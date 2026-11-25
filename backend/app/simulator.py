import asyncio
from dataclasses import dataclass
import logging
import math
import time
from typing import Callable, List, Optional, Tuple
import numpy as np

from .config import settings
from .models import CSIPacket, RoomConfig

logger = logging.getLogger("wifi_spatial.simulator")

WAVELENGTH_24GHZ_M: float = 0.122  # ~2.46 GHz (Canal 11 HT20)


@dataclass
class SimulatedAgent:
    """Agente humano independente com cinemática Random Waypoint em coordenadas métricas (metros)."""

    agent_id: str
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    waypoint_x_m: float
    waypoint_y_m: float
    target_speed_mps: float
    pause_until: float = 0.0
    gait_freq_hz: float = 1.75
    resp_freq_hz: float = 0.28
    gait_phase: float = 0.0
    resp_phase: float = 0.0
    sub_slope_1: float = 0.25
    sub_slope_2: float = 0.43
    phase_seed: float = 0.0


class CSISimulator:
    """
    Simulador Físico Multi-Pessoa Wi-Fi CSI (802.11n HT20, 52 subportadoras válidas):
    - Opera em modo contínuo único controlado por `simulated_people` (0, 1, 2, 3, 4).
    - Modelo cinemático Random Waypoint independente por pessoa:
      * Espaçamento inicial mínimo >= 1.2 m entre agentes.
      * Destinos individuais, velocidades variáveis (0.6 - 1.4 m/s) e pausas aleatórias (2 - 5 s).
      * Desvio suave de paredes e repulsão suave pessoa-pessoa (< 0.9 m).
    - Sinais CSI acoplados fisicamente às coordenadas reais (x, y) em metros dos ESPs e do Linksys E1200.
    """

    def __init__(
        self,
        get_room_config: Callable[[], RoomConfig],
        get_cal_state_fn: Optional[Callable[[], str]] = None,
        is_calibrating_fn: Optional[Callable[[], bool]] = None,
        on_packet_callback: Callable[[CSIPacket, str], None] = lambda pkt, src: None,
        seed: int = 42,
    ):
        self.get_room_config = get_room_config
        self.get_cal_state_fn = get_cal_state_fn
        self.is_calibrating_fn = is_calibrating_fn or (lambda: False)
        self.on_packet_callback = on_packet_callback

        self.enabled: bool = settings.INITIAL_MODE == "simulation"
        self.active_nodes: int = max(1, min(6, settings.INITIAL_SIM_NODES))
        self.simulated_people: int = max(0, min(4, settings.INITIAL_SIM_PEOPLE))
        self.allow_pauses: bool = True

        self._rng = np.random.default_rng(seed)
        self._agents: List[SimulatedAgent] = []
        self._stage2_ref_agent: Optional[SimulatedAgent] = None

        self._pending_agc_scale: float = 1.0
        self._agc_hold_frames: int = 0

        self._subcarrier_k = np.arange(settings.NUM_SUBCARRIERS, dtype=np.float64)
        self._base_profile = 24.0 + 4.5 * np.cos(2.0 * np.pi * (self._subcarrier_k - 26.0) / 52.0)

        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

        self._sync_agents_count()

    @property
    def agents(self) -> List[SimulatedAgent]:
        return self._agents

    def get_agent_positions_m(self) -> List[Tuple[float, float]]:
        """Retorna as posições reais (x_m, y_m) dos agentes simulados para validação em testes."""
        return [(a.x_m, a.y_m) for a in self._agents]

    def set_people_count(self, count: int) -> None:
        count = max(0, min(4, int(count)))
        if count != self.simulated_people or len(self._agents) != count:
            self.simulated_people = count
            self._sync_agents_count()

    def set_scenario(
        self,
        scenario: str,
        active_nodes: Optional[int] = None,
        simulated_people: Optional[int] = None,
    ) -> None:
        """Compatibilidade retroativa caso código legado chame set_scenario."""
        if active_nodes is not None:
            self.active_nodes = max(1, min(6, int(active_nodes)))
        if simulated_people is not None:
            self.set_people_count(simulated_people)
        elif scenario == "empty_room":
            self.set_people_count(0)
        elif self.simulated_people == 0:
            self.set_people_count(1)

    def set_custom_trajectories(
        self,
        starts_and_waypoints_m: List[Tuple[float, float, float, float, float]],
        allow_pauses: bool = False,
    ) -> None:
        """
        Configura agentes determinísticos para testes controlados:
        Cada tupla = (x0_m, y0_m, wp_x_m, wp_y_m, speed_mps).
        """
        self.allow_pauses = allow_pauses
        self.simulated_people = len(starts_and_waypoints_m)
        self._agents = []
        distinct_gaits = [1.55, 2.05, 2.48, 1.32]
        distinct_slopes_1 = [0.19, 0.37, 0.53, 0.29]
        distinct_slopes_2 = [0.41, 0.61, 0.23, 0.49]

        for idx, (x0, y0, wx, wy, spd) in enumerate(starts_and_waypoints_m):
            dx, dy = wx - x0, wy - y0
            dist = max(1e-3, math.hypot(dx, dy))
            self._agents.append(
                SimulatedAgent(
                    agent_id=f"S{idx + 1}",
                    x_m=float(x0),
                    y_m=float(y0),
                    vx_mps=float(spd * dx / dist),
                    vy_mps=float(spd * dy / dist),
                    waypoint_x_m=float(wx),
                    waypoint_y_m=float(wy),
                    target_speed_mps=float(np.clip(spd, 0.6, 1.4)),
                    pause_until=0.0,
                    gait_freq_hz=distinct_gaits[idx % len(distinct_gaits)],
                    resp_freq_hz=0.25 + 0.04 * idx,
                    gait_phase=float(idx * 1.57),
                    resp_phase=float(idx * 1.1),
                    sub_slope_1=distinct_slopes_1[idx % len(distinct_slopes_1)],
                    sub_slope_2=distinct_slopes_2[idx % len(distinct_slopes_2)],
                    phase_seed=float(idx * 2.09),
                )
            )

    def inject_agc_spike(self, factor: float = 3.5, duration_frames: int = 18) -> None:
        self._pending_agc_scale = float(np.clip(factor, 0.2, 8.0))
        self._agc_hold_frames = max(1, duration_frames)

    # ==========================================================================
    # Cinemática Random Waypoint Independente + Repulsão (< 0.9 m) + Paredes
    # ==========================================================================

    def _room_dims(self) -> Tuple[float, float]:
        try:
            cfg = self.get_room_config()
            return float(cfg.width_m), float(cfg.length_m)
        except Exception:
            return 5.0, 4.5

    def _pick_waypoint(self, width_m: float, length_m: float, avoid_xy: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        margin_x = min(0.60, width_m * 0.14)
        margin_y = min(0.65, length_m * 0.15)
        for _ in range(25):
            wx = float(self._rng.uniform(margin_x, max(margin_x + 0.2, width_m - margin_x)))
            wy = float(self._rng.uniform(margin_y, max(margin_y + 0.2, length_m - margin_y)))
            if avoid_xy is None or math.hypot(wx - avoid_xy[0], wy - avoid_xy[1]) >= 1.4:
                return wx, wy
        return float(width_m * 0.5), float(length_m * 0.5)

    def _sync_agents_count(self) -> None:
        width_m, length_m = self._room_dims()
        target_n = self.simulated_people
        if target_n <= 0:
            self._agents = []
            return

        if len(self._agents) > target_n:
            self._agents = self._agents[:target_n]
            return

        distinct_gaits = [1.52, 2.03, 2.44, 1.34]
        distinct_slopes_1 = [0.19, 0.37, 0.53, 0.29]
        distinct_slopes_2 = [0.41, 0.61, 0.23, 0.49]

        # Âncoras bem distribuídas caso a amostragem aleatória precise de garantia de >= 1.2m
        fallback_anchors = [
            (0.22 * width_m, 0.55 * length_m),
            (0.78 * width_m, 0.55 * length_m),
            (0.50 * width_m, 0.32 * length_m),
            (0.50 * width_m, 0.76 * length_m),
        ]

        while len(self._agents) < target_n:
            idx = len(self._agents)
            margin_x = min(0.60, width_m * 0.14)
            margin_y = min(0.65, length_m * 0.15)

            chosen_xy: Optional[Tuple[float, float]] = None
            for _ in range(60):
                cx = float(self._rng.uniform(margin_x, max(margin_x + 0.2, width_m - margin_x)))
                cy = float(self._rng.uniform(margin_y, max(margin_y + 0.2, length_m - margin_y)))
                if all(math.hypot(cx - a.x_m, cy - a.y_m) >= 1.20 for a in self._agents):
                    chosen_xy = (cx, cy)
                    break

            if chosen_xy is None:
                chosen_xy = fallback_anchors[idx % len(fallback_anchors)]

            # Escolhe waypoint em direção oposta ao quadrante atual para garantir diversidade direcional
            opp_x = width_m - chosen_xy[0] + float(self._rng.uniform(-0.35, 0.35))
            opp_y = length_m - chosen_xy[1] + float(self._rng.uniform(-0.35, 0.35))
            wx = float(np.clip(opp_x, margin_x, width_m - margin_x))
            wy = float(np.clip(opp_y, margin_y, length_m - margin_y))

            spd = float(self._rng.uniform(0.65, 1.35))
            dx, dy = wx - chosen_xy[0], wy - chosen_xy[1]
            dist = max(1e-3, math.hypot(dx, dy))

            self._agents.append(
                SimulatedAgent(
                    agent_id=f"S{idx + 1}",
                    x_m=chosen_xy[0],
                    y_m=chosen_xy[1],
                    vx_mps=spd * dx / dist,
                    vy_mps=spd * dy / dist,
                    waypoint_x_m=wx,
                    waypoint_y_m=wy,
                    target_speed_mps=spd,
                    pause_until=0.0,
                    gait_freq_hz=distinct_gaits[idx % len(distinct_gaits)],
                    resp_freq_hz=0.25 + 0.04 * idx,
                    gait_phase=float(self._rng.uniform(0.0, 2.0 * math.pi)),
                    resp_phase=float(self._rng.uniform(0.0, 2.0 * math.pi)),
                    sub_slope_1=distinct_slopes_1[idx % len(distinct_slopes_1)],
                    sub_slope_2=distinct_slopes_2[idx % len(distinct_slopes_2)],
                    phase_seed=float(idx * 2.15 + 0.4),
                )
            )

    def _step_agent_list(
        self,
        agents: List[SimulatedAgent],
        width_m: float,
        length_m: float,
        dt: float,
        now: float,
        allow_pauses: bool,
    ) -> None:
        wall_margin = 0.52
        repulsion_radius = 0.90

        for i, ag in enumerate(agents):
            ag.resp_phase += 2.0 * math.pi * ag.resp_freq_hz * dt

            if allow_pauses and now < ag.pause_until:
                ag.vx_mps *= 0.65
                ag.vy_mps *= 0.65
                continue

            dx = ag.waypoint_x_m - ag.x_m
            dy = ag.waypoint_y_m - ag.y_m
            dist_wp = math.hypot(dx, dy)

            if dist_wp < 0.26:
                ag.waypoint_x_m, ag.waypoint_y_m = self._pick_waypoint(
                    width_m, length_m, avoid_xy=(ag.x_m, ag.y_m)
                )
                ag.target_speed_mps = float(self._rng.uniform(0.60, 1.40))
                if allow_pauses and float(self._rng.random()) < 0.28:
                    ag.pause_until = now + float(self._rng.uniform(2.0, 5.0))
                    ag.vx_mps = 0.0
                    ag.vy_mps = 0.0
                    continue
                dx = ag.waypoint_x_m - ag.x_m
                dy = ag.waypoint_y_m - ag.y_m
                dist_wp = max(1e-3, math.hypot(dx, dy))

            des_vx = (dx / max(1e-3, dist_wp)) * ag.target_speed_mps
            des_vy = (dy / max(1e-3, dist_wp)) * ag.target_speed_mps

            # Desvio suave de paredes
            if ag.x_m < wall_margin:
                des_vx += 1.4 * ((wall_margin - ag.x_m) / wall_margin)
            elif ag.x_m > (width_m - wall_margin):
                des_vx -= 1.4 * ((ag.x_m - (width_m - wall_margin)) / wall_margin)

            if ag.y_m < wall_margin:
                des_vy += 1.4 * ((wall_margin - ag.y_m) / wall_margin)
            elif ag.y_m > (length_m - wall_margin):
                des_vy -= 1.4 * ((ag.y_m - (length_m - wall_margin)) / wall_margin)

            # Repulsão suave pessoa-pessoa (< 0.9 m)
            for j, other in enumerate(agents):
                if i == j:
                    continue
                rx = ag.x_m - other.x_m
                ry = ag.y_m - other.y_m
                d_ij = math.hypot(rx, ry)
                if 1e-4 < d_ij < repulsion_radius:
                    push = 1.35 * ((repulsion_radius - d_ij) / repulsion_radius)
                    des_vx += (rx / d_ij) * push
                    des_vy += (ry / d_ij) * push

            # Suavização de inércia humana e limite físico de velocidade
            ag.vx_mps = 0.78 * ag.vx_mps + 0.22 * des_vx
            ag.vy_mps = 0.78 * ag.vy_mps + 0.22 * des_vy
            spd_cur = math.hypot(ag.vx_mps, ag.vy_mps)
            if spd_cur > 1.55:
                ag.vx_mps *= 1.55 / spd_cur
                ag.vy_mps *= 1.55 / spd_cur

            ag.x_m = float(np.clip(ag.x_m + ag.vx_mps * dt, 0.32, width_m - 0.32))
            ag.y_m = float(np.clip(ag.y_m + ag.vy_mps * dt, 0.35, length_m - 0.35))
            ag.gait_phase += 2.0 * math.pi * ag.gait_freq_hz * max(0.35, spd_cur / 0.95) * dt

    def _step_kinematics(self, dt: float, now: float) -> None:
        width_m, length_m = self._room_dims()
        cal_state = self.get_cal_state_fn() if self.get_cal_state_fn is not None else "IDLE"

        if cal_state == "CAPTURING_STAGE2":
            if self._stage2_ref_agent is None:
                wx, wy = self._pick_waypoint(width_m, length_m)
                self._stage2_ref_agent = SimulatedAgent(
                    agent_id="CAL_1P",
                    x_m=width_m * 0.25,
                    y_m=length_m * 0.50,
                    vx_mps=0.95,
                    vy_mps=0.25,
                    waypoint_x_m=wx,
                    waypoint_y_m=wy,
                    target_speed_mps=1.05,
                    pause_until=0.0,
                )
            self._step_agent_list([self._stage2_ref_agent], width_m, length_m, dt, now, allow_pauses=False)
            return

        self._stage2_ref_agent = None
        if self.simulated_people <= 0:
            return
        if len(self._agents) != self.simulated_people:
            self._sync_agents_count()

        self._step_agent_list(self._agents, width_m, length_m, dt, now, allow_pauses=self.allow_pauses)

    # ==========================================================================
    # Síntese CSI Multi-Caminho Acoplada às Coordenadas Reais dos ESPs e E1200
    # ==========================================================================

    def synthesize_packet(
        self,
        node_id: str,
        rx_x: float,
        rx_y: float,
        tx_x: float,
        tx_y: float,
        now: float,
        force_empty: bool = False,
        agc_scale_override: Optional[float] = None,
        width_m: Optional[float] = None,
        length_m: Optional[float] = None,
    ) -> CSIPacket:
        if width_m is None or length_m is None:
            w_cfg, l_cfg = self._room_dims()
            width_m = width_m or w_cfg
            length_m = length_m or l_cfg

        k = self._subcarrier_k
        amplitudes = self._base_profile.copy()

        # Ruído térmico realista + 3 subportadoras ruidosas fixas (para validar NBVI)
        thermal_noise = self._rng.normal(0.0, 0.042, size=settings.NUM_SUBCARRIERS)
        thermal_noise[0] += float(self._rng.normal(0.0, 1.7))
        thermal_noise[25] += float(self._rng.normal(0.0, 1.5))
        thermal_noise[51] += float(self._rng.normal(0.0, 1.8))
        amplitudes = amplitudes + thermal_noise

        tx_xm, tx_ym = tx_x * width_m, tx_y * length_m
        rx_xm, rx_ym = rx_x * width_m, rx_y * length_m
        link_len_m = max(0.40, math.hypot(rx_xm - tx_xm, rx_ym - tx_ym))

        # Atenuação estática de percurso (ESPs mais próximos têm RSSI e sensibilidade bruta maiores)
        rssi_dbm = -41.0 - 14.0 * math.log10(max(1.0, link_len_m)) + float(self._rng.normal(0.0, 0.22))
        link_dist_gain = (3.6 / max(1.1, link_len_m)) ** 0.45

        cal_state = self.get_cal_state_fn() if self.get_cal_state_fn is not None else "IDLE"
        if cal_state == "CAPTURING_STAGE2" and self._stage2_ref_agent is not None and not force_empty:
            active_agents = [self._stage2_ref_agent]
        elif force_empty or self.simulated_people <= 0:
            active_agents = []
        else:
            if len(self._agents) != self.simulated_people:
                self._sync_agents_count()
            active_agents = self._agents

        link_hash = (sum(ord(ch) for ch in node_id) % 17) * 0.37

        for p_idx, ag in enumerate(active_agents):
            d_tx_m = math.hypot(ag.x_m - tx_xm, ag.y_m - tx_ym)
            d_rx_m = math.hypot(ag.x_m - rx_xm, ag.y_m - rx_ym)
            bistatic_m = d_tx_m + d_rx_m
            excess_m = max(0.0, bistatic_m - link_len_m)

            # Acoplamento espacial de Elipse de Fresnel + multi-caminho indoor em metros reais
            fresnel_coupling = (
                0.90 * math.exp(-excess_m / 0.46) + 0.10 * math.exp(-excess_m / 1.90)
            ) * link_dist_gain

            rssi_dbm -= 2.8 * min(1.5, fresnel_coupling)

            spd_cur = math.hypot(ag.vx_mps, ag.vy_mps)
            is_paused = (now < ag.pause_until) or (spd_cur < 0.06)

            # Fase física Doppler bistática + cadência independente por indivíduo
            bistatic_phase = (2.0 * math.pi / WAVELENGTH_24GHZ_M) * bistatic_m
            u1 = np.cos(k * ag.sub_slope_1 + ag.phase_seed + link_hash)
            u2 = np.sin(k * ag.sub_slope_2 - 0.7 * ag.phase_seed + 0.5 * link_hash)

            if is_paused:
                # Pessoa parada: assinatura suave de respiração/postura para testar retenção estática
                resp_mod = math.sin(ag.resp_phase + 0.25 * bistatic_phase)
                amplitudes += (0.68 * fresnel_coupling * resp_mod) * (u1 + 0.45 * u2)
            else:
                dyn_phase_1 = ag.gait_phase + 0.35 * bistatic_phase
                dyn_phase_2 = 1.41 * ag.gait_phase - 0.22 * bistatic_phase + ag.phase_seed
                wave_1 = math.sin(dyn_phase_1) * u1
                wave_2 = 0.74 * math.cos(dyn_phase_2) * u2
                dyn_amp = (5.4 + 1.8 * min(1.4, spd_cur)) * fresnel_coupling
                amplitudes += dyn_amp * (wave_1 + wave_2)

        amplitudes = np.maximum(amplitudes, 1.5)

        phases = 0.25 * k + 0.3 + 0.08 * link_hash
        base_imag = np.rint(amplitudes * np.sin(phases))
        base_real = np.rint(amplitudes * np.cos(phases))

        agc_factor = agc_scale_override if agc_scale_override is not None else self._pending_agc_scale
        imag_arr = base_imag * agc_factor
        real_arr = base_real * agc_factor

        csi_interleaved: List[float] = []
        for im, re in zip(imag_arr.tolist(), real_arr.tolist()):
            csi_interleaved.append(float(im))
            csi_interleaved.append(float(re))

        node_num = 1
        if node_id.startswith("esp_"):
            try:
                node_num = int(node_id.split("_")[1])
            except ValueError:
                node_num = 1

        return CSIPacket(
            node_id=node_id,
            timestamp=now,
            timestamp_us=int(now * 1_000_000),
            src_mac=f"c0:56:27:e1:20:{node_num:02x}",
            rssi=int(round(np.clip(rssi_dbm, -90.0, -25.0))),
            noise_floor=-93,
            channel=11,
            csi_raw=csi_interleaved,
        )

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Simulador CSI Random Waypoint iniciado ({settings.SIMULATOR_HZ} Hz).")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run_loop(self) -> None:
        interval = 1.0 / settings.SIMULATOR_HZ
        last_time = time.time()

        while self._running:
            now = time.time()
            dt = max(0.005, min(0.1, now - last_time))
            last_time = now

            if self.enabled:
                cal_state = self.get_cal_state_fn() if self.get_cal_state_fn is not None else "IDLE"
                force_empty = cal_state in ("PREPARING", "CAPTURING") or (
                    cal_state == "IDLE" and self.is_calibrating_fn()
                )
                if not force_empty:
                    self._step_kinematics(dt, now)

                if self._agc_hold_frames > 0:
                    self._agc_hold_frames -= 1
                    if self._agc_hold_frames == 0:
                        self._pending_agc_scale = 1.0

                room_cfg = self.get_room_config()
                tx_x, tx_y = room_cfg.tx.x, room_cfg.tx.y

                for rx in room_cfg.rx_nodes[: self.active_nodes]:
                    pkt = self.synthesize_packet(
                        node_id=rx.id,
                        rx_x=rx.x,
                        rx_y=rx.y,
                        tx_x=tx_x,
                        tx_y=tx_y,
                        now=now,
                        force_empty=force_empty,
                        width_m=room_cfg.width_m,
                        length_m=room_cfg.length_m,
                    )
                    self.on_packet_callback(pkt, "simulation")

            await asyncio.sleep(interval)
