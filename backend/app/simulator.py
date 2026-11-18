import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np

from .config import settings
from .models import CSIPacket, RoomConfig

logger = logging.getLogger("wifi_spatial.simulator")


@dataclass
class VirtualPerson:
    """Agente virtual caminhando suavemente no grid normalizado [0.0, 1.0] x [0.0, 1.0]."""

    agent_id: int
    x: float
    y: float
    target_x: float
    target_y: float
    speed: float
    pause_remaining: float = 0.0
    doppler_phase: float = 0.0
    doppler_freq_hz: float = 1.9

    @property
    def is_moving(self) -> bool:
        return self.pause_remaining <= 0.0


class CSISimulator:
    """
    Simulador Físico Integrado de Wi-Fi CSI (52 subportadoras a 20 Hz).
    Modela atenuação na 1ª Zona de Fresnel (Elipse TX-RX), reflexão multi-caminho por distância
    radial, descorrelação espectral multi-pessoa (subcarrier_diversity), efeito Doppler de caminhada
    humana e desvio linear de fase (CFO/SFO do ESP32).
    """

    ANCHOR_REGIONS = [
        (0.20, 0.80, 0.22, 0.80),  # Agente 1: centro e eixo principal
        (0.18, 0.48, 0.24, 0.78),  # Agente 2: setor esquerdo / superior
        (0.52, 0.82, 0.24, 0.78),  # Agente 3: setor direito / inferior
        (0.28, 0.72, 0.25, 0.76),  # Agente 4: diagonais centrais
    ]

    def __init__(
        self,
        get_room_config: Callable[[], RoomConfig],
        is_calibrating_fn: Callable[[], bool],
        on_packet_callback: Callable[[CSIPacket, str], None],
        get_calibration_mode_fn: Optional[Callable[[], str]] = None,
    ):
        self.get_room_config = get_room_config
        self.is_calibrating_fn = is_calibrating_fn
        self.on_packet_callback = on_packet_callback
        self.get_calibration_mode_fn = get_calibration_mode_fn or (lambda: "empty_room")

        self.enabled: bool = settings.INITIAL_MODE == "simulation"
        self.num_people: int = settings.INITIAL_SIM_PEOPLE
        self.num_nodes: int = settings.INITIAL_SIM_NODES
        self.agents: List[VirtualPerson] = []
        self._rng = np.random.default_rng(42)
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._subcarrier_k = np.arange(settings.NUM_SUBCARRIERS, dtype=np.float64)

        self.set_people_count(self.num_people)

    def set_nodes_count(self, count: int) -> None:
        """Ajusta dinamicamente quantos módulos ESP32 são simulados (0, 1, 2 ou 3 ESPs)."""
        self.num_nodes = max(0, min(3, int(count)))
        if self.num_nodes == 1 and self.agents:
            self._reposition_agents_for_single_link()

    def _reposition_agents_for_single_link(self) -> None:
        """Distribui os agentes virtuais ao longo da Zona de Abrangência do único ESP32 ativo."""
        try:
            room_cfg = self.get_room_config()
            if room_cfg.rx_nodes:
                tx_x, tx_y = room_cfg.tx.x, room_cfg.tx.y
                rx_x, rx_y = room_cfg.rx_nodes[0].x, room_cfg.rx_nodes[0].y
                long_slots = [0.24, 0.52, 0.78, 0.38]
                lat_slots = [-0.03, 0.03, 0.01, -0.04]
                for idx, ag in enumerate(self.agents):
                    t = long_slots[idx % len(long_slots)]
                    lat = lat_slots[idx % len(lat_slots)]
                    ag.x = float(np.clip(tx_x + t * (rx_x - tx_x) + lat, 0.25, 0.75))
                    ag.y = float(np.clip(tx_y + t * (rx_y - tx_y), 0.20, 0.80))
                    ag.target_x, ag.target_y = self._pick_next_waypoint(idx, ag.x, ag.y)
        except Exception:
            pass

    def set_people_count(self, count: int) -> None:
        """Ajusta dinamicamente o número de agentes virtuais (0 a 4 pessoas) na área."""
        count = max(0, min(4, int(count)))
        self.num_people = count

        initial_positions = [
            (0.46, 0.36),
            (0.54, 0.66),
            (0.35, 0.52),
            (0.65, 0.50),
        ]

        while len(self.agents) < count:
            idx = len(self.agents)
            ix, iy = initial_positions[idx % len(initial_positions)]
            tx, ty = self._pick_next_waypoint(idx, ix, iy)
            self.agents.append(
                VirtualPerson(
                    agent_id=idx + 1,
                    x=ix,
                    y=iy,
                    target_x=tx,
                    target_y=ty,
                    speed=float(self._rng.uniform(0.065, 0.095)),
                    pause_remaining=0.0,
                    doppler_phase=float(self._rng.uniform(0.0, 2.0 * math.pi)),
                    doppler_freq_hz=float(1.45 + 0.55 * idx),
                )
            )

        if len(self.agents) > count:
            self.agents = self.agents[:count]

        if self.num_nodes == 1 and self.agents:
            self._reposition_agents_for_single_link()

    def _pick_next_waypoint(self, agent_idx: int, curr_x: float, curr_y: float) -> tuple[float, float]:
        # Quando apenas 1 ESP32 está ativo, todos os agentes circulam dentro da elipse de abrangência TX <-> ESP_1
        if self.num_nodes == 1:
            try:
                room_cfg = self.get_room_config()
                if room_cfg.rx_nodes:
                    tx_x, tx_y = room_cfg.tx.x, room_cfg.tx.y
                    rx_x, rx_y = room_cfg.rx_nodes[0].x, room_cfg.rx_nodes[0].y
                    # Divide o corredor em setores não-sobrepostos por agente para evitar interferência destrutiva
                    t_ranges = [(0.16, 0.34), (0.44, 0.60), (0.68, 0.86), (0.22, 0.80)]
                    tmin, tmax = t_ranges[agent_idx % len(t_ranges)]
                    for _ in range(10):
                        t = float(self._rng.uniform(tmin, tmax))
                        lat_offset = float(self._rng.uniform(-0.08, 0.08))
                        nx = float(np.clip(tx_x + t * (rx_x - tx_x) + lat_offset, 0.14, 0.86))
                        ny = float(np.clip(tx_y + t * (rx_y - tx_y), 0.16, 0.84))
                        if math.hypot(nx - curr_x, ny - curr_y) > 0.10:
                            return nx, ny
            except Exception:
                pass

        xmin, xmax, ymin, ymax = self.ANCHOR_REGIONS[agent_idx % len(self.ANCHOR_REGIONS)]
        for _ in range(10):
            nx = float(self._rng.uniform(xmin, xmax))
            ny = float(self._rng.uniform(ymin, ymax))
            if math.hypot(nx - curr_x, ny - curr_y) > 0.16:
                return nx, ny
        return float(self._rng.uniform(0.22, 0.78)), float(self._rng.uniform(0.22, 0.78))

    def _step_agents(self, dt: float) -> None:
        """Atualiza a cinemática dos agentes virtuais com caminhada suave e separação espacial."""
        for idx, ag in enumerate(self.agents):
            if ag.pause_remaining > 0.0:
                ag.pause_remaining = max(0.0, ag.pause_remaining - dt)
                ag.doppler_phase += 2.0 * math.pi * 0.35 * dt
                continue

            dx = ag.target_x - ag.x
            dy = ag.target_y - ag.y
            dist = math.hypot(dx, dy)

            if dist < 0.04:
                ag.target_x, ag.target_y = self._pick_next_waypoint(idx, ag.x, ag.y)
                ag.speed = float(self._rng.uniform(0.068, 0.095))
                continue

            step = min(dist, ag.speed * dt)
            vx = (dx / dist) * step
            vy = (dy / dist) * step

            for other in self.agents:
                if other.agent_id == ag.agent_id:
                    continue
                ox = ag.x - other.x
                oy = ag.y - other.y
                odist = math.hypot(ox, oy)
                if 1e-4 < odist < 0.22:
                    repel = (0.22 - odist) * 0.22 * dt
                    vx += (ox / odist) * repel
                    vy += (oy / odist) * repel

            ag.x = float(np.clip(ag.x + vx, 0.10, 0.90))
            ag.y = float(np.clip(ag.y + vy, 0.14, 0.86))
            ag.doppler_phase += 2.0 * math.pi * ag.doppler_freq_hz * dt

    def _synthesize_node_packet(
        self,
        node_id: str,
        rx_x: float,
        rx_y: float,
        tx_x: float,
        tx_y: float,
        now: float,
        calibrating: bool,
        cal_mode: str,
    ) -> CSIPacket:
        """
        Gera um pacote CSI sintético realista de 52 subportadoras (104 inteiros [Imag, Real]).
        """
        k = self._subcarrier_k
        amplitudes = 22.0 + 2.0 * np.cos(2.0 * np.pi * (k - 26.0) / 52.0)
        true_phases = np.zeros(settings.NUM_SUBCARRIERS, dtype=np.float64)

        link_dx = rx_x - tx_x
        link_dy = rx_y - tx_y
        link_len = max(0.05, math.hypot(link_dx, link_dy))
        # RSSI base correspondente a ~3.1m na equação Log-Distance Path Loss (-50.8 dBm)
        rssi_float = -50.8 + float(self._rng.normal(0.0, 0.30))

        if calibrating and cal_mode == "person_ref":
            # Na Etapa 2 (Referência de 1 Pessoa Caminhando), simula exatamente 1 pessoa andando no feixe
            if self.agents:
                active_agents = [self.agents[0]]
            else:
                active_agents = [
                    VirtualPerson(
                        agent_id=1,
                        x=0.50,
                        y=0.48,
                        target_x=0.50,
                        target_y=0.68,
                        speed=0.08,
                        pause_remaining=0.0,
                        doppler_phase=now * 3.8,
                        doppler_freq_hz=1.8,
                    )
                ]
        elif calibrating:
            active_agents = []
        else:
            active_agents = self.agents

        fresnel_sub_k = np.arange(26, dtype=np.float64)
        radial_sub_k = np.arange(26, dtype=np.float64)

        for idx, ag in enumerate(active_agents):
            d_tx = math.hypot(ag.x - tx_x, ag.y - tx_y)
            d_rx = math.hypot(ag.x - rx_x, ag.y - rx_y)

            excess_d = max(0.0, (d_tx + d_rx) - link_len)
            fresnel_factor = math.exp(-excess_d / 0.24)

            proj_t = ((ag.x - tx_x) * link_dx + (ag.y - tx_y) * link_dy) / (link_len ** 2)
            proj_t = max(0.0, min(1.0, proj_t))

            r_norm = max(0.0, min(1.0, d_rx / 1.42))

            rssi_float -= 4.5 * fresnel_factor

            fresnel_center = proj_t * 25.0
            radial_center = r_norm * 25.0

            env_fresnel = np.exp(-0.5 * ((fresnel_sub_k - fresnel_center) / 1.45) ** 2) * (0.40 + 0.60 * fresnel_factor)
            env_radial = np.exp(-0.5 * ((radial_sub_k - radial_center) / 1.20) ** 2)

            # Cada pessoa adicional introduz modulação seletiva em frequência (multipath decorrelation)
            multipath_harmonic = np.sin(fresnel_sub_k * (0.35 + 0.28 * idx) + ag.doppler_phase * (0.7 + 0.4 * idx))
            radial_harmonic = np.cos(radial_sub_k * (0.40 + 0.31 * idx) - ag.doppler_phase * (0.8 + 0.3 * idx))

            if ag.is_moving:
                osc_amp = math.sin(ag.doppler_phase)
                osc_phase = math.cos(ag.doppler_phase * 1.13)

                amplitudes[:26] += env_fresnel * (3.8 + 5.6 * osc_amp) + 1.35 * multipath_harmonic * fresnel_factor
                amplitudes[26:52] += env_radial * (4.8 + 6.5 * osc_amp) + 1.45 * radial_harmonic * fresnel_factor

                true_phases[:26] += env_fresnel * (0.35 + 0.62 * osc_phase) + 0.22 * multipath_harmonic * fresnel_factor
                true_phases[26:52] += env_radial * (0.45 + 0.72 * osc_phase) + 0.25 * radial_harmonic * fresnel_factor
            else:
                micro_osc = 0.20 * math.sin(ag.doppler_phase)
                amplitudes[:26] += env_fresnel * (4.5 + micro_osc) + 0.9 * multipath_harmonic * fresnel_factor
                amplitudes[26:52] += env_radial * (5.8 + micro_osc) + 1.0 * radial_harmonic * fresnel_factor

                true_phases[:26] += env_fresnel * (0.48 + 0.05 * micro_osc) + 0.14 * multipath_harmonic
                true_phases[26:52] += env_radial * (0.62 + 0.05 * micro_osc) + 0.16 * radial_harmonic

        # Durante calibração no modo 'crowd_compensated', simula ruído leve de plateia fora do feixe
        if calibrating and cal_mode == "crowd_compensated":
            if self._rng.random() < 0.25:
                # Pico esporádico de alguém se mexendo na plateia (que a Mediana + P20 irá filtrar!)
                amplitudes += self._rng.normal(0.0, 0.85, size=settings.NUM_SUBCARRIERS)
                true_phases += self._rng.normal(0.0, 0.06, size=settings.NUM_SUBCARRIERS)
            else:
                amplitudes += self._rng.normal(0.0, 0.22, size=settings.NUM_SUBCARRIERS)
                true_phases += self._rng.normal(0.0, 0.02, size=settings.NUM_SUBCARRIERS)
        else:
            amplitudes += self._rng.normal(0.0, 0.16, size=settings.NUM_SUBCARRIERS)
            true_phases += self._rng.normal(0.0, 0.015, size=settings.NUM_SUBCARRIERS)

        amplitudes = np.clip(amplitudes, 2.0, 62.0)

        # Injeta desvio linear de fase realista (CFO + SFO do oscilador de cristal do ESP32)
        cfo_offset = float(self._rng.uniform(-math.pi, math.pi))
        sfo_slope = float(self._rng.uniform(-0.18, 0.18))
        corrupted_phases = true_phases + sfo_slope * k + cfo_offset

        imag_arr = np.rint(amplitudes * np.sin(corrupted_phases)).astype(int)
        real_arr = np.rint(amplitudes * np.cos(corrupted_phases)).astype(int)

        csi_interleaved: List[int] = []
        for im, re in zip(imag_arr.tolist(), real_arr.tolist()):
            csi_interleaved.append(int(np.clip(im, -127, 127)))
            csi_interleaved.append(int(np.clip(re, -127, 127)))

        return CSIPacket(
            node_id=node_id,
            timestamp=now,
            rssi=int(round(np.clip(rssi_float, -92.0, -25.0))),
            csi_raw=csi_interleaved,
        )

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Simulador físico de Wi-Fi CSI iniciado (20 Hz).")

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
        interval = 1.0 / settings.SIMULATOR_HZ  # 0.05s (20 Hz)
        last_time = time.time()

        while self._running:
            now = time.time()
            dt = max(0.01, min(0.2, now - last_time))
            last_time = now

            if self.enabled:
                calibrating = self.is_calibrating_fn()
                cal_mode = self.get_calibration_mode_fn()
                if not calibrating or cal_mode == "person_ref":
                    self._step_agents(dt)

                room_cfg = self.get_room_config()
                tx_x, tx_y = room_cfg.tx.x, room_cfg.tx.y

                for rx in room_cfg.rx_nodes[: self.num_nodes]:
                    pkt = self._synthesize_node_packet(
                        node_id=rx.id,
                        rx_x=rx.x,
                        rx_y=rx.y,
                        tx_x=tx_x,
                        tx_y=tx_y,
                        now=now,
                        calibrating=calibrating,
                        cal_mode=cal_mode,
                    )
                    self.on_packet_callback(pkt, "simulation")

            await asyncio.sleep(interval)
