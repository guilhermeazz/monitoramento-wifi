from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import settings
from .dsp_engine import NodeRuntimeState
from .models import (
    GeometryWarning,
    NodePosition,
    OccupancyState,
    RoomConfig,
    SpatialTarget,
    TopologyMode,
    UncertaintyMode,
)


class MetricKalmanFilter2D:
    """
    Filtro de Kalman Linear 2D em coordenadas métricas reais [x_m, y_m, vx_mps, vy_mps]^T:
    - Velocidade humana máxima limitada fisicamente a 2.0 m/s.
    - Suporta predição pura (coast) durante oclusões curtas/cruzamentos e retenção estática.
    """

    def __init__(
        self,
        x0_m: float,
        y0_m: float,
        now: float,
        process_noise: float = 0.35,
        measurement_noise: float = 0.22,
        max_speed_mps: float = settings.KALMAN_MAX_SPEED_MPS,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_speed_mps = max_speed_mps
        self.state = np.array([x0_m, y0_m, 0.0, 0.0], dtype=np.float64)
        self.cov = np.diag([0.16, 0.16, 0.50, 0.50]).astype(np.float64)
        self.last_ts: float = now

    def predict(self, now: float, width_m: float, length_m: float, damp_velocity: float = 0.88) -> Tuple[float, float]:
        dt = float(np.clip(now - self.last_ts, 0.0, 0.5))
        if dt <= 1e-4:
            return float(self.state[0]), float(self.state[1])
        self.last_ts = now

        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, damp_velocity, 0.0],
                [0.0, 0.0, 0.0, damp_velocity],
            ],
            dtype=np.float64,
        )
        q_pos = 0.25 * (dt ** 2) * self.process_noise
        q_vel = dt * self.process_noise
        Q = np.diag([q_pos, q_pos, q_vel, q_vel])

        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q
        self._clamp_state(width_m, length_m)
        return float(self.state[0]), float(self.state[1])

    def update(self, meas_x_m: float, meas_y_m: float, now: float, width_m: float, length_m: float) -> Tuple[float, float, float, float]:
        self.predict(now, width_m, length_m, damp_velocity=0.90)

        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        R = np.eye(2, dtype=np.float64) * (self.measurement_noise ** 2)

        z = np.array([meas_x_m, meas_y_m], dtype=np.float64)
        innov = z - (H @ self.state)
        innov_norm = float(np.linalg.norm(innov))
        if innov_norm > 1.35:
            innov = innov * (1.35 / innov_norm)

        S = H @ self.cov @ H.T + R
        K = self.cov @ H.T @ np.linalg.inv(S)

        self.state = self.state + (K @ innov)
        self.cov = (np.eye(4, dtype=np.float64) - K @ H) @ self.cov
        self._clamp_state(width_m, length_m)

        return (
            float(self.state[0]),
            float(self.state[1]),
            float(self.state[2]),
            float(self.state[3]),
        )

    def freeze_static(self, now: float) -> None:
        self.last_ts = now
        self.state[2] *= 0.25
        self.state[3] *= 0.25

    def _clamp_state(self, width_m: float, length_m: float) -> None:
        margin_x = min(0.28, width_m * 0.08)
        margin_y = min(0.28, length_m * 0.08)
        self.state[0] = float(np.clip(self.state[0], margin_x, max(margin_x + 0.1, width_m - margin_x)))
        self.state[1] = float(np.clip(self.state[1], margin_y, max(margin_y + 0.1, length_m - margin_y)))
        spd = math.hypot(float(self.state[2]), float(self.state[3]))
        if spd > self.max_speed_mps:
            scale = self.max_speed_mps / spd
            self.state[2] *= scale
            self.state[3] *= scale


@dataclass
class TrackedPerson:
    """Rastro individual gerenciado pelo Multi-Object Tracker (Hungarian + Kalman 2D)."""

    track_id: str
    kalman: MetricKalmanFilter2D
    hits: int
    created_ts: float
    last_matched_ts: float
    coast_until_ts: float
    confidence: float = 0.85
    state: OccupancyState = "MOTION"


@dataclass
class SpatialTrackingResult:
    """Resultado consolidado do rastreador espacial multi-alvo e contador híbrido."""

    occupancy_state: OccupancyState
    confidence: float
    target_count: int
    count_display: str
    topology_mode: TopologyMode
    uncertainty_mode: UncertaintyMode
    active_nodes_count: int
    active_zone: str
    mvs_current: float
    threshold_on: float
    threshold_off: float
    intensity: float
    targets: List[SpatialTarget]
    heatmap_grid: List[List[float]]
    grid_nx: int = 20
    grid_ny: int = 18
    cell_size_m: float = 0.25
    spectral_modes_count: int = 0
    spatial_peaks_count: int = 0
    link_decorrelation: float = 0.0


class SpatialTracker:
    """
    Rastreador Espacial 2D Multi-Objeto & Estimador Híbrido de Contagem:
    1. Grid Métrico Real (0.25 m/célula) com Matriz de Pesos Fresnel W pré-calculada e Hot-Reload.
    2. Reconstrução Tomográfica de Rádio (RTI) com ganho equalizado G_l, normalização de sensibilidade
       de coluna e supressão tomográfica de links silenciosos.
    3. Extração Multi-Alvo via Non-Maximum Suppression (NMS) com raio humano mínimo de 1.10 m.
    4. Associação Hungarian (`scipy.optimize.linear_sum_assignment`) + Filtro de Kalman 2D por alvo
       (v_max = 2.0 m/s, coasting anti-pisca e retenção estática integrada ao hold-timer).
    5. Estimativa Híbrida de Contagem (0, 1, 2, 3+): combina picos espaciais NMS, autovalores de subespaço
       em janela de 1.0 s e descorrelação entre links, com mediana/histerese temporal de 2.0 s.
    """

    def __init__(
        self,
        cell_size_m: float = settings.RTI_CELL_SIZE_M,
        nms_radius_m: float = settings.NMS_MIN_HUMAN_RADIUS_M,
        grid_size: Optional[int] = None,
    ):
        self.cell_size_m = float(cell_size_m)
        self.nms_radius_m = float(nms_radius_m)
        self._legacy_grid_size = grid_size

        # Estado pré-calculado do layout em memória (atualizado via hot_reload_layout)
        self._layout_signature: Optional[Tuple] = None
        self.grid_nx: int = 20
        self.grid_ny: int = 18
        self.grid_x_m: np.ndarray = np.zeros((18, 20), dtype=np.float64)
        self.grid_y_m: np.ndarray = np.zeros((18, 20), dtype=np.float64)
        self.w_matrix_by_id: Dict[str, np.ndarray] = {}
        self.link_distances_m: Dict[str, float] = {}
        self.geometry_warnings: List[GeometryWarning] = []
        self.interior_prior: np.ndarray = np.ones((18, 20), dtype=np.float64)

        self.heatmap_ema: np.ndarray = np.zeros((18, 20), dtype=np.float64)
        self._tracks: List[TrackedPerson] = []
        self._next_track_num: int = 1

        # Histerese temporal de 2.0s na contagem de pessoas
        self._count_history: deque[Tuple[float, int]] = deque(maxlen=120)
        self._committed_count: int = 0
        self._last_motion_count: int = 1
        self._sector_ema: Dict[str, float] = {}
        self.peak_occupancy: int = 0

    def reset(self) -> None:
        self.heatmap_ema.fill(0.0)
        self._tracks.clear()
        self._next_track_num = 1
        self._count_history.clear()
        self._committed_count = 0
        self._sector_ema.clear()

    # ==========================================================================
    # Layout Métrico (0.25 m/célula), Avisos de Geometria e Hot-Reload da Matriz W
    # ==========================================================================

    @staticmethod
    def _make_layout_signature(room_config: RoomConfig) -> Tuple:
        rx_sig = tuple((r.id, round(r.x, 4), round(r.y, 4)) for r in room_config.rx_nodes)
        return (
            round(room_config.width_m, 3),
            round(room_config.length_m, 3),
            round(room_config.tx.x, 4),
            round(room_config.tx.y, 4),
            rx_sig,
        )

    def hot_reload_layout(self, room_config: RoomConfig) -> List[GeometryWarning]:
        """
        Recalcula imediatamente em memória a malha métrica (0.25 m/célula), a matriz de pesos
        de Fresnel W por link e os diagnósticos geométricos sem reiniciar o servidor.
        """
        w_m = float(room_config.width_m)
        l_m = float(room_config.length_m)
        nx = max(8, int(round(w_m / self.cell_size_m)))
        ny = max(8, int(round(l_m / self.cell_size_m)))

        xs_m = (np.arange(nx, dtype=np.float64) + 0.5) * (w_m / nx)
        ys_m = (np.arange(ny, dtype=np.float64) + 0.5) * (l_m / ny)
        gx_m, gy_m = np.meshgrid(xs_m, ys_m)

        self.grid_nx = nx
        self.grid_ny = ny
        self.grid_x_m = gx_m
        self.grid_y_m = gy_m

        if self.heatmap_ema.shape != (ny, nx):
            self.heatmap_ema = np.zeros((ny, nx), dtype=np.float64)

        # Prior suave de interior de sala (evita artefatos colados na alvenaria a < 0.30m)
        dist_wall = np.minimum.reduce([gx_m, w_m - gx_m, gy_m, l_m - gy_m])
        self.interior_prior = np.clip(0.55 + 0.45 * np.tanh((dist_wall - 0.25) / 0.25), 0.35, 1.0)

        tx_xm = room_config.tx.x * w_m
        tx_ym = room_config.tx.y * l_m
        d_tx_grid = np.hypot(gx_m - tx_xm, gy_m - tx_ym)
        tx_taper = 1.0 - 0.58 * np.exp(-(d_tx_grid ** 2) / (2.0 * (0.48 ** 2)))

        self.w_matrix_by_id = {}
        self.link_distances_m = {}
        warnings: List[GeometryWarning] = []
        unit_vectors: List[Tuple[NodePosition, float, float, float]] = []

        for rx in room_config.rx_nodes:
            rx_xm = rx.x * w_m
            rx_ym = rx.y * l_m
            dx = rx_xm - tx_xm
            dy = rx_ym - tx_ym
            link_dist = float(math.hypot(dx, dy))
            safe_dist = max(0.30, link_dist)
            self.link_distances_m[rx.id] = round(link_dist, 2)

            if link_dist < 1.50:
                warnings.append(
                    GeometryWarning(
                        code="LINK_TOO_SHORT",
                        severity="warning",
                        node_id=rx.id,
                        message=f"{rx.name} está a {link_dist:.2f} m do roteador (< 1.5 m): zona de saturação direta.",
                    )
                )
            elif link_dist > 7.00:
                warnings.append(
                    GeometryWarning(
                        code="LINK_TOO_LONG",
                        severity="warning",
                        node_id=rx.id,
                        message=f"{rx.name} está a {link_dist:.2f} m do roteador (> 7.0 m): sensibilidade reduzida.",
                    )
                )

            unit_vectors.append((rx, dx / safe_dist, dy / safe_dist, link_dist))

            d_rx_grid = np.hypot(gx_m - rx_xm, gy_m - rx_ym)
            excess_m = np.maximum(0.0, (d_tx_grid + d_rx_grid) - safe_dist)
            rx_taper = 1.0 - 0.30 * np.exp(-(d_rx_grid ** 2) / (2.0 * (0.32 ** 2)))

            # Peso elíptico de Fresnel com decaimento exponencial pelo excesso de caminho Delta d_l(x,y)
            w_grid = np.exp(-excess_m / 0.40) * tx_taper * rx_taper * self.interior_prior
            self.w_matrix_by_id[rx.id] = w_grid.ravel()

        # Verifica pares de links quase colineares
        for i in range(len(unit_vectors)):
            for j in range(i + 1, len(unit_vectors)):
                rx_a, ux_a, uy_a, _ = unit_vectors[i]
                rx_b, ux_b, uy_b, _ = unit_vectors[j]
                cross_sin = abs(ux_a * uy_b - uy_a * ux_b)
                dot_cos = ux_a * ux_b + uy_a * uy_b
                if cross_sin < 0.18 and dot_cos > 0.0:
                    warnings.append(
                        GeometryWarning(
                            code="COLLINEAR_LINKS",
                            severity="warning",
                            node_id=rx_b.id,
                            message=(
                                f"{rx_a.name} e {rx_b.name} estão quase colineares em relação ao TX; "
                                "aumente a abertura angular para melhorar a precisão 2D."
                            ),
                        )
                    )

        self.geometry_warnings = warnings
        self._layout_signature = self._make_layout_signature(room_config)
        return warnings

    def _ensure_layout(self, room_config: RoomConfig) -> None:
        sig = self._make_layout_signature(room_config)
        if self._layout_signature != sig:
            self.hot_reload_layout(room_config)

    @staticmethod
    def determine_topology_mode(active_nodes_count: int) -> TopologyMode:
        if active_nodes_count <= 0:
            return "NO_SENSORS"
        if active_nodes_count == 1:
            return "SINGLE_NODE_FRESNEL"
        if active_nodes_count == 2:
            return "DUAL_NODE_SECTOR"
        return "MULTI_NODE_KALMAN_2D"

    @staticmethod
    def format_count_display(count: int) -> str:
        if count <= 0:
            return "0"
        if count >= 3:
            return f"{count}" if count == 3 else "3+"
        return str(count)

    # ==========================================================================
    # Loop Principal de Atualização (20 Hz)
    # ==========================================================================

    def update(
        self,
        room_config: RoomConfig,
        node_states: Dict[str, NodeRuntimeState],
        is_calibrating: bool = False,
        now: Optional[float] = None,
        hold_time_sec: float = settings.DEFAULT_HOLD_TIME_SEC,
    ) -> SpatialTrackingResult:
        if now is None:
            now = time.time()

        self._ensure_layout(room_config)

        active_pairs: List[Tuple[NodePosition, NodeRuntimeState]] = []
        for rx in room_config.rx_nodes:
            st = node_states.get(rx.id)
            if st is not None and st.online:
                active_pairs.append((rx, st))

        active_count = len(active_pairs)
        topology_mode = self.determine_topology_mode(active_count)

        if active_pairs:
            _, dominant_st = max(active_pairs, key=lambda p: p[1].intensity)
            mvs_cur = dominant_st.mvs_current
            th_on = dominant_st.threshold_on
            th_off = dominant_st.threshold_off
            max_intensity = max(p[1].intensity for p in active_pairs)
        else:
            mvs_cur = 0.0
            th_on = 2.5e-6
            th_off = 2.125e-6
            max_intensity = 0.0

        # ======================================================================
        # ESTÁGIO 1: GATE SOBERANO DE PRESENÇA (Zero Fantasmas em Sala Vazia)
        # ======================================================================
        all_empty = all(st.state == "EMPTY" for _, st in active_pairs)
        if active_count == 0 or is_calibrating or all_empty:
            self.heatmap_ema *= 0.20
            self._tracks.clear()
            self._count_history.clear()
            self._committed_count = 0
            zero_grid = [[0.0 for _ in range(self.grid_nx)] for _ in range(self.grid_ny)]
            empty_conf = 0.98 if active_count > 0 else 0.0
            return SpatialTrackingResult(
                occupancy_state="EMPTY",
                confidence=empty_conf,
                target_count=0,
                count_display="0",
                topology_mode=topology_mode,
                uncertainty_mode="NONE",
                active_nodes_count=active_count,
                active_zone="NONE",
                mvs_current=round(mvs_cur, 8),
                threshold_on=round(th_on, 8),
                threshold_off=round(th_off, 8),
                intensity=round(max_intensity, 2),
                targets=[],
                heatmap_grid=zero_grid,
                grid_nx=self.grid_nx,
                grid_ny=self.grid_ny,
                cell_size_m=self.cell_size_m,
                spectral_modes_count=0,
                spatial_peaks_count=0,
                link_decorrelation=0.0,
            )

        any_motion = any(st.state == "MOTION" for _, st in active_pairs)
        global_state: OccupancyState = "MOTION" if any_motion else "PRESENCE_STATIC"
        intensity_clipped = float(np.clip(max_intensity, 0.0, 12.0))

        # Energias por link equalizadas (Z-Score RMS + Ganho Calibrado G_l ou compensação geométrica inicial)
        equalized_energies = self._compute_equalized_link_energies(active_pairs)

        # ======================================================================
        # ESTÁGIO 2A: 1 LINK ATIVO -> Zona/Halo de Incerteza Fresnel + Contagem Espectral
        # ======================================================================
        if topology_mode == "SINGLE_NODE_FRESNEL":
            rx_node, st_node = active_pairs[0]
            w_flat = self.w_matrix_by_id.get(rx_node.id)
            el_grid = w_flat.reshape((self.grid_ny, self.grid_nx)) if w_flat is not None else np.zeros_like(self.heatmap_ema)
            scale_1 = min(1.0, equalized_energies[0] / 3.0)
            self.heatmap_ema = 0.60 * (el_grid * scale_1) + 0.40 * self.heatmap_ema

            spectral_modes, link_decorr = self._estimate_spectral_and_decorrelation(active_pairs, equalized_energies)
            if global_state == "PRESENCE_STATIC":
                final_count = max(1, self._last_motion_count)
            else:
                raw_count = max(1, min(4, spectral_modes))
                final_count = self._apply_temporal_count_hysteresis(raw_count, now)
                self._last_motion_count = final_count

            self.peak_occupancy = max(self.peak_occupancy, final_count)
            conf = float(np.clip(0.84 + 0.03 * min(intensity_clipped, 4.5), 0.84, 0.96)) if global_state == "MOTION" else 0.87
            grid_list = [[round(float(v), 3) for v in row] for row in self.heatmap_ema]

            return SpatialTrackingResult(
                occupancy_state=global_state,
                confidence=round(conf, 2),
                target_count=final_count,
                count_display=self.format_count_display(final_count),
                topology_mode=topology_mode,
                uncertainty_mode="FRESNEL_HALO",
                active_nodes_count=1,
                active_zone="ZONA_FRESNEL_1",
                mvs_current=round(mvs_cur, 8),
                threshold_on=round(th_on, 8),
                threshold_off=round(th_off, 8),
                intensity=round(intensity_clipped, 2),
                targets=[],  # Renderiza halo elíptico honesto ao longo do link sem ponto rígido enganoso
                heatmap_grid=grid_list,
                grid_nx=self.grid_nx,
                grid_ny=self.grid_ny,
                cell_size_m=self.cell_size_m,
                spectral_modes_count=spectral_modes,
                spatial_peaks_count=1,
                link_decorrelation=0.0,
            )

        # ======================================================================
        # ESTÁGIO 2B: 2+ LINKS ATIVOS -> RTI 0.25m + NMS Multi-Pico + Hungarian + Kalman 2D
        # ======================================================================
        rti_norm_grid = self._reconstruct_rti_grid(active_pairs, equalized_energies)
        self.heatmap_ema = 0.65 * rti_norm_grid + 0.35 * self.heatmap_ema

        # Extração Multi-Alvo via NMS (raio humano mínimo 1.10 m)
        peaks_m = self._extract_nms_peaks(self.heatmap_ema, room_config.width_m, room_config.length_m)
        spatial_peaks_n = len(peaks_m)

        # Estimativa Híbrida de Contagem (Autovalores + Descorrelação entre Links + Picos Espaciais NMS)
        spectral_modes, link_decorr = self._estimate_spectral_and_decorrelation(active_pairs, equalized_energies)

        if global_state == "PRESENCE_STATIC":
            final_count = max(1, self._last_motion_count)
        else:
            ref_energies = [st.energy_ref_1p for _, st in active_pairs if st.energy_ref_1p > 0.2]
            ref_e_1p = float(np.median(ref_energies)) if ref_energies else 3.5
            raw_hybrid_count = self._fuse_hybrid_headcount(
                spatial_peaks_n=spatial_peaks_n,
                spectral_modes=spectral_modes,
                link_decorr=link_decorr,
                equalized_energies=equalized_energies,
                active_nodes_count=active_count,
                ref_energy_1p=ref_e_1p,
            )
            final_count = self._apply_temporal_count_hysteresis(raw_hybrid_count, now)
            self._last_motion_count = final_count

        self.peak_occupancy = max(self.peak_occupancy, final_count)

        # Atualiza Multi-Object Tracker (Hungarian + Filtro de Kalman 2D por alvo)
        targets_list = self._update_multi_object_tracks(
            peaks_m=peaks_m,
            target_headcount=final_count,
            global_state=global_state,
            room_config=room_config,
            now=now,
            hold_time_sec=hold_time_sec,
        )

        # Define zona ativa dominante
        if topology_mode == "DUAL_NODE_SECTOR":
            active_zone = self._classify_dual_sector(active_pairs, equalized_energies)
            uncertainty_mode: UncertaintyMode = "SECTOR_ZONE" if len(targets_list) == 0 else "POINT_2D"
        else:
            uncertainty_mode = "POINT_2D"
            if len(targets_list) == 1:
                active_zone = targets_list[0].zone
            elif len(targets_list) > 1:
                zones_unique = sorted(set(t.zone for t in targets_list))
                active_zone = "MULTI_ZONA" if len(zones_unique) > 1 else zones_unique[0]
            else:
                active_zone = "ZONA_CENTRAL"

        conf = (
            float(np.clip(0.85 + 0.03 * min(intensity_clipped, 4.5), 0.85, 0.99))
            if global_state == "MOTION"
            else 0.88
        )
        grid_list = [[round(float(v), 3) for v in row] for row in self.heatmap_ema]

        return SpatialTrackingResult(
            occupancy_state=global_state,
            confidence=round(conf, 2),
            target_count=final_count,
            count_display=self.format_count_display(final_count),
            topology_mode=topology_mode,
            uncertainty_mode=uncertainty_mode,
            active_nodes_count=active_count,
            active_zone=active_zone,
            mvs_current=round(mvs_cur, 8),
            threshold_on=round(th_on, 8),
            threshold_off=round(th_off, 8),
            intensity=round(intensity_clipped, 2),
            targets=targets_list,
            heatmap_grid=grid_list,
            grid_nx=self.grid_nx,
            grid_ny=self.grid_ny,
            cell_size_m=self.cell_size_m,
            spectral_modes_count=spectral_modes,
            spatial_peaks_count=spatial_peaks_n,
            link_decorrelation=round(link_decorr, 3),
        )

    # ==========================================================================
    # Equalização por Link, Reconstrução RTI 2D e Extração Multi-Alvo (NMS)
    # ==========================================================================

    def _compute_equalized_link_energies(
        self,
        active_pairs: List[Tuple[NodePosition, NodeRuntimeState]],
    ) -> np.ndarray:
        energies = np.zeros(len(active_pairs), dtype=np.float64)
        for idx, (rx, st) in enumerate(active_pairs):
            if st.calibrated_energy > 0.0:
                base_e = st.calibrated_energy
            elif st.state != "EMPTY":
                base_e = max(0.25, math.sqrt(max(0.0, st.intensity)) - 0.85) * st.gain_g
            else:
                base_e = 0.0

            # Se ganho empírico da Etapa 2 ainda for unitário (1.0), aplica compensação física de distância TX-RX
            if abs(st.gain_g - 1.0) < 1e-3:
                d_m = self.link_distances_m.get(rx.id, 3.6)
                geom_eq = float(np.clip((d_m / 3.6) ** 0.45, 0.55, 1.85))
                base_e *= geom_eq

            energies[idx] = base_e
        return energies

    def _reconstruct_rti_grid(
        self,
        active_pairs: List[Tuple[NodePosition, NodeRuntimeState]],
        equalized_energies: np.ndarray,
    ) -> np.ndarray:
        n_links = len(active_pairs)
        m_voxels = self.grid_ny * self.grid_nx
        W_sub = np.zeros((n_links, m_voxels), dtype=np.float64)

        for i, (rx, _) in enumerate(active_pairs):
            w_row = self.w_matrix_by_id.get(rx.id)
            if w_row is not None and w_row.size == m_voxels:
                W_sub[i, :] = w_row

        y_vec = np.maximum(0.02, equalized_energies)
        y_max = float(np.max(y_vec))
        if y_max <= 0.03:
            return np.zeros((self.grid_ny, self.grid_nx), dtype=np.float64)

        # Normalização de sensibilidade por coluna (evita concentração artificial na interseção próxima ao TX)
        col_sum = np.sum(W_sub, axis=0, keepdims=True)
        W_norm = W_sub / np.power(col_sum + 0.32, 0.58)

        # Inversão linear regularizada de Tikhonov no espaço dual
        gram = W_norm @ W_norm.T
        reg_lambda = 0.14 * (float(np.trace(gram)) / max(1, n_links)) + 1e-6
        dual_w = np.linalg.solve(gram + reg_lambda * np.eye(n_links), y_vec)
        voxel_raw = np.maximum(0.0, W_norm.T @ dual_w)

        # Veto tomográfico suave: links silenciosos atenuam células que cruzam sua 1ª zona de Fresnel
        quiet_thresh = 0.34 * y_max
        veto = np.ones(m_voxels, dtype=np.float64)
        for i in range(n_links):
            if y_vec[i] < quiet_thresh:
                quiet_ratio = 1.0 - (y_vec[i] / max(quiet_thresh, 1e-6))
                veto *= np.exp(-1.25 * W_sub[i, :] * quiet_ratio)

        img = (voxel_raw * veto).reshape((self.grid_ny, self.grid_nx))
        img = self._smooth_3x3(img)

        peak_v = float(np.max(img))
        if peak_v > 1e-6:
            return img / peak_v
        return np.zeros_like(img)

    @staticmethod
    def _smooth_3x3(grid: np.ndarray) -> np.ndarray:
        """Filtro Gaussiano 3x3 separável rápido ([0.25, 0.5, 0.25]) sem dependências extras."""
        padded = np.pad(grid, pad_width=1, mode="edge")
        horiz = 0.25 * padded[:, :-2] + 0.50 * padded[:, 1:-1] + 0.25 * padded[:, 2:]
        out = 0.25 * horiz[:-2, :] + 0.50 * horiz[1:-1, :] + 0.25 * horiz[2:, :]
        return out

    def _extract_nms_peaks(
        self,
        norm_grid: np.ndarray,
        width_m: float,
        length_m: float,
        max_peaks: int = 5,
    ) -> List[Tuple[float, float, float]]:
        """
        Extrai múltiplos picos simultâneos aplicando Non-Maximum Suppression (NMS)
        com raio mínimo humano de 1.10 m e interpolação sub-célula 3x3.
        Retorna lista de (x_m, y_m, score).
        """
        g_max = float(np.max(norm_grid))
        if g_max < 0.15:
            return []

        g_norm = norm_grid / g_max
        adapt_th = float(np.median(g_norm) + 1.15 * np.std(g_norm))
        peak_thresh = float(np.clip(max(0.32, adapt_th), 0.32, 0.66))

        ny, nx = g_norm.shape
        padded = np.pad(g_norm, pad_width=1, mode="constant", constant_values=-1.0)

        candidates: List[Tuple[float, int, int]] = []
        for iy in range(ny):
            for ix in range(nx):
                val = float(g_norm[iy, ix])
                if val < peak_thresh:
                    continue
                patch = padded[iy : iy + 3, ix : ix + 3]
                if val >= float(np.max(patch)) - 1e-9:
                    candidates.append((val, iy, ix))

        if not candidates:
            flat_idx = int(np.argmax(g_norm))
            iy, ix = divmod(flat_idx, nx)
            candidates.append((float(g_norm[iy, ix]), iy, ix))

        candidates.sort(key=lambda item: item[0], reverse=True)

        accepted: List[Tuple[float, float, float]] = []
        for score, iy, ix in candidates:
            y0, y1 = max(0, iy - 1), min(ny, iy + 2)
            x0, x1 = max(0, ix - 1), min(nx, ix + 2)
            w_patch = np.maximum(0.0, g_norm[y0:y1, x0:x1]) ** 2.0
            w_sum = float(np.sum(w_patch))
            if w_sum > 1e-8:
                px_m = float(np.sum(self.grid_x_m[y0:y1, x0:x1] * w_patch) / w_sum)
                py_m = float(np.sum(self.grid_y_m[y0:y1, x0:x1] * w_patch) / w_sum)
            else:
                px_m = float(self.grid_x_m[iy, ix])
                py_m = float(self.grid_y_m[iy, ix])

            px_m = float(np.clip(px_m, 0.25, width_m - 0.25))
            py_m = float(np.clip(py_m, 0.25, length_m - 0.25))

            if all(math.hypot(px_m - ax, py_m - ay) >= self.nms_radius_m for ax, ay, _ in accepted):
                accepted.append((px_m, py_m, score))
                if len(accepted) >= max_peaks:
                    break

        return accepted

    # ==========================================================================
    # Estimativa Híbrida de Contagem (Autovalores + Descorrelação + NMS + Histerese 2s)
    # ==========================================================================

    def _estimate_spectral_and_decorrelation(
        self,
        active_pairs: List[Tuple[NodePosition, NodeRuntimeState]],
        equalized_energies: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Analisa a matriz conjunta de subespaço (autovalores em janela de 1.0 s) e a
        descorrelação temporal entre links independentes.
        Retorna: (spectral_modes_people_est, mean_decorrelation_0_to_1).
        """
        # 1. Descorrelação temporal entre links ativos com energia relevante
        turb_vecs: List[np.ndarray] = []
        for idx, (_, st) in enumerate(active_pairs):
            if st.turb_1s_resampled is not None and equalized_energies[idx] > 0.35:
                std_t = float(np.std(st.turb_1s_resampled))
                if std_t > 1e-7:
                    turb_vecs.append(st.turb_1s_resampled / std_t)

        decorr_vals: List[float] = []
        for i in range(len(turb_vecs)):
            for j in range(i + 1, len(turb_vecs)):
                corr = float(np.mean(turb_vecs[i] * turb_vecs[j]))
                corr = float(np.clip(corr, -1.0, 1.0))
                decorr_vals.append(1.0 - abs(corr))
        mean_decorr = float(np.mean(decorr_vals)) if decorr_vals else 0.0

        # 2. Subespaço conjunto SNR-normalizado (SVD de 1.0s em todas as subportadoras ativas)
        joint_blocks: List[np.ndarray] = []
        single_node_2p_votes = 0
        for idx, (_, st) in enumerate(active_pairs):
            if st.sub12_1s_resampled is not None and st.state == "MOTION":
                floor_scale = math.sqrt(max(1e-7, st.eig_floor[0] if st.eig_floor else 2.5e-5))
                pkt_dem = st.sub12_1s_resampled - np.mean(st.sub12_1s_resampled, axis=1, keepdims=True)
                joint_blocks.append(pkt_dem / floor_scale)

            # Voto individual por envelope calibrado de autovalor secundário (lambda_2 > 1.55 * lambda_2_ref)
            if len(st.eigenvalues) >= 2 and st.eig2_ref_1p > 0:
                floor2 = st.eig_floor[1] if len(st.eig_floor) > 1 else 1e-5
                excess_l2 = max(0.0, st.eigenvalues[1] - floor2)
                ref_span = max(1e-6, st.eig2_ref_1p - floor2)
                if excess_l2 > 1.65 * ref_span:
                    single_node_2p_votes += 1

        pr_joint = 1.0
        mu3_ratio = 0.0
        mu5_ratio = 0.0
        if joint_blocks:
            X_joint = np.concatenate(joint_blocks, axis=1)
            # Singular values -> autovalores da matriz de covariância conjunta
            s_vals = np.linalg.svd(X_joint, compute_uv=False)
            mu_vals = (s_vals ** 2) / float(max(1, X_joint.shape[0] - 1))
            # Subtrai piso unitário de ruído branco
            mu_clean = np.maximum(0.0, mu_vals - 1.2)
            sum_mu = float(np.sum(mu_clean[:8]))
            sum_sq = float(np.sum(mu_clean[:8] ** 2))
            if sum_sq > 1e-9 and mu_clean[0] > 2.5:
                pr_joint = (sum_mu ** 2) / sum_sq
                mu3_ratio = float(mu_clean[2] / mu_clean[0]) if mu_clean.size > 2 else 0.0
                mu5_ratio = float(mu_clean[4] / mu_clean[0]) if mu_clean.size > 4 else 0.0

        # Cada pessoa em movimento excita ~2 modos harmônicos ortogonais + dispersão Doppler entre links
        if (pr_joint >= 3.05 and mu5_ratio >= 0.038) or (pr_joint >= 3.40):
            spectral_people = 3
        elif (pr_joint >= 2.50 and mu3_ratio >= 0.065) or single_node_2p_votes >= 2 or (
            len(active_pairs) == 1 and single_node_2p_votes >= 1 and pr_joint >= 2.35
        ):
            spectral_people = 2
        else:
            spectral_people = 1

        return spectral_people, mean_decorr

    def _fuse_hybrid_headcount(
        self,
        spatial_peaks_n: int,
        spectral_modes: int,
        link_decorr: float,
        equalized_energies: np.ndarray,
        active_nodes_count: int,
        ref_energy_1p: float = 3.5,
    ) -> int:
        """
        Fusão por Consenso Físico Ponderado entre 3 estimadores independentes:
        1. N_spat: contagem de picos espaciais NMS no grid RTI (raio mínimo humano 1.1 m).
        2. N_spec: modos espectrais (autovalores da covariância em janela de 1.0 s).
        3. N_energy: links independentes com ocupação direta (>= 0.72 * ref_e) + descorrelação temporal.
        """
        ref_e = max(1.2, float(ref_energy_1p))
        total_norm_e = float(np.sum(equalized_energies)) / ref_e
        direct_links = int(np.sum(equalized_energies >= 0.72 * ref_e))
        moderate_links = int(np.sum(equalized_energies >= 0.40 * ref_e))

        # 1. Proteção rigorosa contra superestimação de 1 pessoa:
        # Se há apenas 1 pico espacial no grid RTI e no máximo 1 link diretamente ocupado (ou energia < 2.45x ref), é 1 pessoa
        if spatial_peaks_n <= 1 and direct_links <= 1 and spectral_modes <= 2:
            return 1
        if spatial_peaks_n <= 1 and total_norm_e < 2.45 and spectral_modes <= 2:
            return 1

        n_spat = max(1, min(4, int(spatial_peaks_n)))
        n_spec = max(1, min(3, int(spectral_modes)))

        if active_nodes_count >= 3 and (
            (direct_links >= 3 and link_decorr >= 0.48 and total_norm_e >= 2.80)
            or (moderate_links >= 3 and link_decorr >= 0.56 and total_norm_e >= 3.35)
        ):
            n_energy = 4 if (total_norm_e >= 4.45 and n_spat >= 3) else 3
        elif (
            direct_links >= 2
            or (moderate_links >= 2 and total_norm_e >= 2.05)
            or (moderate_links >= 2 and link_decorr >= 0.58 and n_spat >= 2 and n_spec >= 2)
        ) and (total_norm_e >= 1.70 or link_decorr >= 0.50):
            n_energy = 2
        else:
            n_energy = 1

        # Consenso ponderado: exige pelo menos 2 estimadores concordando em 3+ (ou 3 links diretos fortes)
        consensus = 0.36 * n_spat + 0.30 * n_spec + 0.34 * n_energy

        votes_for_3 = int(n_spat >= 3) + int(n_spec >= 3) + int(n_energy >= 3)
        if votes_for_3 >= 2 or (n_energy >= 3 and total_norm_e >= 3.15 and link_decorr >= 0.55):
            consensus = max(consensus, 2.62)
        elif n_spec >= 2 and n_energy >= 2:
            consensus = max(consensus, 1.72)

        if consensus >= 3.45:
            return 4
        if consensus >= 2.50:
            return 3
        if consensus >= 1.54:
            return 2
        return 1

    def _apply_temporal_count_hysteresis(self, raw_count: int, now: float) -> int:
        """Aplica janela de mediana/histerese temporal de 2.0 s na contagem final."""
        self._count_history.append((now, int(raw_count)))
        cutoff = now - settings.HEADCOUNT_HYSTERESIS_SEC
        while self._count_history and self._count_history[0][0] < cutoff:
            self._count_history.popleft()

        vals = [c for _, c in self._count_history]
        if not vals:
            self._committed_count = raw_count
            return raw_count

        if self._committed_count == 0:
            self._committed_count = raw_count
            return raw_count

        med = int(round(float(np.median(vals))))
        if med != self._committed_count:
            # Exige maioria consistente na janela recente para transicionar entre 1 <-> 2 <-> 3+
            recent_1s = [c for ts, c in self._count_history if ts >= (now - 1.1)]
            if len(recent_1s) >= 4:
                agree_ratio = sum(1 for c in recent_1s if c == med) / float(len(recent_1s))
                if agree_ratio >= 0.60:
                    self._committed_count = med
            elif len(vals) < 4:
                self._committed_count = med

        return max(1, self._committed_count)

    # ==========================================================================
    # Multi-Object Tracking: Hungarian Association + Per-Target 2D Kalman Filter
    # ==========================================================================

    def _update_multi_object_tracks(
        self,
        peaks_m: List[Tuple[float, float, float]],
        target_headcount: int,
        global_state: OccupancyState,
        room_config: RoomConfig,
        now: float,
        hold_time_sec: float,
    ) -> List[SpatialTarget]:
        w_m = room_config.width_m
        l_m = room_config.length_m

        # Se todo o ambiente está em PRESENCE_STATIC, congela suavemente os alvos ativos via hold-timer
        if global_state == "PRESENCE_STATIC":
            for tr in self._tracks:
                tr.kalman.freeze_static(now)
                tr.state = "PRESENCE_STATIC"
                tr.coast_until_ts = max(tr.coast_until_ts, now + min(hold_time_sec, 15.0))
            self._tracks = [tr for tr in self._tracks if now <= tr.coast_until_ts]
            return self._serialize_tracks(self._tracks[:target_headcount], w_m, l_m)

        # Predição Kalman de todos os rastros ativos
        predicted_positions: List[Tuple[float, float]] = []
        for tr in self._tracks:
            px, py = tr.kalman.predict(now, w_m, l_m)
            predicted_positions.append((px, py))

        n_tracks = len(self._tracks)
        n_peaks = len(peaks_m)
        matched_tracks = set()
        matched_peaks = set()

        if n_tracks > 0 and n_peaks > 0:
            cost_mat = np.zeros((n_tracks, n_peaks), dtype=np.float64)
            for ti, (px, py) in enumerate(predicted_positions):
                for pi, (zx, zy, pscore) in enumerate(peaks_m):
                    dist = math.hypot(px - zx, py - zy)
                    cost_mat[ti, pi] = dist - 0.12 * pscore

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            max_gate_dist_m = 1.75

            for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                px, py = predicted_positions[r]
                zx, zy, pscore = peaks_m[c]
                if math.hypot(px - zx, py - zy) <= max_gate_dist_m:
                    tr = self._tracks[r]
                    tr.kalman.update(zx, zy, now, w_m, l_m)
                    tr.hits += 1
                    tr.last_matched_ts = now
                    tr.coast_until_ts = now + settings.TRACK_COAST_SEC
                    tr.state = "MOTION"
                    tr.confidence = float(np.clip(0.82 + 0.15 * pscore, 0.80, 0.99))
                    matched_tracks.add(r)
                    matched_peaks.add(c)

        # Cria novos rastros para picos não associados (respeitando espaçamento humano e headcount)
        for pi, (zx, zy, pscore) in enumerate(peaks_m):
            if pi in matched_peaks:
                continue
            if len(self._tracks) >= max(target_headcount + 1, 4):
                continue
            too_close = any(
                math.hypot(zx - float(t.kalman.state[0]), zy - float(t.kalman.state[1])) < 0.85
                for t in self._tracks
            )
            if too_close:
                continue
            new_id = f"P{self._next_track_num}"
            self._next_track_num += 1
            self._tracks.append(
                TrackedPerson(
                    track_id=new_id,
                    kalman=MetricKalmanFilter2D(zx, zy, now=now),
                    hits=1,
                    created_ts=now,
                    last_matched_ts=now,
                    coast_until_ts=now + settings.TRACK_COAST_SEC,
                    confidence=float(np.clip(0.80 + 0.14 * pscore, 0.78, 0.96)),
                    state="MOTION",
                )
            )

        # Remove rastros cujo tempo de coast expirou; se alguém apenas parou enquanto outro anda,
        # estende coast se a contagem híbrida confirmar que continuam N pessoas na sala
        alive_tracks: List[TrackedPerson] = []
        for idx, tr in enumerate(self._tracks):
            if idx not in matched_tracks and (now - tr.last_matched_ts) > 0.45:
                tr.kalman.state[2] *= 0.75
                tr.kalman.state[3] *= 0.75
                if len(self._tracks) <= target_headcount and tr.hits >= 3:
                    tr.state = "PRESENCE_STATIC"
                    tr.coast_until_ts = max(tr.coast_until_ts, now + min(3.0, hold_time_sec * 0.4))
            if now <= tr.coast_until_ts:
                alive_tracks.append(tr)

        # Ordena por maturidade/confiança e seleciona até target_headcount rastros confirmados
        alive_tracks.sort(key=lambda t: (t.hits >= 2, t.last_matched_ts, t.hits), reverse=True)
        self._tracks = alive_tracks[: max(target_headcount + 1, 4)]

        confirmed = [t for t in self._tracks if t.hits >= 2 or len(self._tracks) <= target_headcount]
        selected = confirmed[:target_headcount]
        selected.sort(key=lambda t: t.track_id)
        return self._serialize_tracks(selected, w_m, l_m)

    def _serialize_tracks(
        self,
        tracks: List[TrackedPerson],
        width_m: float,
        length_m: float,
    ) -> List[SpatialTarget]:
        out: List[SpatialTarget] = []
        for tr in tracks:
            xm = float(tr.kalman.state[0])
            ym = float(tr.kalman.state[1])
            vx = float(tr.kalman.state[2])
            vy = float(tr.kalman.state[3])
            nx = float(np.clip(xm / max(1.0, width_m), 0.02, 0.98))
            ny = float(np.clip(ym / max(1.0, length_m), 0.02, 0.98))
            out.append(
                SpatialTarget(
                    id=tr.track_id,
                    x=round(nx, 3),
                    y=round(ny, 3),
                    x_m=round(xm, 2),
                    y_m=round(ym, 2),
                    vx=round(vx, 2),
                    vy=round(vy, 2),
                    state=tr.state,
                    confidence=round(tr.confidence, 2),
                    zone=self._classify_2d_zone(nx, ny),
                )
            )
        return out

    def _classify_dual_sector(
        self,
        active_pairs: List[Tuple[NodePosition, NodeRuntimeState]],
        equalized_energies: np.ndarray,
    ) -> str:
        (rx_a, st_a), (rx_b, st_b) = active_pairs[0], active_pairs[1]
        ia_raw = max(0.05, float(equalized_energies[0]))
        ib_raw = max(0.05, float(equalized_energies[1]))
        ia = 0.45 * ia_raw + 0.55 * self._sector_ema.get(rx_a.id, ia_raw)
        ib = 0.45 * ib_raw + 0.55 * self._sector_ema.get(rx_b.id, ib_raw)
        self._sector_ema[rx_a.id] = ia
        self._sector_ema[rx_b.id] = ib

        both_active = (st_a.state != "EMPTY" and st_b.state != "EMPTY") or (ia >= 0.75 and ib >= 0.75)
        ratio_ab = ia / (ib + 1e-6)
        if both_active and (0.58 <= ratio_ab <= 1.72):
            return "INTERSECAO"
        return "ZONA_A" if ia >= ib else "ZONA_B"

    @staticmethod
    def _classify_2d_zone(x: float, y: float) -> str:
        if math.hypot(x - 0.5, y - 0.5) <= 0.20:
            return "ZONA_CENTRAL"
        if y < 0.42:
            return "ZONA_NORTE"
        if y > 0.66:
            return "ZONA_SUL"
        if x < 0.5:
            return "ZONA_OESTE"
        return "ZONA_LESTE"
