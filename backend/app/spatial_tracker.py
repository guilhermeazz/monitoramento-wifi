import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from scipy import ndimage

from .config import settings
from .dsp_engine import LinkDSPMetrics
from .models import (
    CoverageZoneInfo,
    GlobalDecision,
    NodePosition,
    PersonCoordinate,
    RoomConfig,
    TopologyInfo,
)


@dataclass
class TrackedTarget:
    """Estado interno de um alvo rastreado pelo filtro cinemático EMA/Kalman 2D."""

    track_id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    confidence: float = 0.75
    motion_ratio: float = 0.5
    longitudinal_t: float = 0.5
    last_updated: float = field(default_factory=time.time)
    hit_streak: int = 1
    missed_frames: int = 0


class SpatialTracker:
    """
    Motor de Topologia Auto-Adaptativa (Adaptive Sensing Topology),
    Auto-Mapeamento da Área de Abrangência via Log-Distance Path Loss,
    Estimador Multi-Pessoa em Feixe Único (subcarrier_diversity + Histerese de 2.5s)
    e Rastreamento Espacial em Grid Normalizado 20x20.
    """

    def __init__(self, grid_size: int = settings.GRID_RESOLUTION):
        self.grid_size = grid_size

        lin = np.linspace(0.5 / grid_size, 1.0 - 0.5 / grid_size, grid_size)
        self.grid_x, self.grid_y = np.meshgrid(lin, lin)

        self.heatmap_ema = np.zeros((grid_size, grid_size), dtype=np.float64)
        self.dynamic_map_ema = np.zeros((grid_size, grid_size), dtype=np.float64)

        self.tracks: Dict[str, TrackedTarget] = {}
        self._next_id_counter: int = 1
        self._last_track_time: float = time.time()
        self._last_mode_code: str = "SINGLE_LINK_CORRIDOR"

        # Filtro de Histerese Temporal para evitar oscilações bruscas no contador
        self._hysteresis_count: int = 0
        self._hysteresis_hold_until: float = 0.0
        self._pending_up_count: int = 0
        self._pending_up_since: float = 0.0

    def force_reset_tracks(self) -> None:
        """Zera instantaneamente a matriz de calor, os alvos rastreados e a histerese (Botão Forçar Zero)."""
        self.heatmap_ema.fill(0.0)
        self.dynamic_map_ema.fill(0.0)
        self.tracks.clear()
        self._hysteresis_count = 0
        self._hysteresis_hold_until = 0.0
        self._pending_up_count = 0
        self._pending_up_since = 0.0

    @staticmethod
    def classify_topology(active_nodes_count: int, avg_dist_m: float = 3.2, area_m2: float = 11.5) -> TopologyInfo:
        """
        Classificador Automático de Topologia (Máquina de Estados de Sensores Ativos):
        - N = 0 -> NO_SENSORS
        - N = 1 -> SINGLE_LINK_CORRIDOR (com Multi-Contagem via Entropia Espectral)
        - N = 2 -> DUAL_LINK_SECTOR
        - N >= 3 -> FULL_MESH_2D
        """
        if active_nodes_count <= 0:
            return TopologyInfo(
                active_nodes=0,
                mode_code="NO_SENSORS",
                mode_title="Em Espera (0 ESP32 Ativos)",
                precision_level="Aguardando Pacotes UDP na Porta 5555",
                capabilities_enabled=[
                    "Servidor UDP escutando pacotes CSI na porta 5555",
                    "Auto-mapeamento instantâneo de distância e área assim que o ESP32 conectar",
                ],
                limitations=[
                    "Nenhum pacote CSI recebido nos últimos 3.0 segundos",
                    "Matriz de calor zerada até conexão de hardware ou ativação do Simulador",
                ],
                setup_recommendation="Conecte ao menos 1 ESP32 na rede 2.4 GHz ou use os botões rápidos para ativar o Simulador.",
            )

        if active_nodes_count == 1:
            return TopologyInfo(
                active_nodes=1,
                mode_code="SINGLE_LINK_CORRIDOR",
                mode_title="1 ESP: Modo Corredor Fresnel",
                precision_level=f"Elipse de Fresnel (~{area_m2:.1f} m²) + Autovalores PCA (SVD)",
                capabilities_enabled=[
                    "Separação Multi-Pessoa (0, 1, 2, 3+) via Autovalores PCA (s1, s2, s3) e CSI Ratio",
                    "Trava Anti-Drop de Pessoa Parada (Banda Vital 0.15-0.55Hz + Sombra Estática)",
                    f"Auto-mapeamento de distância TX-RX (~{avg_dist_m:.1f}m) via Log-Distance Path Loss",
                ],
                limitations=[
                    "Coordenadas distribuídas dentro da elipse de Fresnel (adicione +1 ou +2 ESP32 para triangulação 2D livre)",
                ],
                setup_recommendation="Execute a Etapa 1 (Ambiente/Plateia) e a Etapa 2 (1 Pessoa Andando) para calibrar os autovalores PCA da sala.",
            )

        if active_nodes_count == 2:
            return TopologyInfo(
                active_nodes=2,
                mode_code="DUAL_LINK_SECTOR",
                mode_title="2 ESPs: Modo Setorial",
                precision_level=f"Interpolação Vetorial 2D (~{area_m2:.1f} m²) + PCA",
                capabilities_enabled=[
                    "Separação de zonas laterais (Esquerda/Direita) e interpolação vetorial 2D",
                    f"Auto-mapeamento das distâncias TX-RX (~{avg_dist_m:.1f}m)",
                    "Contagem multi-pessoa com retenção de presença parada nos 2 feixes",
                ],
                limitations=[
                    "Menor resolução nas bordas externas fora do setor em 'V' dos 2 sensores",
                ],
                setup_recommendation="Posicione os 2 ESP32 formando um 'V' aberto em frente ao Roteador TX.",
            )

        return TopologyInfo(
            active_nodes=active_nodes_count,
            mode_code="FULL_MESH_2D",
            mode_title="3+ ESPs: Malha 2D Completa",
            precision_level=f"Triangulação 2D Completa (~{area_m2:.1f} m²)",
            capabilities_enabled=[
                "Triangulação espacial 2D (X, Y) em toda a área de abrangência",
                "Separação de múltiplos indivíduos por clustering tomográfico 20x20",
                f"Auto-mapeamento perimetral automático (~{avg_dist_m:.1f}m TX-RX)",
            ],
            limitations=[
                "Resolução física limitada pelo comprimento de onda em 2.4 GHz (~12.5 cm)",
            ],
            setup_recommendation="Distribua os 3 módulos ESP32 cercando a área de interesse oposta ao Roteador TX.",
        )

    def auto_map_coverage_zone(
        self,
        room_config: RoomConfig,
        active_rx_pairs: List[Tuple[NodePosition, LinkDSPMetrics]],
    ) -> CoverageZoneInfo:
        """
        Calcula a atenuação Log-Distance Path Loss, distâncias estimadas e Área de Abrangência Ativa em m²,
        preservando o posicionamento manual/tático dos nós de hardware configurado pelo usuário.
        """
        lateral_r = settings.LATERAL_SENSITIVITY_RADIUS_M  # 1.8m

        # Inicializa coordenadas do TX apenas se ainda não configuradas (x=0, y=0)
        if room_config.tx.x == 0.0 and room_config.tx.y == 0.0:
            room_config.tx.x = 0.50
            room_config.tx.y = 0.12

        if not active_rx_pairs:
            return CoverageZoneInfo(
                active_area_m2=0.0,
                avg_tx_rx_distance_m=0.0,
                lateral_radius_m=lateral_r,
                auto_mapped=True,
                node_distances_m={},
                description="Área de Abrangência Inativa (Aguardando conexão de sensores ESP32)",
            )

        node_distances: Dict[str, float] = {}
        for rx, m in active_rx_pairs:
            node_distances[rx.id] = round(float(m.estimated_distance_m), 2)
            # Inicializa coordenadas do receptor apenas se virgem (x=0, y=0), respeitando o layout manual do usuário
            if rx.x == 0.0 and rx.y == 0.0:
                if rx.id == "esp_1":
                    rx.x, rx.y = 0.15, 0.84  # Ponta esquerda
                elif rx.id == "esp_2":
                    rx.x, rx.y = 0.85, 0.84  # Ponta direita
                elif rx.id == "esp_3":
                    rx.x, rx.y = 0.50, 0.88  # Centro
                else:
                    rx.x, rx.y = 0.50, 0.84

        avg_dist = float(np.mean(list(node_distances.values())))
        avg_dist = float(np.clip(avg_dist, 1.0, 10.0))
        n_active = len(active_rx_pairs)

        # Cálculo da Área de Abrangência Ativa (Elipse de Fresnel)
        target_length_m = round(float(np.clip(avg_dist / 0.70, 3.2, 14.0)), 1)
        if n_active == 1:
            target_width_m = round(float(np.clip(lateral_r * 2.0 + 1.4, 4.2, 10.0)), 1)
            semi_major = (avg_dist * 0.5) + 0.85
            semi_minor = lateral_r
            active_area_m2 = round(math.pi * semi_major * semi_minor, 1)
        elif n_active == 2:
            target_width_m = round(float(np.clip(avg_dist * 1.25 + lateral_r, 4.8, 12.0)), 1)
            semi_major = (avg_dist * 0.55) + 0.90
            active_sector_width = lateral_r * 1.55
            active_area_m2 = round(math.pi * semi_major * active_sector_width, 1)
        else:
            target_width_m = round(float(np.clip(avg_dist * 1.40 + lateral_r, 5.2, 14.0)), 1)
            active_area_m2 = round(min(target_width_m * target_length_m * 0.78, math.pi * (avg_dist * 0.68) * (lateral_r * 1.9)), 1)

        # Ajuste suave nas dimensões métricas sugeridas
        room_config.width_m = round(0.90 * room_config.width_m + 0.10 * target_width_m, 1)
        room_config.length_m = round(0.90 * room_config.length_m + 0.10 * target_length_m, 1)

        desc = f"Área de Abrangência Ativa: ~{active_area_m2:.1f} m² (Distância TX-RX: {avg_dist:.1f} m)"
        return CoverageZoneInfo(
            active_area_m2=active_area_m2,
            avg_tx_rx_distance_m=round(avg_dist, 2),
            lateral_radius_m=lateral_r,
            auto_mapped=True,
            node_distances_m=node_distances,
            description=desc,
        )

    @staticmethod
    def classify_zone(x: float, y: float, mode_code: str = "FULL_MESH_2D", longitudinal_t: float = 0.5) -> str:
        """Classifica a região semântica dentro da Zona de Abrangência Ativa."""
        if mode_code == "SINGLE_LINK_CORRIDOR":
            if longitudinal_t < 0.36:
                return "Zona de Abrangência (Próx. Roteador TX)"
            if longitudinal_t > 0.66:
                return "Zona de Abrangência (Próx. ESP32 RX)"
            return "Zona de Abrangência (Centro do Feixe)"

        dist_center = np.hypot(x - 0.5, y - 0.5)
        if dist_center <= 0.18:
            return "Centro da Abrangência"
        if x < 0.5 and y < 0.5:
            return "Setor Noroeste"
        if x >= 0.5 and y < 0.5:
            return "Setor Nordeste"
        if x < 0.5 and y >= 0.5:
            return "Setor Sudoeste"
        return "Setor Sudeste"

    def _apply_count_hysteresis(
        self,
        raw_count: int,
        max_perturbation: float,
        now: float,
    ) -> int:
        """
        Filtro de Histerese Temporal e Anti-Glitch (2.0 segundos):
        - Transição de 0 para >=1: exige persistência de pelo menos 0.15s (2 frames consecutivos a 10 Hz)
          para eliminar 100% de picos e ghosts transientes de 50ms causados por jitter/bursts de RF.
        - Quando confirmado, sustenta e renova a janela de 2.0s enquanto houver energia ou retenção.
        - Quando o sinal decai, mantém por 2.0s para evitar oscilações rápidas entre frames.
        """
        if max_perturbation < 0.05 and raw_count == 0:
            if (now - self._hysteresis_hold_until) > 0.0:
                self._hysteresis_count = 0
                self._pending_up_count = 0
                return 0

        if raw_count >= self._hysteresis_count and raw_count > 0:
            if raw_count > self._hysteresis_count:
                # Requer persistência de pelo menos 0.15s para subir a contagem (filtra ruído transiente de 1 único frame)
                if self._pending_up_count != raw_count:
                    self._pending_up_count = raw_count
                    self._pending_up_since = now
                elif (now - self._pending_up_since) >= 0.15:
                    self._hysteresis_count = raw_count
                    self._hysteresis_hold_until = now + settings.COUNT_HYSTERESIS_SEC
            else:
                # Mantém e renova o timer de histerese
                self._hysteresis_hold_until = now + settings.COUNT_HYSTERESIS_SEC
                self._pending_up_count = raw_count
        else:
            # raw_count < self._hysteresis_count: segura pelo tempo de histerese
            if now >= self._hysteresis_hold_until:
                self._hysteresis_count = raw_count
                if raw_count > 0:
                    self._hysteresis_hold_until = now + settings.COUNT_HYSTERESIS_SEC * 0.6

        return self._hysteresis_count

    def _compute_global_decision(
        self,
        active_rx_pairs: List[Tuple[NodePosition, LinkDSPMetrics]],
        room_config: RoomConfig,
    ) -> GlobalDecision:
        """
        Fusão Científica Multi-Sensor Espaço-Temporal via SVD Global (WiCount / Widar):
        Evita a sobrecontagem (3 ESP32 vendo 1 pessoa = 1 pessoa, e NÃO 3 pessoas!).
        - 1 Pessoa em movimento: produz um único componente temporal dominante (λ1 >> λ2, ratio < 0.38).
        - 2 Pessoas em movimento: introduzem trajetórias Doppler independentes ortogonais (ratio >= 0.38 ou λ2 alto).
        """
        if not active_rx_pairs:
            return GlobalDecision(
                decision_people_count=0,
                decision_reason="Sem sensores ativos",
                global_svd_s1=0.0,
                global_svd_s2=0.0,
                global_svd_ratio=0.0,
                global_energy_ratio=0.0,
                nodes_triggered_count=0,
                dominant_node_id=None,
            )

        triggered_pairs = [
            (rx, m) for rx, m in active_rx_pairs
            if m.is_triggered or m.stationary_retention_active or (m.net_motion_energy > 0.0 and m.z_score >= 3.8)
        ]
        triggered_count = len(triggered_pairs)

        if triggered_count == 0:
            return GlobalDecision(
                decision_people_count=0,
                decision_reason="Ambiente Livre (Todos os links abaixo do Gate Z-Score)",
                global_svd_s1=0.0,
                global_svd_s2=0.0,
                global_svd_ratio=0.0,
                global_energy_ratio=0.0,
                nodes_triggered_count=0,
                dominant_node_id=None,
            )

        # Nó com maior energia líquida de perturbação
        dominant_rx, dominant_m = max(active_rx_pairs, key=lambda pair: pair[1].net_motion_energy)
        dominant_node_id = dominant_rx.id

        # Matriz espaço-temporal multi-sensor concatenada: cada nó ativo entra com seus componentes principais
        feature_rows: List[List[float]] = []
        for rx, m in active_rx_pairs:
            weight = max(0.15, m.perturbation_score)
            feature_rows.append([
                float(m.pca_s1) * weight,
                float(m.pca_s2) * weight,
                float(m.pca_s3) * weight,
            ])

        x_mat = np.array(feature_rows, dtype=np.float64)
        try:
            _, s_vals, _ = np.linalg.svd(x_mat, full_matrices=False)
            lambda1 = float(s_vals[0]) if s_vals.size >= 1 else float(dominant_m.pca_s1)
            lambda2 = float(s_vals[1]) if s_vals.size >= 2 else float(dominant_m.pca_s2)
            lambda3 = float(s_vals[2]) if s_vals.size >= 3 else 0.0
        except Exception:
            lambda1 = float(dominant_m.pca_s1)
            lambda2 = float(dominant_m.pca_s2)
            lambda3 = 0.0

        svd_ratio = float(lambda2 / (lambda1 + 1e-6)) if lambda1 > 0.05 else 0.0

        energies = [m.energy_ratio for rx, m in triggered_pairs]
        mean_energy_ratio = float(np.mean(energies)) if energies else 0.0
        max_energy_ratio = float(np.max(energies)) if energies else 0.0
        max_div = max((m.subcarrier_diversity for rx, m in active_rx_pairs), default=0.0)
        s2_over_empty = max((m.pca_s2 / max(0.08, m.pca_s2_empty_floor) for rx, m in active_rx_pairs), default=1.0)

        # Regra de decisão científica (WiCount):
        # 1 PESSOA: λ1 domina fortemente e diversidade/energia são nominais de 1 corpo físico
        is_single_person = (svd_ratio < 0.35) and (mean_energy_ratio < 1.85) and (max_div < 0.48) and (s2_over_empty < 2.3)

        if is_single_person:
            people_count = 1
            reason = (
                f"1 Pessoa Confirmada (SVD Dominante: λ1={lambda1:.2f}, λ2={lambda2:.2f}, "
                f"Ratio={svd_ratio:.2f} < 0.35 — {triggered_count} ESP(s) no mesmo alvo)"
            )
        elif (svd_ratio >= 0.32) or (mean_energy_ratio >= 1.80 and (max_div >= 0.45 or s2_over_empty >= 2.2)):
            if mean_energy_ratio >= 2.50 or svd_ratio >= 0.55:
                people_count = 3
                reason = (
                    f"3+ Pessoas Confirmadas (SVD Multi-Alvo: λ1={lambda1:.2f}, λ2={lambda2:.2f}, "
                    f"Ratio={svd_ratio:.2f}, E_ratio={mean_energy_ratio:.2f})"
                )
            else:
                people_count = 2
                reason = (
                    f"2 Pessoas Confirmadas (SVD Multi-Alvo: λ1={lambda1:.2f}, λ2={lambda2:.2f}, "
                    f"Ratio={svd_ratio:.2f}, E_ratio={mean_energy_ratio:.2f})"
                )
        else:
            people_count = 1
            reason = (
                f"1 Pessoa Confirmada (SVD Dominante: λ1={lambda1:.2f}, λ2={lambda2:.2f}, "
                f"Ratio={svd_ratio:.2f})"
            )

        return GlobalDecision(
            decision_people_count=people_count,
            decision_reason=reason,
            global_svd_s1=round(lambda1, 3),
            global_svd_s2=round(lambda2, 3),
            global_svd_ratio=round(svd_ratio, 3),
            global_energy_ratio=round(mean_energy_ratio, 2),
            nodes_triggered_count=triggered_count,
            dominant_node_id=dominant_node_id,
        )

    def update(
        self,
        room_config: RoomConfig,
        node_metrics: Dict[str, LinkDSPMetrics],
        is_calibrating: bool = False,
        now: float | None = None,
    ) -> Tuple[List[List[float]], List[PersonCoordinate], str, TopologyInfo, CoverageZoneInfo, GlobalDecision]:
        """
        Auto-mapeia a Zona de Abrangência via RSSI/CSI, classifica a topologia e executa
        o rastreamento multi-pessoa com fusão SVD global e filtro de histerese temporal.
        """
        if now is None:
            now = time.time()
        dt = max(0.02, min(0.5, now - self._last_track_time))
        self._last_track_time = now

        active_rx_pairs: List[Tuple[NodePosition, LinkDSPMetrics]] = []
        for rx in room_config.rx_nodes:
            m = node_metrics.get(rx.id)
            if m is not None and m.online:
                active_rx_pairs.append((rx, m))

        active_count = len(active_rx_pairs)
        coverage_info = self.auto_map_coverage_zone(room_config, active_rx_pairs)
        topology_info = self.classify_topology(
            active_nodes_count=active_count,
            avg_dist_m=coverage_info.avg_tx_rx_distance_m,
            area_m2=coverage_info.active_area_m2,
        )

        if topology_info.mode_code != self._last_mode_code:
            if topology_info.mode_code == "NO_SENSORS":
                self.tracks.clear()
                self._hysteresis_count = 0
            self._last_mode_code = topology_info.mode_code

        if active_count == 0:
            self.heatmap_ema.fill(0.0)
            self.dynamic_map_ema.fill(0.0)
            self.tracks.clear()
            self._hysteresis_count = 0
            zero_grid = [[0.0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            empty_decision = GlobalDecision(
                decision_people_count=0,
                decision_reason="Aguardando conexão de ESP32 ou Simulador",
            )
            return zero_grid, [], "Sem Sensores", topology_info, coverage_info, empty_decision

        if is_calibrating:
            self.heatmap_ema *= 0.4
            self.dynamic_map_ema *= 0.4
            self.tracks.clear()
            self._hysteresis_count = 0
            grid_list = [[round(float(val), 3) for val in row] for row in self.heatmap_ema]
            cal_decision = GlobalDecision(
                decision_people_count=0,
                decision_reason="Calibração científica em andamento...",
            )
            return grid_list, [], "Calibrando...", topology_info, coverage_info, cal_decision

        # Fusão SVD Global Científica (WiCount / Widar)
        global_decision = self._compute_global_decision(active_rx_pairs, room_config)

        any_retained = any(m.stationary_retention_active for _, m in active_rx_pairs)
        max_pert = max([m.perturbation_score for _, m in active_rx_pairs] or [0.0])

        if global_decision.decision_people_count == 0 and not any_retained:
            self.force_reset_tracks()
            self.heatmap_ema *= 0.30
            self.dynamic_map_ema *= 0.30
            self.tracks.clear()
            self._hysteresis_count = 0
            grid_list = [[round(float(np.clip(val, 0.0, 1.0)), 3) for val in row] for row in self.heatmap_ema]
            return grid_list, [], "Zona de Abrangência Livre", topology_info, coverage_info, global_decision

        stable_count = self._apply_count_hysteresis(global_decision.decision_people_count, max_pert, now)

        if stable_count == 0 and not any_retained:
            self.force_reset_tracks()
            self.heatmap_ema *= 0.30
            self.dynamic_map_ema *= 0.30
            self.tracks.clear()
            self._hysteresis_count = 0
            grid_list = [[round(float(np.clip(val, 0.0, 1.0)), 3) for val in row] for row in self.heatmap_ema]
            return grid_list, [], "Zona de Abrangência Livre", topology_info, coverage_info, global_decision

        if topology_info.mode_code == "SINGLE_LINK_CORRIDOR":
            rx_node, m_single = active_rx_pairs[0]
            heatmap, dyn_map, detections = self._process_single_link_corridor(
                room_config=room_config,
                rx_node=rx_node,
                metrics=m_single,
                target_people_count=stable_count,
            )
            max_targets = max(1, stable_count) if stable_count > 0 else 0
        elif topology_info.mode_code == "DUAL_LINK_SECTOR":
            heatmap, dyn_map, detections = self._process_dual_link_sector(room_config, active_rx_pairs)
            detections = detections[:stable_count]
            max_targets = stable_count
        else:
            heatmap, dyn_map = self._compute_full_mesh_probability_matrix(room_config, active_rx_pairs)
            self.heatmap_ema = 0.62 * heatmap + 0.38 * self.heatmap_ema
            self.dynamic_map_ema = 0.65 * dyn_map + 0.35 * self.dynamic_map_ema

            detections = self._extract_peak_centroids(
                self.heatmap_ema,
                self.dynamic_map_ema,
                node_metrics,
                room_config=room_config,
                max_targets=stable_count,
                min_separation_m=1.1,
                allow_synthetic_split=(stable_count >= 2),
            )
            max_targets = stable_count

        if topology_info.mode_code in ("SINGLE_LINK_CORRIDOR", "DUAL_LINK_SECTOR"):
            self.heatmap_ema = 0.65 * heatmap + 0.35 * self.heatmap_ema
            self.dynamic_map_ema = 0.65 * dyn_map + 0.35 * self.dynamic_map_ema

        people_coords = self._update_multi_target_tracks(
            detections=detections,
            room_config=room_config,
            dt=dt,
            now=now,
            mode_code=topology_info.mode_code,
            max_targets=max_targets,
        )

        active_zone = self._determine_most_active_zone(people_coords, self.heatmap_ema, topology_info.mode_code)
        grid_list = [[round(float(np.clip(val, 0.0, 1.0)), 3) for val in row] for row in self.heatmap_ema]
        return grid_list, people_coords, active_zone, topology_info, coverage_info, global_decision

    # ==========================================================================
    # CASO N = 1: Zona de Abrangência com Multi-Contagem (1 ESP32 + 1 Roteador)
    # ==========================================================================

    def _process_single_link_corridor(
        self,
        room_config: RoomConfig,
        rx_node: NodePosition,
        metrics: LinkDSPMetrics,
        target_people_count: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float, float, float, float]]]:
        """
        Concentra 100% do peso da matriz de calor dentro da Elipse de Fresnel de Abrangência entre
        o Roteador (TX) e o único ESP32 ativo (RX).
        Quando target_people_count >= 2 (detectado via subcarrier_diversity + energy_ratio),
        posiciona múltiplos marcadores (X, Y) distribuídos dentro da elipse de abrangência.
        """
        p_score = metrics.perturbation_score
        if target_people_count <= 0 or p_score < 0.08:
            return (
                np.zeros((self.grid_size, self.grid_size), dtype=np.float64),
                np.zeros((self.grid_size, self.grid_size), dtype=np.float64),
                [],
            )

        tx_x, tx_y = room_config.tx.x, room_config.tx.y
        rx_x, rx_y = rx_node.x, rx_node.y
        link_dx = rx_x - tx_x
        link_dy = rx_y - tx_y
        link_len = max(0.05, float(np.hypot(link_dx, link_dy)))

        # Vetor unitário perpendicular ao eixo TX -> RX para distribuir alvos dentro da elipse
        perp_x = -link_dy / link_len
        perp_y = link_dx / link_len

        dist_tx = np.hypot(self.grid_x - tx_x, self.grid_y - tx_y)
        dist_rx = np.hypot(self.grid_x - rx_x, self.grid_y - rx_y)

        # Elipse da Zona de Abrangência Ativa (Fresnel + raio de sensibilidade ~1.8m)
        excess_path = np.maximum(0.0, (dist_tx + dist_rx) - link_len)
        ellipse_corridor = np.exp(-excess_path / 0.14)

        proj_t = ((self.grid_x - tx_x) * link_dx + (self.grid_y - tx_y) * link_dy) / (link_len ** 2)
        proj_t = np.clip(proj_t, 0.0, 1.0)

        detections: List[Tuple[float, float, float, float, float]] = []
        multi_lobes = np.zeros_like(proj_t)

        foci_list = metrics.multi_target_foci or [(metrics.longitudinal_focus, 0.0)]
        motion_ratio = float(metrics.motion_score)
        base_conf = float(np.clip(0.62 + 0.34 * p_score, 0.58, 0.97))

        if target_people_count == 1:
            t_focus = float(np.clip(metrics.longitudinal_focus, 0.16, 0.84))
            target_x = float(np.clip(tx_x + t_focus * link_dx, 0.06, 0.94))
            target_y = float(np.clip(tx_y + t_focus * link_dy, 0.08, 0.92))
            multi_lobes = np.exp(-0.5 * ((proj_t - t_focus) / 0.17) ** 2)
            detections.append((target_x, target_y, base_conf, motion_ratio, t_focus))
        else:
            # 2 ou 3+ pessoas detectadas no feixe único: distribui os marcadores na elipse de abrangência
            for idx in range(min(3, target_people_count)):
                t_val, lat_offset = foci_list[idx % len(foci_list)]
                t_val = float(np.clip(t_val, 0.18, 0.84))
                target_x = float(np.clip(tx_x + t_val * link_dx + perp_x * lat_offset, 0.08, 0.92))
                target_y = float(np.clip(tx_y + t_val * link_dy + perp_y * lat_offset, 0.10, 0.90))

                lobe_i = np.exp(-((self.grid_x - target_x) ** 2 + (self.grid_y - target_y) ** 2) / (2.0 * 0.12 ** 2))
                multi_lobes = np.maximum(multi_lobes, lobe_i)
                conf_i = max(0.56, base_conf - idx * 0.04)
                detections.append((target_x, target_y, conf_i, motion_ratio, t_val))

        heatmap = p_score * ellipse_corridor * (0.30 + 0.70 * multi_lobes)
        heatmap = ndimage.gaussian_filter(heatmap, sigma=0.65)
        peak = float(np.max(heatmap))
        if peak > 1e-6:
            heatmap = (heatmap / peak) * min(1.0, p_score * 1.18)

        dyn_map = heatmap * (metrics.motion_score / max(0.1, p_score))
        return np.clip(heatmap, 0.0, 1.0), dyn_map, detections

    # ==========================================================================
    # CASO N = 2: Modo Setorial Duplo (2 ESP32 + 1 Roteador)
    # ==========================================================================

    def _process_dual_link_sector(
        self,
        room_config: RoomConfig,
        active_rx_pairs: List[Tuple[NodePosition, LinkDSPMetrics]],
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float, float, float, float]]]:
        (rx1, m1), (rx2, m2) = active_rx_pairs[0], active_rx_pairs[1]
        s1, s2 = m1.perturbation_score, m2.perturbation_score

        if max(s1, s2) < 0.11:
            return (
                np.zeros((self.grid_size, self.grid_size), dtype=np.float64),
                np.zeros((self.grid_size, self.grid_size), dtype=np.float64),
                [],
            )

        tx_x, tx_y = room_config.tx.x, room_config.tx.y

        map1, dyn1, det1 = self._process_single_link_corridor(
            room_config, rx1, m1, target_people_count=max(1, m1.estimated_people_on_link) if s1 >= 0.12 else 0
        )
        map2, dyn2, det2 = self._process_single_link_corridor(
            room_config, rx2, m2, target_people_count=max(1, m2.estimated_people_on_link) if s2 >= 0.12 else 0
        )

        t1 = float(np.clip(m1.longitudinal_focus, 0.15, 0.85))
        t2 = float(np.clip(m2.longitudinal_focus, 0.15, 0.85))
        p1_x, p1_y = tx_x + t1 * (rx1.x - tx_x), tx_y + t1 * (rx1.y - tx_y)
        p2_x, p2_y = tx_x + t2 * (rx2.x - tx_x), tx_y + t2 * (rx2.y - tx_y)

        if s1 >= 0.13 and s2 < 0.13:
            return map1, dyn1, det1
        if s2 >= 0.13 and s1 < 0.13:
            return map2, dyn2, det2

        w1 = s1 / max(1e-6, s1 + s2)
        w2 = s2 / max(1e-6, s1 + s2)
        interp_x = float(np.clip(w1 * p1_x + w2 * p2_x, 0.06, 0.94))
        interp_y = float(np.clip(w1 * p1_y + w2 * p2_y, 0.08, 0.92))
        interp_t = w1 * t1 + w2 * t2

        sector_lobe = np.exp(-((self.grid_x - interp_x) ** 2 + (self.grid_y - interp_y) ** 2) / (2.0 * 0.11 ** 2))
        combined_map = 0.45 * (map1 + map2) + 0.65 * max(s1, s2) * sector_lobe
        combined_map = np.clip(ndimage.gaussian_filter(combined_map, sigma=0.65), 0.0, 1.0)
        combined_dyn = 0.5 * (dyn1 + dyn2)

        est_total = max(m1.estimated_people_on_link, m2.estimated_people_on_link)
        detections: List[Tuple[float, float, float, float, float]] = []
        if est_total >= 2 or abs(t1 - t2) > 0.10:
            if det1:
                detections.append(det1[0])
            if det2:
                detections.append(det2[0])
        else:
            conf = float(np.clip(0.62 + 0.35 * max(s1, s2), 0.58, 0.97))
            m_ratio = float(np.clip(max(m1.motion_score, m2.motion_score) / max(0.2, max(s1, s2)), 0.0, 1.0))
            detections.append((interp_x, interp_y, conf, m_ratio, interp_t))

        return combined_map, combined_dyn, detections[:3]

    # ==========================================================================
    # CASO N >= 3: Malha Tomográfica 2D Completa (3+ ESP32 + 1 Roteador)
    # ==========================================================================

    def _compute_full_mesh_probability_matrix(
        self,
        room_config: RoomConfig,
        active_rx_pairs: List[Tuple[NodePosition, LinkDSPMetrics]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        tx_x, tx_y = room_config.tx.x, room_config.tx.y
        dist_tx = np.hypot(self.grid_x - tx_x, self.grid_y - tx_y)

        fresnel_accum = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        fresnel_links: List[np.ndarray] = []
        radial_projections: List[np.ndarray] = []
        radial_dyn_projections: List[np.ndarray] = []

        max_perturbation = 0.0

        for rx, metrics in active_rx_pairs:
            p_score = metrics.perturbation_score
            max_perturbation = max(max_perturbation, p_score)
            if p_score < 0.03:
                continue

            rx_x, rx_y = rx.x, rx.y
            link_vec_x = rx_x - tx_x
            link_vec_y = rx_y - tx_y
            link_len = max(0.05, float(np.hypot(link_vec_x, link_vec_y)))

            dist_rx = np.hypot(self.grid_x - rx_x, self.grid_y - rx_y)

            excess_path = np.maximum(0.0, (dist_tx + dist_rx) - link_len)
            ellipse_weight = np.exp(-excess_path / 0.14)

            proj_t = ((self.grid_x - tx_x) * link_vec_x + (self.grid_y - tx_y) * link_vec_y) / (link_len ** 2)
            proj_t = np.clip(proj_t, 0.0, 1.0)

            sub_total = metrics.subcarrier_dyn_profile + metrics.subcarrier_stat_profile
            fresnel_bins = sub_total[:26]
            if float(np.max(fresnel_bins)) > 0.05:
                t_idx = np.clip(proj_t * 25.0, 0.0, 25.0)
                idx_low = np.floor(t_idx).astype(int)
                idx_high = np.minimum(25, idx_low + 1)
                frac = t_idx - idx_low
                long_weight = (1.0 - frac) * fresnel_bins[idx_low] + frac * fresnel_bins[idx_high]
                long_max = float(np.max(long_weight))
                if long_max > 1e-6:
                    long_weight = 0.25 + 0.75 * (long_weight / long_max)
                else:
                    long_weight = np.ones_like(proj_t)
            else:
                long_weight = np.ones_like(proj_t)

            link_map = p_score * ellipse_weight * long_weight
            fresnel_accum += link_map
            fresnel_links.append(link_map)

            radial_bins = sub_total[26:52]
            radial_dyn_bins = metrics.subcarrier_dyn_profile[26:52]
            if float(np.max(radial_bins)) > 0.04:
                r_norm = np.clip(dist_rx / 1.42, 0.0, 1.0)
                r_idx = r_norm * 25.0
                r_low = np.floor(r_idx).astype(int)
                r_high = np.minimum(25, r_low + 1)
                r_frac = r_idx - r_low

                rad_map = (1.0 - r_frac) * radial_bins[r_low] + r_frac * radial_bins[r_high]
                rad_dyn_map = (1.0 - r_frac) * radial_dyn_bins[r_low] + r_frac * radial_dyn_bins[r_high]
                radial_projections.append(rad_map)
                radial_dyn_projections.append(rad_dyn_map)

        if max_perturbation < 0.04:
            return (
                np.zeros((self.grid_size, self.grid_size), dtype=np.float64),
                np.zeros((self.grid_size, self.grid_size), dtype=np.float64),
            )

        synergy_map = np.zeros_like(fresnel_accum)
        n_links = len(fresnel_links)
        if n_links >= 2:
            for i in range(n_links):
                for j in range(i + 1, n_links):
                    synergy_map += np.sqrt(fresnel_links[i] * fresnel_links[j])
            synergy_map /= max(1, (n_links * (n_links - 1)) // 2)

        base_fresnel = (fresnel_accum / max(1, n_links)) * 0.55 + synergy_map * 0.85

        if len(radial_projections) >= 2:
            prod_map = np.ones_like(fresnel_accum)
            sum_map = np.zeros_like(fresnel_accum)
            for r_map in radial_projections:
                r_normed = r_map / max(0.5, float(np.max(r_map)))
                prod_map *= (0.08 + r_normed)
                sum_map += r_normed

            geom_intersection = (prod_map ** (1.0 / len(radial_projections))) - 0.08
            geom_intersection = np.maximum(0.0, geom_intersection)
            tomo_peaks = (geom_intersection ** 1.6) * (sum_map / len(radial_projections))

            dyn_sum = np.zeros_like(fresnel_accum)
            for d_map in radial_dyn_projections:
                dyn_sum += d_map
            dyn_avg = dyn_sum / len(radial_dyn_projections)
        elif len(radial_projections) == 1:
            r_map = radial_projections[0]
            tomo_peaks = (r_map / max(0.5, float(np.max(r_map)))) * base_fresnel
            dyn_avg = radial_dyn_projections[0]
        else:
            tomo_peaks = base_fresnel
            dyn_avg = base_fresnel

        combined = 0.22 * base_fresnel + 0.88 * tomo_peaks
        smoothed = ndimage.gaussian_filter(combined, sigma=0.75)

        peak_val = float(np.max(smoothed))
        if peak_val > 0.08:
            target_scale = min(1.0, max_perturbation * 1.15)
            smoothed = (smoothed / peak_val) * target_scale

        return np.clip(smoothed, 0.0, 1.0), dyn_avg

    def _extract_peak_centroids(
        self,
        heatmap: np.ndarray,
        dyn_map: np.ndarray,
        node_metrics: Dict[str, LinkDSPMetrics],
        room_config: RoomConfig | None = None,
        max_targets: int = 4,
        min_separation_m: float = 1.2,
        allow_synthetic_split: bool = False,
    ) -> List[Tuple[float, float, float, float, float]]:
        if max_targets <= 0:
            return []

        max_val = float(np.max(heatmap))
        if max_val < 0.10:
            return []

        w_m = room_config.width_m if room_config else 5.0
        l_m = room_config.length_m if room_config else 4.5

        abs_threshold = max(0.12, max_val * 0.40)
        neighborhood_max = ndimage.maximum_filter(heatmap, size=4, mode="constant", cval=0.0)
        is_local_max = (heatmap == neighborhood_max) & (heatmap >= abs_threshold)

        peak_indices = np.argwhere(is_local_max)
        candidates: List[Tuple[float, float, float, float, float]] = []
        max_node_motion = float(max([m.motion_score for m in node_metrics.values() if m.online] or [0.0]))

        if peak_indices.size > 0:
            peak_scores = [float(heatmap[r, c]) for r, c in peak_indices]
            sorted_order = np.argsort(peak_scores)[::-1]

            for idx in sorted_order:
                r, c = int(peak_indices[idx][0]), int(peak_indices[idx][1])
                score = float(heatmap[r, c])

                r0, r1 = max(0, r - 1), min(self.grid_size, r + 2)
                c0, c1 = max(0, c - 1), min(self.grid_size, c + 2)
                patch_w = np.maximum(0.0, heatmap[r0:r1, c0:c1] - abs_threshold * 0.5)
                w_sum = float(np.sum(patch_w))

                if w_sum > 1e-6:
                    cx = float(np.sum(self.grid_x[r0:r1, c0:c1] * patch_w) / w_sum)
                    cy = float(np.sum(self.grid_y[r0:r1, c0:c1] * patch_w) / w_sum)
                else:
                    cx = float(self.grid_x[r, c])
                    cy = float(self.grid_y[r, c])

                # Exige separação física mínima em metros (ex: >= 1.1 metros) entre dois picos distintos
                too_close = False
                for ex_x, ex_y, _, _, _ in candidates:
                    dist_m = float(np.hypot((cx - ex_x) * w_m, (cy - ex_y) * l_m))
                    if dist_m < min_separation_m:
                        too_close = True
                        break
                if too_close:
                    continue

                confidence = float(np.clip(0.60 + 0.38 * score, 0.55, 0.99))
                candidates.append((cx, cy, confidence, max_node_motion, 0.5))

                if len(candidates) >= max_targets:
                    break

        # Fallback de máxima verossimilhança: se há alvo ativo mas nenhum pico local isolado, usa o argmax global
        if not candidates and max_targets >= 1 and max_val >= 0.10:
            r_max, c_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            cx = float(self.grid_x[r_max, c_max])
            cy = float(self.grid_y[r_max, c_max])
            conf = float(np.clip(0.60 + 0.38 * max_val, 0.55, 0.95))
            candidates.append((cx, cy, conf, max_node_motion, 0.5))

        if allow_synthetic_split and len(candidates) < max_targets and candidates:
            base_cx, base_cy, base_conf, base_m, _ = candidates[0]
            offsets = [(0.24, -0.14), (-0.24, 0.14), (0.0, 0.25)]
            while len(candidates) < max_targets:
                ox, oy = offsets[(len(candidates) - 1) % len(offsets)]
                nx = float(np.clip(base_cx + ox, 0.14, 0.86))
                ny = float(np.clip(base_cy + oy, 0.18, 0.84))
                candidates.append((nx, ny, max(0.58, base_conf - 0.04), base_m, 0.5))

        return candidates

    def _update_multi_target_tracks(
        self,
        detections: List[Tuple[float, float, float, float, float]],
        room_config: RoomConfig,
        dt: float,
        now: float,
        mode_code: str,
        max_targets: int = 4,
    ) -> List[PersonCoordinate]:
        """Associa detecções aos alvos rastreados e aplica filtro cinemático Alpha-Beta."""
        if max_targets <= 0:
            self.tracks.clear()
            return []

        for track in self.tracks.values():
            track.x = float(np.clip(track.x + track.vx * dt * 0.35, 0.03, 0.97))
            track.y = float(np.clip(track.y + track.vy * dt * 0.35, 0.03, 0.97))

        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = list(range(len(detections)))

        pairs: List[Tuple[float, str, int]] = []
        match_radius = 0.48 if mode_code == "SINGLE_LINK_CORRIDOR" else 0.34
        for t_id in unmatched_tracks:
            tr = self.tracks[t_id]
            for d_idx in unmatched_dets:
                dx, dy, _, _, _ = detections[d_idx]
                dist = float(np.hypot(tr.x - dx, tr.y - dy))
                if dist < match_radius:
                    pairs.append((dist, t_id, d_idx))

        pairs.sort(key=lambda item: item[0])
        matched_tracks = set()
        matched_dets = set()

        alpha_pos = 0.50
        beta_vel = 0.20

        for dist, t_id, d_idx in pairs:
            if t_id in matched_tracks or d_idx in matched_dets:
                continue
            matched_tracks.add(t_id)
            matched_dets.add(d_idx)

            tr = self.tracks[t_id]
            dx, dy, conf, m_ratio, long_t = detections[d_idx]

            res_x = dx - tr.x
            res_y = dy - tr.y

            # Trava de Retenção de Pessoa Parada: quando a banda de locomoção cai (m_ratio < 0.22),
            # congela a pessoa na última coordenada estimada com status 'stationary'
            if m_ratio < 0.22:
                eff_alpha = 0.04
                tr.vx = 0.0
                tr.vy = 0.0
            else:
                eff_alpha = alpha_pos
                inst_vx = res_x / dt
                inst_vy = res_y / dt
                tr.vx = (1.0 - beta_vel) * tr.vx + beta_vel * inst_vx
                tr.vy = (1.0 - beta_vel) * tr.vy + beta_vel * inst_vy

            tr.x = float(np.clip(tr.x + eff_alpha * res_x, 0.04, 0.96))
            tr.y = float(np.clip(tr.y + eff_alpha * res_y, 0.06, 0.94))
            tr.confidence = 0.6 * tr.confidence + 0.4 * conf
            tr.motion_ratio = m_ratio
            tr.longitudinal_t = 0.6 * tr.longitudinal_t + 0.4 * long_t
            tr.last_updated = now
            tr.hit_streak += 1
            tr.missed_frames = 0

        # Mantém os tracks ativos durante a janela de histerese (2.5s = ~25 frames a 10 Hz)
        max_missed_frames = int(settings.COUNT_HYSTERESIS_SEC * settings.WS_BROADCAST_HZ)
        for t_id in list(unmatched_tracks - matched_tracks):
            tr = self.tracks[t_id]
            tr.missed_frames += 1
            tr.vx *= 0.65
            tr.vy *= 0.65
            tr.confidence *= 0.94
            if tr.missed_frames > max_missed_frames or len(self.tracks) > max_targets:
                del self.tracks[t_id]

        for d_idx in unmatched_dets:
            if d_idx in matched_dets:
                continue
            if len(self.tracks) >= max_targets:
                break
            dx, dy, conf, m_ratio, long_t = detections[d_idx]
            new_id = self._allocate_track_id()
            self.tracks[new_id] = TrackedTarget(
                track_id=new_id,
                x=dx,
                y=dy,
                vx=0.0,
                vy=0.0,
                confidence=conf,
                motion_ratio=m_ratio,
                longitudinal_t=long_t,
                last_updated=now,
                hit_streak=1,
                missed_frames=0,
            )

        if len(self.tracks) > max_targets:
            keep_ids = sorted(self.tracks.keys())[:max_targets]
            for tid in list(self.tracks.keys()):
                if tid not in keep_ids:
                    del self.tracks[tid]

        output: List[PersonCoordinate] = []
        for t_id in sorted(self.tracks.keys()):
            tr = self.tracks[t_id]
            is_moving = tr.motion_ratio >= 0.22
            activity = "moving" if is_moving else "stationary"
            posture_state = "standing_moving" if is_moving else "stationary"
            zone = self.classify_zone(tr.x, tr.y, mode_code=mode_code, longitudinal_t=tr.longitudinal_t)

            output.append(
                PersonCoordinate(
                    id=tr.track_id,
                    x=round(tr.x, 3),
                    y=round(tr.y, 3),
                    x_m=round(tr.x * room_config.width_m, 2),
                    y_m=round(tr.y * room_config.length_m, 2),
                    vx=round(tr.vx, 3),
                    vy=round(tr.vy, 3),
                    zone=zone,
                    activity=activity,
                    posture_state=posture_state,
                    confidence=round(float(np.clip(tr.confidence, 0.50, 0.99)), 2),
                )
            )

        return output

    def _allocate_track_id(self) -> str:
        for i in range(1, 10):
            candidate = f"P{i}"
            if candidate not in self.tracks:
                return candidate
        self._next_id_counter += 1
        return f"P{self._next_id_counter}"

    def _determine_most_active_zone(
        self,
        people: List[PersonCoordinate],
        heatmap: np.ndarray,
        mode_code: str = "FULL_MESH_2D",
    ) -> str:
        if mode_code == "NO_SENSORS":
            return "Sem Sensores"

        if people:
            zone_scores: Dict[str, float] = {}
            for p in people:
                weight = 1.5 if p.activity == "moving" else 1.0
                zone_scores[p.zone] = zone_scores.get(p.zone, 0.0) + weight * p.confidence
            return max(zone_scores.items(), key=lambda kv: kv[1])[0]

        max_val = float(np.max(heatmap))
        if max_val < 0.15:
            return "Zona de Abrangência Livre"

        idx = int(np.argmax(heatmap))
        r, c = divmod(idx, self.grid_size)
        return self.classify_zone(float(self.grid_x[r, c]), float(self.grid_y[r, c]), mode_code=mode_code)
