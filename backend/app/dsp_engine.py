import json
import logging
import os
from datetime import datetime
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
from scipy import ndimage, signal

from .config import settings
from .models import CalibrationMode, CalibrationStateCode, CSIPacket, SensitivityLevel

logger = logging.getLogger("wifi_spatial.dsp")


@dataclass
class ProcessedCSISample:
    """Amostra CSI processada com amplitude bruta, normalização L2 e fase sanitizada."""

    timestamp: float
    rssi: int
    amplitudes: np.ndarray        # Shape: (52,)
    amp_norm: np.ndarray          # Shape: (52,) normalizado por norma L2 (cancela AGC)
    sanitized_phases: np.ndarray  # Shape: (52,)


@dataclass
class NodeBaseline:
    """
    Perfil estatístico WiCount / esp-radar calibrado para cada ESP32-S3:
    - Correlação de Pearson Temporal (r) e Motion Index Mt = 1 - r^2
    - Blacklist de subportadoras ruidosas / em deep fade
    - Limiar de ruído Z-Score (Hard Zero Gate Z >= 3.8)
    - Autovalores PCA de referência (s1_ref, s2_ref) para discriminação multi-pessoa
    """

    mean_amplitude: np.ndarray    # Shape: (52,)
    std_amplitude: np.ndarray     # Shape: (52,)
    mean_amp_norm: np.ndarray     # Shape: (52,)
    mean_phase: np.ndarray        # Shape: (52,)
    valid_subcarriers: List[int] = field(default_factory=lambda: list(range(52)))
    subcarrier_blacklist: List[int] = field(default_factory=list)
    empty_motion_mean: float = 0.015
    empty_motion_std: float = 0.005
    empty_motion_max: float = 0.030
    motion_threshold: float = 0.045
    baseline_energy_mean: float = 0.12
    baseline_energy_std: float = 0.04
    baseline_energy_max: float = 0.22
    effective_noise_floor: float = 0.30
    static_clutter_floor: float = 0.25
    s1_empty_floor: float = 0.55
    s2_empty_floor: float = 0.32
    baseline_rssi: float = -50.5
    unit_person_energy: float = 0.55
    s1_ref: float = 1.40
    s2_ref: float = 0.50
    ref_1p_motion: float = 0.25
    ref_1p_s1: float = 1.20
    ref_1p_s2: float = 0.30
    ref_2p_motion: float = 0.50
    ref_2p_s2: float = 0.70
    ref_2p_ratio: float = 0.45
    sample_count: int = 0
    bootstrapped: bool = False
    mode: CalibrationMode = "nominal"
    step1_done: bool = False
    step2_done: bool = False
    step3_done: bool = False
    calibrated_at: float = field(default_factory=time.time)


@dataclass
class LinkDSPMetrics:
    """Métricas científicas extraídas pelo motor DSP para um link Roteador (TX) -> ESP32 (RX)."""

    node_id: str
    rssi: int
    rssi_avg: float
    estimated_distance_m: float
    packets_per_sec: float
    pearson_r: float                     # r temporal com o baseline estático da sala
    motion_index: float                  # M_t = 1 - r^2 (Índice de movimento esp-radar)
    motion_threshold: float              # Limiar de corte Z-score em M_t
    is_triggered: bool                   # True se M_t >= motion_threshold e Z >= 3.8
    current_energy: float                # Energia Doppler atual na banda humana (0.3 - 2.5 Hz)
    effective_noise_floor: float         # Limiar de corte absoluto (Noise Gate Z-Score)
    net_motion_energy: float             # max(0.0, current_energy - effective_noise_floor)
    z_score: float                       # Z-Score estatístico em tempo real (meta >= 3.8)
    ttl_remaining_sec: float             # Segundos restantes de TTL da trava estática (0..8s)
    motion_score: float                  # [0.0, 1.0] Score de locomoção
    presence_score: float                # [0.0, 1.0] Score de presença (com decaimento TTL)
    perturbation_score: float            # [0.0, 1.0] Score consolidado (0.0 se net_motion_energy==0 e sem TTL)
    walking_energy: float
    vital_energy: float
    shadow_attenuation: float
    pca_s1: float                        # 1º valor singular líquido (SVD)
    pca_s2: float                        # 2º valor singular bruto (SVD)
    pca_s2_empty_floor: float            # Baseline do 2º valor singular no vazio
    pca_s3: float                        # 3º valor singular (SVD)
    pca_s2_s1_ratio: float
    unit_person_energy: float
    subcarrier_diversity: float
    energy_ratio: float
    estimated_people_on_link: int
    stationary_retention_active: bool
    top_k_indices: List[int]
    multi_target_foci: List[Tuple[float, float]]
    longitudinal_profile: np.ndarray
    longitudinal_focus: float
    subcarrier_dyn_profile: np.ndarray
    subcarrier_stat_profile: np.ndarray
    amplitudes_preview: List[float]
    phases_preview: List[float]
    csi_ratio_preview: List[float]
    online: bool
    last_seen: float


class DSPEngine:
    """
    Motor DSP Wi-Fi CSI v3.1 (Blindado com Z-Score Noise Gate, High-Pass EMA e TTL Estático):
    1. Descarta subportadoras fracas (< Percentil 30 de amplitude média bruta) para eliminar deep fading.
    2. Remove a divisão instável CSI Ratio e aplica Média Móvel Curta (5 amostras) + Subtração de EMA
       (High-Pass Temporal) sobre as amplitudes normalizadas L2 antes do filtro na banda humana (0.3 a 2.5 Hz).
    3. Limiar Estatístico Z-Score Rigoroso:
         effective_noise_floor = max(baseline_energy_max * 1.35, baseline_energy_mean + 4.5 * baseline_energy_std)
         net_motion_energy = max(0.0, current_energy - effective_noise_floor)
       com Hard Zero Gate (se net_motion_energy == 0.0 e sem TTL ativo -> perturbação = 0.0 absoluto).
    4. Trava de Pessoa Parada (Anti-Drop) condicionada a 1.5s consecutivos de movimento forte (> 2.5x noise_floor)
       e Decaimento Automático (TTL de 8.0 segundos) caso não haja micro-movimento > 1.8x noise_floor.
    """

    COMPLETED_BANNER_HOLD_SEC: float = 4.5
    STATIONARY_ENTRY_CONFIRM_SEC: float = 0.35
    STATIONARY_TTL_MAX_SEC: float = 8.0
    MIN_CONSECUTIVE_TRIGGER_FRAMES: int = 2

    def __init__(self, num_subcarriers: int = settings.NUM_SUBCARRIERS):
        self.num_subcarriers = num_subcarriers
        self.top_k = settings.TOP_K_SUBCARRIERS  # 18 subportadoras
        self.window_sec = settings.BUFFER_WINDOW_SEC  # 2.5s
        self.resample_hz = settings.RESAMPLE_HZ  # 20.0 Hz
        self.resample_n = settings.RESAMPLE_SAMPLES  # 50 amostras
        self.window_size = settings.BUFFER_MAX_PACKETS

        # Buffers circulares por node_id
        self.buffers: Dict[str, deque[ProcessedCSISample]] = {}
        self.packet_timestamps: Dict[str, deque[float]] = {}

        # Estado temporal de confirmação de entrada (0.35s) e TTL de pessoa parada (8.0s) por node_id
        self.count_history_window: Dict[str, deque[Tuple[float, int]]] = {}
        self.last_retained_count: Dict[str, int] = {}
        self.has_confirmed_entry: Dict[str, bool] = {}
        self.entry_motion_accum_sec: Dict[str, float] = {}
        self.last_metric_eval_time: Dict[str, float] = {}
        self.stationary_ttl_deadline: Dict[str, float] = {}
        self.consecutive_motion_frames: Dict[str, int] = {}

        # Baselines calibrados por node_id
        self.baselines: Dict[str, NodeBaseline] = {}
        self.is_calibrated: bool = False
        self.step1_calibrated: bool = False
        self.step2_calibrated: bool = False
        self.calibrated_mode: CalibrationMode = "nominal"
        self.last_calibrated_at_str: Optional[str] = None

        # Máquina de Estados de Calibração
        self.calibration_state: CalibrationStateCode = "idle"
        self.active_calibration_mode: CalibrationMode = "crowd_compensated"
        self.is_calibrating: bool = False
        self.countdown_start_time: float = 0.0
        self.countdown_duration: float = 0.0
        self.calibration_start_time: float = 0.0
        self.calibration_duration: float = settings.DEFAULT_CALIBRATION_SECONDS
        self.completed_timestamp: float = 0.0
        self.calibration_samples: Dict[str, List[ProcessedCSISample]] = {}

        self.sensitivity: SensitivityLevel = (
            settings.DEFAULT_SENSITIVITY
            if settings.DEFAULT_SENSITIVITY in ("low", "normal", "high")
            else "normal"
        )

        # Filtro Butterworth na Banda Humana Unificada (0.3 Hz a 2.5 Hz @ 20 Hz)
        self._sos_human_band = signal.butter(
            N=2,
            Wn=[0.30, 2.50],
            btype="bandpass",
            fs=self.resample_hz,
            output="sos",
        )
        # Banda Vital / Micro-Movimento (0.18 Hz a 0.65 Hz)
        self._sos_vital = signal.butter(
            N=2,
            Wn=[0.18, 0.65],
            btype="bandpass",
            fs=self.resample_hz,
            output="sos",
        )

        self._num_range_bins = settings.GRID_RESOLUTION
        self._subcarrier_indices = np.arange(self.num_subcarriers, dtype=np.float64)

        # Restaura perfil de calibração persistido do disco se existir
        self.load_calibration_from_disk()

    def save_calibration_to_disk(self) -> bool:
        """Salva o perfil calibrado de todos os ESP32 em backend/calibration_profile.json."""
        try:
            profile_path = settings.CALIBRATION_PROFILE_PATH
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            data: Dict[str, Any] = {
                "saved_at": time.time(),
                "saved_at_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_calibrated": self.is_calibrated,
                "step1_calibrated": self.step1_calibrated,
                "step2_calibrated": self.step2_calibrated,
                "calibrated_mode": self.calibrated_mode,
                "nodes": {},
            }
            for node_id, b in self.baselines.items():
                data["nodes"][node_id] = {
                    "mean_amplitude": [round(float(x), 4) for x in b.mean_amplitude.tolist()],
                    "std_amplitude": [round(float(x), 4) for x in b.std_amplitude.tolist()],
                    "mean_amp_norm": [round(float(x), 4) for x in b.mean_amp_norm.tolist()],
                    "mean_phase": [round(float(x), 4) for x in b.mean_phase.tolist()],
                    "valid_subcarriers": [int(x) for x in b.valid_subcarriers],
                    "subcarrier_blacklist": [int(x) for x in b.subcarrier_blacklist],
                    "empty_motion_mean": round(float(b.empty_motion_mean), 5),
                    "empty_motion_std": round(float(b.empty_motion_std), 5),
                    "empty_motion_max": round(float(b.empty_motion_max), 5),
                    "motion_threshold": round(float(b.motion_threshold), 5),
                    "baseline_energy_mean": round(float(b.baseline_energy_mean), 4),
                    "baseline_energy_std": round(float(b.baseline_energy_std), 4),
                    "baseline_energy_max": round(float(b.baseline_energy_max), 4),
                    "effective_noise_floor": round(float(b.effective_noise_floor), 4),
                    "static_clutter_floor": round(float(b.static_clutter_floor), 4),
                    "s1_empty_floor": round(float(b.s1_empty_floor), 4),
                    "s2_empty_floor": round(float(b.s2_empty_floor), 4),
                    "baseline_rssi": round(float(b.baseline_rssi), 2),
                    "unit_person_energy": round(float(b.unit_person_energy), 4),
                    "s1_ref": round(float(b.s1_ref), 4),
                    "s2_ref": round(float(b.s2_ref), 4),
                    "ref_1p_motion": round(float(b.ref_1p_motion), 4),
                    "ref_1p_s1": round(float(b.ref_1p_s1), 4),
                    "ref_1p_s2": round(float(b.ref_1p_s2), 4),
                    "ref_2p_motion": round(float(b.ref_2p_motion), 4),
                    "ref_2p_s2": round(float(b.ref_2p_s2), 4),
                    "ref_2p_ratio": round(float(b.ref_2p_ratio), 4),
                    "sample_count": b.sample_count,
                    "bootstrapped": b.bootstrapped,
                    "mode": b.mode,
                    "step1_done": b.step1_done,
                    "step2_done": b.step2_done,
                    "step3_done": b.step3_done,
                    "calibrated_at": b.calibrated_at,
                }
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Perfil de calibração persistido com sucesso em {profile_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar perfil de calibração em disco: {e}")
            return False

    def load_calibration_from_disk(self) -> bool:
        """Carrega perfil persistido em backend/calibration_profile.json se existir."""
        try:
            profile_path = settings.CALIBRATION_PROFILE_PATH
            if not profile_path.exists():
                return False
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            nodes_data = data.get("nodes", {})
            if not nodes_data:
                return False
            for node_id, nd in nodes_data.items():
                mean_amp = np.array(nd.get("mean_amplitude", []), dtype=np.float64)
                std_amp = np.array(nd.get("std_amplitude", []), dtype=np.float64)
                mean_norm = np.array(nd.get("mean_amp_norm", []), dtype=np.float64)
                mean_ph = np.array(nd.get("mean_phase", []), dtype=np.float64)
                if mean_amp.size != self.num_subcarriers:
                    continue
                valid_subs = [int(x) for x in nd.get("valid_subcarriers", list(range(self.num_subcarriers)))]
                black_subs = [int(x) for x in nd.get("subcarrier_blacklist", [])]
                self.baselines[node_id] = NodeBaseline(
                    mean_amplitude=mean_amp,
                    std_amplitude=std_amp,
                    mean_amp_norm=mean_norm,
                    mean_phase=mean_ph,
                    valid_subcarriers=valid_subs,
                    subcarrier_blacklist=black_subs,
                    empty_motion_mean=float(nd.get("empty_motion_mean", 0.015)),
                    empty_motion_std=float(nd.get("empty_motion_std", 0.005)),
                    empty_motion_max=float(nd.get("empty_motion_max", 0.030)),
                    motion_threshold=float(nd.get("motion_threshold", 0.045)),
                    baseline_energy_mean=float(nd.get("baseline_energy_mean", 0.12)),
                    baseline_energy_std=float(nd.get("baseline_energy_std", 0.04)),
                    baseline_energy_max=float(nd.get("baseline_energy_max", 0.22)),
                    effective_noise_floor=float(nd.get("effective_noise_floor", 0.30)),
                    static_clutter_floor=float(nd.get("static_clutter_floor", 0.25)),
                    s1_empty_floor=float(nd.get("s1_empty_floor", 0.55)),
                    s2_empty_floor=float(nd.get("s2_empty_floor", 0.32)),
                    baseline_rssi=float(nd.get("baseline_rssi", -50.5)),
                    unit_person_energy=float(nd.get("unit_person_energy", 0.55)),
                    s1_ref=float(nd.get("s1_ref", 1.40)),
                    s2_ref=float(nd.get("s2_ref", 0.50)),
                    ref_1p_motion=float(nd.get("ref_1p_motion", 0.25)),
                    ref_1p_s1=float(nd.get("ref_1p_s1", 1.20)),
                    ref_1p_s2=float(nd.get("ref_1p_s2", 0.30)),
                    ref_2p_motion=float(nd.get("ref_2p_motion", 0.50)),
                    ref_2p_s2=float(nd.get("ref_2p_s2", 0.70)),
                    ref_2p_ratio=float(nd.get("ref_2p_ratio", 0.45)),
                    sample_count=int(nd.get("sample_count", 100)),
                    bootstrapped=True,
                    mode=nd.get("mode", "nominal"),
                    step1_done=bool(nd.get("step1_done", False)),
                    step2_done=bool(nd.get("step2_done", False)),
                    step3_done=bool(nd.get("step3_done", False)),
                    calibrated_at=float(nd.get("calibrated_at", time.time())),
                )
            self.is_calibrated = bool(data.get("is_calibrated", False)) or (len(self.baselines) > 0)
            self.step1_calibrated = bool(data.get("step1_calibrated", False)) or any(b.step1_done for b in self.baselines.values())
            self.step2_calibrated = bool(data.get("step2_calibrated", False)) or any(b.step2_done for b in self.baselines.values())
            self.calibrated_mode = data.get("calibrated_mode", "empty_room")
            saved_str = data.get("saved_at_str", "")
            if saved_str:
                self.last_calibrated_at_str = saved_str.split(" ")[-1]
            logger.info(f"Perfil de calibração restaurado com sucesso de {profile_path} ({len(self.baselines)} nós)")
            return True
        except Exception as e:
            logger.warning(f"Não foi possível carregar calibração do disco: {e}")
            return False

    @staticmethod
    def compute_pearson_motion(
        amp_current: np.ndarray,
        baseline_amp: np.ndarray,
        valid_subcarriers: List[int],
    ) -> Tuple[float, float]:
        """
        Calcula a correlação de Pearson temporal (r) entre a curva de amplitude atual e o baseline estático,
        e o Índice de Movimento M_t = 1 - r^2 (esp-radar / WiCount).
        """
        if len(valid_subcarriers) < 6:
            valid_idx = np.arange(amp_current.size)
        else:
            valid_idx = np.array(valid_subcarriers, dtype=int)

        h_t = amp_current[valid_idx]
        h_bar = baseline_amp[valid_idx]

        h_t_c = h_t - np.mean(h_t)
        h_bar_c = h_bar - np.mean(h_bar)

        denom = float(np.sqrt(np.sum(h_t_c ** 2)) * np.sqrt(np.sum(h_bar_c ** 2)))
        if denom < 1e-9:
            return 1.0, 0.0

        r = float(np.clip(np.sum(h_t_c * h_bar_c) / denom, -1.0, 1.0))
        motion_index = float(max(0.0, 1.0 - (r ** 2)))
        return r, motion_index

    @staticmethod
    def estimate_distance_from_rssi(rssi_avg: float, csi_amp_mean: float = 22.0) -> float:
        path_loss_dist = 10.0 ** ((-40.0 - float(rssi_avg)) / (10.0 * 2.2))
        csi_atten_factor = float(np.clip((22.0 / max(8.0, float(csi_amp_mean))) ** 0.12, 0.90, 1.12))
        return float(np.clip(path_loss_dist * csi_atten_factor, 1.0, 10.0))

    @staticmethod
    def compute_l2_normalized_amplitude(amplitudes: np.ndarray) -> np.ndarray:
        """
        Normaliza cada pacote dividindo as amplitudes pela norma L2 do próprio pacote:
            amp_norm = amp / (np.linalg.norm(amp) + 1e-6)
        Sem usar a divisão instável amp[:-1] / amp[1:].
        """
        l2_norm = float(np.linalg.norm(amplitudes)) + 1e-6
        return amplitudes / l2_norm

    @staticmethod
    def hampel_filter_matrix(matrix: np.ndarray, window_size: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
        if matrix.shape[0] < window_size:
            return matrix
        median_mat = ndimage.median_filter(matrix, size=(window_size, 1), mode="nearest")
        abs_dev = np.abs(matrix - median_mat)
        mad_mat = ndimage.median_filter(abs_dev, size=(window_size, 1), mode="nearest")
        threshold = n_sigmas * 1.4826 * (mad_mat + 1e-6)
        outliers = abs_dev > threshold
        return np.where(outliers, median_mat, matrix)

    @staticmethod
    def smooth_and_ema_highpass(norm_matrix: np.ndarray, ma_window: int = 5, ema_alpha: float = 0.12) -> np.ndarray:
        """
        1. Aplica média móvel curta (5 amostras) para eliminar ruído branco de alta frequência dos ESP32-S3.
        2. Aplica Subtração de Média Móvel Exponencial (High-Pass Temporal) sobre as amplitudes normalizadas
           suavizadas, substituindo a razão instável amp[:-1]/amp[1:].
        """
        smoothed = ndimage.uniform_filter1d(norm_matrix, size=ma_window, axis=0, mode="nearest")
        ema = np.empty_like(smoothed)
        ema[0, :] = smoothed[0, :]
        for t in range(1, smoothed.shape[0]):
            ema[t, :] = (1.0 - ema_alpha) * ema[t - 1, :] + ema_alpha * smoothed[t, :]
        highpass_residual = smoothed - ema
        return highpass_residual

    def extract_amplitude_and_phase(self, csi_raw: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        raw_arr = np.asarray(csi_raw, dtype=np.float64)
        if raw_arr.size < 4:
            return (
                np.ones(self.num_subcarriers, dtype=np.float64) * 10.0,
                np.zeros(self.num_subcarriers, dtype=np.float64),
            )

        # Proteção int8_t vs uint8_t: converte bytes sem sinal (128..255) para complemento de dois (-128..-1)
        raw_arr = np.where(raw_arr > 127, raw_arr - 256.0, raw_arr)

        even_len = (raw_arr.size // 2) * 2
        raw_arr = raw_arr[:even_len]

        imag = raw_arr[0::2]
        real = raw_arr[1::2]

        amplitudes = np.sqrt(imag ** 2 + real ** 2)
        phases = np.arctan2(imag, real)

        n_pairs = amplitudes.size
        if n_pairs == self.num_subcarriers:
            pass
        elif n_pairs >= 64:
            active_idx = np.concatenate([np.arange(6, 32), np.arange(33, 59)])
            amplitudes = amplitudes[active_idx]
            phases = phases[active_idx]
        elif n_pairs > self.num_subcarriers:
            start = (n_pairs - self.num_subcarriers) // 2
            amplitudes = amplitudes[start : start + self.num_subcarriers]
            phases = phases[start : start + self.num_subcarriers]
        else:
            x_old = np.linspace(0.0, 1.0, n_pairs)
            x_new = np.linspace(0.0, 1.0, self.num_subcarriers)
            amplitudes = np.interp(x_new, x_old, amplitudes)
            phases = np.interp(x_new, x_old, np.unwrap(phases))

        return amplitudes, phases

    def sanitize_phase(self, raw_phases: np.ndarray) -> np.ndarray:
        if raw_phases.size == 52:
            lower = np.unwrap(raw_phases[:26])
            upper = np.unwrap(raw_phases[26:])
            jump = upper[0] - lower[-1]
            upper = upper - np.round(jump / (2.0 * np.pi)) * (2.0 * np.pi)
            unwrapped = np.concatenate([lower, upper])
        else:
            unwrapped = np.unwrap(raw_phases)

        k = self._subcarrier_indices
        k_mean = np.mean(k)
        u_mean = np.mean(unwrapped)
        denom = np.sum((k - k_mean) ** 2)
        slope = (np.sum((k - k_mean) * (unwrapped - u_mean)) / denom) if denom > 1e-9 else 0.0
        intercept = u_mean - slope * k_mean

        sanitized = unwrapped - (slope * k + intercept)
        return np.clip(sanitized, -np.pi, np.pi)

    def ingest_packet(self, packet: CSIPacket) -> ProcessedCSISample:
        now = time.time()
        self.tick_calibration_state(now)

        amplitudes, raw_phases = self.extract_amplitude_and_phase(packet.csi_raw)
        amp_norm = self.compute_l2_normalized_amplitude(amplitudes)
        sanitized_phases = self.sanitize_phase(raw_phases)

        sample = ProcessedCSISample(
            timestamp=packet.timestamp,
            rssi=packet.rssi,
            amplitudes=amplitudes,
            amp_norm=amp_norm,
            sanitized_phases=sanitized_phases,
        )

        node_id = packet.node_id
        if node_id not in self.buffers:
            self.buffers[node_id] = deque(maxlen=self.window_size)
            self.packet_timestamps[node_id] = deque(maxlen=90)
            self.count_history_window[node_id] = deque(maxlen=40)
            self.last_retained_count[node_id] = 0
            self.has_confirmed_entry[node_id] = False
            self.entry_motion_accum_sec[node_id] = 0.0
            self.last_metric_eval_time[node_id] = now
            self.stationary_ttl_deadline[node_id] = 0.0

        self.buffers[node_id].append(sample)
        self.packet_timestamps[node_id].append(packet.timestamp)

        if node_id not in self.baselines:
            self._init_baseline_from_first_sample(node_id, sample)
        elif not self.is_calibrated and not self.baselines[node_id].bootstrapped and len(self.buffers[node_id]) >= 18:
            self._calibrate_node_from_samples(node_id, list(self.buffers[node_id]), mode="nominal", safety_mult=1.35)
            self.baselines[node_id].bootstrapped = True

        if self.calibration_state == "calibrating":
            if node_id not in self.calibration_samples:
                self.calibration_samples[node_id] = []
            self.calibration_samples[node_id].append(sample)
            self.tick_calibration_state(now)

        return sample

    def _init_baseline_from_first_sample(self, node_id: str, sample: ProcessedCSISample) -> None:
        self.baselines[node_id] = NodeBaseline(
            mean_amplitude=sample.amplitudes.copy(),
            std_amplitude=np.ones(self.num_subcarriers, dtype=np.float64) * 0.65,
            mean_amp_norm=sample.amp_norm.copy(),
            mean_phase=sample.sanitized_phases.copy(),
            baseline_energy_mean=0.14,
            baseline_energy_std=0.05,
            baseline_energy_max=0.25,
            effective_noise_floor=0.38,
            static_clutter_floor=0.30,
            s1_empty_floor=0.65,
            s2_empty_floor=0.38,
            baseline_rssi=float(sample.rssi),
            unit_person_energy=0.55,
            s1_ref=1.40,
            s2_ref=0.50,
            sample_count=1,
            bootstrapped=False,
            mode="nominal",
            step2_done=False,
        )

    def _compute_window_energy_and_svd(
        self,
        amp_u: np.ndarray,
        norm_u: np.ndarray,
        phase_u: np.ndarray,
    ) -> Tuple[float, float, np.ndarray, np.ndarray, float, float, float, np.ndarray]:
        """
        Núcleo unificado de cálculo de energia na banda humana (0.3 Hz a 2.5 Hz) e SVD:
        1. Descarta qualquer subportadora cuja amplitude média bruta seja menor que o percentil 30.
        2. Aplica filtro Hampel (janela 5, 3 sigma).
        3. Aplica média móvel curta (5 amostras) + Subtração de EMA (High-Pass Temporal) sobre norm_u.
        4. Filtra na banda humana (0.3 Hz a 2.5 Hz) para extrair current_energy e autovalores SVD (s1, s2, s3).
        """
        amp_clean = self.hampel_filter_matrix(amp_u, window_size=5, n_sigmas=3.0)
        norm_clean = self.hampel_filter_matrix(norm_u, window_size=5, n_sigmas=3.0)
        phase_clean = self.hampel_filter_matrix(phase_u, window_size=5, n_sigmas=3.0)

        # 1. Descarta subportadoras fracas (< Percentil 30 da amplitude média bruta)
        mean_amp_k = np.mean(amp_clean, axis=0)
        p30_cutoff = float(np.percentile(mean_amp_k, 30))
        valid_mask = mean_amp_k >= p30_cutoff
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size < self.top_k:
            valid_indices = np.argsort(mean_amp_k)[-self.top_k :]

        # 2. Média móvel curta (5 amostras) + Subtração de EMA (High-Pass Temporal)
        hp_norm = self.smooth_and_ema_highpass(norm_clean, ma_window=5, ema_alpha=0.12)
        hp_amp = self.smooth_and_ema_highpass(amp_clean, ma_window=5, ema_alpha=0.12)

        # Seleciona Top-18 subportadoras dentre as que passaram no corte do Percentil 30
        var_candidates = np.var(hp_norm[:, valid_indices], axis=0) * 1800.0 + np.var(hp_amp[:, valid_indices], axis=0) * 0.45
        if valid_indices.size > self.top_k:
            sub_order = np.argsort(var_candidates)[-self.top_k :]
            top_k_idx = np.sort(valid_indices[sub_order])
        else:
            top_k_idx = np.sort(valid_indices)

        # 3. Filtragem Butterworth na Banda Humana (0.3 Hz a 2.5 Hz)
        band_norm = signal.sosfilt(self._sos_human_band, hp_norm, axis=0)[6:, :]
        band_amp = signal.sosfilt(self._sos_human_band, hp_amp, axis=0)[6:, :]

        sub_norm_var = np.var(band_norm, axis=0)
        sub_amp_var = np.var(band_amp, axis=0)

        # Energia Doppler atual na banda humana (0.3 Hz a 2.5 Hz)
        current_energy = float(
            np.mean(sub_norm_var[top_k_idx]) * 850.0
            + np.mean(sub_amp_var[top_k_idx]) * 0.58
        )

        # Banda Vital / Micro-movimento (0.18 Hz a 0.65 Hz)
        vit_norm = signal.sosfilt(self._sos_vital, hp_norm, axis=0)[6:, :]
        vital_energy = float(np.mean(np.var(vit_norm[:, top_k_idx], axis=0)) * 2000.0)

        # 4. Decomposição SVD (PCA) na matriz centrada (44 amostras pós-transiente x 18 subportadoras)
        x_pca = band_norm[:, top_k_idx] * 42.0 + band_amp[:, top_k_idx] * 0.25
        try:
            _, s_vals, _ = np.linalg.svd(x_pca, full_matrices=False)
            norm_f = float(np.sqrt(max(1, x_pca.shape[0])))
            s1 = float(s_vals[0]) / norm_f if s_vals.size >= 1 else 0.0
            s2 = float(s_vals[1]) / norm_f if s_vals.size >= 2 else 0.0
            s3 = float(s_vals[2]) / norm_f if s_vals.size >= 3 else 0.0
        except Exception:
            s1, s2, s3 = 0.0, 0.0, 0.0

        sub_dyn_profile = np.zeros(self.num_subcarriers, dtype=np.float64)
        sub_dyn_profile[valid_indices] = np.clip(
            sub_norm_var[valid_indices] * 1800.0 + sub_amp_var[valid_indices] * 0.42,
            0.0,
            10.0,
        )

        return current_energy, vital_energy, top_k_idx, sub_dyn_profile, s1, s2, s3, hp_norm[-1, :]

    def _calibrate_node_from_samples(
        self,
        node_id: str,
        samples: List[ProcessedCSISample],
        mode: CalibrationMode = "empty_room",
        safety_mult: float = 1.35,
    ) -> None:
        """
        Calibração científica Z-score por sub-janelas:
        - Detecta subcarrier_blacklist e valid_subcarriers.
        - Calcula média, desvio padrão e pico de Pearson Motion Index (M_t = 1 - r^2) e Doppler Energy.
        - Define motion_threshold e effective_noise_floor com margem de segurança.
        - Persiste o perfil no disco via save_calibration_to_disk().
        """
        if len(samples) < 5:
            return

        amp_mat = np.stack([s.amplitudes for s in samples], axis=0)
        norm_mat = np.stack([s.amp_norm for s in samples], axis=0)
        phase_mat = np.stack([s.sanitized_phases for s in samples], axis=0)
        rssi_vals = np.array([s.rssi for s in samples], dtype=np.float64)

        mean_amp_k = np.median(amp_mat, axis=0)
        std_amp_k = np.std(amp_mat, axis=0)
        med_std = float(np.median(std_amp_k))

        # Detecção de subportadoras em deep fade (< 8.0) ou com ruído excessivo (> 2.5x mediana)
        blacklist: List[int] = []
        for k in range(self.num_subcarriers):
            if mean_amp_k[k] < 8.0 or std_amp_k[k] > (2.5 * med_std):
                blacklist.append(k)

        valid_subs = [k for k in range(self.num_subcarriers) if k not in blacklist]
        if len(valid_subs) < 18:
            best_idx = np.argsort(mean_amp_k)[-18:]
            valid_subs = sorted([int(x) for x in best_idx])
            blacklist = [k for k in range(self.num_subcarriers) if k not in valid_subs]

        n_total = len(samples)
        win_energies: List[float] = []
        m_list: List[float] = []
        s1_list: List[float] = []
        s2_list: List[float] = []

        step = max(2, n_total // 8)
        win_len = min(n_total, max(12, int(n_total * 0.55)))
        for start_i in range(0, max(1, n_total - win_len + 1), step):
            sub_samples = samples[start_i : start_i + win_len]
            t_last = sub_samples[-1].timestamp
            a_u, n_u, p_u = self._resample_uniform_window(sub_samples, t_last)
            e_val, _, _, _, s1_v, s2_v, _, _ = self._compute_window_energy_and_svd(a_u, n_u, p_u)
            win_energies.append(e_val)
            s1_list.append(s1_v)
            s2_list.append(s2_v)

            recent_sub_amp = np.median(a_u[-8:, :], axis=0)
            _, m_val = self.compute_pearson_motion(recent_sub_amp, mean_amp_k, valid_subs)
            m_list.append(m_val)

        if not win_energies:
            a_u, n_u, p_u = self._resample_uniform_window(samples, samples[-1].timestamp)
            e_val, _, _, _, s1_v, s2_v, _, _ = self._compute_window_energy_and_svd(a_u, n_u, p_u)
            win_energies = [e_val]
            s1_list = [s1_v]
            s2_list = [s2_v]
            _, m_val = self.compute_pearson_motion(np.median(a_u, axis=0), mean_amp_k, valid_subs)
            m_list = [m_val]

        e_arr = np.array(win_energies, dtype=np.float64)
        m_arr = np.array(m_list, dtype=np.float64)

        if mode == "crowd_compensated":
            e_mean = float(np.median(e_arr))
            e_std = max(0.015, float(np.median(np.abs(e_arr - e_mean)) * 1.4826))
            e_max = float(np.percentile(e_arr, 90))
            m_mean = float(np.median(m_arr))
            m_std = max(0.004, float(np.median(np.abs(m_arr - m_mean)) * 1.4826))
            m_max = float(np.percentile(m_arr, 90))
        else:
            e_mean = float(np.mean(e_arr))
            e_std = max(0.015, float(np.std(e_arr)))
            e_max = float(np.max(e_arr))
            m_mean = float(np.mean(m_arr))
            m_std = max(0.004, float(np.std(m_arr)))
            m_max = float(np.max(m_arr))

        sigma_mult = 3.2 if mode == "crowd_compensated" else settings.HARD_ZERO_Z_GATE
        effective_floor = max(0.12, e_max * safety_mult, e_mean + sigma_mult * e_std)
        motion_threshold = max(0.035, m_max * safety_mult, m_mean + sigma_mult * m_std)

        existing = self.baselines.get(node_id)
        unit_e = existing.unit_person_energy if (existing and existing.step2_done) else max(0.82, effective_floor * 1.85)
        s1_r = existing.s1_ref if existing else 1.40
        s2_r = existing.s2_ref if existing else 0.50
        ref_1p_m = existing.ref_1p_motion if existing else 0.25
        ref_1p_s1_v = existing.ref_1p_s1 if existing else 1.20
        ref_1p_s2_v = existing.ref_1p_s2 if existing else 0.30
        ref_2p_m = existing.ref_2p_motion if existing else 0.50
        ref_2p_s2_v = existing.ref_2p_s2 if existing else 0.70
        ref_2p_rat = existing.ref_2p_ratio if existing else 0.45

        step1_ok = existing.step1_done if existing else False
        step2_ok = existing.step2_done if existing else False
        step3_ok = existing.step3_done if existing else False

        if mode in ("stage1_silence", "empty_room", "crowd_compensated", "nominal", "instant_zero"):
            step1_ok = True
        elif mode in ("stage2_1person", "person_ref"):
            step2_ok = True
            ref_1p_m = float(np.median(m_arr))
            ref_1p_s1_v = float(np.median(s1_list))
            ref_1p_s2_v = float(np.median(s2_list))
            unit_e = float(np.clip(max(0.40, e_mean - (existing.effective_noise_floor if existing else 0.3)), 0.40, 8.0))
        elif mode == "stage3_2persons":
            step3_ok = True
            ref_2p_m = float(np.median(m_arr))
            ref_2p_s2_v = float(np.median(s2_list))
            ref_2p_rat = float(ref_2p_s2_v / max(0.1, np.median(s1_list)))

        self.baselines[node_id] = NodeBaseline(
            mean_amplitude=mean_amp_k,
            std_amplitude=np.maximum(std_amp_k, 0.35),
            mean_amp_norm=np.median(norm_mat, axis=0),
            mean_phase=np.median(phase_mat, axis=0),
            valid_subcarriers=valid_subs,
            subcarrier_blacklist=blacklist,
            empty_motion_mean=round(m_mean, 5),
            empty_motion_std=round(m_std, 5),
            empty_motion_max=round(m_max, 5),
            motion_threshold=round(motion_threshold, 5),
            baseline_energy_mean=round(e_mean, 4),
            baseline_energy_std=round(e_std, 4),
            baseline_energy_max=round(e_max, 4),
            effective_noise_floor=round(effective_floor, 4),
            static_clutter_floor=max(0.20, effective_floor * 0.85),
            s1_empty_floor=max(0.25, float(np.max(s1_list)) * 1.25),
            s2_empty_floor=max(0.16, float(np.max(s2_list)) * 1.25),
            baseline_rssi=float(np.median(rssi_vals)),
            unit_person_energy=round(unit_e, 4),
            s1_ref=round(s1_r, 4),
            s2_ref=round(s2_r, 4),
            ref_1p_motion=round(ref_1p_m, 4),
            ref_1p_s1=round(ref_1p_s1_v, 4),
            ref_1p_s2=round(ref_1p_s2_v, 4),
            ref_2p_motion=round(ref_2p_m, 4),
            ref_2p_s2=round(ref_2p_s2_v, 4),
            ref_2p_ratio=round(ref_2p_rat, 4),
            sample_count=n_total,
            bootstrapped=True,
            mode=mode,
            step1_done=step1_ok,
            step2_done=step2_ok,
            step3_done=step3_ok,
            calibrated_at=time.time(),
        )

        if step1_ok:
            self.step1_calibrated = True
            self.is_calibrated = True
        if step2_ok:
            self.step2_calibrated = True
            self.is_calibrated = True

        # Limpa todo estado residual de contagem/retenção daquele nó
        if node_id in self.count_history_window:
            self.count_history_window[node_id].clear()
        self.last_retained_count[node_id] = 0
        self.has_confirmed_entry[node_id] = False
        self.entry_motion_accum_sec[node_id] = 0.0
        self.stationary_ttl_deadline[node_id] = 0.0
        self.consecutive_motion_frames[node_id] = 0

        # Persiste em disco
        self.save_calibration_to_disk()

    def force_instant_zero(self, now: Optional[float] = None) -> dict:
        """
        Botão '🧹 Forçar Zero (Aqui está Vazio Agora)':
        Captura os últimos ~3 segundos de buffer que já estão na memória dos ESP32-S3,
        define o pico máximo atual + 40% de margem (safety_mult=1.40) como o novo effective_noise_floor,
        limpa todos os rastros/fantasmas e trava o contador em 0 instantaneamente!
        """
        if now is None:
            now = time.time()

        calibrated_nodes: Dict[str, float] = {}
        for node_id, buf in self.buffers.items():
            samples = list(buf)
            if len(samples) >= 5:
                self._calibrate_node_from_samples(
                    node_id=node_id,
                    samples=samples,
                    mode="instant_zero",
                    safety_mult=1.40,
                )
                calibrated_nodes[node_id] = self.baselines[node_id].effective_noise_floor

        self.consecutive_motion_frames.clear()
        self.is_calibrating = False
        self.is_calibrated = True
        self.step1_calibrated = True
        self.calibrated_mode = "instant_zero"
        self.calibration_state = "completed"
        self.completed_timestamp = now
        self.last_calibrated_at_str = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        self.save_calibration_to_disk()
        return calibrated_nodes

    # ==========================================================================
    # Calibração Científica em 3 Estágios (Wizard) & Modos Rápidos
    # ==========================================================================

    def start_calibration(
        self,
        mode: CalibrationMode = "stage1_silence",
        delay_seconds: float = 0.0,
        duration_seconds: float = 15.0,
    ) -> None:
        now = time.time()
        self.active_calibration_mode = mode
        self.countdown_duration = max(0.0, min(30.0, float(delay_seconds)))
        self.calibration_duration = max(2.0, min(60.0, float(duration_seconds)))

        if self.countdown_duration > 0.0:
            self.calibration_state = "countdown"
            self.countdown_start_time = now
            self.consecutive_motion_frames.clear()
            self.is_calibrating = False
            self.calibration_samples = {}
        else:
            self._begin_active_calibration(now)

    def _begin_active_calibration(self, now: float) -> None:
        self.calibration_state = "calibrating"
        self.calibration_start_time = now
        self.calibration_samples = {node_id: [] for node_id in self.buffers.keys()}
        self.is_calibrating = True

    def tick_calibration_state(self, now: Optional[float] = None) -> Tuple[CalibrationStateCode, float, float]:
        if now is None:
            now = time.time()

        if self.calibration_state == "countdown":
            elapsed = now - self.countdown_start_time
            remaining = max(0.0, self.countdown_duration - elapsed)
            if remaining <= 0.0:
                self._begin_active_calibration(now)
                return ("calibrating", 0.0, 0.0)
            return ("countdown", round(remaining, 1), 0.0)

        if self.calibration_state == "calibrating":
            elapsed = now - self.calibration_start_time
            if elapsed >= self.calibration_duration:
                self._finalize_calibration(now)
                return ("completed", 0.0, 1.0)
            progress = float(np.clip(elapsed / self.calibration_duration, 0.0, 0.99))
            return ("calibrating", 0.0, progress)

        if self.calibration_state == "completed":
            if (now - self.completed_timestamp) > self.COMPLETED_BANNER_HOLD_SEC:
                self.calibration_state = "idle"
                return ("idle", 0.0, 1.0 if self.is_calibrated else 0.0)
            return ("completed", 0.0, 1.0)

        return ("idle", 0.0, 1.0 if self.is_calibrated else 0.0)

    def _finalize_calibration(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()

        mode = self.active_calibration_mode

        for node_id in list(self.buffers.keys()):
            samples = self.calibration_samples.get(node_id, [])
            if not samples and len(self.buffers[node_id]) > 0:
                samples = list(self.buffers[node_id])

            if len(samples) < 5:
                continue

            self._calibrate_node_from_samples(
                node_id=node_id,
                samples=samples,
                mode=mode,
                safety_mult=1.35,
            )

        if mode in ("stage1_silence", "empty_room", "crowd_compensated", "nominal", "instant_zero"):
            self.step1_calibrated = True
        elif mode in ("stage2_1person", "person_ref"):
            self.step2_calibrated = True

        self.consecutive_motion_frames.clear()
        self.is_calibrating = False
        self.is_calibrated = True
        self.calibrated_mode = mode
        self.calibration_state = "completed"
        self.completed_timestamp = now
        self.last_calibrated_at_str = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        self.save_calibration_to_disk()

    # ==========================================================================
    # Interpolação Temporal Uniforme & Avaliação Z-Score em Tempo Real
    # ==========================================================================

    def _resample_uniform_window(
        self,
        samples: List[ProcessedCSISample],
        now: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        recent = [s for s in samples if (now - s.timestamp) <= self.window_sec]
        if len(recent) < 4:
            recent = samples[-min(len(samples), 15) :]

        ts = np.array([s.timestamp for s in recent], dtype=np.float64)
        for i in range(1, ts.size):
            if ts[i] <= ts[i - 1]:
                ts[i] = ts[i - 1] + 1e-4

        amp_raw = np.stack([s.amplitudes for s in recent], axis=0)
        norm_raw = np.stack([s.amp_norm for s in recent], axis=0)
        phase_raw = np.stack([s.sanitized_phases for s in recent], axis=0)

        if ts.size < 3 or (ts[-1] - ts[0]) < 0.08:
            idx_grid = np.linspace(0, ts.size - 1, self.resample_n)
            idx_orig = np.arange(ts.size, dtype=np.float64)
            amp_u = np.vstack([np.interp(idx_grid, idx_orig, amp_raw[:, k]) for k in range(self.num_subcarriers)]).T
            norm_u = np.vstack([np.interp(idx_grid, idx_orig, norm_raw[:, k]) for k in range(self.num_subcarriers)]).T
            phase_u = np.vstack([np.interp(idx_grid, idx_orig, phase_raw[:, k]) for k in range(self.num_subcarriers)]).T
            return amp_u, norm_u, phase_u

        t_uniform = np.linspace(max(ts[0], ts[-1] - self.window_sec), ts[-1], self.resample_n)
        amp_u = np.empty((self.resample_n, self.num_subcarriers), dtype=np.float64)
        norm_u = np.empty((self.resample_n, self.num_subcarriers), dtype=np.float64)
        phase_u = np.empty((self.resample_n, self.num_subcarriers), dtype=np.float64)

        for k in range(self.num_subcarriers):
            amp_u[:, k] = np.interp(t_uniform, ts, amp_raw[:, k])
            norm_u[:, k] = np.interp(t_uniform, ts, norm_raw[:, k])
            phase_u[:, k] = np.interp(t_uniform, ts, phase_raw[:, k])

        return amp_u, norm_u, phase_u

    def compute_node_metrics(self, node_id: str, now: Optional[float] = None) -> LinkDSPMetrics:
        if now is None:
            now = time.time()

        buf = self.buffers.get(node_id)
        ts_buf = self.packet_timestamps.get(node_id)

        if not buf or len(buf) == 0:
            return self._empty_metrics(node_id, last_seen=0.0)

        latest = buf[-1]
        last_seen = latest.timestamp
        online = (now - last_seen) <= settings.NODE_ONLINE_TIMEOUT_SEC

        if not online:
            return self._empty_metrics(node_id, last_seen=last_seen)

        prev_eval = self.last_metric_eval_time.get(node_id, now - 0.1)
        dt_eval = float(np.clip(now - prev_eval, 0.02, 0.50))
        self.last_metric_eval_time[node_id] = now

        recent_ts = [t for t in ts_buf if (now - t) <= 2.0] if ts_buf else []
        packets_per_sec = len(recent_ts) / 2.0 if recent_ts else 0.0

        samples_list = list(buf)
        rssi_arr = np.array([s.rssi for s in samples_list], dtype=np.float64)

        baseline = self.baselines.get(node_id)
        if baseline is None:
            self._init_baseline_from_first_sample(node_id, latest)
            baseline = self.baselines[node_id]

        rssi_avg = float(np.median(rssi_arr))
        ref_rssi_for_dist = (
            0.80 * baseline.baseline_rssi + 0.20 * float(np.percentile(rssi_arr, 85))
            if self.is_calibrated
            else max(baseline.baseline_rssi, float(np.percentile(rssi_arr, 85)))
        )
        estimated_distance_m = self.estimate_distance_from_rssi(
            rssi_avg=ref_rssi_for_dist,
            csi_amp_mean=float(np.mean(baseline.mean_amplitude)),
        )

        # 1. Reamostragem Uniforme (50 amostras @ 20 Hz) + Filtro Percentil 30 + High-Pass EMA + SVD
        amp_u, norm_u, phase_u = self._resample_uniform_window(samples_list, now)
        (
            current_energy,
            raw_vital_energy,
            top_k_idx,
            subcarrier_dyn_profile,
            raw_s1,
            raw_s2,
            raw_s3,
            hp_last_row,
        ) = self._compute_window_energy_and_svd(amp_u, norm_u, phase_u)

        # Ajuste do limiar de disparo pelo seletor de Sensibilidade (Baixa | Normal | Alta)
        if self.sensitivity == "low":
            gate_mult = 1.25
            z_trigger = 4.2
        elif self.sensitivity == "normal":
            gate_mult = 1.00
            z_trigger = settings.HARD_ZERO_Z_GATE  # 3.8
        else:  # "high"
            gate_mult = 0.82
            z_trigger = 3.2

        effective_noise_floor = baseline.effective_noise_floor * gate_mult
        motion_threshold = baseline.motion_threshold * gate_mult

        # 2. Correlação de Pearson Temporal e Motion Index M_t = 1 - r^2 (esp-radar / WiCount)
        amp_recent = np.median(amp_u[-8:, :], axis=0)
        pearson_r, motion_index = self.compute_pearson_motion(
            amp_current=amp_recent,
            baseline_amp=baseline.mean_amplitude,
            valid_subcarriers=baseline.valid_subcarriers,
        )

        # Cálculo do Z-Score em Tempo Real (Duplo: Pearson Motion Index + Doppler Energy)
        z_score_m = max(0.0, (motion_index - baseline.empty_motion_mean) / max(0.005, baseline.empty_motion_std))
        z_score_e = max(0.0, (current_energy - baseline.baseline_energy_mean) / max(0.015, baseline.baseline_energy_std))
        z_score = max(z_score_m, z_score_e)

        # Hard Zero Gate Rigoroso:
        # Se motion_index < motion_threshold ou Z < z_trigger, é ruído de oscilador / estático!
        is_triggered = bool(motion_index >= motion_threshold and z_score >= z_trigger)
        net_motion_energy = max(0.0, current_energy - effective_noise_floor) if is_triggered else 0.0

        # Valores singulares SVD descontando o baseline vazio
        pca_s1 = max(0.0, raw_s1 - baseline.s1_empty_floor)
        pca_s2 = raw_s2
        pca_s3 = max(0.0, raw_s3 - baseline.s2_empty_floor * 0.8)
        pca_s2_s1_ratio = float(raw_s2 / (raw_s1 + 1e-6)) if raw_s1 > 0.1 else 0.0

        # Sombra estática (apenas informativa; NUNCA cria ou segura pessoa sozinha sem Doppler/TTL!)
        recent_norm_med = np.median(norm_u[-12:, :], axis=0)
        norm_diff = np.abs(recent_norm_med - baseline.mean_amp_norm)
        raw_shadow = float(np.mean(norm_diff[top_k_idx]) * 18.0)
        shadow_attenuation = max(0.0, raw_shadow - baseline.static_clutter_floor)
        vital_energy = max(0.0, raw_vital_energy - effective_noise_floor * 0.55)

        # 3. Confirmação de Movimento & Anti-Glitch (M-of-N persistence)
        is_motion_candidate = is_triggered and (net_motion_energy > 0.0)
        if is_motion_candidate:
            self.consecutive_motion_frames[node_id] = self.consecutive_motion_frames.get(node_id, 0) + 1
        else:
            self.consecutive_motion_frames[node_id] = 0

        # Movimento confirmado exige pelo menos 2 frames consecutivos (rejeita spikes transientes isolados de 1 frame / 50ms)
        motion_confirmed = self.consecutive_motion_frames[node_id] >= self.MIN_CONSECUTIVE_TRIGGER_FRAMES

        # Máquina de Estados de Confirmação de Entrada (0.35s) e TTL de Pessoa Parada (8.0s)
        # Qualquer movimento persistente com Z >= z_trigger por >= 0.35s confirma entrada e arma a retenção estática
        entry_signal_active = motion_confirmed or (net_motion_energy > 0.0 and z_score >= (z_trigger * 1.15))
        micro_motion_alive = (current_energy > (1.15 * effective_noise_floor)) or (vital_energy > 0.08)

        if entry_signal_active:
            self.entry_motion_accum_sec[node_id] = self.entry_motion_accum_sec.get(node_id, 0.0) + dt_eval
            if self.entry_motion_accum_sec[node_id] >= self.STATIONARY_ENTRY_CONFIRM_SEC:
                self.has_confirmed_entry[node_id] = True
                self.stationary_ttl_deadline[node_id] = now + self.STATIONARY_TTL_MAX_SEC
        else:
            # Decai o acumulador de confirmação caso tenha sido apenas um spike curto (< 0.35s)
            self.entry_motion_accum_sec[node_id] = max(0.0, self.entry_motion_accum_sec.get(node_id, 0.0) - dt_eval * 0.8)

        if self.has_confirmed_entry.get(node_id, False) and micro_motion_alive:
            # Renova o TTL de 8 segundos enquanto houver micro-movimento real ou respiração
            self.stationary_ttl_deadline[node_id] = now + self.STATIONARY_TTL_MAX_SEC

        ttl_remaining = max(0.0, self.stationary_ttl_deadline.get(node_id, 0.0) - now)
        if ttl_remaining <= 0.0 and not entry_signal_active:
            # Expirou o TTL de 8 segundos sem micro-movimento: ZERA a trava estática e atualiza baseline lentamente!
            self.has_confirmed_entry[node_id] = False
            self.last_retained_count[node_id] = 0
            alpha_bg = 0.03
            baseline.mean_amp_norm = (1.0 - alpha_bg) * baseline.mean_amp_norm + alpha_bg * recent_norm_med

        # 4. REGRA DE OURO (Hard Zero Gate com Anti-Glitch):
        # Se não há movimento confirmado nem TTL de retenção ativo, o link é ZERO ABSOLUTO (0.0)!
        retention_active = False
        if not motion_confirmed and ttl_remaining <= 0.0:
            motion_score = 0.0
            presence_score = 0.0
            perturbation_score = 0.0
            subcarrier_diversity = 0.0
            energy_ratio = 0.0
            estimated_people = 0
            longitudinal_profile = np.zeros(self._num_range_bins, dtype=np.float64)
            longitudinal_focus = 0.5
            multi_target_foci = [(0.5, 0.0)]
            subcarrier_dyn_profile.fill(0.0)
            subcarrier_stat_profile = np.zeros(self.num_subcarriers, dtype=np.float64)
            motion_index = 0.0
        else:
            motion_score = float(np.clip(np.tanh(net_motion_energy / max(0.20, baseline.unit_person_energy * 0.80)), 0.0, 1.0))
            ttl_factor = float(ttl_remaining / self.STATIONARY_TTL_MAX_SEC) if ttl_remaining > 0 else 0.0
            presence_score = float(np.clip(max(motion_score * 0.85, ttl_factor * 0.80), 0.0, 1.0))
            perturbation_score = float(np.clip(max(motion_score, presence_score), 0.0, 1.0))

            subcarrier_stat_profile = np.clip(norm_diff * 18.0, 0.0, 10.0)
            subcarrier_diversity, n_spectral_peaks = self._compute_subcarrier_diversity(
                amp_matrix=amp_u,
                phase_matrix=phase_u,
                subcarrier_dyn_profile=subcarrier_dyn_profile,
                subcarrier_stat_profile=subcarrier_stat_profile,
                perturbation_score=perturbation_score,
            )

            energy_ratio = float(net_motion_energy / max(0.20, baseline.unit_person_energy))
            rssi_drop = max(0.0, baseline.baseline_rssi - float(np.median(rssi_arr[-10:])))

            longitudinal_profile, longitudinal_focus, multi_target_foci = self._extract_longitudinal_and_multi_foci(
                amp_residual=np.median(amp_u[-10:, :], axis=0) - baseline.mean_amplitude,
                phase_residual=np.abs(np.median(phase_u[-10:, :], axis=0) - baseline.mean_phase),
                sub_phase_var=subcarrier_dyn_profile * 0.1,
                subcarrier_dyn_profile=subcarrier_dyn_profile,
                subcarrier_stat_profile=subcarrier_stat_profile,
                rssi_drop=rssi_drop,
                perturbation_score=perturbation_score,
            )

            # Estimativa individual do link com limiar calibrado
            if motion_confirmed and net_motion_energy > 0.0 and z_score >= z_trigger:
                s2_ratio_over_empty = raw_s2 / max(0.08, baseline.s2_empty_floor)
                if energy_ratio >= 2.60 and (s2_ratio_over_empty >= 2.8 or subcarrier_diversity >= 0.60):
                    raw_est = 3
                elif energy_ratio >= 1.90 and (s2_ratio_over_empty >= 2.3 or subcarrier_diversity >= 0.50 or n_spectral_peaks >= 2):
                    raw_est = 2
                else:
                    raw_est = 1
                self.last_retained_count[node_id] = max(1, raw_est)
            elif ttl_remaining > 0.0 and self.has_confirmed_entry.get(node_id, False):
                raw_est = max(1, self.last_retained_count.get(node_id, 1))
                retention_active = True
            else:
                raw_est = 0

            hist_q = self.count_history_window[node_id]
            hist_q.append((now, raw_est))
            while len(hist_q) > 1 and (now - hist_q[0][0]) > settings.COUNT_HYSTERESIS_SEC:
                hist_q.popleft()
            recent_counts = [c for _, c in hist_q]
            estimated_people = int(np.rint(np.median(recent_counts))) if len(recent_counts) >= 3 else raw_est

        return LinkDSPMetrics(
            node_id=node_id,
            rssi=latest.rssi,
            rssi_avg=round(rssi_avg, 1),
            estimated_distance_m=round(estimated_distance_m, 2),
            packets_per_sec=round(packets_per_sec, 1),
            pearson_r=round(pearson_r, 4),
            motion_index=round(motion_index, 4),
            motion_threshold=round(motion_threshold, 4),
            is_triggered=is_triggered,
            current_energy=round(current_energy, 3),
            effective_noise_floor=round(effective_noise_floor, 3),
            net_motion_energy=round(net_motion_energy, 3),
            z_score=round(z_score, 2),
            ttl_remaining_sec=round(ttl_remaining, 1),
            motion_score=round(motion_score, 3),
            presence_score=round(presence_score, 3),
            perturbation_score=round(perturbation_score, 3),
            walking_energy=round(net_motion_energy, 3),
            vital_energy=round(vital_energy, 3),
            shadow_attenuation=round(shadow_attenuation, 3),
            pca_s1=round(pca_s1, 3),
            pca_s2=round(pca_s2, 3),
            pca_s2_empty_floor=round(baseline.s2_empty_floor, 3),
            pca_s3=round(pca_s3, 3),
            pca_s2_s1_ratio=round(pca_s2_s1_ratio, 3),
            unit_person_energy=round(baseline.unit_person_energy, 3),
            subcarrier_diversity=round(subcarrier_diversity, 3),
            energy_ratio=round(energy_ratio, 2),
            estimated_people_on_link=estimated_people,
            stationary_retention_active=retention_active,
            top_k_indices=[int(i) for i in top_k_idx.tolist()],
            multi_target_foci=multi_target_foci,
            longitudinal_profile=longitudinal_profile,
            longitudinal_focus=round(longitudinal_focus, 3),
            subcarrier_dyn_profile=subcarrier_dyn_profile,
            subcarrier_stat_profile=subcarrier_stat_profile,
            amplitudes_preview=[round(float(x), 2) for x in latest.amplitudes],
            phases_preview=[round(float(x), 3) for x in latest.sanitized_phases],
            csi_ratio_preview=[round(float(x), 4) for x in hp_last_row.tolist()],
            online=online,
            last_seen=last_seen,
        )

    def _compute_subcarrier_diversity(
        self,
        amp_matrix: np.ndarray,
        phase_matrix: np.ndarray,
        subcarrier_dyn_profile: np.ndarray,
        subcarrier_stat_profile: np.ndarray,
        perturbation_score: float,
    ) -> Tuple[float, int]:
        if perturbation_score < 0.15:
            return 0.0, 0

        centered_amp = amp_matrix - np.mean(amp_matrix, axis=0, keepdims=True)
        band_envelopes = [
            np.mean(centered_amp[:, 0:13], axis=1),
            np.mean(centered_amp[:, 13:26], axis=1),
            np.mean(centered_amp[:, 26:39], axis=1),
            np.mean(centered_amp[:, 39:52], axis=1),
        ]
        corrs: List[float] = []
        for i in range(4):
            for j in range(i + 1, 4):
                std_i = float(np.std(band_envelopes[i]))
                std_j = float(np.std(band_envelopes[j]))
                if std_i > 1e-4 and std_j > 1e-4:
                    c = float(np.corrcoef(band_envelopes[i], band_envelopes[j])[0, 1])
                    if not np.isnan(c):
                        corrs.append(abs(c))
        decorrelation_score = float(np.clip(1.0 - float(np.mean(corrs)), 0.0, 1.0)) if corrs else 0.0

        total_profile = subcarrier_dyn_profile + subcarrier_stat_profile * 0.5
        prof_sum = float(np.sum(total_profile))
        shannon_h = (
            -float(np.sum((total_profile / prof_sum) * np.log((total_profile / prof_sum) + 1e-12)))
            / float(np.log(self.num_subcarriers))
            if prof_sum > 1e-5
            else 0.0
        )

        fresnel_sub = total_profile[:26]
        radial_sub = total_profile[26:52]
        f_max = float(np.max(fresnel_sub)) if fresnel_sub.size else 0.0
        r_max = float(np.max(radial_sub)) if radial_sub.size else 0.0

        n_peaks_f = int(np.sum((ndimage.maximum_filter(fresnel_sub, size=4) == fresnel_sub) & (fresnel_sub >= f_max * 0.42))) if f_max > 0.22 else 0
        n_peaks_r = int(np.sum((ndimage.maximum_filter(radial_sub, size=4) == radial_sub) & (radial_sub >= r_max * 0.42))) if r_max > 0.22 else 0

        n_spectral_peaks = max(n_peaks_f, n_peaks_r)
        peak_diversity = float(np.clip((n_spectral_peaks - 1) * 0.42 + (n_peaks_f + n_peaks_r - 2) * 0.12, 0.0, 1.0))

        diversity = float(
            np.clip(
                0.42 * decorrelation_score + 0.33 * peak_diversity + 0.25 * np.clip((shannon_h - 0.55) / 0.38, 0.0, 1.0),
                0.0,
                1.0,
            )
        )
        return diversity, n_spectral_peaks

    def _extract_longitudinal_and_multi_foci(
        self,
        amp_residual: np.ndarray,
        phase_residual: np.ndarray,
        sub_phase_var: np.ndarray,
        subcarrier_dyn_profile: np.ndarray,
        subcarrier_stat_profile: np.ndarray,
        rssi_drop: float,
        perturbation_score: float,
    ) -> Tuple[np.ndarray, float, List[Tuple[float, float]]]:
        num_bins = self._num_range_bins
        if perturbation_score < 0.10:
            return np.zeros(num_bins, dtype=np.float64), 0.5, [(0.5, 0.0)]

        fresnel_energy = np.abs(amp_residual[:26]) * 0.45 + subcarrier_dyn_profile[:26] * 0.65
        x_sub = np.linspace(0.0, 1.0, 26)
        x_bins = np.linspace(0.0, 1.0, num_bins)
        raw_profile = np.interp(x_bins, x_sub, fresnel_energy)

        kernel = np.array([0.15, 0.25, 0.30, 0.25, 0.15])
        smoothed = np.convolve(raw_profile, kernel, mode="same")

        max_val = float(np.max(smoothed))
        normalized = (smoothed / max_val) if max_val > 1e-6 else np.ones(num_bins, dtype=np.float64) * 0.5

        weights = normalized ** 2.2
        w_sum = float(np.sum(weights))
        spectral_focus = float(np.sum(x_bins * weights) / w_sum) if w_sum > 1e-6 else 0.5
        primary_focus = float(np.clip(spectral_focus, 0.16, 0.84))

        total_sub = subcarrier_dyn_profile + subcarrier_stat_profile * 0.5
        radial_sub = total_sub[26:52]
        local_max = (ndimage.maximum_filter(smoothed, size=4) == smoothed) & (smoothed >= 0.35)
        peak_bins = np.argwhere(local_max).flatten()

        foci: List[Tuple[float, float]] = []
        if peak_bins.size >= 2:
            sorted_bins = sorted(peak_bins.tolist(), key=lambda b: float(smoothed[b]), reverse=True)
            for idx, b in enumerate(sorted_bins[:3]):
                t_val = float(np.clip(x_bins[b], 0.18, 0.84))
                lat_sign = -1.0 if idx % 2 == 0 else 1.0
                lat_mag = 0.07 if idx > 0 else 0.04
                foci.append((t_val, lat_sign * lat_mag))

        if len(foci) == 0:
            foci.append((primary_focus, 0.0))
        if len(foci) == 1:
            t0 = foci[0][0]
            rad_weighted_idx = float(np.argmax(radial_sub)) / 25.0 if float(np.max(radial_sub)) > 0.05 else 0.62
            t1 = float(np.clip(0.28 if t0 > 0.52 else 0.72, 0.20, 0.82))
            t1 = float(np.clip(0.65 * t1 + 0.35 * rad_weighted_idx, 0.20, 0.82))
            if abs(t1 - t0) < 0.18:
                t1 = float(np.clip(t0 - 0.24 if t0 > 0.5 else t0 + 0.24, 0.18, 0.84))
            foci[0] = (t0, -0.055)
            foci.append((t1, 0.055))
        if len(foci) == 2:
            t_mid = float(np.clip(0.50 + (0.18 if abs(0.50 - foci[0][0]) < 0.12 else 0.0), 0.22, 0.78))
            foci.append((t_mid, 0.0))

        return normalized * perturbation_score, primary_focus, foci

    def _empty_metrics(self, node_id: str, last_seen: float) -> LinkDSPMetrics:
        return LinkDSPMetrics(
            node_id=node_id,
            rssi=-95,
            rssi_avg=-95.0,
            estimated_distance_m=3.0,
            packets_per_sec=0.0,
            pearson_r=1.0,
            motion_index=0.0,
            motion_threshold=0.045,
            is_triggered=False,
            current_energy=0.0,
            effective_noise_floor=0.30,
            net_motion_energy=0.0,
            z_score=0.0,
            ttl_remaining_sec=0.0,
            motion_score=0.0,
            presence_score=0.0,
            perturbation_score=0.0,
            walking_energy=0.0,
            vital_energy=0.0,
            shadow_attenuation=0.0,
            pca_s1=0.0,
            pca_s2=0.0,
            pca_s2_empty_floor=0.32,
            pca_s3=0.0,
            pca_s2_s1_ratio=0.0,
            unit_person_energy=1.20,
            subcarrier_diversity=0.0,
            energy_ratio=0.0,
            estimated_people_on_link=0,
            stationary_retention_active=False,
            top_k_indices=list(range(self.top_k)),
            multi_target_foci=[(0.5, 0.0)],
            longitudinal_profile=np.zeros(self._num_range_bins, dtype=np.float64),
            longitudinal_focus=0.5,
            subcarrier_dyn_profile=np.zeros(self.num_subcarriers, dtype=np.float64),
            subcarrier_stat_profile=np.zeros(self.num_subcarriers, dtype=np.float64),
            amplitudes_preview=[0.0] * self.num_subcarriers,
            phases_preview=[0.0] * self.num_subcarriers,
            csi_ratio_preview=[0.0] * self.num_subcarriers,
            online=False,
            last_seen=last_seen,
        )

    def deactivate_nodes_except(self, keep_node_ids: Optional[List[str]] = None) -> None:
        if keep_node_ids is None:
            self.buffers.clear()
            self.packet_timestamps.clear()
            self.count_history_window.clear()
            self.last_retained_count.clear()
            self.has_confirmed_entry.clear()
            self.entry_motion_accum_sec.clear()
            self.stationary_ttl_deadline.clear()
            return
        keep_set = set(keep_node_ids)
        for nid in list(self.buffers.keys()):
            if nid not in keep_set:
                self.buffers[nid].clear()
                if nid in self.packet_timestamps:
                    self.packet_timestamps[nid].clear()
                if nid in self.count_history_window:
                    self.count_history_window[nid].clear()
                self.last_retained_count[nid] = 0
                self.has_confirmed_entry[nid] = False
                self.entry_motion_accum_sec[nid] = 0.0
                self.stationary_ttl_deadline[nid] = 0.0
