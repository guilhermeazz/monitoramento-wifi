from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import signal

from .config import settings
from .models import (
    CalibrationQuality,
    CalibrationState,
    CalibrationStatusPayload,
    CSIPacket,
    NodeDiagnosticDetail,
    NodeSummaryStatus,
    OccupancyState,
)

logger = logging.getLogger("wifi_spatial.dsp")

EPSILON: float = 1e-8
RESAMPLE_BINS_1S: int = 25

# Índices válidos de subportadoras LLTF em HT20 (64 subportadoras -> 52 válidas):
# Remove guard bands inferiores (0..5), subportadora DC central (32) e guard bands superiores (59..63)
HT20_VALID_SUBCARRIER_INDICES: np.ndarray = np.concatenate(
    [np.arange(6, 32, dtype=int), np.arange(33, 59, dtype=int)]
)


@dataclass
class CSIRingSample:
    """Amostra CSI individual armazenada no ring buffer de entrada (Zero-Blocking)."""

    timestamp: float
    rssi: int
    noise_floor: int
    channel: int
    amp_raw: np.ndarray     # Shape: (52,) A_k(t) = sqrt(I_k^2 + Q_k^2)
    amp_norm: np.ndarray    # Shape: (52,) \tilde{A}_k(t) normalizado por L2 (invariante a AGC)
    turbulence_raw: float = 0.0
    turbulence_filt: float = 0.0


@dataclass
class NodeCalibrationProfile:
    """
    Perfil de calibração em 2 etapas de um nó ESP32-S3:
    - Etapa 1 (Sala Vazia): NBVI (12 subportadoras), mu_0, sigma_0, P95, piso de autovalores (lambda_1,0, lambda_2,0).
    - Etapa 2 (Referência 1 Pessoa): ganho de equalização espacial G_l e envelope 1p (lambda_1_ref, lambda_2_ref, E_1_ref).
    """

    node_id: str
    nbvi_indices: List[int] = field(default_factory=lambda: [2, 6, 10, 14, 18, 22, 27, 31, 35, 39, 43, 47])
    static_indices: List[int] = field(default_factory=lambda: [10, 22, 31, 43])
    nbvi_scores: List[float] = field(default_factory=lambda: [0.02] * 52)
    baseline_mean_norm: np.ndarray = field(
        default_factory=lambda: np.full(52, 1.0 / np.sqrt(52.0), dtype=np.float64)
    )
    baseline_std_norm: np.ndarray = field(
        default_factory=lambda: np.full(52, 0.002, dtype=np.float64)
    )
    mvs_mean_empty: float = 6.0e-7
    mvs_std_empty: float = 3.5e-7
    mvs_p95_empty: float = 2.0e-6
    threshold_on_base: float = 2.5e-6
    threshold_off_base: float = 2.125e-6
    static_var_threshold: float = 1.5e-5
    eig_floor: List[float] = field(default_factory=lambda: [2.5e-5, 1.0e-5, 5.0e-6])
    gain_g: float = 1.0
    eig1_ref_1p: float = 1.8e-3
    eig2_ref_1p: float = 2.6e-4
    energy_ref_1p: float = 2.6
    stage2_calibrated: bool = False
    quality: CalibrationQuality = "OPTIMAL"
    measured_pps: float = 50.0
    sample_count: int = 0
    calibrated: bool = False
    calibrated_at: float = field(default_factory=time.time)


@dataclass
class NodeRuntimeState:
    """Estado dinâmico em tempo real avaliado pelo loop DSP para cada nó ESP32-S3."""

    node_id: str
    state: OccupancyState = "EMPTY"
    mvs_current: float = 0.0
    z_score: float = 0.0
    gain_g: float = 1.0
    calibrated_energy: float = 0.0
    turbulence_raw: float = 0.0
    turbulence_filt: float = 0.0
    threshold_on: float = 2.5e-6
    threshold_off: float = 2.125e-6
    intensity: float = 0.0
    eigenvalues: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    eig1_ref_1p: float = 1.8e-3
    eig2_ref_1p: float = 2.6e-4
    eig_floor: List[float] = field(default_factory=lambda: [2.5e-5, 1.0e-5, 5.0e-6])
    energy_ref_1p: float = 2.6
    turb_1s_resampled: Optional[np.ndarray] = None
    sub12_1s_resampled: Optional[np.ndarray] = None
    consecutive_motion_ticks: int = 0
    hold_until: float = 0.0
    pps: float = 0.0
    rssi: int = -55
    noise_floor: int = -92
    channel: int = 6
    online: bool = False
    last_seen: float = 0.0
    mvs_history: deque = field(default_factory=lambda: deque(maxlen=300))


class DSPEngine:
    """
    Motor DSP Vetorizado de Baixa Latência para Monitoramento Espacial Wi-Fi CSI:
    - Sanitização HT20 (64 -> 52 subportadoras) + Normalização L2 (Invariante a saltos de AGC)
    - Seleção NBVI (12 subportadoras ativas, blindada contra subportadoras nulas/mortas)
    - Filtro de Hampel + Butterworth Passa-Baixa (10 Hz) + MVS por Timestamp (0.5 s)
    - Histerese Schmitt Trigger + Hold Timer para retenção transparente de presença estática
    - Normalização por Link (Z-Score + Ganho Calibrado G_l) e Autovalores em janela de 1.0 s
    - Calibração Guiada em 2 Etapas + Adaptação EMA Lenta (> 120s em 0 pessoas)
    """

    def __init__(self, num_subcarriers: int = settings.NUM_SUBCARRIERS):
        self.num_subcarriers = num_subcarriers
        self.nbvi_k = settings.NBVI_TOP_K
        self.static_k = settings.STATIC_TOP_K

        self.ring_buffers: Dict[str, deque[CSIRingSample]] = {}
        self.profiles: Dict[str, NodeCalibrationProfile] = {}
        self.runtime_states: Dict[str, NodeRuntimeState] = {}

        self.sensitivity_multiplier: float = settings.DEFAULT_SENSITIVITY_MULT
        self.hold_time_sec: float = settings.DEFAULT_HOLD_TIME_SEC

        # Wizard de Calibração Guiada em 2 Etapas
        self.cal_state: CalibrationState = "IDLE"
        self.cal_stage: int = 1
        self.cal_prep_duration: float = settings.DEFAULT_PREP_SECONDS
        self.cal_capture_duration: float = settings.DEFAULT_STAGE1_CAPTURE_SECONDS
        self.cal_state_start_time: float = 0.0
        self.cal_completed_time: float = 0.0
        self.cal_buffers: Dict[str, List[CSIRingSample]] = {}
        self.is_calibrated: bool = False
        self.is_stage2_calibrated: bool = False
        self.overall_quality: CalibrationQuality = "UNCALIBRATED"
        self.noise_diagnosis: str = "Aguardando calibração inicial."
        self.last_calibrated_at_str: Optional[str] = None

        # Controle de EMA Lento para sala vazia contínua (> 120s em 0 pessoas)
        self.empty_since_ts: Optional[float] = None
        self._last_ema_update_ts: float = 0.0

        self._default_fs: float = max(25.0, settings.SIMULATOR_HZ)
        self.load_calibration_from_disk()

    # ==========================================================================
    # Sanitização de Subportadoras e Normalização L2 (Anti-AGC)
    # ==========================================================================

    @staticmethod
    def sanitize_and_compute_amplitudes(
        csi_raw: List[float],
        num_subcarriers: int = 52,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw_arr = np.asarray(csi_raw, dtype=np.float64)
        if raw_arr.size < 4:
            amp_raw = np.ones(num_subcarriers, dtype=np.float64)
            amp_norm = amp_raw / np.sqrt(np.sum(amp_raw ** 2) + EPSILON)
            return amp_raw, amp_norm

        if np.min(raw_arr) >= 0.0 and np.max(raw_arr) <= 255.0 and np.any(raw_arr > 127.0):
            if np.all(np.floor(raw_arr) == raw_arr) and np.sum(raw_arr > 180.0) > 2:
                raw_arr = np.where(raw_arr > 127.0, raw_arr - 256.0, raw_arr)

        even_len = (raw_arr.size // 2) * 2
        raw_arr = raw_arr[:even_len]

        imag = raw_arr[0::2]
        real = raw_arr[1::2]
        amplitudes = np.sqrt(imag ** 2 + real ** 2)

        n_pairs = amplitudes.size
        if n_pairs == num_subcarriers:
            valid_amp = amplitudes
        elif n_pairs >= 64:
            valid_amp = amplitudes[HT20_VALID_SUBCARRIER_INDICES]
        elif n_pairs > num_subcarriers:
            start = (n_pairs - num_subcarriers) // 2
            valid_amp = amplitudes[start : start + num_subcarriers]
        else:
            x_old = np.linspace(0.0, 1.0, n_pairs)
            x_new = np.linspace(0.0, 1.0, num_subcarriers)
            valid_amp = np.interp(x_new, x_old, amplitudes)

        l2_norm = float(np.sqrt(np.sum(valid_amp ** 2) + EPSILON))
        amp_norm = valid_amp / l2_norm
        return valid_amp, amp_norm

    # ==========================================================================
    # Seleção Automática de Subportadoras via NBVI (Blindada contra Nulos)
    # ==========================================================================

    @staticmethod
    def compute_nbvi_selection(
        amp_norm_matrix: np.ndarray,
        top_k: int = 12,
        alpha: float = 0.5,
    ) -> Tuple[List[int], List[int], np.ndarray, np.ndarray, np.ndarray]:
        if amp_norm_matrix.ndim != 2 or amp_norm_matrix.shape[0] < 2:
            n_sub = amp_norm_matrix.shape[-1] if amp_norm_matrix.ndim >= 1 else 52
            mu = np.full(n_sub, 1.0 / np.sqrt(n_sub), dtype=np.float64)
            sigma = np.full(n_sub, 1e-3, dtype=np.float64)
            nbvi = np.full(n_sub, 0.01, dtype=np.float64)
            default_12 = [2, 6, 10, 14, 18, 22, 27, 31, 35, 39, 43, 47][:top_k]
            return default_12, default_12[:4], nbvi, mu, sigma

        mu = np.mean(amp_norm_matrix, axis=0)
        sigma = np.std(amp_norm_matrix, axis=0)

        active_mus = mu[mu > 1e-3]
        median_mu = float(np.median(active_mus)) if active_mus.size > 0 else (1.0 / np.sqrt(mu.size))
        # Descarta subportadoras nulas/mortas ou em deep fading severo (< 30% da mediana ativa)
        dead_mask = (mu < max(0.015, 0.30 * median_mu)) | (sigma < 1e-9)

        safe_mu = np.maximum(mu, 1e-4)
        nbvi = alpha * (sigma / (safe_mu ** 2 + EPSILON)) + (1.0 - alpha) * (sigma / (safe_mu + EPSILON))
        nbvi = np.where(dead_mask, 999.0, nbvi)

        sorted_candidates = np.argsort(nbvi).tolist()

        selected: List[int] = []
        for idx in sorted_candidates:
            if len(selected) >= top_k:
                break
            if nbvi[idx] >= 900.0 and len(selected) >= 4:
                break
            if all(abs(idx - s) >= 2 for s in selected):
                selected.append(int(idx))

        if len(selected) < top_k:
            for idx in sorted_candidates:
                if len(selected) >= top_k:
                    break
                if int(idx) not in selected:
                    selected.append(int(idx))

        static_4 = [int(x) for x in sorted_candidates[:4]]
        selected_sorted = sorted(selected)
        return selected_sorted, static_4, nbvi, mu, sigma

    # ==========================================================================
    # Turbulência Espacial, Filtros, MVS e Covariância Espectral (Autovalores)
    # ==========================================================================

    @staticmethod
    def compute_spatial_turbulence(
        amp_norm: np.ndarray,
        selected_indices: List[int],
    ) -> float:
        sub_vals = amp_norm[selected_indices]
        mean_val = float(np.mean(sub_vals))
        diff = sub_vals - mean_val
        return float(np.sqrt(np.mean(diff ** 2)))

    @staticmethod
    def hampel_filter_1d(
        values: np.ndarray,
        window_size: int = settings.HAMPEL_WINDOW,
        tau_sigma: float = settings.HAMPEL_SIGMA,
    ) -> np.ndarray:
        n = values.size
        if n < 3:
            return values.copy()

        out = values.copy()
        half_w = window_size // 2
        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            win = values[start:end]
            med = float(np.median(win))
            mad = 1.4826 * float(np.median(np.abs(win - med)))
            if abs(values[i] - med) > tau_sigma * (mad + 1e-12):
                out[i] = med
        return out

    def butterworth_lowpass_1d(self, values: np.ndarray, fs_hz: float = 50.0) -> np.ndarray:
        if values.size < 6:
            return values.copy()

        effective_fs = float(np.clip(fs_hz, 15.0, 200.0))
        cutoff = min(settings.BUTTER_CUTOFF_HZ, 0.42 * effective_fs)
        sos = signal.butter(N=2, Wn=cutoff, btype="lowpass", fs=effective_fs, output="sos")
        zi = signal.sosfilt_zi(sos) * values[0]
        filtered, _ = signal.sosfilt(sos, values, zi=zi)
        return filtered

    @staticmethod
    def compute_mvs_by_timestamp(
        timestamps: np.ndarray,
        turbulence_filtered: np.ndarray,
        now_ts: float,
        window_sec: float = settings.MVS_WINDOW_SEC,
    ) -> float:
        if timestamps.size < 2:
            return 0.0
        mask = timestamps >= (now_ts - window_sec)
        win_vals = turbulence_filtered[mask]
        if win_vals.size < 2:
            win_vals = turbulence_filtered[-min(4, turbulence_filtered.size) :]
        if win_vals.size < 2:
            return 0.0
        return float(np.var(win_vals))

    @staticmethod
    def compute_subspace_eigenvalues(
        sub12_window: np.ndarray,
        top_n: int = 3,
    ) -> List[float]:
        """
        Calcula os maiores autovalores (lambda_1 >= lambda_2 >= lambda_3) da matriz de
        covariância das 12 subportadoras selecionadas na janela temporal de 1.0 s.
        Remove modo comum instantâneo e média temporal por subportadora.
        """
        if sub12_window.ndim != 2 or sub12_window.shape[0] < 4:
            return [0.0] * top_n

        # Remove média por pacote (invariância a flutuação residual comum) e média temporal por canal
        pkt_centered = sub12_window - np.mean(sub12_window, axis=1, keepdims=True)
        X = pkt_centered - np.mean(pkt_centered, axis=0, keepdims=True)

        n_samples = X.shape[0]
        cov = (X.T @ X) / float(max(1, n_samples - 1))
        eigvals = np.linalg.eigvalsh(cov)
        eigvals_desc = np.maximum(0.0, eigvals[::-1])
        out = [float(eigvals_desc[i]) if i < eigvals_desc.size else 0.0 for i in range(top_n)]
        return out

    @staticmethod
    def _resample_window_1s(
        ts_arr: np.ndarray,
        turb_filt: np.ndarray,
        sub12_mat: np.ndarray,
        now_ts: float,
        window_sec: float = settings.COV_WINDOW_SEC,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Reamostra os últimos 1.0 s em grade uniforme de 25 pontos para correlação entre links."""
        mask = ts_arr >= (now_ts - window_sec)
        t_w = ts_arr[mask]
        if t_w.size < 4 or (t_w[-1] - t_w[0]) < 0.15:
            return None, None

        turb_w = turb_filt[mask]
        sub_w = sub12_mat[mask]

        # Garante timestamps estritamente crescentes para np.interp
        _, uniq_idx = np.unique(t_w, return_index=True)
        if uniq_idx.size < 4:
            return None, None

        t_u = t_w[uniq_idx]
        turb_u = turb_w[uniq_idx]
        sub_u = sub_w[uniq_idx]

        grid_t = np.linspace(max(t_u[0], now_ts - window_sec), t_u[-1], RESAMPLE_BINS_1S)
        turb_rs = np.interp(grid_t, t_u, turb_u)
        turb_rs = turb_rs - np.mean(turb_rs)

        sub_rs = np.zeros((RESAMPLE_BINS_1S, sub_u.shape[1]), dtype=np.float64)
        for k in range(sub_u.shape[1]):
            col = np.interp(grid_t, t_u, sub_u[:, k])
            sub_rs[:, k] = col - np.mean(col)

        return turb_rs, sub_rs

    # ==========================================================================
    # Ingestão UDP Zero-Blocking e Processamento de Tick Assíncrono (20 Hz)
    # ==========================================================================

    def enqueue_packet(self, packet: CSIPacket) -> CSIRingSample:
        node_id = packet.node_id
        if node_id not in self.ring_buffers:
            self.ring_buffers[node_id] = deque(maxlen=settings.RING_BUFFER_MAX_PACKETS)
        if node_id not in self.profiles:
            self.profiles[node_id] = NodeCalibrationProfile(node_id=node_id)
        if node_id not in self.runtime_states:
            self.runtime_states[node_id] = NodeRuntimeState(node_id=node_id)

        amp_raw, amp_norm = self.sanitize_and_compute_amplitudes(packet.csi_raw, self.num_subcarriers)
        prof = self.profiles[node_id]
        turb_raw = self.compute_spatial_turbulence(amp_norm, prof.nbvi_indices)

        sample = CSIRingSample(
            timestamp=packet.timestamp,
            rssi=packet.rssi,
            noise_floor=packet.noise_floor,
            channel=packet.channel,
            amp_raw=amp_raw,
            amp_norm=amp_norm,
            turbulence_raw=turb_raw,
            turbulence_filt=turb_raw,
        )
        self.ring_buffers[node_id].append(sample)

        if self.cal_state in ("CAPTURING", "CAPTURING_STAGE2"):
            if node_id not in self.cal_buffers:
                self.cal_buffers[node_id] = []
            self.cal_buffers[node_id].append(sample)

        return sample

    def ingest_packet(self, packet: CSIPacket) -> CSIRingSample:
        return self.enqueue_packet(packet)

    def process_node_tick(self, node_id: str, now: Optional[float] = None) -> NodeRuntimeState:
        if now is None:
            now = time.time()

        if node_id not in self.runtime_states:
            self.runtime_states[node_id] = NodeRuntimeState(node_id=node_id)
        if node_id not in self.profiles:
            self.profiles[node_id] = NodeCalibrationProfile(node_id=node_id)

        rt = self.runtime_states[node_id]
        prof = self.profiles[node_id]
        buf = self.ring_buffers.get(node_id)

        if not buf or len(buf) == 0:
            rt.online = False
            rt.state = "EMPTY"
            rt.mvs_current = 0.0
            rt.z_score = 0.0
            rt.calibrated_energy = 0.0
            rt.intensity = 0.0
            rt.pps = 0.0
            rt.turb_1s_resampled = None
            rt.sub12_1s_resampled = None
            return rt

        latest = buf[-1]
        rt.last_seen = latest.timestamp
        rt.rssi = latest.rssi
        rt.noise_floor = latest.noise_floor
        rt.channel = latest.channel
        rt.online = (now - latest.timestamp) <= settings.NODE_ONLINE_TIMEOUT_SEC

        if not rt.online:
            rt.state = "EMPTY"
            rt.mvs_current = 0.0
            rt.z_score = 0.0
            rt.calibrated_energy = 0.0
            rt.intensity = 0.0
            rt.pps = 0.0
            rt.consecutive_motion_ticks = 0
            rt.turb_1s_resampled = None
            rt.sub12_1s_resampled = None
            return rt

        recent_samples = [s for s in buf if (latest.timestamp - s.timestamp) <= 2.0]
        if len(recent_samples) < 3:
            recent_samples = list(buf)[-min(len(buf), 15) :]

        ts_arr = np.array([s.timestamp for s in recent_samples], dtype=np.float64)
        span_sec = float(ts_arr[-1] - ts_arr[0]) if ts_arr.size >= 2 else 0.0
        rt.pps = round((ts_arr.size - 1) / span_sec, 1) if span_sec > 0.05 else float(ts_arr.size)

        if not prof.calibrated and self.cal_state == "IDLE":
            buf_span = (buf[-1].timestamp - buf[0].timestamp) if len(buf) >= 2 else 0.0
            if len(buf) >= 35 or (len(buf) >= 12 and buf_span >= 2.5):
                self._auto_bootstrap_profile(node_id, list(buf))

        amp_norm_mat = np.stack([s.amp_norm for s in recent_samples], axis=0)
        sub_12 = amp_norm_mat[:, prof.nbvi_indices]
        mean_12 = np.mean(sub_12, axis=1, keepdims=True)
        turb_series = np.sqrt(np.mean((sub_12 - mean_12) ** 2, axis=1))

        turb_hampel = self.hampel_filter_1d(
            turb_series,
            window_size=settings.HAMPEL_WINDOW,
            tau_sigma=settings.HAMPEL_SIGMA,
        )
        turb_filt = self.butterworth_lowpass_1d(turb_hampel, fs_hz=max(20.0, rt.pps))

        rt.turbulence_raw = float(turb_series[-1])
        rt.turbulence_filt = float(turb_filt[-1])
        latest.turbulence_raw = rt.turbulence_raw
        latest.turbulence_filt = rt.turbulence_filt

        t_latest = float(ts_arr[-1])
        mvs_val = self.compute_mvs_by_timestamp(
            timestamps=ts_arr,
            turbulence_filtered=turb_filt,
            now_ts=t_latest,
            window_sec=settings.MVS_WINDOW_SEC,
        )
        rt.mvs_current = mvs_val
        rt.mvs_history.append((now, mvs_val))

        # Autovalores da matriz de covariância em janela de 1.0 s
        mask_1s = ts_arr >= (t_latest - settings.COV_WINDOW_SEC)
        sub12_1s = sub_12[mask_1s] if np.sum(mask_1s) >= 4 else sub_12
        rt.eigenvalues = self.compute_subspace_eigenvalues(sub12_1s, top_n=3)
        rt.eig1_ref_1p = prof.eig1_ref_1p
        rt.eig2_ref_1p = prof.eig2_ref_1p
        rt.eig_floor = list(prof.eig_floor)
        rt.energy_ref_1p = prof.energy_ref_1p
        rt.gain_g = prof.gain_g

        turb_rs, sub_rs = self._resample_window_1s(
            ts_arr=ts_arr,
            turb_filt=turb_filt,
            sub12_mat=sub_12,
            now_ts=t_latest,
            window_sec=settings.COV_WINDOW_SEC,
        )
        rt.turb_1s_resampled = turb_rs
        rt.sub12_1s_resampled = sub_rs

        # Histerese Schmitt Trigger + Debounce (k >= 3) + Hold Timer
        theta_on = max(1e-9, prof.threshold_on_base * self.sensitivity_multiplier)
        theta_off = 0.85 * theta_on
        rt.threshold_on = theta_on
        rt.threshold_off = theta_off
        rt.intensity = float(np.clip(mvs_val / theta_on, 0.0, 12.0))

        # Z-Score e Energia Normalizada por Link (com Ganho Calibrado G_l em domínio RMS linear)
        safe_std = max(prof.mvs_std_empty, 0.15 * prof.mvs_mean_empty, 5e-8)
        rt.z_score = float(max(0.0, (mvs_val - prof.mvs_mean_empty) / safe_std))

        rms_cur = float(np.sqrt(max(0.0, mvs_val)))
        rms_floor = float(np.sqrt(max(0.0, prof.mvs_p95_empty))) * 0.90
        rms_unit = float(np.sqrt(max(1e-9, theta_on)))
        raw_linear_energy = max(0.0, (rms_cur - rms_floor) / rms_unit)

        if mvs_val > theta_on:
            rt.consecutive_motion_ticks += 1
            if rt.consecutive_motion_ticks >= settings.DEBOUNCE_CONSECUTIVE_TICKS:
                rt.state = "MOTION"
                rt.hold_until = now + self.hold_time_sec
        else:
            rt.consecutive_motion_ticks = 0
            if rt.state == "MOTION":
                if mvs_val < theta_off:
                    rt.state = "PRESENCE_STATIC"
                    rt.hold_until = now + self.hold_time_sec
            elif rt.state == "PRESENCE_STATIC":
                # Decaimento determinístico (TTL): expira para EMPTY ao fim do hold_time_sec sem loop infinito
                if now >= rt.hold_until:
                    rt.state = "EMPTY"
                    rt.intensity = 0.0
                    rt.calibrated_energy = 0.0

        if rt.state == "EMPTY":
            rt.intensity = float(np.clip(mvs_val / theta_on, 0.0, 0.95)) if mvs_val > 0 else 0.0
            rt.calibrated_energy = 0.0  # Hard Zero Gate em sala vazia
        elif rt.state == "PRESENCE_STATIC":
            rt.calibrated_energy = float(max(0.25 * prof.gain_g, prof.gain_g * raw_linear_energy))
        else:
            rt.calibrated_energy = float(prof.gain_g * raw_linear_energy)

        return rt

    def _evaluate_static_branch(
        self,
        amp_norm_mat: np.ndarray,
        prof: NodeCalibrationProfile,
    ) -> bool:
        if amp_norm_mat.shape[0] < 6:
            return False
        s4_idx = prof.static_indices
        s4_mat = amp_norm_mat[:, s4_idx]
        s4_var = float(np.mean(np.var(s4_mat, axis=0)))
        return s4_var > (prof.static_var_threshold * 1.35)

    def process_all_nodes_tick(
        self,
        candidate_node_ids: List[str],
        now: Optional[float] = None,
    ) -> Dict[str, NodeRuntimeState]:
        if now is None:
            now = time.time()
        self.tick_calibration_state(now)
        results: Dict[str, NodeRuntimeState] = {}
        for nid in candidate_node_ids:
            results[nid] = self.process_node_tick(nid, now=now)

        # Atualização lenta (EMA) do baseline vazio APENAS após > 120s contínuos em 0 pessoas
        online_states = [st for st in results.values() if st.online]
        if online_states and all(st.state == "EMPTY" for st in online_states) and self.cal_state == "IDLE":
            if self.empty_since_ts is None:
                self.empty_since_ts = now
            elif (now - self.empty_since_ts) >= settings.EMPTY_EMA_MIN_IDLE_SEC:
                if (now - self._last_ema_update_ts) >= 5.0:
                    self._apply_slow_empty_ema_update(online_states)
                    self._last_ema_update_ts = now
        else:
            self.empty_since_ts = None

        return results

    def _apply_slow_empty_ema_update(self, empty_states: List[NodeRuntimeState], alpha_ema: float = 0.02) -> None:
        """Adapta lentamente o baseline de ruído apenas quando a sala está vazia por > 120s."""
        for st in empty_states:
            prof = self.profiles.get(st.node_id)
            if prof is None or not prof.calibrated:
                continue
            cur_mvs = max(1e-8, st.mvs_current)
            # Proteção contra absorção acidental de spike isolado
            if cur_mvs > prof.threshold_on_base * 0.90:
                continue
            prof.mvs_mean_empty = (1.0 - alpha_ema) * prof.mvs_mean_empty + alpha_ema * cur_mvs
            dev = abs(cur_mvs - prof.mvs_mean_empty)
            prof.mvs_std_empty = max(
                5e-8,
                (1.0 - alpha_ema) * prof.mvs_std_empty + alpha_ema * dev,
            )
            est_p95 = max(1.5e-7, prof.mvs_mean_empty + 2.0 * prof.mvs_std_empty)
            prof.mvs_p95_empty = (1.0 - alpha_ema) * prof.mvs_p95_empty + alpha_ema * est_p95
            prof.threshold_on_base = max(1.25 * prof.mvs_p95_empty, prof.mvs_mean_empty + 3.2 * prof.mvs_std_empty)
            prof.threshold_off_base = 0.85 * prof.threshold_on_base

            if len(st.eigenvalues) >= 3 and st.eigenvalues[0] > 0:
                for idx in range(min(3, len(prof.eig_floor))):
                    prof.eig_floor[idx] = (1.0 - alpha_ema) * prof.eig_floor[idx] + alpha_ema * st.eigenvalues[idx]

    # ==========================================================================
    # Wizard de Calibração Guiada em 2 Etapas
    # Etapa 1: Baseline Sala Vazia (5s saída + 15s coleta)
    # Etapa 2: Referência 1 Pessoa (20s caminhando pela sala)
    # ==========================================================================

    def start_calibration(
        self,
        prep_seconds: Optional[float] = None,
        capture_seconds: Optional[float] = None,
        stage: int = 1,
        now: Optional[float] = None,
    ) -> CalibrationStatusPayload:
        if now is None:
            now = time.time()

        self.cal_stage = 2 if stage == 2 else 1
        if self.cal_stage == 1:
            self.cal_prep_duration = (
                max(0.0, float(prep_seconds))
                if prep_seconds is not None
                else settings.DEFAULT_PREP_SECONDS
            )
            self.cal_capture_duration = (
                max(1.0, float(capture_seconds))
                if capture_seconds is not None
                else settings.DEFAULT_STAGE1_CAPTURE_SECONDS
            )
            self.cal_state = "PREPARING" if self.cal_prep_duration > 0.0 else "CAPTURING"
        else:
            self.cal_prep_duration = max(0.0, float(prep_seconds)) if prep_seconds is not None else 2.0
            self.cal_capture_duration = (
                max(2.0, float(capture_seconds))
                if capture_seconds is not None
                else settings.DEFAULT_STAGE2_CAPTURE_SECONDS
            )
            self.cal_state = "PREPARING_STAGE2" if self.cal_prep_duration > 0.0 else "CAPTURING_STAGE2"

        self.cal_state_start_time = now
        self.cal_buffers = {nid: [] for nid in self.ring_buffers.keys()}
        return self.get_calibration_status(now)

    def cancel_calibration(self, now: Optional[float] = None) -> CalibrationStatusPayload:
        if now is None:
            now = time.time()
        self.cal_state = "IDLE"
        self.cal_buffers.clear()
        return self.get_calibration_status(now)

    def tick_calibration_state(self, now: Optional[float] = None) -> CalibrationStatusPayload:
        if now is None:
            now = time.time()

        if self.cal_state == "PREPARING":
            elapsed = now - self.cal_state_start_time
            if elapsed >= self.cal_prep_duration:
                self.cal_state = "CAPTURING"
                self.cal_state_start_time = now
                self.cal_buffers = {nid: [] for nid in self.ring_buffers.keys()}

        elif self.cal_state == "CAPTURING":
            elapsed = now - self.cal_state_start_time
            if elapsed >= self.cal_capture_duration:
                self._finalize_stage1_calibration(now)

        elif self.cal_state == "PREPARING_STAGE2":
            elapsed = now - self.cal_state_start_time
            if elapsed >= self.cal_prep_duration:
                self.cal_state = "CAPTURING_STAGE2"
                self.cal_state_start_time = now
                self.cal_buffers = {nid: [] for nid in self.ring_buffers.keys()}

        elif self.cal_state == "CAPTURING_STAGE2":
            elapsed = now - self.cal_state_start_time
            if elapsed >= self.cal_capture_duration:
                self._finalize_stage2_calibration(now)

        elif self.cal_state == "COMPLETED":
            if (now - self.cal_completed_time) >= 6.0:
                self.cal_state = "IDLE"

        return self.get_calibration_status(now)

    def get_calibration_status(self, now: Optional[float] = None) -> CalibrationStatusPayload:
        if now is None:
            now = time.time()

        if self.cal_state in ("PREPARING", "PREPARING_STAGE2"):
            elapsed = now - self.cal_state_start_time
            rem = max(0.0, self.cal_prep_duration - elapsed)
            prog = float(np.clip(elapsed / max(0.1, self.cal_prep_duration), 0.0, 1.0))
            return CalibrationStatusPayload(
                state=self.cal_state,
                stage=self.cal_stage,
                progress=round(prog, 2),
                remaining_sec=round(rem, 1),
                quality=self.overall_quality,
                calibrated=self.is_calibrated,
                stage2_calibrated=self.is_stage2_calibrated,
                noise_diagnosis=self.noise_diagnosis,
                last_calibrated_at=self.last_calibrated_at_str,
            )

        if self.cal_state in ("CAPTURING", "CAPTURING_STAGE2"):
            elapsed = now - self.cal_state_start_time
            rem = max(0.0, self.cal_capture_duration - elapsed)
            prog = float(np.clip(elapsed / max(0.1, self.cal_capture_duration), 0.0, 0.99))
            return CalibrationStatusPayload(
                state=self.cal_state,
                stage=self.cal_stage,
                progress=round(prog, 2),
                remaining_sec=round(rem, 1),
                quality=self.overall_quality,
                calibrated=self.is_calibrated,
                stage2_calibrated=self.is_stage2_calibrated,
                noise_diagnosis=self.noise_diagnosis,
                last_calibrated_at=self.last_calibrated_at_str,
            )

        if self.cal_state in ("STAGE1_DONE", "COMPLETED"):
            return CalibrationStatusPayload(
                state=self.cal_state,
                stage=self.cal_stage,
                progress=1.0,
                remaining_sec=0.0,
                quality=self.overall_quality,
                calibrated=self.is_calibrated,
                stage2_calibrated=self.is_stage2_calibrated,
                noise_diagnosis=self.noise_diagnosis,
                last_calibrated_at=self.last_calibrated_at_str,
            )

        return CalibrationStatusPayload(
            state="IDLE",
            stage=2 if self.is_stage2_calibrated else 1,
            progress=1.0 if self.is_calibrated else 0.0,
            remaining_sec=0.0,
            quality=self.overall_quality,
            calibrated=self.is_calibrated,
            stage2_calibrated=self.is_stage2_calibrated,
            noise_diagnosis=self.noise_diagnosis,
            last_calibrated_at=self.last_calibrated_at_str,
        )

    def calibrate_node_from_samples(
        self,
        node_id: str,
        samples: List[CSIRingSample],
        alpha: float = 0.5,
    ) -> NodeCalibrationProfile:
        """
        Etapa 1 — Baseline Sala Vazia:
        Calcula NBVI (12 subportadoras válidas), mu_0, sigma_0, P95(MVS) por link e piso de autovalores (lambda_1,0, lambda_2,0).
        """
        if len(samples) < 4:
            prof = self.profiles.get(node_id, NodeCalibrationProfile(node_id=node_id))
            self.profiles[node_id] = prof
            return prof

        prev_prof = self.profiles.get(node_id)
        amp_norm_mat = np.stack([s.amp_norm for s in samples], axis=0)
        ts_arr = np.array([s.timestamp for s in samples], dtype=np.float64)

        nbvi_12, static_4, nbvi_scores, mu, sigma = self.compute_nbvi_selection(
            amp_norm_matrix=amp_norm_mat,
            top_k=self.nbvi_k,
            alpha=alpha,
        )

        sub_12 = amp_norm_mat[:, nbvi_12]
        mean_12 = np.mean(sub_12, axis=1, keepdims=True)
        turb_series = np.sqrt(np.mean((sub_12 - mean_12) ** 2, axis=1))

        span_sec = float(ts_arr[-1] - ts_arr[0]) if ts_arr.size >= 2 else 1.0
        measured_pps = (ts_arr.size - 1) / span_sec if span_sec > 0.05 else 50.0

        turb_hampel = self.hampel_filter_1d(turb_series)
        turb_filt = self.butterworth_lowpass_1d(turb_hampel, fs_hz=max(20.0, measured_pps))

        mvs_empty_values: List[float] = []
        eig1_empty: List[float] = []
        eig2_empty: List[float] = []
        eig3_empty: List[float] = []

        step_stride = max(1, len(samples) // 80)
        for i in range(2, ts_arr.size, step_stride):
            t_cur = float(ts_arr[i])
            mvs_i = self.compute_mvs_by_timestamp(
                timestamps=ts_arr[: i + 1],
                turbulence_filtered=turb_filt[: i + 1],
                now_ts=t_cur,
                window_sec=settings.MVS_WINDOW_SEC,
            )
            if mvs_i > 0.0:
                mvs_empty_values.append(mvs_i)

            mask_cov = ts_arr[: i + 1] >= (t_cur - settings.COV_WINDOW_SEC)
            win_sub = sub_12[: i + 1][mask_cov]
            if win_sub.shape[0] >= 6:
                eigs = self.compute_subspace_eigenvalues(win_sub, top_n=3)
                eig1_empty.append(eigs[0])
                eig2_empty.append(eigs[1])
                eig3_empty.append(eigs[2])

        if not mvs_empty_values:
            mvs_empty_values = [float(np.var(turb_filt))]

        mu_0 = max(1e-8, float(np.mean(mvs_empty_values)))
        sigma_0 = max(5e-8, float(np.std(mvs_empty_values)))
        p95_mvs = max(1.5e-7, float(np.percentile(mvs_empty_values, 95)))
        max_mvs = max(p95_mvs, float(np.max(mvs_empty_values)))
        # Noise Gate rigoroso: acima do pico máximo observado no vazio (1.35x max) e Z-score >= 4.5 sigma
        theta_on = max(1.35 * max_mvs, 1.50 * p95_mvs, mu_0 + 4.5 * sigma_0)
        theta_off = 0.85 * theta_on

        if eig1_empty:
            eig_floor = [
                max(1e-7, float(np.percentile(eig1_empty, 95))),
                max(5e-8, float(np.percentile(eig2_empty, 95))),
                max(2e-8, float(np.percentile(eig3_empty, 95))),
            ]
        else:
            eigs_all = self.compute_subspace_eigenvalues(sub_12, top_n=3)
            eig_floor = [max(1e-7, v * 1.25) for v in eigs_all]

        s4_var = float(np.mean(np.var(amp_norm_mat[:, static_4], axis=0)))
        static_thresh = max(1e-7, s4_var * 1.5)

        valid_nbvi_vals = [nbvi_scores[idx] for idx in nbvi_12 if nbvi_scores[idx] < 900.0]
        mean_nbvi_12 = float(np.mean(valid_nbvi_vals)) if valid_nbvi_vals else 0.30
        if measured_pps >= 30.0 and mean_nbvi_12 < 0.12:
            quality: CalibrationQuality = "OPTIMAL"
        elif measured_pps >= 15.0 and mean_nbvi_12 < 0.25:
            quality = "ACCEPTABLE"
        else:
            quality = "NOISY"

        # Preserva parâmetros da Etapa 2 se já calibrada anteriormente, senão deriva envelope inicial coerente
        gain_g = prev_prof.gain_g if (prev_prof and prev_prof.stage2_calibrated) else 1.0
        eig1_ref = (
            prev_prof.eig1_ref_1p
            if (prev_prof and prev_prof.stage2_calibrated)
            else max(1.4e-3, eig_floor[0] * 55.0)
        )
        eig2_ref = (
            prev_prof.eig2_ref_1p
            if (prev_prof and prev_prof.stage2_calibrated)
            else max(2.2e-4, eig_floor[1] * 14.0)
        )
        energy_ref = prev_prof.energy_ref_1p if (prev_prof and prev_prof.stage2_calibrated) else 2.6

        profile = NodeCalibrationProfile(
            node_id=node_id,
            nbvi_indices=nbvi_12,
            static_indices=static_4,
            nbvi_scores=[round(float(x), 5) for x in nbvi_scores.tolist()],
            baseline_mean_norm=mu,
            baseline_std_norm=sigma,
            mvs_mean_empty=mu_0,
            mvs_std_empty=sigma_0,
            mvs_p95_empty=p95_mvs,
            threshold_on_base=theta_on,
            threshold_off_base=theta_off,
            static_var_threshold=static_thresh,
            eig_floor=eig_floor,
            gain_g=gain_g,
            eig1_ref_1p=eig1_ref,
            eig2_ref_1p=eig2_ref,
            energy_ref_1p=energy_ref,
            stage2_calibrated=bool(prev_prof and prev_prof.stage2_calibrated),
            quality=quality,
            measured_pps=round(measured_pps, 1),
            sample_count=len(samples),
            calibrated=True,
            calibrated_at=time.time(),
        )
        self.profiles[node_id] = profile

        if node_id in self.runtime_states:
            rt = self.runtime_states[node_id]
            rt.state = "EMPTY"
            rt.consecutive_motion_ticks = 0
            rt.hold_until = 0.0
            rt.intensity = 0.0
            rt.calibrated_energy = 0.0

        return profile

    def calibrate_stage2_single_person(
        self,
        samples_by_node: Dict[str, List[CSIRingSample]],
    ) -> Dict[str, NodeCalibrationProfile]:
        """
        Etapa 2 — Referência 1 Pessoa (20s caminhando pela sala):
        Calcula o ganho de equalização espacial G_l por link (para que o ESP mais próximo do E1200 não domine)
        e o envelope espectral/energético de 1 pessoa (lambda_1_ref, lambda_2_ref, E_1_ref).
        """
        raw_link_energies: Dict[str, float] = {}
        node_eigs: Dict[str, Tuple[float, float]] = {}

        for nid, samples in samples_by_node.items():
            if len(samples) < 8:
                continue
            prof = self.profiles.get(nid, NodeCalibrationProfile(node_id=nid))
            amp_norm_mat = np.stack([s.amp_norm for s in samples], axis=0)
            ts_arr = np.array([s.timestamp for s in samples], dtype=np.float64)

            sub_12 = amp_norm_mat[:, prof.nbvi_indices]
            mean_12 = np.mean(sub_12, axis=1, keepdims=True)
            turb_series = np.sqrt(np.mean((sub_12 - mean_12) ** 2, axis=1))
            span_sec = float(ts_arr[-1] - ts_arr[0]) if ts_arr.size >= 2 else 1.0
            pps = (ts_arr.size - 1) / span_sec if span_sec > 0.05 else 50.0
            turb_filt = self.butterworth_lowpass_1d(self.hampel_filter_1d(turb_series), fs_hz=max(20.0, pps))

            rms_floor = float(np.sqrt(max(0.0, prof.mvs_p95_empty))) * 0.90
            rms_unit = float(np.sqrt(max(1e-9, prof.threshold_on_base)))

            lin_energies: List[float] = []
            eig1_list: List[float] = []
            eig2_list: List[float] = []

            stride = max(1, len(samples) // 80)
            for i in range(4, ts_arr.size, stride):
                t_cur = float(ts_arr[i])
                mvs_v = self.compute_mvs_by_timestamp(
                    timestamps=ts_arr[: i + 1],
                    turbulence_filtered=turb_filt[: i + 1],
                    now_ts=t_cur,
                    window_sec=settings.MVS_WINDOW_SEC,
                )
                if mvs_v > 0.0:
                    e_lin = max(0.0, (math.sqrt(max(0.0, mvs_v)) - rms_floor) / rms_unit)
                    if e_lin > 0.25:
                        lin_energies.append(e_lin)

                mask_cov = ts_arr[: i + 1] >= (t_cur - settings.COV_WINDOW_SEC)
                win_sub = sub_12[: i + 1][mask_cov]
                if win_sub.shape[0] >= 6:
                    eigs = self.compute_subspace_eigenvalues(win_sub, top_n=3)
                    if eigs[0] > prof.eig_floor[0] * 2.0:
                        eig1_list.append(eigs[0])
                        eig2_list.append(eigs[1])

            if lin_energies:
                raw_link_energies[nid] = float(np.percentile(lin_energies, 75))
            if eig1_list:
                node_eigs[nid] = (
                    float(np.percentile(eig1_list, 80)),
                    float(np.percentile(eig2_list, 85)),
                )

        if raw_link_energies:
            target_ref_energy = float(np.median(list(raw_link_energies.values())))
            target_ref_energy = float(np.clip(target_ref_energy, 1.4, 5.0))
        else:
            target_ref_energy = 2.6

        for nid, prof in self.profiles.items():
            if nid in raw_link_energies and raw_link_energies[nid] > 0.15:
                g_l = float(np.clip(target_ref_energy / raw_link_energies[nid], 0.15, 3.0))
                prof.gain_g = round(g_l, 4)
                prof.energy_ref_1p = round(target_ref_energy, 4)
                prof.stage2_calibrated = True
            if nid in node_eigs:
                prof.eig1_ref_1p = max(prof.eig_floor[0] * 8.0, node_eigs[nid][0])
                prof.eig2_ref_1p = max(prof.eig_floor[1] * 4.0, node_eigs[nid][1])
                prof.stage2_calibrated = True

        self.is_stage2_calibrated = any(p.stage2_calibrated for p in self.profiles.values())
        return self.profiles

    def _auto_bootstrap_profile(self, node_id: str, samples: List[CSIRingSample]) -> None:
        prof = self.calibrate_node_from_samples(node_id, samples)
        prof.threshold_on_base *= 1.35
        prof.threshold_off_base = 0.85 * prof.threshold_on_base
        prof.calibrated = True

    def _finalize_stage1_calibration(self, now: float) -> None:
        qualities: List[CalibrationQuality] = []
        pps_list: List[float] = []
        for nid in list(self.ring_buffers.keys()):
            samples = self.cal_buffers.get(nid, [])
            if len(samples) < 5 and nid in self.ring_buffers:
                samples = list(self.ring_buffers[nid])
            if len(samples) >= 4:
                prof = self.calibrate_node_from_samples(nid, samples)
                qualities.append(prof.quality)
                pps_list.append(prof.measured_pps)

        if qualities:
            if all(q == "OPTIMAL" for q in qualities):
                self.overall_quality = "OPTIMAL"
                self.noise_diagnosis = (
                    f"Ruído de fundo excelente ({min(pps_list):.0f}+ Hz). "
                    "Pronto para a Etapa 2 (Referência 1 Pessoa)."
                )
            elif any(q == "NOISY" for q in qualities):
                self.overall_quality = "NOISY"
                self.noise_diagnosis = (
                    "Ruído térmico/multipath elevado ou baixa taxa de pacotes em algum nó. "
                    "Verifique canal fixo HT20 e distância TX-RX."
                )
            else:
                self.overall_quality = "ACCEPTABLE"
                self.noise_diagnosis = (
                    "Baseline vazio aceitável. Recomendado executar a Etapa 2 caminhando pela sala."
                )
        else:
            self.overall_quality = "ACCEPTABLE"
            self.noise_diagnosis = "Baseline inicial registrado."

        self.is_calibrated = True
        self.cal_state = "STAGE1_DONE"
        self.cal_completed_time = now
        self.last_calibrated_at_str = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        self.save_calibration_to_disk()

    def _finalize_stage2_calibration(self, now: float) -> None:
        samples_map: Dict[str, List[CSIRingSample]] = {}
        for nid in list(self.ring_buffers.keys()):
            samples = self.cal_buffers.get(nid, [])
            if len(samples) >= 8:
                samples_map[nid] = samples

        if samples_map:
            self.calibrate_stage2_single_person(samples_map)
            gains_txt = ", ".join(f"{nid}: {p.gain_g:.2f}x" for nid, p in sorted(self.profiles.items()))
            self.noise_diagnosis = f"Calibração 2 Etapas concluída! Ganhos equalizados ({gains_txt})."
        else:
            self.noise_diagnosis = "Etapa 2 finalizada (poucas amostras de movimento; mantendo ganho nominal)."

        self.is_calibrated = True
        self.cal_state = "COMPLETED"
        self.cal_completed_time = now
        self.last_calibrated_at_str = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        self.save_calibration_to_disk()

    # ==========================================================================
    # Persistência em calibration_profile.json (Versão 4.2 compatível)
    # ==========================================================================

    def save_calibration_to_disk(self) -> bool:
        try:
            path = settings.CALIBRATION_PROFILE_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, Any] = {
                "version": "4.2",
                "saved_at": time.time(),
                "saved_at_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_calibrated": self.is_calibrated,
                "is_stage2_calibrated": self.is_stage2_calibrated,
                "overall_quality": self.overall_quality,
                "noise_diagnosis": self.noise_diagnosis,
                "nodes": {},
            }
            for nid, prof in self.profiles.items():
                payload["nodes"][nid] = {
                    "nbvi_indices": prof.nbvi_indices,
                    "static_indices": prof.static_indices,
                    "nbvi_scores": prof.nbvi_scores,
                    "baseline_mean_norm": [round(float(x), 6) for x in prof.baseline_mean_norm.tolist()],
                    "baseline_std_norm": [round(float(x), 6) for x in prof.baseline_std_norm.tolist()],
                    "mvs_mean_empty": prof.mvs_mean_empty,
                    "mvs_std_empty": prof.mvs_std_empty,
                    "mvs_p95_empty": prof.mvs_p95_empty,
                    "threshold_on_base": prof.threshold_on_base,
                    "threshold_off_base": prof.threshold_off_base,
                    "static_var_threshold": prof.static_var_threshold,
                    "eig_floor": [round(float(v), 8) for v in prof.eig_floor],
                    "gain_g": prof.gain_g,
                    "eig1_ref_1p": prof.eig1_ref_1p,
                    "eig2_ref_1p": prof.eig2_ref_1p,
                    "energy_ref_1p": prof.energy_ref_1p,
                    "stage2_calibrated": prof.stage2_calibrated,
                    "quality": prof.quality,
                    "measured_pps": prof.measured_pps,
                    "sample_count": prof.sample_count,
                    "calibrated": prof.calibrated,
                    "calibrated_at": prof.calibrated_at,
                }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return True
        except Exception as exc:
            logger.warning(f"Erro ao salvar calibration_profile.json: {exc}")
            return False

    def load_calibration_from_disk(self) -> bool:
        try:
            path = settings.CALIBRATION_PROFILE_PATH
            if not path.exists():
                return False
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") not in ("4.1", "4.2"):
                return False
            nodes_dict = data.get("nodes", {})
            for nid, nd in nodes_dict.items():
                mean_arr = np.asarray(nd.get("baseline_mean_norm", []), dtype=np.float64)
                std_arr = np.asarray(nd.get("baseline_std_norm", []), dtype=np.float64)
                if mean_arr.size != self.num_subcarriers:
                    continue
                nbvi_idx = [int(i) for i in nd.get("nbvi_indices", [])]
                # Descarta perfis sub-amostrados (< 3 Hz / < 25 pacotes) ou com subportadoras nulas
                if float(nd.get("measured_pps", 0.0)) < 3.0 or int(nd.get("sample_count", 0)) < 25:
                    continue
                if any(mean_arr[i] < 1e-3 for i in nbvi_idx if 0 <= i < mean_arr.size):
                    continue
                self.profiles[nid] = NodeCalibrationProfile(
                    node_id=nid,
                    nbvi_indices=nbvi_idx,
                    static_indices=[int(i) for i in nd.get("static_indices", [])],
                    nbvi_scores=[float(x) for x in nd.get("nbvi_scores", [])],
                    baseline_mean_norm=mean_arr,
                    baseline_std_norm=std_arr,
                    mvs_mean_empty=float(nd.get("mvs_mean_empty", 6.0e-7)),
                    mvs_std_empty=float(nd.get("mvs_std_empty", 3.5e-7)),
                    mvs_p95_empty=float(nd.get("mvs_p95_empty", 2.0e-6)),
                    threshold_on_base=float(nd.get("threshold_on_base", 2.5e-6)),
                    threshold_off_base=float(nd.get("threshold_off_base", 2.125e-6)),
                    static_var_threshold=float(nd.get("static_var_threshold", 1.5e-5)),
                    eig_floor=[float(x) for x in nd.get("eig_floor", [2.5e-5, 1.0e-5, 5.0e-6])],
                    gain_g=float(nd.get("gain_g", 1.0)),
                    eig1_ref_1p=float(nd.get("eig1_ref_1p", 1.8e-3)),
                    eig2_ref_1p=float(nd.get("eig2_ref_1p", 2.6e-4)),
                    energy_ref_1p=float(nd.get("energy_ref_1p", 2.6)),
                    stage2_calibrated=bool(nd.get("stage2_calibrated", False)),
                    quality=nd.get("quality", "OPTIMAL"),
                    measured_pps=float(nd.get("measured_pps", 50.0)),
                    sample_count=int(nd.get("sample_count", 100)),
                    calibrated=bool(nd.get("calibrated", True)),
                    calibrated_at=float(nd.get("calibrated_at", time.time())),
                )
            if self.profiles:
                self.is_calibrated = bool(data.get("is_calibrated", True))
                self.is_stage2_calibrated = bool(data.get("is_stage2_calibrated", False))
                self.overall_quality = data.get("overall_quality", "OPTIMAL")
                self.noise_diagnosis = data.get("noise_diagnosis", "Perfil calibrado carregado.")
                saved_str = data.get("saved_at_str", "")
                if saved_str:
                    self.last_calibrated_at_str = saved_str.split(" ")[-1]
                return True
            return False
        except Exception as exc:
            logger.debug(f"Falha ao carregar calibration_profile.json: {exc}")
            return False

    # ==========================================================================
    # Builders de Status Enxuto e Telemetria Diagnostics
    # ==========================================================================

    def build_node_summary(self, node_id: str, name: str, x: float, y: float) -> NodeSummaryStatus:
        rt = self.runtime_states.get(node_id, NodeRuntimeState(node_id=node_id))
        return NodeSummaryStatus(
            id=node_id,
            name=name,
            online=rt.online,
            state=rt.state if rt.online else "EMPTY",
            intensity=round(rt.intensity, 2),
            x=x,
            y=y,
        )

    def build_node_diagnostic(self, node_id: str, name: str, now: Optional[float] = None) -> NodeDiagnosticDetail:
        if now is None:
            now = time.time()
        rt = self.runtime_states.get(node_id, NodeRuntimeState(node_id=node_id))
        prof = self.profiles.get(node_id, NodeCalibrationProfile(node_id=node_id))
        buf = self.ring_buffers.get(node_id)
        latest_amp_norm = (
            [round(float(v), 4) for v in buf[-1].amp_norm.tolist()]
            if (buf and len(buf) > 0)
            else [round(1.0 / np.sqrt(self.num_subcarriers), 4)] * self.num_subcarriers
        )
        hold_rem = max(0.0, rt.hold_until - now) if rt.state == "PRESENCE_STATIC" else 0.0

        return NodeDiagnosticDetail(
            id=node_id,
            name=name,
            online=rt.online,
            state=rt.state if rt.online else "EMPTY",
            pps=rt.pps,
            rssi=rt.rssi,
            noise_floor=rt.noise_floor,
            channel=rt.channel,
            turbulence=round(rt.turbulence_raw, 6),
            turbulence_filtered=round(rt.turbulence_filt, 6),
            mvs=round(rt.mvs_current, 8),
            z_score=round(rt.z_score, 2),
            gain_g=round(prof.gain_g, 3),
            calibrated_energy=round(rt.calibrated_energy, 3),
            threshold_on=round(rt.threshold_on, 8),
            threshold_off=round(rt.threshold_off, 8),
            intensity=round(rt.intensity, 2),
            hold_remaining_sec=round(hold_rem, 1),
            quality=prof.quality if prof.calibrated else "UNCALIBRATED",
            stage2_calibrated=prof.stage2_calibrated,
            eigenvalues=[round(float(v), 7) for v in rt.eigenvalues],
            nbvi_indices=prof.nbvi_indices,
            nbvi_scores=prof.nbvi_scores,
            amplitudes_norm=latest_amp_norm,
        )

    def clear_inactive_nodes(self, active_node_ids: List[str]) -> None:
        keep = set(active_node_ids)
        for nid in list(self.ring_buffers.keys()):
            if nid not in keep:
                self.ring_buffers[nid].clear()
                if nid in self.runtime_states:
                    self.runtime_states[nid].online = False
                    self.runtime_states[nid].state = "EMPTY"
                    self.runtime_states[nid].mvs_current = 0.0
                    self.runtime_states[nid].calibrated_energy = 0.0
                    self.runtime_states[nid].intensity = 0.0
