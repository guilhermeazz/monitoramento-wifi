import time
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

OccupancyState = Literal["EMPTY", "PRESENCE_STATIC", "MOTION"]
CalibrationState = Literal[
    "IDLE",
    "PREPARING",
    "CAPTURING",
    "STAGE1_DONE",
    "PREPARING_STAGE2",
    "CAPTURING_STAGE2",
    "COMPLETED",
]
CalibrationQuality = Literal["OPTIMAL", "ACCEPTABLE", "NOISY", "UNCALIBRATED"]
TopologyMode = Literal[
    "NO_SENSORS",
    "SINGLE_NODE_FRESNEL",
    "DUAL_NODE_SECTOR",
    "MULTI_NODE_KALMAN_2D",
]
UncertaintyMode = Literal["NONE", "FRESNEL_HALO", "SECTOR_ZONE", "POINT_2D"]


class NodePosition(BaseModel):
    """Posição normalizada [0.0, 1.0] de um nó RF (Cisco Linksys E1200 TX ou ESP32-S3 RX)."""

    id: str = Field(..., description="ID único (ex: tx_router, esp_1)")
    name: str = Field(..., description="Nome de exibição")
    mac: str = Field(default="", description="MAC Address opcional do dispositivo")
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)


class GeometryWarning(BaseModel):
    """Aviso discreto de geometria física (distância TX-RX fora da faixa ou colinearidade)."""

    code: str
    severity: Literal["info", "warning"] = "warning"
    node_id: Optional[str] = None
    message: str


class RoomConfig(BaseModel):
    """Configuração geométrica da área monitorada em metros reais."""

    width_m: float = Field(default=5.0, gt=1.0, le=30.0)
    length_m: float = Field(default=4.5, gt=1.0, le=30.0)
    tx: NodePosition
    rx_nodes: List[NodePosition] = Field(..., min_length=1)


class RoomLayoutResponse(BaseModel):
    """Resposta do endpoint de layout da sala com diagnósticos geométricos e matriz RTI."""

    room_config: RoomConfig
    grid_nx: int
    grid_ny: int
    cell_size_m: float
    link_distances_m: Dict[str, float] = Field(default_factory=dict)
    warnings: List[GeometryWarning] = Field(default_factory=list)


class CSIPacket(BaseModel):
    """Pacote CSI bruto decodificado do protocolo UDP binário (0xC510) ou Simulador."""

    node_id: str
    timestamp: float = Field(default_factory=time.time)
    timestamp_us: int = 0
    src_mac: str = "00:00:00:00:00:00"
    rssi: int = -50
    noise_floor: int = -92
    channel: int = 6
    csi_raw: List[float] = Field(..., description="Pares intercalados [Imag_0, Real_0, ...]")

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, values: dict) -> dict:
        if isinstance(values, dict):
            if "csi_raw" not in values and "csi" in values:
                values["csi_raw"] = values.pop("csi")
            if "timestamp" not in values or values["timestamp"] is None:
                values["timestamp"] = time.time()
        return values


class NodeSummaryStatus(BaseModel):
    """Status enxuto de um nó ESP32-S3 para o payload leve do WebSocket (< 1 KB)."""

    id: str
    name: str
    online: bool = False
    state: OccupancyState = "EMPTY"
    intensity: float = 0.0
    x: float = 0.5
    y: float = 0.85


class NodeDiagnosticDetail(BaseModel):
    """Telemetria detalhada de engenharia enviada apenas quando diagnostics=true."""

    id: str
    name: str
    online: bool
    state: OccupancyState
    pps: float
    rssi: int
    noise_floor: int
    channel: int
    turbulence: float
    turbulence_filtered: float
    mvs: float
    z_score: float = 0.0
    gain_g: float = 1.0
    calibrated_energy: float = 0.0
    threshold_on: float
    threshold_off: float
    intensity: float
    hold_remaining_sec: float
    quality: CalibrationQuality
    stage2_calibrated: bool = False
    eigenvalues: List[float] = Field(default_factory=list)
    nbvi_indices: List[int] = Field(default_factory=list)
    nbvi_scores: List[float] = Field(default_factory=list)
    amplitudes_norm: List[float] = Field(default_factory=list)


class SpatialTarget(BaseModel):
    """Alvo espacial rastreado por Filtro de Kalman 2D Multi-Objeto (MOT)."""

    id: str = "P1"
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    x_m: float
    y_m: float
    vx: float = 0.0
    vy: float = 0.0
    state: OccupancyState = "MOTION"
    confidence: float = Field(..., ge=0.0, le=1.0)
    zone: str = "ZONA_CENTRAL"


class CalibrationStatusPayload(BaseModel):
    """Estado em tempo real do Wizard de Calibração em 2 Etapas."""

    state: CalibrationState = "IDLE"
    stage: int = 1
    progress: float = 0.0
    remaining_sec: float = 0.0
    quality: CalibrationQuality = "UNCALIBRATED"
    calibrated: bool = False
    stage2_calibrated: bool = False
    noise_diagnosis: str = "Aguardando calibração inicial."
    last_calibrated_at: Optional[str] = None


class LLMInsightPayload(BaseModel):
    """Insight contextual assíncrono gerado pela camada analítica."""

    summary: str = "Monitoramento contínuo de ocupação e contagem ativo."
    updated_at: str = "--:--:--"
    provider: str = "statistical-engine"


class LLMReportResponse(BaseModel):
    """Resposta completa da análise executiva ou pergunta livre à LLM."""

    status: str = "ok"
    mode: Literal["quick", "full_report", "ask"] = "full_report"
    provider: str = "statistical-engine"
    model: str = "qwen2.5:7b"
    summary: str = Field(..., description="Resumo curto de 2 frases")
    markdown_report: str = Field(..., description="Análise detalhada estruturada em Markdown")
    updated_at: str
    events_analyzed: int = 0


class LLMStatusResponse(BaseModel):
    """Status do motor Ollama Local e modelos disponíveis."""

    online: bool
    base_url: str
    active_model: str
    installed_models: List[str] = Field(default_factory=list)
    available_models: List[str] = Field(default_factory=list)
    badge_text: str


class AggregatedEvent(BaseModel):
    """Evento semântico de transição de estado com debounce."""

    timestamp: str
    epoch: float = Field(default_factory=time.time)
    from_state: OccupancyState
    to_state: OccupancyState
    people_count: int = 0
    active_zone: str
    confidence: float
    mvs_peak: float
    summary: str


class DiagnosticsPayload(BaseModel):
    """Bloco pesado de telemetria de engenharia (enviado apenas se diagnostics=true)."""

    room_config: RoomConfig
    nodes_detail: Dict[str, NodeDiagnosticDetail]
    heatmap_grid: List[List[float]]
    grid_nx: int = 20
    grid_ny: int = 18
    cell_size_m: float = 0.25
    geometry_warnings: List[GeometryWarning] = Field(default_factory=list)
    spectral_modes_count: int = 0
    spatial_peaks_count: int = 0
    link_decorrelation: float = 0.0
    recent_events: List[AggregatedEvent]
    sensitivity_multiplier: float
    hold_time_sec: float
    simulated_people: int = 1
    peak_occupancy: int = 1
    zone_dwell_seconds: Dict[str, float] = Field(default_factory=dict)


class LiveStatePayload(BaseModel):
    """
    Payload WebSocket a 10 Hz (< 1 KB no modo padrão; enriquecido quando diagnostics=true).
    Opera em Modo Único Contínuo: Monitoramento de Ocupação e Contagem de Pessoas (0, 1, 2, 3+).
    """

    timestamp: float
    mode: Literal["simulation", "hardware"]
    occupancy_state: OccupancyState
    confidence: float
    target_count: int
    count_display: str = "0"
    peak_occupancy: int = 0
    topology_mode: TopologyMode
    uncertainty_mode: UncertaintyMode = "NONE"
    active_nodes_count: int
    active_zone: str
    mvs_current: float
    threshold_on: float
    threshold_off: float
    intensity: float
    calibration: CalibrationStatusPayload
    nodes: List[NodeSummaryStatus]
    targets: List[SpatialTarget]
    llm_insight: LLMInsightPayload
    diagnostics: Optional[DiagnosticsPayload] = None


class CalibrationStartRequest(BaseModel):
    """
    Parâmetros para iniciar o Wizard de Calibração em 2 Etapas:
    - stage=1: Baseline Sala Vazia (5s saída + 15s coleta) -> NBVI, mu_0, sigma_0, P95, piso de autovalores.
    - stage=2: Referência 1 Pessoa (prep opcional + 20s caminhando) -> ganho G_l e envelope de 1 pessoa.
    """

    model_config = ConfigDict(extra="ignore")

    stage: Literal[1, 2] = 1
    prep_seconds: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    capture_seconds: Optional[float] = Field(default=None, ge=1.0, le=60.0)

    @model_validator(mode="before")
    @classmethod
    def support_legacy_fields(cls, values: dict) -> dict:
        if isinstance(values, dict):
            if "prep_seconds" not in values and "delay_seconds" in values:
                values["prep_seconds"] = values.pop("delay_seconds")
            if "capture_seconds" not in values and "duration_seconds" in values:
                values["capture_seconds"] = values.pop("duration_seconds")
        return values


class SimulatorControlRequest(BaseModel):
    """
    Controle unificado da fonte de dados (Hardware Real vs Simulador),
    quantidade de pessoas simuladas (0..4), nós ativos, salto de AGC e parâmetros DSP.
    """

    model_config = ConfigDict(extra="ignore")

    mode: Optional[Literal["simulation", "hardware"]] = None
    source: Optional[Literal["simulation", "hardware"]] = None
    simulated_people: Optional[int] = Field(default=None, ge=0, le=4)
    active_nodes: Optional[int] = Field(default=None, ge=1, le=6)
    inject_agc_spike: Optional[float] = Field(
        default=None,
        description="Fator escalar de salto de AGC a injetar (ex: 2.0, 5.0, 0.3)",
    )
    sensitivity_multiplier: Optional[float] = Field(default=None, ge=0.4, le=3.0)
    hold_time_sec: Optional[float] = Field(default=None, ge=2.0, le=60.0)

    @model_validator(mode="before")
    @classmethod
    def unify_source_field(cls, values: dict) -> dict:
        if isinstance(values, dict):
            if values.get("mode") is None and values.get("source") is not None:
                values["mode"] = values["source"]
        return values


class AnalysisSummarizeRequest(BaseModel):
    """Requisição para síntese rápida, relatório executivo completo ou pergunta livre à LLM."""

    mode: Literal["quick", "full_report", "ask"] = "full_report"
    question: Optional[str] = None
    focus_area: Optional[str] = None
    model: Optional[str] = None
