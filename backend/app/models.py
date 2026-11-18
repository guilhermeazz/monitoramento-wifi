import time
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


TopologyModeCode = Literal[
    "NO_SENSORS",
    "SINGLE_LINK_CORRIDOR",
    "DUAL_LINK_SECTOR",
    "FULL_MESH_2D",
]

CalibrationMode = Literal[
    "instant_zero",
    "stage1_silence",
    "stage2_1person",
    "stage3_2persons",
    "crowd_compensated",
    "empty_room",
    "person_ref",
    "nominal",
]
CalibrationStateCode = Literal["idle", "countdown", "calibrating", "completed"]
SensitivityLevel = Literal["low", "normal", "high"]


class NodePosition(BaseModel):
    """Posição normalizada (0.0 a 1.0) de um nó de RF (TX ou RX) na área de abrangência."""

    id: str = Field(..., description="Identificador único do nó (ex: tx_router, esp_1)")
    name: str = Field(..., description="Nome amigável exibido na interface")
    x: float = Field(..., ge=0.0, le=1.0, description="Coordenada X normalizada [0.0, 1.0]")
    y: float = Field(..., ge=0.0, le=1.0, description="Coordenada Y normalizada [0.0, 1.0]")


class RoomConfig(BaseModel):
    """Configuração geométrica da área de abrangência (auto-mapeada via RSSI/CSI)."""

    width_m: float = Field(default=5.0, gt=1.0, le=50.0, description="Largura efetiva da área em metros (Eixo X)")
    length_m: float = Field(default=4.5, gt=1.0, le=50.0, description="Comprimento efetivo da área em metros (Eixo Y)")
    tx: NodePosition = Field(..., description="Posição do Roteador Wi-Fi Transmissor (TX)")
    rx_nodes: List[NodePosition] = Field(..., min_length=1, description="Lista de nós receptores ESP32 (RX)")


class CoverageZoneInfo(BaseModel):
    """Métricas de auto-mapeamento da Zona de Abrangência Ativa (Log-Distance Path Loss + Elipse de Fresnel)."""

    active_area_m2: float = Field(default=13.3, description="Área efetiva monitorada em metros quadrados")
    avg_tx_rx_distance_m: float = Field(default=3.0, description="Distância média estimada Roteador <-> ESP32 (m)")
    lateral_radius_m: float = Field(default=1.8, description="Raio lateral de sensibilidade ao redor do feixe (m)")
    auto_mapped: bool = Field(default=True, description="Indica posicionamento automático via atenuação RF")
    node_distances_m: Dict[str, float] = Field(default_factory=dict, description="Distância TX-RX estimada por nó (m)")
    description: str = Field(
        default="Área de Abrangência Ativa: ~13.3 m² (Distância TX-RX: 3.0 m)",
        description="Resumo textual da área de abrangência ativa",
    )


class CSIPacket(BaseModel):
    """Pacote bruto de Channel State Information (CSI) recebido via UDP ou Simulador."""

    node_id: str = Field(..., description="Identificador do ESP32 (ex: esp_1)")
    timestamp: float = Field(default_factory=time.time, description="Timestamp UNIX em segundos")
    rssi: int = Field(default=-50, description="Indicador de potência do sinal recebido (dBm)")
    csi_raw: List[int] = Field(..., description="Vetor intercalado de pares [Imag_0, Real_0, Imag_1, Real_1, ...]")

    @model_validator(mode="before")
    @classmethod
    def normalize_csi_fields(cls, values: dict) -> dict:
        if isinstance(values, dict):
            if "csi_raw" not in values and "csi" in values:
                values["csi_raw"] = values.pop("csi")
            if "timestamp" not in values or values["timestamp"] is None:
                values["timestamp"] = time.time()
        return values


class NodeStatus(BaseModel):
    """Telemetria DSP avançada (Pearson Correlation, Z-Score Noise Gate, PCA/SVD) de cada receptor ESP32."""

    node_id: str
    name: str
    x: float
    y: float
    rssi: int = -90
    rssi_avg: float = -90.0
    estimated_distance_m: float = 3.0
    packets_per_sec: float = 0.0
    current_energy: float = 0.0
    effective_noise_floor: float = 0.04
    net_motion_energy: float = 0.0
    z_score: float = 0.0
    pearson_r: float = 1.0
    motion_index: float = 0.0
    motion_threshold: float = 0.04
    is_triggered: bool = False
    ttl_remaining_sec: float = 0.0
    motion_score: float = 0.0
    presence_score: float = 0.0
    perturbation_score: float = 0.0
    walking_energy: float = 0.0
    vital_energy: float = 0.0
    shadow_attenuation: float = 0.0
    pca_s1: float = 0.0
    pca_s2: float = 0.0
    pca_s3: float = 0.0
    pca_s2_s1_ratio: float = 0.0
    unit_person_energy: float = 1.0
    subcarrier_diversity: float = 0.0
    energy_ratio: float = 0.0
    estimated_people_on_link: int = 0
    longitudinal_focus: float = 0.5
    top_k_indices: List[int] = Field(default_factory=list)
    online: bool = False
    last_seen: float = 0.0
    amplitudes_preview: List[float] = Field(default_factory=list)
    phases_preview: List[float] = Field(default_factory=list)
    csi_ratio_preview: List[float] = Field(default_factory=list)


class TopologyInfo(BaseModel):
    """Diagnóstico dinâmico da topologia de sensores ativos e capacidades físicas habilitadas."""

    active_nodes: int = Field(..., ge=0, description="Quantidade de módulos ESP32 online nos últimos 3.0s")
    mode_code: TopologyModeCode = Field(..., description="Código da máquina de estados de topologia")
    mode_title: str = Field(..., description="Título amigável do modo de operação atual")
    precision_level: str = Field(..., description="Nível de precisão espacial física do setup")
    capabilities_enabled: List[str] = Field(default_factory=list, description="Lista do que este setup faz agora")
    limitations: List[str] = Field(default_factory=list, description="Limitações físicas atuais da topologia")
    setup_recommendation: str = Field(..., description="Recomendação prática de posicionamento dos sensores")


class PersonCoordinate(BaseModel):
    """Pessoa rastreada no plano 2D/3D com estado cinemático e retenção de presença parada."""

    id: str = Field(..., description="ID de rastreamento (ex: P1, P2)")
    x: float = Field(..., ge=0.0, le=1.0, description="Posição X normalizada [0.0, 1.0]")
    y: float = Field(..., ge=0.0, le=1.0, description="Coordenada Y normalizada [0.0, 1.0]")
    x_m: float = Field(..., description="Posição X em metros")
    y_m: float = Field(..., description="Posição Y em metros")
    vx: float = Field(default=0.0, description="Velocidade estimada no eixo X")
    vy: float = Field(default=0.0, description="Velocidade estimada no eixo Y")
    zone: str = Field(..., description="Quadrante, corredor ou zona atual")
    activity: Literal["moving", "stationary"] = Field(..., description="Estado de atividade dinâmico")
    posture_state: Literal["standing_moving", "stationary"] = Field(
        default="standing_moving",
        description="Estado volumétrico para renderização 3D",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Índice de confiança da detecção [0.0, 1.0]")


class AggregatedEvent(BaseModel):
    """Evento semântico temporal gerado pelo EventAggregator para consumo humano e da LLM."""

    timestamp: str = Field(..., description="Horário formatado HH:MM:SS")
    epoch: float = Field(default_factory=time.time, description="Timestamp UNIX")
    people_count: int = Field(..., ge=0)
    active_zones: List[str] = Field(default_factory=list)
    activity_level: Literal["none", "low", "moderate", "high"] = "none"
    summary: str = Field(..., description="Resumo em linguagem natural do evento")


class GlobalDecision(BaseModel):
    """Métricas da fusão matemática multi-sensor global e justificativa científica da decisão."""

    decision_people_count: int = 0
    decision_reason: str = "Zona de Abrangência Livre"
    global_svd_s1: float = 0.0
    global_svd_s2: float = 0.0
    global_svd_ratio: float = 0.0
    global_energy_ratio: float = 0.0
    nodes_triggered_count: int = 0
    dominant_node_id: Optional[str] = None


class LiveStatePayload(BaseModel):
    """Payload consolidado transmitido via WebSocket (/ws/live) a 10 Hz."""

    mode: Literal["simulation", "hardware"]
    simulated_people: int
    simulated_nodes: int = 1
    sensitivity: SensitivityLevel = "normal"
    calibrated: bool
    step1_calibrated: bool = False
    step2_calibrated: bool = False
    calibrating: bool = False
    calibration_state: CalibrationStateCode = "idle"
    calibration_mode: CalibrationMode = "nominal"
    countdown_remaining: float = 0.0
    calibration_progress: float = 0.0
    last_calibrated_at: Optional[str] = None
    overall_confidence: float = 0.92
    agitation_level: Literal["none", "low", "moderate", "high"] = "none"
    agitation_score: float = 0.0
    coverage_info: CoverageZoneInfo
    topology_info: TopologyInfo
    room_config: RoomConfig
    nodes_status: Dict[str, NodeStatus]
    people_count: int
    people_coordinates: List[PersonCoordinate]
    heatmap_grid: List[List[float]]
    active_zone: str
    total_packets_per_sec: float
    recent_events: List[AggregatedEvent]
    global_decision: GlobalDecision = Field(default_factory=GlobalDecision)


class ModeChangeRequest(BaseModel):
    """Requisição para alternar entre modo Simulador, Hardware Real e Sensibilidade de Contagem."""

    mode: Optional[Literal["simulation", "hardware"]] = None
    simulated_people: Optional[int] = Field(default=None, ge=0, le=4)
    simulated_nodes: Optional[int] = Field(default=None, ge=0, le=3)
    sensitivity: Optional[SensitivityLevel] = None


class CalibrateRequest(BaseModel):
    """Parâmetros para Calibração Científica em 3 Estágios (Silêncio, 1 Pessoa, 2+ Pessoas)."""

    mode: Literal[
        "instant_zero",
        "stage1_silence",
        "stage2_1person",
        "stage3_2persons",
        "crowd_compensated",
        "empty_room",
        "person_ref",
    ] = Field(
        default="stage1_silence",
        description="Estágio da calibração: 'instant_zero' (3s imediato), 'stage1_silence' (15s vazio), 'stage2_1person' (20s 1 pessoa), 'stage3_2persons' (15s 2 pessoas)",
    )
    delay_seconds: float = Field(
        default=0.0,
        ge=0.0,
        le=30.0,
        description="Segundos de contagem regressiva (COUNTDOWN) antes de iniciar a coleta CSI",
    )
    duration_seconds: float = Field(
        default=15.0,
        ge=2.0,
        le=60.0,
        description="Duração da coleta de pacotes CSI em segundos (CALIBRATING)",
    )

    @model_validator(mode="before")
    @classmethod
    def support_legacy_duration_field(cls, values: dict) -> dict:
        if isinstance(values, dict):
            if "duration_seconds" not in values and "duration_sec" in values:
                values["duration_seconds"] = values.pop("duration_sec")
        return values


class LLMReportRequest(BaseModel):
    """Requisição opcionalmente customizada para geração de relatório analítico via Ollama Local."""

    focus_area: Optional[str] = Field(default=None, description="Foco específico opcional para o relatório")
    model: Optional[str] = Field(default=None, description="Modelo Ollama opcional (ex: qwen2.5:7b, qwen3:8b, gemma3:4b)")


class LLMAskRequest(BaseModel):
    """Pergunta em linguagem natural enviada pelo operador sobre o histórico da sala."""

    question: str = Field(..., min_length=2, max_length=600)
    model: Optional[str] = Field(default=None, description="Modelo Ollama opcional")


class LLMStatusResponse(BaseModel):
    """Status de conectividade do servidor Ollama Local (http://localhost:11434) e modelos baixados."""

    online: bool
    base_url: str
    active_model: str
    installed_models: List[str] = Field(default_factory=list)
    available_models: List[str] = Field(default_factory=list)
    badge_text: str


class LLMResponse(BaseModel):
    """Resposta estruturada do serviço de inteligência analítica (Ollama Local / Fallback Estatístico)."""

    status: str = "ok"
    provider: str = Field(..., description="'ollama-local' ou 'local-statistical-fallback'")
    model: str
    content: str = Field(..., description="Conteúdo analítico em Markdown")
    generated_at: str
    events_analyzed: int
