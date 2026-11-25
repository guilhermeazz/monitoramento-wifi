import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"


class Settings:
    """Configurações centralizadas do sistema Monitoramento Espacial Wi-Fi CSI."""

    PROJECT_NAME: str = "Monitoramento Espacial Wi-Fi CSI"
    VERSION: str = "4.2.0"

    # Rede e Portas
    HOST: str = os.getenv("HOST", "0.0.0.0")
    HTTP_PORT: int = int(os.getenv("HTTP_PORT", "8000"))
    UDP_HOST: str = os.getenv("UDP_HOST", "0.0.0.0")
    UDP_PORT: int = int(os.getenv("UDP_PORT", "5555"))

    # Taxas de Atualização e Pipeline DSP
    SIMULATOR_HZ: float = float(os.getenv("SIMULATOR_HZ", "50.0"))
    DSP_TICK_HZ: float = float(os.getenv("DSP_TICK_HZ", "20.0"))
    WS_BROADCAST_HZ: float = float(os.getenv("WS_BROADCAST_HZ", "10.0"))

    # Parâmetros Físicos OFDM HT20 (Cisco Linksys E1200 + ESP32-S3)
    RAW_LLTF_SUBCARRIERS: int = 64
    NUM_SUBCARRIERS: int = 52
    NBVI_TOP_K: int = 12
    STATIC_TOP_K: int = 4
    RING_BUFFER_MAX_PACKETS: int = 300  # ~6s a 50 Hz

    # Filtros, MVS, Subespaço Espectral (1.0s) e EMA Lento (120s)
    HAMPEL_WINDOW: int = 7
    HAMPEL_SIGMA: float = 3.5
    BUTTER_CUTOFF_HZ: float = 10.0
    MVS_WINDOW_SEC: float = 0.50
    COV_WINDOW_SEC: float = 1.00
    HEADCOUNT_HYSTERESIS_SEC: float = 2.00
    EMPTY_EMA_MIN_IDLE_SEC: float = 120.0
    DEBOUNCE_CONSECUTIVE_TICKS: int = 3
    DEFAULT_HOLD_TIME_SEC: float = float(os.getenv("HOLD_TIME_SEC", "15.0"))
    DEFAULT_SENSITIVITY_MULT: float = float(os.getenv("SENSITIVITY_MULT", "1.0"))

    # Grid RTI Métrico e Multi-Object Tracking (MOT)
    RTI_CELL_SIZE_M: float = 0.25
    NMS_MIN_HUMAN_RADIUS_M: float = 1.10
    KALMAN_MAX_SPEED_MPS: float = 2.00
    TRACK_COAST_SEC: float = 2.20

    # Calibração Guiada em 2 Etapas e Geometria Persistida
    CALIBRATION_PROFILE_PATH: Path = BACKEND_DIR / "calibration_profile.json"
    ROOM_CONFIG_PATH: Path = BACKEND_DIR / "room_config.json"
    DEFAULT_PREP_SECONDS: float = 5.0
    DEFAULT_STAGE1_CAPTURE_SECONDS: float = 15.0
    DEFAULT_STAGE2_CAPTURE_SECONDS: float = 20.0
    DEFAULT_CAPTURE_SECONDS: float = 15.0

    # Fonte de Dados Inicial e Timeout de Sensores (Modo Operacional Único Contínuo)
    INITIAL_MODE: str = os.getenv("INITIAL_MODE", "simulation")  # "simulation" | "hardware"
    INITIAL_SIM_NODES: int = int(os.getenv("INITIAL_SIM_NODES", "3"))
    INITIAL_SIM_PEOPLE: int = int(os.getenv("INITIAL_SIM_PEOPLE", "1"))
    NODE_ONLINE_TIMEOUT_SEC: float = 3.0
    GRID_RESOLUTION: int = 20

    # Eventos e LLM Assíncrona (Ollama Local / OpenAI-compatible / Fallback)
    EVENT_DEBOUNCE_SEC: float = 15.0
    MAX_EVENT_HISTORY: int = 100
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    OLLAMA_TIMEOUT_SEC: float = float(os.getenv("OLLAMA_TIMEOUT_SEC", "12.0"))

    # Geometria Física Padrão (Distância recomendada TX-RX: 3m a 6m)
    DEFAULT_ROOM_WIDTH_M: float = 5.0
    DEFAULT_ROOM_LENGTH_M: float = 4.5
    DEFAULT_TX_NODE: dict = {
        "id": "tx_router",
        "name": "Cisco Linksys E1200 (TX)",
        "mac": "20:aa:4b:4f:67:1d",
        "x": 0.50,
        "y": 0.10,
    }
    DEFAULT_RX_NODES: list = [
        {"id": "esp_1", "name": "ESP32-S3 #1", "mac": "c0:56:27:e1:20:01", "x": 0.14, "y": 0.84},
        {"id": "esp_2", "name": "ESP32-S3 #2", "mac": "c0:56:27:e1:20:02", "x": 0.86, "y": 0.84},
        {"id": "esp_3", "name": "ESP32-S3 #3", "mac": "c0:56:27:e1:20:03", "x": 0.50, "y": 0.88},
    ]


settings = Settings()
