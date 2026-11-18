import os
from pathlib import Path
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env caso exista
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


class Settings:
    """Configurações centralizadas do sistema de monitoramento espacial Wi-Fi CSI v3.0 (Alta Precisão)."""

    PROJECT_NAME: str = "Wi-Fi CSI Spatial Radar, PCA Multi-Target & 3D Volumetric Sensing"
    VERSION: str = "3.0.0"

    # Rede e Portas
    HOST: str = os.getenv("HOST", "0.0.0.0")
    HTTP_PORT: int = int(os.getenv("HTTP_PORT", "8000"))
    UDP_HOST: str = os.getenv("UDP_HOST", "0.0.0.0")
    UDP_PORT: int = int(os.getenv("UDP_PORT", "5555"))

    # Taxas de Atualização e Janelas DSP (Reamostragem Uniforme 2.5s @ 20 Hz = 50 amostras)
    SIMULATOR_HZ: float = float(os.getenv("SIMULATOR_HZ", "20.0"))
    WS_BROADCAST_HZ: float = float(os.getenv("WS_BROADCAST_HZ", "10.0"))
    NUM_SUBCARRIERS: int = 52
    TOP_K_SUBCARRIERS: int = 18
    BUFFER_WINDOW_SEC: float = 2.5
    RESAMPLE_HZ: float = 20.0
    RESAMPLE_SAMPLES: int = 50    # 2.5s * 20 Hz = 50 amostras uniformes
    BUFFER_MAX_PACKETS: int = 85  # Comporta até ~30 Hz durante 2.5s
    GRID_RESOLUTION: int = 20     # Matriz espacial 20x20

    # Calibração Científica em 3 Estágios, Persistência em JSON e Histerese
    CALIBRATION_PROFILE_PATH: Path = BASE_DIR / "backend" / "calibration_profile.json"
    HARD_ZERO_Z_GATE: float = 3.8
    DEFAULT_CALIBRATION_SECONDS: float = float(os.getenv("CALIBRATION_SECONDS", "15.0"))
    DEFAULT_SENSITIVITY: str = os.getenv("COUNTING_SENSITIVITY", "normal")  # "low" | "normal" | "high"
    COUNT_HYSTERESIS_SEC: float = 2.0

    # Estado Inicial e Timeout de Sensores (Padrão no Modo Simulação com 2 Pessoas)
    INITIAL_MODE: str = os.getenv("INITIAL_MODE", "simulation")  # "simulation" | "hardware"
    INITIAL_SIM_PEOPLE: int = int(os.getenv("INITIAL_SIM_PEOPLE", "2"))
    INITIAL_SIM_NODES: int = int(os.getenv("INITIAL_SIM_NODES", "3"))
    NODE_ONLINE_TIMEOUT_SEC: float = 3.0

    # LLM Local via Ollama (100% Offline)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    OLLAMA_TIMEOUT_SEC: float = float(os.getenv("OLLAMA_TIMEOUT_SEC", "30.0"))
    RECOMMENDED_OLLAMA_MODELS: list = [
        "qwen2.5:7b",
        "qwen3:8b",
        "qwen3:4b",
        "gemma3:4b",
        "llama3.2:3b",
    ]

    EVENT_AGGREGATION_INTERVAL_SEC: float = 5.0
    MAX_EVENT_HISTORY: int = 200

    # Geometria Base da Área de Abrangência (Auto-Mapeada via Log-Distance Path Loss)
    DEFAULT_ROOM_WIDTH_M: float = 5.0
    DEFAULT_ROOM_LENGTH_M: float = 4.5
    LATERAL_SENSITIVITY_RADIUS_M: float = 1.8
    DEFAULT_TX_NODE: dict = {
        "id": "tx_router",
        "name": "Roteador 2.4GHz (TX)",
        "x": 0.50,
        "y": 0.12,
    }
    DEFAULT_RX_NODES: list = [
        {"id": "esp_1", "name": "ESP32 #1 (Ponta Esq.)", "x": 0.15, "y": 0.84},
        {"id": "esp_2", "name": "ESP32 #2 (Ponta Dir.)", "x": 0.85, "y": 0.84},
        {"id": "esp_3", "name": "ESP32 #3 (Centro)", "x": 0.50, "y": 0.88},
    ]


settings = Settings()
