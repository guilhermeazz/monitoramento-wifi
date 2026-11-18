import asyncio
from contextlib import asynccontextmanager
import logging
import time
from typing import Dict, List, Literal, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_DIR, settings
from .dsp_engine import DSPEngine, LinkDSPMetrics
from .event_aggregator import EventAggregator
from .llm_service import LLMService
from .models import (
    CalibrateRequest,
    CSIPacket,
    LiveStatePayload,
    LLMAskRequest,
    LLMReportRequest,
    LLMResponse,
    LLMStatusResponse,
    ModeChangeRequest,
    NodePosition,
    NodeStatus,
    RoomConfig,
)
from .simulator import CSISimulator
from .spatial_tracker import SpatialTracker
from .udp_receiver import UDPReceiverServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wifi_spatial.main")


class SpatialMonitoringController:
    """Orquestrador central de estado, recepção UDP, Simulador, Motor DSP PCA/SVD, Tracker 2D/3D e LLM."""

    def __init__(self) -> None:
        self.mode: Literal["simulation", "hardware"] = (
            "simulation" if settings.INITIAL_MODE == "simulation" else "hardware"
        )
        self.room_config = RoomConfig(
            width_m=settings.DEFAULT_ROOM_WIDTH_M,
            length_m=settings.DEFAULT_ROOM_LENGTH_M,
            tx=NodePosition(**settings.DEFAULT_TX_NODE),
            rx_nodes=[NodePosition(**node) for node in settings.DEFAULT_RX_NODES],
        )

        self.dsp_engine = DSPEngine(num_subcarriers=settings.NUM_SUBCARRIERS)
        self.spatial_tracker = SpatialTracker(grid_size=settings.GRID_RESOLUTION)
        self.event_aggregator = EventAggregator(max_history=settings.MAX_EVENT_HISTORY)
        self.llm_service = LLMService()

        self.udp_server = UDPReceiverServer(on_packet_callback=self.on_csi_packet)
        self.simulator = CSISimulator(
            get_room_config=lambda: self.room_config,
            is_calibrating_fn=lambda: self.dsp_engine.calibration_state == "calibrating",
            on_packet_callback=self.on_csi_packet,
            get_calibration_mode_fn=lambda: self.dsp_engine.active_calibration_mode,
        )

        self.ws_clients: Set[WebSocket] = set()
        self._broadcast_task: asyncio.Task | None = None
        self._running: bool = False
        self._latest_state: LiveStatePayload | None = None
        self._prev_cal_state: str = "idle"

    def on_csi_packet(self, packet: CSIPacket, source: str) -> None:
        # Se chegar um pacote UDP real de um ESP32 físico, muda automaticamente para o modo Hardware Real!
        if source == "hardware" and self.mode != "hardware":
            logger.info(f"Pacote UDP real detectado de [{packet.node_id}]! Alternando automaticamente para modo HARDWARE.")
            self.mode = "hardware"
            self.simulator.enabled = False
            self.dsp_engine.deactivate_nodes_except([])
            self.spatial_tracker.tracks.clear()
            self.spatial_tracker._hysteresis_count = 0

        if self.mode == "simulation" and source != "simulation":
            return
        if self.mode == "hardware" and source != "hardware":
            return

        self.dsp_engine.ingest_packet(packet)

    def build_live_state(self) -> LiveStatePayload:
        """Computa o estado consolidado de 100ms (10 Hz) para streaming WebSocket (/ e /3d) e REST."""
        now = time.time()
        cal_state, countdown_rem, cal_progress = self.dsp_engine.tick_calibration_state(now)
        is_calibrating = cal_state == "calibrating"
        in_cal_flow = cal_state in ("countdown", "calibrating")

        if self._prev_cal_state == "calibrating" and cal_state == "completed":
            active_m = self.dsp_engine.active_calibration_mode
            if active_m == "person_ref":
                mode_pt = "Etapa 2: Referência de 1 Pessoa (PCA s1_ref / s2_ref / unit_person_energy)"
            elif active_m == "crowd_compensated":
                mode_pt = "Etapa 1: Modo Sala com Plateia (Mediana + Percentil 15)"
                self.spatial_tracker.tracks.clear()
                self.spatial_tracker._hysteresis_count = 0
            else:
                mode_pt = "Etapa 1: Sala Vazia (Cancelamento de Objetos Estáticos)"
                self.spatial_tracker.tracks.clear()
                self.spatial_tracker._hysteresis_count = 0
            self.event_aggregator.record_system_event(
                f"Calibração concluída ({mode_pt}) às {self.dsp_engine.last_calibrated_at_str}.",
                people_count=0,
            )
        self._prev_cal_state = cal_state

        # 1. Extrai métricas DSP avançadas (Jitter Resampling + CSI Ratio + Top-18 + PCA SVD)
        dsp_metrics_map: Dict[str, LinkDSPMetrics] = {}
        total_pps = 0.0
        for rx in self.room_config.rx_nodes:
            m = self.dsp_engine.compute_node_metrics(rx.id, now=now)
            dsp_metrics_map[rx.id] = m
            total_pps += m.packets_per_sec

        # 2. Atualiza Auto-Mapeamento da Área de Abrangência e coordenadas 2D com fusão SVD global
        heatmap_grid, people_coords, active_zone, topology_info, coverage_info, global_decision = self.spatial_tracker.update(
            room_config=self.room_config,
            node_metrics=dsp_metrics_map,
            is_calibrating=in_cal_flow,
            now=now,
        )

        # 3. Constrói status atualizado dos nós
        nodes_status_map: Dict[str, NodeStatus] = {}
        online_motions: List[float] = []
        online_perts: List[float] = []

        for rx in self.room_config.rx_nodes:
            m = dsp_metrics_map[rx.id]
            if m.online:
                online_motions.append(m.motion_score)
                online_perts.append(m.perturbation_score)
            nodes_status_map[rx.id] = NodeStatus(
                node_id=rx.id,
                name=rx.name,
                x=rx.x,
                y=rx.y,
                rssi=m.rssi,
                rssi_avg=m.rssi_avg,
                estimated_distance_m=m.estimated_distance_m,
                packets_per_sec=m.packets_per_sec,
                current_energy=m.current_energy,
                effective_noise_floor=m.effective_noise_floor,
                net_motion_energy=m.net_motion_energy,
                z_score=m.z_score,
                pearson_r=m.pearson_r,
                motion_index=m.motion_index,
                motion_threshold=m.motion_threshold,
                is_triggered=m.is_triggered,
                ttl_remaining_sec=m.ttl_remaining_sec,
                motion_score=m.motion_score,
                presence_score=m.presence_score,
                perturbation_score=m.perturbation_score,
                walking_energy=m.walking_energy,
                vital_energy=m.vital_energy,
                shadow_attenuation=m.shadow_attenuation,
                pca_s1=m.pca_s1,
                pca_s2=m.pca_s2,
                pca_s3=m.pca_s3,
                pca_s2_s1_ratio=m.pca_s2_s1_ratio,
                unit_person_energy=m.unit_person_energy,
                subcarrier_diversity=m.subcarrier_diversity,
                energy_ratio=m.energy_ratio,
                estimated_people_on_link=m.estimated_people_on_link,
                longitudinal_focus=m.longitudinal_focus,
                top_k_indices=m.top_k_indices,
                online=m.online,
                last_seen=m.last_seen,
                amplitudes_preview=m.amplitudes_preview,
                phases_preview=m.phases_preview,
                csi_ratio_preview=m.csi_ratio_preview,
            )

        max_motion = max(online_motions, default=0.0)
        max_pert = max(online_perts, default=0.0)
        agitation_score = float(max(max_motion, max_pert * 0.75))
        if len(people_coords) == 0 and agitation_score < 0.14:
            agitation_level: Literal["none", "low", "moderate", "high"] = "none"
        elif agitation_score >= 0.62:
            agitation_level = "high"
        elif agitation_score >= 0.30:
            agitation_level = "moderate"
        else:
            agitation_level = "low"

        # Confiança global do sistema (elevada quando Etapa 1 e Etapa 2 estão calibradas)
        if people_coords:
            base_conf = sum(p.confidence for p in people_coords) / len(people_coords)
        else:
            base_conf = 0.92 if topology_info.active_nodes > 0 else 0.0
        cal_bonus = (0.04 if self.dsp_engine.step1_calibrated else 0.0) + (0.04 if self.dsp_engine.step2_calibrated else 0.0)
        overall_confidence = round(min(0.99, base_conf + cal_bonus), 2) if topology_info.active_nodes > 0 else 0.0

        self.event_aggregator.ingest_state(
            people=people_coords,
            nodes_status=nodes_status_map,
            is_calibrating=in_cal_flow,
        )

        effective_cal_mode = (
            self.dsp_engine.active_calibration_mode
            if in_cal_flow
            else self.dsp_engine.calibrated_mode
        )

        state = LiveStatePayload(
            mode=self.mode,
            simulated_people=self.simulator.num_people,
            simulated_nodes=self.simulator.num_nodes,
            sensitivity=self.dsp_engine.sensitivity,
            calibrated=self.dsp_engine.is_calibrated,
            step1_calibrated=self.dsp_engine.step1_calibrated,
            step2_calibrated=self.dsp_engine.step2_calibrated,
            calibrating=is_calibrating,
            calibration_state=cal_state,
            calibration_mode=effective_cal_mode,
            countdown_remaining=countdown_rem,
            calibration_progress=round(cal_progress, 2),
            last_calibrated_at=self.dsp_engine.last_calibrated_at_str,
            overall_confidence=overall_confidence,
            agitation_level=agitation_level,
            agitation_score=round(agitation_score, 2),
            coverage_info=coverage_info,
            topology_info=topology_info,
            room_config=self.room_config,
            nodes_status=nodes_status_map,
            people_count=len(people_coords),
            people_coordinates=people_coords,
            heatmap_grid=heatmap_grid,
            active_zone=active_zone,
            total_packets_per_sec=round(total_pps, 1),
            recent_events=self.event_aggregator.get_recent_events(limit=25),
            global_decision=global_decision,
        )
        self._latest_state = state
        return state

    async def start(self) -> None:
        self._running = True
        await self.udp_server.start(host=settings.UDP_HOST, port=settings.UDP_PORT)
        await self.simulator.start()
        self.event_aggregator.record_system_event(
            f"Sistema v3.0 iniciado em modo {self.mode.upper()} ({self.simulator.num_nodes} ESP, {self.simulator.num_people} pessoas).",
            people_count=self.simulator.num_people if self.mode == "simulation" else 0,
        )
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        self._running = False
        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        await self.simulator.stop()
        await self.udp_server.stop()

    async def _broadcast_loop(self) -> None:
        interval = 1.0 / settings.WS_BROADCAST_HZ  # 0.1s (10 Hz)
        while self._running:
            try:
                state = self.build_live_state()
                if self.ws_clients:
                    payload_json = state.model_dump(mode="json")
                    dead_clients: List[WebSocket] = []
                    for ws in list(self.ws_clients):
                        try:
                            await ws.send_json(payload_json)
                        except Exception:
                            dead_clients.append(ws)
                    for ws in dead_clients:
                        self.ws_clients.discard(ws)
            except Exception as exc:
                logger.error(f"Erro no loop de broadcast 10Hz: {exc}")

            await asyncio.sleep(interval)


controller = SpatialMonitoringController()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await controller.start()
    yield
    await controller.stop()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Sistema de Monitoramento Espacial Wi-Fi CSI v3.0 (PCA/SVD, CSI Ratio, Calibração 2 Etapas, Hero Counter & Página /3d)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/state", response_model=LiveStatePayload)
async def get_current_state() -> LiveStatePayload:
    if controller._latest_state is not None:
        return controller._latest_state
    return controller.build_live_state()


@app.get("/api/config/room", response_model=RoomConfig)
async def get_room_config() -> RoomConfig:
    return controller.room_config


@app.post("/api/config/room", response_model=RoomConfig)
async def update_room_config(config: RoomConfig) -> RoomConfig:
    controller.room_config = config
    return controller.room_config


@app.post("/api/config/room/reset", response_model=RoomConfig)
async def reset_room_config() -> RoomConfig:
    controller.room_config = RoomConfig(
        width_m=settings.DEFAULT_ROOM_WIDTH_M,
        length_m=settings.DEFAULT_ROOM_LENGTH_M,
        tx=NodePosition(**settings.DEFAULT_TX_NODE),
        rx_nodes=[NodePosition(**node) for node in settings.DEFAULT_RX_NODES],
    )
    controller.event_aggregator.record_system_event(
        "Posições dos nós restauradas para o layout padrão (ESP1 Ponta Esq, ESP2 Ponta Dir, ESP3 Centro).",
        people_count=0,
    )
    return controller.room_config


@app.post("/api/mode")
async def switch_mode(req: ModeChangeRequest) -> dict:
    prev_mode = controller.mode
    if req.mode is not None:
        controller.mode = req.mode
        controller.simulator.enabled = req.mode == "simulation"

    if req.simulated_people is not None:
        controller.simulator.set_people_count(req.simulated_people)

    if req.simulated_nodes is not None:
        controller.simulator.set_nodes_count(req.simulated_nodes)

    if req.sensitivity is not None:
        controller.dsp_engine.sensitivity = req.sensitivity

    if controller.mode == "simulation":
        active_sim_ids = [rx.id for rx in controller.room_config.rx_nodes[: controller.simulator.num_nodes]]
        controller.dsp_engine.deactivate_nodes_except(active_sim_ids)
        controller.udp_server.clear_heartbeats(active_sim_ids)
    elif prev_mode != "hardware":
        controller.dsp_engine.deactivate_nodes_except([])
        controller.udp_server.clear_heartbeats([])

    mode_desc = (
        f"Simulador ({controller.simulator.num_nodes} ESP(s), {controller.simulator.num_people} pessoa(s), Sensibilidade: {controller.dsp_engine.sensitivity.upper()})"
        if controller.mode == "simulation"
        else f"Hardware Real (UDP:{controller.udp_server.bound_port}, Sensibilidade: {controller.dsp_engine.sensitivity.upper()})"
    )
    controller.event_aggregator.record_system_event(
        f"Configuração atualizada: {mode_desc}",
        people_count=controller.simulator.num_people if controller.mode == "simulation" else 0,
    )

    return {
        "status": "ok",
        "mode": controller.mode,
        "simulated_people": controller.simulator.num_people,
        "simulated_nodes": controller.simulator.num_nodes,
        "sensitivity": controller.dsp_engine.sensitivity,
        "udp_port": controller.udp_server.bound_port,
    }


@app.post("/api/calibrate/wizard")
async def trigger_calibration_wizard(req: CalibrateRequest) -> dict:
    """
    Assistente Científico de Calibração em 3 Estágios (WiCount / esp-radar):
    - 'instant_zero': 3s imediato (Zera quarto vazio na hora)
    - 'stage1_silence': 15s sala vazia (Mapeia baseline, blacklist de subportadoras e piso de ruído Z-Score)
    - 'stage2_1person': 20s 1 pessoa andando (Mapeia autovalores SVD λ1/λ2 e unit_person_energy)
    - 'stage3_2persons': 15s 2 pessoas andando (Mapeia autovalores SVD ortogonais e ratio de referência)
    """
    stage = req.mode
    if stage == "instant_zero":
        return await trigger_instant_zero()

    delay_sec = req.delay_seconds
    if delay_sec == 0.0 and stage == "stage1_silence":
        delay_sec = 10.0  # Tempo padrão para o operador sair da sala se não especificado

    duration_sec = req.duration_seconds
    if duration_sec <= 2.0:
        if stage == "stage1_silence":
            duration_sec = 15.0
        elif stage == "stage2_1person":
            duration_sec = 20.0
        elif stage == "stage3_2persons":
            duration_sec = 15.0

    controller.dsp_engine.start_calibration(
        mode=stage,
        delay_seconds=delay_sec,
        duration_seconds=duration_sec,
    )

    stage_names = {
        "stage1_silence": "Estágio 1: Silêncio da Sala Vazia (15s)",
        "stage2_1person": "Estágio 2: Referência de 1 Pessoa Caminhando (20s)",
        "stage3_2persons": "Estágio 3: Referência de 2 Pessoas Caminhando (15s)",
    }
    label = stage_names.get(stage, stage)
    controller.event_aggregator.record_system_event(
        f"Calibração Científica: {label} iniciada (Delay {delay_sec:.0f}s, Coleta {duration_sec:.0f}s).",
        people_count=1 if stage == "stage2_1person" else (2 if stage == "stage3_2persons" else 0),
    )
    return {
        "status": controller.dsp_engine.calibration_state,
        "stage": stage,
        "delay_seconds": delay_sec,
        "duration_seconds": duration_sec,
        "message": f"{label} iniciada.",
    }


@app.post("/api/calibrate/step1_baseline")
async def trigger_calibration_step1(req: CalibrateRequest | None = None) -> dict:
    """
    Etapa 1: Calibração de Sala / Cancelamento de Objetos Estáticos:
    - 'crowd_compensated': Modo Sala com Plateia (Rápido 8s - Mediana + Percentil 15).
    - 'empty_room': Modo Sala Vazia com Temporizador (Delay 10s para sair + 8s coleta).
    """
    mode: Literal["crowd_compensated", "empty_room"] = (
        "empty_room" if (req and req.mode == "empty_room") else "crowd_compensated"
    )
    delay_sec = req.delay_seconds if req is not None else (10.0 if mode == "empty_room" else 0.0)
    duration_sec = req.duration_seconds if req is not None else 8.0

    controller.dsp_engine.start_calibration(
        mode=mode,
        delay_seconds=delay_sec,
        duration_seconds=duration_sec,
    )

    label = "Etapa 1 (Modo Sala com Plateia 8s)" if mode == "crowd_compensated" else "Etapa 1 (Sala Vazia Delay 10s + 8s)"
    controller.event_aggregator.record_system_event(
        f"Calibração {label} iniciada.",
        people_count=0,
    )
    return {
        "status": controller.dsp_engine.calibration_state,
        "step": 1,
        "mode": mode,
        "delay_seconds": delay_sec,
        "duration_seconds": duration_sec,
        "message": f"Calibração {label} iniciada.",
    }


@app.post("/api/calibrate/step2_person_ref")
async def trigger_calibration_step2(req: CalibrateRequest | None = None) -> dict:
    """
    Etapa 2: Calibração de Referência de 1 Pessoa Caminhando (8s):
    Grava unit_person_energy, s1_ref e s2_ref da sala para máxima precisão multi-pessoa.
    """
    delay_sec = req.delay_seconds if req is not None else 0.0
    duration_sec = req.duration_seconds if req is not None else 8.0

    controller.dsp_engine.start_calibration(
        mode="person_ref",
        delay_seconds=delay_sec,
        duration_seconds=duration_sec,
    )

    controller.event_aggregator.record_system_event(
        f"Calibração Etapa 2 (Referência de 1 Pessoa Caminhando por {duration_sec:.0f}s) iniciada.",
        people_count=1,
    )
    return {
        "status": controller.dsp_engine.calibration_state,
        "step": 2,
        "mode": "person_ref",
        "delay_seconds": delay_sec,
        "duration_seconds": duration_sec,
        "message": "Caminhe normalmente entre o Roteador e o ESP32 por 8 segundos.",
    }


@app.post("/api/calibrate/instant_zero")
async def trigger_instant_zero() -> dict:
    """
    Botão '🧹 Forçar Zero (Aqui está Vazio Agora)':
    Usa os últimos ~3 segundos já armazenados nos buffers dos ESP32-S3, define o pico máximo
    atual + 40% de margem como o novo effective_noise_floor, limpa todos os fantasmas e
    trava o contador em 0 instantaneamente!
    """
    floors = controller.dsp_engine.force_instant_zero()
    controller.spatial_tracker.force_reset_tracks()
    controller.event_aggregator.record_system_event(
        "🧹 Forçar Zero executado: Quarto vazio calibrado instantaneamente (Pico + 40% = Novo Piso Zero).",
        people_count=0,
    )
    return {
        "status": "completed",
        "mode": "instant_zero",
        "effective_noise_floors": floors,
        "message": "Piso de ruído recalibrado instantaneamente e contador zerado!",
    }


@app.post("/api/calibrate")
async def trigger_calibration(req: CalibrateRequest | None = None) -> dict:
    """Rota unificada de Calibração (compatível com Wizard em 3 estágios e legado)."""
    mode = req.mode if req is not None else "stage1_silence"
    if mode in ("stage1_silence", "stage2_1person", "stage3_2persons", "instant_zero"):
        return await trigger_calibration_wizard(req or CalibrateRequest(mode=mode))
    if mode == "person_ref":
        return await trigger_calibration_step2(req)
    return await trigger_calibration_step1(req)


@app.get("/api/events")
async def get_aggregated_events() -> dict:
    return {
        "events": controller.event_aggregator.get_recent_events(limit=50),
        "peak_occupancy": controller.event_aggregator.peak_occupancy,
        "zone_dwell_seconds": controller.event_aggregator.zone_dwell_seconds,
    }


@app.get("/api/llm/status", response_model=LLMStatusResponse)
async def get_llm_status(model: str | None = None) -> LLMStatusResponse:
    return await controller.llm_service.get_status(preferred_model=model)


@app.post("/api/llm/report", response_model=LLMResponse)
async def generate_llm_report(req: LLMReportRequest | None = None) -> LLMResponse:
    state = controller._latest_state or controller.build_live_state()
    digest = controller.event_aggregator.get_analytical_digest(
        room_config=controller.room_config,
        current_people=state.people_coordinates,
        nodes_status=state.nodes_status,
        topology_info=state.topology_info,
        coverage_info=state.coverage_info,
        calibration_mode=state.calibration_mode,
    )
    focus = req.focus_area if req is not None else None
    model_override = req.model if req is not None else None
    return await controller.llm_service.generate_report(
        digest=digest,
        focus_area=focus,
        model_override=model_override,
    )


@app.post("/api/llm/ask", response_model=LLMResponse)
async def ask_llm_about_room(req: LLMAskRequest) -> LLMResponse:
    state = controller._latest_state or controller.build_live_state()
    digest = controller.event_aggregator.get_analytical_digest(
        room_config=controller.room_config,
        current_people=state.people_coordinates,
        nodes_status=state.nodes_status,
        topology_info=state.topology_info,
        coverage_info=state.coverage_info,
        calibration_mode=state.calibration_mode,
    )
    return await controller.llm_service.ask_question(
        question=req.question,
        digest=digest,
        model_override=req.model,
    )


@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    controller.ws_clients.add(websocket)
    try:
        initial_state = controller._latest_state or controller.build_live_state()
        await websocket.send_json(initial_state.model_dump(mode="json"))

        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        controller.ws_clients.discard(websocket)
    except Exception:
        controller.ws_clients.discard(websocket)


@app.get("/")
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "Frontend index.html ainda não encontrado."})


FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
