import asyncio
from contextlib import asynccontextmanager
import json
import logging
import time
from typing import Dict, Literal, Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_DIR, settings
from .dsp_engine import DSPEngine
from .event_aggregator import EventAggregator
from .llm_service import LLMService
from .models import (
    AggregatedEvent,
    AnalysisSummarizeRequest,
    CalibrationStartRequest,
    CSIPacket,
    DiagnosticsPayload,
    LiveStatePayload,
    LLMStatusResponse,
    NodeDiagnosticDetail,
    NodePosition,
    RoomConfig,
    RoomLayoutResponse,
    SimulatorControlRequest,
)
from .simulator import CSISimulator
from .spatial_tracker import SpatialTracker, SpatialTrackingResult
from .udp_receiver import UDPReceiverServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wifi_spatial.main")


class SpatialMonitoringController:
    """
    Orquestrador Assíncrono em Modo Operacional Único Contínuo:
    - Monitoramento de Ocupação e Contagem de Pessoas (0, 1, 2, 3+) + Tracking 2D Multi-Objeto.
    - Fonte de dados selecionável: Hardware Real (ESP32 UDP) vs. Simulador Random Waypoint.
    - Calibração Guiada em 2 Etapas (Sala Vazia -> Referência 1 Pessoa) + Hot-Reload de Layout RTI.
    """

    def __init__(self) -> None:
        self.mode: Literal["simulation", "hardware"] = (
            "simulation" if settings.INITIAL_MODE == "simulation" else "hardware"
        )
        self.room_config = self._load_or_default_room_config()

        self.dsp_engine = DSPEngine(num_subcarriers=settings.NUM_SUBCARRIERS)
        self.spatial_tracker = SpatialTracker(
            cell_size_m=settings.RTI_CELL_SIZE_M,
            nms_radius_m=settings.NMS_MIN_HUMAN_RADIUS_M,
        )
        self.spatial_tracker.hot_reload_layout(self.room_config)

        self.llm_service = LLMService()
        self.event_aggregator = EventAggregator(
            max_history=settings.MAX_EVENT_HISTORY,
            debounce_sec=settings.EVENT_DEBOUNCE_SEC,
            on_transition_callback=self._on_semantic_transition,
        )

        self.udp_server = UDPReceiverServer(on_packet_callback=self.on_csi_packet)
        self.simulator = CSISimulator(
            get_room_config=lambda: self.room_config,
            get_cal_state_fn=lambda: self.dsp_engine.cal_state,
            is_calibrating_fn=lambda: self.dsp_engine.cal_state in ("PREPARING", "CAPTURING"),
            on_packet_callback=self.on_csi_packet,
        )

        self.ws_clients: Dict[WebSocket, bool] = {}

        self._dsp_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._latest_tracking: Optional[SpatialTrackingResult] = None
        self._prev_cal_state: str = "IDLE"

    @staticmethod
    def _default_room_config() -> RoomConfig:
        return RoomConfig(
            width_m=settings.DEFAULT_ROOM_WIDTH_M,
            length_m=settings.DEFAULT_ROOM_LENGTH_M,
            tx=NodePosition(**settings.DEFAULT_TX_NODE),
            rx_nodes=[NodePosition(**node) for node in settings.DEFAULT_RX_NODES],
        )

    def _load_or_default_room_config(self) -> RoomConfig:
        try:
            if settings.ROOM_CONFIG_PATH.exists():
                raw = json.loads(settings.ROOM_CONFIG_PATH.read_text(encoding="utf-8"))
                return RoomConfig.model_validate(raw)
        except Exception as exc:
            logger.warning(f"Falha ao ler room_config.json ({exc}); usando geometria padrão.")
        return self._default_room_config()

    def save_room_config(self) -> None:
        try:
            settings.ROOM_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            settings.ROOM_CONFIG_PATH.write_text(
                json.dumps(self.room_config.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"Erro ao salvar room_config.json: {exc}")

    def apply_room_layout_hot_reload(self, cfg: RoomConfig) -> RoomLayoutResponse:
        """Persiste o novo layout e recalcula imediatamente a matriz W do RTI em memória."""
        self.room_config = cfg
        self.save_room_config()
        warnings = self.spatial_tracker.hot_reload_layout(self.room_config)
        return RoomLayoutResponse(
            room_config=self.room_config,
            grid_nx=self.spatial_tracker.grid_nx,
            grid_ny=self.spatial_tracker.grid_ny,
            cell_size_m=self.spatial_tracker.cell_size_m,
            link_distances_m=dict(self.spatial_tracker.link_distances_m),
            warnings=warnings,
        )

    def build_layout_response(self) -> RoomLayoutResponse:
        self.spatial_tracker._ensure_layout(self.room_config)
        return RoomLayoutResponse(
            room_config=self.room_config,
            grid_nx=self.spatial_tracker.grid_nx,
            grid_ny=self.spatial_tracker.grid_ny,
            cell_size_m=self.spatial_tracker.cell_size_m,
            link_distances_m=dict(self.spatial_tracker.link_distances_m),
            warnings=list(self.spatial_tracker.geometry_warnings),
        )

    def _on_semantic_transition(self, event: AggregatedEvent) -> None:
        active_n = self._latest_tracking.active_nodes_count if self._latest_tracking else 1
        self.llm_service.schedule_background_update(
            event=event,
            recent_events=self.event_aggregator.get_recent_events(limit=8),
            active_nodes=active_n,
        )

    def on_csi_packet(self, packet: CSIPacket, source: str) -> None:
        if source == "hardware" and self.mode != "hardware":
            logger.info(f"Pacote UDP físico recebido de [{packet.node_id}]. Alternando para HARDWARE.")
            self.mode = "hardware"
            self.simulator.enabled = False
            self.dsp_engine.clear_inactive_nodes([])
            self.spatial_tracker.reset()

        if self.mode == "simulation" and source != "simulation":
            return
        if self.mode == "hardware" and source != "hardware":
            return

        # Atualiza MAC observado no layout se enviado pelo hardware
        if source == "hardware" and packet.src_mac and packet.src_mac != "00:00:00:00:00:00":
            for rx in self.room_config.rx_nodes:
                if rx.id == packet.node_id and not rx.mac:
                    rx.mac = packet.src_mac

        self.dsp_engine.enqueue_packet(packet)

    def run_dsp_tick(self, now: Optional[float] = None) -> SpatialTrackingResult:
        if now is None:
            now = time.time()

        candidate_ids = [rx.id for rx in self.room_config.rx_nodes]
        node_states = self.dsp_engine.process_all_nodes_tick(candidate_ids, now=now)

        cal_status = self.dsp_engine.get_calibration_status(now)
        is_calibrating = cal_status.state in (
            "PREPARING",
            "CAPTURING",
            "PREPARING_STAGE2",
            "CAPTURING_STAGE2",
        )

        if self._prev_cal_state == "CAPTURING" and cal_status.state == "STAGE1_DONE":
            self.spatial_tracker.reset()
            self.event_aggregator.record_system_event(
                f"Calibração Etapa 1 (Sala Vazia) concluída com qualidade {cal_status.quality}.",
                state="EMPTY",
                zone="NONE",
                people_count=0,
            )
        elif self._prev_cal_state == "CAPTURING_STAGE2" and cal_status.state == "COMPLETED":
            self.spatial_tracker.reset()
            self.event_aggregator.record_system_event(
                "Calibração Etapa 2 (Referência 1 Pessoa e Ganhos G_l) concluída.",
                state="EMPTY",
                zone="NONE",
                people_count=0,
            )
        self._prev_cal_state = cal_status.state

        tracking = self.spatial_tracker.update(
            room_config=self.room_config,
            node_states=node_states,
            is_calibrating=is_calibrating,
            now=now,
            hold_time_sec=self.dsp_engine.hold_time_sec,
        )
        self._latest_tracking = tracking

        self.event_aggregator.ingest_tracking_state(
            occupancy_state=tracking.occupancy_state,
            active_zone=tracking.active_zone,
            confidence=tracking.confidence,
            mvs_current=tracking.mvs_current,
            threshold_on=tracking.threshold_on,
            people_count=tracking.target_count,
            is_calibrating=is_calibrating,
            now=now,
        )
        return tracking

    def build_live_payload(self, include_diagnostics: bool = False) -> LiveStatePayload:
        now = time.time()
        tracking = self._latest_tracking or self.run_dsp_tick(now)
        cal_status = self.dsp_engine.get_calibration_status(now)
        peak_occ = max(
            self.spatial_tracker.peak_occupancy,
            self.event_aggregator.peak_occupancy,
            tracking.target_count,
        )

        nodes_summary = [
            self.dsp_engine.build_node_summary(rx.id, rx.name, rx.x, rx.y)
            for rx in self.room_config.rx_nodes
        ]

        diag_payload: Optional[DiagnosticsPayload] = None
        if include_diagnostics:
            nodes_detail: Dict[str, NodeDiagnosticDetail] = {
                rx.id: self.dsp_engine.build_node_diagnostic(rx.id, rx.name, now=now)
                for rx in self.room_config.rx_nodes
            }
            diag_payload = DiagnosticsPayload(
                room_config=self.room_config,
                nodes_detail=nodes_detail,
                heatmap_grid=tracking.heatmap_grid,
                grid_nx=tracking.grid_nx,
                grid_ny=tracking.grid_ny,
                cell_size_m=tracking.cell_size_m,
                geometry_warnings=list(self.spatial_tracker.geometry_warnings),
                spectral_modes_count=tracking.spectral_modes_count,
                spatial_peaks_count=tracking.spatial_peaks_count,
                link_decorrelation=tracking.link_decorrelation,
                recent_events=self.event_aggregator.get_recent_events(limit=15),
                sensitivity_multiplier=round(self.dsp_engine.sensitivity_multiplier, 2),
                hold_time_sec=round(self.dsp_engine.hold_time_sec, 1),
                simulated_people=self.simulator.simulated_people,
                peak_occupancy=peak_occ,
                zone_dwell_seconds={
                    k: round(v, 1) for k, v in self.event_aggregator.zone_dwell_seconds.items()
                },
            )

        return LiveStatePayload(
            timestamp=round(now, 3),
            mode=self.mode,
            occupancy_state=tracking.occupancy_state,
            confidence=tracking.confidence,
            target_count=tracking.target_count,
            count_display=tracking.count_display,
            peak_occupancy=peak_occ,
            topology_mode=tracking.topology_mode,
            uncertainty_mode=tracking.uncertainty_mode,
            active_nodes_count=tracking.active_nodes_count,
            active_zone=tracking.active_zone,
            mvs_current=tracking.mvs_current,
            threshold_on=tracking.threshold_on,
            threshold_off=tracking.threshold_off,
            intensity=tracking.intensity,
            calibration=cal_status,
            nodes=nodes_summary,
            targets=tracking.targets,
            llm_insight=self.llm_service.latest_insight,
            diagnostics=diag_payload,
        )

    async def start(self) -> None:
        self._running = True
        await self.udp_server.start(host=settings.UDP_HOST, port=settings.UDP_PORT)
        await self.simulator.start()
        self._dsp_task = asyncio.create_task(self._dsp_loop())
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info(
            f"Controlador Wi-Fi CSI iniciado (Fonte: {self.mode}, DSP: {settings.DSP_TICK_HZ} Hz, WS: {settings.WS_BROADCAST_HZ} Hz)"
        )

    async def stop(self) -> None:
        self._running = False
        for task in (self._dsp_task, self._broadcast_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self.simulator.stop()
        await self.udp_server.stop()

    async def _dsp_loop(self) -> None:
        interval = 1.0 / settings.DSP_TICK_HZ
        while self._running:
            try:
                self.run_dsp_tick(time.time())
            except Exception as exc:
                logger.error(f"Erro no tick DSP: {exc}")
            await asyncio.sleep(interval)

    async def _broadcast_loop(self) -> None:
        interval = 1.0 / settings.WS_BROADCAST_HZ
        while self._running:
            if self.ws_clients:
                try:
                    light_json: Optional[dict] = None
                    diag_json: Optional[dict] = None
                    stale: list[WebSocket] = []

                    for ws, wants_diag in list(self.ws_clients.items()):
                        try:
                            if wants_diag:
                                if diag_json is None:
                                    diag_json = self.build_live_payload(include_diagnostics=True).model_dump(
                                        mode="json", exclude_none=True
                                    )
                                await ws.send_json(diag_json)
                            else:
                                if light_json is None:
                                    light_json = self.build_live_payload(include_diagnostics=False).model_dump(
                                        mode="json", exclude_none=True
                                    )
                                await ws.send_json(light_json)
                        except Exception:
                            stale.append(ws)

                    for ws in stale:
                        self.ws_clients.pop(ws, None)
                except Exception as exc:
                    logger.error(f"Erro no loop WebSocket: {exc}")
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
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "version": settings.VERSION,
        "mode": controller.mode,
        "udp_port": controller.udp_server.bound_port,
        "calibrated": controller.dsp_engine.is_calibrated,
        "stage2_calibrated": controller.dsp_engine.is_stage2_calibrated,
        "quality": controller.dsp_engine.overall_quality,
    }


@app.get("/api/state", response_model=LiveStatePayload)
async def get_current_state(diagnostics: bool = Query(default=False)) -> LiveStatePayload:
    controller.run_dsp_tick(time.time())
    return controller.build_live_payload(include_diagnostics=diagnostics)


@app.get("/api/llm/status", response_model=LLMStatusResponse)
async def get_llm_status(model: Optional[str] = None) -> LLMStatusResponse:
    return await controller.llm_service.get_status(preferred_model=model)


@app.post("/api/calibration/start")
@app.post("/api/calibrate")
async def start_guided_calibration(req: Optional[CalibrationStartRequest] = None) -> dict:
    stage = req.stage if req is not None else 1
    prep_s = req.prep_seconds if req is not None else None
    cap_s = req.capture_seconds if req is not None else None

    status = controller.dsp_engine.start_calibration(
        prep_seconds=prep_s,
        capture_seconds=cap_s,
        stage=stage,
    )
    controller.spatial_tracker.reset()
    return {
        "status": "ok",
        "calibration": status.model_dump(),
        "message": (
            f"Etapa {stage} iniciada ({controller.dsp_engine.cal_prep_duration:.0f}s preparo + "
            f"{controller.dsp_engine.cal_capture_duration:.0f}s coleta)."
        ),
    }


@app.post("/api/calibration/cancel")
async def cancel_guided_calibration() -> dict:
    status = controller.dsp_engine.cancel_calibration()
    return {"status": "ok", "calibration": status.model_dump()}


@app.get("/api/calibration/status")
async def get_calibration_status() -> dict:
    return controller.dsp_engine.get_calibration_status().model_dump()


@app.post("/api/simulator/control")
@app.post("/api/config/mode")
async def control_simulator_and_dsp(req: SimulatorControlRequest) -> dict:
    if req.mode is not None:
        controller.mode = req.mode
        controller.simulator.enabled = req.mode == "simulation"
        controller.dsp_engine.clear_inactive_nodes([])
        controller.spatial_tracker.reset()

    if req.active_nodes is not None:
        controller.simulator.active_nodes = max(1, min(len(controller.room_config.rx_nodes), int(req.active_nodes)))
        active_ids = [
            rx.id for rx in controller.room_config.rx_nodes[: controller.simulator.active_nodes]
        ]
        controller.dsp_engine.clear_inactive_nodes(active_ids)

    if req.simulated_people is not None:
        controller.simulator.set_people_count(req.simulated_people)
        if controller.simulator.simulated_people == 0:
            controller.spatial_tracker.reset()
            for rt in controller.dsp_engine.runtime_states.values():
                rt.state = "EMPTY"
                rt.intensity = 0.0
                rt.calibrated_energy = 0.0
                rt.consecutive_motion_ticks = 0

    if req.inject_agc_spike is not None:
        controller.simulator.inject_agc_spike(factor=req.inject_agc_spike)

    if req.sensitivity_multiplier is not None:
        controller.dsp_engine.sensitivity_multiplier = float(req.sensitivity_multiplier)

    if req.hold_time_sec is not None:
        controller.dsp_engine.hold_time_sec = float(req.hold_time_sec)

    return {
        "status": "ok",
        "mode": controller.mode,
        "simulated_people": controller.simulator.simulated_people,
        "active_nodes": controller.simulator.active_nodes,
        "sensitivity_multiplier": controller.dsp_engine.sensitivity_multiplier,
        "hold_time_sec": controller.dsp_engine.hold_time_sec,
    }


@app.post("/api/analysis/summarize")
@app.post("/api/llm/report")
@app.post("/api/llm/ask")
async def summarize_analysis(req: Optional[AnalysisSummarizeRequest] = None) -> dict:
    now = time.time()
    tracking = controller._latest_tracking or controller.run_dsp_tick(now)
    cal_status = controller.dsp_engine.get_calibration_status(now)
    peak_occ = max(
        controller.spatial_tracker.peak_occupancy,
        controller.event_aggregator.peak_occupancy,
        tracking.target_count,
    )
    nodes_detail: Dict[str, NodeDiagnosticDetail] = {
        rx.id: controller.dsp_engine.build_node_diagnostic(rx.id, rx.name, now=now)
        for rx in controller.room_config.rx_nodes
    }

    req_mode = req.mode if req else "full_report"
    if req and req.question and req.question.strip():
        req_mode = "ask"

    report_res = await controller.llm_service.generate_analysis(
        mode=req_mode,
        occupancy_state=tracking.occupancy_state,
        people_count=tracking.target_count,
        peak_occupancy=peak_occ,
        active_zone=tracking.active_zone,
        confidence=tracking.confidence,
        active_nodes=tracking.active_nodes_count,
        topology_mode=tracking.topology_mode,
        mvs_current=tracking.mvs_current,
        threshold_on=tracking.threshold_on,
        intensity=tracking.intensity,
        calibration_quality=cal_status.quality,
        zone_dwell_seconds=controller.event_aggregator.zone_dwell_seconds,
        nodes_detail=nodes_detail,
        recent_events=controller.event_aggregator.get_recent_events(limit=12),
        question=req.question if req else None,
        model_override=req.model if req else None,
    )

    return {
        "status": "ok",
        "insight": controller.llm_service.latest_insight.model_dump(),
        "report": report_res.model_dump(),
    }


@app.get("/api/config/room", response_model=RoomConfig)
async def get_room_config() -> RoomConfig:
    return controller.room_config


@app.get("/api/config/layout", response_model=RoomLayoutResponse)
async def get_room_layout() -> RoomLayoutResponse:
    return controller.build_layout_response()


@app.post("/api/config/room", response_model=RoomConfig)
async def update_room_config(cfg: RoomConfig) -> RoomConfig:
    controller.apply_room_layout_hot_reload(cfg)
    return controller.room_config


@app.post("/api/config/layout", response_model=RoomLayoutResponse)
async def update_room_layout(cfg: RoomConfig) -> RoomLayoutResponse:
    return controller.apply_room_layout_hot_reload(cfg)


@app.post("/api/config/room/reset", response_model=RoomConfig)
async def reset_room_config() -> RoomConfig:
    cfg = controller._default_room_config()
    controller.apply_room_layout_hot_reload(cfg)
    return controller.room_config


@app.websocket("/ws/live")
async def websocket_live_endpoint(
    websocket: WebSocket,
    diagnostics: bool = Query(default=False),
) -> None:
    await websocket.accept()
    controller.ws_clients[websocket] = bool(diagnostics)
    try:
        initial = controller.build_live_payload(include_diagnostics=bool(diagnostics))
        await websocket.send_json(initial.model_dump(mode="json", exclude_none=True))

        while True:
            raw_msg = await websocket.receive_text()
            msg = raw_msg.strip()
            if msg == "ping":
                await websocket.send_text("pong")
            elif msg == "diagnostics:on":
                controller.ws_clients[websocket] = True
                snap = controller.build_live_payload(include_diagnostics=True)
                await websocket.send_json(snap.model_dump(mode="json", exclude_none=True))
            elif msg == "diagnostics:off":
                controller.ws_clients[websocket] = False
            elif msg.startswith("{"):
                try:
                    cmd = json.loads(msg)
                    if "diagnostics" in cmd:
                        controller.ws_clients[websocket] = bool(cmd["diagnostics"])
                except Exception:
                    pass
    except WebSocketDisconnect:
        controller.ws_clients.pop(websocket, None)
    except Exception:
        controller.ws_clients.pop(websocket, None)


@app.get("/")
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "Frontend index.html não encontrado."})


FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
