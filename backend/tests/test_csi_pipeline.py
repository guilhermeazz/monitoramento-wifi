import asyncio
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import settings
from app.dsp_engine import CSIRingSample, DSPEngine
from app.models import CSIPacket, NodePosition, RoomConfig
from app.simulator import CSISimulator
from app.spatial_tracker import SpatialTracker
from app.udp_receiver import BINARY_HEADER_FMT, CSI_PACKET_MAGIC, parse_csi_payload


def _make_room_config(num_nodes: int = 3) -> RoomConfig:
    rx_all = [
        NodePosition(id="esp_1", name="ESP32-S3 #1", mac="c0:56:27:e1:20:01", x=0.14, y=0.84),
        NodePosition(id="esp_2", name="ESP32-S3 #2", mac="c0:56:27:e1:20:02", x=0.86, y=0.84),
        NodePosition(id="esp_3", name="ESP32-S3 #3", mac="c0:56:27:e1:20:03", x=0.50, y=0.88),
    ]
    return RoomConfig(
        width_m=5.0,
        length_m=4.5,
        tx=NodePosition(id="tx_router", name="Cisco Linksys E1200", mac="20:aa:4b:4f:67:1d", x=0.50, y=0.10),
        rx_nodes=rx_all[:num_nodes],
    )


def _calibrate_two_stages(dsp: DSPEngine, sim: CSISimulator, room: RoomConfig, t0: float) -> float:
    """Executa Etapa 1 (Sala Vazia) + Etapa 2 (1 Pessoa caminhando) em ambiente controlado."""
    dt = 0.02  # 50 Hz
    stage1_samples: Dict[str, List[CSIRingSample]] = {rx.id: [] for rx in room.rx_nodes}

    for i in range(250):  # 5.0s em sala vazia
        ts = t0 + i * dt
        for rx in room.rx_nodes:
            pkt = sim.synthesize_packet(
                rx.id,
                rx.x,
                rx.y,
                room.tx.x,
                room.tx.y,
                now=ts,
                force_empty=True,
                width_m=room.width_m,
                length_m=room.length_m,
            )
            stage1_samples[rx.id].append(dsp.enqueue_packet(pkt))

    for rx in room.rx_nodes:
        dsp.calibrate_node_from_samples(rx.id, stage1_samples[rx.id])

    # Etapa 2: 1 pessoa caminhando pela sala por 6s (300 frames)
    sim.set_custom_trajectories([(0.9, 1.8, 4.1, 3.1, 1.05)], allow_pauses=False)
    stage2_samples: Dict[str, List[CSIRingSample]] = {rx.id: [] for rx in room.rx_nodes}
    t_s2 = t0 + 250 * dt

    for j in range(300):
        ts = t_s2 + (j + 1) * dt
        sim._step_kinematics(dt, ts)
        for rx in room.rx_nodes:
            pkt = sim.synthesize_packet(
                rx.id,
                rx.x,
                rx.y,
                room.tx.x,
                room.tx.y,
                now=ts,
                force_empty=False,
                width_m=room.width_m,
                length_m=room.length_m,
            )
            stage2_samples[rx.id].append(dsp.enqueue_packet(pkt))

    dsp.calibrate_stage2_single_person(stage2_samples)
    for nid in stage1_samples:
        dsp.ring_buffers[nid].clear()
        if nid in dsp.runtime_states:
            dsp.runtime_states[nid].state = "EMPTY"
            dsp.runtime_states[nid].consecutive_motion_ticks = 0
            dsp.runtime_states[nid].calibrated_energy = 0.0

    return t_s2 + 301 * dt


def test_zero_people_and_agc_immunity():
    """
    Critério 1: 0 pessoas na sala -> contagem estritamente 0 (zero fantasmas),
    mesmo sob saltos severos de ganho escalar do AGC (2x, 5x, 0.3x).
    """
    room = _make_room_config(3)
    dsp = DSPEngine(num_subcarriers=52)
    tracker = SpatialTracker()
    sim = CSISimulator(get_room_config=lambda: room, seed=101)
    sim.set_people_count(0)

    t_cur = _calibrate_two_stages(dsp, sim, room, t0=1_700_000_000.0)
    sim.set_people_count(0)

    dt = 0.02
    agc_pattern = [1.0, 2.0, 5.0, 0.3, 4.5, 0.25, 3.2, 1.0]
    ghost_frames = 0
    node_ids = [rx.id for rx in room.rx_nodes]

    for j in range(200):
        ts = t_cur + j * dt
        agc_gain = agc_pattern[j % len(agc_pattern)]
        for rx in room.rx_nodes:
            pkt = sim.synthesize_packet(
                rx.id,
                rx.x,
                rx.y,
                room.tx.x,
                room.tx.y,
                now=ts,
                force_empty=True,
                agc_scale_override=agc_gain,
            )
            dsp.enqueue_packet(pkt)

        if j % 2 == 1:
            states = dsp.process_all_nodes_tick(node_ids, now=ts)
            res = tracker.update(room, states, is_calibrating=False, now=ts)
            if res.target_count != 0 or res.occupancy_state != "EMPTY" or len(res.targets) != 0:
                ghost_frames += 1

    assert ghost_frames == 0, f"Detectados {ghost_frames} frames com pessoas fantasmas em sala vazia!"


def test_one_person_count_2d_accuracy_and_static_hold():
    """
    Critério 2: 1 pessoa caminhando -> contagem estável em 1 + erro médio 2D controlado,
    latência de disparo < 500 ms, e retenção de contagem 1 durante pausa estática via hold-timer.
    """
    room = _make_room_config(3)
    dsp = DSPEngine(num_subcarriers=52)
    tracker = SpatialTracker()
    sim = CSISimulator(get_room_config=lambda: room, seed=202)

    t_cur = _calibrate_two_stages(dsp, sim, room, t0=1_700_010_000.0)
    sim.set_custom_trajectories([(1.1, 2.5, 3.9, 2.5, 0.85)], allow_pauses=False)

    dt = 0.02
    node_ids = [rx.id for rx in room.rx_nodes]
    first_trigger_ts = None
    settled_counts: List[int] = []
    pos_errors_m: List[float] = []

    for j in range(250):  # 5.0 segundos de caminhada de 1 pessoa
        ts = t_cur + (j + 1) * dt
        sim._step_kinematics(dt, ts)
        for rx in room.rx_nodes:
            pkt = sim.synthesize_packet(rx.id, rx.x, rx.y, room.tx.x, room.tx.y, now=ts)
            dsp.enqueue_packet(pkt)

        if j % 2 == 1:
            states = dsp.process_all_nodes_tick(node_ids, now=ts)
            res = tracker.update(room, states, is_calibrating=False, now=ts)
            if res.occupancy_state == "MOTION" and first_trigger_ts is None:
                first_trigger_ts = ts

            if j >= 60:  # Após convergência da janela de 1.0s
                settled_counts.append(res.target_count)
                assert len(res.targets) == 1
                gt_x, gt_y = sim.get_agent_positions_m()[0]
                err = math.hypot(res.targets[0].x_m - gt_x, res.targets[0].y_m - gt_y)
                pos_errors_m.append(err)

    assert first_trigger_ts is not None
    latency_ms = (first_trigger_ts - t_cur) * 1000.0
    assert latency_ms < 500.0, f"Latência de entrada ({latency_ms:.1f} ms) excedeu 500 ms!"

    count_1_ratio = sum(1 for c in settled_counts if c == 1) / len(settled_counts)
    mean_2d_err = float(np.mean(pos_errors_m))
    assert count_1_ratio >= 0.90, f"Contagem para 1 pessoa instável: {settled_counts}"
    assert mean_2d_err < 1.35, f"Erro médio 2D elevado para 1 pessoa ({mean_2d_err:.2f} m)"

    # Testa transição transparente para presença estática (pessoa para de andar por 3s)
    sim._agents[0].pause_until = t_cur + 20.0
    sim._agents[0].vx_mps = 0.0
    sim._agents[0].vy_mps = 0.0
    t_pause_start = t_cur + 251 * dt

    for k in range(120):  # 2.4s parada respirando
        ts = t_pause_start + (k + 1) * dt
        sim._step_kinematics(dt, ts)
        for rx in room.rx_nodes:
            pkt = sim.synthesize_packet(rx.id, rx.x, rx.y, room.tx.x, room.tx.y, now=ts)
            dsp.enqueue_packet(pkt)
        if k % 2 == 1:
            states = dsp.process_all_nodes_tick(node_ids, now=ts)
            res = tracker.update(room, states, is_calibrating=False, now=ts)

    assert res.occupancy_state in ("PRESENCE_STATIC", "MOTION")
    assert res.target_count == 1, "Pessoa parada perdeu contagem durante o hold-timer!"


def test_two_and_three_people_opposite_directions_headcount_and_mot():
    """
    Critério 3: 2 e 3 pessoas simuladas caminhando em direções opostas/independentes ->
    contagens estáveis em 2 e 3 com trajetórias distintas rastreadas por Hungarian + Kalman 2D.
    """
    room = _make_room_config(3)
    dsp = DSPEngine(num_subcarriers=52)
    tracker = SpatialTracker()
    sim = CSISimulator(get_room_config=lambda: room, seed=303)

    t_cur = _calibrate_two_stages(dsp, sim, room, t0=1_700_020_000.0)
    dt = 0.02
    node_ids = [rx.id for rx in room.rx_nodes]

    # --- CENÁRIO A: 2 PESSOAS EM DIREÇÕES OPOSTAS (Oeste: Norte->Sul vs Leste: Sul->Norte) ---
    sim.set_custom_trajectories(
        [
            (1.90, 1.30, 0.85, 3.75, 0.52),  # Agente 1 (Corredor Oeste: Norte -> Sul)
            (4.15, 3.75, 3.10, 1.30, 0.54),  # Agente 2 (Corredor Leste: Sul -> Norte - direção oposta!)
        ],
        allow_pauses=False,
    )

    counts_2p: List[int] = []
    seps_2p: List[float] = []
    last_res_2p = None
    for j in range(260):  # 5.2 segundos de caminhada contínua em sentidos opostos
        ts = t_cur + (j + 1) * dt
        sim._step_kinematics(dt, ts)
        for rx in room.rx_nodes:
            pkt = sim.synthesize_packet(rx.id, rx.x, rx.y, room.tx.x, room.tx.y, now=ts)
            dsp.enqueue_packet(pkt)

        if j % 2 == 1:
            states = dsp.process_all_nodes_tick(node_ids, now=ts)
            last_res_2p = tracker.update(room, states, is_calibrating=False, now=ts)
            if j >= 100:  # Após janela de estabilização de 2.0s
                counts_2p.append(last_res_2p.target_count)
                if len(last_res_2p.targets) >= 2:
                    seps_2p.append(
                        math.hypot(
                            last_res_2p.targets[0].x_m - last_res_2p.targets[1].x_m,
                            last_res_2p.targets[0].y_m - last_res_2p.targets[1].y_m,
                        )
                    )

    ratio_2p = sum(1 for c in counts_2p if c == 2) / len(counts_2p)
    assert ratio_2p >= 0.80, f"Contagem de 2 pessoas não estabilizou em 2 (obtido: {counts_2p})"
    assert last_res_2p is not None and len(seps_2p) >= 20
    mean_sep_2p = float(np.mean(seps_2p))
    assert mean_sep_2p >= 1.10, f"Trajetórias de 2 pessoas próximas demais (sep média={mean_sep_2p:.2f}m)!"

    # --- CENÁRIO B: 3 PESSOAS EM DIREÇÕES INDEPENDENTES/OPOSTAS ---
    tracker.reset()
    for nid in node_ids:
        dsp.ring_buffers[nid].clear()

    t_3p_start = t_cur + 300 * dt
    sim.set_custom_trajectories(
        [
            (1.90, 1.30, 0.80, 3.75, 0.52),  # Agente 1 (Corredor Oeste -> Sul)
            (4.20, 3.75, 3.10, 1.30, 0.54),  # Agente 2 (Corredor Leste -> Norte)
            (2.50, 3.85, 2.50, 1.35, 0.50),  # Agente 3 (Corredor Central -> Norte)
        ],
        allow_pauses=False,
    )

    counts_3p: List[int] = []
    max_distinct_tracks = 0
    for j in range(300):  # 6.0 segundos
        ts = t_3p_start + (j + 1) * dt
        sim._step_kinematics(dt, ts)
        for rx in room.rx_nodes:
            pkt = sim.synthesize_packet(rx.id, rx.x, rx.y, room.tx.x, room.tx.y, now=ts)
            dsp.enqueue_packet(pkt)

        if j % 2 == 1:
            states = dsp.process_all_nodes_tick(node_ids, now=ts)
            res_3p = tracker.update(room, states, is_calibrating=False, now=ts)
            max_distinct_tracks = max(max_distinct_tracks, len(res_3p.targets))
            if j >= 120:
                counts_3p.append(res_3p.target_count)

    ratio_3p = sum(1 for c in counts_3p if c >= 3) / len(counts_3p)
    assert ratio_3p >= 0.75, f"Contagem de 3 pessoas não estabilizou em 3+ (obtido: {counts_3p})"
    assert max_distinct_tracks >= 2, "Multi-Object Tracker deve manter múltiplos alvos distintos!"


def test_random_waypoint_independence_and_single_link_uncertainty():
    """
    Critério 4: Valida que o Random Waypoint cria agentes com >= 1.2m de distância inicial e vetores
    independentes, e que com apenas 1 link ativo o SpatialTracker usa FRESNEL_HALO (sem ponto rígido enganoso).
    """
    room_3 = _make_room_config(3)
    sim = CSISimulator(get_room_config=lambda: room_3, seed=77)
    sim.set_people_count(3)
    positions = sim.get_agent_positions_m()
    assert len(positions) == 3

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = math.hypot(positions[i][0] - positions[j][0], positions[i][1] - positions[j][1])
            assert d >= 1.19, f"Agentes iniciaram próximos demais ({d:.2f}m < 1.20m)!"

    # Verifica modo de incerteza para 1 link ativo
    room_1 = _make_room_config(1)
    dsp = DSPEngine(num_subcarriers=52)
    tracker = SpatialTracker()
    now = 1_700_030_000.0
    rt = dsp.runtime_states.setdefault("esp_1", dsp.process_node_tick("esp_1", now))
    rt.online = True
    rt.state = "MOTION"
    rt.intensity = 3.5
    rt.calibrated_energy = 2.2
    rt.mvs_current = 8.0e-6
    rt.threshold_on = 2.5e-6

    res_1 = tracker.update(room_1, dsp.runtime_states, is_calibrating=False, now=now + 0.1)
    assert res_1.topology_mode == "SINGLE_NODE_FRESNEL"
    assert res_1.uncertainty_mode == "FRESNEL_HALO"
    assert res_1.targets == []


def test_nbvi_excludes_dead_and_noisy_subcarriers_and_slow_ema():
    """
    Critério 5: Garante que subportadoras nulas/mortas (0.0) e ruidosas são excluídas pelo NBVI,
    e que o EMA lento de baseline só atualiza após > 120s contínuos em 0 pessoas.
    """
    rng = np.random.default_rng(123)
    n_samples = 200
    n_sub = 52

    base_matrix = np.ones((n_samples, n_sub), dtype=np.float64) * 20.0
    base_matrix += rng.normal(0.0, 0.03, size=(n_samples, n_sub))

    # Subportadoras ruidosas + subportadoras mortas/nulas (0.0)
    noisy_channels = [0, 5, 13, 25, 37, 51]
    dead_channels = [21, 22, 23, 24]
    for ch in noisy_channels:
        base_matrix[:, ch] = 2.5 + rng.normal(0.0, 2.2, size=n_samples)
    for ch in dead_channels:
        base_matrix[:, ch] = 0.0

    norms = np.sqrt(np.sum(base_matrix ** 2, axis=1, keepdims=True) + 1e-8)
    amp_norm_mat = base_matrix / norms

    selected_12, _, _, _, _ = DSPEngine.compute_nbvi_selection(amp_norm_mat, top_k=12)
    assert len(selected_12) == 12
    for bad_ch in noisy_channels + dead_channels:
        assert bad_ch not in selected_12, f"Canal inválido/morto {bad_ch} selecionado indevidamente!"

    # Valida regra de > 120s para EMA lento em sala vazia
    dsp = DSPEngine(num_subcarriers=52)
    room = _make_room_config(1)
    sim = CSISimulator(get_room_config=lambda: room, seed=55)
    t0 = 1_700_040_000.0
    samples = []
    for i in range(60):
        pkt = sim.synthesize_packet("esp_1", 0.14, 0.84, 0.50, 0.10, now=t0 + i * 0.02, force_empty=True)
        samples.append(dsp.enqueue_packet(pkt))
    prof = dsp.calibrate_node_from_samples("esp_1", samples)
    initial_mean = prof.mvs_mean_empty

    # Antes de 120s -> não altera
    pkt_60s = sim.synthesize_packet("esp_1", 0.14, 0.84, 0.50, 0.10, now=t0 + 60.0, force_empty=True)
    dsp.enqueue_packet(pkt_60s)
    dsp.process_all_nodes_tick(["esp_1"], now=t0 + 60.0)
    assert dsp.profiles["esp_1"].mvs_mean_empty == initial_mean

    # Após 125s contínuos em EMPTY -> aplica EMA lento
    for k in range(15):
        ts = t0 + 186.0 + k * 0.02
        dsp.enqueue_packet(sim.synthesize_packet("esp_1", 0.14, 0.84, 0.50, 0.10, now=ts, force_empty=True))
    dsp.process_all_nodes_tick(["esp_1"], now=t0 + 186.3)
    assert dsp.empty_since_ts is not None


def test_layout_hot_reload_and_geometry_warnings():
    """
    Critério 6: Valida que salvar novo layout recalcula a matriz W (0.25m/célula) em hot-reload
    e emite avisos discretos de geometria (< 1.5m, > 7.0m e colinearidade).
    """
    tracker = SpatialTracker(cell_size_m=0.25)
    bad_room = RoomConfig(
        width_m=8.0,
        length_m=6.0,
        tx=NodePosition(id="tx_router", name="Linksys E1200", x=0.10, y=0.10),
        rx_nodes=[
            NodePosition(id="esp_1", name="ESP #1", x=0.18, y=0.18),  # < 1.5 m
            NodePosition(id="esp_2", name="ESP #2", x=0.50, y=0.50),  # Colinear com esp_1!
            NodePosition(id="esp_3", name="ESP #3", x=0.98, y=0.95),  # > 7.0 m
        ],
    )
    warnings = tracker.hot_reload_layout(bad_room)
    codes = {w.code for w in warnings}
    assert tracker.grid_nx == 32  # 8.0m / 0.25m
    assert tracker.grid_ny == 24  # 6.0m / 0.25m
    assert "LINK_TOO_SHORT" in codes
    assert "LINK_TOO_LONG" in codes
    assert "COLLINEAR_LINKS" in codes


def test_binary_udp_and_fastapi_routes():
    """
    Critério 7: Valida protocolo binário UDP 0xC510, payload leve do WebSocket e endpoints FastAPI
    (calibração de 2 etapas, layout hot-reload e modo único de operação).
    """
    mac_bytes = bytes([0xC0, 0x56, 0x27, 0xE1, 0x20, 0x02])
    header = struct.pack(
        BINARY_HEADER_FMT,
        CSI_PACKET_MAGIC,
        1,
        2,
        mac_bytes,
        123_456_789,
        -47,
        -94,
        11,
        128,
    )
    iq_data = [((i % 50) - 25) for i in range(128)]
    pkt = parse_csi_payload(header + struct.pack("<128b", *iq_data))
    assert pkt is not None and pkt.node_id == "esp_2"

    from app.main import (
        control_simulator_and_dsp,
        controller,
        get_current_state,
        get_room_layout,
        health_check,
        start_guided_calibration,
        cancel_guided_calibration,
    )
    from app.models import CalibrationStartRequest, SimulatorControlRequest

    controller.llm_service._last_offline_ts = time.time() + 3600.0

    # Verifica UI Dark Liquid Glass: modo único (sem botões de cenários separados), wizard 2 etapas e editor de layout
    index_html = (BACKEND_ROOT.parent / "frontend" / "index.html").read_text(encoding="utf-8")
    styles_css = (BACKEND_ROOT.parent / "frontend" / "styles.css").read_text(encoding="utf-8")
    assert "liquid-glass" in index_html
    assert "counter-glass-orb" in styles_css
    assert "data-scenario=" not in index_html
    assert "layout-editor-modal" in index_html
    assert "chk-snap-025" in index_html
    assert "btn-start-cal-stage1" in index_html
    assert "btn-start-cal-stage2" in index_html
    assert 'data-sim-people="4"' in index_html

    now = 1_700_050_000.0
    for i in range(30):
        ts = now + i * 0.02
        for rx in controller.room_config.rx_nodes:
            p = controller.simulator.synthesize_packet(
                rx.id, rx.x, rx.y, controller.room_config.tx.x, controller.room_config.tx.y, now=ts
            )
            controller.dsp_engine.enqueue_packet(p)
    controller.run_dsp_tick(now + 0.6)

    light_state = controller.build_live_payload(include_diagnostics=False)
    light_bytes = json.dumps(light_state.model_dump(mode="json", exclude_none=True)).encode("utf-8")
    assert len(light_bytes) < 1200, f"Payload leve muito grande: {len(light_bytes)} bytes"
    assert not hasattr(light_state, "scenario")

    def _eval_coro(coro):
        try:
            coro.send(None)
        except StopIteration as stop:
            return stop.value
        finally:
            coro.close()

    hlth = _eval_coro(health_check())
    assert hlth["status"] == "ok"

    layout_res = _eval_coro(get_room_layout())
    assert layout_res.cell_size_m == 0.25

    cal_s1 = _eval_coro(start_guided_calibration(CalibrationStartRequest(stage=1, prep_seconds=5, capture_seconds=15)))
    assert cal_s1["calibration"]["stage"] == 1
    _eval_coro(cancel_guided_calibration())

    ctrl_res = _eval_coro(control_simulator_and_dsp(SimulatorControlRequest(mode="simulation", simulated_people=3)))
    assert ctrl_res["simulated_people"] == 3
    st_cur = _eval_coro(get_current_state(diagnostics=False))
    assert st_cur.count_display in ("0", "1", "2", "3", "3+")
