import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Literal, Optional

from .config import settings
from .models import (
    AggregatedEvent,
    CoverageZoneInfo,
    NodeStatus,
    PersonCoordinate,
    RoomConfig,
    TopologyInfo,
)


class EventAggregator:
    """
    Agregador Semântico Temporal.
    Converte coordenadas e estados de 100ms em eventos semânticos de alto nível
    a cada 5 segundos (ou em mudanças relevantes de ocupação/zona), evitando o envio
    de vetores CSI brutos para a camada de LLM.
    """

    def __init__(self, max_history: int = settings.MAX_EVENT_HISTORY):
        self.history: deque[AggregatedEvent] = deque(maxlen=max_history)
        self._last_record_time: float = 0.0
        self._last_people_count: int = -1
        self._last_active_zones: tuple = ()
        self._last_tick_time: float = time.time()

        self.session_start_time: str = datetime.now().strftime("%H:%M:%S")
        self.zone_dwell_seconds: Dict[str, float] = {}
        self.zone_motion_seconds: Dict[str, float] = {}
        self.node_perturbation_acc: Dict[str, List[float]] = {}
        self.peak_occupancy: int = 0
        self.total_zone_transitions: int = 0

    def ingest_state(
        self,
        people: List[PersonCoordinate],
        nodes_status: Dict[str, NodeStatus],
        is_calibrating: bool = False,
    ) -> Optional[AggregatedEvent]:
        now = time.time()
        dt = max(0.0, min(1.0, now - self._last_tick_time))
        self._last_tick_time = now

        if is_calibrating:
            return None

        people_count = len(people)
        self.peak_occupancy = max(self.peak_occupancy, people_count)

        for p in people:
            z = p.zone or "Zona de Abrangência"
            self.zone_dwell_seconds[z] = self.zone_dwell_seconds.get(z, 0.0) + dt
            if p.activity == "moving":
                self.zone_motion_seconds[z] = self.zone_motion_seconds.get(z, 0.0) + dt

        for node_id, st in nodes_status.items():
            if node_id not in self.node_perturbation_acc:
                self.node_perturbation_acc[node_id] = []
            self.node_perturbation_acc[node_id].append(st.perturbation_score)
            if len(self.node_perturbation_acc[node_id]) > 300:
                self.node_perturbation_acc[node_id] = self.node_perturbation_acc[node_id][-300:]

        active_zones_sorted = sorted(list({p.zone for p in people}))
        active_zones_tuple = tuple(active_zones_sorted)

        elapsed = now - self._last_record_time
        count_changed = (people_count != self._last_people_count) and (elapsed >= 1.2)
        zone_changed = (active_zones_tuple != self._last_active_zones) and (elapsed >= 1.8)
        periodic_due = elapsed >= settings.EVENT_AGGREGATION_INTERVAL_SEC

        if not (periodic_due or count_changed or zone_changed):
            return None

        if zone_changed and self._last_people_count > 0:
            self.total_zone_transitions += 1

        self._last_record_time = now
        self._last_people_count = people_count
        self._last_active_zones = active_zones_tuple

        event = self._build_event(people, nodes_status, now)
        self.history.append(event)
        return event

    def record_system_event(self, summary_text: str, people_count: int = 0) -> AggregatedEvent:
        """Registra um evento operacional explícito (ex: Calibração concluída, Troca de modo)."""
        now = time.time()
        event = AggregatedEvent(
            timestamp=datetime.fromtimestamp(now).strftime("%H:%M:%S"),
            epoch=now,
            people_count=people_count,
            active_zones=[],
            activity_level="low",
            summary=summary_text,
        )
        self.history.append(event)
        self._last_record_time = now
        return event

    def _build_event(
        self,
        people: List[PersonCoordinate],
        nodes_status: Dict[str, NodeStatus],
        now: float,
    ) -> AggregatedEvent:
        ts_str = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        count = len(people)

        if count == 0:
            max_node_score = max((n.perturbation_score for n in nodes_status.values()), default=0.0)
            act_level: Literal["none", "low", "moderate", "high"] = "low" if max_node_score > 0.12 else "none"
            summary = "Zona de Abrangência livre (0 pessoas detectadas; links RF estáveis)"
            return AggregatedEvent(
                timestamp=ts_str,
                epoch=now,
                people_count=0,
                active_zones=[],
                activity_level=act_level,
                summary=summary,
            )

        moving_count = sum(1 for p in people if p.activity == "moving")
        if moving_count >= 2 or (count >= 3 and moving_count >= 1):
            activity_level = "high"
        elif moving_count == 1:
            activity_level = "moderate"
        else:
            activity_level = "low"

        zone_group: Dict[tuple[str, str], int] = {}
        active_zones_list: List[str] = []
        for p in people:
            if p.zone not in active_zones_list:
                active_zones_list.append(p.zone)
            state_pt = "em movimento" if p.activity == "moving" else "parada"
            key = (state_pt, p.zone)
            zone_group[key] = zone_group.get(key, 0) + 1

        parts: List[str] = []
        for (state_pt, z_name), qty in zone_group.items():
            prep = "no" if z_name.startswith(("Centro", "Setor")) else "na"
            state_label = "paradas" if (qty > 1 and state_pt == "parada") else state_pt
            parts.append(f"{qty} {state_label} {prep} {z_name}")

        noun = "pessoa detectada" if count == 1 else "pessoas detectadas"
        summary = f"{count} {noun} ({', '.join(parts)})"

        return AggregatedEvent(
            timestamp=ts_str,
            epoch=now,
            people_count=count,
            active_zones=active_zones_list,
            activity_level=activity_level,
            summary=summary,
        )

    def get_recent_events(self, limit: int = 25) -> List[AggregatedEvent]:
        items = list(self.history)[-limit:]
        items.reverse()
        return items

    def get_analytical_digest(
        self,
        room_config: RoomConfig,
        current_people: List[PersonCoordinate],
        nodes_status: Dict[str, NodeStatus],
        topology_info: Optional[TopologyInfo] = None,
        coverage_info: Optional[CoverageZoneInfo] = None,
        calibration_mode: str = "nominal",
    ) -> dict:
        events_list = [
            {
                "timestamp": e.timestamp,
                "people_count": e.people_count,
                "active_zones": e.active_zones,
                "activity_level": e.activity_level,
                "summary": e.summary,
            }
            for e in list(self.history)[-60:]
        ]

        total_dwell = sum(self.zone_dwell_seconds.values())
        zone_stats = {}
        for z, dwell_val in self.zone_dwell_seconds.items():
            dwell = round(dwell_val, 1)
            motion = round(self.zone_motion_seconds.get(z, 0.0), 1)
            pct = round((dwell / total_dwell) * 100.0, 1) if total_dwell > 0 else 0.0
            zone_stats[z] = {
                "dwell_person_seconds": dwell,
                "motion_person_seconds": motion,
                "share_percent": pct,
            }

        nodes_digest = {}
        for nid, st in nodes_status.items():
            hist_scores = self.node_perturbation_acc.get(nid, [st.perturbation_score])
            nodes_digest[nid] = {
                "name": st.name,
                "online": st.online,
                "position_xy": [round(st.x, 2), round(st.y, 2)],
                "estimated_distance_m": st.estimated_distance_m,
                "current_rssi_dbm": st.rssi,
                "current_perturbation": st.perturbation_score,
                "subcarrier_diversity": st.subcarrier_diversity,
                "energy_ratio": st.energy_ratio,
                "estimated_people_on_link": st.estimated_people_on_link,
                "longitudinal_focus": st.longitudinal_focus,
                "avg_perturbation": round(sum(hist_scores) / max(1, len(hist_scores)), 3),
                "max_perturbation": round(max(hist_scores, default=0.0), 3),
            }

        return {
            "session_start": self.session_start_time,
            "current_time": datetime.now().strftime("%H:%M:%S"),
            "calibration_mode": calibration_mode,
            "coverage_info": coverage_info.model_dump() if coverage_info else None,
            "topology_info": topology_info.model_dump() if topology_info else None,
            "room_dimensions_m": {
                "width_m": room_config.width_m,
                "length_m": room_config.length_m,
                "area_m2": coverage_info.active_area_m2 if coverage_info else round(room_config.width_m * room_config.length_m, 1),
            },
            "current_occupancy": len(current_people),
            "peak_occupancy": self.peak_occupancy,
            "total_zone_transitions": self.total_zone_transitions,
            "zone_statistics": zone_stats,
            "rf_nodes_telemetry": nodes_digest,
            "event_timeline": events_list,
        }
