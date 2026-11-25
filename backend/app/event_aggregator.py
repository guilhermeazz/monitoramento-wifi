from collections import deque
from datetime import datetime
import time
from typing import Callable, Dict, List, Optional

from .config import settings
from .models import AggregatedEvent, OccupancyState


class EventAggregator:
    """
    Agregador Semântico de Transições de Estado e Ocupação com Debounce (15–30 s).
    Mantém estatísticas completas da sessão (permanência por zona, pico de pessoas,
    histórico de eventos e transições) para alimentar a camada de Análise Contextual da LLM.
    """

    def __init__(
        self,
        max_history: int = settings.MAX_EVENT_HISTORY,
        debounce_sec: float = settings.EVENT_DEBOUNCE_SEC,
        on_transition_callback: Optional[Callable[[AggregatedEvent], None]] = None,
    ):
        self.history: deque[AggregatedEvent] = deque(maxlen=max_history)
        self.debounce_sec = debounce_sec
        self.on_transition_callback = on_transition_callback

        self.session_start_str: str = datetime.now().strftime("%H:%M:%S")
        self._current_state: OccupancyState = "EMPTY"
        self._current_people_count: int = 0
        self._last_event_time: float = 0.0
        self._state_entered_at: float = time.time()
        self._peak_mvs_in_window: float = 0.0
        self.peak_occupancy: int = 0
        self.total_transitions: int = 0
        self.zone_dwell_seconds: Dict[str, float] = {}
        self._last_tick: float = time.time()

    def ingest_tracking_state(
        self,
        occupancy_state: OccupancyState,
        active_zone: str,
        confidence: float,
        mvs_current: float,
        threshold_on: float,
        people_count: int = 0,
        is_calibrating: bool = False,
        now: Optional[float] = None,
    ) -> Optional[AggregatedEvent]:
        if now is None:
            now = time.time()

        dt = max(0.0, min(1.0, now - self._last_tick))
        self._last_tick = now

        if is_calibrating:
            return None

        self.peak_occupancy = max(self.peak_occupancy, people_count)

        if occupancy_state != "EMPTY" and active_zone and active_zone != "NONE":
            self.zone_dwell_seconds[active_zone] = self.zone_dwell_seconds.get(active_zone, 0.0) + dt

        self._peak_mvs_in_window = max(self._peak_mvs_in_window, mvs_current)

        state_changed = occupancy_state != self._current_state
        people_changed = people_count != self._current_people_count
        elapsed_since_event = now - self._last_event_time

        is_major_entry_or_exit = (
            (self._current_state == "EMPTY" and occupancy_state == "MOTION")
            or (self._current_state != "EMPTY" and occupancy_state == "EMPTY")
            or (people_changed and elapsed_since_event >= 1.5)
        )
        should_emit = False
        if state_changed or people_changed:
            if is_major_entry_or_exit and elapsed_since_event >= 1.2:
                should_emit = True
            elif elapsed_since_event >= min(4.0, self.debounce_sec * 0.35):
                should_emit = True
        elif occupancy_state != "EMPTY" and elapsed_since_event >= self.debounce_sec:
            should_emit = True

        if not should_emit:
            return None

        prev_state = self._current_state
        self._current_state = occupancy_state
        self._current_people_count = people_count
        self._last_event_time = now
        self._state_entered_at = now
        self.total_transitions += 1

        summary = self._format_transition_summary(
            from_state=prev_state,
            to_state=occupancy_state,
            people_count=people_count,
            zone=active_zone,
            confidence=confidence,
            mvs_peak=self._peak_mvs_in_window,
            threshold_on=threshold_on,
        )
        self._peak_mvs_in_window = mvs_current

        event = AggregatedEvent(
            timestamp=datetime.fromtimestamp(now).strftime("%H:%M:%S"),
            epoch=now,
            from_state=prev_state,
            to_state=occupancy_state,
            people_count=people_count,
            active_zone=active_zone,
            confidence=round(confidence, 2),
            mvs_peak=round(mvs_current, 8),
            summary=summary,
        )
        self.history.append(event)

        if self.on_transition_callback is not None:
            self.on_transition_callback(event)

        return event

    def record_system_event(
        self,
        summary_text: str,
        state: OccupancyState = "EMPTY",
        zone: str = "NONE",
        people_count: int = 0,
    ) -> AggregatedEvent:
        now = time.time()
        event = AggregatedEvent(
            timestamp=datetime.fromtimestamp(now).strftime("%H:%M:%S"),
            epoch=now,
            from_state=self._current_state,
            to_state=state,
            people_count=people_count,
            active_zone=zone,
            confidence=0.98,
            mvs_peak=0.0,
            summary=summary_text,
        )
        self.history.append(event)
        self._last_event_time = now
        if self.on_transition_callback is not None:
            self.on_transition_callback(event)
        return event

    @staticmethod
    def _format_transition_summary(
        from_state: OccupancyState,
        to_state: OccupancyState,
        people_count: int,
        zone: str,
        confidence: float,
        mvs_peak: float,
        threshold_on: float,
    ) -> str:
        conf_pct = int(round(confidence * 100))
        ratio = mvs_peak / max(1e-9, threshold_on)
        zone_clean = zone.replace("_", " ").title() if zone != "NONE" else "Área Monitorada"
        p_label = f"{people_count} pessoa" if people_count == 1 else f"{people_count} pessoas"

        if to_state == "EMPTY" or people_count == 0:
            return f"Ambiente livre (0 pessoas). Canal RF estável em repouso ({conf_pct}% confiança)."
        if from_state == "EMPTY" and to_state == "MOTION":
            return f"Entrada detectada: {p_label} em movimento em {zone_clean} ({ratio:.1f}x limiar, {conf_pct}% confiança)."
        if to_state == "PRESENCE_STATIC":
            return f"Presença estática: {p_label} retida(s) em {zone_clean} ({conf_pct}% confiança)."
        return f"Atividade contínua: {p_label} em movimento em {zone_clean} (intensidade {ratio:.1f}x, {conf_pct}% confiança)."

    def get_recent_events(self, limit: int = 20) -> List[AggregatedEvent]:
        items = list(self.history)[-limit:]
        items.reverse()
        return items
