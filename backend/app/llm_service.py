import asyncio
from datetime import datetime
import json
import logging
from typing import Dict, List, Literal, Optional
import httpx

from .config import settings
from .models import (
    AggregatedEvent,
    LLMInsightPayload,
    LLMReportResponse,
    LLMStatusResponse,
    NodeDiagnosticDetail,
    OccupancyState,
)

logger = logging.getLogger("wifi_spatial.llm")

RECOMMENDED_MODELS = [
    "qwen2.5:7b",
    "qwen3:8b",
    "gemma3:4b",
    "llama3.2:3b",
]


class LLMService:
    """
    Serviço Assíncrono Completo de Inteligência Analítica Espacial (Ollama Local + Motor Estatístico):
    - 100% desacoplado do loop DSP/WebSocket (atualizações automáticas via asyncio.create_task).
    - Suporta:
      1. Insight Rápido em Tempo Real (2 frases diretas sobre a transição atual).
      2. Relatório Executivo Completo em Markdown (Ocupação, Contagem, Permanência por Zonas, Diagnóstico RF/NBVI e Recomendações).
      3. Chat Interativo de Perguntas Livres em Linguagem Natural sobre o ambiente.
    """

    def __init__(self) -> None:
        self.base_url: str = settings.OLLAMA_BASE_URL
        self.model_name: str = settings.OLLAMA_MODEL
        self.timeout_sec: float = settings.OLLAMA_TIMEOUT_SEC
        self.latest_insight = LLMInsightPayload(
            summary="Monitoramento espacial Wi-Fi CSI ativo. Aguardando eventos na Zona de Fresnel.",
            updated_at=datetime.now().strftime("%H:%M:%S"),
            provider="statistical-engine",
        )
        self._bg_task: Optional[asyncio.Task] = None
        self._last_offline_ts: float = 0.0

    async def _is_ollama_reachable(self) -> bool:
        """Verifica rapidamente em <120ms se a porta local do Ollama está aberta."""
        import time as _time
        now = _time.time()
        if (now - self._last_offline_ts) < 3.0:
            return False
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", 11434),
                timeout=0.12,
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except Exception:
            self._last_offline_ts = now
            return False

    async def get_status(self, preferred_model: Optional[str] = None) -> LLMStatusResponse:
        if preferred_model and preferred_model.strip():
            self.model_name = preferred_model.strip()

        try:
            if not await self._is_ollama_reachable():
                raise ConnectionError("Porta 11434 fechada")
            timeout = httpx.Timeout(1.5, connect=0.3)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()

            installed = [
                m.get("name", "")
                for m in data.get("models", [])
                if isinstance(m, dict) and m.get("name")
            ]
            if installed and not preferred_model and self.model_name not in installed:
                self.model_name = installed[0]

            available: List[str] = []
            for m in installed + RECOMMENDED_MODELS:
                if m and m not in available:
                    available.append(m)

            return LLMStatusResponse(
                online=True,
                base_url=self.base_url,
                active_model=self.model_name,
                installed_models=installed,
                available_models=available,
                badge_text=f"Ollama Online ({self.model_name})",
            )
        except Exception:
            available = [self.model_name] + [m for m in RECOMMENDED_MODELS if m != self.model_name]
            return LLMStatusResponse(
                online=False,
                base_url=self.base_url,
                active_model=self.model_name,
                installed_models=[],
                available_models=available,
                badge_text="Motor Analítico Local (Offline)",
            )

    def schedule_background_update(
        self,
        event: AggregatedEvent,
        recent_events: List[AggregatedEvent],
        active_nodes: int,
    ) -> None:
        """Agenda atualização não-bloqueante em background a cada transição de estado."""
        now_str = datetime.now().strftime("%H:%M:%S")
        self.latest_insight = LLMInsightPayload(
            summary=self._build_fast_summary(event, active_nodes),
            updated_at=now_str,
            provider="statistical-engine",
        )

        try:
            loop = asyncio.get_running_loop()
            if self._bg_task is None or self._bg_task.done():
                self._bg_task = loop.create_task(
                    self._refine_with_llm_async(event, recent_events, active_nodes)
                )
        except RuntimeError:
            pass

    async def generate_analysis(
        self,
        mode: Literal["quick", "full_report", "ask"],
        occupancy_state: OccupancyState,
        people_count: int,
        peak_occupancy: int,
        active_zone: str,
        confidence: float,
        active_nodes: int,
        topology_mode: str,
        mvs_current: float,
        threshold_on: float,
        intensity: float,
        calibration_quality: str,
        zone_dwell_seconds: Dict[str, float],
        nodes_detail: Dict[str, NodeDiagnosticDetail],
        recent_events: List[AggregatedEvent],
        question: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> LLMReportResponse:
        target_model = (model_override or self.model_name).strip()
        self.model_name = target_model
        now_str = datetime.now().strftime("%H:%M:%S")

        digest = {
            "horario": now_str,
            "estado_ocupacao": occupancy_state,
            "pessoas_na_sala_agora": people_count,
            "pico_pessoas_sessao": peak_occupancy,
            "zona_ativa": active_zone,
            "confianca_pct": int(round(confidence * 100)),
            "topologia": topology_mode,
            "nos_esp32_ativos": active_nodes,
            "mvs_atual": round(mvs_current, 8),
            "limiar_theta_on": round(threshold_on, 8),
            "intensidade_x": round(intensity, 2),
            "qualidade_calibracao_nbvi": calibration_quality,
            "tempo_permanencia_zonas_seg": {k: round(v, 1) for k, v in zone_dwell_seconds.items()},
            "sensores": {
                nid: {
                    "nome": nd.name,
                    "online": nd.online,
                    "estado": nd.state,
                    "taxa_hz": nd.pps,
                    "rssi_dbm": nd.rssi,
                    "intensidade": nd.intensity,
                    "qualidade_nbvi": nd.quality,
                }
                for nid, nd in nodes_detail.items()
            },
            "eventos_recentes": [
                {"hora": e.timestamp, "pessoas": e.people_count, "zona": e.active_zone, "resumo": e.summary}
                for e in recent_events[:10]
            ],
        }

        if mode == "quick":
            short_text = self._build_fallback_short(
                occupancy_state=occupancy_state,
                people_count=people_count,
                peak_occupancy=peak_occupancy,
                active_zone=active_zone,
                confidence=confidence,
                active_nodes=active_nodes,
                recent_events=recent_events,
            )
            self.latest_insight = LLMInsightPayload(
                summary=short_text,
                updated_at=now_str,
                provider="statistical-engine",
            )
            return LLMReportResponse(
                status="ok",
                mode="quick",
                provider="statistical-engine",
                model=target_model,
                summary=short_text,
                markdown_report=short_text,
                updated_at=now_str,
                events_analyzed=len(recent_events),
            )

        # Tenta consultar o Ollama Local para relatório completo ou pergunta interativa
        system_prompt = (
            "Você é o Especialista Analítico de um Sistema de Monitoramento Espacial Wi-Fi CSI (ESP32-S3 + Cisco Linksys E1200).\n"
            "Responda SEMPRE em Português do Brasil com formatação Markdown limpa, estruturada e objetiva.\n"
            "Use estritamente as métricas reais do JSON fornecido (contagem de pessoas, zonas, MVS, RSSI e histórico)."
        )

        if mode == "ask" and question:
            user_prompt = (
                f"Pergunta do Operador: \"{question}\"\n\n"
                "Responda de forma completa, clara e técnica em Markdown com base nos dados em tempo real abaixo:\n"
                f"```json\n{json.dumps(digest, ensure_ascii=False)}\n```"
            )
        else:
            user_prompt = (
                "Gere um **Relatório Analítico Completo de Monitoramento Espacial Wi-Fi CSI** em Markdown contendo:\n"
                "1. **Panorama de Ocupação e Contagem de Pessoas** (pessoas presentes agora, pico da sessão, estado de movimento/estático e confiança)\n"
                "2. **Mapeamento por Zonas e Tempo de Permanência**\n"
                "3. **Diagnóstico Físico RF & Sensores ESP32-S3** (relação MVS/Limiar, RSSI, taxa em Hz e qualidade NBVI)\n"
                "4. **Conclusão e Recomendações Operacionais**\n\n"
                f"Dados Reais da Sessão:\n```json\n{json.dumps(digest, ensure_ascii=False)}\n```"
            )

        try:
            llm_md = await self._call_ollama(system_prompt, user_prompt, target_model, max_tokens=550)
            if llm_md:
                first_line = llm_md.splitlines()[0].lstrip("#*>- ").strip()
                short_summary = first_line[:220] if len(first_line) > 15 else self.latest_insight.summary
                self.latest_insight = LLMInsightPayload(
                    summary=short_summary,
                    updated_at=now_str,
                    provider=f"ollama:{target_model}",
                )
                return LLMReportResponse(
                    status="ok",
                    mode=mode,
                    provider=f"ollama:{target_model}",
                    model=target_model,
                    summary=short_summary,
                    markdown_report=llm_md,
                    updated_at=now_str,
                    events_analyzed=len(recent_events),
                )
        except Exception as exc:
            logger.debug(f"Ollama local indisponível ({exc}); gerando análise estatística completa.")

        # Fallback Analítico Completo e Rico em Markdown (100% funcional offline)
        if mode == "ask" and question:
            md_report, short_sum = self._build_comprehensive_fallback_answer(question, digest)
        else:
            md_report, short_sum = self._build_comprehensive_fallback_report(digest)

        self.latest_insight = LLMInsightPayload(
            summary=short_sum,
            updated_at=now_str,
            provider="statistical-engine",
        )
        return LLMReportResponse(
            status="ok",
            mode=mode,
            provider="statistical-engine",
            model=f"Motor Analítico Local ({target_model} offline)",
            summary=short_sum,
            markdown_report=md_report,
            updated_at=now_str,
            events_analyzed=len(recent_events),
        )

    async def summarize_on_demand(
        self,
        occupancy_state: OccupancyState,
        active_zone: str,
        confidence: float,
        active_nodes: int,
        recent_events: List[AggregatedEvent],
        question: Optional[str] = None,
        model_override: Optional[str] = None,
        people_count: int = 0,
        peak_occupancy: int = 0,
    ) -> LLMInsightPayload:
        """Compatibilidade direta com chamadas que esperam LLMInsightPayload."""
        short_text = self._build_fallback_short(
            occupancy_state=occupancy_state,
            people_count=people_count,
            peak_occupancy=peak_occupancy,
            active_zone=active_zone,
            confidence=confidence,
            active_nodes=active_nodes,
            recent_events=recent_events,
        )
        self.latest_insight = LLMInsightPayload(
            summary=short_text,
            updated_at=datetime.now().strftime("%H:%M:%S"),
            provider="statistical-engine",
        )
        return self.latest_insight

    async def _refine_with_llm_async(
        self,
        event: AggregatedEvent,
        recent_events: List[AggregatedEvent],
        active_nodes: int,
    ) -> None:
        try:
            system_prompt = (
                "Você é o motor analítico de um radar espacial Wi-Fi CSI. "
                "Resuma a transição de estado e ocupação em Português em NO MÁXIMO 2 frases curtas e diretas."
            )
            payload_str = json.dumps(
                {
                    "transicao": f"{event.from_state} -> {event.to_state}",
                    "pessoas": event.people_count,
                    "zona": event.active_zone,
                    "confianca": event.confidence,
                    "sensores": active_nodes,
                    "resumo_base": event.summary,
                },
                ensure_ascii=False,
            )
            text = await self._call_ollama(system_prompt, payload_str, self.model_name, max_tokens=110)
            if text:
                clean = " ".join(ln.strip() for ln in text.splitlines() if ln.strip())[:260]
                self.latest_insight = LLMInsightPayload(
                    summary=clean,
                    updated_at=datetime.now().strftime("%H:%M:%S"),
                    provider=f"ollama:{self.model_name}",
                )
        except Exception:
            pass

    async def _call_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        max_tokens: int = 500,
    ) -> str:
        if not await self._is_ollama_reachable():
            raise ConnectionError("Ollama local offline")
        timeout = httpx.Timeout(3.5, connect=0.35)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.25, "num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return ((data.get("message") or {}).get("content") or "").strip()

    @staticmethod
    def _build_fast_summary(event: AggregatedEvent, active_nodes: int) -> str:
        topo_desc = f"Malha de {active_nodes} sensores ativa." if active_nodes > 1 else "Corredor de Fresnel único ativo."
        return f"{event.summary} {topo_desc}"

    @staticmethod
    def _build_fallback_short(
        occupancy_state: OccupancyState,
        people_count: int,
        peak_occupancy: int,
        active_zone: str,
        confidence: float,
        active_nodes: int,
        recent_events: List[AggregatedEvent],
    ) -> str:
        conf_pct = int(round(confidence * 100))
        zone_label = active_zone.replace("_", " ").title() if active_zone != "NONE" else "Área Monitorada"

        if occupancy_state == "EMPTY" or people_count == 0:
            s1 = f"Sala vazia (0 pessoas) confirmada com {conf_pct}% de confiança em {active_nodes} sensor(es) ESP32-S3 (Pico da sessão: {peak_occupancy})."
        elif occupancy_state == "PRESENCE_STATIC":
            s1 = f"Contagem atual: {people_count} pessoa(s) em presença estática na {zone_label} ({conf_pct}% de confiança, pico: {peak_occupancy})."
        else:
            s1 = f"Contagem atual: {people_count} pessoa(s) em movimento ativo na {zone_label} ({conf_pct}% de confiança, pico: {peak_occupancy})."

        if recent_events:
            s2 = f"Última transição às {recent_events[0].timestamp}: {recent_events[0].summary}"
        else:
            s2 = "Canal RF monitorado continuamente via seleção NBVI de 12 subportadoras."
        return f"{s1} {s2}"

    @staticmethod
    def _build_comprehensive_fallback_report(digest: dict) -> tuple[str, str]:
        occ = digest["estado_ocupacao"]
        p_now = digest["pessoas_na_sala_agora"]
        p_peak = digest["pico_pessoas_sessao"]
        zone = digest["zona_ativa"].replace("_", " ").title() if digest["zona_ativa"] != "NONE" else "Nenhuma (Sala Livre)"
        conf = digest["confianca_pct"]
        n_nodes = digest["nos_esp32_ativos"]
        topo = digest["topologia"]
        mvs = digest["mvs_atual"]
        theta = digest["limiar_theta_on"]
        intensity = digest["intensidade_x"]
        qual = digest["qualidade_calibracao_nbvi"]
        dwell_map: Dict[str, float] = digest.get("tempo_permanencia_zonas_seg", {})
        sensores: Dict[str, dict] = digest.get("sensores", {})
        eventos: List[dict] = digest.get("eventos_recentes", [])

        state_pt = {
            "EMPTY": "Sala Vazia (Repouso RF)",
            "PRESENCE_STATIC": "Presença Estática Retida",
            "MOTION": "Movimento Humano Ativo",
        }.get(occ, occ)

        short_sum = (
            f"Ocupação atual: {p_now} pessoa(s) [{state_pt}] na zona {zone} ({conf}% confiança, "
            f"intensidade {intensity:.1f}x, pico na sessão: {p_peak} pessoa(s))."
        )

        lines = [
            f"# Relatório Analítico Espacial Wi-Fi CSI ({digest['horario']})",
            f"> **Status:** `{state_pt}` | **Contagem Atual:** **`{p_now} pessoa(s)`** | **Pico na Sessão:** `{p_peak} pessoa(s)` | **Confiança:** `{conf}%`",
            "",
            "## 1. Panorama de Ocupação e Contagem",
            f"- **Pessoas Detectadas Agora:** **`{p_now}`** ({state_pt}).",
            f"- **Zona Espacial Ativa:** **`{zone}`** (Topologia `{topo}` com `{n_nodes}` nó(s) ESP32-S3).",
            f"- **Relação Sinal/Limiar (MVS):** `M(t) = {mvs*1e5:.2f}e-5` vs `θ_on = {theta*1e5:.2f}e-5` (**`{intensity:.2f}x`** o limiar de disparo).",
            "",
            "## 2. Distribuição Espacial e Tempo de Permanência por Zona",
        ]

        if dwell_map:
            total_dwell = max(0.1, sum(dwell_map.values()))
            for z_key, secs in sorted(dwell_map.items(), key=lambda kv: kv[1], reverse=True):
                pct = (secs / total_dwell) * 100.0
                z_title = z_key.replace("_", " ").title()
                lines.append(f"- **{z_title}:** `{secs:.1f} s` acumulados (`{pct:.0f}%` da ocupação registrada).")
        else:
            lines.append("- Nenhuma permanência acumulada registrada até o momento nesta sessão.")

        lines.extend([
            "",
            "## 3. Diagnóstico Físico dos Sensores ESP32-S3 (NBVI & Link RF)",
            "| Sensor | Status | Estado | Taxa CSI | RSSI | Intensidade | Qualidade NBVI |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for nid, s in sensores.items():
            st_online = "ONLINE" if s["online"] else "OFFLINE"
            lines.append(
                f"| **{s['nome']}** (`{nid}`) | `{st_online}` | `{s['estado']}` | `{s['taxa_hz']:.0f} Hz` | `{s['rssi_dbm']} dBm` | `{s['intensidade']:.2f}x` | `{s['qualidade_nbvi']}` |"
            )

        lines.extend(["", "## 4. Histórico Recente de Transições e Recomendações"])
        if eventos:
            for ev in eventos[:5]:
                lines.append(f"- **`[{ev['hora']}]`** {ev['resumo']}")
        else:
            lines.append("- Sem transições recentes registradas.")

        rec = (
            "Ambiente calibrado com qualidade excelente (`OPTIMAL`). Limiares Schmitt Trigger operando com imunidade total a saltos de AGC."
            if qual == "OPTIMAL"
            else "Recomenda-se executar a Calibração Automática Guiada (5s + 10s) com a sala vazia para otimizar a seleção das 12 subportadoras NBVI."
        )
        lines.extend(["", f"**Recomendação Técnica:** {rec}"])
        return "\n".join(lines), short_sum

    @staticmethod
    def _build_comprehensive_fallback_answer(question: str, digest: dict) -> tuple[str, str]:
        p_now = digest["pessoas_na_sala_agora"]
        p_peak = digest["pico_pessoas_sessao"]
        occ = digest["estado_ocupacao"]
        zone = digest["zona_ativa"].replace("_", " ").title() if digest["zona_ativa"] != "NONE" else "Sala Livre"
        conf = digest["confianca_pct"]
        intensity = digest["intensidade_x"]
        n_nodes = digest["nos_esp32_ativos"]
        qual = digest["qualidade_calibracao_nbvi"]
        dwell_map: Dict[str, float] = digest.get("tempo_permanencia_zonas_seg", {})

        state_pt = {
            "EMPTY": "Sala Vazia (0 pessoas)",
            "PRESENCE_STATIC": f"{p_now} pessoa(s) em Presença Estática",
            "MOTION": f"{p_now} pessoa(s) em Movimento Ativo",
        }.get(occ, occ)

        short_sum = f"Resposta Analítica: {state_pt} na zona {zone} ({conf}% de confiança, pico da sessão: {p_peak})."

        dwell_summary = (
            ", ".join(f"**{k.replace('_', ' ').title()}** (`{v:.1f}s`)" for k, v in dwell_map.items())
            if dwell_map
            else "Sem tempo acumulado registrado ainda"
        )

        md = "\n".join([
            f"### Análise Contextual — *\"{question}\"*",
            "",
            f"- **Contagem e Estado Atual:** **`{p_now} pessoa(s)`** no ambiente (`{state_pt}`) com **`{conf}%`** de confiança.",
            f"- **Pico de Ocupação na Sessão:** **`{p_peak} pessoa(s)`**.",
            f"- **Localização / Zona Ativa:** **`{zone}`** (monitorada por `{n_nodes}` sensor(es) ESP32-S3, intensidade MVS `{intensity:.2f}x`).",
            f"- **Permanência Acumulada por Zonas:** {dwell_summary}.",
            f"- **Qualidade do Canal CSI (NBVI):** `{qual}` (Normalização $L_2$ anti-AGC + Filtro de Hampel e Butterworth 10 Hz ativos).",
        ])
        return md, short_sum
