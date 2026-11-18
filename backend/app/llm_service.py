from datetime import datetime
import json
import logging
from typing import List, Optional
import httpx

from .config import settings
from .models import LLMResponse, LLMStatusResponse

logger = logging.getLogger("wifi_spatial.llm")


class LLMService:
    """
    Serviço Assíncrono de Inteligência Analítica Espacial 100% Offline via Ollama Local.
    Comunica-se com http://localhost:11434 (/api/chat, /api/generate e /api/tags) usando httpx.AsyncClient,
    suportando qwen2.5:7b, qwen3:8b, qwen3:4b, gemma3:4b, llama3.2:3b e qualquer modelo instalado localmente,
    com fallback estatístico imediato caso o serviço do Ollama esteja fechado.
    """

    def __init__(self) -> None:
        self.base_url: str = settings.OLLAMA_BASE_URL
        self.model_name: str = settings.OLLAMA_MODEL
        self.timeout_sec: float = settings.OLLAMA_TIMEOUT_SEC

    @staticmethod
    def _friendly_model_label(model_tag: str) -> str:
        low = model_tag.lower()
        if "qwen3" in low:
            return f"Qwen 3 ({model_tag})"
        if "qwen2.5" in low or "qwen" in low:
            return f"Qwen 2.5 ({model_tag})" if ":" in model_tag else "Qwen 2.5"
        if "gemma3" in low or "gemma" in low:
            return f"Gemma 3 ({model_tag})" if ":" in model_tag else "Gemma 3"
        if "llama3" in low or "llama" in low:
            return f"Llama 3.2 ({model_tag})" if ":" in model_tag else "Llama 3.2"
        return model_tag

    async def get_status(self, preferred_model: Optional[str] = None) -> LLMStatusResponse:
        """
        Verifica se o servidor local do Ollama está rodando em http://localhost:11434/api/tags
        e lista automaticamente os modelos instalados na máquina.
        """
        if preferred_model and preferred_model.strip():
            self.model_name = preferred_model.strip()

        recommended = list(settings.RECOMMENDED_OLLAMA_MODELS)
        try:
            timeout = httpx.Timeout(2.0, connect=0.6)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()

            installed: List[str] = [
                m.get("name", "")
                for m in data.get("models", [])
                if isinstance(m, dict) and m.get("name")
            ]

            # Se o usuário tiver modelos instalados e o modelo padrão não estiver entre eles,
            # seleciona automaticamente o primeiro modelo instalado caso preferred_model não tenha sido forçado
            if installed and not preferred_model and self.model_name not in installed:
                self.model_name = installed[0]

            available: List[str] = []
            for m in installed + recommended:
                if m and m not in available:
                    available.append(m)

            friendly = self._friendly_model_label(self.model_name)
            return LLMStatusResponse(
                online=True,
                base_url=self.base_url,
                active_model=self.model_name,
                installed_models=installed,
                available_models=available,
                badge_text=f"LLM Local Online ({friendly})",
            )
        except Exception as exc:
            logger.debug(f"Ollama local indisponível em {self.base_url}/api/tags: {exc}")
            available = [self.model_name] + [m for m in recommended if m != self.model_name]
            return LLMStatusResponse(
                online=False,
                base_url=self.base_url,
                active_model=self.model_name,
                installed_models=[],
                available_models=available,
                badge_text="Ollama Offline (Fallback Estatístico)",
            )

    def _compact_digest_for_local_llm(self, digest: dict) -> dict:
        compact = dict(digest)
        timeline = compact.get("event_timeline", [])
        if len(timeline) > 15:
            compact["event_timeline"] = timeline[-15:]
        return compact

    def _build_concise_system_prompt(self, digest: dict) -> str:
        topo = digest.get("topology_info") or {}
        cov = digest.get("coverage_info") or {}
        cal_mode = digest.get("calibration_mode", "nominal")
        mode_code = topo.get("mode_code", "SINGLE_LINK_CORRIDOR")
        active_nodes = topo.get("active_nodes", 1)
        mode_title = topo.get("mode_title", "Zona de Abrangência Elíptica")
        area_m2 = cov.get("active_area_m2", 11.5)
        dist_m = cov.get("avg_tx_rx_distance_m", 3.2)

        cal_desc = (
            "Calibrado com Plateia na Sala (Mediana + Percentil 20)"
            if cal_mode == "crowd_compensated"
            else "Calibrado com Sala Vazia"
            if cal_mode == "empty_room"
            else "Baseline Nominal"
        )

        return (
            "Você é um assistente especialista em Monitoramento Espacial Wi-Fi CSI (sem câmeras).\n"
            "Responda SEMPRE em Português do Brasil, de forma direta, técnica e formatada em Markdown.\n"
            f"Topologia Atual: {mode_title} ({active_nodes} ESP32 online, modo {mode_code}).\n"
            f"Zona de Abrangência Auto-Mapeada: ~{area_m2} m² (Distância TX-RX: {dist_m} m).\n"
            f"Estado de Calibração: {cal_desc}.\n"
            "Mesmo com 1 único ESP32, o sistema conta múltiplas pessoas (0, 1, 2, 3+) via Entropia Espectral (subcarrier_diversity).\n"
            "Use apenas os números reais fornecidos no JSON do usuário."
        )

    async def _call_ollama(self, system_prompt: str, user_prompt: str, target_model: str) -> str:
        timeout = httpx.Timeout(self.timeout_sec, connect=0.8)
        async with httpx.AsyncClient(timeout=timeout) as client:
            chat_payload = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.25,
                    "num_predict": 700,
                },
            }
            resp = await client.post(f"{self.base_url}/api/chat", json=chat_payload)
            if resp.status_code == 404:
                gen_payload = {
                    "model": target_model,
                    "system": system_prompt,
                    "prompt": user_prompt,
                    "stream": False,
                    "options": {"temperature": 0.25, "num_predict": 700},
                }
                resp = await client.post(f"{self.base_url}/api/generate", json=gen_payload)
                resp.raise_for_status()
                return (resp.json().get("response") or "").strip()

            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message") or {}
            return (msg.get("content") or data.get("response") or "").strip()

    async def generate_report(
        self,
        digest: dict,
        focus_area: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> LLMResponse:
        now_str = datetime.now().strftime("%H:%M:%S")
        events_count = len(digest.get("event_timeline", []))
        target_model = (model_override or self.model_name or settings.OLLAMA_MODEL).strip()
        self.model_name = target_model

        compact_digest = self._compact_digest_for_local_llm(digest)
        system_prompt = self._build_concise_system_prompt(compact_digest)
        focus_line = f"\nFoco adicional solicitado: {focus_area}\n" if focus_area else ""

        user_prompt = (
            "Com base nos dados agregados abaixo, gere um **Relatório Inteligente da Área de Abrangência** em Markdown contendo:\n"
            "1. **Diagnóstico de Abrangência Auto-Mapeada e Calibração**\n"
            "2. **Resumo Executivo de Contagem de Pessoas**\n"
            "3. **Análise Espectral dos Sensores ESP32 (Subcarrier Diversity & Energia)**\n"
            "4. **Recomendações Operacionais**\n"
            f"{focus_line}\n"
            f"JSON da Sessão:\n```json\n{json.dumps(compact_digest, ensure_ascii=False)}\n```"
        )

        try:
            content = await self._call_ollama(system_prompt, user_prompt, target_model)
            if content:
                return LLMResponse(
                    status="ok",
                    provider="ollama-local",
                    model=f"Ollama ({target_model})",
                    content=content,
                    generated_at=now_str,
                    events_analyzed=events_count,
                )
        except Exception as exc:
            logger.info(f"Ollama local indisponível ou modelo '{target_model}' não carregado ({exc}). Usando Fallback Estatístico.")

        fallback_md = self._build_local_fallback_report(digest, focus_area)
        return LLMResponse(
            status="ok",
            provider="local-statistical-fallback",
            model="Fallback Estatístico Local (Ollama Offline)",
            content=fallback_md,
            generated_at=now_str,
            events_analyzed=events_count,
        )

    async def ask_question(
        self,
        question: str,
        digest: dict,
        model_override: Optional[str] = None,
    ) -> LLMResponse:
        now_str = datetime.now().strftime("%H:%M:%S")
        events_count = len(digest.get("event_timeline", []))
        target_model = (model_override or self.model_name or settings.OLLAMA_MODEL).strip()
        self.model_name = target_model

        compact_digest = self._compact_digest_for_local_llm(digest)
        system_prompt = self._build_concise_system_prompt(compact_digest)
        user_prompt = (
            f"Pergunta do Operador: \"{question}\"\n\n"
            "Responda de forma objetiva em Markdown com base nos dados reais abaixo:\n"
            f"```json\n{json.dumps(compact_digest, ensure_ascii=False)}\n```"
        )

        try:
            content = await self._call_ollama(system_prompt, user_prompt, target_model)
            if content:
                return LLMResponse(
                    status="ok",
                    provider="ollama-local",
                    model=f"Ollama ({target_model})",
                    content=content,
                    generated_at=now_str,
                    events_analyzed=events_count,
                )
        except Exception as exc:
            logger.info(f"Ollama local indisponível para pergunta ({exc}). Usando Fallback Estatístico.")

        fallback_answer = self._build_local_fallback_answer(question, digest)
        return LLMResponse(
            status="ok",
            provider="local-statistical-fallback",
            model="Fallback Estatístico Local (Ollama Offline)",
            content=fallback_answer,
            generated_at=now_str,
            events_analyzed=events_count,
        )

    def _build_local_fallback_report(self, digest: dict, focus_area: Optional[str] = None) -> str:
        topo = digest.get("topology_info") or {}
        cov = digest.get("coverage_info") or {}
        cal_mode = digest.get("calibration_mode", "nominal")
        active_nodes = topo.get("active_nodes", 1)
        mode_title = topo.get("mode_title", "Zona de Abrangência Elíptica")
        area_m2 = cov.get("active_area_m2", 11.5)
        dist_m = cov.get("avg_tx_rx_distance_m", 3.2)

        curr_occ = digest.get("current_occupancy", 0)
        peak_occ = digest.get("peak_occupancy", 0)
        nodes_telemetry = digest.get("rf_nodes_telemetry", {})
        timeline = digest.get("event_timeline", [])

        cal_label = (
            "Modo Plateia / Pessoas na Sala (Mediana + Percentil 20)"
            if cal_mode == "crowd_compensated"
            else "Sala Vazia (Média + Desvio Padrão)"
            if cal_mode == "empty_room"
            else "Baseline Nominal Automático"
        )

        lines = [
            "# Relatório Inteligente da Área de Abrangência (Wi-Fi CSI)",
            f"> **Topologia:** `{mode_title}` (`{active_nodes}` ESP32 online) | "
            f"**Abrangência Ativa:** `~{area_m2} m²` (Distância TX-RX: `{dist_m} m`)",
            "",
            "## 1. Auto-Mapeamento RF e Estado de Calibração",
            f"- **Área Efetiva Monitorada:** `~{area_m2} m²` (Elipse de Fresnel + raio lateral de `1.8 m`).",
            f"- **Distância Física Estimada (Log-Distance Path Loss):** `{dist_m} m` entre o Roteador (TX) e o(s) ESP32 (RX).",
            f"- **Perfil de Calibração Ativo:** **{cal_label}**.",
            "",
            "## 2. Contagem de Pessoas e Ocupação da Zona",
            f"- **Pessoas na Zona de Abrangência Agora:** **`{curr_occ}` pessoa(s)**.",
            f"- **Pico Registrado na Sessão:** **`{peak_occ}` pessoa(s)** simultânea(s).",
            "",
            "## 3. Telemetria Espectral CSI & Descorrelação Multi-Pessoa",
        ]

        for nid, nd in nodes_telemetry.items():
            state_badge = "ONLINE" if nd.get("online") else "OFFLINE"
            lines.append(
                f"- **{nd.get('name', nid)} (`{nid}` — {state_badge}):** "
                f"Distância TX-RX `{nd.get('estimated_distance_m', dist_m)} m` (`{nd.get('current_rssi_dbm', -50)} dBm`) | "
                f"Perturbação: `{nd.get('current_perturbation', 0.0)*100:.1f}%` | "
                f"Diversidade Espectral (`subcarrier_diversity`): `{nd.get('subcarrier_diversity', 0.0)*100:.0f}%` | "
                f"Razão de Energia: `{nd.get('energy_ratio', 0.0):.2f}x`."
            )

        if timeline:
            latest = timeline[-1]
            lines.append(f"- **Último Evento (`{latest.get('timestamp')}`):** *{latest.get('summary')}*.")

        if focus_area:
            lines.append(f"\n- **Foco Solicitado:** {focus_area}.")

        lines.extend([
            "",
            "---",
            f"*Nota: Relatório gerado pelo Fallback Estatístico Local 100% Offline. Para inferência neural via Ollama, execute `ollama run {self.model_name}`.*",
        ])
        return "\n".join(lines)

    def _build_local_fallback_answer(self, question: str, digest: dict) -> str:
        topo = digest.get("topology_info") or {}
        cov = digest.get("coverage_info") or {}
        mode_title = topo.get("mode_title", "Zona de Abrangência Elíptica")
        active_nodes = topo.get("active_nodes", 1)
        area_m2 = cov.get("active_area_m2", 11.5)
        dist_m = cov.get("avg_tx_rx_distance_m", 3.2)
        curr_occ = digest.get("current_occupancy", 0)
        peak_occ = digest.get("peak_occupancy", 0)
        timeline = digest.get("event_timeline", [])
        recent_summary = timeline[-1].get("summary", "Monitoramento ativo") if timeline else "Sessão ativa"

        return (
            f"### Análise Instantânea — **{mode_title}**\n\n"
            f"- **Contagem Atual na Zona de Abrangência:** **`{curr_occ}` pessoa(s)** (Pico na sessão: `{peak_occ}`).\n"
            f"- **Área de Abrangência Auto-Mapeada:** `~{area_m2} m²` (Distância TX-RX estimada: `{dist_m} m` com `{active_nodes}` ESP32 online).\n"
            f"- **Status Recente:** {recent_summary}."
        )
