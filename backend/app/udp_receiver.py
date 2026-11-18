import asyncio
import json
import logging
import struct
import time
from typing import Callable, Dict, List, Optional

from .config import settings
from .models import CSIPacket

logger = logging.getLogger("wifi_spatial.udp")


class CSIDatagramProtocol(asyncio.DatagramProtocol):
    """
    Protocolo UDP Assíncrono (asyncio.DatagramProtocol) na porta 5555
    para recepção de pacotes CSI (JSON ou Binário Compactado) vindos dos módulos ESP32.
    """

    def __init__(self, on_packet_callback: Callable[[CSIPacket, str], None]):
        super().__init__()
        self.on_packet_callback = on_packet_callback
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]
        sockname = transport.get_extra_info("sockname")
        logger.info(f"Servidor UDP CSI escutando em {sockname}")

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        packet = parse_csi_payload(data)
        if packet is not None:
            self.on_packet_callback(packet, "hardware")

    def error_received(self, exc: Exception) -> None:
        logger.warning(f"Erro no socket UDP CSI: {exc}")

    def connection_lost(self, exc: Optional[Exception]) -> None:
        logger.info("Socket UDP CSI encerrado.")


def parse_csi_payload(data: bytes) -> Optional[CSIPacket]:
    """
    Decodifica pacotes CSI recebidos via UDP:
    1. Formato JSON (ex: {"node_id":"esp_1","rssi":-48,"csi":[...]} ou com "csi_raw" e "timestamp")
    2. Formato Binário Compactado: Header 'CSI1' (4B) + node_idx (1B) + rssi (1B int8) + count (2B uint16) + int8[]
    """
    if not data:
        return None

    try:
        # Verifica se é pacote binário compactado com magic header b"CSI1"
        if len(data) >= 8 and data[:4] == b"CSI1":
            node_idx, rssi, byte_count = struct.unpack("<bbH", data[4:8])
            payload_bytes = data[8 : 8 + byte_count]
            csi_ints = list(struct.unpack(f"<{len(payload_bytes)}b", payload_bytes))
            return CSIPacket(
                node_id=f"esp_{node_idx}",
                timestamp=time.time(),
                rssi=int(rssi),
                csi_raw=csi_ints,
            )

        # Caso contrário, decodifica como JSON UTF-8
        text = data.decode("utf-8", errors="ignore").strip()
        if not text.startswith("{"):
            return None

        raw_dict = json.loads(text)
        return CSIPacket.model_validate(raw_dict)
    except Exception as exc:
        logger.debug(f"Pacote UDP descartado (formato inválido): {exc}")
        return None


class UDPReceiverServer:
    """
    Gerenciador do servidor UDP assíncrono na porta 5555 e detector de atividade (heartbeat 3.0s)
    dos módulos ESP32.
    """

    def __init__(self, on_packet_callback: Callable[[CSIPacket, str], None]):
        self.on_packet_callback = on_packet_callback
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.protocol: Optional[CSIDatagramProtocol] = None
        self.bound_port: int = settings.UDP_PORT
        self.last_seen_by_node: Dict[str, float] = {}

    def record_heartbeat(self, node_id: str, timestamp: Optional[float] = None) -> None:
        """Registra o instante do último pacote CSI válido recebido de um módulo ESP32."""
        self.last_seen_by_node[node_id] = timestamp if timestamp is not None else time.time()

    def is_node_online(self, node_id: str, now: Optional[float] = None) -> bool:
        """
        Considera um nó ESP32 como ATIVO (online) se ele enviou pacotes CSI
        nos últimos 3.0 segundos (settings.NODE_ONLINE_TIMEOUT_SEC).
        """
        if now is None:
            now = time.time()
        last_ts = self.last_seen_by_node.get(node_id, 0.0)
        return last_ts > 0.0 and (now - last_ts) <= settings.NODE_ONLINE_TIMEOUT_SEC

    def get_active_node_ids(self, candidate_ids: List[str], now: Optional[float] = None) -> List[str]:
        """Retorna a lista de node_ids que estão online nos últimos 3.0 segundos."""
        if now is None:
            now = time.time()
        return [nid for nid in candidate_ids if self.is_node_online(nid, now=now)]

    def clear_heartbeats(self, keep_node_ids: Optional[List[str]] = None) -> None:
        """Limpa imediatamente o status online de nós desativados (ex: troca de topologia no simulador)."""
        if keep_node_ids is None:
            self.last_seen_by_node.clear()
            return
        keep_set = set(keep_node_ids)
        for nid in list(self.last_seen_by_node.keys()):
            if nid not in keep_set:
                del self.last_seen_by_node[nid]

    async def start(self, host: str = settings.UDP_HOST, port: int = settings.UDP_PORT) -> None:
        loop = asyncio.get_running_loop()
        try:
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: CSIDatagramProtocol(self._handle_packet),
                local_addr=(host, port),
            )
            self.transport = transport  # type: ignore[assignment]
            self.protocol = protocol  # type: ignore[assignment]
            self.bound_port = port
        except OSError as exc:
            logger.warning(
                f"Porta UDP {port} ocupada ({exc}). Usando porta efêmera de fallback."
            )
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: CSIDatagramProtocol(self._handle_packet),
                local_addr=(host, 0),
            )
            self.transport = transport  # type: ignore[assignment]
            self.protocol = protocol  # type: ignore[assignment]
            sockname = transport.get_extra_info("sockname")
            if sockname:
                self.bound_port = sockname[1]

    def _handle_packet(self, packet: CSIPacket, source: str) -> None:
        self.record_heartbeat(packet.node_id, packet.timestamp)
        self.on_packet_callback(packet, source)

    async def stop(self) -> None:
        if self.transport is not None:
            self.transport.close()
            self.transport = None
