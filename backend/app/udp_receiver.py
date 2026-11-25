import asyncio
import json
import logging
import struct
import time
from typing import Callable, Dict, List, Optional

from .config import settings
from .models import CSIPacket

logger = logging.getLogger("wifi_spatial.udp")

BINARY_HEADER_FMT = "<HBB6sQbbBH"
BINARY_HEADER_SIZE = struct.calcsize(BINARY_HEADER_FMT)  # 23 bytes
CSI_PACKET_MAGIC = 0xC510


def parse_csi_payload(data: bytes) -> Optional[CSIPacket]:
    """
    Decodifica pacotes CSI recebidos via UDP (Zero-Blocking):
    1. Protocolo Binário Compacto v1 (magic 0xC510, little-endian, 23B header + int8[] I/Q).
    2. Protocolo Binário Legado ('CSI1').
    3. Formato JSON UTF-8.
    """
    if not data:
        return None

    try:
        # 1. Protocolo Binário Compacto (0xC510) do Firmware ESP32-S3
        if len(data) >= BINARY_HEADER_SIZE:
            magic = struct.unpack_from("<H", data, 0)[0]
            if magic == CSI_PACKET_MAGIC:
                (
                    _magic,
                    _version,
                    node_num,
                    src_mac_bytes,
                    timestamp_us,
                    rssi,
                    noise_floor,
                    channel,
                    csi_len,
                ) = struct.unpack_from(BINARY_HEADER_FMT, data, 0)

                end_idx = BINARY_HEADER_SIZE + csi_len
                if len(data) < end_idx or csi_len < 4:
                    return None

                csi_ints = list(struct.unpack_from(f"<{csi_len}b", data, BINARY_HEADER_SIZE))
                mac_str = ":".join(f"{b:02x}" for b in src_mac_bytes)
                return CSIPacket(
                    node_id=f"esp_{node_num}",
                    timestamp=time.time(),
                    timestamp_us=int(timestamp_us),
                    src_mac=mac_str,
                    rssi=int(rssi),
                    noise_floor=int(noise_floor),
                    channel=int(channel),
                    csi_raw=csi_ints,
                )

        # 2. Formato Binário Legado ('CSI1')
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

        # 3. Fallback JSON UTF-8
        text = data.decode("utf-8", errors="ignore").strip()
        if not text.startswith("{"):
            return None

        raw_dict = json.loads(text)
        return CSIPacket.model_validate(raw_dict)
    except Exception as exc:
        logger.debug(f"Pacote UDP inválido descartado: {exc}")
        return None


class CSIDatagramProtocol(asyncio.DatagramProtocol):
    """
    Protocolo UDP Assíncrono Zero-Blocking na porta 5555.
    Decodifica o pacote, registra o endpoint (IP, porta) do ESP32-S3 para estímulo ativo
    de downlink HT20 via roteador Cisco Linksys E1200 e empilha no ring buffer.
    """

    def __init__(
        self,
        on_packet_callback: Callable[[CSIPacket, str], None],
        on_peer_addr_callback: Optional[Callable[[str, tuple], None]] = None,
    ):
        super().__init__()
        self.on_packet_callback = on_packet_callback
        self.on_peer_addr_callback = on_peer_addr_callback
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]
        sockname = transport.get_extra_info("sockname")
        logger.info(f"Receptor UDP CSI (Binário 0xC510 / JSON) escutando em {sockname}")

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        # Ignora pacotes curtos de probe/ping ("PING")
        if len(data) <= 12:
            return
        packet = parse_csi_payload(data)
        if packet is not None:
            if self.on_peer_addr_callback is not None and addr:
                self.on_peer_addr_callback(packet.node_id, addr)
            self.on_packet_callback(packet, "hardware")

    def error_received(self, exc: Exception) -> None:
        logger.warning(f"Erro no socket UDP CSI: {exc}")

    def connection_lost(self, exc: Optional[Exception]) -> None:
        logger.info("Socket UDP CSI encerrado.")


class UDPReceiverServer:
    """
    Gerenciador do servidor UDP assíncrono, monitor de heartbeat e gerador de estímulo
    downlink UDP (40 Hz) para os nós ESP32-S3 através do roteador Cisco Linksys E1200
    (evita que o rate-limit ICMP de 1 Hz do roteador reduza a taxa de amostragem CSI).
    """

    def __init__(self, on_packet_callback: Callable[[CSIPacket, str], None]):
        self.on_packet_callback = on_packet_callback
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.protocol: Optional[CSIDatagramProtocol] = None
        self.bound_port: int = settings.UDP_PORT
        self.last_seen_by_node: Dict[str, float] = {}
        self.peer_addrs_by_node: Dict[str, tuple] = {}
        self._stimulus_task: Optional[asyncio.Task] = None
        self._running: bool = False

    def record_heartbeat(self, node_id: str, timestamp: Optional[float] = None) -> None:
        self.last_seen_by_node[node_id] = timestamp if timestamp is not None else time.time()

    def record_peer_addr(self, node_id: str, addr: tuple) -> None:
        self.peer_addrs_by_node[node_id] = addr

    def is_node_online(self, node_id: str, now: Optional[float] = None) -> bool:
        if now is None:
            now = time.time()
        last_ts = self.last_seen_by_node.get(node_id, 0.0)
        return last_ts > 0.0 and (now - last_ts) <= settings.NODE_ONLINE_TIMEOUT_SEC

    def clear_heartbeats(self, keep_node_ids: Optional[List[str]] = None) -> None:
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
                lambda: CSIDatagramProtocol(self._handle_packet, self.record_peer_addr),
                local_addr=(host, port),
            )
            self.transport = transport  # type: ignore[assignment]
            self.protocol = protocol  # type: ignore[assignment]
            self.bound_port = port
        except OSError as exc:
            logger.warning(f"Porta UDP {port} ocupada ({exc}). Alocando porta efêmera.")
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: CSIDatagramProtocol(self._handle_packet, self.record_peer_addr),
                local_addr=(host, 0),
            )
            self.transport = transport  # type: ignore[assignment]
            self.protocol = protocol  # type: ignore[assignment]
            sockname = transport.get_extra_info("sockname")
            if sockname:
                self.bound_port = sockname[1]

        self._running = True
        self._stimulus_task = asyncio.create_task(self._downlink_stimulus_loop())

    async def _downlink_stimulus_loop(self) -> None:
        """
        Envia pacotes UDP leves de 32 bytes a 40 Hz para cada ESP32-S3 ativo.
        Ao passar pelo roteador Cisco Linksys E1200 (AP -> STA), cada pacote gera um
        quadro 802.11n HT20 OFDM válido (sig_len >= 60) com MAC do roteador, garantindo
        taxa CSI estável de 40+ Hz mesmo quando o roteador limita ICMP a 1 Hz.
        """
        probe_pkt = b"CSI_HT20_STIMULUS_FRAME_PAD_32B!"
        interval = 1.0 / 40.0
        while self._running:
            try:
                now = time.time()
                if self.transport is not None and self.peer_addrs_by_node:
                    for nid, addr in list(self.peer_addrs_by_node.items()):
                        last_ts = self.last_seen_by_node.get(nid, 0.0)
                        if (now - last_ts) <= 10.0:
                            self.transport.sendto(probe_pkt, addr)
            except Exception:
                pass
            await asyncio.sleep(interval)

    def _handle_packet(self, packet: CSIPacket, source: str) -> None:
        self.record_heartbeat(packet.node_id, packet.timestamp)
        self.on_packet_callback(packet, source)

    async def stop(self) -> None:
        self._running = False
        if self._stimulus_task is not None:
            self._stimulus_task.cancel()
            try:
                await self._stimulus_task
            except asyncio.CancelledError:
                pass
            self._stimulus_task = None
        if self.transport is not None:
            self.transport.close()
            self.transport = None
