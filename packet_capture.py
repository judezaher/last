"""Packet capture module.

This layer captures live packets from an interface, extracts a compact event,
and broadcasts that event to one or more queues:
- IOC queue for stateless blacklist/whitelist checks
- ML queue for Phase 4 flow-based deep-learning checks
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from scapy.all import ICMP, IP, TCP, UDP, sniff  # type: ignore

LOGGER = logging.getLogger("ids.packet_capture")


def _port_text(port: Optional[int]) -> str:
    return "-" if port is None else str(port)


def timestamp_to_utc(ts: float) -> str:
    """Convert a Unix timestamp to readable UTC time."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class PacketEvent:
    timestamp: float
    src_ip: str
    dst_ip: str
    protocol: str
    src_port: Optional[int]
    dst_port: Optional[int]
    length: int

    # Phase 4 fields used by CICIDS flow extraction.
    ip_header_length: int = 0
    transport_header_length: int = 0
    payload_length: int = 0
    tcp_flags: str = ""
    tcp_window: int = 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["timestamp_utc"] = timestamp_to_utc(self.timestamp)
        return data

    def summary(self) -> str:
        return (
            f"{self.protocol} "
            f"{self.src_ip}:{_port_text(self.src_port)} -> "
            f"{self.dst_ip}:{_port_text(self.dst_port)} "
            f"len={self.length}"
        )


class PacketCapture:
    """Capture packets and place PacketEvent objects into queues."""

    def __init__(
        self,
        iface: str,
        output_queues: list["queue.Queue[PacketEvent]"],
        stop_event: threading.Event,
        bpf_filter: str = "ip",
        debug_packets: bool = False,
        debug_every: int = 1,
    ) -> None:
        self.iface = iface
        self.output_queues = output_queues
        self.stop_event = stop_event
        self.bpf_filter = bpf_filter
        self.debug_packets = debug_packets
        self.debug_every = max(1, int(debug_every))
        self._packet_counter = 0
        self._dropped_counter = 0

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self._run, name="packet-capture", daemon=True)
        thread.start()
        return thread

    def _run(self) -> None:
        LOGGER.info("Starting packet capture on iface=%s filter=%r", self.iface, self.bpf_filter)
        try:
            sniff(
                iface=self.iface,
                filter=self.bpf_filter,
                prn=self._handle_packet,
                store=False,
                stop_filter=lambda _pkt: self.stop_event.is_set(),
            )
        except Exception:
            LOGGER.exception("Packet capture stopped because of an error")
            self.stop_event.set()
        finally:
            LOGGER.info("Packet capture stopped dropped_events=%d", self._dropped_counter)

    def _handle_packet(self, pkt) -> None:  # scapy packet type kept loose for beginner readability
        event = self._packet_to_event(pkt)
        if event is None:
            return

        self._packet_counter += 1

        if self.debug_packets and self._packet_counter % self.debug_every == 0:
            LOGGER.info("PACKET #%d %s", self._packet_counter, event.summary())

        for out_queue in self.output_queues:
            try:
                out_queue.put_nowait(event)
            except queue.Full:
                self._dropped_counter += 1
                LOGGER.warning("Packet queue full; dropped event: %s", event.summary())

    @staticmethod
    def _packet_to_event(pkt) -> Optional[PacketEvent]:
        if IP not in pkt:
            return None

        ip = pkt[IP]
        protocol = "OTHER"
        src_port: Optional[int] = None
        dst_port: Optional[int] = None
        tcp_flags = ""
        tcp_window = 0
        transport_header_length = 0

        if TCP in pkt:
            tcp = pkt[TCP]
            protocol = "TCP"
            src_port = int(tcp.sport)
            dst_port = int(tcp.dport)
            tcp_flags = str(tcp.flags)
            tcp_window = int(tcp.window)
            transport_header_length = int(tcp.dataofs or 0) * 4
        elif UDP in pkt:
            udp = pkt[UDP]
            protocol = "UDP"
            src_port = int(udp.sport)
            dst_port = int(udp.dport)
            transport_header_length = 8
        elif ICMP in pkt:
            protocol = "ICMP"
            transport_header_length = 8

        ip_header_length = int(getattr(ip, "ihl", 0) or 0) * 4
        total_length = int(getattr(ip, "len", 0) or len(pkt))
        payload_length = max(0, total_length - ip_header_length - transport_header_length)

        # Scapy's pkt.time can be Decimal in some versions; convert to float.
        timestamp = float(getattr(pkt, "time", time.time()))

        return PacketEvent(
            timestamp=timestamp,
            src_ip=str(ip.src),
            dst_ip=str(ip.dst),
            protocol=protocol,
            src_port=src_port,
            dst_port=dst_port,
            length=total_length,
            ip_header_length=ip_header_length,
            transport_header_length=transport_header_length,
            payload_length=payload_length,
            tcp_flags=tcp_flags,
            tcp_window=tcp_window,
        )
