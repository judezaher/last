"""IOC detection engine.

This module receives PacketEvent objects and checks them against:
- whitelisted IPs
- blacklisted IPs
- blacklisted ports

Phase 2 improvement:
- clearer alert details
- optional decision debugging for whitelist suppression
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ioc_loader import IOCCache
from packet_capture import PacketEvent

LOGGER = logging.getLogger("ids.ioc_engine")


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _port_text(port: Optional[int]) -> str:
    return "-" if port is None else str(port)


@dataclass
class AlertDeduplicator:
    """Suppress repeated alerts for a short time window.

    Important for threading:
    - main.py creates ONE AlertDeduplicator object.
    - The same object is passed to all IOC worker threads.
    - The lock makes allow() thread-safe.

    So if worker-0 alerts now, worker-1 will also see that history and will
    suppress a duplicate alert inside hold_seconds.
    """

    hold_seconds: int = 30
    _last_seen: dict[tuple, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def allow(self, key: tuple) -> bool:
        now = time.time()
        with self._lock:
            previous = self._last_seen.get(key, 0)
            if now - previous < self.hold_seconds:
                return False
            self._last_seen[key] = now
            self._cleanup_old_entries(now)
            return True

    def _cleanup_old_entries(self, now: float) -> None:
        """Prevent the deduplication dictionary from growing forever."""
        max_age = max(self.hold_seconds * 4, 60)
        old_keys = [key for key, last_seen in self._last_seen.items() if now - last_seen > max_age]
        for key in old_keys:
            self._last_seen.pop(key, None)


def detect_ioc(event: PacketEvent, ioc_cache: IOCCache, debug_decisions: bool = False) -> Optional[dict]:
    """Return an alert dict if a packet matches an IOC rule."""
    iocs = ioc_cache.snapshot()

    # Whitelist has priority in this starter version.
    # Important: do not put your victim/protected server here unless you really
    # want to suppress alerts involving that IP.
    if event.src_ip in iocs.whitelist_ips or event.dst_ip in iocs.whitelist_ips:
        if debug_decisions:
            LOGGER.info("SUPPRESSED_BY_WHITELIST %s", event.summary())
        return None

    reasons: list[str] = []

    if event.src_ip in iocs.blacklist_ips:
        reasons.append(f"SRC_IP_BLACKLISTED:{event.src_ip}")
    if event.dst_ip in iocs.blacklist_ips:
        reasons.append(f"DST_IP_BLACKLISTED:{event.dst_ip}")

    if event.src_port is not None and event.src_port in iocs.blacklist_ports:
        reasons.append(f"SRC_PORT_BLACKLISTED:{event.src_port}")
    if event.dst_port is not None and event.dst_port in iocs.blacklist_ports:
        reasons.append(f"DST_PORT_BLACKLISTED:{event.dst_port}")

    if not reasons:
        return None

    return {
        "alert_type": "IOC_MATCH",
        "severity": "HIGH",
        "created_at": time.time(),
        "created_at_utc": _utc_now(),
        "reasons": reasons,
        "packet": event.to_dict(),
        "summary": event.summary(),
    }


def build_dedup_key(event: PacketEvent, alert: dict, mode: str) -> tuple:
    """Build the key used to decide whether an alert is a duplicate.

    Modes:
    - source: suppress all repeated alerts from the same source IP.
    - source-reason: suppress repeated alerts from the same source IP for the same reason.
    - exact: suppress only alerts with the same src, dst, protocol, ports, and reasons.
    """
    reasons = tuple(sorted(alert["reasons"]))

    if mode == "source":
        return ("source", event.src_ip)

    if mode == "source-reason":
        return ("source-reason", event.src_ip, reasons)

    return (
        "exact",
        event.src_ip,
        event.dst_ip,
        event.protocol,
        event.src_port,
        event.dst_port,
        reasons,
    )


def worker_loop(
    worker_id: int,
    packet_queue: "queue.Queue[PacketEvent]",
    alert_queue: "queue.Queue[dict]",
    block_queue: "queue.Queue[dict]",
    ioc_cache: IOCCache,
    stop_event: threading.Event,
    deduplicator: AlertDeduplicator,
    dedup_mode: str,
    debug_decisions: bool = False,
    ips_mode: str = "alert-only",
) -> None:
    LOGGER.info("Worker %d started", worker_id)

    while not stop_event.is_set() or not packet_queue.empty():
        try:
            event = packet_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            alert = detect_ioc(event, ioc_cache, debug_decisions=debug_decisions)
            if alert:
                key = build_dedup_key(event, alert, dedup_mode)
                if deduplicator.allow(key):
                    alert_queue.put_nowait(alert)
                    if ips_mode != "alert-only":
                        try:
                            block_queue.put_nowait(alert)
                        except queue.Full:
                            LOGGER.warning("IPS block queue full; block request dropped for packet: %s", event.summary())
                    LOGGER.warning(
                        "ALERT worker=%d %s %s:%s -> %s:%s len=%d reasons=%s",
                        worker_id,
                        event.protocol,
                        event.src_ip,
                        _port_text(event.src_port),
                        event.dst_ip,
                        _port_text(event.dst_port),
                        event.length,
                        alert["reasons"],
                    )
                elif debug_decisions:
                    LOGGER.info("DUPLICATE_SUPPRESSED worker=%d %s reasons=%s", worker_id, event.summary(), alert["reasons"])
        except queue.Full:
            LOGGER.warning("Alert queue full; alert dropped for packet: %s", event.summary())
        except Exception:
            LOGGER.exception("Worker %d failed to process packet", worker_id)
        finally:
            packet_queue.task_done()

    LOGGER.info("Worker %d stopped", worker_id)
