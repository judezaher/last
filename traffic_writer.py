"""Traffic log writer thread.

This module writes every observed packet summary to a JSONL file so the GUI
Dashboard can show an "All Logs" page without slowing down packet capture.

Design:
- PacketCapture sends PacketEvent objects to traffic_queue.
- TrafficWriter is the only thread that writes logs/traffic.jsonl.
- The dashboard reads the JSONL file through API endpoints.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

from packet_capture import PacketEvent

LOGGER = logging.getLogger("ids.traffic_writer")


class TrafficWriter:
    """Write observed packet events to JSONL."""

    def __init__(
        self,
        traffic_queue: "queue.Queue[PacketEvent]",
        stop_event: threading.Event,
        output_file: Path,
    ) -> None:
        self.traffic_queue = traffic_queue
        self.stop_event = stop_event
        self.output_file = output_file

    def start(self) -> threading.Thread:
        thread = threading.Thread(
            target=self._write_loop,
            name="traffic-writer",
            daemon=True,
        )
        thread.start()
        return thread

    def _write_loop(self) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Writing traffic logs to %s", self.output_file)

        with self.output_file.open("a", encoding="utf-8", buffering=1) as f:
            while not self.stop_event.is_set() or not self.traffic_queue.empty():
                try:
                    event = self.traffic_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                try:
                    record = {
                        "event_type": "TRAFFIC",
                        "severity": "INFO",
                        "created_at": time.time(),
                        "created_at_utc": event.to_dict().get("timestamp_utc"),
                        "source": "packet_capture",
                        "protocol": event.protocol,
                        "src_ip": event.src_ip,
                        "dst_ip": event.dst_ip,
                        "src_port": event.src_port,
                        "dst_port": event.dst_port,
                        "length": event.length,
                        "summary": event.summary(),
                        "packet": event.to_dict(),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                finally:
                    self.traffic_queue.task_done()
