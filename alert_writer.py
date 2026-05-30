"""Alert writer thread.

Workers should not write to disk directly.
They put alerts into a queue, and this one thread writes JSONL alerts.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path

LOGGER = logging.getLogger("ids.alert_writer")


class AlertWriter:
    def __init__(
        self,
        alert_queue: "queue.Queue[dict]",
        stop_event: threading.Event,
        output_file: Path,
    ) -> None:
        self.alert_queue = alert_queue
        self.stop_event = stop_event
        self.output_file = output_file

    def start(self) -> threading.Thread:
        thread = threading.Thread(
            target=self._write_loop,
            name="alert-writer",
            daemon=True,
        )
        thread.start()
        return thread

    def _write_loop(self) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Writing alerts to %s", self.output_file)

        with self.output_file.open("a", encoding="utf-8", buffering=1) as f:
            while not self.stop_event.is_set() or not self.alert_queue.empty():
                try:
                    alert = self.alert_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                f.write(json.dumps(alert, ensure_ascii=False) + "\n")
                self.alert_queue.task_done()
