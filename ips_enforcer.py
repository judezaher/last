"""IPS enforcement module.

Phase 3 adds this module.

The IDS workers still detect alerts. This module receives allowed alerts through a
queue and can block the suspicious source IP using iptables.

Safety design:
- Default project mode is still alert-only.
- dry-run mode prints what WOULD be blocked without changing firewall rules.
- enforce mode is required before iptables rules are added.
- Rules inserted by this module include a comment, so we can clean them later.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

LOGGER = logging.getLogger("ids.ips_enforcer")


@dataclass(frozen=True)
class BlockAction:
    """A normalized block decision created from an IDS alert."""

    ip: str
    reasons: tuple[str, ...]
    summary: str
    created_at: float = field(default_factory=time.time)


class IPSEnforcer:
    """Consumes IDS alerts and optionally blocks source IPs with iptables."""

    def __init__(
        self,
        block_queue: "queue.Queue[dict]",
        stop_event: threading.Event,
        mode: str = "alert-only",
        chains: Optional[list[str]] = None,
        block_seconds: int = 300,
        comment: str = "IDS_IOC_BLOCK",
        cleanup_on_exit: bool = True,
    ) -> None:
        self.block_queue = block_queue
        self.stop_event = stop_event
        self.mode = mode
        self.chains = chains or ["INPUT", "FORWARD"]
        self.block_seconds = block_seconds
        self.comment = comment
        self.cleanup_on_exit = cleanup_on_exit
        self._installed: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, name="ips-enforcer", daemon=True)
        thread.start()
        return thread

    def run(self) -> None:
        LOGGER.info(
            "IPS enforcer started mode=%s chains=%s block_seconds=%s cleanup_on_exit=%s",
            self.mode,
            ",".join(self.chains),
            self.block_seconds,
            self.cleanup_on_exit,
        )

        while not self.stop_event.is_set() or not self.block_queue.empty():
            try:
                alert = self.block_queue.get(timeout=0.5)
            except queue.Empty:
                self._expire_old_rules()
                continue

            try:
                action = self._alert_to_block_action(alert)
                if action is None:
                    LOGGER.info("IPS_NO_ACTION alert had no source IP to block: %s", alert.get("summary"))
                else:
                    self._handle_action(action)
            except Exception:
                LOGGER.exception("IPS failed while handling alert")
            finally:
                self.block_queue.task_done()
                self._expire_old_rules()

        if self.cleanup_on_exit:
            self.cleanup_rules()

        LOGGER.info("IPS enforcer stopped")

    def _alert_to_block_action(self, alert: dict) -> Optional[BlockAction]:
        """Convert an alert to a block action.

        Beginner rule for Phase 3:
        - Block the packet source IP, because that is the system that generated
          suspicious traffic in our lab.
        - Do not block if source IP is missing.
        """
        packet = alert.get("packet", {})
        flow = alert.get("flow", {})
        src_ip = packet.get("src_ip") or flow.get("src_ip")
        if not src_ip:
            return None

        reasons = tuple(alert.get("reasons", []))
        summary = str(alert.get("summary", ""))
        return BlockAction(ip=str(src_ip), reasons=reasons, summary=summary)

    def _handle_action(self, action: BlockAction) -> None:
        if self.mode == "alert-only":
            return

        if self.mode == "dry-run":
            LOGGER.warning(
                "IPS_DRY_RUN would block source_ip=%s chains=%s reasons=%s packet=%s",
                action.ip,
                self.chains,
                list(action.reasons),
                action.summary,
            )
            return

        if self.mode != "enforce":
            LOGGER.warning("Unknown IPS mode %s; no block applied", self.mode)
            return

        for chain in self.chains:
            self._ensure_drop_rule(chain, action.ip, action.reasons)

    def _ensure_drop_rule(self, chain: str, ip: str, reasons: tuple[str, ...]) -> None:
        key = (chain, ip)
        now = time.time()

        with self._lock:
            # If we already installed this rule, just refresh its expiry time.
            if key in self._installed:
                self._installed[key] = now
                return

        if self._iptables_rule_exists(chain, ip):
            with self._lock:
                self._installed[key] = now
            LOGGER.info("IPS_BLOCK_ALREADY_EXISTS chain=%s source_ip=%s", chain, ip)
            return

        cmd = [
            "iptables",
            "-I",
            chain,
            "1",
            "-s",
            ip,
            "-m",
            "comment",
            "--comment",
            self.comment,
            "-j",
            "DROP",
        ]
        self._run_iptables(cmd)

        with self._lock:
            self._installed[key] = now

        LOGGER.warning(
            "IPS_BLOCK_APPLIED chain=%s source_ip=%s reasons=%s",
            chain,
            ip,
            list(reasons),
        )

    def _iptables_rule_exists(self, chain: str, ip: str) -> bool:
        cmd = [
            "iptables",
            "-C",
            chain,
            "-s",
            ip,
            "-m",
            "comment",
            "--comment",
            self.comment,
            "-j",
            "DROP",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def _delete_drop_rule(self, chain: str, ip: str) -> None:
        cmd = [
            "iptables",
            "-D",
            chain,
            "-s",
            ip,
            "-m",
            "comment",
            "--comment",
            self.comment,
            "-j",
            "DROP",
        ]

        # Delete repeatedly in case the same rule exists more than once.
        while self._iptables_rule_exists(chain, ip):
            self._run_iptables(cmd)
            LOGGER.warning("IPS_BLOCK_REMOVED chain=%s source_ip=%s", chain, ip)

    def _expire_old_rules(self) -> None:
        if self.mode != "enforce" or self.block_seconds <= 0:
            return

        now = time.time()
        expired: list[tuple[str, str]] = []

        with self._lock:
            for key, last_seen in list(self._installed.items()):
                if now - last_seen >= self.block_seconds:
                    expired.append(key)

        for chain, ip in expired:
            try:
                self._delete_drop_rule(chain, ip)
            except Exception:
                LOGGER.exception("Could not expire IPS block chain=%s source_ip=%s", chain, ip)
            finally:
                with self._lock:
                    self._installed.pop((chain, ip), None)

    def cleanup_rules(self) -> None:
        if self.mode != "enforce":
            return

        with self._lock:
            rules = list(self._installed.keys())

        for chain, ip in rules:
            try:
                self._delete_drop_rule(chain, ip)
            except Exception:
                LOGGER.exception("Could not cleanup IPS block chain=%s source_ip=%s", chain, ip)
            finally:
                with self._lock:
                    self._installed.pop((chain, ip), None)

    @staticmethod
    def _run_iptables(cmd: list[str]) -> None:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"iptables command failed: {' '.join(cmd)}\n"
                f"stdout={result.stdout.strip()}\n"
                f"stderr={result.stderr.strip()}"
            )
