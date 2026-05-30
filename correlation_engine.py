"""Alert Correlation Engine.

Combines multiple remaining IDS/ML signals from the same source IP into stronger,
escalated alerts.

Examples of correlations detected in this cleaned version:
- ML alert + IOC blacklist match from the same source → CRITICAL
- Multiple destination IPs from the same source within 60s → likely scan/host sweep
- Repeated ML probability-guard alerts → HIGH

Flow Rate Guard and Isolation Forest anomaly correlation rules were removed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

LOGGER = logging.getLogger("ids.correlation_engine")


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class _SourceHistory:
    """Rolling alert history for one source IP."""
    alerts: deque = field(default_factory=lambda: deque(maxlen=200))
    dst_ips: deque = field(default_factory=lambda: deque(maxlen=500))
    dst_ip_times: deque = field(default_factory=lambda: deque(maxlen=500))

    def add_alert(self, alert: dict, now: float) -> None:
        self.alerts.append((now, alert))

    def add_dst(self, dst_ip: str, now: float) -> None:
        self.dst_ips.append(dst_ip)
        self.dst_ip_times.append(now)

    def prune(self, cutoff: float) -> None:
        while self.alerts and self.alerts[0][0] < cutoff:
            self.alerts.popleft()
        while self.dst_ip_times and self.dst_ip_times[0] < cutoff:
            self.dst_ip_times.popleft()
            if self.dst_ips:
                self.dst_ips.popleft()

    def recent_alert_types(self, window: float, now: float) -> set[str]:
        cutoff = now - window
        return {a["alert_type"] for ts, a in self.alerts if ts >= cutoff}

    def unique_dst_count(self, window: float, now: float) -> int:
        cutoff = now - window
        seen = set()
        for i, ts in enumerate(self.dst_ip_times):
            if ts >= cutoff and i < len(self.dst_ips):
                seen.add(self.dst_ips[i])
        return len(seen)

    def alert_count_by_type(self, alert_type: str, window: float, now: float) -> int:
        cutoff = now - window
        return sum(1 for ts, a in self.alerts if ts >= cutoff and a.get("alert_type") == alert_type)


class CorrelationEngine:
    """Correlate alerts from multiple engines by source IP.

    Call feed(alert) for every alert before it goes to the alert writer.
    If feed() returns a correlated alert, put BOTH the original and the
    correlated alert into the alert queue.
    """

    def __init__(
        self,
        scan_window_seconds: float = 60.0,
        scan_unique_dst_threshold: int = 10,
        correlation_window_seconds: float = 30.0,
        cooldown_seconds: float = 20.0,
    ) -> None:
        self.scan_window = float(scan_window_seconds)
        self.scan_dst_threshold = int(scan_unique_dst_threshold)
        self.corr_window = float(correlation_window_seconds)
        self.cooldown = float(cooldown_seconds)

        self._history: dict[str, _SourceHistory] = defaultdict(_SourceHistory)
        self._last_corr_alert: dict[str, float] = {}
        self._lock = threading.Lock()

    def feed(self, alert: dict) -> Optional[dict]:
        """Record alert and return a correlated escalation alert if warranted, else None."""
        now = time.time()
        src_ip = self._extract_src(alert)
        dst_ip = self._extract_dst(alert)

        if not src_ip:
            return None

        with self._lock:
            hist = self._history[src_ip]
            hist.prune(now - max(self.scan_window, self.corr_window) * 2)
            hist.add_alert(alert, now)
            if dst_ip:
                hist.add_dst(dst_ip, now)

            corr = self._evaluate(src_ip, hist, alert, now)

        return corr

    def _evaluate(
        self,
        src_ip: str,
        hist: _SourceHistory,
        latest_alert: dict,
        now: float,
    ) -> Optional[dict]:
        # Cooldown per source so we don't spam correlated alerts
        last = self._last_corr_alert.get(src_ip, 0.0)
        if now - last < self.cooldown:
            return None

        recent_types = hist.recent_alert_types(self.corr_window, now)
        unique_dsts = hist.unique_dst_count(self.scan_window, now)

        reasons = []
        severity = "HIGH"

        # Rule 1: ML + IOC from same source = very high confidence attack
        if "ML_FLOW_MATCH" in recent_types and "IOC_MATCH" in recent_types:
            reasons.append("CORR_ML_AND_IOC_SAME_SOURCE")
            severity = "CRITICAL"

        # Rule 2: Many unique destination IPs = port scan / host scan
        if unique_dsts >= self.scan_dst_threshold:
            reasons.append(f"CORR_SCAN_UNIQUE_DSTS:{unique_dsts}_in_{int(self.scan_window)}s")
            severity = "HIGH"

        # Rule 3: Multiple ML probability guard alerts = escalate
        guard_count = hist.alert_count_by_type("ML_ATTACK_PROBABILITY_GUARD", self.corr_window, now)
        if guard_count >= 3:
            reasons.append(f"CORR_REPEATED_ML_GUARD:{guard_count}")

        if not reasons:
            return None

        self._last_corr_alert[src_ip] = now

        corr_alert = {
            "alert_type": "CORRELATED_ALERT",
            "severity": severity,
            "created_at": now,
            "created_at_utc": _utc_now(),
            "reasons": reasons,
            "source_ip": src_ip,
            "recent_alert_types": sorted(recent_types),
            "unique_dst_ips_in_window": unique_dsts,
            "summary": (
                f"Correlated alert src={src_ip} severity={severity} "
                f"reasons={reasons} recent_types={sorted(recent_types)}"
            ),
        }
        LOGGER.warning(
            "CORRELATED_ALERT src=%s severity=%s reasons=%s",
            src_ip, severity, reasons,
        )
        return corr_alert

    @staticmethod
    def _extract_src(alert: dict) -> Optional[str]:
        if "source_ip" in alert:
            return alert["source_ip"]
        pkt = alert.get("packet") or {}
        flow = alert.get("flow") or {}
        return pkt.get("src_ip") or flow.get("src_ip")

    @staticmethod
    def _extract_dst(alert: dict) -> Optional[str]:
        pkt = alert.get("packet") or {}
        flow = alert.get("flow") or {}
        return pkt.get("dst_ip") or flow.get("dst_ip")
