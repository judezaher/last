"""Phase 4 ML/deep-learning engine — cleaned v17.

Kept improvements:
  - ICMP flows are skipped before feature extraction.
  - Per-class ML thresholds replace the single global threshold.
  - Smart flow-key-mode selection: service mode for web ports, five-tuple for others.
  - Alert correlation engine can escalate combined remaining signals.

Removed from this cleaned version:
  - Isolation Forest anomaly detector.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

from cicids_live_features import CICIDSFeatureVector, CICIDSFlowExtractor, CICIDSNormalizer
from cicids_model import CICIDSPureNumpyModel, load_feature_columns
from packet_capture import PacketEvent
from correlation_engine import CorrelationEngine

LOGGER = logging.getLogger("ids.ml_engine")

# ---------------------------------------------------------------------------
# Per-class confidence thresholds
# Some attack classes have weaker signatures in the live normalizer.
# Lowering their threshold catches more without flooding false positives
# on other classes.
# ---------------------------------------------------------------------------
DEFAULT_PER_CLASS_THRESHOLDS: dict[str, float] = {
    "DoS Hulk":             0.50,
    "DoS slowloris":        0.55,
    "DoS Slowhttptest":     0.55,
    "DoS GoldenEye":        0.50,
    "DDoS":                 0.50,
    "PortScan":             0.45,
    "Bot":                  0.55,
    "Web Attack":           0.55,
    "Infiltration":         0.55,
    "Heartbleed":           0.55,
    "FTP-Patator":          0.55,
    "SSH-Patator":          0.55,
    "BENIGN":               0.99,   # never alert on benign via this path
}

# Web service ports — use service flow-key mode for these
WEB_PORTS: frozenset[int] = frozenset({80, 443, 8000, 8080, 8443, 3000, 5000})


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class DeduplicatorLike(Protocol):
    def allow(self, key: tuple) -> bool: ...


class CICIDSLiveModelAdapter:
    """Small wrapper around the doctor's CICIDS H5 model."""

    def __init__(self, model_path: Path, mapping_path: Path | None = None) -> None:
        LOGGER.info("Loading CICIDS model from %s", model_path)
        self.model = CICIDSPureNumpyModel(model_path, mapping_path)
        self._lock = threading.Lock()

    def predict_one(self, features: CICIDSFeatureVector):
        with self._lock:
            return self.model.predict(features.as_numpy_2d())[0]


def get_class_threshold(predicted_label: str, global_threshold: float) -> float:
    """Return the per-class threshold, falling back to the global threshold."""
    for key, val in DEFAULT_PER_CLASS_THRESHOLDS.items():
        if key.lower() in predicted_label.lower():
            return val
    return global_threshold


def build_ml_alert(flow: CICIDSFeatureVector, prediction, threshold: float) -> dict:
    severity = "HIGH" if prediction.confidence >= max(0.90, threshold) else "MEDIUM"
    return {
        "alert_type": "ML_FLOW_MATCH",
        "severity": severity,
        "created_at": time.time(),
        "created_at_utc": _utc_now(),
        "reasons": [
            f"ML_CLASS:{prediction.predicted_label}",
            f"ML_CONFIDENCE:{prediction.confidence:.4f}",
            f"ML_THRESHOLD:{threshold:.4f}",
        ],
        "flow": {
            "src_ip": flow.src_ip,
            "dst_ip": flow.dst_ip,
            "src_port": flow.src_port,
            "dst_port": flow.dst_port,
            "protocol": flow.protocol,
            "packet_count": flow.packet_count,
            "flow_id": str(flow.flow_id),
        },
        "features_normalized": flow.normalized_dict(),
        "summary": (
            f"ML flow alert class={prediction.predicted_label} "
            f"confidence={prediction.confidence:.4f} {flow.summary()}"
        ),
    }


def should_alert(predicted_label: str, confidence: float, threshold: float, alert_on_benign: bool = False) -> bool:
    if confidence < threshold:
        return False
    if predicted_label.upper() == "BENIGN" and not alert_on_benign:
        return False
    return True


def top_probabilities(prediction, n: int = 3) -> list[tuple[str, float]]:
    probs = getattr(prediction, "probabilities", {}) or {}
    return sorted(((str(k), float(v)) for k, v in probs.items()), key=lambda item: item[1], reverse=True)[:n]


def best_attack_probability(prediction) -> tuple[str, float]:
    probs = getattr(prediction, "probabilities", {}) or {}
    attacks = [(str(label), float(prob)) for label, prob in probs.items() if str(label).upper() != "BENIGN"]
    if not attacks:
        return "", 0.0
    return max(attacks, key=lambda item: item[1])


def log_flow_features(worker_id: int, flow: CICIDSFeatureVector, debug_every: int, counter: int) -> None:
    if counter % debug_every != 0:
        return
    vals = flow.normalized_dict()
    preview_keys = ["Dst Port", "Protocol", "Flow Duration", "Flow_Packets", "Flow_Bytes", "Packet Length Mean"]
    preview = {k: round(float(vals.get(k, 0.0)), 6) for k in preview_keys}
    LOGGER.info(
        "ML_FLOW_FEATURES worker=%d flow=%s preview=%s vector_len=%d",
        worker_id, flow.summary(), preview, len(flow.normalized_values),
    )


def smart_flow_key_mode(dst_port: int, global_mode: str) -> str:
    """Use service mode for web ports, five-tuple for everything else.

    Service mode is good for Slowloris/HULK (many sockets to same service).
    Five-tuple mode is better for port scans and exploit flows.
    """
    if dst_port in WEB_PORTS:
        return "service"
    return "five-tuple"


def process_finalized_flow(
    *,
    worker_id: int,
    flow: CICIDSFeatureVector,
    mode: str,
    model_adapter: Optional[CICIDSLiveModelAdapter],
    threshold: float,
    deduplicator: DeduplicatorLike,
    alert_queue: "queue.Queue[dict]",
    block_queue: "queue.Queue[dict]",
    ips_mode: str,
    debug_decisions: bool,
    alert_on_benign: bool,
    attack_prob_threshold: float = 0.35,
    correlation_engine: Optional[CorrelationEngine] = None,
) -> None:
    # ------------------------------------------------------------------ #
    # Improvement 1: Skip ICMP entirely — CICIDS model not trained on it  #
    # ------------------------------------------------------------------ #
    if flow.protocol.upper() == "ICMP":
        if debug_decisions:
            LOGGER.info("ML_SKIP_ICMP flow=%s", flow.summary())
        return

    if mode == "features-only":
        log_flow_features(worker_id, flow, 1, 1)
        feature_record = {
            "alert_type": "ML_FLOW_FEATURES",
            "severity": "INFO",
            "created_at": time.time(),
            "created_at_utc": _utc_now(),
            "reasons": ["ML_MODE:features-only"],
            "flow": {
                "src_ip": flow.src_ip,
                "dst_ip": flow.dst_ip,
                "src_port": flow.src_port,
                "dst_port": flow.dst_port,
                "protocol": flow.protocol,
                "packet_count": flow.packet_count,
                "flow_id": str(flow.flow_id),
            },
            "features_normalized": flow.normalized_dict(),
            "summary": f"ML feature capture {flow.summary()}",
        }
        try:
            alert_queue.put_nowait(feature_record)
        except queue.Full:
            LOGGER.warning("Alert queue full; feature record dropped for flow: %s", flow.summary())
        return

    if mode != "predict":
        return

    if model_adapter is None:
        LOGGER.warning("ML predict mode requested but model adapter is missing")
        return

    prediction = model_adapter.predict_one(flow)
    top3 = top_probabilities(prediction, n=3)
    best_attack_label, best_attack_prob = best_attack_probability(prediction)

    # ------------------------------------------------------------------ #
    # Improvement 2: Per-class threshold                                  #
    # ------------------------------------------------------------------ #
    class_threshold = get_class_threshold(prediction.predicted_label, threshold)

    if debug_decisions:
        LOGGER.info(
            "ML_PREDICTION worker=%d class=%s confidence=%.4f class_threshold=%.4f "
            "best_attack=%s:%.4f top3=%s flow=%s",
            worker_id,
            prediction.predicted_label,
            prediction.confidence,
            class_threshold,
            best_attack_label or "NONE",
            best_attack_prob,
            top3,
            flow.summary(),
        )

    alert_by_top_class = should_alert(
        prediction.predicted_label, prediction.confidence, class_threshold, alert_on_benign
    )
    alert_by_attack_probability = (
        prediction.predicted_label.upper() == "BENIGN"
        and best_attack_prob >= attack_prob_threshold
    )

    def _emit(alert: dict) -> None:
        dedup_key = (
            "ml-flow",
            flow.src_ip,
            flow.dst_ip,
            flow.protocol,
            prediction.predicted_label,
            best_attack_label if alert_by_attack_probability else "top",
        )
        if not deduplicator.allow(dedup_key):
            if debug_decisions:
                LOGGER.info("ML_DUPLICATE_SUPPRESSED worker=%d flow=%s", worker_id, flow.summary())
            return

        # Feed into correlation engine
        corr_alert = None
        if correlation_engine is not None:
            corr_alert = correlation_engine.feed(alert)

        alert_queue.put_nowait(alert)
        if corr_alert is not None:
            try:
                alert_queue.put_nowait(corr_alert)
            except queue.Full:
                pass

        if ips_mode != "alert-only":
            try:
                block_queue.put_nowait(alert)
            except queue.Full:
                LOGGER.warning("IPS block queue full; ML block request dropped for flow: %s", flow.summary())

        LOGGER.warning(
            "ML_ALERT worker=%d class=%s confidence=%.4f flow=%s",
            worker_id, prediction.predicted_label, prediction.confidence, flow.summary(),
        )

    if alert_by_top_class or alert_by_attack_probability:
        alert = build_ml_alert(flow, prediction, class_threshold)
        if alert_by_attack_probability:
            alert["alert_type"] = "ML_ATTACK_PROBABILITY_GUARD"
            alert["severity"] = "MEDIUM"
            alert["reasons"].append(f"BEST_ATTACK_CLASS:{best_attack_label}")
            alert["reasons"].append(f"BEST_ATTACK_PROBABILITY:{best_attack_prob:.4f}")
            alert["reasons"].append(f"ATTACK_PROB_THRESHOLD:{attack_prob_threshold:.4f}")
            alert["summary"] = (
                f"ML attack-probability guard best_attack={best_attack_label} "
                f"probability={best_attack_prob:.4f} top_class={prediction.predicted_label} "
                f"confidence={prediction.confidence:.4f} {flow.summary()}"
            )
        alert["model_top3_probabilities"] = top3
        _emit(alert)


def ml_worker_loop(
    worker_id: int,
    ml_queue: "queue.Queue[PacketEvent]",
    alert_queue: "queue.Queue[dict]",
    block_queue: "queue.Queue[dict]",
    stop_event: threading.Event,
    mode: str,
    model_adapter: Optional[CICIDSLiveModelAdapter],
    threshold: float,
    deduplicator: DeduplicatorLike,
    feature_columns_path: Path,
    normalizer_path: Path | None,
    flow_idle_timeout: float,
    flow_active_timeout: float,
    min_flow_packets: int,
    flow_key_mode: str = "service",
    http_remap_ports: tuple[int, ...] = (),
    debug_decisions: bool = False,
    debug_every: int = 20,
    ips_mode: str = "alert-only",
    alert_on_benign: bool = False,
    attack_prob_threshold: float = 0.35,
    correlation_engine: Optional[CorrelationEngine] = None,
) -> None:
    """ML worker thread with all v17 improvements active."""
    feature_columns = load_feature_columns(feature_columns_path)
    if len(feature_columns) != 72:
        raise ValueError(f"Expected 72 CICIDS feature columns, got {len(feature_columns)}")

    normalizer = CICIDSNormalizer.from_json(normalizer_path, feature_columns)

    extractor = CICIDSFlowExtractor(
        feature_columns=feature_columns,
        normalizer=normalizer,
        idle_timeout_seconds=flow_idle_timeout,
        active_timeout_seconds=flow_active_timeout,
        min_packets=min_flow_packets,
        flow_key_mode=flow_key_mode,
        http_remap_ports=http_remap_ports,
    )

    LOGGER.info(
        "ML worker %d started mode=%s threshold=%.3f attack_prob_threshold=%.3f "
        "flow_key_mode=%s http_remap_ports=%s flow_idle_timeout=%.1fs "
        "flow_active_timeout=%.1fs min_packets=%d",
        worker_id, mode, threshold, attack_prob_threshold,
        flow_key_mode, ",".join(str(p) for p in http_remap_ports) or "none",
        flow_idle_timeout, flow_active_timeout, min_flow_packets,
    )

    processed_packets = 0
    debug_every = max(1, int(debug_every))

    def _process(flow: CICIDSFeatureVector) -> None:
        process_finalized_flow(
            worker_id=worker_id,
            flow=flow,
            mode=mode,
            model_adapter=model_adapter,
            threshold=threshold,
            deduplicator=deduplicator,
            alert_queue=alert_queue,
            block_queue=block_queue,
            ips_mode=ips_mode,
            debug_decisions=debug_decisions,
            alert_on_benign=alert_on_benign,
            attack_prob_threshold=attack_prob_threshold,
            correlation_engine=correlation_engine,
        )

    while not stop_event.is_set() or not ml_queue.empty():
        try:
            event = ml_queue.get(timeout=0.5)
        except queue.Empty:
            for flow in extractor.flush_expired():
                try:
                    _process(flow)
                except Exception:
                    LOGGER.exception("ML worker %d failed to process expired flow", worker_id)
            continue

        try:
            processed_packets += 1
            finalized_flows = extractor.add_packet(event)
            for flow in finalized_flows:
                if mode == "features-only" and processed_packets % debug_every != 0:
                    continue
                _process(flow)
        except queue.Full:
            LOGGER.warning("Alert queue full; ML alert dropped for packet: %s", event.summary())
        except Exception:
            LOGGER.exception("ML worker %d failed to process packet", worker_id)
        finally:
            ml_queue.task_done()

    # Final flush on shutdown
    for flow in extractor.flush_all():
        try:
            _process(flow)
        except Exception:
            LOGGER.exception("ML worker %d failed to process final flow", worker_id)

    LOGGER.info("ML worker %d stopped", worker_id)
