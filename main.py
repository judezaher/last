"""Threaded IDS/IPS with IOC detection, CICIDS ML flow detection, and alert correlation.

This cleaned version removes:
  - Flow Rate Guard
  - Isolation Forest anomaly detection

Kept:
  - IOC blacklist/whitelist detection
  - IPS dry-run/enforce blocking
  - CICIDS supervised ML model
  - ICMP skip in ML pipeline
  - Per-class ML thresholds
  - Alert correlation for remaining IDS/ML alerts
"""

from __future__ import annotations

import argparse
import logging
import queue
import signal
import threading
from pathlib import Path

from alert_writer import AlertWriter
from traffic_writer import TrafficWriter
from ioc_engine import AlertDeduplicator, worker_loop
from ioc_loader import IOCCache
from packet_capture import PacketCapture, PacketEvent
from ips_enforcer import IPSEnforcer
from ml_engine import CICIDSLiveModelAdapter, ml_worker_loop
from correlation_engine import CorrelationEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Threaded IOC IDS/IPS + CICIDS ML + Correlation"
    )
    # --- Core ---
    parser.add_argument("--iface", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--alerts", default="logs/alerts.jsonl")
    parser.add_argument("--traffic-log", default="logs/traffic.jsonl")
    parser.add_argument("--log-traffic", action="store_true", help="Write all observed packet summaries to logs/traffic.jsonl for the dashboard All Logs page.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=10000)
    parser.add_argument("--reload-seconds", type=int, default=30)
    parser.add_argument("--dedup-seconds", type=int, default=30)
    parser.add_argument(
        "--dedup-mode",
        choices=["source", "source-reason", "exact"],
        default="source-reason",
    )
    parser.add_argument("--bpf", default="ip")
    parser.add_argument("--debug-packets", action="store_true")
    parser.add_argument("--debug-every", type=int, default=1)
    parser.add_argument("--debug-decisions", action="store_true")

    # --- IPS ---
    parser.add_argument(
        "--ips-mode",
        choices=["alert-only", "dry-run", "enforce"],
        default="alert-only",
    )
    parser.add_argument("--block-chains", default="INPUT,FORWARD")
    parser.add_argument("--block-seconds", type=int, default=300)
    parser.add_argument("--no-cleanup-on-exit", action="store_true")

    # --- ML (CICIDS) ---
    parser.add_argument(
        "--ml-mode",
        choices=["off", "features-only", "predict"],
        default="predict",
    )
    parser.add_argument("--ml-model-path", default="models/CICIDS_baseline (2).h5")
    parser.add_argument("--ml-features-path", default="models/cicids_feature_columns.json")
    parser.add_argument("--ml-mapping-path", default="models/Mapping")
    parser.add_argument("--ml-normalizer-path", default="models/live_feature_normalizer.json")
    parser.add_argument("--ml-workers", type=int, default=1)
    parser.add_argument(
        "--ml-threshold",
        type=float,
        default=0.60,
        help="Global ML alert threshold. Per-class thresholds override this for known attack classes.",
    )
    parser.add_argument("--ml-alert-benign", action="store_true")
    parser.add_argument("--ml-attack-prob-threshold", type=float, default=0.15)
    parser.add_argument("--ml-debug-every", type=int, default=1)
    parser.add_argument("--flow-idle-timeout", type=float, default=90.0)
    parser.add_argument("--flow-active-timeout", type=float, default=30.0)
    parser.add_argument("--min-flow-packets", type=int, default=2)
    parser.add_argument(
        "--ml-flow-key-mode",
        choices=["five-tuple", "service"],
        default="service",
        help=(
            "Default flow key mode. In v17, web port flows automatically use service mode "
            "and all others use five-tuple, regardless of this setting. "
            "This setting still controls non-TCP flows."
        ),
    )
    parser.add_argument("--ml-http-remap-ports", default="8080,8000,8443")

    # --- Correlation Engine ---
    parser.add_argument(
        "--correlation",
        action="store_true",
        default=True,
        help="Enable alert correlation engine. Default: on.",
    )
    parser.add_argument("--no-correlation", dest="correlation", action="store_false")
    parser.add_argument(
        "--scan-dst-threshold",
        type=int,
        default=10,
        help="Unique destination IPs within scan-window before a scan correlation fires. Default: 10.",
    )

    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    def parse_port_list(text: str) -> tuple[int, ...]:
        ports = []
        for item in str(text or "").split(","):
            item = item.strip()
            if not item:
                continue
            ports.append(int(item))
        return tuple(ports)

    args.ml_http_remap_ports_tuple = parse_port_list(args.ml_http_remap_ports)

    log = logging.getLogger("ids.main")

    if args.ml_workers != 1:
        log.warning(
            "Using ml-workers=%d. For flow-based ML, 1 is safest because a flow must stay in one worker.",
            args.ml_workers,
        )

    stop_event = threading.Event()

    def request_stop(_signum=None, _frame=None) -> None:
        log.info("Stop requested")
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    packet_queue: "queue.Queue[PacketEvent]" = queue.Queue(maxsize=args.queue_size)
    ml_queue: "queue.Queue[PacketEvent]" = queue.Queue(maxsize=args.queue_size)
    alert_queue: "queue.Queue[dict]" = queue.Queue(maxsize=args.queue_size)
    block_queue: "queue.Queue[dict]" = queue.Queue(maxsize=args.queue_size)
    traffic_queue: "queue.Queue[PacketEvent] | None" = None
    traffic_thread: threading.Thread | None = None
    if args.log_traffic:
        traffic_queue = queue.Queue(maxsize=args.queue_size)

    # --- IOC cache ---
    ioc_cache = IOCCache(Path(args.data_dir))
    ioc_cache.load_once()
    reload_thread = threading.Thread(
        target=ioc_cache.reload_loop,
        args=(stop_event, args.reload_seconds),
        name="ioc-reloader",
        daemon=True,
    )
    reload_thread.start()

    # --- Alert writer ---
    alert_writer = AlertWriter(alert_queue, stop_event, Path(args.alerts))
    alert_thread = alert_writer.start()

    # --- Traffic writer for dashboard All Logs page ---
    if traffic_queue is not None:
        traffic_writer = TrafficWriter(traffic_queue, stop_event, Path(args.traffic_log))
        traffic_thread = traffic_writer.start()

    # --- IPS ---
    chains = [c.strip().upper() for c in args.block_chains.split(",") if c.strip()]
    ips_enforcer = IPSEnforcer(
        block_queue=block_queue,
        stop_event=stop_event,
        mode=args.ips_mode,
        chains=chains,
        block_seconds=args.block_seconds,
        cleanup_on_exit=not args.no_cleanup_on_exit,
    )
    ips_thread = ips_enforcer.start()

    # --- Correlation engine ---
    correlation_engine: CorrelationEngine | None = None
    if args.correlation:
        correlation_engine = CorrelationEngine(
            scan_unique_dst_threshold=args.scan_dst_threshold,
        )
        log.info("Alert correlation engine enabled (scan_dst_threshold=%d)", args.scan_dst_threshold)

    # --- IOC workers ---
    deduplicator = AlertDeduplicator(hold_seconds=args.dedup_seconds)
    workers: list[threading.Thread] = []
    for worker_id in range(args.workers):
        t = threading.Thread(
            target=_ioc_worker,
            args=(
                worker_id,
                packet_queue,
                alert_queue,
                block_queue,
                ioc_cache,
                stop_event,
                deduplicator,
                args.dedup_mode,
                args.debug_decisions,
                args.ips_mode,
                correlation_engine,
            ),
            name=f"ioc-worker-{worker_id}",
            daemon=True,
        )
        t.start()
        workers.append(t)

    # --- ML workers ---
    ml_model_adapter = None
    if args.ml_mode == "predict":
        ml_model_adapter = CICIDSLiveModelAdapter(
            Path(args.ml_model_path), Path(args.ml_mapping_path)
        )

    ml_workers: list[threading.Thread] = []
    if args.ml_mode != "off":
        ml_deduplicator = AlertDeduplicator(hold_seconds=args.dedup_seconds)
        for worker_id in range(args.ml_workers):
            t = threading.Thread(
                target=ml_worker_loop,
                args=(
                    worker_id,
                    ml_queue,
                    alert_queue,
                    block_queue,
                    stop_event,
                    args.ml_mode,
                    ml_model_adapter,
                    args.ml_threshold,
                    ml_deduplicator,
                    Path(args.ml_features_path),
                    Path(args.ml_normalizer_path) if args.ml_normalizer_path else None,
                    args.flow_idle_timeout,
                    args.flow_active_timeout,
                    args.min_flow_packets,
                    args.ml_flow_key_mode,
                    args.ml_http_remap_ports_tuple,
                    args.debug_decisions,
                    args.ml_debug_every,
                    args.ips_mode,
                    args.ml_alert_benign,
                    args.ml_attack_prob_threshold,
                    correlation_engine,
                ),
                name=f"ml-worker-{worker_id}",
                daemon=True,
            )
            t.start()
            ml_workers.append(t)

    # --- Packet capture ---
    capture_output_queues = [packet_queue]
    if args.ml_mode != "off":
        capture_output_queues.append(ml_queue)
    if traffic_queue is not None:
        capture_output_queues.append(traffic_queue)

    capture = PacketCapture(
        iface=args.iface,
        output_queues=capture_output_queues,
        stop_event=stop_event,
        bpf_filter=args.bpf,
        debug_packets=args.debug_packets,
        debug_every=args.debug_every,
    )
    capture.start()

    log.info(
        "IDS v17 started — IOC + ML (CICIDS) + Correlation. Press Ctrl+C to stop."
    )

    try:
        while not stop_event.is_set():
            stop_event.wait(1)
    finally:
        stop_event.set()
        packet_queue.join()
        ml_queue.join()
        alert_queue.join()
        block_queue.join()
        if traffic_queue is not None:
            traffic_queue.join()
        alert_thread.join(timeout=2)
        if traffic_thread is not None:
            traffic_thread.join(timeout=2)
        ips_thread.join(timeout=2)
        log.info("IDS stopped cleanly")


def _ioc_worker(
    worker_id: int,
    packet_queue: "queue.Queue[PacketEvent]",
    alert_queue: "queue.Queue[dict]",
    block_queue: "queue.Queue[dict]",
    ioc_cache: "IOCCache",
    stop_event: threading.Event,
    deduplicator: "AlertDeduplicator",
    dedup_mode: str,
    debug_decisions: bool,
    ips_mode: str,
    correlation_engine: "CorrelationEngine | None",
) -> None:
    """IOC worker thread extended with rate guard and correlation feeding."""
    from ioc_engine import detect_ioc, build_dedup_key
    import queue as _queue

    log = logging.getLogger(f"ids.ioc_worker.{worker_id}")
    log.info("IOC worker %d started", worker_id)

    def _port_text(port) -> str:
        return "-" if port is None else str(port)

    while not stop_event.is_set() or not packet_queue.empty():
        try:
            event = packet_queue.get(timeout=0.5)
        except _queue.Empty:
            continue

        try:
            # IOC blacklist check
            alert = detect_ioc(event, ioc_cache, debug_decisions=debug_decisions)
            if alert:
                key = build_dedup_key(event, alert, dedup_mode)
                if deduplicator.allow(key):
                    corr = correlation_engine.feed(alert) if correlation_engine else None
                    alert_queue.put_nowait(alert)
                    if corr is not None:
                        try:
                            alert_queue.put_nowait(corr)
                        except _queue.Full:
                            pass
                    if ips_mode != "alert-only":
                        try:
                            block_queue.put_nowait(alert)
                        except _queue.Full:
                            log.warning("IPS block queue full; block request dropped for packet: %s", event.summary())
                    log.warning(
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
                    log.info(
                        "DUPLICATE_SUPPRESSED worker=%d %s reasons=%s",
                        worker_id, event.summary(), alert["reasons"],
                    )
        except _queue.Full:
            log.warning("Alert queue full; alert dropped for packet: %s", event.summary())
        except Exception:
            log.exception("IOC worker %d failed to process packet", worker_id)
        finally:
            packet_queue.task_done()

    log.info("IOC worker %d stopped", worker_id)


if __name__ == "__main__":
    main()
