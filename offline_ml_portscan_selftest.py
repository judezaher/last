"""Offline self-test: verify that the ML model classifies a tiny TCP reset flow as PortScan.

Run from the ids_ioc_starter folder:
    python3 offline_ml_portscan_selftest.py
"""
from __future__ import annotations

from pathlib import Path

from packet_capture import PacketEvent
from cicids_live_features import CICIDSFlowExtractor, CICIDSNormalizer
from cicids_model import CICIDSPureNumpyModel, load_feature_columns


def main() -> None:
    feature_columns = load_feature_columns(Path("models/cicids_feature_columns.json"))
    normalizer = CICIDSNormalizer.from_json(Path("models/live_feature_normalizer.json"), feature_columns)
    model = CICIDSPureNumpyModel(Path("models/CICIDS_baseline (2).h5"), Path("models/Mapping"))

    extractor = CICIDSFlowExtractor(
        feature_columns=feature_columns,
        normalizer=normalizer,
        idle_timeout_seconds=3,
        active_timeout_seconds=10,
        min_packets=2,
        flow_key_mode="five-tuple",
        http_remap_ports=(),
    )

    t = 1000.0
    events = [
        PacketEvent(
            timestamp=t,
            src_ip="192.168.10.10",
            dst_ip="192.168.10.1",
            protocol="TCP",
            src_port=50000,
            dst_port=80,
            length=40,
            ip_header_length=20,
            transport_header_length=20,
            payload_length=0,
            tcp_flags="S",
            tcp_window=64240,
        ),
        PacketEvent(
            timestamp=t + 0.000058,
            src_ip="192.168.10.1",
            dst_ip="192.168.10.10",
            protocol="TCP",
            src_port=80,
            dst_port=50000,
            length=40,
            ip_header_length=20,
            transport_header_length=20,
            payload_length=0,
            tcp_flags="RA",
            tcp_window=0,
        ),
    ]

    for event in events:
        extractor.add_packet(event)

    flows = extractor.flush_all()
    if not flows:
        raise RuntimeError("No flow was finalized")

    flow = flows[0]
    prediction = model.predict(flow.as_numpy_2d())[0]
    print(f"Flow: {flow.summary()}")
    print(f"Prediction: {prediction.predicted_label} confidence={prediction.confidence:.4f}")
    print("Top probabilities:")
    for label, prob in sorted(prediction.probabilities.items(), key=lambda item: item[1], reverse=True)[:5]:
        print(f"  {label}: {prob:.6f}")


if __name__ == "__main__":
    main()
