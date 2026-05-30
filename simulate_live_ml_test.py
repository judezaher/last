"""Small Phase 4B-D simulation test without real network traffic.

It creates fake PacketEvent objects, turns them into a CICIDS flow, normalizes the
72 features, and runs the doctor's model. Use this to verify Phase 4 code before
capturing live packets.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from cicids_live_features import CICIDSFlowExtractor, CICIDSNormalizer
from cicids_model import CICIDSPureNumpyModel, load_feature_columns
from packet_capture import PacketEvent


def main() -> None:
    feature_columns = load_feature_columns("models/cicids_feature_columns.json")
    normalizer = CICIDSNormalizer.from_json("models/live_feature_normalizer.json", feature_columns)
    extractor = CICIDSFlowExtractor(
        feature_columns=feature_columns,
        normalizer=normalizer,
        idle_timeout_seconds=1,
        active_timeout_seconds=30,
        min_packets=1,
    )

    now = time.time()
    packets = [
        PacketEvent(now, "192.168.10.10", "192.168.10.1", "ICMP", None, None, 84, 20, 8, 56),
        PacketEvent(now + 0.1, "192.168.10.1", "192.168.10.10", "ICMP", None, None, 84, 20, 8, 56),
        PacketEvent(now + 0.2, "192.168.10.10", "192.168.10.1", "ICMP", None, None, 84, 20, 8, 56),
        PacketEvent(now + 0.3, "192.168.10.1", "192.168.10.10", "ICMP", None, None, 84, 20, 8, 56),
    ]

    flows = []
    for packet in packets:
        flows.extend(extractor.add_packet(packet))
    flows.extend(extractor.flush_all())

    print(f"Finalized flows: {len(flows)}")
    for flow in flows:
        print("Flow:", flow.summary())
        print("Vector length:", len(flow.normalized_values))
        preview = {k: round(v, 6) for k, v in list(flow.normalized_dict().items())[:10]}
        print("First 10 normalized features:", json.dumps(preview, indent=2))

        model = CICIDSPureNumpyModel("models/CICIDS_baseline (2).h5", "models/Mapping")
        pred = model.predict(flow.as_numpy_2d())[0]
        print(f"Prediction: {pred.predicted_label} confidence={pred.confidence:.4f}")


if __name__ == "__main__":
    main()
