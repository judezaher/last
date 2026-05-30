"""Offline self-test for the lab Slowloris/slow HTTP traffic shape.

This does not send any traffic. It creates PacketEvent objects similar to the
user's lab script and verifies that the ML model predicts DoS Slowhttptest.
"""
from packet_capture import PacketEvent
from cicids_live_features import CICIDSFlowExtractor, CICIDSNormalizer
from cicids_model import CICIDSPureNumpyModel, load_feature_columns

ATTACKER_IP = "192.168.10.10"
TARGET_IP = "192.168.10.20"
TARGET_PORT = 8080
CONNECTIONS = 20
SLEEP_SECONDS = 5
DURATION_SECONDS = 60

features = load_feature_columns("models/cicids_feature_columns.json")
normalizer = CICIDSNormalizer.from_json("models/live_feature_normalizer.json", features)
model = CICIDSPureNumpyModel("models/CICIDS_baseline (2).h5", "models/Mapping")
extractor = CICIDSFlowExtractor(
    features,
    normalizer,
    idle_timeout_seconds=90,
    active_timeout_seconds=30,
    min_packets=2,
    flow_key_mode="service",
    http_remap_ports=(8080, 8000, 8443),
)

flows = []
t0 = 1000.0
for i in range(CONNECTIONS):
    sport = 34000 + i
    ts = t0 + i * 0.01
    events = [
        PacketEvent(ts, ATTACKER_IP, TARGET_IP, "TCP", sport, TARGET_PORT, 60, 20, 20, 0, "S", 64240),
        PacketEvent(ts + 0.001, TARGET_IP, ATTACKER_IP, "TCP", TARGET_PORT, sport, 60, 20, 20, 0, "SA", 64240),
        PacketEvent(ts + 0.002, ATTACKER_IP, TARGET_IP, "TCP", sport, TARGET_PORT, 52, 20, 20, 0, "A", 64240),
        PacketEvent(ts + 0.003, ATTACKER_IP, TARGET_IP, "TCP", sport, TARGET_PORT, 56, 20, 20, len(b"GET / HTTP/1.1\r\n"), "PA", 64240),
        PacketEvent(ts + 0.004, ATTACKER_IP, TARGET_IP, "TCP", sport, TARGET_PORT, 72, 20, 20, len(f"Host: {TARGET_IP}\r\n".encode()), "PA", 64240),
    ]
    for event in events:
        flows.extend(extractor.add_packet(event))

for sec in range(SLEEP_SECONDS, DURATION_SECONDS + 1, SLEEP_SECONDS):
    for i in range(CONNECTIONS):
        sport = 34000 + i
        ts = t0 + sec + i * 0.001
        event = PacketEvent(ts, ATTACKER_IP, TARGET_IP, "TCP", sport, TARGET_PORT, 70, 20, 20, len(b"X-a: keepalive\r\n"), "PA", 64240)
        flows.extend(extractor.add_packet(event))

flows.extend(extractor.flush_all())
print(f"Finalized ML flows: {len(flows)}")

ok = False
for flow in flows:
    prediction = model.predict(flow.as_numpy_2d())[0]
    top3 = sorted(prediction.probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
    print(flow.summary())
    print("Prediction:", prediction.predicted_label, f"confidence={prediction.confidence:.4f}", "top3=", top3)
    if prediction.predicted_label in {"DoS Slowhttptest", "DoS slowloris"}:
        ok = True

if not ok:
    raise SystemExit("Self-test failed: ML did not predict slow HTTP class")
print("OK: ML predicts slow HTTP attack class")
