#!/usr/bin/env bash
# IDS v17 ML-only runner — includes all graduation project improvements:
#   - ICMP filtered from ML pipeline
#   - Per-class ML thresholds
# # #   - Alert correlation engine
#
# Usage:
#   sudo ./run_ml_only_ids.sh eth0
#   sudo ./run_ml_only_ids.sh eth0 alert-only
#   sudo ./run_ml_only_ids.sh eth0 enforce
#
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

IFACE="${1:-eth0}"
IPS_MODE="${2:-dry-run}"
FLOW_KEY_MODE="${FLOW_KEY_MODE:-service}"
BPF_FILTER="${BPF_FILTER:-tcp}"
ML_ATTACK_PROB_THRESHOLD="${ML_ATTACK_PROB_THRESHOLD:-0.35}"
ALERT_FILE="${ALERT_FILE:-logs/alerts_v17.jsonl}"
CONSOLE_LOG="${CONSOLE_LOG:-logs/ids_console_v17.log}"

case "$IPS_MODE" in
  alert-only|dry-run|enforce) ;;
  *)
    echo "Invalid IPS mode: $IPS_MODE. Use: alert-only, dry-run, or enforce"
    exit 2
    ;;
esac

mkdir -p logs

if [[ $EUID -ne 0 ]]; then
  echo "Needs root for packet capture. Re-running with sudo..."
  exec sudo --preserve-env=FLOW_KEY_MODE,BPF_FILTER,ML_ATTACK_PROB_THRESHOLD,ALERT_FILE,CONSOLE_LOG bash "$0" "$IFACE" "$IPS_MODE"
fi

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
  PYTHON="venv/bin/python"
else
  PYTHON="python3"
fi

cat <<INFO
============================================================
 IDS v17 started
============================================================
 Interface        : $IFACE
 BPF filter       : $BPF_FILTER
 IPS mode         : $IPS_MODE
 ML mode          : predict
 Flow key mode    : $FLOW_KEY_MODE (auto-overridden per port)
Alert file       : $ALERT_FILE
Attack-prob guard: $ML_ATTACK_PROB_THRESHOLD

Active detection layers:
  [1] IOC blacklist/whitelist (rule-based, every packet)
  [3] CICIDS ML model (deep learning, per-class thresholds)
  [3] ML probability guard (early attack signal)
  [4] Alert correlation engine (escalates combined signals)

 ICMP flows: skipped from ML pipeline (not in training data)
 Press Ctrl+C to stop.
============================================================
INFO

CMD=(
  "$PYTHON" main.py
  --iface "$IFACE"
  --data-dir data
  --alerts "$ALERT_FILE"
  --workers 4
  --queue-size 20000
  --reload-seconds 30
  --dedup-seconds 10
  --dedup-mode source-reason
  --bpf "$BPF_FILTER"
  --debug-decisions
  --ips-mode "$IPS_MODE"
  --block-chains INPUT,FORWARD
  --block-seconds 300
  --ml-mode predict
  --ml-model-path "models/CICIDS_baseline (2).h5"
  --ml-features-path "models/cicids_feature_columns.json"
  --ml-mapping-path "models/Mapping"
  --ml-normalizer-path "models/live_feature_normalizer.json"
  --ml-workers 1
  --ml-threshold 0.60
  --ml-attack-prob-threshold "$ML_ATTACK_PROB_THRESHOLD"
  --flow-idle-timeout 90
  --flow-active-timeout 30
  --min-flow-packets 2
  --ml-flow-key-mode "$FLOW_KEY_MODE"
  --ml-http-remap-ports "8080,8000,8443"
  --correlation
  --scan-dst-threshold 10
)

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" 2>&1 | tee -a "$CONSOLE_LOG"
