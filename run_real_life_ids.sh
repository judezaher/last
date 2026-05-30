#!/usr/bin/env bash
# ML-only real-life style IDS runner for this lab project.
# Port-scan detection is handled by the CICIDS ML model only.
# No separate scan_detector module is started.
#
# Usage examples:
#   sudo ./run_real_life_ids.sh eth0
#   sudo ./run_real_life_ids.sh eth0 alert-only
#   sudo ./run_real_life_ids.sh eth0 dry-run
#   sudo ./run_real_life_ids.sh eth0 enforce
#
# Default: dry-run. It alerts and shows what would be blocked, but does not block.
# Use enforce only inside your own isolated VirtualBox lab.

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

IFACE="${1:-eth0}"
IPS_MODE="${2:-dry-run}"          # alert-only | dry-run | enforce
FLOW_KEY_MODE="${FLOW_KEY_MODE:-service}"  # service mode lets ML see Slowloris as one CICIDS flow
BPF_FILTER="${BPF_FILTER:-tcp}"
ML_ATTACK_PROB_THRESHOLD="${ML_ATTACK_PROB_THRESHOLD:-0.35}"
ALERT_FILE="${ALERT_FILE:-logs/alerts_ml_only_real_life.jsonl}"
CONSOLE_LOG="${CONSOLE_LOG:-logs/ids_console_ml_only_real_life.log}"

case "$IPS_MODE" in
  alert-only|dry-run|enforce) ;;
  *)
    echo "Invalid IPS mode: $IPS_MODE"
    echo "Use: alert-only, dry-run, or enforce"
    exit 2
    ;;
esac

mkdir -p logs

if [[ $EUID -ne 0 ]]; then
  echo "This IDS needs root/sudo for packet capture. Re-running with sudo..."
  exec sudo --preserve-env=FLOW_KEY_MODE,BPF_FILTER,ML_ATTACK_PROB_THRESHOLD,ALERT_FILE,CONSOLE_LOG bash "$0" "$IFACE" "$IPS_MODE"
fi

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
  PYTHON="venv/bin/python"
else
  PYTHON="python3"
fi

if [[ ! -f "main.py" ]]; then
  echo "main.py was not found. Put this script inside the ids_ioc_starter folder."
  exit 1
fi

if [[ ! -f "models/CICIDS_baseline (2).h5" ]]; then
  echo "ML model file is missing: models/CICIDS_baseline (2).h5"
  exit 1
fi

cat <<INFO
============================================================
 ML-only IDS runner started
============================================================
 Interface        : $IFACE
 BPF filter       : $BPF_FILTER
 IPS mode         : $IPS_MODE
ML mode          : predict
ML flow key mode : $FLOW_KEY_MODE
Attack-prob guard: $ML_ATTACK_PROB_THRESHOLD
Alert file       : $ALERT_FILE
Console log      : $CONSOLE_LOG

 Active detection modules:
  - IOC detection: blacklist IPs, whitelist IPs, blacklist ports
  - ML detection: CICIDS 72-feature model prediction
  - ML guard: non-BENIGN probability guard
  - IPS action queue: $IPS_MODE

 Removed:
  - Separate port-scan detector is OFF/REMOVED
  - PortScan/Slow HTTP alerts must come from ML_PREDICTION / ML_ALERT only

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
)

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" 2>&1 | tee -a "$CONSOLE_LOG"
