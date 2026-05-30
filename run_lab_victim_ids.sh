#!/usr/bin/env bash
# Recommended lab runner.
#
# Use this when Kali is acting as the attacker-side router/sensor and you want
# the IDS to evaluate one protected lab victim instead of all Internet traffic.
#
# Usage:
#   sudo ./run_lab_victim_ids.sh eth0 192.168.10.20
#   sudo ./run_lab_victim_ids.sh eth0 192.168.10.20 dry-run
#   sudo ./run_lab_victim_ids.sh eth0 192.168.10.20 alert-only
#   sudo ./run_lab_victim_ids.sh eth0 192.168.10.20 enforce

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

IFACE="${1:-eth0}"
VICTIM_IP="${2:-}"
IPS_MODE="${3:-dry-run}"

if [[ -z "$VICTIM_IP" ]]; then
  echo "Usage: sudo ./run_lab_victim_ids.sh <iface> <victim-ip> [alert-only|dry-run|enforce]"
  echo "Example: sudo ./run_lab_victim_ids.sh eth0 192.168.10.20 dry-run"
  exit 2
fi

case "$IPS_MODE" in
  alert-only|dry-run|enforce) ;;
  *)
    echo "Invalid IPS mode: $IPS_MODE"
    echo "Use: alert-only, dry-run, or enforce"
    exit 2
    ;;
esac

VICTIM_TAG="${VICTIM_IP//[^A-Za-z0-9_.-]/_}"
FLOW_KEY_MODE="${FLOW_KEY_MODE:-service}"
BPF_FILTER="${BPF_FILTER:-tcp and host $VICTIM_IP}"
ML_ATTACK_PROB_THRESHOLD="${ML_ATTACK_PROB_THRESHOLD:-0.35}"
ALERT_FILE="${ALERT_FILE:-logs/alerts_lab_${VICTIM_TAG}.jsonl}"
CONSOLE_LOG="${CONSOLE_LOG:-logs/ids_console_lab_${VICTIM_TAG}.log}"

mkdir -p logs

if [[ $EUID -ne 0 ]]; then
  echo "This IDS needs root/sudo for packet capture. Re-running with sudo..."
  exec sudo --preserve-env=FLOW_KEY_MODE,BPF_FILTER,ML_ATTACK_PROB_THRESHOLD,ALERT_FILE,CONSOLE_LOG bash "$0" "$IFACE" "$VICTIM_IP" "$IPS_MODE"
fi

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
  PYTHON="venv/bin/python"
else
  PYTHON="python3"
fi

if [[ ! -f "models/CICIDS_baseline (2).h5" ]]; then
  echo "ML model file is missing: models/CICIDS_baseline (2).h5"
  exit 1
fi

cat <<INFO
============================================================
 Lab victim IDS runner started
============================================================
 Interface                : $IFACE
 Protected victim         : $VICTIM_IP
 BPF filter               : $BPF_FILTER
 IPS mode                 : $IPS_MODE
 ML mode                  : predict
 ML flow key mode         : $FLOW_KEY_MODE
 Attack-prob guard        : $ML_ATTACK_PROB_THRESHOLD
 Alert file               : $ALERT_FILE
 Console log              : $CONSOLE_LOG

 This runner intentionally ignores unrelated Internet traffic.
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
