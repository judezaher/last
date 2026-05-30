#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
sudo .venv/bin/python main.py \
  --iface eth0 \
  --log-traffic \
  --ips-mode dry-run \
  --ml-mode predict \
  --debug-decisions
