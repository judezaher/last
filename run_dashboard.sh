#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
python dashboard.py --host 0.0.0.0 --port 8000
