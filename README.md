# IDS IOC + CICIDS ML Lab Project

This build is ML-only for PortScan and Slow HTTP / Slowloris-style detection. The separate scan detector is removed.

For the full Kali setup, topology notes, normal-traffic training flow, and attack test commands, read:

```text
KALI_RUN_AND_TEST_GUIDE.md
```

Run:

```bash
cd ~/Desktop/IDS/ids_ioc_starter
chmod +x run_real_life_ids.sh
sudo ./run_real_life_ids.sh eth0
```

Default ML settings are tuned for the lab:
- ML mode: predict
- Flow key mode: service
- HTTP remap: 8080,8000,8443 -> 80 inside the ML feature vector
- Active timeout: 30 seconds
- Idle timeout: 90 seconds

Offline tests:

```bash
python3 offline_ml_portscan_selftest.py
python3 offline_ml_slow_http_selftest.py
```

## GUI Dashboard

This version includes a professional local SOC-style dashboard.

Run the IDS with traffic logging:

```bash
sudo .venv/bin/python main.py --iface eth0 --log-traffic --ips-mode dry-run --ml-mode predict --debug-decisions
```

Run the dashboard:

```bash
python dashboard.py --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

See `docs/DASHBOARD_V17_GUIDE.md` for details.
