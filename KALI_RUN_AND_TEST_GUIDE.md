# Kali IDS Run And Test Guide

This project is a Python IDS/IPS lab that watches packets on an interface, checks IOC rules, builds CICIDS-style flow features, runs the CICIDS ML model, and can optionally block attackers with `iptables`.

## What You Are Building

Your Kali box is the sensor. It must see the packets before the IDS can detect anything.

Good lab options:

1. Simple one-interface demo: attacker sends traffic directly to Kali `eth0`.
   - Easiest setup.
   - Good for proving the IDS works.
   - Less realistic because the attacker is attacking the IDS machine itself.

2. Better router demo: Kali routes traffic between attacker network and victim network.
   - Best simulation of an outside attacker reaching an internal victim.
   - Usually needs two interfaces, for example `eth0` attacker side and `eth1` victim side.
   - The IDS can sniff `eth0` for inbound attack traffic.

3. Passive sensor demo: Kali is not the gateway, but receives mirrored traffic.
   - Realistic for IDS-only monitoring.
   - Needs switch port mirroring, a tap, or VirtualBox/VMware network setup that lets Kali see the traffic.

Important: if attacker and victim are on the same subnet, and Kali is only their default gateway, their direct LAN traffic may not pass through Kali. In that case the IDS will not see the attack unless you attack Kali itself, put the victim on the other side of Kali, or mirror the traffic.

## Install On Kali

From inside the extracted project folder:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip tcpdump iptables nmap slowhttptest hping3 curl jq

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

chmod +x run_ml_only_ids.sh run_real_life_ids.sh
ls -lh "models/CICIDS_baseline (2).h5"
```

If the model file is missing, the IDS can still do IOC/rate checks, but ML prediction will not work. The expected model path is:

```text
models/CICIDS_baseline (2).h5
```

## Check The Interface

You said you are using `eth0`. Confirm Kali really sees traffic there:

```bash
ip -br addr
ip route
sudo tcpdump -i eth0 -n tcp
```

While `tcpdump` is running, generate traffic from the attacker. If `tcpdump` shows nothing, the IDS will also see nothing.

## Offline Tests First

Run these before live packet capture:

```bash
source .venv/bin/activate
python3 offline_ml_portscan_selftest.py
python3 offline_ml_slow_http_selftest.py
python3 -m py_compile *.py
```

Expected results:

```text
Prediction: PortScan confidence=...
OK: ML predicts slow HTTP attack class
```

## Start The IDS

Use dry-run first. Dry-run alerts and prints what it would block, but it does not change firewall rules.

Recommended for your lab:

```bash
sudo ./run_lab_victim_ids.sh eth0 <victim-ip> dry-run
```

This watches only traffic involving the protected victim. Use this first when
Kali is also the attacker's default route, because otherwise normal Internet
traffic such as browser, GitHub, Google, and `apt update` can create noisy ML
false positives.

General runner:

```bash
sudo ./run_ml_only_ids.sh eth0 dry-run
```

Watch logs from another terminal:

```bash
tail -f logs/ids_console_v17.log
tail -f logs/alerts_v17.jsonl
```

Use `enforce` only after dry-run works:

```bash
sudo ./run_ml_only_ids.sh eth0 enforce
```

Do not use `enforce` over SSH until you are sure the block rules will not lock you out.

## Do We Need Normal Traffic?

Yes, but for two different reasons:

- The CICIDS supervised model is already trained. It does not need your normal traffic to start predicting.
- You still need normal traffic to prove the IDS is not alerting on everyday behavior.
- You need normal traffic if you want to train the optional Isolation Forest anomaly detector.

Generate normal traffic for 5 to 10 minutes:

```bash
curl http://<victim-ip>/
curl http://<victim-ip>:8080/
ssh <user>@<victim-ip>
sudo apt update
```

Do not expect zero log lines. Expect no repeated `HIGH`, `CRITICAL`, `ML_ALERT`, or `IPS_DRY_RUN would block` messages for normal behavior.

## Train The Optional Anomaly Model

Capture normal flow features:

```bash
sudo .venv/bin/python main.py \
  --iface eth0 \
  --ml-mode features-only \
  --alerts logs/normal_flows.jsonl \
  --bpf tcp \
  --debug-decisions \
  --no-rate-guard \
  --no-correlation
```

Let it run during normal traffic, then stop with `Ctrl+C`.

Train:

```bash
python3 anomaly_engine.py \
  --train \
  --normal-log logs/normal_flows.jsonl \
  --model-out models/isolation_forest.pkl
```

After that, `run_ml_only_ids.sh` loads `models/isolation_forest.pkl` automatically.

## Attack Tests

Run these only inside your own isolated lab.

Port scan from attacker:

```bash
nmap -sS -Pn -p 1-1000 <victim-ip>
```

Expected IDS output:

```text
ML_PREDICTION ... class=PortScan ...
ML_ALERT ... class=PortScan ...
```

Slow HTTP test from attacker against an HTTP service:

```bash
slowhttptest -c 200 -H -g -o slowhttp \
  -i 10 -r 50 -t GET \
  -u http://<victim-ip>:8080/ \
  -x 24 -p 3
```

Expected IDS output:

```text
ML_PREDICTION ... class=DoS Slowhttptest ...
ML_ALERT ... ML_CLASS:DoS Slowhttptest ...
```

Rate/flood guard test:

```bash
sudo hping3 -S --fast -p 80 <victim-ip>
```

Expected IDS output:

```text
RATE_FLOOD_GUARD ...
IPS_DRY_RUN would block source_ip=...
```

## IOC Rule Test

To force a simple IOC alert, add the attacker IP:

```bash
printf '\n<attacker-ip>\n' | sudo tee -a data/blacklist_ips.csv
```

Then send any TCP traffic from attacker to victim or Kali. Expected:

```text
ALERT ... SRC_IP_BLACKLISTED:<attacker-ip>
```

The IDS reloads CSV rules every 30 seconds.

## If Alerts Do Not Appear

Check in this order:

1. `sudo tcpdump -i eth0 -n tcp` sees the traffic.
2. The model file exists at `models/CICIDS_baseline (2).h5`.
3. You started with `sudo`, because packet capture needs root.
4. Attacker and victim traffic actually crosses Kali.
5. The target has the service open, for example port `80` or `8080`.
6. You are reading the right log file: `logs/ids_console_v17.log` and `logs/alerts_v17.jsonl`.
