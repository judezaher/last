"""Professional IDS/IPS Dashboard for v17.

No external web framework is required. This uses Python's standard library.

Pages:
- Alerts: reads logs/alerts.jsonl and visualizes confirmed IDS/ML/correlation alerts.
- All Logs: reads logs/traffic.jsonl and merges alert records so you can see observed traffic
  and security events together.

Run:
    python dashboard.py --host 0.0.0.0 --port 8000
Open:
    http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
from collections import Counter, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


DEFAULT_ALERTS = Path("logs/alerts.jsonl")
DEFAULT_TRAFFIC = Path("logs/traffic.jsonl")


def _read_jsonl_tail(path: Path, limit: int = 5000) -> list[dict[str, Any]]:
    """Read the last N valid JSONL records from a log file."""
    if not path.exists():
        return []
    records: deque[dict[str, Any]] = deque(maxlen=max(1, int(limit)))
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                continue
    return list(records)


def _parse_time(record: dict[str, Any]) -> float:
    """Return a comparable timestamp in seconds."""
    val = record.get("created_at")
    if isinstance(val, (int, float)):
        return float(val)
    text = record.get("created_at_utc") or record.get("timestamp_utc")
    if isinstance(text, str):
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    pkt = record.get("packet") or {}
    if isinstance(pkt, dict):
        val = pkt.get("timestamp")
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


def _iso_from_ts(ts: float) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _extract_packet_or_flow(record: dict[str, Any]) -> dict[str, Any]:
    pkt = record.get("packet")
    if isinstance(pkt, dict):
        return pkt
    flow = record.get("flow")
    if isinstance(flow, dict):
        return flow
    return {}


def _detection_source(alert_type: str) -> str:
    if alert_type.startswith("IOC"):
        return "IOC Engine"
    if alert_type.startswith("ML"):
        return "CICIDS ML Model"
    if alert_type.startswith("CORRELATED"):
        return "Correlation Engine"
    if "SCAN" in alert_type:
        return "Scan/Correlation Logic"
    if "IPS" in alert_type:
        return "IPS Enforcer"
    return "IDS Core"


def _normalize_alert(record: dict[str, Any]) -> dict[str, Any]:
    pkt = _extract_packet_or_flow(record)
    alert_type = str(record.get("alert_type") or record.get("event_type") or "UNKNOWN")
    reasons = record.get("reasons")
    if isinstance(reasons, list):
        reason_text = " | ".join(str(x) for x in reasons)
    elif reasons is None:
        reason_text = ""
    else:
        reason_text = str(reasons)
    ts = _parse_time(record)
    src_ip = record.get("source_ip") or pkt.get("src_ip") or record.get("src_ip") or ""
    dst_ip = pkt.get("dst_ip") or record.get("dst_ip") or ""
    confidence = ""
    for r in reasons or []:
        if isinstance(r, str) and ("CONFIDENCE" in r or "PROBABILITY" in r):
            confidence = r.split(":", 1)[-1]
            break
    return {
        "kind": "ALERT",
        "time": _iso_from_ts(ts),
        "ts": ts,
        "type": alert_type,
        "source_module": _detection_source(alert_type),
        "severity": str(record.get("severity") or "INFO"),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": pkt.get("src_port"),
        "dst_port": pkt.get("dst_port"),
        "protocol": pkt.get("protocol", ""),
        "confidence": confidence,
        "reasons": reason_text,
        "summary": str(record.get("summary") or reason_text or alert_type),
        "raw": record,
    }


def _normalize_traffic(record: dict[str, Any]) -> dict[str, Any]:
    pkt = _extract_packet_or_flow(record)
    ts = _parse_time(record)
    return {
        "kind": "TRAFFIC",
        "time": _iso_from_ts(ts),
        "ts": ts,
        "type": "TRAFFIC",
        "source_module": "Packet Capture",
        "severity": "INFO",
        "src_ip": record.get("src_ip") or pkt.get("src_ip") or "",
        "dst_ip": record.get("dst_ip") or pkt.get("dst_ip") or "",
        "src_port": record.get("src_port") if record.get("src_port") is not None else pkt.get("src_port"),
        "dst_port": record.get("dst_port") if record.get("dst_port") is not None else pkt.get("dst_port"),
        "protocol": record.get("protocol") or pkt.get("protocol") or "",
        "length": record.get("length") if record.get("length") is not None else pkt.get("length"),
        "summary": str(record.get("summary") or ""),
        "raw": record,
    }


def _bucket_timeline(records: list[dict[str, Any]], bucket_seconds: int = 30) -> list[dict[str, Any]]:
    ts_values = [float(r.get("ts") or 0) for r in records if r.get("ts")]
    if not ts_values:
        return []
    end = max(ts_values)
    start = max(min(ts_values), end - bucket_seconds * 24)
    buckets: Counter[int] = Counter()
    for ts in ts_values:
        if ts < start:
            continue
        b = int((ts - start) // bucket_seconds)
        buckets[b] += 1
    out = []
    total_buckets = int((end - start) // bucket_seconds) + 1
    for b in range(total_buckets):
        t = start + b * bucket_seconds
        out.append({"time": _iso_from_ts(t), "count": buckets.get(b, 0)})
    return out


def _summary(alerts: list[dict[str, Any]], logs: list[dict[str, Any]]) -> dict[str, Any]:
    alert_types = Counter(a.get("type", "UNKNOWN") for a in alerts)
    alert_sources = Counter(a.get("source_module", "UNKNOWN") for a in alerts)
    severities = Counter(a.get("severity", "INFO") for a in alerts)
    protocols = Counter(l.get("protocol", "UNKNOWN") or "UNKNOWN" for l in logs if l.get("kind") == "TRAFFIC")
    top_src = Counter((x.get("src_ip") or "unknown") for x in logs).most_common(10)
    top_dst = Counter((x.get("dst_ip") or "unknown") for x in logs).most_common(10)
    return {
        "alerts_total": len(alerts),
        "logs_total": len(logs),
        "alert_types": dict(alert_types),
        "alert_sources": dict(alert_sources),
        "severities": dict(severities),
        "protocols": dict(protocols),
        "top_sources": top_src,
        "top_destinations": top_dst,
        "alert_timeline": _bucket_timeline(alerts, 30),
        "log_timeline": _bucket_timeline(logs, 30),
    }


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>IDS/IPS SOC Dashboard</title>
<style>
:root{--bg:#090d18;--panel:#121a2a;--panel2:#182234;--border:#26364f;--text:#e6edf7;--muted:#94a3b8;--blue:#38bdf8;--green:#22c55e;--yellow:#f59e0b;--red:#ef4444;--purple:#a78bfa;--cyan:#06b6d4}
*{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at top right,#13213a 0,#090d18 44%,#060914 100%);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif} 
header{padding:18px 22px;border-bottom:1px solid var(--border);background:rgba(9,13,24,.86);position:sticky;top:0;z-index:2;backdrop-filter:blur(10px)}
h1{margin:0;font-size:22px;letter-spacing:.3px}.subtitle{color:var(--muted);margin-top:5px;font-size:13px}.row{display:flex;gap:14px;flex-wrap:wrap}.tabs{padding:14px 22px;border-bottom:1px solid var(--border)}button{background:#0e1728;color:var(--text);border:1px solid var(--border);border-radius:8px;padding:10px 14px;cursor:pointer}button.active{background:#075985;border-color:#38bdf8}.page{display:none;padding:18px 22px}.page.active{display:block}.card{background:rgba(18,26,42,.92);border:1px solid var(--border);border-radius:12px;padding:14px;box-shadow:0 8px 28px rgba(0,0,0,.25)}.metric{min-width:190px;flex:1}.metric .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}.metric .value{font-size:30px;margin-top:8px}.grid{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:14px;margin:14px 0}@media(max-width:900px){.grid{grid-template-columns:1fr}}.chart-title{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}.chart{height:210px;display:flex;align-items:center;justify-content:center}.bars{height:180px;display:flex;gap:5px;align-items:end;width:100%;border-left:1px solid #334155;border-bottom:1px solid #334155;padding:8px}.bar{flex:1;min-width:4px;background:linear-gradient(180deg,var(--blue),#1d4ed8);border-radius:4px 4px 0 0;position:relative}.bar:hover:after{content:attr(data-tip);position:absolute;bottom:105%;left:0;background:#020617;border:1px solid var(--border);padding:6px;border-radius:6px;font-size:11px;white-space:nowrap;z-index:5}.hbar{display:grid;grid-template-columns:150px 1fr 45px;gap:8px;align-items:center;margin:8px 0}.hbar-label{color:#cbd5e1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.hbar-fill{height:12px;background:#1e293b;border-radius:99px;overflow:hidden}.hbar-fill div{height:100%;background:linear-gradient(90deg,var(--cyan),var(--purple));}.filters{display:flex;gap:10px;margin:14px 0;flex-wrap:wrap}.filters input,.filters select{background:#0b1220;color:var(--text);border:1px solid var(--border);border-radius:8px;padding:9px 10px}table{width:100%;border-collapse:collapse;background:rgba(18,26,42,.84);border:1px solid var(--border);border-radius:12px;overflow:hidden}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #223047;font-size:13px;vertical-align:top}th{background:#071020;color:#cbd5e1;text-transform:uppercase;font-size:11px;letter-spacing:.06em}tr:hover{background:#162238}.pill{display:inline-block;border:1px solid var(--border);border-radius:999px;padding:3px 8px;font-size:12px;background:#0b1220}.sev-HIGH,.sev-CRITICAL{color:#fecaca;border-color:#ef4444;background:rgba(239,68,68,.13)}.sev-MEDIUM{color:#fde68a;border-color:#f59e0b;background:rgba(245,158,11,.13)}.sev-INFO{color:#bfdbfe;border-color:#38bdf8;background:rgba(56,189,248,.10)}.proto{color:#bfdbfe;border-color:#2563eb}.small{font-size:12px;color:var(--muted)}.footer{padding:12px 22px;color:var(--muted);font-size:12px}.status-dot{display:inline-block;width:8px;height:8px;border-radius:99px;background:var(--green);box-shadow:0 0 10px var(--green);margin-right:6px}.source-tag{color:#d8b4fe}.summary{max-width:520px;white-space:normal}.raw-toggle{font-size:11px;color:#93c5fd;cursor:pointer}.raw{display:none;white-space:pre-wrap;max-height:180px;overflow:auto;background:#020617;border:1px solid var(--border);padding:8px;border-radius:8px;margin-top:6px;color:#cbd5e1}
</style>
</head>
<body>
<header><h1><span class="status-dot"></span>IDS/IPS SOC Dashboard</h1><div class="subtitle">Professional dashboard for v17 — auto-refreshes every 3 seconds. Alerts page + All Logs page with timelines and source analytics.</div></header>
<div class="tabs"><button id="tabAlerts" class="active" onclick="showPage('alerts')">Alerts</button><button id="tabLogs" onclick="showPage('logs')">All Logs</button><span class="small" style="margin-left:12px">Last refresh: <span id="lastRefresh">-</span></span></div>
<section id="pageAlerts" class="page active">
  <div class="row" id="alertMetrics"></div>
  <div class="grid"><div class="card"><div class="chart-title">Alert timeline</div><div id="alertTimeline" class="chart"></div></div><div class="card"><div class="chart-title">Where alerts are coming from</div><div id="alertSources"></div></div><div class="card"><div class="chart-title">Alert types</div><div id="alertTypes"></div></div><div class="card"><div class="chart-title">Top alerting sources</div><div id="alertTopSources"></div></div></div>
  <div class="filters"><input id="alertSearch" placeholder="Search alerts: IP, type, reason..." oninput="renderAlerts()"><select id="alertSeverity" onchange="renderAlerts()"><option value="">All severities</option><option>CRITICAL</option><option>HIGH</option><option>MEDIUM</option><option>INFO</option></select><select id="alertSource" onchange="renderAlerts()"><option value="">All engines</option></select></div>
  <div class="card"><div class="chart-title">Recent confirmed alerts</div><table><thead><tr><th>Time</th><th>Type</th><th>Engine</th><th>Severity</th><th>Source</th><th>Destination</th><th>Protocol</th><th>Confidence</th><th>Reasons / Summary</th></tr></thead><tbody id="alertsTable"></tbody></table></div>
</section>
<section id="pageLogs" class="page">
  <div class="row" id="logMetrics"></div>
  <div class="grid"><div class="card"><div class="chart-title">Log timeline</div><div id="logTimeline" class="chart"></div></div><div class="card"><div class="chart-title">Protocol distribution</div><div id="protocols"></div></div><div class="card"><div class="chart-title">Top sources</div><div id="topSources"></div></div><div class="card"><div class="chart-title">Top destinations</div><div id="topDestinations"></div></div></div>
  <div class="filters"><input id="logSearch" placeholder="Search logs: IP, protocol, summary..." oninput="renderLogs()"><select id="logKind" onchange="renderLogs()"><option value="">All log kinds</option><option>TRAFFIC</option><option>ALERT</option></select><select id="logProtocol" onchange="renderLogs()"><option value="">All protocols</option></select></div>
  <div class="card"><div class="chart-title">All logs: traffic + alert records</div><table><thead><tr><th>Time</th><th>Kind</th><th>Engine</th><th>Protocol</th><th>Source</th><th>Src Port</th><th>Destination</th><th>Dst Port</th><th>Length</th><th>Summary</th></tr></thead><tbody id="logsTable"></tbody></table></div>
</section>
<div class="footer">Tip: run IDS with <code>--log-traffic</code> to fill the All Logs page. Alerts are read from <code>logs/alerts.jsonl</code>; traffic is read from <code>logs/traffic.jsonl</code>.</div>
<script>
let alerts=[], logs=[], summary={};
function showPage(name){document.getElementById('pageAlerts').classList.toggle('active',name==='alerts');document.getElementById('pageLogs').classList.toggle('active',name==='logs');document.getElementById('tabAlerts').classList.toggle('active',name==='alerts');document.getElementById('tabLogs').classList.toggle('active',name==='logs')}
async function loadData(){try{const [a,l,s]=await Promise.all([fetch('/api/alerts?limit=1000').then(r=>r.json()),fetch('/api/logs?limit=2000').then(r=>r.json()),fetch('/api/summary?limit=2000').then(r=>r.json())]);alerts=a.records||[];logs=l.records||[];summary=s;document.getElementById('lastRefresh').textContent=new Date().toLocaleTimeString();populateFilters();renderAll();}catch(e){console.error(e)}}
function populateFilters(){const srcSel=document.getElementById('alertSource');const current=srcSel.value;const engines=[...new Set(alerts.map(a=>a.source_module).filter(Boolean))].sort();srcSel.innerHTML='<option value="">All engines</option>'+engines.map(e=>`<option>${esc(e)}</option>`).join('');srcSel.value=current;const pSel=document.getElementById('logProtocol');const pc=pSel.value;const prots=[...new Set(logs.map(l=>l.protocol).filter(Boolean))].sort();pSel.innerHTML='<option value="">All protocols</option>'+prots.map(p=>`<option>${esc(p)}</option>`).join('');pSel.value=pc;}
function metric(label,value){return `<div class="card metric"><div class="label">${esc(label)}</div><div class="value">${value}</div></div>`}
function renderAll(){renderMetrics();renderCharts();renderAlerts();renderLogs()}
function renderMetrics(){const high=alerts.filter(a=>['HIGH','CRITICAL'].includes(a.severity)).length;const ioc=alerts.filter(a=>(a.type||'').startsWith('IOC')).length;const ml=alerts.filter(a=>(a.type||'').startsWith('ML')||String(a.source_module).includes('ML')).length;document.getElementById('alertMetrics').innerHTML=metric('Recent alerts',alerts.length)+metric('High / Critical',high)+metric('IOC alerts',ioc)+metric('ML / Correlation alerts',ml);const traffic=logs.filter(l=>l.kind==='TRAFFIC').length;const alertLogs=logs.filter(l=>l.kind==='ALERT').length;const tcp=logs.filter(l=>l.protocol==='TCP').length;const udp=logs.filter(l=>l.protocol==='UDP').length;document.getElementById('logMetrics').innerHTML=metric('All log records',logs.length)+metric('Traffic packets',traffic)+metric('Alert records',alertLogs)+metric('TCP / UDP',`${tcp} / ${udp}`)}
function renderCharts(){barTimeline('alertTimeline',summary.alert_timeline||[]);barTimeline('logTimeline',summary.log_timeline||[]);hbars('alertSources',summary.alert_sources||{});hbars('alertTypes',summary.alert_types||{});hbars('protocols',summary.protocols||{});hbarsFromPairs('topSources',summary.top_sources||[]);hbarsFromPairs('topDestinations',summary.top_destinations||[]);const ats={};alerts.forEach(a=>ats[a.src_ip||'unknown']=(ats[a.src_ip||'unknown']||0)+1);hbars('alertTopSources',ats)}
function barTimeline(id,data){const el=document.getElementById(id);if(!data.length){el.innerHTML='<div class="small">No timestamped data yet</div>';return}const max=Math.max(...data.map(x=>x.count),1);el.innerHTML=`<div class="bars">${data.map(x=>`<div class="bar" style="height:${Math.max(2,x.count/max*100)}%" data-tip="${new Date(x.time).toLocaleTimeString()} : ${x.count}"></div>`).join('')}</div>`}
function hbars(id,obj){const entries=Object.entries(obj).sort((a,b)=>b[1]-a[1]).slice(0,8);const el=document.getElementById(id);if(!entries.length){el.innerHTML='<div class="small">No data yet</div>';return}const max=Math.max(...entries.map(e=>e[1]),1);el.innerHTML=entries.map(([k,v])=>`<div class="hbar"><div class="hbar-label">${esc(k)}</div><div class="hbar-fill"><div style="width:${v/max*100}%"></div></div><div>${v}</div></div>`).join('')}
function hbarsFromPairs(id,pairs){const obj={};pairs.forEach(p=>obj[p[0]]=p[1]);hbars(id,obj)}
function renderAlerts(){const q=document.getElementById('alertSearch').value.toLowerCase();const sev=document.getElementById('alertSeverity').value;const eng=document.getElementById('alertSource').value;let rows=alerts.filter(a=>(!sev||a.severity===sev)&&(!eng||a.source_module===eng));if(q){rows=rows.filter(a=>JSON.stringify(a).toLowerCase().includes(q))}rows=rows.sort((a,b)=>(b.ts||0)-(a.ts||0)).slice(0,300);document.getElementById('alertsTable').innerHTML=rows.map((a,i)=>`<tr><td>${fmtTime(a.time)}</td><td><span class="pill">${esc(a.type)}</span></td><td><span class="source-tag">${esc(a.source_module)}</span></td><td><span class="pill sev-${esc(a.severity)}">${esc(a.severity)}</span></td><td>${esc(a.src_ip||'')}</td><td>${esc(a.dst_ip||'')}</td><td>${a.protocol?`<span class="pill proto">${esc(a.protocol)}</span>`:''}</td><td>${esc(a.confidence||'')}</td><td class="summary">${esc(a.reasons||a.summary||'')}<div class="raw-toggle" onclick="toggleRaw('araw${i}')">raw</div><pre id="araw${i}" class="raw">${esc(JSON.stringify(a.raw,null,2))}</pre></td></tr>`).join('')||'<tr><td colspan="9" class="small">No alerts yet</td></tr>'}
function renderLogs(){const q=document.getElementById('logSearch').value.toLowerCase();const kind=document.getElementById('logKind').value;const proto=document.getElementById('logProtocol').value;let rows=logs.filter(l=>(!kind||l.kind===kind)&&(!proto||l.protocol===proto));if(q){rows=rows.filter(l=>JSON.stringify(l).toLowerCase().includes(q))}rows=rows.sort((a,b)=>(b.ts||0)-(a.ts||0)).slice(0,500);document.getElementById('logsTable').innerHTML=rows.map(l=>`<tr><td>${fmtTime(l.time)}</td><td><span class="pill ${l.kind==='ALERT'?'sev-HIGH':''}">${esc(l.kind)}</span></td><td><span class="source-tag">${esc(l.source_module||'')}</span></td><td>${l.protocol?`<span class="pill proto">${esc(l.protocol)}</span>`:''}</td><td>${esc(l.src_ip||'')}</td><td>${val(l.src_port)}</td><td>${esc(l.dst_ip||'')}</td><td>${val(l.dst_port)}</td><td>${val(l.length)}</td><td class="summary">${esc(l.summary||'')}</td></tr>`).join('')||'<tr><td colspan="10" class="small">No logs yet. Start IDS with --log-traffic.</td></tr>'}
function toggleRaw(id){const el=document.getElementById(id);el.style.display=el.style.display==='block'?'none':'block'}
function fmtTime(t){if(!t)return'';try{return new Date(t).toLocaleString()}catch(e){return t}}
function val(v){return v===null||v===undefined?'':esc(String(v))}
function esc(s){return String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]))}
loadData();setInterval(loadData,3000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    alerts_path = DEFAULT_ALERTS
    traffic_path = DEFAULT_TRAFFIC

    def _send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, text: str) -> None:
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        limit = int(params.get("limit", ["2000"])[0])

        if parsed.path in ("/", "/index.html"):
            self._send_html(HTML)
            return

        if parsed.path == "/api/alerts":
            alerts = [_normalize_alert(x) for x in _read_jsonl_tail(self.alerts_path, limit)]
            self._send_json({"records": alerts, "count": len(alerts)})
            return

        if parsed.path == "/api/traffic":
            traffic = [_normalize_traffic(x) for x in _read_jsonl_tail(self.traffic_path, limit)]
            self._send_json({"records": traffic, "count": len(traffic)})
            return

        if parsed.path == "/api/logs":
            traffic = [_normalize_traffic(x) for x in _read_jsonl_tail(self.traffic_path, limit)]
            alerts = [_normalize_alert(x) for x in _read_jsonl_tail(self.alerts_path, max(200, limit // 4))]
            logs = traffic + alerts
            logs.sort(key=lambda r: r.get("ts") or 0, reverse=True)
            self._send_json({"records": logs[:limit], "count": len(logs[:limit])})
            return

        if parsed.path == "/api/summary":
            alerts = [_normalize_alert(x) for x in _read_jsonl_tail(self.alerts_path, limit)]
            traffic = [_normalize_traffic(x) for x in _read_jsonl_tail(self.traffic_path, limit)]
            logs = traffic + alerts
            logs.sort(key=lambda r: r.get("ts") or 0, reverse=True)
            self._send_json(_summary(alerts, logs))
            return

        self._send_json({"error": "not found"}, status=404)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep terminal clean; comment this line if you want HTTP access logs.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="IDS/IPS SOC dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--alerts", default=str(DEFAULT_ALERTS))
    parser.add_argument("--traffic", default=str(DEFAULT_TRAFFIC))
    args = parser.parse_args()

    DashboardHandler.alerts_path = Path(args.alerts)
    DashboardHandler.traffic_path = Path(args.traffic)

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running on http://{args.host}:{args.port}")
    print(f"Alerts log:  {DashboardHandler.alerts_path}")
    print(f"Traffic log: {DashboardHandler.traffic_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
