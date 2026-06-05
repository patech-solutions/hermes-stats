import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

app = FastAPI(title="Hermes Dashboard")

DB_PATH = Path.home() / ".hermes/state.db"
SESSIONS_DIR = Path.home() / ".hermes/sessions"
AGENT_LOG = Path.home() / ".hermes/logs/agent.log"
MAX_CTX = 40960
CHARS_PER_TOKEN = 4  # rough estimate for JSONL sessions without token counts


# ── Helpers ──────────────────────────────────────────────────────────────────

def db():
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def read_jsonl(session_id: str) -> list[dict]:
    path = SESSIONS_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return []
    msgs = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                msgs.append(json.loads(line))
            except Exception:
                pass
    return msgs


def jsonl_metrics(session_id: str) -> dict:
    """Extract turns, estimated tokens, tools, duration from a JSONL file."""
    msgs = read_jsonl(session_id)
    if not msgs:
        return {}

    turns = 0
    tool_names: set[str] = set()
    total_chars = 0
    timestamps = []
    model = None

    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        ts = m.get("timestamp")
        if ts:
            try:
                timestamps.append(float(ts) if not isinstance(ts, str) else
                    datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
            except Exception:
                pass

        if role == "system" and not model:
            model = m.get("model")

        total_chars += len(str(content))

        if role == "assistant":
            tool_calls = m.get("tool_calls")
            if not tool_calls:
                turns += 1  # final assistant response = 1 turn
            else:
                try:
                    calls = json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls
                    for c in calls:
                        name = c.get("function", {}).get("name") or c.get("name", "")
                        if name:
                            tool_names.add(name)
                except Exception:
                    pass

        elif role == "tool":
            name = m.get("name") or m.get("tool_name", "")
            if name:
                tool_names.add(name)

    # First/last timestamp for duration
    started_at = min(timestamps) if timestamps else None
    ended_at = max(timestamps) if timestamps else None
    duration_s = round(ended_at - started_at, 1) if (started_at and ended_at and ended_at > started_at) else None

    # Estimate tokens from total content length
    est_input_tokens = (total_chars // CHARS_PER_TOKEN // max(turns, 1)) * max(turns, 1)
    est_input_tokens = min(est_input_tokens, MAX_CTX)

    return {
        "api_call_count": turns,
        "tool_call_count": len(tool_names),
        "tools_used": list(tool_names),
        "input_tokens": est_input_tokens,
        "duration_s": duration_s,
        "started_at": started_at,
        "ended_at": ended_at,
        "model": model,
        "source": "jsonl",
    }


def parse_log_timings() -> list:
    if not AGENT_LOG.exists():
        return []
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*response ready:.*time=([\d.]+)s api_calls=(\d+)"
    )
    result = []
    for line in AGENT_LOG.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            result.append({"ts": ts, "time_s": float(m.group(2)), "api_calls": int(m.group(3))})
    return result


def parse_memory_log() -> list:
    if not AGENT_LOG.exists():
        return []
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[MEMORY\] .*rss=(\d+)MB")
    result = []
    for line in AGENT_LOG.read_text(errors="replace").splitlines():
        m = pattern.search(line)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            result.append({"ts": ts, "ts_str": m.group(1), "rss_mb": int(m.group(2))})
    return result


def db_messages_metrics(session_id: str) -> dict:
    """Derive turns, estimated tokens, and tools from state.db messages table."""
    con = db()
    cur = con.cursor()
    cur.execute(
        "SELECT role, token_count, LENGTH(content) as clen, tool_name, timestamp "
        "FROM messages WHERE session_id=? AND active=1 ORDER BY timestamp",
        (session_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    if not rows:
        return {}
    turns = sum(1 for r in rows if r["role"] == "user")
    tool_names = {r["tool_name"] for r in rows if r["tool_name"]}
    total_chars = sum(r["clen"] or 0 for r in rows)
    timestamps = [r["timestamp"] for r in rows if r["timestamp"]]
    started_at = min(timestamps) if timestamps else None
    ended_at = max(timestamps) if timestamps else None
    duration_s = round(ended_at - started_at, 1) if (started_at and ended_at and ended_at > started_at) else None
    est_input = min(total_chars // CHARS_PER_TOKEN, MAX_CTX)
    return {
        "api_call_count": turns,
        "tool_call_count": len(tool_names),
        "tools_used": list(tool_names),
        "input_tokens": est_input,
        "duration_s": duration_s,
        "started_at": started_at,
        "ended_at": ended_at,
        "token_source": "estimated",
    }


def enrich_session(row: dict, timings: list) -> dict:
    r = dict(row)
    started = r.get("started_at") or 0
    ended = r.get("ended_at")

    needs_fallback = not r.get("input_tokens") and not r.get("api_call_count")
    if needs_fallback:
        # Try JSONL first, then state.db messages
        jm = jsonl_metrics(r["id"]) or db_messages_metrics(r["id"])
        if jm:
            for k in ("api_call_count", "tool_call_count", "input_tokens", "model"):
                if not r.get(k):
                    r[k] = jm.get(k)
            if not started and jm.get("started_at"):
                started = jm["started_at"]
                r["started_at"] = started
            if not ended and jm.get("ended_at"):
                ended = jm["ended_at"]
                r["ended_at"] = ended
            if not r.get("tools_used"):
                r["tools_used"] = jm.get("tools_used", [])
            r["token_source"] = jm.get("token_source", "estimated")
        else:
            r["token_source"] = "unavailable"
    else:
        r["token_source"] = "exact"

    r["duration_s"] = round(ended - started, 1) if (ended and started and ended > started) else None
    r["started_str"] = datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M") if started else None
    base_url = r.get("billing_base_url") or ""
    r["is_local"] = not base_url or "localhost" in base_url or "127.0.0.1" in base_url or r.get("billing_provider") == "custom"
    r["context_pct"] = round((r.get("input_tokens") or 0) / MAX_CTX * 100, 1)

    # Compression: inactive messages in state.db
    con = db()
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM messages WHERE session_id=? AND active=0", (r["id"],))
    r["compressed_count"] = cur.fetchone()[0]
    r["compression_used"] = r["compressed_count"] > 0

    # Tools from state.db if not already set from JSONL
    if not r.get("tools_used"):
        cur.execute(
            "SELECT DISTINCT tool_name FROM messages WHERE session_id=? AND tool_name IS NOT NULL AND active=1",
            (r["id"],),
        )
        r["tools_used"] = [t[0] for t in cur.fetchall() if t[0]]
    con.close()

    # Response timing from log
    if timings and started:
        end_ts = ended or (started + 86400)
        session_timings = [t for t in timings if started <= t["ts"] <= end_ts]
        if session_timings:
            r["avg_response_s"] = round(sum(t["time_s"] for t in session_timings) / len(session_timings), 1)
            r["max_response_s"] = round(max(t["time_s"] for t in session_timings), 1)
        else:
            r["avg_response_s"] = None
            r["max_response_s"] = None
    else:
        r["avg_response_s"] = None
        r["max_response_s"] = None

    return r


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/sessions")
def get_sessions(limit: int = 200, offset: int = 0):
    con = db()
    cur = con.cursor()
    cur.execute(
        "SELECT * FROM sessions WHERE archived=0 OR archived IS NULL ORDER BY started_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    timings = parse_log_timings()
    return [enrich_session(r, timings) for r in rows]


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    con = db()
    cur = con.cursor()
    cur.execute("SELECT * FROM sessions WHERE id=?", (session_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(404, "Session not found")
    timings = parse_log_timings()
    result = enrich_session(dict(row), timings)

    # Build turns from state.db messages first
    cur.execute(
        "SELECT role, content, tool_calls, tool_name, timestamp, token_count, active "
        "FROM messages WHERE session_id=? ORDER BY timestamp",
        (session_id,),
    )
    db_msgs = [dict(m) for m in cur.fetchall()]
    con.close()

    # Fall back to JSONL if no db messages
    raw_msgs = db_msgs if db_msgs else read_jsonl(session_id)

    turns = []
    current_turn = None
    for msg in raw_msgs:
        role = msg.get("role", "")
        if role == "session_meta":
            continue
        if role == "user":
            if current_turn:
                turns.append(current_turn)
            current_turn = {
                "turn": len(turns) + 1,
                "user_ts": msg.get("timestamp"),
                "user_msg": (str(msg.get("content") or ""))[:200],
                "assistant_ts": None,
                "tools": [],
                "token_count": msg.get("token_count") or 0,
                "duration_s": None,
            }
        elif role == "assistant" and current_turn:
            current_turn["assistant_ts"] = msg.get("timestamp")
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                try:
                    calls = json.loads(tool_calls) if isinstance(tool_calls, str) else tool_calls
                    for c in calls:
                        name = c.get("function", {}).get("name") or c.get("name", "")
                        if name and name not in current_turn["tools"]:
                            current_turn["tools"].append(name)
                except Exception:
                    pass
            current_turn["token_count"] += msg.get("token_count") or 0
        elif role == "tool" and current_turn:
            name = msg.get("tool_name") or msg.get("name", "")
            if name and name not in current_turn["tools"]:
                current_turn["tools"].append(name)

    if current_turn:
        turns.append(current_turn)

    for t in turns:
        if t["user_ts"] and t["assistant_ts"]:
            t["duration_s"] = round(t["assistant_ts"] - t["user_ts"], 1)

    result["turns"] = turns
    result["max_ctx"] = MAX_CTX
    return result


@app.get("/api/stats")
def get_stats():
    con = db()
    cur = con.cursor()
    cur.execute("SELECT * FROM sessions WHERE archived=0 OR archived IS NULL ORDER BY started_at DESC")
    rows = [dict(r) for r in cur.fetchall()]
    cur.execute("SELECT COUNT(DISTINCT session_id) FROM messages WHERE active=0")
    sessions_with_compression = cur.fetchone()[0]
    cur.execute(
        "SELECT tool_name, COUNT(*) as cnt FROM messages WHERE tool_name IS NOT NULL "
        "GROUP BY tool_name ORDER BY cnt DESC LIMIT 10"
    )
    top_tools_db = [{"tool": r[0], "count": r[1]} for r in cur.fetchall()]
    con.close()

    timings = parse_log_timings()

    # Enrich all sessions
    enriched = [enrich_session(r, timings) for r in rows]

    completed = [s for s in enriched if s.get("ended_at") and s.get("started_at") and s["ended_at"] > s["started_at"]]
    with_tokens = [s for s in enriched if s.get("input_tokens", 0) > 0]
    with_turns = [s for s in enriched if s.get("api_call_count", 0) > 0]

    def avg(lst, key):
        vals = [s[key] for s in lst if s.get(key)]
        return round(sum(vals) / len(vals), 1) if vals else None

    def mx(lst, key):
        vals = [s[key] for s in lst if s.get(key)]
        return round(max(vals), 1) if vals else None

    # Tools: prefer state.db aggregation, supplement from JSONL
    tool_counts: dict[str, int] = {t["tool"]: t["count"] for t in top_tools_db}
    for s in enriched:
        for tool in s.get("tools_used", []):
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
    top_tools = sorted(
        [{"tool": k, "count": v} for k, v in tool_counts.items()],
        key=lambda x: -x["count"]
    )[:10]

    # Sessions per day
    day_counts: dict[str, int] = {}
    for s in enriched:
        if s.get("started_at"):
            day = datetime.fromtimestamp(s["started_at"]).strftime("%Y-%m-%d")
            day_counts[day] = day_counts.get(day, 0) + 1
    sessions_per_day = [{"day": d, "count": c} for d, c in sorted(day_counts.items())[-30:]]

    local_count = sum(1 for s in enriched if s.get("is_local"))

    stats = {
        "total": len(enriched),
        "completed": len(completed),
        "avg_duration": avg(completed, "duration_s"),
        "max_duration": mx(completed, "duration_s"),
        "avg_input_tokens": avg(with_tokens, "input_tokens"),
        "max_input_tokens": mx(with_tokens, "input_tokens"),
        "avg_output_tokens": avg([s for s in enriched if s.get("output_tokens", 0) > 0], "output_tokens"),
        "max_output_tokens": mx(enriched, "output_tokens"),
        "avg_turns": avg(with_turns, "api_call_count"),
        "max_turns": mx(enriched, "api_call_count"),
        "avg_tools": avg([s for s in enriched if s.get("tool_call_count", 0) > 0], "tool_call_count"),
        "max_tools": mx(enriched, "tool_call_count"),
        "total_tool_calls": sum(s.get("tool_call_count", 0) for s in enriched),
        "local_count": local_count,
        "sessions_with_compression": sessions_with_compression,
        "top_tools": top_tools,
        "sessions_per_day": sessions_per_day,
        "max_ctx": MAX_CTX,
        "avg_response_s": None,
        "max_response_s": None,
        "total_responses_logged": 0,
    }

    if timings:
        times = [t["time_s"] for t in timings]
        stats["avg_response_s"] = round(sum(times) / len(times), 1)
        stats["max_response_s"] = round(max(times), 1)
        stats["total_responses_logged"] = len(times)

    return stats


@app.get("/api/gpu")
def get_gpu():
    result = {"error": None, "models": []}
    # GPU stats via nvidia-smi
    try:
        smi = subprocess.run(
            ["/usr/lib/wsl/lib/nvidia-smi",
             "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if smi.returncode == 0:
            parts = [p.strip() for p in smi.stdout.strip().split(",")]
            result.update({
                "name": parts[0],
                "temp_c": parts[1],
                "util_pct": parts[2],
                "vram_used_mb": int(parts[3]),
                "vram_total_mb": int(parts[4]),
                "power_w": parts[5],
                "vram_pct": round(int(parts[3]) / int(parts[4]) * 100, 1),
            })
        else:
            result["error"] = smi.stderr.strip()
    except Exception as e:
        result["error"] = str(e)

    # Loaded models via ollama ps
    try:
        ps = subprocess.run(
            ["ollama", "ps"],
            capture_output=True, text=True, timeout=5
        )
        if ps.returncode == 0:
            lines = ps.stdout.strip().splitlines()
            for line in lines[1:]:  # skip header
                parts = line.split()
                if len(parts) >= 5:
                    result["models"].append({
                        "name": parts[0],
                        "size": parts[2] + " " + parts[3],
                        "processor": parts[4],
                        "context": parts[5] if len(parts) > 5 else "—",
                    })
    except Exception as e:
        result["models_error"] = str(e)

    return result


@app.get("/api/memory-log")
def get_memory_log(limit: int = 100):
    return parse_memory_log()[-limit:]


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(content=HTML)


HTML = r"""<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hermes Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0f1117; --card: #1a1d2e; --card2: #242740;
    --border: #2e3250; --text: #e2e8f0; --muted: #8892b0;
    --accent: #7c6af7; --accent2: #56d9a0; --warn: #f5a623; --danger: #e05252;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }

  header { background: var(--card); border-bottom: 1px solid var(--border); padding: 12px 24px;
    display: flex; align-items: center; justify-content: space-between; }
  header h1 { font-size: 18px; font-weight: 600; }
  .subtitle { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .nav-btn { background: var(--card2); border: 1px solid var(--border); color: var(--text);
    padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
  .nav-btn:hover { border-color: var(--accent); color: var(--accent); }

  main { padding: 20px 24px; max-width: 1400px; margin: 0 auto; }

  /* GPU Bar */
  .gpu-bar { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 18px; margin-bottom: 16px; }
  .gpu-title { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing:.5px; margin-bottom: 10px; }
  .gpu-row { display: flex; gap: 28px; flex-wrap: wrap; align-items: center; }
  .gpu-metric .gl { font-size: 11px; color: var(--muted); margin-bottom: 2px; }
  .gpu-metric .gv { font-size: 16px; font-weight: 600; }
  .progress-wrap { flex: 1; min-width: 180px; }
  .progress-label { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .progress { background: var(--card2); border-radius: 4px; height: 8px; overflow: hidden; }
  .progress-fill { height: 100%; border-radius: 4px; transition: width .3s; }
  .fill-green { background: var(--accent2); } .fill-warn { background: var(--warn); } .fill-danger { background: var(--danger); } .fill-accent { background: var(--accent); }

  /* Models section */
  .models-row { margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border);
    display: flex; gap: 12px; flex-wrap: wrap; }
  .model-chip { background: var(--card2); border: 1px solid var(--border); border-radius: 8px;
    padding: 6px 12px; font-size: 12px; display: flex; flex-direction: column; gap: 2px; }
  .model-chip .mn { font-weight: 600; color: var(--text); }
  .model-chip .ms { color: var(--muted); font-size: 11px; }

  /* Stats grid */
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(175px, 1fr)); gap: 12px; margin-bottom: 16px; }
  .stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .stat-card .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing:.5px; margin-bottom: 6px; }
  .stat-card .value { font-size: 22px; font-weight: 700; }
  .stat-card .sub { color: var(--muted); font-size: 11px; margin-top: 3px; }
  .stat-card.accent .value { color: var(--accent); }
  .stat-card.green .value { color: var(--accent2); }
  .stat-card.warn .value { color: var(--warn); }

  /* Charts */
  .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .chart-card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
  .chart-card h3 { font-size: 13px; color: var(--muted); margin-bottom: 12px; font-weight: 500; }
  .chart-card canvas { max-height: 200px; }

  /* Table */
  .section-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
  .section-header h2 { font-size: 15px; font-weight: 600; }
  .badge { font-size: 11px; background: var(--card2); border: 1px solid var(--border); padding: 3px 8px; border-radius: 10px; color: var(--muted); }
  .table-wrap { background: var(--card); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
  .table-scroll { overflow-x: auto; max-height: 520px; overflow-y: auto; }
  table { width: 100%; border-collapse: collapse; }
  thead th { background: var(--card2); color: var(--muted); font-size: 11px; text-transform: uppercase;
    letter-spacing:.5px; padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 1; }
  tbody tr { border-bottom: 1px solid var(--border); cursor: pointer; transition: background .1s; }
  tbody tr:hover { background: var(--card2); }
  td { padding: 8px 10px; vertical-align: middle; font-size: 13px; }

  /* Pills */
  .pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }
  .pill-local { background: rgba(86,217,160,.15); color: var(--accent2); border: 1px solid rgba(86,217,160,.3); }
  .pill-cloud { background: rgba(245,166,35,.15); color: var(--warn); border: 1px solid rgba(245,166,35,.3); }
  .pill-compressed { background: rgba(124,106,247,.15); color: var(--accent); border: 1px solid rgba(124,106,247,.3); }
  .pill-tool { background: rgba(124,106,247,.1); color: #a89cf7; border: 1px solid rgba(124,106,247,.2); margin: 1px; font-size: 10px; padding: 2px 6px; border-radius: 8px; }
  .pill-est { background: rgba(245,166,35,.1); color: var(--warn); border: 1px solid rgba(245,166,35,.2); font-size: 10px; padding: 1px 5px; border-radius: 6px; }

  /* Context mini bar */
  .mini-ctx { display: inline-flex; align-items: center; gap: 6px; min-width: 100px; }
  .mini-bar { flex: 1; background: var(--card2); border-radius: 3px; height: 6px; overflow: hidden; }
  .mini-fill { height: 100%; border-radius: 3px; }

  /* Detail */
  #detail { display: none; }
  .back-btn { background: none; border: none; color: var(--accent); cursor: pointer; font-size: 14px;
    padding: 0; margin-bottom: 16px; display: flex; align-items: center; gap: 6px; }
  .detail-header { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 20px; margin-bottom: 14px; }
  .detail-title { font-size: 18px; font-weight: 600; margin-bottom: 8px; }
  .detail-meta { display: flex; gap: 16px; flex-wrap: wrap; }
  .detail-meta span { color: var(--muted); font-size: 12px; }
  .detail-meta strong { color: var(--text); }

  .loading { color: var(--muted); text-align: center; padding: 40px; }
  .error-msg { color: var(--danger); text-align: center; padding: 20px; }

  @media (max-width: 768px) {
    .charts-row { grid-template-columns: 1fr; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>⚕ Hermes Dashboard</h1>
    <div class="subtitle" id="header-sub">laden...</div>
  </div>
  <button class="nav-btn" onclick="refresh()">↻ Vernieuwen</button>
</header>

<main>
  <div id="overview">
    <div class="gpu-bar" id="gpu-bar">
      <div class="gpu-title">GPU &amp; Geladen modellen</div>
      <div id="gpu-content"><span style="color:var(--muted)">laden...</span></div>
    </div>

    <div class="stats-grid" id="stats-grid"><div class="loading">laden...</div></div>

    <div class="charts-row">
      <div class="chart-card"><h3>Sessies per dag (30 dagen)</h3><canvas id="chart-sessions"></canvas></div>
      <div class="chart-card"><h3>Top tools gebruikt</h3><canvas id="chart-tools"></canvas></div>
    </div>

    <div class="section-header">
      <h2>Sessies</h2>
      <span class="badge" id="session-count">—</span>
    </div>
    <div class="table-wrap">
      <div class="table-scroll">
        <table>
          <thead><tr>
            <th>Datum</th><th>Titel</th><th>Platform</th>
            <th>Turns</th><th>Input tokens</th><th>Output tokens</th>
            <th>Context %</th><th>Duur</th><th>Reactietijd</th>
            <th>Tools</th><th>Backend</th><th>Compressie</th>
          </tr></thead>
          <tbody id="sessions-tbody"><tr><td colspan="12" class="loading">laden...</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>

  <div id="detail">
    <button class="back-btn" onclick="showOverview()">← Terug naar overzicht</button>
    <div id="detail-content"><div class="loading">laden...</div></div>
  </div>
</main>

<script>
let chartSessions, chartTools;

const fmtNum = n => n == null ? '—' : Math.round(n).toLocaleString('nl-NL');
const fmtDur = s => {
  if (!s || s <= 0) return '—';
  if (s < 60) return s.toFixed(1) + 's';
  return Math.floor(s/60) + 'm ' + Math.round(s%60) + 's';
};
const ctxColor = p => p > 80 ? 'fill-danger' : p > 50 ? 'fill-warn' : 'fill-accent';

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

async function loadGPU() {
  try {
    const gpu = await fetchJSON('/api/gpu');
    let html = '';

    if (!gpu.error) {
      const vp = gpu.vram_pct;
      const vc = vp > 90 ? 'fill-danger' : vp > 70 ? 'fill-warn' : 'fill-green';
      html += `<div class="gpu-row">
        <div class="gpu-metric"><div class="gl">GPU</div><div class="gv">${gpu.name}</div></div>
        <div class="gpu-metric"><div class="gl">Temp</div><div class="gv">${gpu.temp_c}°C</div></div>
        <div class="gpu-metric"><div class="gl">Gebruik</div><div class="gv">${gpu.util_pct}%</div></div>
        <div class="gpu-metric"><div class="gl">Vermogen</div><div class="gv">${gpu.power_w}W</div></div>
        <div class="progress-wrap">
          <div class="progress-label">
            <span>VRAM</span>
            <span>${(gpu.vram_used_mb/1024).toFixed(1)} / ${(gpu.vram_total_mb/1024).toFixed(1)} GB (${vp}%)</span>
          </div>
          <div class="progress"><div class="progress-fill ${vc}" style="width:${Math.min(vp,100)}%"></div></div>
        </div>
      </div>`;
    } else {
      html += `<span style="color:var(--muted)">${gpu.error}</span>`;
    }

    if (gpu.models && gpu.models.length > 0) {
      html += '<div class="models-row">' + gpu.models.map(m => `
        <div class="model-chip">
          <span class="mn">${m.name}</span>
          <span class="ms">${m.size} · ${m.processor} · ctx ${m.context}</span>
        </div>`).join('') + '</div>';
    } else {
      html += '<div class="models-row"><span style="color:var(--muted);font-size:12px">Geen modellen geladen</span></div>';
    }

    document.getElementById('gpu-content').innerHTML = html;
  } catch(e) {
    document.getElementById('gpu-content').innerHTML = `<span style="color:var(--muted)">GPU niet beschikbaar: ${e.message}</span>`;
  }
}

function renderStats(stats) {
  document.getElementById('header-sub').textContent =
    stats.total + ' sessies · ' + stats.local_count + '/' + stats.total + ' lokaal · Max ctx: ' + (stats.max_ctx||40960).toLocaleString('nl-NL') + ' tokens';

  const cards = [
    { label: 'Totaal sessies',      value: stats.total,                         sub: stats.completed + ' afgerond',                     cls: '' },
    { label: 'Gem. turns',          value: fmtNum(stats.avg_turns),             sub: 'max ' + fmtNum(stats.max_turns),                   cls: 'accent' },
    { label: 'Gem. input tokens',   value: fmtNum(stats.avg_input_tokens),      sub: 'max ' + fmtNum(stats.max_input_tokens),            cls: '' },
    { label: 'Gem. output tokens',  value: fmtNum(stats.avg_output_tokens),     sub: 'max ' + fmtNum(stats.max_output_tokens),           cls: '' },
    { label: 'Gem. duur',           value: fmtDur(stats.avg_duration),          sub: 'max ' + fmtDur(stats.max_duration),                cls: '' },
    { label: 'Gem. reactietijd',    value: stats.avg_response_s ? stats.avg_response_s + 's' : '—', sub: 'max ' + (stats.max_response_s ? stats.max_response_s + 's' : '—'), cls: 'warn' },
    { label: 'Totaal tool calls',   value: fmtNum(stats.total_tool_calls),      sub: 'gem. ' + fmtNum(stats.avg_tools) + '/sessie',      cls: 'accent' },
    { label: 'Compressie',          value: stats.sessions_with_compression,     sub: 'sessies met compressie',                           cls: 'accent' },
    { label: 'Lokale API',          value: stats.local_count + '/' + stats.total, sub: '100% lokaal',                                   cls: 'green' },
  ];

  document.getElementById('stats-grid').innerHTML = cards.map(c => `
    <div class="stat-card ${c.cls}">
      <div class="label">${c.label}</div>
      <div class="value">${c.value}</div>
      <div class="sub">${c.sub}</div>
    </div>`).join('');

  const days = stats.sessions_per_day || [];
  const ctxS = document.getElementById('chart-sessions').getContext('2d');
  if (chartSessions) chartSessions.destroy();
  chartSessions = new Chart(ctxS, {
    type: 'bar',
    data: { labels: days.map(d => d.day.slice(5)), datasets: [{ label: 'Sessies',
      data: days.map(d => d.count), backgroundColor: 'rgba(124,106,247,0.6)', borderRadius: 4 }] },
    options: { plugins: { legend: { display: false } },
      scales: { x: { ticks: { color: '#8892b0' }, grid: { color: '#2e3250' } },
                y: { ticks: { color: '#8892b0' }, grid: { color: '#2e3250' }, beginAtZero: true } } }
  });

  const tools = stats.top_tools || [];
  const ctxT = document.getElementById('chart-tools').getContext('2d');
  if (chartTools) chartTools.destroy();
  chartTools = new Chart(ctxT, {
    type: 'bar',
    data: { labels: tools.map(t => t.tool), datasets: [{ label: 'Aanroepen',
      data: tools.map(t => t.count), backgroundColor: 'rgba(86,217,160,0.6)', borderRadius: 4 }] },
    options: { indexAxis: 'y', plugins: { legend: { display: false } },
      scales: { x: { ticks: { color: '#8892b0' }, grid: { color: '#2e3250' }, beginAtZero: true },
                y: { ticks: { color: '#8892b0', font: { size: 11 } }, grid: { color: '#2e3250' } } } }
  });
}

function renderSessions(sessions) {
  document.getElementById('session-count').textContent = sessions.length + ' sessies';
  const tbody = document.getElementById('sessions-tbody');
  if (!sessions.length) { tbody.innerHTML = '<tr><td colspan="12" class="loading">Geen sessies</td></tr>'; return; }

  tbody.innerHTML = sessions.map(s => {
    const cp = s.context_pct || 0;
    const ctxBar = `<div class="mini-ctx">
      <div class="mini-bar"><div class="mini-fill ${ctxColor(cp)}" style="width:${Math.min(cp,100)}%"></div></div>
      <span style="font-size:11px;color:var(--muted)">${cp}%</span>
      ${s.token_source==='estimated' ? '<span class="pill-est">~</span>' : ''}
    </div>`;
    const tools = (s.tools_used||[]).slice(0,3).map(t=>`<span class="pill pill-tool">${t.replace(/_tool$/,'')}</span>`).join('') +
      ((s.tools_used||[]).length > 3 ? `<span class="pill pill-tool">+${s.tools_used.length-3}</span>` : '');
    const backend = s.is_local ? '<span class="pill pill-local">lokaal</span>' : '<span class="pill pill-cloud">cloud</span>';
    const comp = s.compression_used ? '<span class="pill pill-compressed">ja</span>' : '<span style="color:var(--muted)">—</span>';
    const turns = s.api_call_count || 0;
    const turnsEst = s.token_source === 'estimated' ? '<span class="pill-est">~</span>' : '';

    return `<tr onclick="showDetail('${s.id}')">
      <td style="white-space:nowrap;color:var(--muted);font-size:12px">${s.started_str||'—'}</td>
      <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${s.title||s.id}">
        ${s.title || '<span style="color:var(--muted)">geen titel</span>'}
      </td>
      <td style="color:var(--muted);font-size:12px">${s.source||'—'}</td>
      <td style="text-align:center">${turns}${turnsEst}</td>
      <td style="text-align:right">${fmtNum(s.input_tokens)}${s.token_source==='estimated'?' <span class="pill-est">~</span>':''}</td>
      <td style="text-align:right">${fmtNum(s.output_tokens)}</td>
      <td>${ctxBar}</td>
      <td style="white-space:nowrap;color:var(--muted)">${fmtDur(s.duration_s)}</td>
      <td style="text-align:center;color:var(--warn)">${s.avg_response_s ? s.avg_response_s+'s' : '—'}</td>
      <td>${tools||'<span style="color:var(--muted);font-size:12px">—</span>'}</td>
      <td>${backend}</td>
      <td>${comp}</td>
    </tr>`;
  }).join('');
}

async function showDetail(id) {
  document.getElementById('overview').style.display = 'none';
  document.getElementById('detail').style.display = 'block';
  document.getElementById('detail-content').innerHTML = '<div class="loading">laden...</div>';
  try {
    const s = await fetchJSON(`/api/sessions/${id}`);
    const turns = s.turns || [];
    const cp = s.context_pct || 0;
    const backend = s.is_local ? '<span class="pill pill-local">lokaal</span>' : '<span class="pill pill-cloud">cloud</span>';
    const comp = s.compression_used ? `<span class="pill pill-compressed">${s.compressed_count} berichten gecomprimeerd</span>` : '—';
    const estNote = s.token_source === 'estimated' ? ' <span class="pill-est">geschat</span>' : '';

    const turnsHtml = turns.length ? `
      <div class="table-wrap" style="margin-top:14px">
        <div style="padding:12px 16px;border-bottom:1px solid var(--border)">
          <h3 style="font-size:13px;color:var(--muted)">Turns (${turns.length})</h3>
        </div>
        <div class="table-scroll">
          <table>
            <thead><tr><th>#</th><th>Gebruikersbericht</th><th>Duur</th><th>Tokens</th><th>Tools</th></tr></thead>
            <tbody>${turns.map(t => `
              <tr>
                <td style="color:var(--muted)">${t.turn}</td>
                <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font-size:12px" title="${t.user_msg}">${t.user_msg||'—'}</td>
                <td style="white-space:nowrap;color:var(--warn)">${fmtDur(t.duration_s)}</td>
                <td style="text-align:right">${fmtNum(t.token_count)||'—'}</td>
                <td>${(t.tools||[]).map(tool=>`<span class="pill pill-tool">${tool.replace(/_tool$/,'')}</span>`).join('')||'<span style="color:var(--muted);font-size:12px">—</span>'}</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>` : '';

    document.getElementById('detail-content').innerHTML = `
      <div class="detail-header">
        <div class="detail-title">${s.title||'Sessie '+s.id}</div>
        <div class="detail-meta">
          <span>Start: <strong>${s.started_str}</strong></span>
          <span>Duur: <strong>${fmtDur(s.duration_s)}</strong></span>
          <span>Platform: <strong>${s.source||'—'}</strong></span>
          <span>Backend: ${backend}</span>
          <span>${comp}</span>
        </div>
      </div>
      <div class="stats-grid">
        <div class="stat-card accent"><div class="label">Turns</div><div class="value">${s.api_call_count||0}</div><div class="sub">${s.tool_call_count||0} tool calls</div></div>
        <div class="stat-card"><div class="label">Input tokens${estNote}</div><div class="value">${fmtNum(s.input_tokens)}</div><div class="sub">cache read: ${fmtNum(s.cache_read_tokens)}</div></div>
        <div class="stat-card"><div class="label">Output tokens</div><div class="value">${fmtNum(s.output_tokens)}</div><div class="sub">cache write: ${fmtNum(s.cache_write_tokens)}</div></div>
        <div class="stat-card warn"><div class="label">Gem. reactietijd</div><div class="value">${s.avg_response_s?s.avg_response_s+'s':'—'}</div><div class="sub">max ${s.max_response_s?s.max_response_s+'s':'—'}</div></div>
        <div class="stat-card">
          <div class="label">Context window${estNote}</div>
          <div class="value">${cp}%</div>
          <div class="sub">${fmtNum(s.input_tokens)} / ${fmtNum(s.max_ctx)} tokens</div>
        </div>
        <div class="stat-card accent"><div class="label">Compressie</div><div class="value">${s.compressed_count||0}</div><div class="sub">berichten gecomprimeerd</div></div>
      </div>
      <div style="background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:14px">
        <div class="progress-label">
          <span>Context window gebruik${estNote}</span>
          <span>${fmtNum(s.input_tokens)} / ${fmtNum(s.max_ctx)} tokens (${cp}%)</span>
        </div>
        <div class="progress" style="height:12px">
          <div class="progress-fill ${ctxColor(cp)}" style="width:${Math.min(cp,100)}%"></div>
        </div>
      </div>
      ${(s.tools_used||[]).length ? `
      <div style="background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin-bottom:14px">
        <div style="font-size:11px;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px">Gebruikte tools</div>
        ${s.tools_used.map(t=>`<span class="pill pill-tool" style="font-size:12px;padding:4px 10px">${t}</span>`).join('')}
      </div>` : ''}
      ${turnsHtml}`;
  } catch(e) {
    document.getElementById('detail-content').innerHTML = `<div class="error-msg">Fout: ${e.message}</div>`;
  }
}

function showOverview() {
  document.getElementById('detail').style.display = 'none';
  document.getElementById('overview').style.display = 'block';
}

async function refresh() {
  try {
    const [stats, sessions] = await Promise.all([fetchJSON('/api/stats'), fetchJSON('/api/sessions')]);
    renderStats(stats);
    renderSessions(sessions);
  } catch(e) {
    document.getElementById('stats-grid').innerHTML = `<div class="error-msg">Fout: ${e.message}</div>`;
  }
}

loadGPU();
setInterval(loadGPU, 10000);
refresh();
</script>
</body>
</html>
"""
