#!/usr/bin/env python3
"""agent-viewer — a live window into the local/cloud LLM agents Claude calls.

When Claude offloads to MiMo / DeepSeek / Codex / subagents, those outputs land
in Claude's context, invisible in your terminal. This serves a live web page
(localhost) that streams them as they happen, so you can read along in a browser.

Sources, merged + live (SSE):
  1. ~/.local/share/lamu/conversations.db  (turns table) — every `cloud_query`
     that was called with a conversation_id is auto-logged here. AUTOMATIC.
  2. ~/.local/state/lamu/agent-stream.jsonl — append-only stream for codex /
     subagents / narration. One JSON object per line:
        {"ts":<epoch>, "source":"codex", "label":"bible review", "role":"response", "text":"..."}
     Append with clog.py, or any `echo '{...}' >> the file`.

Run:  python viewer.py [--port 8787]     then open http://127.0.0.1:8787
Local-only bind (127.0.0.1); read-only; no external deps (stdlib).
"""
from __future__ import annotations
import argparse, html, json, os, pathlib, sqlite3, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOME = pathlib.Path.home()
CONV_DB = HOME / ".local/share/lamu/conversations.db"
STREAM = HOME / ".local/state/lamu/agent-stream.jsonl"
STREAM.parent.mkdir(parents=True, exist_ok=True)
STREAM.touch(exist_ok=True)


def _db_records() -> list[dict]:
    if not CONV_DB.exists():
        return []
    out = []
    try:
        c = sqlite3.connect(f"file:{CONV_DB}?mode=ro", uri=True, timeout=2)
        for cid, idx, role, content, ts, meta in c.execute(
                "select conversation_id, idx, role, content, ts, metadata from turns"):
            model = ""
            if meta:
                try:
                    model = (json.loads(meta) or {}).get("model", "")
                except Exception:
                    pass
            out.append({"ts": float(ts or 0), "source": model or "cloud",
                        "label": str(cid), "role": role or "", "text": content or "",
                        "key": f"db:{cid}:{idx}"})
        c.close()
    except Exception:
        pass
    return out


def _jsonl_records() -> list[dict]:
    out = []
    try:
        for i, line in enumerate(STREAM.read_text(errors="replace").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            out.append({"ts": float(o.get("ts") or 0), "source": o.get("source", "agent"),
                        "label": o.get("label", ""), "role": o.get("role", ""),
                        "text": o.get("text", ""), "key": f"jsonl:{i}"})
    except Exception:
        pass
    return out


def all_records() -> list[dict]:
    recs = _db_records() + _jsonl_records()
    recs.sort(key=lambda r: r["ts"])
    return recs


PAGE = """<!doctype html><meta charset=utf-8><title>agent-viewer</title>
<style>
:root{color-scheme:dark}body{margin:0;font:14px/1.5 ui-monospace,Menlo,monospace;background:#0d1117;color:#c9d1d9}
header{position:sticky;top:0;background:#161b22;padding:8px 14px;border-bottom:1px solid #30363d;font-weight:600}
#feed{padding:14px;max-width:1000px;margin:auto}
.turn{margin:10px 0;border:1px solid #30363d;border-radius:8px;overflow:hidden}
.head{padding:5px 10px;background:#161b22;font-size:12px;display:flex;gap:8px;align-items:center}
.badge{padding:1px 7px;border-radius:10px;font-weight:700;color:#0d1117}
.role{opacity:.6}.ts{margin-left:auto;opacity:.5;font-size:11px}
.body{padding:8px 12px;white-space:pre-wrap;word-break:break-word}
pre.code{background:#0a0e14;border:1px solid #21262d;border-radius:6px;padding:8px;overflow-x:auto;white-space:pre}
.prompt .body{opacity:.72}
</style>
<header>🛰 agent-viewer — live MiMo / DeepSeek / Codex / subagent output</header>
<div id=feed></div>
<script>
const feed=document.getElementById('feed');
const colors={mimo:'#7ee787',deepseek:'#79c0ff',codex:'#ffa657',claude:'#d2a8ff',cloud:'#8b949e',agent:'#8b949e'};
function badge(s){const k=Object.keys(colors).find(k=>s.toLowerCase().includes(k))||'agent';return colors[k];}
function esc(t){return t.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function render(t){ // escape, then turn ```fences``` into <pre>
  const parts=t.split(/```/);let o='';
  for(let i=0;i<parts.length;i++){o+= i%2 ? '<pre class=code>'+esc(parts[i].replace(/^[a-z]*\\n/,''))+'</pre>' : esc(parts[i]);}
  return o;
}
function add(r){
  const d=document.createElement('div');d.className='turn '+(r.role==='prompt'?'prompt':'');
  const ts=r.ts?new Date(r.ts*1000).toLocaleTimeString():'';
  d.innerHTML=`<div class=head><span class=badge style="background:${badge(r.source)}">${esc(r.source)}</span>`+
    `<span class=role>${esc(r.label||'')} ${esc(r.role||'')}</span><span class=ts>${ts}</span></div>`+
    `<div class=body>${render(r.text||'')}</div>`;
  feed.appendChild(d);window.scrollTo(0,document.body.scrollHeight);
}
const es=new EventSource('/stream');
es.onmessage=e=>{try{add(JSON.parse(e.data))}catch(_){}}
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            seen: set[str] = set()
            try:
                while True:
                    for r in all_records():
                        if r["key"] in seen:
                            continue
                        seen.add(r["key"])
                        payload = json.dumps({k: r[k] for k in ("ts", "source", "label", "role", "text")})
                        self.wfile.write(f"data: {payload}\n\n".encode())
                        self.wfile.flush()
                    time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)  # local-only bind (Bible R14)
    print(f"agent-viewer → http://127.0.0.1:{a.port}   (sources: conversations.db + {STREAM})")
    srv.serve_forever()
