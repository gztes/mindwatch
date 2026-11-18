"""Hand-label a sample of saves, then measure how well the classifier agrees with you.

The sample is stratified by the classifier's score, because a plain random
sample of 50 would contain about one "signal" item (signal is 1.7% of saves).
Recall is then re-weighted by how big each band really is.
"""

import json
import random
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mindwatch.report import SIGNAL

# Sampling strata. Fixed to the v1 scores the sample was drawn from, whatever the
# current threshold is, so the re-weighting stays valid after a re-score.
BANDS = {"high": (7, 10), "mid": (4, 6), "low": (0, 3)}
SIZES = {"high": 20, "mid": 15, "low": 15}


def band(score: int) -> str:
    return next(b for b, (lo, hi) in BANDS.items() if lo <= score <= hi)


def judgeable(r: dict) -> bool:
    return bool((r.get("title") or "").strip() or (r.get("summary") or "").strip())


def sample(rows: list[dict], sizes: dict = SIZES, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    picked = []
    for b, k in sizes.items():
        pool = [r for r in rows if judgeable(r) and band(r["score"]) == b]
        picked += rng.sample(pool, min(k, len(pool)))
    rng.shuffle(picked)
    return picked


def load_gold(path: Path) -> dict[str, bool]:
    gold = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                g = json.loads(line)
                gold[g["id"]] = g["worth"]  # last answer wins, so relabelling works
    return gold


def evaluate(rows: list[dict], gold: dict[str, bool], strata: list[dict] | None = None,
             threshold: int = SIGNAL, exclude: set[str] = frozenset()) -> dict:
    """Precision/recall of `score >= threshold` against hand labels.

    `strata` holds the scores the sample was drawn with (defaults to `rows`). Each
    labelled item stands in for N_band / n_band saves of its stratum, which turns a
    stratified sample back into estimates for the whole collection.
    """
    by_id = {r["id"]: r for r in rows}
    strat = {r["id"]: band(r["score"]) for r in (strata or rows)}
    sizes = {b: sum(1 for r in (strata or rows) if judgeable(r) and band(r["score"]) == b) for b in BANDS}
    gold = {i: w for i, w in gold.items() if i in by_id and i in strat and i not in exclude}

    per = {}
    for b in BANDS:
        answers = [w for i, w in gold.items() if strat[i] == b]
        per[b] = {"n": len(answers), "worth": sum(answers), "rate": sum(answers) / len(answers) if answers else None}
    weight = {b: sizes[b] / per[b]["n"] if per[b]["n"] else 0.0 for b in BANDS}

    tp = fp = fn = agree = tot = 0.0
    for i, w in gold.items():
        x, pred = weight[strat[i]], by_id[i]["score"] >= threshold
        tp += x * (pred and w)
        fp += x * (pred and not w)
        fn += x * (w and not pred)
        agree += x * (pred == w)
        tot += x
    disagree = [by_id[i] | {"you": w} for i, w in gold.items() if (by_id[i]["score"] >= threshold) != w]
    return {"bands": per, "sizes": sizes, "n": len(gold), "threshold": threshold,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "agreement": agree / tot if tot else None,
            "est_worth": round(tp + fn), "disagree": disagree}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>mindwatch labelling</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1c1c1c;--mute:#6b6b6b;--line:#e2e0da;--yes:#1f7a4d;--no:#b23b3b}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--card:#1e1e1e;--ink:#ececec;--mute:#9a9a9a;--line:#2e2e2e;--yes:#4fb985;--no:#e07070}}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:640px;margin:0 auto;padding:32px 16px}
.bar{height:4px;background:var(--line);border-radius:2px;margin-bottom:24px}.bar i{display:block;height:100%;background:var(--ink);border-radius:2px;transition:width .2s}
.meta{color:var(--mute);font-size:13px;margin-bottom:8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:24px}
h1{font-size:20px;margin:0 0 12px}p{margin:0 0 16px;color:var(--ink)}a{color:var(--mute)}
.q{margin:24px 0 12px;color:var(--mute);font-size:14px}
.btns{display:flex;gap:12px}button{flex:1;padding:14px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink);font:inherit;cursor:pointer}
button.y{border-color:var(--yes);color:var(--yes)}button.n{border-color:var(--no);color:var(--no)}
kbd{font-size:12px;opacity:.7;margin-left:6px}.back{margin-top:16px;background:none;border:none;color:var(--mute);flex:none;padding:0}
</style></head><body><main>
<div class="bar"><i id="bar"></i></div><div class="meta" id="meta"></div>
<div class="card" id="card"></div>
<div class="q">Would you actually go back to this? Think ML, quant, agents, Claude Code tooling, the job hunt, founder lessons, mindset with a real idea. One key per card, no holding.</div>
<div class="btns"><button class="n" onclick="ans(false)">Not worth it<kbd>N</kbd></button><button class="y" onclick="ans(true)">Worth it<kbd>Y</kbd></button></div>
<button class="back" onclick="back()">&larr; back <kbd>B</kbd></button>
</main><script>
let items=[],i=0;
const esc=s=>(s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function load(){items=await (await fetch('/items')).json();i=items.findIndex(x=>x.done===null);if(i<0)i=items.length;show()}
function show(){
  const done=items.filter(x=>x.done!==null).length;
  document.getElementById('bar').style.width=(100*done/items.length)+'%';
  document.getElementById('meta').textContent=done+' / '+items.length+' labelled';
  const c=document.getElementById('card');
  if(i>=items.length){c.innerHTML='<h1>Done.</h1><p>Run <code>uv run mindwatch eval</code> in the terminal.</p>';return}
  const x=items[i];
  c.innerHTML='<div class="meta">'+esc(x.type)+(x.done!==null?' &middot; you said '+(x.done?'worth it':'not worth it'):'')+'</div><h1>'+esc(x.title||'(untitled)')+'</h1><p>'+esc(x.summary)+'</p>'+(x.url?'<a href="'+esc(x.url)+'" target="_blank" rel="noopener">open original</a>':'');
}
let busy=false;
async function ans(w){if(busy||i>=items.length)return;busy=true;const x=items[i];
  try{await fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:x.id,worth:w})});
  x.done=w;i++;show()}finally{setTimeout(()=>busy=false,250)}}
function back(){if(i>0){i--;show()}}
// e.repeat: holding a key down must not answer several cards in a row
document.addEventListener('keydown',e=>{if(e.repeat)return;const k=e.key.toLowerCase();if(k==='y')ans(true);else if(k==='n')ans(false);else if(k==='b')back()});
load();
</script></body></html>"""


def serve(items: list[dict], gold_path: Path, port: int = 8765) -> None:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/items":
                gold = load_gold(gold_path)
                out = [{"id": r["id"], "title": r.get("title"), "summary": r.get("summary"), "url": r.get("url"),
                        "type": r.get("type"), "done": gold.get(r["id"])} for r in items]
                self._send(json.dumps(out).encode(), "application/json")
            else:
                self._send(PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock, gold_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": body["id"], "worth": bool(body["worth"])}) + "\n")
            self._send(b"{}", "application/json")

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"labelling {len(items)} saves at {url}  (Y = worth it, N = not, B = back; Ctrl+C to stop)")
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped; answers are saved, run it again to resume")
