"""Turn labels into numbers you can post and a digest you can act on."""

import json
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SIGNAL = 6  # score threshold for "worth going back to" (v2 prompt)

# Allowlist, not blocklist: Instagram alone uses ~15 opaque share tokens (igsh, stkn,
# xtok...) and LinkedIn's rcm carries the sharer's member id. Keep only what a link
# needs to resolve; drop everything else before a URL is written anywhere.
KEEP_PARAMS = {"v", "list", "img_index"}
YOUTUBE = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def clean_url(url: str | None) -> str | None:
    if not url:
        return url
    parts = urlsplit(url)
    keep = KEEP_PARAMS | ({"t"} if parts.netloc in YOUTUBE else set())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k in keep])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def join(saves: list[dict], labels: list[dict]) -> list[dict]:
    lab = {l["id"]: l for l in labels}
    return [{**s, **lab[s["id"]]} for s in saves if s["id"] in lab]


def stats(rows: list[dict]) -> dict:
    by_cat = Counter(r["cat"] for r in rows)
    total = len(rows) or 1
    return {
        "total": len(rows),
        "by_cat": dict(by_cat.most_common()),
        "share": {c: n / total for c, n in by_cat.items()},
        "signal": sum(r["score"] >= SIGNAL for r in rows),
        "scores": dict(sorted(Counter(r["score"] for r in rows).items())),
    }


def top(rows: list[dict], n: int = 25, cat: str | None = None) -> list[dict]:
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: (-r["score"], r.get("created") or "")):
        if r["score"] < SIGNAL or (cat and r["cat"] != cat):
            continue
        key = (r.get("title") or "").strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) == n:
            break
    return out


def _link(r: dict) -> str:
    title = r.get("title") or "(untitled)"
    return f"[{title}]({clean_url(r['url'])})" if r.get("url") else title


def digest_md(rows: list[dict]) -> str:
    s = stats(rows)
    pct = lambda c: f"{100 * s['share'].get(c, 0):.0f}%"
    lines = [
        f"# mindwatch digest, {date.today().isoformat()}",
        "",
        f"**{s['total']} saves read.** {s['signal']} worth your time (score {SIGNAL}+), "
        f"{pct('hustle')} hustle bait, {pct('other')} off-goal.",
        "",
        "| Category | Saves | Share |",
        "|---|---|---|",
        *[f"| {c} | {n} | {pct(c)} |" for c, n in s["by_cat"].items()],
        "",
        "## Top 25 overall",
        "",
    ]
    for r in top(rows, 25):
        lines.append(f"- **{r['score']}** `{r['cat']}` {_link(r)}: {r['why']}")
    for cat in ("ml", "agents", "quant", "career", "build", "mindset"):
        picks = top(rows, 10, cat)
        if not picks:
            continue
        lines += ["", f"## Best of {cat}", ""]
        for r in picks:
            lines.append(f"- **{r['score']}** {_link(r)}: {r['why']}")
    return "\n".join(lines) + "\n"
