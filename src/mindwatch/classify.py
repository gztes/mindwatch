"""Sort every save into signal vs noise with a local model (Ollama)."""

import json
import time
from pathlib import Path

import httpx

OLLAMA = "http://localhost:11434"

CATEGORIES = {
    "ml": "machine learning, deep learning, LLM internals, AI research, GPUs and AI hardware",
    "agents": "AI agents, AI coding tools, Claude/ChatGPT workflows, building with AI, dev tools",
    "quant": "quant finance, trading systems, markets, investing analysis",
    "career": "tech careers, internships, interviews, job search, recruiting",
    "build": "startups, product, shipping projects, founders, distribution",
    "content": "growing on X/social media, personal brand, content strategy",
    "hustle": "get-rich-quick, 'make $X/month with AI', 'sell X to businesses for $Y', faceless channels, "
    "clipping, cold-calling small businesses, side-hustle and money-claim bait",
    "other": "anything else: lifestyle, fashion, fitness, dating, news, entertainment, memes",
}

GOALS = (
    "The user is a student pivoting into ML/AI engineering. Goals: go deep on ML, quant and AI agents, "
    "land an internship at a top AI company (NVIDIA, AMD, YC startups), ship impressive projects, grow on X."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "cat": {"type": "string", "enum": list(CATEGORIES)},
                    "score": {"type": "integer"},
                    "why": {"type": "string"},
                },
                "required": ["i", "cat", "score", "why"],
            },
        }
    },
    "required": ["labels"],
}


def build_prompt(items: list[dict]) -> str:
    cats = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    lines = []
    for n, it in enumerate(items):
        text = f"{it.get('title') or '(no title)'} :: {(it.get('summary') or '(no summary)')[:400]}"
        lines.append(f"[{n}] ({it.get('type') or 'unknown'}) {text}")
    return (
        f"{GOALS}\n\nCategories:\n{cats}\n\n"
        "For each saved item below, pick ONE category and a score 0-10 for how useful it is for the user's goals "
        "(10 = concrete, technical, worth an hour; 0 = noise). Be strict: most saves are 2-5. "
        "Reserve 8+ for things that teach real ML/quant/agent skills or are directly useful for landing the job. "
        "Any item whose hook is a money claim is 'hustle' and scores 0-2, even if it mentions AI. "
        "Generic motivation and self-help score 0-3. Lifestyle is 'other', never 'hustle'. "
        "Items with no title and no summary score 0. 'why' is max 12 words.\n\n"
        + "\n".join(lines)
    )


def parse_labels(raw: str, items: list[dict]) -> list[dict]:
    start, end = raw.find("{"), raw.rfind("}")  # tolerate code fences / chatter around the JSON
    try:
        data = json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    out, seen = [], set()
    for lab in data.get("labels", []):
        i = lab.get("i")
        if not isinstance(i, int) or not 0 <= i < len(items) or i in seen:
            continue
        seen.add(i)
        cat = lab.get("cat") if lab.get("cat") in CATEGORIES else "other"
        score = max(0, min(10, int(lab.get("score", 0))))
        out.append({"id": items[i]["id"], "cat": cat, "score": score, "why": str(lab.get("why", ""))[:120]})
    return out


def classify_batch(items: list[dict], model: str, client: httpx.Client) -> list[dict]:
    r = client.post(
        f"{OLLAMA}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": build_prompt(items)}],
            "format": SCHEMA,
            "think": False,
            "stream": False,
            "options": {"temperature": 0},
        },
    )
    r.raise_for_status()
    return parse_labels(r.json()["message"]["content"], items)


CLAUDE_FORMAT = (
    '\n\nReply with ONLY a JSON object, no prose, shaped like: '
    '{"labels": [{"i": 0, "cat": "ml", "score": 7, "why": "..."}]}. One entry per item.'
)


def classify_batch_claude(items: list[dict], model: str) -> list[dict]:
    """Headless Claude Code (`claude -p`): uses the user's existing subscription, no API key."""
    import subprocess

    proc = subprocess.run(
        ["claude", "-p", "--model", model, "--output-format", "json"],
        input=build_prompt(items) + CLAUDE_FORMAT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )
    if proc.returncode != 0:
        print(f"    claude failed: {proc.stderr[:200]}", flush=True)
        return []
    return parse_labels(json.loads(proc.stdout).get("result", ""), items)


def run(
    saves: Path,
    labels: Path,
    model: str,
    batch: int = 10,
    limit: int | None = None,
    backend: str = "ollama",
    workers: int = 1,
) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = [json.loads(l) for l in saves.open(encoding="utf-8")]
    done = set()
    if labels.exists():
        done = {json.loads(l)["id"] for l in labels.open(encoding="utf-8")}
    todo = [r for r in rows if r["id"] not in done][:limit]
    print(f"{len(done)} already labelled, {len(todo)} to go with {backend}:{model}")

    client = httpx.Client(timeout=600)

    def do(chunk: list[dict]) -> list[dict]:
        if backend == "claude":
            return classify_batch_claude(chunk, model)
        return classify_batch(chunk, model, client)

    chunks = [todo[s : s + batch] for s in range(0, len(todo), batch)]
    t0, n = time.time(), 0
    with labels.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(do, c): c for c in chunks}
        for fut in as_completed(futures):
            for lab in fut.result():
                f.write(json.dumps(lab, ensure_ascii=False) + "\n")
            f.flush()
            n += len(futures[fut])
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(todo)}  ({rate:.1f}/s, ~{(len(todo) - n) / rate / 60:.0f} min left)", flush=True)
