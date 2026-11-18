"""Actually watch the best saves: download audio, transcribe locally, extract what matters."""

import json
import subprocess
from pathlib import Path

from mindwatch.report import clean_url

VIDEO_TYPES = {"YouTubeVideo", "InstagramReel", "TikTokPost", "Video", "XPost"}
MAX_SECONDS = 25 * 60  # longer videos are courses: flag them, don't transcribe them
MIN_WORDS = 40  # below this the video is music or on-screen text, nothing to extract


def usable(transcript: str) -> bool:
    return len(transcript.split()) >= MIN_WORDS


def pick(rows: list[dict], n: int, min_score: int, done: set[str]) -> list[dict]:
    cands = [
        r for r in rows
        if r.get("type") in VIDEO_TYPES and r.get("url") and r["score"] >= min_score and r["id"] not in done
    ]
    return sorted(cands, key=lambda r: -r["score"])[:n]


def download_audio(url: str, out_dir: Path, item_id: str, cookies: str | None = None) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp", "-q", "--no-warnings", "-x", "--audio-format", "mp3",
        "--match-filter", f"duration < {MAX_SECONDS}",
        "-o", str(out_dir / f"{item_id}.%(ext)s"),
    ]
    if cookies:
        cmd += ["--cookies", cookies]
    try:
        proc = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print("    download timed out")
        return None
    path = out_dir / f"{item_id}.mp3"
    if proc.returncode == 0 and not path.exists():
        print(f"    skipped: longer than {MAX_SECONDS // 60} min, watch it yourself")
        return None
    if proc.returncode != 0 or not path.exists():
        print(f"    download failed: {proc.stderr.strip()[:150]}")
        return None
    return path


_whisper = None


def transcribe(audio: Path, model_size: str = "small") -> str:
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        _whisper = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = _whisper.transcribe(str(audio), vad_filter=True)
    return " ".join(s.text.strip() for s in segments)


def extraction_prompt(item: dict, transcript: str) -> str:
    return (
        "You are reading the transcript of a video the user saved. The user is pivoting into ML/AI engineering "
        "(ML, quant, agents) and wants concrete, reusable takeaways, not vibes.\n\n"
        f"Title: {item.get('title')}\nSaved summary: {item.get('summary')}\n\nTranscript:\n{transcript[:24000]}\n\n"
        "Reply with ONLY a JSON object with keys: tldr (1-2 sentences), tools (named tools/libraries/products), "
        "repos (GitHub repos or papers named), steps (concrete steps or techniques taught, max 8), "
        "claims (factual or numeric claims worth checking, max 5), verdict (one line: build it / learn it / "
        "skim / skip, and why). Use empty lists when nothing qualifies. Never invent names not in the transcript."
    )


def extract(item: dict, transcript: str, model: str = "haiku") -> dict | None:
    proc = subprocess.run(
        ["claude", "-p", "--model", model, "--output-format", "json"],
        input=extraction_prompt(item, transcript),
        capture_output=True, text=True, encoding="utf-8", timeout=600,
    )
    if proc.returncode != 0:
        return None
    raw = json.loads(proc.stdout).get("result", "")
    try:
        return json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def note_md(item: dict, x: dict) -> str:
    lines = [
        f"# {item.get('title') or '(untitled)'}",
        "",
        f"Source: {clean_url(item.get('url'))} · score {item.get('score')} · `{item.get('cat')}`",
        "",
        f"**TL;DR:** {x.get('tldr', '')}",
        "",
        f"**Verdict:** {x.get('verdict', '')}",
    ]
    for key, head in (("tools", "Tools"), ("repos", "Repos and papers"), ("steps", "Steps"), ("claims", "Claims")):
        if x.get(key):
            lines += ["", f"## {head}", "", *[f"- {v}" for v in x[key]]]
    return "\n".join(lines) + "\n"


def run(rows: list[dict], data: Path, n: int, min_score: int, cookies: str | None) -> None:
    out_dir = data / "watched"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in out_dir.glob("*.md")}
    todo = pick(rows, n, min_score, done)
    print(f"watching {len(todo)} videos")
    for k, item in enumerate(todo, 1):
        print(f"[{k}/{len(todo)}] {item['title'][:70]}")
        audio = download_audio(item["url"], data / "media", item["id"], cookies)
        if not audio:
            continue
        text = transcribe(audio)
        (out_dir / f"{item['id']}.txt").write_text(text, encoding="utf-8")
        if not usable(text):
            x = {"tldr": "No usable speech (music or on-screen text only).", "verdict": "skim: open it and read the captions"}
            (out_dir / f"{item['id']}.md").write_text(note_md(item, x), encoding="utf-8")
            print("    no speech, flagged for a manual look")
            continue
        x = extract(item, text)
        if not x:
            print("    extraction failed")
            continue
        (out_dir / f"{item['id']}.md").write_text(note_md(item, x), encoding="utf-8")
        print(f"    {x.get('verdict', '')[:100]}")
