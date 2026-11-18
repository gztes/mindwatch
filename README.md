# mindwatch

I saved 3,075 things to [mymind](https://mymind.com) in eight months, mostly Instagram reels, and almost never went back to any of them. mindwatch reads every save, scores it against my goals, watches the best videos, and tells me what was actually worth it.

## Results

Measured on 30 hand-labelled saves the prompt was never tuned on, sampled across score bands and re-weighted to the full collection:

| | v1 prompt | v2 prompt |
|---|---|---|
| Agreement with my labels | 64% | **80%** |
| Precision (flagged saves I agree with) | 90% | **99%** |
| Recall (saves I want that it flags) | 4% | **48%** |

v1 was accurate but far too strict: it buried Claude Code tooling, job-hunt tactics and founder lessons at 4 to 6 out of 10. v2 fixes that without losing precision. With n = 30, treat recall as a range (roughly 30 to 65%), not a point.

Of the 26 top-scoring videos, 21 were transcribed and summarised. 10 had no speech at all (the content is on-screen text), and 2 were worth learning from properly.

## How it works

```
export    mymind API  ->  data/saves.jsonl            (signed per-request JWT, read-only key)
classify  Claude Haiku, batches of 50, 4 workers  ->  category + score 0-10 per save
report    stats + a markdown digest of the best saves (tracking params stripped from links)
watch     yt-dlp  ->  faster-whisper (local)  ->  Claude extraction: tools, repos, steps, verdict
label     local page: judge a stratified sample blind (score hidden), one key per card
eval      re-weighted precision / recall / agreement, held-out mode, list of disagreements
```

Design choices worth knowing:

- **Stratified eval.** Only a small share of saves score high, so a plain random sample of 50 contains almost no flagged items and says nothing about precision. The sample is drawn per score band and each label is weighted by its band's size. The strata stay fixed to the scores the sample was drawn from, so the weights stay valid after a re-score.
- **Local model tried first.** qwen3:4b on Ollama ran at ~0.1 saves/s (about 8 hours for the collection) and scored far too generously, so classification moved to Claude Haiku through the `claude` CLI.
- **Long or silent videos are skipped.** Anything over 25 minutes is flagged to watch yourself; transcripts under 40 words get a "read the captions" note instead of an LLM call.
- **Privacy.** `.env`, cookies and all data stay out of git. Links keep only an allowlist of query parameters (`v`, `list`, `img_index`, YouTube `t`), because share links embed account tokens.

## Run it

```bash
uv sync --extra dev
cp .env.example .env          # add a READ-ONLY key from mymind > Extensions
uv run mindwatch export
uv run mindwatch classify --backend claude --batch 50 --workers 4
uv run mindwatch report --out path/to/vault/digest.md
uv run mindwatch watch -n 30 --min-score 7 --cookies cookies.txt   # cookies only for Instagram
uv run mindwatch label --fresh --out path/outside/onedrive/gold.jsonl
uv run mindwatch eval --gold path/to/gold.jsonl --labels labels_v1.jsonl --threshold 7
uv run pytest
```

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/), the [Claude Code](https://claude.com/claude-code) CLI, ffmpeg, and yt-dlp for `watch`.
