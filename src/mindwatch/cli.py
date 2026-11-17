import argparse
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from mindwatch.client import MyMind, slim

DATA = Path("data")


def cmd_export(args: argparse.Namespace) -> None:
    load_dotenv()
    kid, secret = os.environ.get("MYMIND_KID"), os.environ.get("MYMIND_SECRET")
    if not kid or not secret:
        raise SystemExit("Set MYMIND_KID and MYMIND_SECRET in .env (create a read-only key on mymind's Extensions page).")

    mm = MyMind(kid, secret)
    objects = mm.list_objects()
    DATA.mkdir(exist_ok=True)
    out = DATA / "saves.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for o in objects:
            f.write(json.dumps(slim(o), ensure_ascii=False) + "\n")

    types = Counter(slim(o)["type"] or "unknown" for o in objects)
    print(f"Exported {len(objects)} saves to {out}")
    for t, n in types.most_common(12):
        print(f"  {n:>6}  {t}")


def cmd_classify(args: argparse.Namespace) -> None:
    from mindwatch.classify import run

    model = args.model or ("haiku" if args.backend == "claude" else "qwen3:4b")
    run(DATA / "saves.jsonl", DATA / "labels.jsonl", model, args.batch, args.limit, args.backend, args.workers)


def cmd_report(args: argparse.Namespace) -> None:
    from mindwatch.report import digest_md, join, load, stats

    rows = join(load(DATA / "saves.jsonl"), load(DATA / "labels.jsonl"))
    s = stats(rows)
    print(f"{s['total']} labelled, {s['signal']} signal")
    for c, n in s["by_cat"].items():
        print(f"  {n:>5}  {100 * s['share'][c]:4.1f}%  {c}")
    print("scores:", s["scores"])
    out = Path(args.out) if args.out else DATA / "digest.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(digest_md(rows), encoding="utf-8")
    print(f"digest written to {out}")


def cmd_watch(args: argparse.Namespace) -> None:
    from mindwatch.report import join, load
    from mindwatch.watch import run

    rows = join(load(DATA / "saves.jsonl"), load(DATA / "labels.jsonl"))
    run(rows, DATA, args.n, args.min_score, args.cookies)


def main() -> None:
    p = argparse.ArgumentParser(prog="mindwatch")
    sub = p.add_subparsers(required=True)
    sub.add_parser("export", help="pull every save from mymind into data/saves.jsonl").set_defaults(func=cmd_export)
    c = sub.add_parser("classify", help="label every save as signal or noise with a local model")
    c.add_argument("--backend", choices=["ollama", "claude"], default="ollama")
    c.add_argument("--model", default=None, help="default: qwen3:4b (ollama) or haiku (claude)")
    c.add_argument("--batch", type=int, default=10)
    c.add_argument("--workers", type=int, default=1)
    c.add_argument("--limit", type=int, default=None)
    c.set_defaults(func=cmd_classify)
    r = sub.add_parser("report", help="stats + a markdown digest of what was worth it")
    r.add_argument("--out", default=None, help="where to write the digest (e.g. your Obsidian vault)")
    r.set_defaults(func=cmd_report)
    w = sub.add_parser("watch", help="download, transcribe and extract the top-scoring videos")
    w.add_argument("-n", type=int, default=10)
    w.add_argument("--min-score", type=int, default=8)
    w.add_argument("--cookies", default=None, help="cookies.txt for Instagram (YouTube needs none)")
    w.set_defaults(func=cmd_watch)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
