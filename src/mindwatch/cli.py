import argparse
import json
import os
import sys
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


def cmd_label(args: argparse.Namespace) -> None:
    from mindwatch.label import sample, serve
    from mindwatch.report import join, load

    from mindwatch.label import load_gold

    # always sample from the v1 scores, so the same saves come back after a re-score
    rows = _strata()
    if args.fresh:
        seen = {i for p in DATA.glob("gold*.jsonl") for i in load_gold(p)}
        rows = [r for r in rows if r["id"] not in seen]
        sizes = {"high": 10, "mid": 10, "low": 10}
    else:
        from mindwatch.label import SIZES as sizes
    out = Path(args.out) if args.out else DATA / "gold.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    serve(sample(rows, sizes, seed=args.seed), out, args.port)


def _strata() -> list[dict]:
    from mindwatch.report import join, load

    v1 = DATA / "labels_v1.jsonl"
    return join(load(DATA / "saves.jsonl"), load(v1 if v1.exists() else DATA / "labels.jsonl"))


def cmd_eval(args: argparse.Namespace) -> None:
    from mindwatch.label import evaluate, load_gold
    from mindwatch.report import SIGNAL, join, load

    rows = join(load(DATA / "saves.jsonl"), load(DATA / args.labels))
    gold = load_gold(Path(args.gold) if args.gold else DATA / "gold.jsonl")
    if not gold:
        raise SystemExit("No labels yet. Run: uv run mindwatch label")
    exclude = set(load_gold(DATA / "gold_tune.jsonl")) if args.holdout else set()
    e = evaluate(rows, gold, strata=_strata(), threshold=args.threshold or SIGNAL, exclude=exclude)
    scope = "held-out" if args.holdout else "all"
    print(f"{e['n']} hand labels ({scope}), scores from {args.labels}, worth it = score {e['threshold']}+\n")
    for b, s in e["bands"].items():
        rate = f"{100 * s['rate']:.0f}%" if s["rate"] is not None else "-"
        print(f"  {b:<5} band ({e['sizes'][b]:>4} saves): you said worth it to {s['worth']}/{s['n']} = {rate}")
    pct = lambda v: f"{100 * v:.0f}%" if v is not None else "-"
    print(f"\nagreement (share of saves where you and it say the same): {pct(e['agreement'])}")
    print(f"precision (of what it flagged, share you agree with): {pct(e['precision'])}")
    print(f"recall    (of what you'd want, share it flagged):     {pct(e['recall'])}  "
          f"(you'd want ~{e['est_worth']} saves in total)")
    if e["disagree"]:
        print("\ndisagreements:")
        for d in sorted(e["disagree"], key=lambda d: -d["score"]):
            side = "you: worth it  " if d["you"] else "you: not worth "
            print(f"  {side} model: {d['score']}  {(d.get('title') or d.get('summary') or '')[:70]}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows console defaults to cp1252
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
    lb = sub.add_parser("label", help="hand-label a stratified sample of 50 saves in your browser")
    lb.add_argument("--seed", type=int, default=0)
    lb.add_argument("--port", type=int, default=8765)
    lb.add_argument("--fresh", action="store_true", help="30 new saves you have not labelled before")
    lb.add_argument("--out", default=None, help="where answers go (default data/gold.jsonl)")
    lb.set_defaults(func=cmd_label)
    ev = sub.add_parser("eval", help="precision/recall of the classifier against your hand labels")
    ev.add_argument("--labels", default="labels.jsonl", help="which score file to grade (e.g. labels_v1.jsonl)")
    ev.add_argument("--threshold", type=int, default=None, help="score that counts as 'worth it' (default SIGNAL)")
    ev.add_argument("--holdout", action="store_true", help="skip the labels used to tune the prompt")
    ev.add_argument("--gold", default=None, help="hand-label file (default data/gold.jsonl)")
    ev.set_defaults(func=cmd_eval)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
