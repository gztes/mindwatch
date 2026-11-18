import json

import pytest

from mindwatch.label import band, evaluate, load_gold, sample


def rows():
    out = []
    for i in range(100):
        out.append({"id": f"lo{i}", "title": f"t{i}", "summary": "", "score": i % 4})  # 0-3
    for i in range(40):
        out.append({"id": f"mid{i}", "title": f"t{i}", "summary": "", "score": 4 + i % 3})  # 4-6
    for i in range(10):
        out.append({"id": f"hi{i}", "title": f"t{i}", "summary": "", "score": 7 + i % 3})  # 7-9
    out.append({"id": "blank", "title": "", "summary": "", "score": 9})
    return out


def test_band_edges():
    assert band(0) == "low" and band(3) == "low"
    assert band(4) == "mid" and band(6) == "mid"
    assert band(7) == "high" and band(10) == "high"


def test_sample_is_stratified_deterministic_and_skips_blanks():
    s = sample(rows(), {"high": 5, "mid": 5, "low": 5}, seed=1)
    assert len(s) == 15
    assert sum(band(r["score"]) == "high" for r in s) == 5
    assert "blank" not in {r["id"] for r in s}
    assert [r["id"] for r in s] == [r["id"] for r in sample(rows(), {"high": 5, "mid": 5, "low": 5}, seed=1)]


def test_sample_caps_at_pool_size():
    assert sum(band(r["score"]) == "high" for r in sample(rows(), {"high": 50, "mid": 0, "low": 0})) == 10


def test_load_gold_last_answer_wins(tmp_path):
    p = tmp_path / "gold.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in [{"id": "a", "worth": True}, {"id": "a", "worth": False}]) + "\n")
    assert load_gold(p) == {"a": False}
    assert load_gold(tmp_path / "missing.jsonl") == {}


def test_evaluate_reweights_recall_by_band_size():
    r = rows()
    # high: 4 of 5 worth it; mid: 1 of 5; low: 0 of 5
    gold = {f"hi{i}": i < 4 for i in range(5)} | {f"mid{i}": i < 1 for i in range(5)} | {f"lo{i}": False for i in range(5)}
    e = evaluate(r, gold, threshold=7)
    assert e["precision"] == pytest.approx(0.8)
    # estimated worth-it items: high 10*0.8=8, mid 40*0.2=8, low 0 -> recall 8/16
    assert e["recall"] == pytest.approx(0.5)
    assert e["est_worth"] == 16
    assert {d["id"] for d in e["disagree"]} == {"hi4", "mid0"}


def test_evaluate_keeps_original_strata_after_rescore_and_excludes_tuning_set():
    v1 = rows()
    gold = {f"hi{i}": i < 4 for i in range(5)} | {f"mid{i}": i < 1 for i in range(5)} | {f"lo{i}": False for i in range(5)}
    # a re-score that promotes mid0 to 8: it is now a hit, but still weighs as a mid-band item
    v2 = [r | {"score": 8} if r["id"] == "mid0" else r for r in v1]
    e = evaluate(v2, gold, strata=v1, threshold=7)
    # tp = 4 high * 2 + 1 mid * 8 = 16, fp = 1 high * 2 = 2 -> precision 16/18, recall 16/16
    assert e["precision"] == pytest.approx(16 / 18)
    assert e["recall"] == pytest.approx(1.0)
    assert evaluate(v2, gold, strata=v1, threshold=7, exclude={"mid0"})["n"] == 14
