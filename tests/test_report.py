from mindwatch.report import join, stats, top


SAVES = [
    {"id": "a", "title": "micrograd", "summary": "s", "url": "u1", "type": "YouTubeVideo", "created": "2026-05-01"},
    {"id": "b", "title": "$10k/mo clipping", "summary": "s", "url": "u2", "type": "InstagramReel", "created": "2026-06-01"},
    {"id": "c", "title": "micrograd", "summary": "dupe", "url": "u3", "type": "InstagramReel", "created": "2026-07-01"},
    {"id": "d", "title": "unlabelled", "summary": "", "url": None, "type": None, "created": None},
]
LABELS = [
    {"id": "a", "cat": "ml", "score": 9, "why": "real"},
    {"id": "b", "cat": "hustle", "score": 0, "why": "bait"},
    {"id": "c", "cat": "ml", "score": 8, "why": "dupe"},
]


def test_join_drops_unlabelled():
    rows = join(SAVES, LABELS)
    assert [r["id"] for r in rows] == ["a", "b", "c"]
    assert rows[0]["cat"] == "ml" and rows[0]["title"] == "micrograd"


def test_stats_counts_categories_and_shares():
    s = stats(join(SAVES, LABELS))
    assert s["total"] == 3
    assert s["by_cat"]["ml"] == 2
    assert round(s["share"]["hustle"], 3) == round(1 / 3, 3)
    assert s["signal"] == 2  # score >= 7


def test_top_sorts_by_score_and_dedupes_titles():
    t = top(join(SAVES, LABELS), n=5)
    assert [r["id"] for r in t] == ["a"]  # b is noise, c duplicates a's title
