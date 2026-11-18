from mindwatch.report import clean_url, digest_md, join, stats, top


def test_clean_url_drops_tracking_but_keeps_what_links_need():
    assert clean_url("https://www.instagram.com/reel/abc/?igsh=x&stkn=y&xtok=z") == "https://www.instagram.com/reel/abc/"
    assert clean_url("https://www.instagram.com/p/abc/?img_index=3&igsh=x") == "https://www.instagram.com/p/abc/?img_index=3"
    assert clean_url("https://www.linkedin.com/posts/x?utm_source=share&rcm=ACoAAB123") == "https://www.linkedin.com/posts/x"
    assert clean_url("https://www.youtube.com/watch?v=abc&si=track&t=42") == "https://www.youtube.com/watch?v=abc&t=42"
    assert clean_url("https://x.com/u/status/1?s=46&t=track") == "https://x.com/u/status/1"
    assert clean_url(None) is None


def test_digest_never_writes_tracking_params():
    rows = join(
        [{"id": "a", "title": "t", "summary": "", "url": "https://www.linkedin.com/x?rcm=ACoAAB123", "created": "2026-05-01"}],
        [{"id": "a", "cat": "ml", "score": 9, "why": "w"}],
    )
    assert "rcm" not in digest_md(rows) and "https://www.linkedin.com/x" in digest_md(rows)


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
    assert s["signal"] == 2  # score >= SIGNAL


def test_top_sorts_by_score_and_dedupes_titles():
    t = top(join(SAVES, LABELS), n=5)
    assert [r["id"] for r in t] == ["a"]  # b is noise, c duplicates a's title
