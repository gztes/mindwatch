from mindwatch.watch import VIDEO_TYPES, extraction_prompt, note_md, pick, usable


def test_usable_rejects_near_empty_transcripts():
    assert not usable("")
    assert not usable("music " * 10)
    assert usable("word " * 60)


def test_pick_takes_top_watchable_videos_only():
    rows = [
        {"id": "a", "type": "YouTubeVideo", "score": 9, "url": "u", "title": "A"},
        {"id": "b", "type": "WebPage", "score": 10, "url": "u", "title": "B"},
        {"id": "c", "type": "InstagramReel", "score": 8, "url": "u", "title": "C"},
        {"id": "d", "type": "InstagramReel", "score": 6, "url": "u", "title": "D"},
        {"id": "e", "type": "YouTubeVideo", "score": 9, "url": None, "title": "E"},
    ]
    assert [r["id"] for r in pick(rows, n=10, min_score=8, done=set())] == ["a", "c"]
    assert [r["id"] for r in pick(rows, n=10, min_score=8, done={"a"})] == ["c"]
    assert "InstagramReel" in VIDEO_TYPES and "WebPage" not in VIDEO_TYPES


def test_extraction_prompt_includes_transcript_and_title():
    p = extraction_prompt({"title": "micrograd", "summary": "autograd"}, "today we build backprop")
    assert "micrograd" in p and "today we build backprop" in p


def test_note_md_renders_sections():
    md = note_md(
        {"title": "micrograd", "url": "https://y/1", "score": 9, "cat": "ml"},
        {
            "tldr": "Build autograd.",
            "tools": ["PyTorch"],
            "repos": ["karpathy/micrograd"],
            "steps": ["write Value class"],
            "claims": [],
            "verdict": "build it",
        },
    )
    assert md.startswith("# micrograd")
    assert "karpathy/micrograd" in md and "write Value class" in md and "build it" in md
    assert "## Claims" not in md  # empty sections are skipped
