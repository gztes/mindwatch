from mindwatch.classify import CATEGORIES, build_prompt, parse_labels


def test_prompt_numbers_items_and_lists_categories():
    items = [
        {"id": "x1", "title": "Karpathy micrograd", "summary": "autograd from scratch", "type": "YouTubeVideo"},
        {"id": "x2", "title": "", "summary": "", "type": "InstagramReel"},
    ]
    p = build_prompt(items)
    assert "[0]" in p and "[1]" in p
    assert "Karpathy micrograd" in p
    for c in CATEGORIES:
        assert c in p


def test_parse_maps_indices_back_to_ids_and_clamps():
    items = [{"id": "x1"}, {"id": "x2"}]
    raw = '{"labels": [{"i": 0, "cat": "ml", "score": 14, "why": "core ML"}, {"i": 1, "cat": "bogus", "score": -2, "why": ""}, {"i": 7, "cat": "ml", "score": 5, "why": "out of range"}]}'
    out = parse_labels(raw, items)
    assert out == [
        {"id": "x1", "cat": "ml", "score": 10, "why": "core ML"},
        {"id": "x2", "cat": "other", "score": 0, "why": ""},
    ]


def test_parse_survives_garbage():
    assert parse_labels("not json", [{"id": "x1"}]) == []
