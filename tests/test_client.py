import base64
import hashlib
import hmac
import json

from mindwatch.client import sign_token, slim


def _b64decode(seg: str) -> bytes:
    return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))


def test_token_is_bound_to_method_and_path_and_verifies():
    secret = base64.b64encode(b"k" * 32).decode()
    token = sign_token("kid123", secret, "get", "/objects", now=1_000)

    header_seg, payload_seg, sig_seg = token.split(".")
    assert "=" not in token  # base64url, no padding

    header = json.loads(_b64decode(header_seg))
    payload = json.loads(_b64decode(payload_seg))
    assert header == {"alg": "HS256", "kid": "kid123"}
    assert payload == {"method": "GET", "path": "/objects", "iat": 1_000, "exp": 1_300}

    expected = hmac.new(b"k" * 32, f"{header_seg}.{payload_seg}".encode(), hashlib.sha256).digest()
    assert _b64decode(sig_seg) == expected


def test_slim_keeps_what_the_classifier_needs():
    obj = {
        "id": "a" * 22,
        "title": "How to Build a Data Center",
        "summary": "GPU racks and selling compute.",
        "source": {"url": "https://www.instagram.com/reel/X/"},
        "tags": [{"name": "GPU", "flags": 2}, {"name": "data center", "flags": 8}],
        "mainEntity": {"@type": "InstagramReel"},
        "created": "2026-09-25T11:43:52Z",
        "embeddings": [{"vector": [0.1]}],
    }
    assert slim(obj) == {
        "id": "a" * 22,
        "title": "How to Build a Data Center",
        "summary": "GPU racks and selling compute.",
        "url": "https://www.instagram.com/reel/X/",
        "type": "InstagramReel",
        "tags": ["GPU", "data center"],
        "created": "2026-09-25T11:43:52Z",
    }


def test_slim_handles_missing_fields_and_multi_type():
    obj = {"id": "b" * 22, "mainEntity": {"@type": ["Video", "YouTubeVideo"]}, "tags": []}
    s = slim(obj)
    assert s["type"] == "YouTubeVideo"
    assert s["url"] is None and s["title"] == "" and s["summary"] == ""
