"""Minimal mymind API client: per-request HS256 tokens, list objects, slim them down."""

import base64
import hashlib
import hmac
import json
import time

import httpx

BASE_URL = "https://api.mymind.com"
USER_AGENT = "mindwatch/0.1"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def sign_token(kid: str, secret_b64: str, method: str, path: str, now: int | None = None) -> str:
    """mymind wants a fresh JWT per request, bound to the method and path."""
    iat = int(time.time()) if now is None else now
    header = _b64url(json.dumps({"alg": "HS256", "kid": kid}, separators=(",", ":")).encode())
    payload = _b64url(
        json.dumps(
            {"method": method.upper(), "path": path, "iat": iat, "exp": iat + 300},
            separators=(",", ":"),
        ).encode()
    )
    sig = hmac.new(base64.b64decode(secret_b64), f"{header}.{payload}".encode(), hashlib.sha256)
    return f"{header}.{payload}.{_b64url(sig.digest())}"


class MyMind:
    def __init__(self, kid: str, secret_b64: str):
        self.kid = kid
        self.secret = secret_b64
        self.http = httpx.Client(base_url=BASE_URL, timeout=120, follow_redirects=True)

    def _get(self, path: str, params: dict | None = None) -> httpx.Response:
        token = sign_token(self.kid, self.secret, "GET", path)
        r = self.http.get(
            path,
            params=params,
            headers={"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT},
        )
        r.raise_for_status()
        return r

    def list_objects(self, limit: int = 10000) -> list[dict]:
        data = self._get("/objects", {"limit": limit}).json()
        return data if isinstance(data, list) else data.get("objects", data.get("items", []))

    def spaces(self) -> list[dict]:
        data = self._get("/spaces").json()
        return data if isinstance(data, list) else data.get("spaces", [])


def _entity_type(obj: dict) -> str | None:
    t = (obj.get("mainEntity") or {}).get("@type")
    if isinstance(t, list):
        return t[-1] if t else None  # most specific type tends to come last
    return t


def slim(obj: dict) -> dict:
    """Only what the classifier needs; drops embeddings, blobs, prose bodies."""
    return {
        "id": obj["id"],
        "title": obj.get("title") or "",
        "summary": obj.get("summary") or "",
        "url": (obj.get("source") or {}).get("url"),
        "type": _entity_type(obj),
        "tags": [t["name"] for t in obj.get("tags", [])],
        "created": obj.get("created"),
    }
