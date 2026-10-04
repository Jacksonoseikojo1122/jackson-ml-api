from __future__ import annotations

import pytest

from .conftest import TOKEN

BODY = {"a": "hello", "b": "hello"}


def test_no_token_configured_allows_anonymous(client):
    assert client.post("/semantic-similarity", json=BODY).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": f"Bearer {TOKEN}x"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": TOKEN},
    ],
    ids=["missing", "wrong", "prefix-match", "wrong-scheme", "no-scheme"],
)
def test_bad_or_missing_token_is_rejected(make_client, headers):
    client = make_client(api_token=TOKEN)
    resp = client.post("/semantic-similarity", json=BODY, headers=headers)
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


def test_correct_token_is_accepted(make_client):
    client = make_client(api_token=TOKEN)
    resp = client.post("/semantic-similarity", json=BODY, headers={"Authorization": f"Bearer {TOKEN}"})
    assert resp.status_code == 200


@pytest.mark.parametrize("path", ["/classify-image", "/vision-embed-search", "/duplicate-clusters", "/tag-suggest"])
def test_every_post_route_requires_token(make_client, path):
    # Auth runs before body validation, so an empty body still yields 401.
    assert make_client(api_token=TOKEN).post(path, json={}).status_code == 401


def test_health_is_public(make_client):
    assert make_client(api_token=TOKEN).get("/health").status_code == 200


def test_settings_from_env(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("ML_API_TOKEN", "abc")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example, https://b.example")
    s = Settings.from_env()
    assert s.api_token == "abc"
    assert s.allowed_origins == ["https://a.example", "https://b.example"]

    monkeypatch.setenv("ML_API_TOKEN", "")
    monkeypatch.delenv("ALLOWED_ORIGINS")
    s = Settings.from_env()
    assert s.api_token is None
    assert s.allowed_origins == ["*"]


def test_cors_allows_configured_origin_only(make_client):
    client = make_client(allowed_origins=["https://app.example"])
    ok = client.options(
        "/tag-suggest",
        headers={"Origin": "https://app.example", "Access-Control-Request-Method": "POST"},
    )
    assert ok.headers.get("access-control-allow-origin") == "https://app.example"
    denied = client.options(
        "/tag-suggest",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in denied.headers
