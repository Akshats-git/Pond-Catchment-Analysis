"""Place search proxy. Nominatim is replaced by a recorded payload; nothing is fetched."""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.routers import places

PAYLOAD = [
    {"name": "Raipur", "display_name": "Raipur, Chhattisgarh, India", "lat": "21.2380",
     "lon": "81.6337", "type": "city", "boundingbox": ["21.08", "21.39", "81.48", "81.79"]},
    {"display_name": "Broken", "lat": "x", "lon": "1"},
]


@pytest.fixture
def fake(monkeypatch):
    calls = []

    def urlopen(request, timeout=None):
        calls.append(request)
        return io.BytesIO(json.dumps(PAYLOAD).encode())

    monkeypatch.setattr(places.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(places, "_CACHE", places.OrderedDict())
    return calls


def test_a_search_returns_points_and_boxes(fake):
    body = TestClient(app).get(f"{settings.api.api_prefix}/places", params={"q": "Raipur"}).json()
    assert body["results"] == [{
        "name": "Raipur", "display_name": "Raipur, Chhattisgarh, India", "lat": 21.238,
        "lon": 81.6337, "type": "city", "bbox": [81.48, 21.08, 81.79, 21.39],
    }]
    assert "PondCatchmentAnalysis" in fake[0].get_header("User-agent")
    assert "q=Raipur" in fake[0].full_url


def test_a_repeated_search_is_cached(fake):
    client = TestClient(app)
    client.get(f"{settings.api.api_prefix}/places", params={"q": "Raipur"})
    client.get(f"{settings.api.api_prefix}/places", params={"q": "raipur "})
    assert len(fake) == 1


def test_an_unreachable_search_says_so(monkeypatch):
    def boom(request, timeout=None):
        raise places.urllib.error.URLError("down")

    monkeypatch.setattr(places.urllib.request, "urlopen", boom)
    monkeypatch.setattr(places, "_CACHE", places.OrderedDict())
    monkeypatch.setattr(places.time, "sleep", lambda s: None)
    response = TestClient(app).get(f"{settings.api.api_prefix}/places", params={"q": "Nowhere"})
    assert response.status_code == 503 and response.json()["code"] == "places_unavailable"


def test_a_transient_failure_is_retried_once(monkeypatch):
    """The lab's own link resets for a few seconds at a time; one hiccup should not turn
    into a failed search when trying again immediately would have worked."""
    calls = []

    def flaky(request, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise places.urllib.error.URLError("reset")
        return io.BytesIO(json.dumps(PAYLOAD).encode())

    monkeypatch.setattr(places.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(places, "_CACHE", places.OrderedDict())
    monkeypatch.setattr(places.time, "sleep", lambda s: None)
    response = TestClient(app).get(f"{settings.api.api_prefix}/places", params={"q": "Raipur"})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1
    assert len(calls) == 2


def test_a_one_letter_search_is_refused():
    assert TestClient(app).get(f"{settings.api.api_prefix}/places", params={"q": "R"}).status_code == 422
