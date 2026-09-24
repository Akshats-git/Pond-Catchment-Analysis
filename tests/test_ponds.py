"""Phase 24. Saving, listing, reopening and deleting analyses (FR-9)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.jobs import reset_dispatcher
from app.main import app
from app.store import Store, reset_store

P = settings.api.api_prefix
SHEET = [81.2814, 21.2398, 81.3126, 21.2636]


@pytest.fixture
def client(tmp_path):
    reset_store(Store(path=tmp_path / "ponds.sqlite3"))
    reset_dispatcher()
    with TestClient(app) as c:
        yield c
    reset_store(None)


@pytest.fixture
def finished_job(client):
    job = client.post(f"{P}/jobs", json={"bbox": SHEET, "ensemble": False}).json()
    while job["status"] not in ("done", "failed"):
        time.sleep(0.05)
        job = client.get(f"{P}/jobs/{job['job_id']}").json()
    assert job["status"] == "done"
    return job


def test_save_a_job_list_it_and_reopen_it(client, finished_job):
    saved = client.post(f"{P}/ponds", json={"name": "Sample village", "job_id": finished_job["job_id"]})
    assert saved.status_code == 201
    record = saved.json()
    assert record["source"] == "elevation_tiles"
    assert record["summary"]["catchment_ha"] == finished_job["result"]["recommended_site"]["catchment"]["area_ha"]

    listing = client.get(f"{P}/ponds").json()
    assert listing["total"] == 1 and listing["items"][0]["name"] == "Sample village"
    assert "response" not in listing["items"][0]

    full = client.get(f"{P}/ponds/{record['id']}").json()
    assert full["response"] == finished_job["result"]


def test_save_a_response_directly(client, finished_job):
    saved = client.post(f"{P}/ponds", json={"name": "From the client", "response": finished_job["result"]})
    assert saved.status_code == 201


def test_a_bad_response_is_refused(client):
    response = client.post(f"{P}/ponds", json={"name": "x", "response": {"not": "an analysis"}})
    assert response.status_code == 422


def test_exactly_one_source(client, finished_job):
    both = client.post(f"{P}/ponds", json={"name": "x", "job_id": finished_job["job_id"],
                                           "response": finished_job["result"]})
    assert both.status_code == 422
    assert client.post(f"{P}/ponds", json={"name": "x"}).status_code == 422


def test_an_unknown_job_is_a_404(client):
    assert client.post(f"{P}/ponds", json={"name": "x", "job_id": "nope"}).status_code == 404


def test_delete(client, finished_job):
    record = client.post(f"{P}/ponds", json={"name": "Gone", "job_id": finished_job["job_id"]}).json()
    assert client.delete(f"{P}/ponds/{record['id']}").status_code == 204
    assert client.get(f"{P}/ponds/{record['id']}").status_code == 404
    assert client.delete(f"{P}/ponds/{record['id']}").status_code == 404


def test_newest_first_and_geojson(client, finished_job):
    for name in ("first", "second", "third"):
        client.post(f"{P}/ponds", json={"name": name, "job_id": finished_job["job_id"]})
    names = [i["name"] for i in client.get(f"{P}/ponds").json()["items"]]
    assert names == ["third", "second", "first"]
    collection = client.get(f"{P}/ponds.geojson").json()
    assert len(collection["features"]) == 3
    assert collection["features"][0]["geometry"]["type"] == "Point"


def test_the_store_survives_a_restart(tmp_path, finished_job, client):
    path = tmp_path / "persist.sqlite3"
    first = Store(path=path)
    first.save("kept", finished_job["result"])
    assert Store(path=path).list()[1] == 1


def test_the_store_is_bounded(tmp_path, finished_job):
    import dataclasses

    small = Store(config=dataclasses.replace(settings.store, max_saved=3), path=tmp_path / "b.sqlite3")
    for i in range(5):
        small.save(f"site {i}", finished_job["result"])
    items, total = small.list()
    assert total == 3 and [i["name"] for i in items] == ["site 4", "site 3", "site 2"]
