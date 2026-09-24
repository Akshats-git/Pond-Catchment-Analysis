"""Phase 19 and 20. Async jobs, the result cache, and the gateway's dispatch.

The gateway tests start real worker processes on local ports and dispatch to them over
HTTP, so what is tested is the code that runs on the four lab containers, not a mock of it.
"""

from __future__ import annotations

import dataclasses
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.jobs import Dispatcher, reset_dispatcher
from app.main import app

PREFIX = settings.api.api_prefix
SHEET = [81.2814, 21.2398, 81.3126, 21.2636]


def wait_for(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"{PREFIX}/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


@pytest.fixture
def local():
    reset_dispatcher()
    with TestClient(app) as client:
        yield client
    reset_dispatcher()


# --------------------------------------------------------------------------- #
# Jobs, here
# --------------------------------------------------------------------------- #
def test_a_job_is_accepted_at_once_and_finishes(local):
    response = local.post(f"{PREFIX}/jobs", json={"bbox": SHEET, "ensemble": False})
    assert response.status_code == 202
    assert response.headers["location"].endswith(response.json()["job_id"])
    body = wait_for(local, response.json()["job_id"])
    assert body["status"] == "done" and body["progress"] == 1.0
    assert body["result"]["input"]["source"] == "elevation_tiles"
    assert body["worker"] == "local"


def test_progress_follows_the_stages_in_order(local):
    job_id = local.post(f"{PREFIX}/jobs", json={"bbox": SHEET}).json()["job_id"]
    seen = []
    while True:
        body = local.get(f"{PREFIX}/jobs/{job_id}").json()
        seen.append(body["progress"])
        if body["status"] in ("done", "failed"):
            break
        time.sleep(0.01)
    assert seen == sorted(seen)
    from app.jobs import dispatcher

    assert dispatcher().store.get(job_id).stages == [
        "elevation", "flow", "ensemble", "siting", "hydrology", "geojson"
    ]


def test_a_repeated_area_is_answered_from_the_cache(local):
    first = wait_for(local, local.post(f"{PREFIX}/jobs", json={"bbox": SHEET, "top_n": 2}).json()["job_id"])
    second = local.post(f"{PREFIX}/jobs", json={"bbox": SHEET, "top_n": 2}).json()
    assert second["status"] == "done" and second["cached"] is True
    assert second["result"] == first["result"]
    # A different parameter is a different answer.
    third = local.post(f"{PREFIX}/jobs", json={"bbox": SHEET, "top_n": 1}).json()
    assert third["cached"] is False


def test_the_synchronous_route_shares_the_cache(local):
    body = {"bbox": SHEET, "top_n": 4, "ensemble": False}
    wait_for(local, local.post(f"{PREFIX}/jobs", json=body).json()["job_id"])
    started = time.perf_counter()
    assert local.post(f"{PREFIX}/analyzeArea", json=body).status_code == 200
    assert time.perf_counter() - started < 0.5


def test_a_failed_job_carries_the_structured_error(local):
    job_id = local.post(f"{PREFIX}/jobs", json={"bbox": [10.0, 45.0, 10.02, 45.02]}).json()["job_id"]
    body = wait_for(local, job_id)
    assert body["status"] == "failed"
    assert body["error"]["code"] == "elevation_unavailable" and body["error"]["status"] == 503


def test_a_bad_selection_is_refused_before_it_becomes_a_job(local):
    response = local.post(f"{PREFIX}/jobs", json={"bbox": [80.0, 20.0, 81.5, 21.5]})
    assert response.status_code == 422 and response.json()["code"] == "aoi_too_large"


def test_an_unknown_job_is_a_404(local):
    response = local.get(f"{PREFIX}/jobs/nope")
    assert response.status_code == 404 and response.json()["code"] == "job_not_found"


def test_a_full_queue_is_a_503_not_a_timeout():
    reset_dispatcher(Dispatcher(config=dataclasses.replace(settings.jobs, max_queued=0)))
    with TestClient(app) as client:
        response = client.post(f"{PREFIX}/jobs", json={"bbox": SHEET, "top_n": 5, "ensemble": False})
        assert response.status_code == 503 and response.json()["code"] == "busy"
    reset_dispatcher()


# --------------------------------------------------------------------------- #
# The gateway, dispatching to real workers
# --------------------------------------------------------------------------- #
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def workers():
    env = {
        **os.environ,
        "POND_ELEVATION_NETWORK_ENABLED": "false",
        "POND_RAINFALL_ENABLED": "false",
        "POND_JOBS_WORKERS": "",
    }
    procs, urls = [], []
    for _ in range(2):
        port = free_port()
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
        urls.append(f"http://127.0.0.1:{port}")
    deadline = time.time() + 30
    for url in urls:
        while True:
            try:
                if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                raise RuntimeError("worker did not start")
            time.sleep(0.2)
    yield urls
    for proc in procs:
        proc.terminate()
        proc.wait(timeout=10)


def gateway(urls, strategy="least_busy"):
    return reset_dispatcher(Dispatcher(config=dataclasses.replace(
        settings.jobs, workers=tuple(urls), strategy=strategy, worker_poll_s=0.05
    )))


def test_the_gateway_hands_the_job_to_a_worker(workers):
    gateway(workers)
    with TestClient(app) as client:
        assert client.get("/health").json()["role"] == "gateway"
        job = client.post(f"{PREFIX}/jobs", json={"bbox": SHEET, "top_n": 1, "ensemble": False}).json()
        body = wait_for(client, job["job_id"])
        assert body["status"] == "done", body
        assert body["worker"] in workers
        assert body["result"]["recommended_site"]["catchment"]["area_ha"] > 0
    reset_dispatcher()


def test_least_busy_spreads_concurrent_jobs_over_the_workers(workers):
    gateway(workers)
    with TestClient(app) as client:
        ids = [
            client.post(f"{PREFIX}/jobs", json={"bbox": SHEET, "top_n": n, "curve_number": 70 + n}).json()["job_id"]
            for n in (1, 2, 3, 4)
        ]
        done = [wait_for(client, i) for i in ids]
        assert all(d["status"] == "done" for d in done)
        assert {d["worker"] for d in done} == set(workers)
        status = client.get(f"{PREFIX}/cluster").json()
        assert status["role"] == "gateway" and len(status["workers"]) == 2
    reset_dispatcher()


def test_a_dead_worker_is_routed_around(workers):
    gateway([f"http://127.0.0.1:{free_port()}", workers[0]])
    with TestClient(app) as client:
        job = client.post(f"{PREFIX}/jobs", json={"bbox": SHEET, "top_n": 2, "curve_number": 60}).json()
        body = wait_for(client, job["job_id"])
        assert body["status"] == "done" and body["worker"] == workers[0]
    reset_dispatcher()


def test_the_synchronous_route_goes_through_the_gateway_too(workers):
    gateway(workers, strategy="round_robin")
    with TestClient(app) as client:
        response = client.post(f"{PREFIX}/analyzeArea", json={"bbox": SHEET, "top_n": 1, "curve_number": 65})
        assert response.status_code == 200
        bad = client.post(f"{PREFIX}/analyzeArea", json={"bbox": [10.0, 45.0, 10.02, 45.02]})
        assert bad.status_code == 503 and bad.json()["code"] == "elevation_unavailable"
    reset_dispatcher()
