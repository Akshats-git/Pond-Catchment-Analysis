"""Async jobs, the result cache, and the dispatcher that spreads them over workers.

`POST /jobs` returns at once with an id; `GET /jobs/{id}` says where the analysis is. Two
reasons for it, one from the HLD and one from the user. Heavy work should not hold an HTTP
connection open for its whole duration; and a progress bar that reports the stage the
analysis is actually in (fetching tiles, routing flow, siting, runoff) is what makes a
several-second wait feel like work rather than a hang.

**One process, two roles.** With `POND_JOBS_WORKERS` empty, jobs run here, in a thread,
behind the same one-at-a-time semaphore as the synchronous routes. That is a *worker*.
With it set, this process is the *gateway*: it keeps the job store and the result cache,
and hands each job to a worker over HTTP by calling that worker's own job API, mirroring
its stage and progress back to whoever is polling. Same code, same endpoints, so a worker
can be tested alone and the gateway adds nothing but the dispatch.

**Two dispatch strategies, measured rather than assumed** (PLAN3 §6.2). Each worker runs
one analysis at a time, because two at once on 512 MB kill both. `round_robin` hands jobs
out in turn whether or not the next worker is busy; `least_busy` hands each job to a
worker with a free slot and queues only when none has one. docs/SCALING.md has the
numbers.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from app.config import JobsConfig, settings

__all__ = ["Job", "JobStore", "Dispatcher", "STAGE_WEIGHTS", "dispatcher"]

STAGE_WEIGHTS: dict[str, float] = {
    "elevation": 0.12,
    "land": 0.08,
    "flow": 0.22,
    "ensemble": 0.30,
    "siting": 0.14,
    "hydrology": 0.08,
    "geojson": 0.06,
}
"""Share of a typical map-path analysis each stage takes, measured on the sample area.
Progress is the sum over finished stages, so the bar moves when the work does."""

STAGE_LABELS: dict[str, str] = {
    "queued": "Waiting for a worker",
    "dispatched": "Sent to a worker",
    "elevation": "Fetching elevation tiles",
    "land": "Reading the imagery for available land",
    "flow": "Routing water across the terrain",
    "ensemble": "Cross-checking on three more grids",
    "siting": "Choosing pond sites",
    "hydrology": "Working out rainfall and runoff",
    "geojson": "Drawing the answer",
    "done": "Done",
    "failed": "Failed",
}


@dataclass
class Job:
    id: str
    request: dict
    key: str
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    result: dict | None = None
    error: dict | None = None
    worker: str | None = None
    cached: bool = False
    stages: list[str] = field(default_factory=list)

    def enter(self, stage: str) -> None:
        """A stage has started: everything before it is finished."""
        done = sum(STAGE_WEIGHTS.get(s, 0.0) for s in self.stages)
        expected = sum(STAGE_WEIGHTS.values()) - (
            0.0 if self.request.get("avoid_unavailable_land") else STAGE_WEIGHTS["land"]
        ) - (0.0 if self.request.get("ensemble", True) else STAGE_WEIGHTS["ensemble"])
        self.stages.append(stage)
        self.stage = stage
        self.progress = round(min(0.99, done / max(expected, 1e-9)), 3)

    def view(self, *, include_result: bool = True) -> dict:
        now = self.finished or time.time()
        body: dict[str, Any] = {
            "job_id": self.id,
            "status": self.status,
            "stage": self.stage,
            "stage_label": STAGE_LABELS.get(self.stage, self.stage),
            "progress": 1.0 if self.status == "done" else self.progress,
            "cached": self.cached,
            "worker": self.worker,
            "created": self.created,
            "elapsed_s": round(now - self.created, 3),
            "run_s": round(now - self.started, 3) if self.started else None,
        }
        if self.error is not None:
            body["error"] = self.error
        if include_result and self.result is not None:
            body["result"] = self.result
        return body


class JobStore:
    """Jobs by id and finished results by request hash. Bounded, thread-safe."""

    def __init__(self, config: JobsConfig | None = None) -> None:
        self.config = config or settings.jobs
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._results: OrderedDict[str, dict] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def add(self, job: Job) -> Job:
        with self.lock:
            self._expire()
            self._jobs[job.id] = job
            while len(self._jobs) > self.config.max_jobs:
                # Oldest finished first; a running job is never forgotten mid-flight.
                victim = next((k for k, j in self._jobs.items() if j.finished), None)
                if victim is None:
                    break
                self._jobs.pop(victim)
        return job

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self.lock:
            return list(self._jobs.values())

    def _expire(self) -> None:
        cutoff = time.time() - self.config.job_ttl_s
        for key in [k for k, j in self._jobs.items() if j.finished and j.finished < cutoff]:
            self._jobs.pop(key)

    # ---- results ---- #
    def cached(self, key: str) -> dict | None:
        with self.lock:
            hit = self._results.get(key)
            if hit is not None:
                self._results.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
            return hit

    def remember(self, key: str, result: dict) -> None:
        with self.lock:
            self._results[key] = result
            self._results.move_to_end(key)
            while len(self._results) > self.config.result_cache_size:
                self._results.popitem(last=False)

    def counts(self) -> dict:
        with self.lock:
            by_status: dict[str, int] = {}
            for job in self._jobs.values():
                by_status[job.status] = by_status.get(job.status, 0) + 1
            return {
                "jobs": by_status,
                "cached_results": len(self._results),
                "cache_hits": self.hits,
                "cache_misses": self.misses,
            }


# --------------------------------------------------------------------------- #
# Workers, as the gateway sees them
# --------------------------------------------------------------------------- #
@dataclass
class Worker:
    url: str
    slots: int
    in_flight: int = 0
    healthy: bool = True
    completed: int = 0
    failed: int = 0
    last_error: str | None = None
    checked: float = 0.0

    def view(self) -> dict:
        return {
            "url": self.url, "healthy": self.healthy, "in_flight": self.in_flight,
            "slots": self.slots, "completed": self.completed, "failed": self.failed,
            "last_error": self.last_error,
        }


class BusyError(Exception):
    """Every slot is taken and the queue is full."""


class Dispatcher:
    """Runs jobs locally, or hands them to workers. See the module docstring."""

    def __init__(
        self,
        store: JobStore | None = None,
        config: JobsConfig | None = None,
        runner: Callable | None = None,
    ) -> None:
        self.config = config or settings.jobs
        self.store = store or JobStore(self.config)
        self.workers = [Worker(url.rstrip("/"), self.config.worker_slots) for url in self.config.workers]
        self.strategy = self.config.strategy
        self._runner = runner
        self._rr = itertools.count()
        self._condition: asyncio.Condition | None = None
        self._queued = 0
        self._tasks: set[asyncio.Task] = set()

    @property
    def role(self) -> str:
        return "gateway" if self.workers else "worker"

    def _cond(self) -> asyncio.Condition:
        if self._condition is None:
            self._condition = asyncio.Condition()
        return self._condition

    # ---- submit ---- #
    def submit(self, request_body: dict, key: str) -> Job:
        """Create a job and start it. A repeated request is answered from the cache."""
        cached = self.store.cached(key)
        job = Job(id=uuid.uuid4().hex[:16], request=request_body, key=key)
        if cached is not None:
            job.status = job.stage = "done"
            job.progress = 1.0
            job.cached = True
            job.started = job.finished = time.time()
            job.result = cached
            job.worker = "cache"
            return self.store.add(job)

        if self._queued >= self.config.max_queued:
            raise BusyError(
                f"{self._queued} analyses are already waiting; the limit is "
                f"{self.config.max_queued}."
            )
        self.store.add(job)
        self._queued += 1
        task = asyncio.get_running_loop().create_task(self._run(job))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job: Job) -> None:
        try:
            if self.workers:
                await self._run_remote(job)
            else:
                await self._run_local(job)
        except Exception as exc:  # noqa: BLE001, a job must end in a state, whatever happens
            self._fail(job, "internal_error", f"The analysis failed unexpectedly: {exc}", "")

    def _finish(self, job: Job, result: dict) -> None:
        with self.store.lock:
            job.status = job.stage = "done"
            job.progress = 1.0
            job.result = result
            job.finished = time.time()
        self.store.remember(job.key, result)

    def _fail(self, job: Job, code: str, detail: str, hint: str, status: int = 500) -> None:
        with self.store.lock:
            if job.finished:
                return
            job.status = job.stage = "failed"
            job.error = {"status": status, "code": code, "detail": detail, "hint": hint}
            job.finished = time.time()

    # ---- here ---- #
    async def _run_local(self, job: Job) -> None:
        from fastapi.concurrency import run_in_threadpool

        from app.routers.analyze import _ANALYSIS_ERRORS, _analysis_limiter, run_area, structured
        from app.schemas.requests import AreaRequest
        from app.schemas.responses import analysis_response

        request = AreaRequest(**job.request)

        def progress(stage: str) -> None:
            with self.store.lock:
                job.enter(stage)

        try:
            async with _analysis_limiter():
                self._queued -= 1
                with self.store.lock:
                    job.status = "running"
                    job.started = time.time()
                    job.worker = "local"
                result = await asyncio.wait_for(
                    run_in_threadpool(
                        lambda: analysis_response(run_area(request, progress)).model_dump(mode="json")
                    ),
                    timeout=self.config.worker_timeout_s,
                )
        except asyncio.TimeoutError:
            self._fail(job, "analysis_timeout", "The analysis did not finish in time.",
                       "Draw a smaller area.", 504)
            return
        except _ANALYSIS_ERRORS as exc:
            err = structured(exc)
            self._fail(job, err.code, err.detail, err.hint, err.status_code)
            return
        self._finish(job, result)

    # ---- elsewhere ---- #
    async def _acquire(self, exclude: set[str]) -> Worker:
        """A worker with a free slot, by the configured strategy. Waits if none has one."""
        cond = self._cond()
        async with cond:
            while True:
                healthy = [w for w in self.workers if w.healthy and w.url not in exclude]
                if not healthy:
                    healthy = [w for w in self.workers if w.url not in exclude]
                    for worker in healthy:
                        if time.time() - worker.checked > self.config.health_interval_s:
                            worker.healthy = True  # give it another chance
                if not healthy:
                    raise BusyError("No worker is reachable.")
                if self.strategy == "round_robin":
                    worker = healthy[next(self._rr) % len(healthy)]
                    while worker.in_flight >= worker.slots:
                        await cond.wait()
                else:
                    free = [w for w in healthy if w.in_flight < w.slots]
                    if not free:
                        await cond.wait()
                        continue
                    worker = min(free, key=lambda w: (w.in_flight, w.completed))
                worker.in_flight += 1
                return worker

    async def _release(self, worker: Worker) -> None:
        cond = self._cond()
        async with cond:
            worker.in_flight -= 1
            cond.notify_all()

    async def _run_remote(self, job: Job) -> None:
        tried: set[str] = set()
        prefix = settings.api.api_prefix
        dequeued = False
        async with httpx.AsyncClient(timeout=10.0) as client:
            for _ in range(max(1, len(self.workers))):
                try:
                    worker = await self._acquire(tried)
                except BusyError as exc:
                    if not dequeued:
                        self._queued -= 1
                    self._fail(job, "no_worker", str(exc), "Try again shortly.", 503)
                    return
                if not dequeued:
                    self._queued -= 1
                    dequeued = True
                tried.add(worker.url)
                with self.store.lock:
                    job.status = "running"
                    job.stage = "dispatched"
                    job.worker = worker.url
                    job.started = job.started or time.time()
                try:
                    outcome = await self._drive(client, worker, job, prefix)
                except (httpx.HTTPError, OSError) as exc:
                    worker.healthy = False
                    worker.checked = time.time()
                    worker.last_error = str(exc) or exc.__class__.__name__
                    worker.failed += 1
                    await self._release(worker)
                    continue  # another worker gets it
                await self._release(worker)
                if outcome == "done":
                    worker.completed += 1
                else:
                    worker.failed += 1
                return
        self._fail(job, "no_worker", "Every worker tried failed to answer.", "Try again shortly.", 503)

    async def _drive(self, client: httpx.AsyncClient, worker: Worker, job: Job, prefix: str) -> str:
        """Submit to one worker and mirror its job until it ends. Raises on transport
        failure, so the caller can try another worker."""
        response = await client.post(f"{worker.url}{prefix}/jobs", json=job.request)
        if response.status_code >= 500 and response.status_code != 503:
            raise httpx.HTTPError(f"worker answered {response.status_code}")
        body = response.json()
        if response.status_code not in (200, 202):
            self._fail(job, body.get("code", "worker_error"), body.get("detail", ""),
                       body.get("hint", ""), response.status_code)
            return "failed"
        remote_id = body["job_id"]
        deadline = time.time() + self.config.worker_timeout_s
        while time.time() < deadline:
            if body.get("status") in ("done", "failed"):
                break
            await asyncio.sleep(self.config.worker_poll_s)
            poll = await client.get(f"{worker.url}{prefix}/jobs/{remote_id}")
            if poll.status_code != 200:
                raise httpx.HTTPError(f"worker lost job {remote_id}: {poll.status_code}")
            body = poll.json()
            with self.store.lock:
                job.stage = body.get("stage", job.stage)
                job.progress = body.get("progress", job.progress)
        else:
            self._fail(job, "analysis_timeout", "The worker did not finish in time.",
                       "Draw a smaller area.", 504)
            return "failed"

        if body["status"] == "done":
            self._finish(job, body["result"])
            return "done"
        error = body.get("error") or {}
        self._fail(job, error.get("code", "worker_error"), error.get("detail", ""),
                   error.get("hint", ""), error.get("status", 500))
        return "failed"

    # ---- status ---- #
    async def check_workers(self) -> None:
        prefix_free = [w for w in self.workers]
        async with httpx.AsyncClient(timeout=3.0) as client:
            for worker in prefix_free:
                try:
                    response = await client.get(f"{worker.url}/health")
                    worker.healthy = response.status_code == 200
                    worker.last_error = None if worker.healthy else f"HTTP {response.status_code}"
                except (httpx.HTTPError, OSError) as exc:
                    worker.healthy = False
                    worker.last_error = str(exc) or exc.__class__.__name__
                worker.checked = time.time()

    def status(self) -> dict:
        return {
            "role": self.role,
            "strategy": self.strategy if self.workers else "local",
            "queued": self._queued,
            "workers": [w.view() for w in self.workers],
            **self.store.counts(),
        }


_DISPATCHER: Dispatcher | None = None


def dispatcher() -> Dispatcher:
    global _DISPATCHER
    if _DISPATCHER is None:
        _DISPATCHER = Dispatcher()
    return _DISPATCHER


def reset_dispatcher(new: Dispatcher | None = None) -> Dispatcher:
    """For tests and for a gateway reconfigured at startup."""
    global _DISPATCHER
    _DISPATCHER = new or Dispatcher()
    return _DISPATCHER
