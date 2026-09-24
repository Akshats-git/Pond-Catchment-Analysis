"""The async job API (PLAN3 §6.1) and the cluster status the gateway reports."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.config import settings

from app.core.raster_dem import AreaError
from app.errors import APIError
from app.jobs import BusyError, dispatcher
from app.routers.analyze import structured
from app.schemas.requests import AreaRequest
from app.schemas.responses import ErrorResponse

router = APIRouter(tags=["jobs"])


@router.post(
    "/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        422: {"model": ErrorResponse, "description": "Selection or parameters unusable."},
        503: {"model": ErrorResponse, "description": "Every worker is busy and the queue is full."},
    },
    summary="Start an area analysis and return at once",
)
async def submit_job(request: AreaRequest, response: Response) -> dict:
    """The same body as `/analyzeArea`. Answers `202` with a `job_id` straight away; poll
    `GET /jobs/{job_id}` for `status` (`queued`, `running`, `done`, `failed`), the `stage`
    the analysis is in, `progress` from 0 to 1, and the full `/analyzeArea` response under
    `result` once it is done.

    A selection analysed before, with the same parameters, comes back `done` immediately
    with `cached: true`.
    """
    try:
        request.area().check_size()
    except AreaError as exc:
        raise structured(exc) from exc
    try:
        job = dispatcher().submit(request.model_dump(mode="json"), request.cache_key())
    except BusyError as exc:
        raise APIError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "busy",
            str(exc),
            "Every worker is busy. Try again in a minute.",
        ) from exc
    response.headers["Location"] = f"{settings.api.api_prefix}/jobs/{job.id}"
    return job.view()


@router.get(
    "/jobs/{job_id}",
    responses={404: {"model": ErrorResponse, "description": "No such job, or it expired."}},
    summary="Where a job is, and its result once it is done",
)
async def get_job(job_id: str, include_result: bool = True) -> dict:
    job = dispatcher().store.get(job_id)
    if job is None:
        raise APIError(
            status.HTTP_404_NOT_FOUND,
            "job_not_found",
            f"There is no job {job_id!r}. Finished jobs are kept for an hour.",
            "Submit the area again; if it was analysed recently the answer is cached.",
        )
    return job.view(include_result=include_result)


@router.get("/jobs", summary="Recent jobs, newest first, without their results")
async def list_jobs(limit: int = 50) -> dict:
    jobs = sorted(dispatcher().store.all(), key=lambda j: -j.created)[: max(1, min(limit, 500))]
    return {"jobs": [j.view(include_result=False) for j in jobs]}


@router.get("/cluster", tags=["service"], summary="Dispatcher, workers and cache status")
async def cluster() -> dict:
    """What the gateway knows: its role, the dispatch strategy, each worker's health and
    load, the queue, and the result cache's hit rate."""
    d = dispatcher()
    if d.workers:
        await d.check_workers()
    return d.status()
