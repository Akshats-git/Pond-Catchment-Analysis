"""Existing water and land availability, from satellite imagery (Phase 23).

The HLD's `POST /api/imagery/detect-ponds` and `POST /api/land/available`. Answered in
this process, or forwarded to the separate CV service when `POND_CV_SERVICE_URL` is set.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, status
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.core.raster_dem import AreaError
from app.cv.imagery import ImageryUnavailable
from app.errors import APIError
from app.routers.analyze import structured
from app.schemas.requests import SelectionRequest
from app.schemas.responses import ErrorResponse

router = APIRouter(tags=["imagery"])

_RESPONSES = {
    422: {"model": ErrorResponse, "description": "The selection is unusable."},
    503: {"model": ErrorResponse, "description": "The imagery could not be fetched."},
}


async def _answer(path: str, request: SelectionRequest, local):
    try:
        aoi = request.area()
        aoi.check_size()
    except AreaError as exc:
        raise structured(exc) from exc
    if settings.cv.service_url:
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    settings.cv.service_url.rstrip("/") + path,
                    json=request.model_dump(exclude_none=True),
                )
        except httpx.HTTPError as exc:
            raise APIError(
                status.HTTP_503_SERVICE_UNAVAILABLE, "imagery_unavailable",
                f"The CV service could not be reached: {exc}",
                "The terrain analysis does not need it; only the imagery layers do.",
            ) from exc
        body = response.json()
        if response.status_code != 200:
            raise APIError(response.status_code, body.get("code", "imagery_unavailable"),
                           body.get("detail", ""), body.get("hint", ""))
        return body
    try:
        return await run_in_threadpool(local, aoi)
    except (AreaError, ImageryUnavailable) as exc:
        raise structured(exc) from exc


@router.post("/imagery/detectPonds", responses=_RESPONSES, summary="Existing ponds and water in an area")
async def detect_ponds(request: SelectionRequest) -> dict:
    """Water bodies already on the ground, read from satellite imagery: one polygon each,
    with its area, largest first. Village ponds, tanks and rivers.

    Worth checking before digging a new pond: an existing one nearby may only need
    desilting, and a site on top of one is not a new pond at all.
    """
    from app.cv.service import detect_ponds as run

    return await _answer("/api/v1/imagery/detectPonds", request, run)


@router.post("/land/available", responses=_RESPONSES, summary="Land available for excavation (FR-3)")
async def land_available(request: SelectionRequest) -> dict:
    """The selection split into water, built-up ground and tree cover, each as a
    (Multi)Polygon with its area and share, plus the share left available.

    Send `avoid_unavailable_land: true` to `/analyzeArea` to keep the pond off
    everything this marks.
    """
    from app.cv.service import land_availability as run

    return await _answer("/api/v1/land/available", request, run)
