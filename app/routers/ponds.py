"""Saved analyses (FR-9): the HLD's `GET/POST /api/ponds`."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Response, status
from pydantic import BaseModel, Field, ValidationError

from app.errors import APIError
from app.jobs import dispatcher
from app.schemas.responses import AnalysisResponse, ErrorResponse
from app.store import store

router = APIRouter(tags=["saved sites"])


class SaveRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="What to call this site.")
    notes: str = Field(default="", max_length=2000)
    job_id: str | None = Field(default=None, description="Save the result of this finished job.")
    response: dict[str, Any] | None = Field(
        default=None, description="Or the full analysis response, as /analyzeContour returned it."
    )


def _not_found(pond_id: str) -> APIError:
    return APIError(status.HTTP_404_NOT_FOUND, "pond_not_found", f"There is no saved site {pond_id!r}.", "")


@router.post("/ponds", status_code=status.HTTP_201_CREATED,
             responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
             summary="Save an analysis")
async def save(request: SaveRequest) -> dict:
    """Keep an analysis under a name. Send the `job_id` of a finished map analysis, or the
    whole response of any analysis (an uploaded sheet's included)."""
    if (request.job_id is None) == (request.response is None):
        raise APIError(422, "invalid_request", "Send exactly one of job_id or response.", "")
    if request.job_id is not None:
        job = dispatcher().store.get(request.job_id)
        if job is None or job.status != "done":
            raise APIError(404, "job_not_found",
                           f"Job {request.job_id!r} is unknown, unfinished or expired.", "")
        response = job.result
    else:
        try:
            response = AnalysisResponse.model_validate(request.response).model_dump(mode="json")
        except ValidationError as exc:
            raise APIError(422, "invalid_request",
                           "response is not an analysis response: " + str(exc.errors()[0]["msg"]), "") from exc
    return store().save(request.name, response, request.notes)


@router.get("/ponds", summary="Saved analyses, newest first")
async def list_ponds(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)) -> dict:
    items, total = store().list(limit, offset)
    return {"total": total, "items": items}


@router.get("/ponds.geojson", summary="Every saved site as GeoJSON points")
async def ponds_geojson() -> dict:
    items, _ = store().list(500, 0)
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [i["summary"]["location"]["lon"], i["summary"]["location"]["lat"]]},
                "properties": {"id": i["id"], "name": i["name"], "created": i["created"],
                               "catchment_ha": i["summary"]["catchment_ha"],
                               "annual_runoff_m3": i["summary"]["annual_runoff_m3"],
                               "marker-color": "#08519c"},
            }
            for i in items
        ],
    }


@router.get("/ponds/{pond_id}", responses={404: {"model": ErrorResponse}}, summary="One saved analysis, in full")
async def get_pond(pond_id: str) -> dict:
    record = store().get(pond_id)
    if record is None:
        raise _not_found(pond_id)
    return record


@router.delete("/ponds/{pond_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response,
               responses={404: {"model": ErrorResponse}}, summary="Delete a saved analysis")
async def delete_pond(pond_id: str) -> Response:
    if not store().delete(pond_id):
        raise _not_found(pond_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
