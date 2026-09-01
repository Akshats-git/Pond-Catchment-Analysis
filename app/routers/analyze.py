"""The HTTP surface: read the request, run the pipeline, map the failures.

No analysis happens here. The route's whole job is the boundary: turn a multipart body
into an `AnalysisParams`, hand it to `app.pipeline.analyse`, and turn whatever comes back
(an answer or one of the core modules' structured errors) into a status code.

**Error mapping is a table, not a chain of `except`s.** Every core module raises the same
`(code, detail, hint)` triple, and `_STATUS_BY_CODE` says what each code means over HTTP.
A new failure mode in `app/core/` therefore needs one line here, and until it gets one it
falls back to `400`. An unrecognised code means the request could not be analysed, which
is the honest default. The three groups:

* **400** for a file that cannot yield an answer: unparseable XML, no contour lines, no
  resolvable elevations, degenerate geometry.
* **413** for too much data: over the upload limit, or a sheet that needs more cells than
  the service will allocate even at its coarsest grid.
* **422** for a request understood but impossible as asked: a parameter out
  of range, a pour point off the sheet, or a sheet on which the siting rules find no
  candidate at all. The last of those is not a malformed request, but it is the same
  thing to a client: nothing about the file changes, and only a different ask can help.

**The analysis runs in a worker thread.** It is several seconds of numpy with the GIL
held in places, and blocking the event loop would stall every other request including
`/health`. `anyio.fail_after` bounds how long a client waits; it cannot interrupt numpy
mid-array, so the thread finishes on its own. The timeout is a promise to the caller,
not a kill switch, and it is documented as such rather than overstated.

**`/contours` is the exception to the semaphore.** It holds a parsed file and nothing
else — no grid, no flow field — so it is not what the concurrency limit is protecting
against, and gating it would defeat the point: the demo page asks for it *while* an
analysis of the same sheet is queued or running.
"""

from __future__ import annotations

import asyncio
import weakref

import anyio
from fastapi import APIRouter, File, Form, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.config import settings
from app.core.dem_builder import DEMBuildError
from app.core.geojson import GeoJSONError, contour_drawing
from app.core.hydrology import HydrologyError
from app.core.kml_parser import ContourParseError, parse_contours
from app.core.pond_siting import SitingError
from app.errors import APIError
from app.pipeline import AnalysisError, Stopwatch, analyse
from app.providers.rainfall import rainfall_for
from app.schemas.requests import AnalysisParams
from app.schemas.responses import (
    AnalysisResponse,
    ContourResponse,
    ErrorResponse,
    RainfallResponse,
    analysis_response,
    contour_response,
    rainfall_response,
)

__all__ = ["router"]

router = APIRouter(tags=["analysis"])

_ACCEPTED_SUFFIXES = (".kml", ".kmz")

_LIMITERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
"""One analysis semaphore per event loop, created on first use.

A module-level semaphore would bind to whichever loop happened to touch it first, which
is wrong under a test client that stands up a fresh loop per call. Keyed weakly so a
finished loop takes its semaphore with it."""


def _analysis_limiter() -> asyncio.Semaphore:
    """The gate that keeps concurrent analyses from exhausting memory.

    See `APIConfig.max_concurrent_analyses`: two analyses at once need more than a
    gigabyte, and on a small container they do not queue, they OOM.
    """
    loop = asyncio.get_running_loop()
    limiter = _LIMITERS.get(loop)
    if limiter is None:
        limiter = asyncio.Semaphore(settings.api.max_concurrent_analyses)
        _LIMITERS[loop] = limiter
    return limiter


_STATUS_BY_CODE: dict[str, int] = {
    # 413 for more data than the service will take on.
    "file_too_large": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    "sheet_too_large": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    # 422 for a problem with the ask rather than the file.
    "invalid_resolution": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "curve_number_out_of_range": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "bad_target_depth": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "bad_rainfall_series": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "invalid_parameters": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "pour_point_outside_map": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "pour_point_unusable": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "no_stream_network": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "no_buildable_ground": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "no_ground_clear_of_watercourse": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "no_site_found": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "ensemble_unavailable": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "invalid_simplify": status.HTTP_422_UNPROCESSABLE_ENTITY,
}
"""Codes that are not 400. Everything else the core raises is a file that cannot be
analysed, which is what 400 means here."""

_ANALYSIS_ERRORS = (
    ContourParseError,
    DEMBuildError,
    SitingError,
    HydrologyError,
    GeoJSONError,
    AnalysisError,
)
"""Every structured error the pipeline can raise. They share the `(code, detail, hint)`
shape by construction and not by coincidence. See each module's error class."""

_RESPONSES: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "The file cannot be analysed."},
    413: {"model": ErrorResponse, "description": "Upload or sheet too large."},
    422: {"model": ErrorResponse, "description": "Parameters or pour point unusable."},
    504: {"model": ErrorResponse, "description": "The analysis exceeded the time limit."},
}


async def _read_upload(file: UploadFile) -> bytes:
    """Read the body, stopping as soon as it passes the limit.

    Starlette has already spooled the upload to a temp file by the time the route runs,
    so this cannot refuse the transfer. What it refuses is materialising an oversized
    file as one `bytes` object and handing it to the parser. Chunked, so an upload ten
    times the limit costs one chunk of memory rather than ten times the limit.
    """
    limit = settings.parser.max_upload_bytes
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise APIError(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "file_too_large",
                f"The upload exceeds the {limit / 1e6:.0f} MB limit.",
                "Clip the contour sheet to the area of interest and retry.",
            )
        chunks.append(chunk)
    if total == 0:
        raise APIError(
            status.HTTP_400_BAD_REQUEST,
            "empty_upload",
            "The uploaded file is empty.",
            "Attach a .kml or .kmz contour sheet as the `file` field.",
        )
    return b"".join(chunks)


def _params(raw: dict) -> AnalysisParams:
    """Validate the form fields, letting omitted ones fall back to their config default.

    Only the keys the client actually sent are passed on: a `None` would override the
    default rather than defer to it.
    """
    try:
        return AnalysisParams(**{k: v for k, v in raw.items() if v is not None})
    except ValidationError as exc:
        # Pydantic prefixes every custom message with "Value error, ". The sentence
        # underneath it is written to be read by whoever sent the request, so the
        # framework's label is dropped rather than passed on.
        problems = "; ".join(
            f"{'.'.join(str(p) for p in error['loc']) or 'request'}: "
            f"{error['msg'].removeprefix('Value error, ')}"
            for error in exc.errors()
        )
        raise APIError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "invalid_parameters",
            problems,
            "See /docs for each parameter's accepted range.",
        ) from exc


def _extension_warning(filename: str) -> str | None:
    """The parser sniffs the content and does not care about the name, so an odd suffix
    is worth saying rather than worth refusing."""
    if filename.lower().endswith(_ACCEPTED_SUFFIXES):
        return None
    return (
        f"{filename!r} is not named .kml or .kmz; it was parsed by content. Check that "
        "the right file was uploaded."
    )


@router.post(
    "/analyzeContour",
    response_model=AnalysisResponse,
    responses=_RESPONSES,
    summary="Analyse a contour map and recommend a pond site with its catchment",
)
async def analyze_contour(
    file: UploadFile = File(
        ..., description="Contour map as .kml or .kmz. Contour lines, not point labels."
    ),
    grid_resolution: float | None = Form(
        None, description="Grid cell size in metres. Leave it out and the contour spacing\n        sets it."
    ),
    top_n: int | None = Form(None, description="How many separate basins to return."),
    lat: float | None = Form(None, description="Latitude of a spot you have chosen yourself."),
    lon: float | None = Form(None, description="Longitude of that spot. Send it with lat."),
    curve_number: float | None = Form(
        None, description="SCS curve number, 30 to 98. How readily this ground sheds rain."
    ),
    rainfall_mm: float | None = Form(
        None,
        description="Yearly rainfall in millimetres. Leave it out and ten years of "
        "Open-Meteo records for the site fill it in.",
    ),
    rain_days: int | None = Form(None, description="Days a year that rain falls on."),
    target_depth_m: float | None = Form(None, description="How deep to build the pond, in metres."),
    ensemble: bool | None = Form(
        None, description="Cross-check each site on three grids. Slower, and honest."
    ),
) -> AnalysisResponse:
    """Send a contour map. Get back where the pond goes, what drains into it, and how
    much water that is worth in an average year.

    The catchment is traced by steepest descent on a grid built from the contour lines.
    Unless you send `ensemble=false` it is traced again on three more grids, so the area
    arrives with an error bar instead of a false precision.

    Sites are ranked by how much ground drains into them, on buildable low-slope land,
    and each site has to stand 3 m above any channel already draining more than 150 ha.
    That last rule is what keeps the answer out of the river.

    Runoff is SCS-CN worked out for each rain day and added up. Never for the whole year
    as one storm, which overstates the yield about sixfold. Rainfall comes from ten years
    of Open-Meteo records for the site unless you send a figure of your own.

    Send `lat` and `lon` together to analyse a spot you have already chosen instead.
    """
    params = _params(
        {
            "grid_resolution": grid_resolution,
            "top_n": top_n,
            "lat": lat,
            "lon": lon,
            "curve_number": curve_number,
            "rainfall_mm": rainfall_mm,
            "rain_days": rain_days,
            "target_depth_m": target_depth_m,
            "ensemble": ensemble,
        }
    )
    if params.ensemble and not settings.api.allow_ensemble:
        # Refused rather than attempted: see `APIConfig.allow_ensemble`. Dying here would
        # take every other request in flight with it, so the honest answer is a 422 that
        # names the limit and points at the analysis the host *can* do.
        raise APIError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "ensemble_unavailable",
            "This host does not have the memory to run the resolution ensemble.",
            "Send ensemble=false, or omit it. The analysis is unchanged except that the "
            "catchment area comes back without its cross-resolution error bar, and "
            "confidence reads `unassessed`.",
        )

    filename = file.filename or "upload.kml"
    data = await _read_upload(file)

    try:
        # The queue wait sits inside the timeout on purpose: a client is promised an
        # answer or an error within `request_timeout_s`, and waiting for a slot is part
        # of the wait. A queue too long to clear in time gets the same honest 504.
        with anyio.fail_after(settings.api.request_timeout_s):
            async with _analysis_limiter():
                result = await run_in_threadpool(analyse, data, filename, params)
    except TimeoutError as exc:
        raise APIError(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "analysis_timeout",
            f"The analysis did not finish within {settings.api.request_timeout_s:.0f} s.",
            "Request a coarser grid_resolution, or set ensemble=false. If several "
            "analyses were sent at once they run one at a time, so try again in a minute.",
        ) from exc
    except _ANALYSIS_ERRORS as exc:
        raise APIError(
            _STATUS_BY_CODE.get(exc.code, status.HTTP_400_BAD_REQUEST),
            exc.code,
            exc.detail,
            exc.hint,
        ) from exc

    response = analysis_response(result)
    extension = _extension_warning(filename)
    if extension is not None:
        response.warnings.insert(0, extension)
    return response


_CONTOUR_RESPONSES: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "The file cannot be read."},
    413: {"model": ErrorResponse, "description": "Upload too large."},
    422: {"model": ErrorResponse, "description": "The simplification asked for is not usable."},
    504: {"model": ErrorResponse, "description": "Reading the file exceeded the time limit."},
}


@router.post(
    "/contours",
    response_model=ContourResponse,
    responses=_CONTOUR_RESPONSES,
    summary="Draw the uploaded contour sheet, without analysing it",
)
async def contours(
    file: UploadFile = File(
        ..., description="Contour map as .kml or .kmz. The same file /analyzeContour takes."
    ),
    simplify_m: float | None = Form(
        None,
        description="How far a drawn line may depart from the one in the file, in "
        "metres. 0 sends every vertex. Leave it out for the default, which is finer "
        "than the grid the analysis runs on.",
    ),
) -> ContourResponse:
    """The contour lines in an uploaded sheet, styled and thinned for a map.

    The parser and nothing else, so this answers in a fraction of a second where
    `/analyzeContour` takes seconds. It exists because a catchment boundary drawn on
    satellite imagery is not checkable by eye: the ground is right when the boundary
    follows the ridges, and only the contours show where those are. Ask for these, lay
    the analysis over them, and the answer can be read rather than taken on trust.

    Every line comes back as a `LineString` carrying its `elevation_m`, an `index` flag
    for the heavy lines a topographic sheet prints every fifth level, and simplestyle
    colours off an elevation ramp, so the collection draws the same in this service's
    demo page and in geojson.io.

    Not gated by the analysis semaphore: parsing costs a few megabytes and a moment,
    which is the whole reason to have this as its own endpoint rather than a flag on the
    analysis. Sending it back with the catchment would put a megabyte on every response
    whether or not the client draws it.
    """
    if simplify_m is not None and not 0.0 <= simplify_m <= 1000.0:
        raise APIError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "invalid_simplify",
            f"simplify_m is {simplify_m}, and it has to be between 0 and 1000 metres.",
            "Leave it out to get the default, which is finer than the analysis grid.",
        )

    filename = file.filename or "upload.kml"
    data = await _read_upload(file)
    watch = Stopwatch()

    def work():
        with watch.stage("parse"):
            parsed = parse_contours(data, filename)
        with watch.stage("draw"):
            return parsed, contour_drawing(parsed, tolerance_m=simplify_m)

    try:
        with anyio.fail_after(settings.api.request_timeout_s):
            parsed, drawing = await run_in_threadpool(work)
    except TimeoutError as exc:
        raise APIError(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "analysis_timeout",
            f"Reading the contours did not finish within "
            f"{settings.api.request_timeout_s:.0f} s.",
            "Ask for a coarser simplify_m, or clip the sheet to the area of interest.",
        ) from exc
    except _ANALYSIS_ERRORS as exc:
        raise APIError(
            _STATUS_BY_CODE.get(exc.code, status.HTTP_400_BAD_REQUEST),
            exc.code,
            exc.detail,
            exc.hint,
        ) from exc

    response = contour_response(parsed, drawing, filename, watch.finish())
    extension = _extension_warning(filename)
    if extension is not None:
        response.warnings.insert(0, extension)
    return response


@router.get(
    "/rainfall",
    response_model=RainfallResponse,
    responses={422: {"model": ErrorResponse, "description": "Coordinates out of range."}},
    summary="Rainfall for a point, from the free Open-Meteo archive",
    tags=["analysis"],
)
async def rainfall(
    lat: float = Query(..., description="Latitude in degrees.", ge=-90.0, le=90.0),
    lon: float = Query(..., description="Longitude in degrees.", ge=-180.0, le=180.0),
) -> RainfallResponse:
    """Ten years of daily rainfall records for one point, averaged to a year.

    The same feed `/analyzeContour` uses when no rainfall figure is given, exposed on its
    own so a client can show the number before committing to an analysis. The demo page
    calls this as soon as a pour point is dropped, which is why the rainfall box on it
    fills itself in.

    Never fails on a rainfall service that is down. It answers with the documented
    regional climatology instead, `is_measured` false, and the reason in `warnings`.
    """
    series = await run_in_threadpool(rainfall_for, lon, lat)
    return rainfall_response(series, lon, lat)


@router.post(
    "/findCatchment",
    response_model=AnalysisResponse,
    responses=_RESPONSES,
    summary="Alias of /analyzeContour",
    include_in_schema=False,
)
async def find_catchment(
    file: UploadFile = File(...),
    grid_resolution: float | None = Form(None),
    top_n: int | None = Form(None),
    lat: float | None = Form(None),
    lon: float | None = Form(None),
    curve_number: float | None = Form(None),
    rainfall_mm: float | None = Form(None),
    rain_days: int | None = Form(None),
    target_depth_m: float | None = Form(None),
    ensemble: bool | None = Form(None),
) -> AnalysisResponse:
    """The name the assignment brief uses. Same request, same response, one piece of
    code behind both. Hidden from the schema so `/docs` shows one endpoint and not two
    identical ones."""
    return await analyze_contour(
        file=file,
        grid_resolution=grid_resolution,
        top_n=top_n,
        lat=lat,
        lon=lon,
        curve_number=curve_number,
        rainfall_mm=rainfall_mm,
        rain_days=rain_days,
        target_depth_m=target_depth_m,
        ensemble=ensemble,
    )
