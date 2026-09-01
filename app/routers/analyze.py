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

**The file arrives as `contour_map`.** That is the field name the assignment brief
fixes, so it is the one `/docs` shows and the one every example sends. `file` is still
accepted, unnamed in the schema, because the demo page and this project's own history
used it and a working client should not break on a rename. `_resolve_upload` takes
whichever came, and a request carrying neither is a 422 that names both.

**`/contours` is the exception to the semaphore.** It holds a parsed file and nothing
else — no grid, no flow field — so it is not what the concurrency limit is protecting
against, and gating it would defeat the point: the demo page asks for it *while* an
analysis of the same sheet is queued or running.
"""

from __future__ import annotations

import asyncio
import math
import weakref

import anyio
from fastapi import APIRouter, File, Form, Query, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.config import settings
from app.core.dem_builder import DEMBuildError
from app.core.geojson import GeoJSONError, contour_drawing
from app.core.hydrology import HydrologyError
from app.core.kml_parser import ContourParseError, parse_contours
from app.core.pond_siting import SitingError
from app.core.render import RenderError, render_png
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

__all__ = ["UPLOAD_FIELD", "UPLOAD_FIELD_ALIAS", "router"]

router = APIRouter(tags=["analysis"])

_ACCEPTED_SUFFIXES = (".kml", ".kmz")

UPLOAD_FIELD = "contour_map"
"""The multipart field the contour map is expected under. Fixed by the assignment
brief, so it is what `/docs`, the README and every curl example use."""

UPLOAD_FIELD_ALIAS = "file"
"""Also accepted, and kept out of the schema so `/docs` offers one file picker rather
than two. It is what this service asked for before the brief named a field, and what
the demo page sent, so dropping it would break a client to gain nothing."""


def _resolve_upload(
    contour_map: UploadFile | None, file: UploadFile | None
) -> UploadFile:
    """The uploaded sheet, under whichever of the two accepted field names it came.

    `contour_map` wins when a request sends both, because it is the documented name and
    a client sending both has already agreed with itself about the content.
    """
    upload = contour_map if contour_map is not None else file
    if upload is None:
        raise APIError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "missing_file",
            f"No contour map was uploaded. Attach the .kml or .kmz file as the "
            f"`{UPLOAD_FIELD}` field.",
            f"With curl: -F {UPLOAD_FIELD}=@contours.kml. In Postman: Body, form-data, "
            f"a row of type File named {UPLOAD_FIELD}. The field `{UPLOAD_FIELD_ALIAS}` "
            f"is accepted too.",
        )
    return upload


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
    "no_available_ground": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "exclusion_mask_shape": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "no_site_found": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "ensemble_unavailable": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "invalid_simplify": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "invalid_image_size": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "invalid_basemap": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "invalid_frame": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "render_too_large": status.HTTP_422_UNPROCESSABLE_ENTITY,
}
"""Codes that are not 400. Everything else the core raises is a file that cannot be
analysed, which is what 400 means here."""

_ANALYSIS_ERRORS = (
    ContourParseError,
    DEMBuildError,
    SitingError,
    HydrologyError,
    GeoJSONError,
    RenderError,
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
            f"Attach a .kml or .kmz contour sheet as the `{UPLOAD_FIELD}` field.",
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


def _span_metres(bbox) -> float:
    """Rough east-west ground width of a bbox, for choosing a simplification budget.

    Rough is the whole requirement: it decides how finely to thin contour lines for a
    raster, where being out by a few percent moves a vertex by a fraction of a pixel.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = math.radians((min_lat + max_lat) / 2.0)
    return abs(max_lon - min_lon) * 111_320.0 * math.cos(mid_lat)


def _image_name(filename: str) -> str:
    """`contours_1m.kml` -> `contours_1m-catchment.png`, with anything a header cannot
    carry stripped out."""
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0] or "contour-map"
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in stem)[:60]
    return f"{safe.strip('-') or 'contour-map'}-catchment.png"


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
    contour_map: UploadFile | None = File(
        None, description="Contour map as .kml or .kmz. Contour lines, not point labels."
    ),
    file: UploadFile | None = File(None, include_in_schema=False),
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
    upload = _resolve_upload(contour_map, file)
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

    filename = upload.filename or "upload.kml"
    data = await _read_upload(upload)

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
    contour_map: UploadFile | None = File(
        None, description="Contour map as .kml or .kmz. The same file /analyzeContour takes."
    ),
    file: UploadFile | None = File(None, include_in_schema=False),
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

    upload = _resolve_upload(contour_map, file)
    filename = upload.filename or "upload.kml"
    data = await _read_upload(upload)
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


_RENDER_RESPONSES: dict[int | str, dict] = {
    200: {
        "content": {"image/png": {}},
        "description": "The rendered map.",
    },
    400: {"model": ErrorResponse, "description": "The file cannot be analysed."},
    413: {"model": ErrorResponse, "description": "Upload or sheet too large."},
    422: {"model": ErrorResponse, "description": "Parameters, size or basemap unusable."},
    504: {"model": ErrorResponse, "description": "The analysis exceeded the time limit."},
}

WARNINGS_HEADER = "X-Pond-Warnings"
"""Where a PNG response puts what a JSON one would have put in `warnings`.

A picture has nowhere to carry a caveat, and silently dropping "the rainfall is a
climatology, not an observation" because the client asked for an image would be the
service choosing what the client is allowed to know. Header values are single-line
latin-1, so the list is joined and truncated; the full set is always available by asking
`/analyzeContour` for the same file."""

_HEADER_LIMIT = 900
"""Servers commonly cap a single header near 4 kB after the field name and CRLF. Well
under it, because this is a signpost to the JSON endpoint and not a payload."""


def _warning_header(warnings: list[str]) -> dict[str, str]:
    """The warnings as one header-safe line, or no header when there are none."""
    if not warnings:
        return {}
    joined = " | ".join(w.replace("\n", " ").strip() for w in warnings)
    # Non-latin-1 characters cannot go in a header at all, and a warning is not worth an
    # encoding error. The replacement marks that something was dropped.
    safe = joined.encode("latin-1", "replace").decode("latin-1")
    if len(safe) > _HEADER_LIMIT:
        safe = safe[: _HEADER_LIMIT - 3] + "..."
    return {WARNINGS_HEADER: safe}


def _legend_rows(result) -> tuple[str, list[tuple[str, str]]]:
    """The recommended site's headline numbers, so the image answers without the JSON."""
    site = result.recommended
    balance = result.recommended_balance
    catchment = site.catchment
    storage = balance.storage

    area = f"{catchment.area_ha:,.1f} ha"
    if site.ensemble is not None:
        area = f"{catchment.area_ha:,.1f} +/- {site.ensemble.std_area_ha:,.1f} ha"

    runoff = balance.runoff
    ratio = balance.fill_ratio
    rows = [
        ("Catchment", area),
        (
            f"Storage at {result.params.target_depth_m:g} m",
            f"{storage.usable_capacity_m3:,.0f} m3",
        ),
        ("Annual runoff", f"{balance.annual_runoff_m3:,.0f} m3"),
        # The single most decision-relevant number on the page: under 1 the pond does not
        # fill in an average year, well over it the spillway is the design problem.
        ("Fill ratio", "no capacity" if ratio == float("inf") else f"{ratio:,.2f}x"),
        ("Rainfall", f"{runoff.rainfall_mm:,.0f} mm / {runoff.rain_days} days"),
    ]
    return f"Site 1 of {len(result.sites)}, recommended", rows


@router.post(
    "/renderMap",
    responses=_RENDER_RESPONSES,
    response_class=Response,
    summary="The same analysis, drawn as a PNG map",
)
async def render_map(
    contour_map: UploadFile | None = File(
        None, description="Contour map as .kml or .kmz. The same file /analyzeContour takes."
    ),
    file: UploadFile | None = File(None, include_in_schema=False),
    grid_resolution: float | None = Form(None, description="Grid cell size in metres."),
    top_n: int | None = Form(None, description="How many basins to draw."),
    lat: float | None = Form(None, description="Latitude of a spot you have chosen yourself."),
    lon: float | None = Form(None, description="Longitude of that spot. Send it with lat."),
    curve_number: float | None = Form(None, description="SCS curve number, 30 to 98."),
    rainfall_mm: float | None = Form(None, description="Yearly rainfall in millimetres."),
    rain_days: int | None = Form(None, description="Days a year that rain falls on."),
    target_depth_m: float | None = Form(None, description="Pond depth in metres."),
    ensemble: bool | None = Form(None, description="Cross-check each site on three grids."),
    width: int | None = Form(None, description="Image width in pixels."),
    height: int | None = Form(None, description="Image height in pixels."),
    basemap: str | None = Form(
        None,
        description="satellite, street, hillshade or none. `hillshade` draws the DEM the "
        "analysis ran on and needs no network.",
    ),
    contours: bool = Form(
        True, description="Draw the uploaded contour lines under the answer."
    ),
    frame: str | None = Form(
        None, description="`sheet` for the whole uploaded map, `sites` to zoom to the answer."
    ),
    legend: bool = Form(True, description="Draw the recommended site's numbers on the image."),
) -> Response:
    """The answer as a picture: the catchment, the pond and the ranked sites, drawn over
    satellite imagery and the contour lines they were derived from.

    Same input as `/analyzeContour`, same analysis, same colours. What comes back is a
    PNG instead of JSON, for a reader who has no map client in front of them. The one
    check that matters is visual and no number can make it: a catchment boundary is right
    when it runs along the ridges, and that is legible at a glance with the contours
    underneath and invisible without them.

    The basemap is allowed to fail. If the tile server cannot be reached the image falls
    back to a hillshade of the uploaded sheet, which needs no network, and says so in the
    `X-Pond-Warnings` header. Every warning the JSON response would have carried is in
    that header, since a PNG has nowhere else to put one.

    Costs a full analysis, so it is as slow as `/analyzeContour` and shares the same
    one-at-a-time queue. Ask for the JSON if you want both; the GeoJSON in it draws the
    same map client-side.
    """
    upload = _resolve_upload(contour_map, file)
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
        raise APIError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "ensemble_unavailable",
            "This host does not have the memory to run the resolution ensemble.",
            "Send ensemble=false, or omit it. The map is unchanged except that the "
            "catchment area is drawn without its error bar.",
        )
    wanted_frame = (frame or "sheet").lower()
    if wanted_frame not in ("sheet", "sites"):
        raise APIError(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "invalid_frame",
            f"frame is {wanted_frame!r}; it has to be `sheet` or `sites`.",
            "`sheet` frames the whole uploaded map, `sites` zooms to the catchments.",
        )

    filename = upload.filename or "upload.kml"
    data = await _read_upload(upload)

    def work() -> tuple[bytes, list[str]]:
        result = analyse(data, filename, params)
        warnings = list(result.warnings)

        analysis_bbox = tuple(result.geojson["bbox"])
        sheet_bbox = tuple(result.contours.metadata.bbox)
        bbox = sheet_bbox if wanted_frame == "sheet" else analysis_bbox

        drawing = None
        if contours:
            # Thinned to the picture rather than to the analysis: a vertex that moves
                # by less than half a pixel cannot change what is drawn, and at a 1200 px
                # width over a 3 km sheet that budget is metres rather than the 1.5 m the
                # vector endpoint spends. Same lines, a fraction of the rasterising.
                span_m = _span_metres(bbox)
                tolerance = max(
                    settings.geojson.contour_simplify_tolerance_m,
                    span_m / max(width or settings.render.default_width, 1) * 0.5,
                )
                drawing = contour_drawing(result.contours, tolerance_m=tolerance)
                warnings.extend(drawing.warnings)

        title, rows = _legend_rows(result)
        png, render_warnings = render_png(
            analysis=result.geojson,
            dem=result.dem,
            contours=drawing.geojson if drawing is not None else None,
            legend_rows=rows if legend else None,
            legend_title=title,
            width=width,
            height=height,
            basemap=basemap,
            frame_bbox=bbox,
        )
        warnings.extend(render_warnings)
        return png, warnings

    try:
        with anyio.fail_after(settings.api.request_timeout_s):
            async with _analysis_limiter():
                png, warnings = await run_in_threadpool(work)
    except TimeoutError as exc:
        raise APIError(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "analysis_timeout",
            f"The analysis did not finish within {settings.api.request_timeout_s:.0f} s.",
            "Request a coarser grid_resolution, set ensemble=false, or ask for a smaller "
            "image. If several requests were sent at once they run one at a time.",
        ) from exc
    except _ANALYSIS_ERRORS as exc:
        raise APIError(
            _STATUS_BY_CODE.get(exc.code, status.HTTP_400_BAD_REQUEST),
            exc.code,
            exc.detail,
            exc.hint,
        ) from exc

    extension = _extension_warning(filename)
    if extension is not None:
        warnings.insert(0, extension)
    return Response(
        content=png,
        media_type="image/png",
        headers={
            # Named so a browser "save image as" lands on something recognisable rather
            # than on `renderMap`.
            "Content-Disposition": f'inline; filename="{_image_name(filename)}"',
            **_warning_header(warnings),
        },
    )


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
    contour_map: UploadFile | None = File(None),
    file: UploadFile | None = File(None, include_in_schema=False),
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
        contour_map=contour_map,
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
