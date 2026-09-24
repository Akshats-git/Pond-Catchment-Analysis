"""The imagery answers, as GeoJSON: existing water (HLD §6.8) and land availability (FR-3).

Runs in the API process by default. `app` below is also a complete service of its own,
the HLD's container 3, started with `uvicorn app.cv.service:app`; set `POND_CV_SERVICE_URL`
on the API and it calls that instead. Keeping the CV apart is what lets it be scaled, or
swapped for a trained model, without the terrain code noticing.
"""

from __future__ import annotations

import io
import time
from dataclasses import replace

import numpy as np
from fastapi import FastAPI, Response, status
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from scipy import ndimage

from app.config import settings
from app.core.geojson import GeoJSONError, feature_collection, mask_geometry
from app.core.raster_dem import AreaError, AreaOfInterest, polygon_mask
from app.cv.imagery import (
    BUILT, CLASSES, TREES, WATER, ImageryUnavailable, LandClasses, classes_for_area,
)

__all__ = ["detect_ponds", "land_availability", "land_classes_png", "app"]

CAVEAT = (
    "Classified from satellite imagery by colour and texture rules, not surveyed. Use it to "
    "screen ground before a site visit, not to settle ownership or boundaries."
)
STYLE = {
    WATER: {"fill": "#38bdf8", "stroke": "#0369a1"},
    BUILT: {"fill": "#f87171", "stroke": "#b91c1c"},
    TREES: {"fill": "#4ade80", "stroke": "#15803d"},
}
_MAX_PONDS = 300


def _classes(aoi: AreaOfInterest, fetch=None) -> tuple[LandClasses, np.ndarray]:
    land = classes_for_area(aoi.bbox, fetch=fetch)
    inside = polygon_mask(aoi.ring, land.frame) if aoi.kind == "polygon" else np.ones(land.frame.shape, bool)
    return land, inside


def _component_geometry(mask: np.ndarray, frame, r0: int, c0: int) -> dict:
    """Vectorise a cropped mask, placing it back in the full frame."""
    x0, y0 = frame.origin_xy
    sub = replace(frame, origin_xy=(x0 + c0 * frame.resolution_m, y0 + r0 * frame.resolution_m))
    geometry, _ = mask_geometry(mask, sub)
    return geometry


def detect_ponds(aoi: AreaOfInterest, *, fetch=None) -> dict:
    """Existing water bodies inside the selection, one feature each, largest first."""
    started = time.perf_counter()
    land, inside = _classes(aoi, fetch)
    water = land.mask(WATER) & inside
    labels, count = ndimage.label(water)
    frame = land.frame
    cell = frame.resolution_m ** 2
    features = []
    if count:
        sizes = ndimage.sum(water, labels, index=np.arange(1, count + 1)) * cell
        order = np.argsort(-sizes)[:_MAX_PONDS]
        slices = ndimage.find_objects(labels)
        for rank, index in enumerate(order, start=1):
            window = slices[index]
            component = labels[window] == index + 1
            try:
                geometry = _component_geometry(component, frame, window[0].start, window[1].start)
            except GeoJSONError:
                continue
            rows, cols = np.nonzero(component)
            centre = frame.lonlat_of(rows.mean() + window[0].start, cols.mean() + window[1].start)
            features.append({
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "role": "existing_water",
                    "id": rank,
                    "area_m2": round(float(sizes[index])),
                    "area_ha": round(float(sizes[index]) / 1e4, 2),
                    "centroid": [round(float(centre[0]), 6), round(float(centre[1]), 6)],
                    "fill-opacity": 0.45,
                    **STYLE[WATER],
                },
            })
    total = float(water.sum()) * cell
    return {
        "status": "ok",
        "count": len(features),
        "total_area_ha": round(total / 1e4, 2),
        "imagery_zoom": land.imagery.zoom,
        "resolution_m": round(frame.resolution_m, 2),
        "geojson": feature_collection(features, bbox=aoi.bbox),
        "warnings": [CAVEAT] + ([f"{land.imagery.missing} imagery tiles were missing."] if land.imagery.missing else []),
        "timing_ms": {"total": round((time.perf_counter() - started) * 1e3, 1)},
    }


def land_availability(aoi: AreaOfInterest, *, fetch=None) -> dict:
    """Ground that is water, built on, or under trees, and the share of the rest."""
    started = time.perf_counter()
    land, inside = _classes(aoi, fetch)
    frame = land.frame
    cell = frame.resolution_m ** 2
    seen = inside & (land.classes != 255)
    total = max(float(seen.sum()) * cell, 1.0)
    features = []
    classes = {}
    for cls in (WATER, BUILT, TREES):
        mask = land.mask(cls) & inside
        area = float(mask.sum()) * cell
        classes[CLASSES[cls]] = {"area_ha": round(area / 1e4, 2), "share": round(area / total, 4)}
        if not mask.any():
            continue
        geometry, _ = mask_geometry(mask, frame)
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "role": "land_unavailable",
                "class": CLASSES[cls],
                "area_ha": round(area / 1e4, 2),
                "share": round(area / total, 4),
                "fill-opacity": 0.35,
                **STYLE[cls],
            },
        })
    available = float((land.mask(0) & seen).sum()) * cell
    return {
        "status": "ok",
        "available_ha": round(available / 1e4, 2),
        "available_share": round(available / total, 4),
        "classes": classes,
        "imagery_zoom": land.imagery.zoom,
        "resolution_m": round(frame.resolution_m, 2),
        "geojson": feature_collection(features, bbox=aoi.bbox),
        "warnings": [CAVEAT] + ([f"{land.imagery.missing} imagery tiles were missing."] if land.imagery.missing else []),
        "timing_ms": {"total": round((time.perf_counter() - started) * 1e3, 1)},
    }


def land_classes_png(bbox, *, fetch=None) -> tuple[bytes, dict]:
    """The class grid as a lossless PNG plus the frame to place it, for a remote client."""
    land = classes_for_area(tuple(bbox), fetch=fetch)
    buffer = io.BytesIO()
    Image.fromarray(land.classes).save(buffer, format="PNG")
    return buffer.getvalue(), {
        "X-Frame-Bbox": ",".join(f"{v:.8f}" for v in bbox),
        "X-Frame-Resolution": f"{land.frame.resolution_m:.6f}",
        "X-Imagery-Zoom": str(land.imagery.zoom),
    }


# --------------------------------------------------------------------------- #
# Container 3: the same functions as a service of their own
# --------------------------------------------------------------------------- #
app = FastAPI(title="Pond CV service", version=settings.api.version)


def _error(exc) -> Response:
    code = status.HTTP_503_SERVICE_UNAVAILABLE if exc.code == "imagery_unavailable" else 422
    import json

    return Response(
        json.dumps({"status": "error", "code": exc.code, "detail": exc.detail, "hint": exc.hint}),
        status_code=code, media_type="application/json",
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "pond-cv"}


@app.post("/api/v1/imagery/detectPonds")
async def _detect(body: dict):
    try:
        return await run_in_threadpool(detect_ponds, _aoi(body))
    except (AreaError, ImageryUnavailable) as exc:
        return _error(exc)


@app.post("/api/v1/land/available")
async def _available(body: dict):
    try:
        return await run_in_threadpool(land_availability, _aoi(body))
    except (AreaError, ImageryUnavailable) as exc:
        return _error(exc)


@app.post("/api/v1/land/classes")
async def _classes_png(body: dict):
    try:
        png, headers = await run_in_threadpool(land_classes_png, body["bbox"])
    except (AreaError, ImageryUnavailable) as exc:
        return _error(exc)
    return Response(png, media_type="image/png", headers=headers)


def _aoi(body: dict) -> AreaOfInterest:
    from app.schemas.requests import SelectionRequest

    return SelectionRequest(**body).area()
