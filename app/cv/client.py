"""How the analysis asks the CV for ground to leave alone.

In-process by default; over HTTP when `POND_CV_SERVICE_URL` names a separate CV service.
Either way the answer is a boolean grid on the analysis DEM, handed to the siting step as
its exclusion mask (`PondSiteSelector.available_mask`), which has been waiting for exactly
this since Phase 2.
"""

from __future__ import annotations

import io

import httpx
import numpy as np
from PIL import Image

from app.config import settings
from app.core.dem_builder import DEM
from app.cv.imagery import BUILT, TREES, WATER, ImageryUnavailable, LandClasses, _frame, classes_for_area, imagery_grid

__all__ = ["unavailable_mask_for", "dem_bbox"]


def dem_bbox(dem: DEM) -> tuple[float, float, float, float]:
    ny, nx = dem.shape
    corners = dem.lonlat_of(np.array([0, ny - 1]), np.array([0, nx - 1]))
    return (float(corners[0, 0]), float(corners[0, 1]), float(corners[1, 0]), float(corners[1, 1]))


def _remote_classes(bbox) -> LandClasses:
    url = settings.cv.service_url.rstrip("/") + "/api/v1/land/classes"
    try:
        response = httpx.post(url, json={"bbox": list(bbox)}, timeout=60.0)
    except httpx.HTTPError as exc:
        raise ImageryUnavailable("imagery_unavailable", f"The CV service could not be reached: {exc}") from exc
    if response.status_code != 200:
        body = response.json()
        raise ImageryUnavailable(body.get("code", "imagery_unavailable"), body.get("detail", ""), body.get("hint", ""))
    classes = np.asarray(Image.open(io.BytesIO(response.content)))
    frame_bbox = tuple(float(v) for v in response.headers["X-Frame-Bbox"].split(","))
    frame = _frame(frame_bbox, float(response.headers["X-Frame-Resolution"]))
    return LandClasses(classes=classes, frame=frame, imagery=None)  # type: ignore[arg-type]


def unavailable_mask_for(dem: DEM) -> np.ndarray:
    """(ny, nx) bool on the DEM: cells mostly covered by water, buildings or trees."""
    bbox = dem_bbox(dem)
    land = _remote_classes(bbox) if settings.cv.service_url else classes_for_area(bbox)
    return imagery_grid(land, dem, (WATER, BUILT, TREES))
