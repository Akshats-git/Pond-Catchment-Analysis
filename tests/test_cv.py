"""Phase 23. Existing water and land availability from imagery.

The imagery is synthetic, with a known answer: a smooth dark pond of known size in
textured fields, beside a rough bright village block. The tiles are real JPEG-free PNGs
served through the same fetch path as Esri's, so decoding and resampling are exercised.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.core.projection import EquirectangularENU
from app.core.raster_dem import AreaOfInterest, RasterSurface
from app.cv import imagery
from app.cv.imagery import BUILT, WATER, classify, fetch_imagery, imagery_grid
from app.cv.service import detect_ponds, land_availability
from app.main import app
from tests.fixtures.synthetic_tiles import pixel_lonlat

BBOX = (81.30, 21.25, 81.31, 21.259)
CENTRE = (81.303, 21.2545)
POND_RADIUS_M = 60.0
VILLAGE = (81.3065, 21.2525, 81.3085, 21.2555)  # a ~200 x 330 m block
PROJ = EquirectangularENU(lon0=CENTRE[0], lat0=CENTRE[1])


def scene(lon, lat, *, pond_radius=POND_RADIUS_M, extra_ponds=()):
    rng = np.random.default_rng(abs(hash((round(float(lon[0, 0]), 5), round(float(lat[0, 0]), 5)))) % 2**32)
    xy = PROJ.forward(np.stack([lon, lat], axis=-1))
    # Fields: mid green with strong pixel texture.
    rgb = np.stack([0.26, 0.33, 0.20]) + rng.normal(0, 0.07, lon.shape + (3,))
    pond = np.hypot(xy[..., 0], xy[..., 1]) < pond_radius
    for (px, py, pr) in extra_ponds:
        pond |= np.hypot(xy[..., 0] - px, xy[..., 1] - py) < pr
    rgb[pond] = (0.15, 0.18, 0.17)
    village = (lon > VILLAGE[0]) & (lon < VILLAGE[2]) & (lat > VILLAGE[1]) & (lat < VILLAGE[3])
    grey = 0.55 + rng.normal(0, 0.18, village.sum())
    rgb[village] = np.stack([grey, grey * 0.97, grey * 0.9], axis=-1)
    return np.clip(rgb, 0, 1)


def fetcher(**kwargs):
    def fetch(z, x, y):
        lon, lat = pixel_lonlat(z, x, y)
        buffer = io.BytesIO()
        Image.fromarray((scene(lon, lat, **kwargs) * 255).astype(np.uint8)).save(buffer, format="PNG")
        return buffer.getvalue()
    return fetch


@pytest.fixture(scope="module")
def land():
    return classify(fetch_imagery(BBOX, fetch=fetcher()))


def test_the_pond_is_water(land):
    area = land.mask(WATER).sum() * land.frame.resolution_m ** 2
    assert area == pytest.approx(math.pi * POND_RADIUS_M ** 2, rel=0.15)


def test_the_village_is_built(land):
    frame = land.frame
    rows, cols = np.nonzero(land.mask(BUILT))
    lonlat = frame.lonlat_of(rows, cols)
    inside = (lonlat[:, 0] > VILLAGE[0] - 3e-4) & (lonlat[:, 0] < VILLAGE[2] + 3e-4)
    assert inside.mean() > 0.9
    assert land.mask(BUILT).sum() * frame.resolution_m ** 2 > 0.5 * 200 * 330


def test_fields_are_available(land):
    assert land.fractions()["available"] > 0.7


def test_detect_ponds_returns_one_polygon_per_pond():
    aoi = AreaOfInterest.from_bbox(BBOX)
    body = detect_ponds(aoi, fetch=fetcher(extra_ponds=[(400, 200, 40)]))
    assert body["count"] == 2
    first = body["geojson"]["features"][0]["properties"]
    assert first["role"] == "existing_water"
    assert first["area_m2"] == pytest.approx(math.pi * POND_RADIUS_M ** 2, rel=0.15)
    assert first["centroid"] == pytest.approx(CENTRE, abs=2e-4)


def test_a_puddle_is_not_a_pond():
    aoi = AreaOfInterest.from_bbox(BBOX)
    body = detect_ponds(aoi, fetch=fetcher(pond_radius=15.0))  # ~700 m2, under the minimum
    assert body["count"] == 0


def test_land_availability_adds_up():
    body = land_availability(AreaOfInterest.from_bbox(BBOX), fetch=fetcher())
    shares = sum(c["share"] for c in body["classes"].values()) + body["available_share"]
    assert shares == pytest.approx(1.0, abs=0.01)
    assert {f["properties"]["class"] for f in body["geojson"]["features"]} >= {"water", "built"}


def test_the_mask_lands_on_the_analysis_grid(land):
    from app.cv.imagery import _frame

    grid = _frame(BBOX, 20.0)
    mask = imagery_grid(land, grid, (WATER,))
    area = mask.sum() * 400.0
    assert area == pytest.approx(math.pi * POND_RADIUS_M ** 2, rel=0.35)


# --------------------------------------------------------------------------- #
# Over HTTP
# --------------------------------------------------------------------------- #
@pytest.fixture
def client(monkeypatch):
    fetch = fetcher()

    def tile(url, key):
        _, z, x, y = key
        return fetch(z, x, y)

    monkeypatch.setattr(imagery, "_tile", tile)
    return TestClient(app)


def test_detect_ponds_endpoint(client):
    body = client.post(f"{settings.api.api_prefix}/imagery/detectPonds", json={"bbox": list(BBOX)}).json()
    assert body["status"] == "ok" and body["count"] == 1


def test_land_available_endpoint_with_a_polygon(client):
    polygon = [[81.300, 21.250], [81.305, 21.250], [81.305, 21.259], [81.300, 21.259]]
    body = client.post(f"{settings.api.api_prefix}/land/available", json={"polygon": polygon}).json()
    # The polygon holds the pond and none of the village.
    assert body["classes"]["built"]["area_ha"] == 0
    assert body["classes"]["water"]["area_ha"] > 0.8


def test_no_imagery_is_a_503(monkeypatch):
    monkeypatch.setattr(imagery, "_tile", lambda url, key: None)
    response = TestClient(app).post(f"{settings.api.api_prefix}/imagery/detectPonds", json={"bbox": list(BBOX)})
    assert response.status_code == 503 and response.json()["code"] == "imagery_unavailable"


def test_the_analysis_keeps_off_unavailable_land(monkeypatch):
    from app.cv import client as cv_client

    monkeypatch.setattr(cv_client, "unavailable_mask_for", lambda dem: np.ones(dem.shape, dtype=bool))
    response = TestClient(app).post(
        f"{settings.api.api_prefix}/analyzeArea",
        json={"bbox": [81.2814, 21.2398, 81.3126, 21.2636], "avoid_unavailable_land": True, "ensemble": False},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "no_site_in_selection"
    assert "available land" in response.json()["detail"]
