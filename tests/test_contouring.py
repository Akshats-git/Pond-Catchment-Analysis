"""Phase 16. Contours generated from a DEM (FR-2), and the bbox modes that use them."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.contouring import (
    ContourError,
    _join,
    _segments,
    contour_levels,
    contours_from_dem,
    nice_interval,
)
from app.core.raster_dem import AreaOfInterest, RasterSurface
from app.main import app

SHEET = "81.2814,21.2398,81.3126,21.2636"
PREFIX = settings.api.api_prefix


@pytest.fixture(scope="module")
def sheet_dem():
    return RasterSurface(AreaOfInterest.from_bbox([float(v) for v in SHEET.split(",")])).sample()


def cone(ny=40, nx=50):
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    return np.hypot(xx - nx / 2, yy - ny / 2)


# --------------------------------------------------------------------------- #
# Marching squares
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("radius", [3.0, 7.5, 12.0])
def test_a_cone_gives_one_closed_circle_per_level(radius):
    lines = _join(*_segments(cone(), radius))
    assert len(lines) == 1
    ring = lines[0]
    assert np.allclose(ring[0], ring[-1])
    r = np.hypot(ring[:, 0] - 25, ring[:, 1] - 20)
    assert np.abs(r - radius).max() < 0.1


def test_a_line_crossing_the_border_stays_open():
    yy, xx = np.mgrid[0:10, 0:10].astype(float)
    lines = _join(*_segments(xx, 4.5))
    assert len(lines) == 1 and len(lines[0]) == 10
    assert np.allclose(lines[0][:, 0], 4.5)


def test_a_saddle_does_not_cross_itself():
    z = np.array([[1.0, 0.0], [0.0, 1.0]])
    lines = _join(*_segments(z, 0.5))
    assert len(lines) == 2


def test_no_data_breaks_the_line_rather_than_bridging_it():
    z = cone()
    z[:, 25] = np.nan
    lines = _join(*_segments(z, 10.0))
    assert len(lines) == 2


@pytest.mark.parametrize("span, interval", [(38, 2.0), (3, 0.5), (400, 20.0), (1200, 50.0)])
def test_the_interval_is_round(span, interval):
    assert nice_interval(span) == interval


def test_levels_are_strictly_inside_the_range():
    assert contour_levels(266.2, 301.1, 5.0).tolist() == [270, 275, 280, 285, 290, 295, 300]


# --------------------------------------------------------------------------- #
# On a real DEM
# --------------------------------------------------------------------------- #
def test_every_vertex_lies_at_its_level(sheet_dem):
    from scipy.ndimage import map_coordinates

    drawn = contours_from_dem(sheet_dem)
    xy = sheet_dem.projection.forward(drawn.points)
    c = (xy[:, 0] - sheet_dem.origin_xy[0]) / sheet_dem.resolution_m
    r = (xy[:, 1] - sheet_dem.origin_xy[1]) / sheet_dem.resolution_m
    z = map_coordinates(sheet_dem.z, [r, c], order=1, mode="nearest")
    assert np.abs(z - drawn.elevations).max() < 1e-6


def test_the_result_is_a_contour_set_like_the_parsers(sheet_dem):
    drawn = contours_from_dem(sheet_dem, interval_m=5.0)
    meta = drawn.metadata
    assert meta.elevation_source == "dem" and meta.interval_m == 5.0
    assert all(level % 5 == 0 for level in meta.levels)
    assert drawn.line_starts[-1] == drawn.vertex_count
    assert drawn.line_count == len(drawn.line_starts) - 1


def test_too_fine_an_interval_is_refused(sheet_dem):
    with pytest.raises(ContourError) as err:
        contours_from_dem(sheet_dem, interval_m=0.01)
    assert err.value.code == "invalid_interval"


# --------------------------------------------------------------------------- #
# The bbox modes
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_contours_for_an_area(client):
    response = client.post(f"{PREFIX}/contours", data={"bbox": SHEET})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["elevation_source"] == "dem"
    assert body["contour_count"] > 50
    assert body["geojson"]["features"][0]["properties"]["role"] == "contour"


def test_contours_with_an_interval(client):
    body = client.post(f"{PREFIX}/contours", data={"bbox": SHEET, "interval_m": "5"}).json()
    assert body["interval_m"] == 5.0
    assert body["index_interval_m"] == 25.0


def test_a_bad_bbox_field_is_a_422(client):
    response = client.post(f"{PREFIX}/contours", data={"bbox": "1,2,3"})
    assert response.status_code == 422 and response.json()["code"] == "invalid_aoi"


def test_the_upload_still_wins_when_both_are_sent(client):
    from tests.fixtures.make_synthetic import VALLEY

    response = client.post(
        f"{PREFIX}/contours",
        files={"contour_map": ("v.kml", VALLEY.to_kml(), "application/xml")},
        data={"bbox": SHEET},
    )
    assert response.status_code == 200
    assert response.json()["elevation_source"] != "dem"


def test_render_an_area(client):
    response = client.post(
        f"{PREFIX}/renderMap", data={"bbox": SHEET, "basemap": "hillshade", "width": "600", "height": "450"}
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert "nearest metre" in response.headers.get("x-pond-warnings", "")


def test_render_an_area_rejects_a_bad_frame(client):
    response = client.post(f"{PREFIX}/renderMap", data={"bbox": SHEET, "frame": "moon"})
    assert response.status_code == 422
