"""Phase 15. `POST /analyzeArea`, the map path through the HTTP surface.

Runs on the committed seed tiles for the sample sheet, so it is offline and fast: the
whole analysis, ensemble included, is under a second here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

AREA = f"{settings.api.api_prefix}/analyzeArea"
SHEET_BBOX = [81.2814, 21.2398, 81.3126, 21.2636]
POLYGON = [[81.285, 21.241], [81.31, 21.243], [81.30, 21.262], [81.288, 21.255]]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def sheet(client):
    response = client.post(AREA, json={"bbox": SHEET_BBOX})
    assert response.status_code == 200, response.text
    return response.json()


def inside(lonlat, bbox):
    return bbox[0] <= lonlat["lon"] <= bbox[2] and bbox[1] <= lonlat["lat"] <= bbox[3]


# --------------------------------------------------------------------------- #
# The answer
# --------------------------------------------------------------------------- #
def test_the_response_has_the_contour_paths_shape(sheet):
    for key in ("input", "dem", "parameters", "recommended_site", "alternative_sites",
                "search", "geojson", "warnings", "timing_ms"):
        assert key in sheet
    assert sheet["input"]["source"] == "elevation_tiles"
    assert sheet["area"]["kind"] == "bbox"


def test_every_site_is_inside_the_selection(sheet):
    for site in [sheet["recommended_site"], *sheet["alternative_sites"]]:
        assert inside(site["location"], SHEET_BBOX)


def test_the_analysed_area_is_the_buffered_selection(sheet):
    area = sheet["area"]
    assert area["analysed_area_ha"] > 2 * area["selection_area_ha"]
    a, s = area["analysed_bbox"], area["selection_bbox"]
    assert a[0] < s[0] and a[1] < s[1] and a[2] > s[2] and a[3] > s[3]


def test_the_sheet_gives_a_catchment_near_the_surveys(sheet):
    """docs/VALIDATION.md: the survey's site 1 is 66.3 ha; the map path's is ~87 ha."""
    area = sheet["recommended_site"]["catchment"]["area_ha"]
    assert 40 <= area <= 150


def test_the_ensemble_runs_by_default_on_this_path(sheet):
    catchment = sheet["recommended_site"]["catchment"]
    assert catchment["confidence"] in ("high", "medium", "low")
    assert len(catchment["grid_resolutions_m"]) == 3


def test_it_says_what_the_heights_are(sheet):
    assert any("nearest metre" in w for w in sheet["warnings"])
    assert sheet["recommended_site"]["catchment"]["method"].startswith("D8 steepest-descent on SRTM")


def test_the_network_layer_is_there(sheet):
    network = sheet["network"]
    assert network["type"] == "FeatureCollection" and network["features"]
    assert all(f["properties"]["role"] == "stream" for f in network["features"])


def test_it_is_fast(sheet):
    assert sheet["timing_ms"]["total"] < 5000


def test_a_polygon_works_and_keeps_sites_inside_it(client):
    response = client.post(AREA, json={"polygon": POLYGON, "ensemble": False})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["area"]["kind"] == "polygon"
    from app.core.raster_dem import AreaOfInterest

    aoi = AreaOfInterest.from_polygon(POLYGON)
    for site in [body["recommended_site"], *body["alternative_sites"]]:
        assert aoi.contains(site["location"]["lon"], site["location"]["lat"])


def test_geojson_geometry_is_accepted(client):
    geometry = {"type": "Polygon", "coordinates": [POLYGON + [POLYGON[0]]]}
    assert client.post(AREA, json={"geometry": geometry, "ensemble": False}).status_code == 200


def test_a_named_pour_point_is_analysed(client):
    body = client.post(
        AREA, json={"bbox": SHEET_BBOX, "lat": 21.2451, "lon": 81.2911, "ensemble": False}
    ).json()
    assert body["search"] is None
    assert len(body["alternative_sites"]) == 0


# --------------------------------------------------------------------------- #
# Asking wrongly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("body, code", [
    ({}, "invalid_request"),
    ({"bbox": [1, 2, 3]}, "invalid_aoi"),
    ({"bbox": [81.3, 21.2, 81.2, 21.3]}, "invalid_aoi"),
    ({"bbox": [80.0, 20.0, 81.5, 21.5]}, "aoi_too_large"),
    ({"bbox": [81.29, 21.25, 81.2901, 21.2501]}, "aoi_too_small"),
    ({"bbox": SHEET_BBOX, "polygon": POLYGON}, "invalid_request"),
    ({"bbox": SHEET_BBOX, "curve_number": 200}, "invalid_request"),
    ({"bbox": SHEET_BBOX, "grid_resolution": 1.0}, "invalid_resolution"),
    ({"bbox": SHEET_BBOX, "lat": 10.0, "lon": 10.0}, "pour_point_outside_map"),
    ({"polygon": [[81.0, 21.0]]}, "invalid_aoi"),
])
def test_every_bad_ask_is_a_named_422(client, body, code):
    response = client.post(AREA, json=body)
    assert response.status_code == 422, response.text
    assert response.json()["code"] == code
    assert "Value error" not in response.json()["detail"]


def test_an_area_with_no_tiles_is_a_503(client):
    """Nowhere near the committed tiles, with the network off in tests."""
    response = client.post(AREA, json={"bbox": [10.0, 45.0, 10.02, 45.02]})
    assert response.status_code == 503
    assert response.json()["code"] == "elevation_unavailable"
