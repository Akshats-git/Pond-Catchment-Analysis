"""Phase 9. The HTTP surface.

The analysis itself is validated elsewhere: `test_catchment_analytic.py` proves the area
against a surface with a known answer, `test_massbalance.py` proves nothing leaks, and
`test_hydrology.py` proves the runoff is event-based. What is left for this file is the
boundary: that a request reaches the pipeline carrying the parameters it named, that the
answer arrives in the documented shape, and that every way of asking wrongly comes back
as one status code and one error envelope rather than a stack trace.

Most tests run on the synthetic valley, which goes through the whole service in about a
tenth of a second, so the error paths and the response contract cost nothing to cover.
Two things still need the real sheet. The figures PLAN §3 reports, and the ensemble
confidence on terrain that has some, and those share one module-scoped request.

One property is worth stating in advance, because it looks like a bug in the assertions
below: D8 splits a symmetric valley. The channel of `z = 0.05|x| + 0.01y` falls between
two grid columns, so each column collects half the hillslope and the service returns two
sites of about half the analytic area. The pair sums to the whole valley, which is what
`test_the_channel_sites_account_for_the_whole_valley` checks.
"""

from __future__ import annotations

import io
import os
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import settings
from app.main import app
from app.routers import analyze as analyze_module
from app.core import render as render_module
from app.routers.analyze import UPLOAD_FIELD, UPLOAD_FIELD_ALIAS
from tests.fixtures import make_variants as variants
from tests.fixtures.make_synthetic import VALLEY

SAMPLE = "data/contours_1m.kml"
ENDPOINT = f"{settings.api.api_prefix}/analyzeContour"
ALIAS = f"{settings.api.api_prefix}/findCatchment"
CONTOURS = f"{settings.api.api_prefix}/contours"
RENDER = f"{settings.api.api_prefix}/renderMap"

VALLEY_KML = VALLEY.to_kml()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def post(
    client: TestClient,
    payload: bytes,
    *,
    name: str = "valley.kml",
    url: str = ENDPOINT,
    field: str = UPLOAD_FIELD,
    **form,
):
    """One analysis request. Form values go out as strings, exactly as a browser sends
    them, so the tests exercise the same coercion a real client would.

    The default `field` is the name the brief fixes, so every test in this file that does
    not say otherwise is a test that the documented request works."""
    return client.post(
        url,
        files={field: (name, payload, "application/vnd.google-earth.kml+xml")},
        data={key: str(value) for key, value in form.items()},
    )


def outlet_xy(site: dict) -> np.ndarray:
    """A returned outlet back in the valley's own metres, where the geometry is checkable."""
    location = site["location"]
    return VALLEY.projection().forward(
        np.array([[location["lon"], location["lat"]]], dtype=np.float64)
    )[0]


def sites_of(body: dict) -> list[dict]:
    return [body["recommended_site"], *body["alternative_sites"]]


@pytest.fixture(scope="module")
def valley(client: TestClient) -> dict:
    """One successful analysis of the synthetic valley, reused by the contract tests."""
    response = post(client, VALLEY_KML, ensemble=False)
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def sample(client: TestClient) -> dict:
    """The provided sheet, analysed once with the ensemble on. The acceptance run of
    PLAN Phase 9. About ten seconds, which is why it is shared rather than repeated."""
    if not os.path.exists(SAMPLE):
        pytest.skip(f"{SAMPLE} is not present")
    with open(SAMPLE, "rb") as handle:
        response = post(client, handle.read(), name="contours_1m.kml")
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------- #
# The service surface
# --------------------------------------------------------------------------- #
def test_health_answers_without_doing_any_work(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"] == settings.api.version


def test_health_reports_what_the_host_can_afford(client: TestClient, monkeypatch) -> None:
    """The demo page reads its ensemble switch off /health, so /health has to tell it.

    Without this the page ships with the cross-check ticked, sends `ensemble=true` to a
    host that refuses it, and every upload a grader tries comes back a 422 on a page that
    looks perfectly healthy.
    """
    body = client.get("/health").json()
    assert body["ensemble_available"] is settings.api.allow_ensemble

    small_host = replace(
        settings, api=replace(settings.api, allow_ensemble=False, default_ensemble=True)
    )
    monkeypatch.setattr(main, "settings", small_host)
    body = client.get("/health").json()
    assert body["ensemble_available"] is False
    # Not merely "off by default": a host that cannot run it must not advertise it as the
    # default either, or the page turns the switch back on.
    assert body["ensemble_default"] is False


def test_root_falls_back_to_the_docs_without_the_demo_page(
    client: TestClient, monkeypatch, tmp_path
) -> None:
    """The page is a single static file, so "is it there" is the whole of the Phase 11
    deployment check. If a build ever ships without it, the service stays usable."""
    monkeypatch.setattr(main, "STATIC_DIR", tmp_path)
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == settings.api.docs_url


def test_root_serves_the_demo_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert ENDPOINT in response.text, "the page must post to the endpoint it documents"
    assert client.get("/static/index.html").status_code == 200


def test_demo_page_loads_nothing_from_a_third_party(client: TestClient) -> None:
    """PLAN §5 asks for a CDN-free single file, and the reason is the deployment: the
    page has to work behind a proxy that blocks unpkg and on a free-tier container with
    no build step. Map tiles are the one exception. They are imagery, requested by the
    viewer's browser and credited on the map, so this checks for external *code*.
    """
    page = client.get("/static/index.html").text
    for tag in ("<script src=", "<link rel=\"stylesheet\"", "@import"):
        assert tag not in page, f"{tag!r} would pull code from somewhere else"
    assert page.count("<script") == 1, "one inline script, no external ones"


def test_openapi_documents_the_endpoint_and_its_failures(client: TestClient) -> None:
    """The rubric's API-documentation line is satisfied by `/docs`, which is only worth
    anything if the schema actually describes the errors as well as the success."""
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][ENDPOINT]["post"]
    assert {"400", "413", "422"} <= set(operation["responses"])
    assert client.get(settings.api.docs_url).status_code == 200


def test_alias_is_the_same_endpoint_not_a_second_one(client: TestClient) -> None:
    """The brief names `findCatchment`; the plan names `analyzeContour`. One
    implementation answers both, and only one of them is in the schema."""
    alias = post(client, VALLEY_KML, url=ALIAS, ensemble=False)
    assert alias.status_code == 200
    canonical = post(client, VALLEY_KML, ensemble=False).json()
    assert alias.json()["recommended_site"]["location"] == (
        canonical["recommended_site"]["location"]
    )
    assert ALIAS not in client.get("/openapi.json").json()["paths"]


# --------------------------------------------------------------------------- #
# The upload field
#
# The brief fixes the multipart field name at `contour_map`, and a grader sending it is
# the one request that has to work. `file` is the name this service used first and stays
# accepted, so these three tests pin the whole contract: the documented name works, the
# old name still works, and neither is a legible error rather than a framework one.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", [ENDPOINT, ALIAS, CONTOURS])
@pytest.mark.parametrize("field", [UPLOAD_FIELD, UPLOAD_FIELD_ALIAS])
def test_every_route_takes_the_file_under_either_name(
    client: TestClient, url: str, field: str
) -> None:
    assert post(client, VALLEY_KML, url=url, field=field, ensemble=False).status_code == 200


def test_the_documented_name_wins_when_a_request_sends_both(client: TestClient) -> None:
    """A client that sends both has already agreed with itself about the content, so the
    tie goes to the documented field rather than to multipart ordering."""
    response = client.post(
        ENDPOINT,
        files=[
            (UPLOAD_FIELD, ("valley.kml", VALLEY_KML, "text/xml")),
            (UPLOAD_FIELD_ALIAS, ("junk.kml", b"not a contour sheet", "text/xml")),
        ],
        data={"ensemble": "false"},
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("url", [ENDPOINT, ALIAS, CONTOURS])
def test_a_request_with_no_file_names_the_field_it_wanted(
    client: TestClient, url: str
) -> None:
    """The likeliest failed first request there is. It should say what to send and how,
    not `field required` against a name the sender never chose."""
    response = client.post(url)
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "missing_file"
    assert UPLOAD_FIELD in body["detail"]
    assert "form-data" in body["hint"] and UPLOAD_FIELD_ALIAS in body["hint"]


def test_a_text_row_where_a_file_belongs_names_the_postman_fix(
    client: TestClient,
) -> None:
    """Postman's form-data rows are type Text until somebody changes them, so a row left
    that way sends the filename as a string. Pydantic's own words for that are `Expected
    UploadFile, received str`, which is true and no help. The answer has to name the
    control to change."""
    response = client.post(ENDPOINT, data={UPLOAD_FIELD: "contours_1m.kml"})
    body = response.json()

    assert response.status_code == 422
    assert body["code"] == "invalid_request"
    assert "not as a file" in body["detail"]
    assert "Postman" in body["hint"] and "File" in body["hint"]
    assert "UploadFile" not in body["detail"] and "UploadFile" not in body["hint"]


def test_the_alias_field_gets_the_same_answer_when_sent_as_text(
    client: TestClient,
) -> None:
    body = client.post(ENDPOINT, data={UPLOAD_FIELD_ALIAS: "contours_1m.kml"}).json()

    assert body["code"] == "invalid_request"
    assert "not as a file" in body["detail"]


def test_docs_advertise_the_brief_field_and_only_that_one(client: TestClient) -> None:
    """`/docs` is where a grader looks for the field name, so it must show the one the
    brief fixed and must not offer a second file picker beside it."""
    schema = client.get("/openapi.json").json()
    for url in (ENDPOINT, CONTOURS):
        content = schema["paths"][url]["post"]["requestBody"]["content"]
        model = content["multipart/form-data"]["schema"]["$ref"].rsplit("/", 1)[-1]
        properties = schema["components"]["schemas"][model]["properties"]
        assert UPLOAD_FIELD in properties
        assert UPLOAD_FIELD_ALIAS not in properties


# --------------------------------------------------------------------------- #
# The successful response
# --------------------------------------------------------------------------- #
def test_response_carries_every_documented_block(valley: dict) -> None:
    assert valley["status"] == "ok"
    assert set(valley) >= {
        "input", "dem", "parameters", "recommended_site", "alternative_sites",
        "search", "geojson", "warnings", "timing_ms",
    }
    assert valley["input"]["contour_count"] > 0
    assert valley["input"]["elevation_source"] == "placemark_name"
    assert valley["dem"]["resolution_source"] == "auto"
    assert valley["recommended_site"]["rank"] == 1
    assert [site["rank"] for site in sites_of(valley)] == list(
        range(1, len(sites_of(valley)) + 1)
    )


def test_the_channel_sites_account_for_the_whole_valley(client: TestClient) -> None:
    """The end-to-end area check, in the one place where the answer is known on paper.

    Both halves of the split channel outlet at the same distance up the valley, so their
    catchments partition the ground above that line: `1000 * (1000 - y)` m^2.
    """
    body = post(client, VALLEY_KML, ensemble=False, top_n=2).json()
    sites = sites_of(body)
    assert len(sites) == 2

    xy = [outlet_xy(site) for site in sites]
    resolution = body["dem"]["resolution_m"]
    assert abs(xy[0][1] - xy[1][1]) < resolution, "the two outlets are not level"
    assert all(abs(point[0]) < 2 * resolution for point in xy), "not on the channel"

    total_ha = sum(site["catchment"]["area_ha"] for site in sites)
    analytic_ha = VALLEY.analytic_catchment_area(float(xy[0][1])) / 1e4
    assert total_ha == pytest.approx(analytic_ha, rel=0.03)


def test_explicit_pour_point_replaces_the_search(client: TestClient) -> None:
    """`lat`/`lon` analyse a chosen place: one site, no ranking, and the outlet lands on
    the channel near where it was asked about rather than wherever it liked."""
    lon, lat = VALLEY.channel_point(500.0)
    body = post(client, VALLEY_KML, ensemble=False, lat=lat, lon=lon).json()

    assert body["search"] is None, "no candidate search was run, so none is reported"
    assert body["alternative_sites"] == []
    site = body["recommended_site"]

    x, y = outlet_xy(site)
    resolution = body["dem"]["resolution_m"]
    snap_radius = (
        settings.catchment.snap_radius_spacing_multiple
        * body["dem"]["mean_contour_spacing_m"]
    )
    assert abs(x) < 2 * resolution, "the outlet did not snap onto the channel"
    assert abs(y - 500.0) <= snap_radius * 2 ** 0.5
    assert site["snap_distance_m"] > 0

    # Half the analytic area, give or take, for the reason in the module docstring.
    analytic_ha = VALLEY.analytic_catchment_area(float(y)) / 1e4
    assert 0.4 * analytic_ha <= site["catchment"]["area_ha"] <= 1.1 * analytic_ha


def test_top_n_limits_the_alternatives(client: TestClient) -> None:
    body = post(client, VALLEY_KML, ensemble=False, top_n=1).json()
    assert body["alternative_sites"] == []
    assert body["recommended_site"]["rank"] == 1


def test_parameters_are_echoed_with_defaults_resolved(client: TestClient) -> None:
    """The response says what the analysis used, not what the client typed, so a report
    built from it can be reproduced without knowing the service's defaults.

    Rainfall is the one field that stays null when it was left out, because what fills it
    is a record fetched for the chosen site. What was actually used is under `runoff`.
    """
    body = post(client, VALLEY_KML, ensemble=False, curve_number=85, rain_days=40).json()
    parameters = body["parameters"]
    assert parameters["curve_number"] == 85
    assert parameters["rain_days"] == 40
    assert parameters["rainfall_mm"] is None
    runoff = body["recommended_site"]["runoff"]
    assert runoff["rain_days"] == 40
    assert runoff["rainfall_mm"] == settings.hydrology.default_annual_rainfall_mm
    assert parameters["target_depth_m"] == settings.hydrology.default_target_depth_m
    assert body["recommended_site"]["runoff"]["curve_number"] == 85
    assert body["recommended_site"]["runoff"]["rain_days"] == 40


def test_runoff_is_event_based_not_annual(valley: dict) -> None:
    """PLAN §4's pitfall, guarded at the API: SCS-CN applied to a year's rain as one
    storm returns about 92%, roughly six times the real yield. Both numbers are
    reported, and the one the service uses has to be the smaller."""
    runoff = valley["recommended_site"]["runoff"]
    assert runoff["runoff_coefficient"] < runoff["single_event_coefficient"]
    assert runoff["overestimate_factor"] > 3.0
    assert 0 < runoff["contributing_days"] <= runoff["rain_days"]
    assert runoff["annual_runoff_m3"] > 0


def test_storage_is_measured_off_the_terrain(valley: dict) -> None:
    storage = valley["recommended_site"]["storage"]
    assert storage["max_depth_m"] == settings.hydrology.default_target_depth_m
    assert storage["capacity_m3"] <= storage["capacity_at_target_depth_m3"]
    assert len(storage["stage_storage"]) >= 2
    depths = [row[0] for row in storage["stage_storage"]]
    volumes = [row[2] for row in storage["stage_storage"]]
    assert depths == sorted(depths)
    assert volumes == sorted(volumes), "storage cannot fall as the water rises"


def test_ensemble_puts_an_error_bar_on_the_area(client: TestClient) -> None:
    body = post(client, VALLEY_KML, ensemble=True).json()
    catchment = body["recommended_site"]["catchment"]
    assert catchment["grid_resolutions_m"] == list(settings.catchment.ensemble_resolutions_m)
    assert catchment["area_uncertainty_ha"] is not None
    assert catchment["confidence"] in {"high", "medium", "low"}


def test_ensemble_off_says_unassessed_rather_than_guessing(valley: dict) -> None:
    catchment = valley["recommended_site"]["catchment"]
    assert catchment["confidence"] == "unassessed"
    assert catchment["area_uncertainty_ha"] is None
    assert catchment["grid_resolutions_m"] == []


def test_why_explains_the_pick_in_the_numbers_that_made_it(valley: dict) -> None:
    why = valley["recommended_site"]["why"]
    assert why and all(isinstance(reason, str) for reason in why)
    assert "largest upstream area" in why[0]
    assert f"{valley['recommended_site']['catchment']['area_ha']:.1f} ha" in why[0]


def test_geojson_is_a_drawable_feature_collection(valley: dict) -> None:
    collection = valley["geojson"]
    assert collection["type"] == "FeatureCollection"
    assert len(collection["bbox"]) == 4
    roles = {feature["properties"]["role"] for feature in collection["features"]}
    assert {"catchment", "pond_site", "flow_path"} <= roles
    for feature in collection["features"]:
        assert feature["geometry"]["coordinates"], "an empty geometry draws nothing"


def test_timings_account_for_every_stage(valley: dict) -> None:
    timings = valley["timing_ms"]
    assert {"parse", "dem", "flow", "siting", "hydrology", "geojson", "total"} <= set(timings)
    assert timings["total"] >= max(v for k, v in timings.items() if k != "total")


def test_warnings_do_not_repeat_themselves(valley: dict) -> None:
    """Several stages warn about the same clipped edge; a response that says it four
    times reads like a fault in the service rather than a caveat about the map."""
    assert len(valley["warnings"]) == len(set(valley["warnings"]))


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #
BAD_FILES = [
    (variants.malformed_xml(), 400, "unparseable_xml"),
    (variants.points_only(), 400, "no_contours"),
    (variants.single_level(), 400, "too_few_levels"),
    (variants.unlabelled_contours(), 400, "no_elevations"),
    (b"", 400, "empty_upload"),
]


@pytest.mark.parametrize("payload,expected_status,expected_code", BAD_FILES)
def test_unusable_files_are_400(
    client: TestClient, payload: bytes, expected_status: int, expected_code: str
) -> None:
    response = post(client, payload)
    assert response.status_code == expected_status
    assert response.json()["code"] == expected_code


BAD_REQUESTS = [
    ({"curve_number": 200}, "invalid_parameters"),
    ({"top_n": 0}, "invalid_parameters"),
    ({"top_n": settings.siting.max_top_n + 1}, "invalid_parameters"),
    ({"rain_days": 0}, "invalid_parameters"),
    ({"target_depth_m": 0}, "invalid_parameters"),
    ({"grid_resolution": -1}, "invalid_parameters"),
    ({"lat": 21.24}, "invalid_parameters"),
    ({"grid_resolution": settings.dem.min_resolution_m / 4}, "invalid_resolution"),
]


@pytest.mark.parametrize("form,expected_code", BAD_REQUESTS)
def test_bad_parameters_are_422(client: TestClient, form: dict, expected_code: str) -> None:
    response = post(client, VALLEY_KML, **form)
    assert response.status_code == 422
    assert response.json()["code"] == expected_code


def test_parameters_are_checked_before_the_terrain_work(client: TestClient) -> None:
    """A curve number of 200 is knowable as wrong without building a DEM. The `code`
    proves which layer rejected it: `invalid_parameters` is the door, and
    `curve_number_out_of_range` would mean the request had already paid for an analysis
    it was never going to keep."""
    body = post(client, VALLEY_KML, curve_number=200).json()
    assert body["code"] == "invalid_parameters"
    assert "curve_number" in body["detail"]


def test_half_a_pour_point_names_the_missing_half(client: TestClient) -> None:
    body = post(client, VALLEY_KML, lat=21.24).json()
    assert "lon" in body["detail"]


def test_pour_point_off_the_sheet_is_422(client: TestClient) -> None:
    body = post(client, VALLEY_KML, ensemble=False, lat=0.0, lon=0.0)
    assert body.status_code == 422
    assert body.json()["code"] == "pour_point_outside_map"
    assert body.json()["hint"], "a rejected pour point should say what to do instead"


def test_a_form_field_of_the_wrong_type_is_a_structured_422(client: TestClient) -> None:
    """FastAPI rejects a mistyped form field before the route body runs; the handler has
    to catch it, or clients meet a second error format.

    A missing file no longer lands here. The route accepts the sheet under either of two
    field names, so it cannot be declared required and is checked in the body instead,
    which is what lets the answer name the field: see
    `test_a_request_with_no_file_names_the_field_it_wanted`."""
    response = client.post(
        ENDPOINT,
        files={UPLOAD_FIELD: ("valley.kml", VALLEY_KML, "text/xml")},
        data={"top_n": "not a number"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "invalid_request"
    assert "top_n" in body["detail"]


def test_upload_over_the_limit_is_413(client: TestClient, monkeypatch) -> None:
    """Refused while reading the spooled upload, so an oversized sheet never becomes a
    single `bytes` object on its way to the parser."""
    tiny = replace(settings, parser=replace(settings.parser, max_upload_bytes=1024))
    monkeypatch.setattr(analyze_module, "settings", tiny)
    response = post(client, VALLEY_KML)
    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"


def test_unexpected_extension_warns_rather_than_refuses(client: TestClient) -> None:
    """The parser sniffs content, so a misnamed file still analyses. Saying so beats
    rejecting a valid contour sheet over its name."""
    response = post(client, VALLEY_KML, name="contours.txt", ensemble=False)
    assert response.status_code == 200
    assert any("contours.txt" in warning for warning in response.json()["warnings"])


def test_kmz_is_accepted(client: TestClient) -> None:
    response = post(client, variants.kmz(VALLEY_KML), name="valley.kmz", ensemble=False)
    assert response.status_code == 200
    assert response.json()["recommended_site"]["catchment"]["area_ha"] > 0


@pytest.mark.parametrize(
    "payload,form",
    [
        (variants.malformed_xml(), {}),
        (VALLEY_KML, {"curve_number": 200}),
        (VALLEY_KML, {"lat": 0.0, "lon": 0.0}),
        (b"", {}),
    ],
)
def test_every_failure_uses_one_envelope(client: TestClient, payload: bytes, form: dict) -> None:
    """One shape for every error in the service, whatever raised it. Getting that is the
    whole reason the core modules carry `(code, detail, hint)` instead of bare messages."""
    body = post(client, payload, **form).json()
    assert set(body) == {"status", "code", "detail", "hint"}
    assert body["status"] == "error"
    assert body["code"] and body["detail"]


# --------------------------------------------------------------------------- #
# The provided sheet. PLAN §3, through the API
# --------------------------------------------------------------------------- #
def test_sample_sheet_reproduces_the_documented_analysis(sample: dict) -> None:
    """PLAN Phase 9's acceptance criterion: the sample map returns the §3 results.

    The catchment is checked against the ensemble's own error bar rather than against a
    literal, so the test measures agreement between the grids, which is the claim, and
    does not have to be edited whenever the derived resolution moves.
    """
    assert sample["input"]["interval_m"] == pytest.approx(1.0)
    assert sample["input"]["elevation_source"] == "placemark_name"
    assert sample["input"]["mapped_area_ha"] == pytest.approx(830.9, rel=0.02)

    site = sample["recommended_site"]
    catchment = site["catchment"]
    assert site["is_recommended"]
    assert catchment["confidence"] in {"high", "medium"}
    assert 20 < catchment["area_ha"] < 150, "a village pond catchment, not a river basin"
    assert catchment["area_ha"] == pytest.approx(
        catchment["ensemble_mean_area_ha"], abs=3 * catchment["area_uncertainty_ha"]
    )
    assert not catchment["is_lower_bound"]
    assert catchment["edge_contact_pct"] < settings.catchment.edge_contact_warn_fraction * 100


def test_sample_sheet_runoff_is_in_the_expected_band(sample: dict) -> None:
    """11-19% is what this terrain yields (PLAN §4). Outside it, either the curve number
    or the per-day summation is wrong, and both are worth failing over."""
    low, high = settings.hydrology.expected_runoff_coefficient_range
    runoff = sample["recommended_site"]["runoff"]
    site = sample["recommended_site"]
    assert low <= runoff["runoff_coefficient"] <= high
    assert runoff["annual_runoff_m3"] == pytest.approx(
        runoff["runoff_depth_mm"] / 1000 * site["catchment"]["area_ha"] * 1e4, rel=0.01
    )


def test_sample_alternatives_are_independent_basins(sample: dict) -> None:
    """Suppression removes each pick's whole catchment, so the alternatives are separate
    sub-basins rather than five points strung along one stream (PLAN §11.7)."""
    sites = sites_of(sample)
    assert len(sites) == settings.siting.default_top_n
    areas = [site["catchment"]["area_ha"] for site in sites]
    assert areas == sorted(areas, reverse=True)
    locations = {(site["location"]["lat"], site["location"]["lon"]) for site in sites}
    assert len(locations) == len(sites)


def test_sample_search_reports_what_it_considered(sample: dict) -> None:
    search = sample["search"]
    assert 0 < search["candidate_cells"] <= min(
        search["stream_cells"], search["buildable_cells"]
    )
    # Both sides are reported to 0.1 ha, so the tolerance is the rounding, not a fudge.
    assert search["stream_threshold_ha"] == pytest.approx(
        sample["input"]["mapped_area_ha"] * settings.siting.stream_threshold_fraction,
        abs=0.1,
    )



# --------------------------------------------------------------------------- #
# Concurrency: the limiter that keeps a small host alive
# --------------------------------------------------------------------------- #
def test_concurrent_analyses_run_one_at_a_time(monkeypatch) -> None:
    """Analyses arriving together must queue, not run side by side.

    This is a memory bound with a body count. Unlimited, a second concurrent request
    does not merely slow the first one down: on the 512 MB container both peak together,
    the kernel kills the worker they share, and *both* clients get an empty reply. That
    is measured rather than guessed - it is what the deployed service did before
    `max_concurrent_analyses` existed.

    The real analysis runs, so the response still has to validate; only the counting is
    added around it.
    """
    import asyncio
    import threading

    import httpx

    real_analyse = analyze_module.analyse
    lock = threading.Lock()
    running = 0
    peak = 0

    def counting_analyse(*args, **kwargs):
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        try:
            return real_analyse(*args, **kwargs)
        finally:
            with lock:
                running -= 1

    monkeypatch.setattr(analyze_module, "analyse", counting_analyse)

    async def hammer() -> list[int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            calls = [
                ac.post(
                    ENDPOINT,
                    files={UPLOAD_FIELD: ("valley.kml", VALLEY_KML, "text/xml")},
                    data={"ensemble": "false"},
                )
                for _ in range(3)
            ]
            return [response.status_code for response in await asyncio.gather(*calls)]

    statuses = asyncio.run(hammer())

    assert statuses == [200, 200, 200], "every queued analysis still gets served"
    assert peak <= settings.api.max_concurrent_analyses, (
        f"{peak} analyses ran at once, limit is {settings.api.max_concurrent_analyses}"
    )


def test_ensemble_is_refused_when_the_host_cannot_afford_it(client: TestClient, monkeypatch) -> None:
    """A host too small for the ensemble says so, rather than dying trying.

    `default_ensemble=false` alone leaves a hole: a client that reads /docs and asks for
    `ensemble=true` anyway would take the worker's memory with it. The answer is a 422
    naming the limit, and a service still standing to serve the next request.
    """
    small_host = replace(settings, api=replace(settings.api, allow_ensemble=False))
    monkeypatch.setattr(analyze_module, "settings", small_host)

    refused = post(client, VALLEY_KML, ensemble=True)
    assert refused.status_code == 422
    body = refused.json()
    assert body["status"] == "error"
    assert body["code"] == "ensemble_unavailable"
    assert "ensemble=false" in body["hint"]

    # The point of refusing: the very next request still works.
    assert post(client, VALLEY_KML, ensemble=False).status_code == 200


# --------------------------------------------------------------------------- #
# The contour overlay
#
# `/contours` exists for one reason: a catchment boundary on satellite imagery cannot be
# checked by eye, because imagery does not show where the ridges are. So what these tests
# hold it to is that the lines it draws are the same lines the analysis read, and that it
# is cheap enough to ask for while the reader is still filling in the panel.
# --------------------------------------------------------------------------- #
def test_contours_come_back_as_a_drawable_collection(client: TestClient) -> None:
    body = post(client, VALLEY_KML, url=CONTOURS).json()
    collection = body["geojson"]

    assert body["status"] == "ok"
    assert collection["type"] == "FeatureCollection"
    assert len(collection["features"]) == body["contour_count"] > 0
    assert len(collection["bbox"]) == 4
    for feature in collection["features"]:
        assert feature["geometry"]["type"] == "LineString"
        assert isinstance(feature["properties"]["elevation_m"], float)
        assert feature["properties"]["stroke"].startswith("#")


def test_the_lines_drawn_are_the_lines_analysed(client: TestClient) -> None:
    """The whole value of the overlay is that it is the same reading of the file. Two
    parses of one sheet that disagreed on how many lines are in it would make the picture
    a second opinion rather than evidence."""
    drawn = post(client, VALLEY_KML, url=CONTOURS).json()
    analysed = post(client, VALLEY_KML, ensemble=False).json()["input"]

    assert drawn["contour_count"] == analysed["contour_count"]
    assert drawn["source_vertex_count"] == analysed["vertex_count"]
    assert drawn["elevation_source"] == analysed["elevation_source"]
    assert drawn["interval_m"] == analysed["interval_m"]
    assert drawn["elevation_range_m"] == analysed["elevation_range_m"]
    assert drawn["bbox"] == analysed["bbox"]


def test_the_overlay_is_thinned_and_says_by_how_much(client: TestClient) -> None:
    body = post(client, VALLEY_KML, url=CONTOURS).json()
    assert body["vertex_count"] <= body["source_vertex_count"]
    assert body["simplify_tolerance_m"] == settings.geojson.contour_simplify_tolerance_m
    # Below the finest grid the service will ever build, so the overlay cannot disagree
    # with the catchment by more than the analysis could resolve anyway.
    assert body["simplify_tolerance_m"] < settings.dem.min_resolution_m


def test_simplification_is_the_clients_to_choose(client: TestClient) -> None:
    coarse = post(client, VALLEY_KML, url=CONTOURS, simplify_m=20).json()
    exact = post(client, VALLEY_KML, url=CONTOURS, simplify_m=0).json()
    assert coarse["vertex_count"] < exact["vertex_count"]
    assert exact["vertex_count"] == exact["source_vertex_count"]
    assert coarse["simplify_tolerance_m"] == 20.0


def test_an_impossible_simplification_is_422(client: TestClient) -> None:
    refused = post(client, VALLEY_KML, url=CONTOURS, simplify_m=-1)
    assert refused.status_code == 422
    assert refused.json()["code"] == "invalid_simplify"


def test_a_file_with_no_contours_fails_the_same_way_here(client: TestClient) -> None:
    """One parser, so one error envelope and one code, whichever endpoint was asked."""
    refused = post(client, b"<kml><Document/></kml>", url=CONTOURS)
    assert refused.status_code == 400
    body = refused.json()
    assert body["status"] == "error" and body["code"] == "no_contours"
    assert body["hint"]


def test_drawing_the_contours_is_far_cheaper_than_analysing_them(client: TestClient) -> None:
    """The reason this is its own endpoint rather than a flag on the analysis. It has to
    answer while a reader is still typing, and it has to not put a megabyte of coordinates
    on every analysis response that will never draw them."""
    drawn = post(client, VALLEY_KML, url=CONTOURS).json()
    analysed = post(client, VALLEY_KML, ensemble=False).json()

    assert drawn["timing_ms"]["total"] < analysed["timing_ms"]["total"]
    assert set(drawn["timing_ms"]) == {"parse", "draw", "total"}
    assert "geojson" not in str(analysed["input"])
    roles = {f["properties"]["role"] for f in analysed["geojson"]["features"]}
    assert "contour" not in roles


def test_the_sample_sheet_draws_within_the_vertex_budget(client: TestClient) -> None:
    """The real file, which is where the size question actually bites: 159,113 vertices
    in, and a response a browser can hold."""
    with open(SAMPLE, "rb") as handle:
        body = post(client, handle.read(), name="contours_1m.kml", url=CONTOURS).json()

    assert body["source_vertex_count"] > 100_000
    assert body["vertex_count"] <= settings.geojson.contour_max_vertices
    assert body["vertex_count"] < body["source_vertex_count"] / 4
    assert body["interval_m"] == 1.0
    assert body["index_interval_m"] == 5.0
    heavy = [f for f in body["geojson"]["features"] if f["properties"]["index"]]
    assert heavy and all(f["properties"]["elevation_m"] % 5 == 0 for f in heavy)


def test_a_megabyte_of_contours_travels_compressed(client: TestClient) -> None:
    """Uncompressed, the sample's overlay is about a megabyte, which is a visible wait on
    a village connection. It is the one response on this service big enough to care."""
    with open(SAMPLE, "rb") as handle:
        payload = handle.read()
    response = client.post(
        CONTOURS,
        files={UPLOAD_FIELD: ("contours_1m.kml", payload, "application/vnd.google-earth.kml+xml")},
        headers={"Accept-Encoding": "gzip"},
    )
    assert response.status_code == 200
    assert response.headers.get("content-encoding") == "gzip"


def test_openapi_documents_the_contour_endpoint(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][CONTOURS]["post"]
    assert {"400", "413", "422"} <= set(operation["responses"])


def test_the_demo_page_can_turn_the_contours_on(client: TestClient) -> None:
    """The map is where this feature is actually used, and the page is one static file
    with no build step, so the wiring is checkable by reading it."""
    page = client.get("/static/index.html").text
    assert 'id="contour-toggle"' in page
    assert '/api/v1/contours' in page


# --------------------------------------------------------------------------- #
# The rendered map
#
# `/renderMap` is the same analysis with a PNG on the end of it, and everything worth
# testing here is a consequence of that: the picture has to be of the answer, it has to
# carry the caveats the JSON would have carried, and it must not fail because somebody
# else's tile server did.
#
# Every test below asks for a basemap that needs no network. The fallback path is the one
# place a fetch is simulated, and it is simulated rather than performed: a suite that
# reaches the internet fails in a lab with no route out, which is exactly where this
# service is deployed.
# --------------------------------------------------------------------------- #
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def render(client: TestClient, payload: bytes = VALLEY_KML, **form):
    form.setdefault("ensemble", False)
    form.setdefault("basemap", "hillshade")
    return post(client, payload, url=RENDER, **form)


def test_the_map_comes_back_as_a_png(client: TestClient) -> None:
    response = render(client)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(PNG_MAGIC)
    assert "valley-catchment.png" in response.headers["content-disposition"]


def test_the_image_is_the_size_that_was_asked_for(client: TestClient) -> None:
    from PIL import Image

    response = render(client, width=640, height=480)
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.size == (640, 480)


def test_the_default_size_is_the_configured_one(client: TestClient) -> None:
    from PIL import Image

    with Image.open(io.BytesIO(render(client).content)) as image:
        assert image.size == (settings.render.default_width, settings.render.default_height)


def test_a_png_carries_the_warnings_a_json_response_would_have(client: TestClient) -> None:
    """The one thing an image cannot do is carry a caveat, and dropping "this rainfall is
    a climatology, not an observation" because the client asked for a picture would be the
    service deciding what the client is allowed to know."""
    analysed = post(client, VALLEY_KML, ensemble=False).json()
    header = render(client).headers.get(analyze_module.WARNINGS_HEADER, "")

    assert analysed["warnings"], "the valley should warn about something"
    for warning in analysed["warnings"]:
        assert warning[:40] in header


def test_the_warning_header_stays_inside_what_a_header_can_hold(client: TestClient) -> None:
    """Header values are single-line latin-1 and servers cap their length. A warning list
    that grew past either would take the whole response down with it."""
    monstrous = ["x" * 400, "a caveat with a — dash in it", "another\nline"] * 6
    header = analyze_module._warning_header(monstrous)[analyze_module.WARNINGS_HEADER]

    assert len(header) <= analyze_module._HEADER_LIMIT
    assert "\n" not in header
    header.encode("latin-1")  # raises if anything survived that cannot be sent


def test_no_warnings_means_no_header(client: TestClient) -> None:
    assert analyze_module._warning_header([]) == {}


def test_the_picture_is_of_the_same_analysis_as_the_json(client: TestClient) -> None:
    """Two endpoints, one pipeline. If the render ran its own analysis with its own
    defaults the image could show a different answer from the numbers beside it, and a
    reader has no way to notice."""
    body = post(client, VALLEY_KML, ensemble=False, top_n=2).json()
    response = render(client, top_n=2)

    assert response.status_code == 200
    assert len(sites_of(body)) == 2
    # Same parameters reach the pipeline, so the same warnings come out of it.
    assert body["warnings"][0][:40] in response.headers[analyze_module.WARNINGS_HEADER]


def test_a_render_refuses_a_size_it_cannot_afford(client: TestClient) -> None:
    """The overlay is drawn supersampled in RGBA, so the pixel count is a memory bound
    and not a matter of taste."""
    too_big = render(client, width=settings.render.max_size_px + 1)
    too_small = render(client, height=settings.render.min_size_px - 1)

    for response in (too_big, too_small):
        assert response.status_code == 422
        assert response.json()["code"] == "invalid_image_size"
        assert "hint" in response.json()


def test_an_unknown_basemap_names_the_ones_that_exist(client: TestClient) -> None:
    body = render(client, basemap="moon").json()

    assert body["code"] == "invalid_basemap"
    for name in settings.render.basemaps:
        assert name in body["detail"] or name in body["hint"]


def test_an_unknown_frame_is_refused(client: TestClient) -> None:
    body = render(client, frame="galaxy").json()
    assert body["code"] == "invalid_frame"


def test_the_render_rejects_what_the_analysis_rejects(client: TestClient) -> None:
    """One error table, one envelope. A bad curve number is a bad curve number whether
    the client wanted JSON or a picture."""
    body = render(client, curve_number=200).json()

    assert body["status"] == "error"
    assert body["code"] == "invalid_parameters"


def test_a_missing_file_names_both_accepted_fields(client: TestClient) -> None:
    response = client.post(RENDER, data={"basemap": "hillshade"})
    body = response.json()

    assert response.status_code == 422
    assert body["code"] == "missing_file"
    assert UPLOAD_FIELD in body["detail"] and UPLOAD_FIELD_ALIAS in body["hint"]


def test_the_hillshade_needs_no_network_at_all(client: TestClient, monkeypatch) -> None:
    """The reason it is the fallback. This test fails loudly if any code path under
    `basemap=hillshade` ever grows a fetch."""

    def forbidden(*args, **kwargs):
        raise AssertionError("the hillshade basemap reached for the network")

    monkeypatch.setattr(render_module.urllib.request, "urlopen", forbidden)
    assert render(client).status_code == 200


def test_a_tile_server_that_is_down_degrades_to_the_hillshade(
    client: TestClient, monkeypatch
) -> None:
    """Somebody else's outage must not become this service's. A 502 here would be the
    render refusing to draw a catchment it had already computed."""
    monkeypatch.setattr(render_module, "_TILE_CACHE", type(render_module._TILE_CACHE)())

    def unreachable(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(render_module.urllib.request, "urlopen", unreachable)
    response = render(client, basemap="satellite")

    assert response.status_code == 200
    assert response.content.startswith(PNG_MAGIC)
    assert "hillshade" in response.headers[analyze_module.WARNINGS_HEADER]


def test_a_failed_tile_is_not_asked_for_twice(monkeypatch) -> None:
    """A tile that 404s at this zoom will 404 on every retry, and re-asking a provider
    that is rate-limiting you is how a service gets itself blocked for good."""
    monkeypatch.setattr(render_module, "_TILE_CACHE", type(render_module._TILE_CACHE)())
    calls = []

    def unreachable(*args, **kwargs):
        calls.append(1)
        raise OSError("nope")

    monkeypatch.setattr(render_module.urllib.request, "urlopen", unreachable)
    for _ in range(3):
        render_module._tile_bytes("https://example.invalid/1/2/3", ("sat", 1, 2, 3), settings.render)
    assert len(calls) == 1


def test_the_view_fills_the_frame_it_was_given() -> None:
    """An integer-zoom fit leaves up to half the canvas empty whenever the extent falls
    just past a power of two. The drawn extent should touch the padding on one axis."""
    bbox = (81.28, 21.24, 81.31, 21.26)
    view = render_module.fit_view(bbox, 1200, 900, cfg=settings.render)

    x0, y0 = view.to_px(bbox[0], bbox[3])
    x1, y1 = view.to_px(bbox[2], bbox[1])
    used_w = float(x1 - x0) / (1200 - 2 * settings.render.padding_px)
    used_h = float(y1 - y0) / (900 - 2 * settings.render.padding_px)

    assert max(used_w, used_h) == pytest.approx(1.0, abs=0.01)
    assert view.tile_zoom >= view.zoom  # never stretched, only downsampled


def test_dense_contours_thin_to_index_lines_and_say_so(client: TestClient) -> None:
    """One-pixel lines six pixels apart stop reading as lines and become a haze over the
    imagery, hiding the ground the contours were drawn to let the reader check. A printed
    sheet drops to every fifth line for the same reason, and a reader counting intervals
    off the picture has to be told the interval changed."""
    small = render(client, width=settings.render.min_size_px, height=settings.render.min_size_px)
    header = small.headers.get(analyze_module.WARNINGS_HEADER, "")

    assert small.status_code == 200
    assert "contour lines fall" in header


def test_the_contours_can_be_left_off(client: TestClient) -> None:
    bare = render(client, contours=False)
    drawn = render(client, contours=True)

    assert bare.status_code == drawn.status_code == 200
    # Fewer strokes is less entropy, so a PNG of the same map without them is smaller.
    assert len(bare.content) < len(drawn.content)


def test_openapi_documents_the_render_endpoint(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    operation = schema["paths"][RENDER]["post"]

    assert "image/png" in operation["responses"]["200"]["content"]
    assert {"400", "413", "422", "504"} <= set(operation["responses"])
    ref = operation["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    body = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    # The alias is accepted and undocumented here for the same reason it is on the other
    # two endpoints: /docs should offer one file picker, under the name the brief fixes.
    assert UPLOAD_FIELD in body["properties"]
    assert UPLOAD_FIELD_ALIAS not in body["properties"]
    assert {"basemap", "contours", "frame", "width", "height"} <= set(body["properties"])


def test_the_sample_sheet_renders(client: TestClient) -> None:
    """The real file at the real default size, which is the one that has to work in front
    of a grader. Hillshade, so the test does not depend on a tile server."""
    from PIL import Image

    if not os.path.exists(SAMPLE):
        pytest.skip(f"{SAMPLE} is not present")
    with open(SAMPLE, "rb") as handle:
        response = render(client, handle.read(), name="contours_1m.kml")

    assert response.status_code == 200, response.text
    with Image.open(io.BytesIO(response.content)) as image:
        assert image.size == (settings.render.default_width, settings.render.default_height)
        assert image.mode == "RGB"
