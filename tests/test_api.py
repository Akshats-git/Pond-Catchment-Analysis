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

import os
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import settings
from app.main import app
from app.routers import analyze as analyze_module
from tests.fixtures import make_variants as variants
from tests.fixtures.make_synthetic import VALLEY

SAMPLE = "data/contours_1m.kml"
ENDPOINT = f"{settings.api.api_prefix}/analyzeContour"
ALIAS = f"{settings.api.api_prefix}/findCatchment"
CONTOURS = f"{settings.api.api_prefix}/contours"

VALLEY_KML = VALLEY.to_kml()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def post(client: TestClient, payload: bytes, *, name: str = "valley.kml", url: str = ENDPOINT, **form):
    """One analysis request. Form values go out as strings, exactly as a browser sends
    them, so the tests exercise the same coercion a real client would."""
    return client.post(
        url,
        files={"file": (name, payload, "application/vnd.google-earth.kml+xml")},
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


def test_missing_file_is_a_structured_422(client: TestClient) -> None:
    """FastAPI rejects this before the route body runs; the handler has to catch it, or
    clients meet a second error format."""
    response = client.post(ENDPOINT, data={"top_n": "1"})
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


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
                    files={"file": ("valley.kml", VALLEY_KML, "text/xml")},
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
        files={"file": ("contours_1m.kml", payload, "application/vnd.google-earth.kml+xml")},
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
