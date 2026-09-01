"""Phase 8. GeoJSON export.

The acceptance criterion for this phase is visual: the output has to land on the right
piece of ground in geojson.io. Nothing here can look at a map, so the tests check the
things that would make it land in the wrong place or draw the wrong shape. The area the
ring encloses, the winding RFC 7946 requires, holes attached to the ring that contains
them, and coordinates that survive a round trip back through the projection to the cells
they came from.

Shapes with known geometry do most of the work: a square, a donut, two squares that touch
at a corner. What the ring should be is arithmetic on those, not judgement.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from app.config import GeoJSONConfig, settings
from app.core.dem_builder import ContourSurface
from app.core.geojson import (
    GeoJSONError,
    build_geojson,
    catchment_feature,
    contour_drawing,
    feature_collection,
    flow_path_geometry,
    mask_geometry,
    outlet_feature,
    pond_feature,
    simplify,
    site_features,
    trace_rings,
)
from app.core.hydrology import water_balance
from app.core.kml_parser import parse_contour_file
from app.core.pond_siting import PondSiteSelector
from app.core.projection import projection_for
from app.core.terrain import analyse_terrain
from app.providers.rainfall import DefaultRainfallProvider
from tests.test_terrain import plane_east, synthetic_dem

SAMPLE = "data/contours_1m.kml"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ring_area_m2(ring, dem) -> float:
    """Area a lon/lat ring encloses, measured back in projected metres."""
    xy = dem.projection.forward(np.asarray(ring, dtype=np.float64))
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])))


def signed_area(ring) -> float:
    """Twice the signed area in lon/lat: positive is counter-clockwise."""
    return sum(
        x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:])
    )


def drawn_area_m2(geometry, dem) -> float:
    """Area of a Polygon or MultiPolygon, holes subtracted."""
    return sum(
        ring_area_m2(polygon[0], dem)
        - sum(ring_area_m2(hole, dem) for hole in polygon[1:])
        for polygon in polygons_of(geometry)
    )


def all_rings(geometry):
    for polygon in polygons_of(geometry):
        yield from polygon


def polygons_of(geometry) -> list:
    return (
        [geometry["coordinates"]]
        if geometry["type"] == "Polygon"
        else geometry["coordinates"]
    )


def block_mask(shape, rows, cols) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[rows, cols] = True
    return mask


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #
def test_a_square_traces_to_one_ring_of_its_own_corners():
    """Four cells: the ring is the eight corners of the square, and nothing else."""
    rings = trace_rings(block_mask((10, 10), slice(3, 5), slice(3, 5)))
    assert len(rings.exteriors) == 1 and not rings.holes
    ring = rings.exteriors[0]
    assert ring[0] == ring[-1]
    assert set(ring) == {(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4), (5, 5)}


def test_an_exterior_winds_counter_clockwise_and_a_hole_the_other_way():
    """RFC 7946's right-hand rule, and the cheapest way to tell the two apart."""
    donut = block_mask((12, 12), slice(2, 10), slice(2, 10))
    donut[4:8, 4:8] = False
    rings = trace_rings(donut)

    assert len(rings.exteriors) == 1 and len(rings.holes) == 1
    from app.core.geojson import _ring_area

    assert _ring_area(rings.exteriors[0]) > 0
    assert _ring_area(rings.holes[0]) < 0


def test_two_components_trace_as_two_rings():
    mask = block_mask((12, 12), slice(1, 4), slice(1, 4))
    mask[7:10, 7:10] = True
    rings = trace_rings(mask)
    assert len(rings.exteriors) == 2 and not rings.holes


def test_cells_touching_at_a_corner_stay_two_rings():
    """The pinch case. Joining them at the shared corner would make a figure-eight, which
    is not a valid polygon and which renderers fill wrongly."""
    mask = block_mask((8, 8), 3, 3)
    mask[4, 4] = True
    rings = trace_rings(mask)
    assert len(rings.exteriors) == 2
    assert all(len(ring) == 5 for ring in rings.exteriors)


def test_an_empty_mask_is_an_error():
    with pytest.raises(GeoJSONError) as raised:
        trace_rings(np.zeros((5, 5), dtype=bool))
    assert raised.value.code == "empty_mask"


# --------------------------------------------------------------------------- #
# Simplification
# --------------------------------------------------------------------------- #
def test_simplification_keeps_the_corners_of_a_staircase():
    """A diagonal staircase is a straight line at any tolerance above half a step."""
    staircase = [(float(i // 2), float((i + 1) // 2)) for i in range(21)]
    assert simplify(staircase, 2.0) == [staircase[0], staircase[-1]]
    assert len(simplify(staircase, 0.0)) == len(staircase)


def test_simplification_below_one_cell_preserves_area(dem, sites):
    """The tolerance is under a cell on purpose: a vertex cannot move into a neighbouring
    cell, so the enclosed area survives (PLAN Phase 8)."""
    mask = sites[0].catchment.mask
    geometry, _ = mask_geometry(mask, dem)
    traced = sum(len(r) for r in trace_rings(mask).exteriors)
    drawn = sum(len(r) for polygon in polygons_of(geometry) for r in polygon)

    assert drawn < traced / 4, "simplification should actually remove vertices"
    assert drawn_area_m2(geometry, dem) == pytest.approx(dem.area_of(mask), rel=0.01)


def test_a_pond_too_small_to_simplify_keeps_its_traced_shape():
    """Four cells cannot survive Douglas-Peucker at three quarters of a cell. Dropping the
    geometry would lose the thing a village most wants to see, so the staircase is kept."""
    dem = plane_east(ny=12, nx=12, res=5.0)
    geometry, _ = mask_geometry(block_mask(dem.shape, slice(5, 7), slice(5, 7)), dem)
    ring = geometry["coordinates"][0]
    assert len(ring) == 5
    assert ring_area_m2(ring, dem) == pytest.approx(100.0, rel=0.02)


def test_the_vertex_budget_is_enforced_and_reported(dem, sites):
    tight = replace(settings.geojson, max_polygon_vertices=60)
    geometry, warnings = mask_geometry(sites[0].catchment.mask, dem, config=tight)

    drawn = sum(len(r) for polygon in polygons_of(geometry) for r in polygon)
    assert drawn <= 60
    assert warnings and "stay under 60 vertices" in warnings[0]


def test_a_budget_that_cannot_be_met_is_reported_rather_than_chased(dem, flow):
    """A stream network is dozens of separate pieces, and a ring needs four vertices
    however hard it is thinned. The export says so; it does not double the tolerance
    forever looking for a number it cannot reach."""
    mask = flow.dem.valid & (flow.accumulation > 500)
    geometry, warnings = mask_geometry(
        mask, dem, config=replace(settings.geojson, max_polygon_vertices=60)
    )
    assert geometry["type"] == "MultiPolygon"
    assert any("above the 60" in w for w in warnings)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_a_square_mask_becomes_a_square_polygon_in_the_right_place():
    """Corner coordinates, not cell centres: the ring must run half a cell outside the
    cells it encloses, or every drawn shape is a cell too small."""
    dem = plane_east(ny=12, nx=12, res=10.0)
    mask = block_mask(dem.shape, slice(4, 8), slice(4, 8))
    geometry, _ = mask_geometry(mask, dem)
    ring = geometry["coordinates"][0]

    assert geometry["type"] == "Polygon"
    assert ring_area_m2(ring, dem) == pytest.approx(dem.area_of(mask), rel=0.01)
    lon, lat = dem.lonlat_of(4, 4)
    assert min(p[0] for p in ring) < lon and min(p[1] for p in ring) < lat


def test_a_hole_is_attached_to_the_ring_that_contains_it():
    dem = plane_east(ny=20, nx=20, res=10.0)
    mask = block_mask(dem.shape, slice(4, 16), slice(4, 16))
    mask[8:12, 8:12] = False
    geometry, _ = mask_geometry(mask, dem)

    assert len(geometry["coordinates"]) == 2
    outer, inner = geometry["coordinates"]
    assert signed_area(outer) > 0 > signed_area(inner)
    assert ring_area_m2(outer, dem) - ring_area_m2(inner, dem) == pytest.approx(
        dem.area_of(mask), rel=0.01
    )


def test_disjoint_components_become_a_multipolygon():
    dem = plane_east(ny=20, nx=20, res=10.0)
    mask = block_mask(dem.shape, slice(2, 6), slice(2, 6))
    mask[12:18, 12:18] = True
    geometry, _ = mask_geometry(mask, dem)

    assert geometry["type"] == "MultiPolygon"
    assert len(geometry["coordinates"]) == 2
    assert drawn_area_m2(geometry, dem) == pytest.approx(dem.area_of(mask), rel=0.01)


def test_coordinates_are_rounded_to_the_configured_precision():
    dem = plane_east(ny=12, nx=12, res=10.0)
    geometry, _ = mask_geometry(block_mask(dem.shape, slice(4, 8), slice(4, 8)), dem)
    for lon, lat in geometry["coordinates"][0]:
        assert lon == round(lon, settings.geojson.coordinate_precision)
        assert lat == round(lat, settings.geojson.coordinate_precision)


# --------------------------------------------------------------------------- #
# On the real sheet
# --------------------------------------------------------------------------- #
def test_the_catchment_polygon_encloses_the_area_that_was_reported(sites, dem):
    """The number in the response and the shape on the map have to be the same catchment.
    Anything else is a report that cannot be checked against its own picture."""
    for site in sites:
        geometry, _ = mask_geometry(site.catchment.mask, dem)
        assert drawn_area_m2(geometry, dem) == pytest.approx(
            site.catchment.area_m2, rel=0.01
        )


def test_a_diagonally_attached_cell_becomes_its_own_polygon(sites, dem):
    """D8 routes diagonally, so a catchment can hang off a corner. Splitting the pinch is
    what keeps the geometry valid; the cost is a sliver polygon, and the area is unchanged
    either way."""
    geometry, _ = mask_geometry(sites[0].catchment.mask, dem)
    assert geometry["type"] == "MultiPolygon"
    assert drawn_area_m2(geometry, dem) == pytest.approx(
        sites[0].catchment.area_m2, rel=0.01
    )


def test_every_exterior_ring_obeys_the_right_hand_rule(sites, dem):
    for site in sites:
        geometry, _ = mask_geometry(site.catchment.mask, dem)
        for polygon in polygons_of(geometry):
            assert signed_area(polygon[0]) > 0
            for hole in polygon[1:]:
                assert signed_area(hole) < 0


def test_every_ring_is_closed_and_has_four_positions(collection):
    for feature in collection["features"]:
        geometry = feature["geometry"]
        if geometry["type"] not in {"Polygon", "MultiPolygon"}:
            continue
        for polygon in polygons_of(geometry):
            for ring in polygon:
                assert len(ring) >= 4
                assert ring[0] == ring[-1]


def test_drawn_coordinates_land_back_on_the_cells_they_came_from(sites, dem):
    """A projection used one way in the analysis and another in the export puts a correct
    catchment on the wrong hillside. Every vertex has to come back inside the mask, or
    within a cell of it."""
    site = sites[0]
    geometry, _ = mask_geometry(site.catchment.mask, dem)
    vertices = [point for ring in all_rings(geometry) for point in ring]
    xy = dem.projection.forward(np.asarray(vertices, dtype=np.float64))

    for x, y in xy.tolist():
        row, col = dem.index_of(float(x), float(y))
        window = site.catchment.mask[
            max(0, row - 1) : row + 2, max(0, col - 1) : col + 2
        ]
        assert window.any(), f"({x:.1f}, {y:.1f}) is not on the catchment"


def test_the_flow_path_runs_from_the_far_end_of_the_basin_to_the_outlet(sites, flow, dem):
    site = sites[0]
    line = flow_path_geometry(flow, site.catchment, dem)
    coordinates = line["coordinates"]

    assert line["type"] == "LineString"
    assert coordinates[-1] == [
        round(v, settings.geojson.coordinate_precision) for v in site.lonlat
    ]
    start_row, start_col = np.divmod(site.catchment.flow_path_cell, dem.shape[1])
    assert coordinates[0] == [
        round(float(v), settings.geojson.coordinate_precision)
        for v in dem.lonlat_of(int(start_row), int(start_col))
    ]

    walked = sum(
        math.dist(
            dem.projection.forward(np.asarray(a)), dem.projection.forward(np.asarray(b))
        )
        for a, b in zip(coordinates, coordinates[1:])
    )
    assert walked == pytest.approx(site.catchment.longest_flow_path_m, rel=0.02)


# --------------------------------------------------------------------------- #
# Features and the collection
# --------------------------------------------------------------------------- #
def test_the_catchment_feature_carries_the_numbers_that_justify_it(sites, dem):
    feature = catchment_feature(sites[0], dem)
    properties = feature["properties"]
    assert properties["role"] == "catchment"
    assert properties["area_ha"] == pytest.approx(sites[0].catchment.area_ha, abs=0.01)
    assert properties["edge_contact_pct"] == pytest.approx(
        sites[0].catchment.edge_contact * 100, abs=0.01
    )
    assert properties["fill"].startswith("#")


def test_the_outlet_feature_carries_the_water_balance(sites, balances):
    feature = outlet_feature(sites[0], balances[0])
    properties = feature["properties"]
    assert feature["geometry"]["type"] == "Point"
    assert properties["role"] == "pond_site"
    assert properties["rank"] == 1 and properties["recommended"]
    assert properties["annual_runoff_m3"] == round(balances[0].annual_runoff_m3)
    assert properties["capacity_m3"] == round(balances[0].storage.usable_capacity_m3)


def test_the_pond_feature_is_the_water_surface_that_was_costed(sites, balances, dem):
    feature = pond_feature(balances[0].storage, dem)
    assert drawn_area_m2(feature["geometry"], dem) == pytest.approx(
        balances[0].storage.usable_area_m2, rel=0.05
    )
    assert feature["properties"]["capacity_m3"] == round(
        balances[0].storage.usable_capacity_m3
    )


def test_a_pond_of_one_cell_is_not_drawn(sites, balances, dem):
    """A 25 m^2 square on a satellite image says less than the numbers already do."""
    storage = replace(
        balances[0].storage, pond_mask=np.zeros(dem.shape, dtype=bool)
    )
    assert pond_feature(storage, dem) is None


def test_alternates_get_a_pond_and_a_point_but_no_flow_path(sites, flow, balances):
    """The flow path is the costed site's alone; the pond is every site's, because the
    question asked of an alternate is where its water would sit."""
    detailed = site_features(flow, sites[0], balances[0], detailed=True)
    outline = site_features(flow, sites[1], balances[1], detailed=False)

    assert [f["properties"]["role"] for f in outline] == [
        "catchment", "pond", "pond_site"
    ]
    assert "flow_path" in [f["properties"]["role"] for f in detailed]
    assert "pond" in [f["properties"]["role"] for f in detailed]


def test_an_alternates_pond_is_drawn_paler_than_the_recommended_one(dem, balances):
    """simplestyle is all geojson.io has to separate the pick from the offers."""
    chosen = pond_feature(balances[0].storage, dem, rank=1)["properties"]
    other = pond_feature(balances[1].storage, dem, rank=2)["properties"]

    assert chosen["fill"] != other["fill"]
    assert other["fill-opacity"] < chosen["fill-opacity"]


def test_the_collection_draws_areas_first_and_points_last(collection):
    """Draw order is the only layering GeoJSON has, and geojson.io honours it."""
    kinds = [f["geometry"]["type"] for f in collection["features"]]
    rank = {"Polygon": 0, "MultiPolygon": 0, "LineString": 1, "Point": 2}
    assert [rank[k] for k in kinds] == sorted(rank[k] for k in kinds)


def test_the_collection_covers_every_site_and_bounds_them(collection, sites):
    roles = [f["properties"]["role"] for f in collection["features"]]
    assert roles.count("catchment") == len(sites)
    assert roles.count("pond_site") == len(sites)
    assert roles.count("pond") == len(sites)
    assert roles.count("flow_path") == 1

    min_lon, min_lat, max_lon, max_lat = collection["bbox"]
    for feature in collection["features"]:
        if feature["geometry"]["type"] == "Point":
            lon, lat = feature["geometry"]["coordinates"]
            assert min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def test_the_collection_survives_a_json_round_trip(collection):
    """Every value crosses a JSON boundary next. A numpy float32 does not survive that,
    and the failure would land in the API layer rather than here."""
    blob = json.dumps(collection)
    assert json.loads(blob) == collection
    assert len(blob) < 1_500_000


def test_building_without_a_water_balance_still_draws_the_terrain(flow, sites):
    """The hydrology is optional: a caller that asked only for the catchment gets the
    catchment, not an exception."""
    collection = build_geojson(flow, sites[:2])
    roles = [f["properties"]["role"] for f in collection["features"]]
    assert "catchment" in roles and "pond" not in roles
    assert json.dumps(collection)


def test_drawing_nothing_is_an_error(flow):
    with pytest.raises(GeoJSONError) as raised:
        build_geojson(flow, [])
    assert raised.value.code == "no_sites"


# --------------------------------------------------------------------------- #
# Contours
#
# These are the input drawn back out, not the answer, and the only claim being made about
# them is that they are the same lines. So the tests are about fidelity: every line kept,
# every line still carrying its own height, and no vertex moved further than the tolerance
# the response admits to.
# --------------------------------------------------------------------------- #
def test_every_contour_in_the_file_comes_back_as_a_feature(sample_contours):
    drawing = contour_drawing(sample_contours)
    features = drawing.geojson["features"]
    assert len(features) == sample_contours.line_count
    assert {f["geometry"]["type"] for f in features} == {"LineString"}
    assert all(len(f["geometry"]["coordinates"]) >= 2 for f in features)
    assert json.dumps(drawing.geojson)


def test_each_line_keeps_its_own_height(sample_contours):
    """Positional, and worth stating: a line thinned out of the drawing would otherwise
    shift every height after it onto the wrong line."""
    drawing = contour_drawing(sample_contours)
    drawn = [f["properties"]["elevation_m"] for f in drawing.geojson["features"]]
    assert drawn == [round(float(z), 2) for z in sample_contours.line_elevations]


def test_no_ground_moves_further_than_the_tolerance_claimed(sample_contours):
    """The whole claim the response makes about the drawing: a line on the map is the
    line in the file, to within `simplify_tolerance_m`. Checked as the furthest any
    original vertex sits from the polyline drawn in its place, which is the quantity
    Douglas-Peucker bounds and the one a reader comparing a boundary against a ridge
    would notice."""
    tolerance = 2.0
    drawing = contour_drawing(sample_contours, tolerance_m=tolerance)
    projection = projection_for(sample_contours.points)

    worst = 0.0
    for index, feature in enumerate(drawing.geojson["features"][:250]):
        original = projection.forward(sample_contours.line_coords(index))
        drawn = projection.forward(
            np.asarray(feature["geometry"]["coordinates"], dtype=np.float64)
        )
        assert len(drawn) <= len(original)
        worst = max(worst, max(_distance_to_polyline(p, drawn) for p in original))
    # Coordinates are rounded to six decimals on the way out, which is under 0.2 m.
    assert worst <= tolerance + 0.2


def _distance_to_polyline(point, line) -> float:
    """Shortest distance from a point to a polyline, measured to the segments and not
    only to the vertices."""
    start, end = line[:-1], line[1:]
    span = end - start
    length2 = (span * span).sum(axis=1)
    t = np.where(
        length2 > 0.0,
        np.clip(((point - start) * span).sum(axis=1) / np.where(length2 > 0.0, length2, 1.0), 0.0, 1.0),
        0.0,
    )
    nearest = start + t[:, None] * span
    return float(np.hypot(*(point - nearest).T).min())


def test_index_contours_are_the_round_levels_not_every_fifth_line(sample_contours):
    """The sample starts at 267 m. Counting five lines up from there would make 272 the
    heavy one; a reader expects 270, so the count runs off the interval instead."""
    drawing = contour_drawing(sample_contours)
    heavy = {
        f["properties"]["elevation_m"]
        for f in drawing.geojson["features"]
        if f["properties"]["index"]
    }
    assert heavy == {270.0, 275.0, 280.0, 285.0, 290.0, 295.0}
    assert all(z % 5.0 == 0.0 for z in heavy)


def test_colour_carries_the_elevation(sample_contours):
    """Without it a reader sees nested loops and cannot tell a hill from a hollow."""
    drawing = contour_drawing(sample_contours)
    features = drawing.geojson["features"]
    lowest = min(features, key=lambda f: f["properties"]["elevation_m"])
    highest = max(features, key=lambda f: f["properties"]["elevation_m"])
    ramp = settings.geojson.contour_ramp
    assert lowest["properties"]["stroke"] == ramp[0]
    assert highest["properties"]["stroke"] == ramp[2]
    assert len({f["properties"]["stroke"] for f in features}) > 8


def test_a_tight_vertex_budget_thins_further_and_says_so(sample_contours):
    """The budget is a promise about the response size, so it has to be kept even when
    the requested tolerance would blow it. What is not allowed is keeping it quietly."""
    config = replace(settings.geojson, contour_max_vertices=8_000)
    drawing = contour_drawing(sample_contours, tolerance_m=0.5, config=config)
    assert drawing.vertex_count <= 8_000
    assert drawing.tolerance_m > 0.5
    assert drawing.warnings and "thinned" in drawing.warnings[0]


def test_a_finer_tolerance_keeps_more_of_the_file(sample_contours):
    coarse = contour_drawing(sample_contours, tolerance_m=8.0)
    fine = contour_drawing(sample_contours, tolerance_m=0.5)
    assert coarse.vertex_count < fine.vertex_count < sample_contours.vertex_count


def test_asking_for_no_simplification_draws_every_vertex(sample_contours):
    config = replace(settings.geojson, contour_max_vertices=10_000_000)
    drawing = contour_drawing(sample_contours, tolerance_m=0.0, config=config)
    assert drawing.vertex_count == sample_contours.vertex_count
    assert not drawing.warnings


def test_the_drawing_covers_the_ground_the_file_covers(sample_contours):
    drawing = contour_drawing(sample_contours)
    assert drawing.geojson["bbox"] == [
        round(v, settings.geojson.coordinate_precision)
        for v in sample_contours.metadata.bbox
    ]


def test_a_file_with_no_lines_is_an_error(sample_contours):
    empty = replace(
        sample_contours,
        points=np.zeros((0, 2)),
        elevations=np.zeros(0),
        line_starts=np.zeros(1, dtype=np.int64),
    )
    with pytest.raises(GeoJSONError) as raised:
        contour_drawing(empty)
    assert raised.value.code == "no_contours"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample_contours():
    return parse_contour_file(SAMPLE)


@pytest.fixture(scope="module")
def flow():
    return analyse_terrain(ContourSurface(parse_contour_file(SAMPLE)).sample(5.0))


@pytest.fixture(scope="module")
def dem(flow):
    return flow.dem


@pytest.fixture(scope="module")
def sites(flow):
    return PondSiteSelector(flow).select(5).sites


@pytest.fixture(scope="module")
def balances(flow, sites):
    series = DefaultRainfallProvider().daily_series(81.3, 21.25)
    return [water_balance(flow, site.catchment, series) for site in sites]


@pytest.fixture(scope="module")
def collection(flow, sites, balances):
    return build_geojson(flow, sites, balances)
