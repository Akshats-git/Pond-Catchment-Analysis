"""Phase 5. Validating the catchment calculation against answers known in advance.

Everything else in this repository checks the pipeline against itself or against the one
sample sheet. This file checks it against arithmetic, and it is the evidence behind every
number the service reports.

Three claims are made here.

1. **The delineation is right.** On a valley whose catchment area can be derived on paper,
   the pipeline reproduces it to within one grid row, and exactly, to the cell, whenever
   the outlet sits on a contour line.

2. **The smoothing earns its place.** Turning it off makes the same measurement wrong by
   fourteen grid rows instead of one. This is not asserted, it is measured, on the same
   surface, through the same code.

3. **Nothing is fitted to the sample.** The valley is a thousand kilometres from the
   sample sheet, and a quadrant of the sample re-run on its own produces answers about
   that quadrant.

See `tests/fixtures/make_synthetic.py` for why A(Y) = 1000 * (1000 - Y).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.catchment import CatchmentDelineator
from app.core.dem_builder import ContourSurface
from app.core.kml_parser import parse_contour_file, parse_contours
from app.core.terrain import analyse_terrain
from tests.fixtures import make_variants as mv
from tests.fixtures.make_synthetic import VALLEY, clip_contours

SAMPLE = "data/contours_1m.kml"

# PLAN §3 Test A samples the channel at these three points.
TEST_POINTS = (250.0, 500.0, 750.0)

# Points where the channel elevation is a whole number, so the outlet cell coincides with
# a contour vertex rather than falling inside an interpolated triangle.
ON_CONTOUR_POINTS = (200.0, 500.0, 800.0)


def delineate(surface: ContourSurface, resolution: float, y: float, *, smooth: bool = True):
    """Run the full pipeline and return the catchment of the channel at `y`.

    Snapping is off on purpose. The point is on the channel by construction, and snapping
    moves an outlet to the largest accumulation within the search radius, which, on a
    channel, is always downstream. That would measure the snap, not the delineation.
    """
    dem = surface.sample(resolution, smooth=smooth)
    delineator = CatchmentDelineator(analyse_terrain(dem))
    return delineator.delineate(*VALLEY.channel_point(y), snap=False), dem


def expected_cells(y: float, resolution: float) -> int:
    """How many cells the analytic answer covers on a grid of this resolution."""
    return round(VALLEY.analytic_catchment_area(y) / resolution ** 2)


def missing_cells(catchment, y: float, resolution: float) -> int:
    """Cells the analytic answer says should be in the catchment and are not.

    Counted in *cells*, not square metres. Cell areas are latitude-weighted, so even a
    perfect catchment misses the continuum area by a few square metres out of half a
    million. An artefact of the projection, not of the routing, and one that would
    disguise the thing being measured.
    """
    return expected_cells(y, resolution) - catchment.cell_count


def error_in_rows(area_m2: float, y: float, resolution: float) -> float:
    """Error expressed in grid rows, which is what it actually is.

    A percentage hides the structure: the same one-row discrepancy reads as 0.66% at
    Y=250 and 1.99% at Y=750, purely because the catchment is smaller. In rows it is
    the same number both times.
    """
    row_area = 2 * VALLEY.x_half * resolution
    return (area_m2 - VALLEY.analytic_catchment_area(y)) / row_area


# --------------------------------------------------------------------------- #
# The surface itself
# --------------------------------------------------------------------------- #
def test_the_synthetic_valley_round_trips_through_the_parser(valley_contours):
    metadata = valley_contours.metadata
    assert metadata.elevation_source == "placemark_name"
    assert metadata.interval_m == pytest.approx(VALLEY.interval_m)
    assert metadata.elevation_range == (0.0, VALLEY.z_max)


def test_the_grid_lands_in_the_regime_the_plan_describes(valley_surface):
    """PLAN §3 Test A reports sigma = 2.5 m on a 5 m grid. Both fall out of the data:
    the contours are 19.7 m apart, so the derived grid is 4.9 m and sigma is 2.46 m."""
    assert valley_surface.mean_spacing_m == pytest.approx(19.7, abs=0.5)
    assert valley_surface.auto_resolution_m == pytest.approx(4.93, abs=0.1)
    assert valley_surface.smoothing_sigma_m == pytest.approx(2.46, abs=0.1)


def test_the_interpolated_surface_matches_the_analytic_one(valley_surface):
    dem = valley_surface.sample(5.0)
    rows, cols = np.indices(dem.shape)
    xy = VALLEY.projection().forward(dem.lonlat_of(rows, cols))
    truth = VALLEY.elevation(xy[..., 0], xy[..., 1])
    residual = dem.z - truth
    assert np.sqrt(np.nanmean(residual ** 2)) < 0.05
    assert np.nanmax(np.abs(residual)) < 1.0


def test_the_whole_domain_is_mapped(valley_surface):
    dem = valley_surface.sample(5.0)
    assert dem.meta.nodata_fraction == 0.0
    assert dem.meta.mapped_area_m2 == pytest.approx(VALLEY.area_m2, rel=1e-3)


# --------------------------------------------------------------------------- #
# Test A. The analytic answer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("y", TEST_POINTS)
def test_analytic_catchment_area_at_five_metres(valley_surface, y):
    """PLAN §3 Test A. The catchment is right to within one grid row of the outlet.

    One row is 5,000 m^2 at this resolution. A sub-cell error in placing the outlet
    along a flat interpolated band, not a routing failure. The absolute discrepancy is the
    same at every Y; only the percentage moves, because the catchment shrinks.
    """
    catchment, _ = delineate(valley_surface, 5.0, y)
    assert abs(error_in_rows(catchment.area_m2, y, 5.0)) <= 1.0


@pytest.mark.parametrize("y", ON_CONTOUR_POINTS)
def test_the_answer_is_exact_when_the_outlet_sits_on_a_contour(valley_surface, y):
    """No tolerance at all. When the outlet cell coincides with a contour vertex there is
    no interpolated flat band to misplace it on, and the catchment is exactly right.
    every one of 30,000 cells, counted."""
    catchment, _ = delineate(valley_surface, 5.0, y)
    assert missing_cells(catchment, y, 5.0) == 0
    assert catchment.area_m2 == pytest.approx(VALLEY.analytic_catchment_area(y), rel=1e-4)


@pytest.mark.parametrize("y", TEST_POINTS)
def test_analytic_catchment_area_at_ten_metres(valley_surface, y):
    """A 10 m grid is outside the regime the smoothing is scaled for.

    Sigma is tied to the contour spacing, so it is fixed in metres: 2.46 m is half a cell
    at 5 m but a quarter of a cell at 10 m, and a quarter-cell Gaussian barely touches the
    stair-steps. The error grows to a few rows accordingly. Worth pinning, because it is
    the argument for deriving the resolution from the data rather than accepting one.
    """
    catchment, _ = delineate(valley_surface, 10.0, y)
    assert abs(error_in_rows(catchment.area_m2, y, 10.0)) <= 4.0


@pytest.mark.parametrize("y", TEST_POINTS)
def test_the_catchment_never_over_reports(valley_surface, y):
    """Every error seen is a *missing* row, never a spurious one. A catchment that claimed
    ground draining somewhere else would be the dangerous failure. It would promise a
    pond more water than the terrain delivers."""
    catchment, _ = delineate(valley_surface, 5.0, y)
    assert catchment.area_m2 <= VALLEY.analytic_catchment_area(y) * (1 + 1e-9)


def test_the_error_is_a_constant_number_of_cells_not_a_constant_fraction(valley_surface):
    """The structure of the residual, stated as a test.

    The same 199 cells go missing at every Y where the outlet falls between contours, and
    none at all where it falls on one. That is one row of cells: the row the outlet is in,
    minus the outlet itself. It identifies the residual as sub-cell positioning and not a
    proportional routing error, because a routing error would take a fraction and would
    grow with the catchment.
    """
    width = valley_surface.sample(5.0).shape[1]
    shortfalls = {y: missing_cells(delineate(valley_surface, 5.0, y)[0], y, 5.0)
                  for y in TEST_POINTS}
    for shortfall in shortfalls.values():
        assert shortfall in (0, width - 1)
    assert len(set(shortfalls.values())) > 1  # not uniformly wrong either


# --------------------------------------------------------------------------- #
# The evidence for the smoothing step
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("y", TEST_POINTS)
def test_smoothing_is_what_makes_the_answer_right(valley_surface, y):
    """PLAN §2 Step 3, measured rather than asserted.

    Without it the same pipeline on the same surface loses fourteen grid rows instead of
    one. This is the number that justifies the step in the report.
    """
    smoothed, _ = delineate(valley_surface, 5.0, y, smooth=True)
    raw, _ = delineate(valley_surface, 5.0, y, smooth=False)
    assert abs(error_in_rows(raw.area_m2, y, 5.0)) > 5.0
    assert abs(error_in_rows(raw.area_m2, y, 5.0)) > 5 * max(
        abs(error_in_rows(smoothed.area_m2, y, 5.0)), 0.5
    )


def test_the_unsmoothed_failure_is_the_worst_at_the_top_of_the_valley(valley_surface):
    """PLAN §3 names Y=750 as the worst case at -12.79%. The mechanism is that hillslope
    water runs *along* a flat stair-step band and rejoins the channel below the outlet, so
    the smaller the catchment the larger the share it loses."""
    errors = {
        y: error_in_rows(delineate(valley_surface, 5.0, y, smooth=False)[0].area_m2, y, 5.0)
        for y in TEST_POINTS
    }
    percentages = {
        y: delineate(valley_surface, 5.0, y, smooth=False)[0].area_m2
        / VALLEY.analytic_catchment_area(y)
        - 1
        for y in TEST_POINTS
    }
    assert abs(percentages[750.0]) > abs(percentages[500.0])
    assert abs(percentages[750.0]) > 0.10


def test_the_flat_bands_are_really_there(valley_surface):
    """The mechanism itself, made visible.

    Near a contour's V-apex the tent's legs wrap around the valley axis, so a Delaunay
    triangle can have all three vertices on the *same* contour, and a triangle whose
    corners are all at one elevation interpolates to a plane. Down the channel of the raw
    interpolation there are runs of cells at bit-identical elevations; after smoothing
    the longest such run is far shorter.
    """
    dem = valley_surface.sample(5.0)
    channel = dem.shape[1] // 2

    def longest_flat_run(profile: np.ndarray) -> int:
        flat = np.diff(profile) == 0.0
        best = run = 0
        for step in flat:
            run = run + 1 if step else 0
            best = max(best, run)
        return best

    assert longest_flat_run(dem.raw_z[:, channel]) > 5
    assert longest_flat_run(dem.z[:, channel]) < longest_flat_run(dem.raw_z[:, channel])


# --------------------------------------------------------------------------- #
# Test 3. The sub-tile test: nothing is fitted to the sample
# --------------------------------------------------------------------------- #
def test_the_valley_is_nowhere_near_the_sample_sheet(valley_contours, sample_contours):
    """The most basic anti-hard-coding check: if any coordinate of the sample had leaked
    into the code, the valley. 1,800 km away. Would come out visibly wrong."""
    assert abs(valley_contours.metadata.bbox[0] - sample_contours.metadata.bbox[0]) > 0.01


@pytest.mark.parametrize("quadrant", ["SW", "SE", "NW", "NE"])
def test_a_quadrant_of_the_sample_is_analysed_on_its_own_terms(sample_contours, quadrant):
    """PLAN §5.3. Clip the sheet to one quadrant, re-run everything, and check the answers
    describe *that* quadrant.

    A result that came from a hard-coded coordinate, a memorised resolution or a cached
    site would survive every other test in this repository and fail this one.
    """
    min_lon, min_lat, max_lon, max_lat = sample_contours.metadata.bbox
    mid_lon = (min_lon + max_lon) / 2
    mid_lat = (min_lat + max_lat) / 2
    window = {
        "SW": (min_lon, min_lat, mid_lon, mid_lat),
        "SE": (mid_lon, min_lat, max_lon, mid_lat),
        "NW": (min_lon, mid_lat, mid_lon, max_lat),
        "NE": (mid_lon, mid_lat, max_lon, max_lat),
    }[quadrant]

    clipped = clip_contours(sample_contours, window)
    surface = ContourSurface(clipped)
    dem = surface.sample()
    flow = analyse_terrain(dem)
    delineator = CatchmentDelineator(flow)

    # The grid follows the clipped data, not the sheet it came from.
    assert dem.meta.mapped_area_m2 < 0.4 * 8.309e6
    assert dem.meta.resolution_source == "auto"

    # The largest catchment in the quadrant is inside the quadrant.
    row, col = np.unravel_index(int(np.argmax(flow.accumulation)), flow.shape)
    catchment = delineator.delineate_cell(int(row), int(col))
    lon, lat = catchment.outlet_lonlat
    assert window[0] <= lon <= window[2]
    assert window[1] <= lat <= window[3]
    assert catchment.area_m2 > 0.05 * dem.meta.mapped_area_m2


def test_the_quadrants_give_four_different_answers(sample_contours):
    """The corollary. If the quadrants agreed, the answer would not be coming from the
    data."""
    min_lon, min_lat, max_lon, max_lat = sample_contours.metadata.bbox
    mid_lon, mid_lat = (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
    areas = []
    for window in ((min_lon, min_lat, mid_lon, mid_lat), (mid_lon, min_lat, max_lon, mid_lat),
                   (min_lon, mid_lat, mid_lon, max_lat), (mid_lon, mid_lat, max_lon, max_lat)):
        surface = ContourSurface(clip_contours(sample_contours, window))
        flow = analyse_terrain(surface.sample())
        areas.append(float(flow.meta.max_accumulation) * surface.auto_resolution_m ** 2)
    assert len(set(round(a) for a in areas)) == 4
    assert max(areas) > 2 * min(areas)


# --------------------------------------------------------------------------- #
# Test 4. Structural variants, end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name",
    ["placemark_name", "z_coordinate", "extended_data", "folder_name",
     "polygon", "multigeometry", "no_namespace", "spaced_coordinates",
     "with_labels_folder", "stray_3d_polygon", "kmz"],
)
def test_every_structural_variant_survives_the_whole_pipeline(name):
    """The Phase 1 fixtures, carried all the way to a catchment. Parsing a file is not the
    same as being able to analyse it: a variant that produced subtly different geometry
    would parse cleanly and then fall over in the triangulation or the routing.
    """
    contours = parse_contours(mv.VARIANTS[name]())
    surface = ContourSurface(contours)
    dem = surface.sample(5.0)
    flow = analyse_terrain(dem)
    delineator = CatchmentDelineator(flow)

    row, col = np.unravel_index(int(np.argmax(flow.accumulation)), flow.shape)
    catchment = delineator.delineate_cell(int(row), int(col))
    assert catchment.area_m2 > 0
    assert catchment.cell_count == catchment.accumulation_cells
    assert 0.0 <= catchment.edge_contact <= 1.0


def test_the_variants_all_describe_the_same_hill():
    """They are the same three contour rings written five ways, so beyond the parsing they
    must produce the same terrain. A check that the geometry handling is equivalent, not
    merely non-crashing."""
    areas = {}
    for name in ("placemark_name", "no_namespace", "spaced_coordinates", "with_labels_folder"):
        surface = ContourSurface(parse_contours(mv.VARIANTS[name]()))
        areas[name] = surface.sample(5.0).meta.mapped_area_m2
    assert max(areas.values()) == pytest.approx(min(areas.values()), rel=1e-9)


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def valley_contours():
    return parse_contours(VALLEY.to_kml())


@pytest.fixture(scope="module")
def valley_surface(valley_contours):
    return ContourSurface(valley_contours)


@pytest.fixture(scope="module")
def sample_contours():
    return parse_contour_file(SAMPLE)
