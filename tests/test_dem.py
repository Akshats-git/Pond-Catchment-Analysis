"""Phase 2. Contour-interpolated DEM.

Two halves again. Synthetic surfaces with known answers test the mechanics, because a
plane interpolates to itself exactly and a NaN-aware Gaussian has an invariant that can
be checked rather than eyeballed. The sample sheet then pins the Phase 2 acceptance
numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import DEMConfig
from app.core.dem_builder import (
    ContourSurface,
    DEMBuildError,
    build_dem,
    contour_metrics,
    row_cell_areas,
)
from app.core.kml_parser import parse_contour_file, parse_contours
from app.core.projection import projection_for
from tests.fixtures import make_variants as mv

SAMPLE = "data/contours_1m.kml"


@pytest.fixture(scope="module")
def contours():
    return parse_contour_file(SAMPLE)


@pytest.fixture(scope="module")
def surface(contours):
    return ContourSurface(contours)


@pytest.fixture(scope="module")
def dem(surface):
    return surface.sample()


# --------------------------------------------------------------------------- #
# Resolution is derived from the data, not chosen
# --------------------------------------------------------------------------- #
def test_resolution_follows_from_the_contour_spacing(surface):
    """PLAN §2 Step 2: spacing = mapped area / total length, resolution = spacing / 4."""
    assert surface.mean_spacing_m == pytest.approx(
        surface.hull_area_m2 / surface.total_length_m
    )
    assert surface.auto_resolution_m == pytest.approx(surface.mean_spacing_m / 4.0)
    assert surface.smoothing_sigma_m == pytest.approx(surface.mean_spacing_m / 8.0)


def test_sample_resolution_is_in_the_expected_regime(surface):
    """Not pinned to a literal, which is the point. This is a sanity band around the
    12 to 15 m contour spacing this sheet has."""
    assert 12.0 < surface.mean_spacing_m < 16.0
    assert 3.0 < surface.auto_resolution_m < 4.0


def test_resolution_tracks_the_data_rather_than_the_sheet(contours):
    """Halve the contour density and the derived resolution must coarsen with it.

    This is the anti-hard-coding guarantee of PLAN §9 made testable: drop every second
    contour level and the grid should get coarser, because the data got coarser.
    """
    keep = np.isin(contours.line_elevations, contours.metadata.levels[::2])
    starts, chunks, elevs = [0], [], []
    for i in np.flatnonzero(keep):
        chunks.append(contours.line_coords(i))
        elevs.append(contours.line_elevations[i])
        starts.append(starts[-1] + len(chunks[-1]))

    from app.core.kml_parser import ContourSet

    thinned = ContourSet(
        points=np.concatenate(chunks),
        elevations=np.repeat(elevs, np.diff(starts)),
        line_starts=np.asarray(starts, dtype=np.int64),
        metadata=contours.metadata,
    )
    dense = ContourSurface(contours).auto_resolution_m
    sparse = ContourSurface(thinned).auto_resolution_m
    assert sparse > dense * 1.5


def test_length_is_summed_per_line_not_across_the_flat_array(contours):
    """The point array runs one contour straight into the next. Counting the jumps
    triples the total on this sheet. 1,900 km against 568 km."""
    proj = projection_for(contours.points)
    xy = proj.forward(contours.points)
    per_line, _ = contour_metrics(xy, contours.line_starts)
    naive = float(np.hypot(*np.diff(xy, axis=0).T).sum())
    assert per_line < naive / 3


def test_explicit_resolution_overrides_the_derived_one(surface):
    dem = surface.sample(5.0)
    assert dem.resolution_m == 5.0
    assert dem.meta.resolution_source == "requested"


def test_sigma_is_tied_to_the_contours_not_to_the_grid(surface):
    """The staircase is an artefact of the *contour spacing*, so the smoothing that
    removes it must not change when the user asks for a different grid."""
    assert surface.sample(5.0).meta.smoothing_sigma_m == pytest.approx(
        surface.sample(2.5).meta.smoothing_sigma_m
    )


def test_out_of_range_resolution_is_rejected(surface):
    with pytest.raises(DEMBuildError) as excinfo:
        surface.sample(0.5)
    assert excinfo.value.code == "invalid_resolution"


def test_derived_resolution_is_clamped(contours):
    tight = DEMConfig(min_resolution_m=6.0, max_resolution_m=20.0)
    dem = ContourSurface(contours, config=tight).sample()
    assert dem.resolution_m == 6.0
    assert any("clamped" in w for w in dem.meta.warnings)


def test_memory_ceiling_coarsens_rather_than_failing(contours):
    """A usable coarse answer with a warning beats a 500 on a large sheet."""
    capped = DEMConfig(max_grid_cells=50_000)
    dem = ContourSurface(contours, config=capped).sample()
    assert dem.meta.resolution_source == "coarsened"
    assert dem.z.size <= 50_000
    assert any("coarsened" in w for w in dem.meta.warnings)


def test_the_cell_ceiling_is_a_hard_limit(contours):
    """It has to hold exactly, not approximately: the ceiling exists so a request cannot
    exhaust the 512 MB free tier. Scaling the resolution analytically in one step lands
    just *over* the cap, because the cell count floors the extent before adding the
    fencepost cell."""
    surface = ContourSurface(contours)
    # Caps must be reachable within max_resolution_m; this sheet needs 21,516 cells
    # even at the coarsest allowed 20 m.
    for cap in (25_000, 50_000, 123_457, 400_000):
        dem = ContourSurface(contours, config=DEMConfig(max_grid_cells=cap)).sample()
        ny, nx = dem.shape
        assert ny * nx <= cap
        assert surface.grid_shape(dem.resolution_m) == (ny, nx)


def test_a_sheet_that_cannot_fit_at_any_resolution_is_rejected(contours):
    impossible = DEMConfig(max_grid_cells=100, max_resolution_m=20.0)
    with pytest.raises(DEMBuildError) as excinfo:
        ContourSurface(contours, config=impossible).sample()
    assert excinfo.value.code == "sheet_too_large"


# --------------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------------- #
def test_a_plane_interpolates_to_itself():
    """Linear interpolation over a Delaunay triangulation reproduces a plane exactly.
    If this drifts, the interpolation is not doing what it claims."""
    surface = ContourSurface(parse_contours(mv.z_coordinate()))
    dem = surface.sample(5.0, smooth=False)
    lonlat = dem.lonlat_of(*np.indices(dem.shape))
    valid = dem.valid
    # mv's rings are nested squares whose level rises inward: not a plane, but every
    # interpolated value must still be a convex combination of the contour levels.
    assert dem.z[valid].min() >= min(mv.LEVELS) - 1e-9
    assert dem.z[valid].max() <= max(mv.LEVELS) + 1e-9
    assert lonlat.shape == (*dem.shape, 2)


def test_interpolation_stays_within_the_contour_levels(dem, contours):
    """A real invariant, not a tolerance: linear interpolation inside the hull is a
    convex combination of the vertex elevations, so it cannot overshoot. This is exactly
    what the mean-fill smoothing bug violated, by 59 m."""
    lo, hi = contours.metadata.elevation_range
    assert dem.raw_z[dem.valid].min() >= lo - 1e-9
    assert dem.raw_z[dem.valid].max() <= hi + 1e-9


def test_cells_outside_the_hull_are_nodata(dem):
    assert dem.nodata.any()
    assert np.isnan(dem.z[dem.nodata]).all()
    assert np.array_equal(dem.nodata, ~np.isfinite(dem.z))


# --------------------------------------------------------------------------- #
# Smoothing. PLAN §11.2, the 357 m peak
# --------------------------------------------------------------------------- #
def test_smoothing_cannot_leave_the_input_range(dem, contours):
    """The normalised Gaussian is a weighted average of valid neighbours, so this holds
    by construction. The bug it guards against produced a 357 m peak on a 298 m map."""
    lo, hi = contours.metadata.elevation_range
    assert dem.z[dem.valid].min() >= lo - 1e-9
    assert dem.z[dem.valid].max() <= hi + 1e-9


def test_mean_fill_smoothing_would_inflate_the_edges(dem):
    """Reproduce the bug to show the guard is load-bearing.

    Filling no-data with the *mean* leaves a phantom contribution in the numerator that
    the denominator does not account for, and cells near the data edge blow past the
    contour range. Filling with zero cancels exactly.
    """
    from scipy.ndimage import gaussian_filter

    sigma = 4.0  # exaggerated so the effect is unmistakable
    raw, nodata = dem.raw_z, dem.nodata
    valid = (~nodata).astype(float)
    mean = np.nanmean(raw)

    filled = np.where(nodata, mean, raw)
    bad = gaussian_filter(filled, sigma) / np.maximum(
        gaussian_filter(valid, sigma), 1e-6
    )
    good = gaussian_filter(np.where(nodata, 0.0, raw), sigma) / np.maximum(
        gaussian_filter(valid, sigma), 1e-6
    )

    hi = raw[~nodata].max()
    assert bad[~nodata].max() > hi + 10.0     # the bug: tens of metres of overshoot
    assert good[~nodata].max() <= hi + 1e-6   # the fix: none at all


def test_smoothing_preserves_the_data_footprint(dem):
    """The division produces finite values just outside the hull. Keeping them would
    invent terrain and quietly enlarge the mapped area every result is measured against."""
    unsmoothed = ContourSurface(
        parse_contour_file(SAMPLE)
    ).sample(dem.resolution_m, smooth=False)
    assert np.array_equal(dem.nodata, unsmoothed.nodata)


def test_smoothing_moves_the_surface_by_less_than_a_contour_interval(dem):
    """PLAN Phase 2 acceptance, on the 99.9th percentile.

    The raw *maximum* is 1.16 m, marginally over the 1 m interval, at 13 cells of 622,227.
    All 13 sit on the convex hull where Delaunay bridges contour 273 to contour 283 with
    a sliver triangle, so the raw surface steps 9.5 m between adjacent cells; averaging
    across that step is the smoothing working, not failing.
    """
    m = dem.meta
    assert m.smoothing_shift_p999_m < m.contour_interval_m
    assert m.cells_over_interval < dem.z.size * 1e-4


def test_disabling_smoothing_is_a_no_op_on_the_surface(surface):
    """Phase 5 has to run the pipeline with smoothing off to show it earns its place."""
    unsmoothed = surface.sample(5.0, smooth=False)
    assert unsmoothed.meta.smoothing_sigma_m == 0.0
    assert np.array_equal(
        np.nan_to_num(unsmoothed.z), np.nan_to_num(unsmoothed.raw_z)
    )


def _d8_dead_end_fraction(z: np.ndarray, valid: np.ndarray) -> float:
    """Fraction of cells with no strictly lower neighbour, which is where D8 has nowhere
    to send its water. This, and not visual smoothness, is what the stair steps cost."""
    ny, nx = z.shape
    core, centre = valid[1:-1, 1:-1], z[1:-1, 1:-1]
    has_lower = np.zeros_like(core)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            rows, cols = slice(1 + dr, ny - 1 + dr), slice(1 + dc, nx - 1 + dc)
            has_lower |= valid[rows, cols] & (z[rows, cols] < centre)
    return float((core & ~has_lower).sum() / core.sum())


def test_smoothing_halves_the_cells_where_d8_has_nowhere_to_go(surface):
    """The reason the step exists at all (PLAN §2 Step 3).

    Linear interpolation between two contour lines makes the ground between them a *flat
    band*, and on a flat band every cell is exactly level with its neighbours, so D8 has
    no downhill direction to pick. In the analytic validation this cost up to 12.79% of
    the catchment: hillslope water ran along a step and joined the stream below the
    outlet.

    The fix does not need to reshape the terrain. It only needs to break the exact
    ties, which is why a sigma of a third of a cell is enough. Measured on the sample:
    13.91% of cells are D8 dead ends before smoothing, 7.07% after. A 49% reduction.
    The remainder is what Phase 3's pit filling is for.
    """
    dem = surface.sample(5.0)
    before = _d8_dead_end_fraction(dem.raw_z, ~dem.nodata)
    after = _d8_dead_end_fraction(dem.z, ~dem.nodata)
    assert before > 0.10
    assert after < before * 0.55


# --------------------------------------------------------------------------- #
# Grid geometry. PLAN §11.8
# --------------------------------------------------------------------------- #
def test_index_round_trip_hits_the_same_cell(dem):
    """`int(round((v - origin) / res))`, with the division inside `round`. Getting this
    wrong returns a neighbouring cell, and a perfectly plausible wrong catchment."""
    rng = np.random.default_rng(1)
    ny, nx = dem.shape
    for _ in range(500):
        row = int(rng.integers(0, ny))
        col = int(rng.integers(0, nx))
        lon, lat = dem.lonlat_of(row, col)
        assert dem.index_of_lonlat(float(lon), float(lat)) == (row, col)


def test_index_of_snaps_to_the_nearest_centre(dem):
    x, y = dem.xy_of(10, 20)
    third = dem.resolution_m / 3
    assert dem.index_of(float(x) + third, float(y) - third) == (10, 20)


def test_row_zero_is_the_south_edge(dem):
    assert dem.lonlat_of(0, 0)[1] < dem.lonlat_of(dem.shape[0] - 1, 0)[1]


def test_contains_rejects_out_of_bounds(dem):
    ny, nx = dem.shape
    assert dem.contains(0, 0) and dem.contains(ny - 1, nx - 1)
    assert not dem.contains(-1, 0) and not dem.contains(ny, 0)


# --------------------------------------------------------------------------- #
# Area. PLAN §2 Step 6
# --------------------------------------------------------------------------- #
def test_cell_area_is_latitude_weighted(dem):
    """res^2 * cos(lat)/cos(lat0): cells shrink towards the pole."""
    areas = dem.row_cell_areas
    assert areas[0] > areas[-1]
    assert np.allclose(areas, dem.resolution_m ** 2, rtol=1e-3)


def test_area_of_a_mask_sums_the_right_cells(dem):
    mask = np.zeros(dem.shape, dtype=bool)
    mask[5, 5] = mask[5, 6] = True
    assert dem.area_of(mask) == pytest.approx(2 * dem.row_cell_areas[5])


def test_mapped_area_is_the_sum_of_valid_cells(dem):
    assert dem.meta.mapped_area_m2 == pytest.approx(dem.area_of(dem.valid))


def test_mapped_area_agrees_with_the_analytic_hull(dem):
    """Two independent routes to the same number: counting valid cells, and the convex
    hull of the vertices. They must agree, or the grid is not covering the data."""
    assert dem.meta.mapped_area_m2 == pytest.approx(dem.meta.hull_area_m2, rel=1e-3)


def test_row_cell_areas_helper_matches_the_dem(dem):
    np.testing.assert_allclose(
        row_cell_areas(dem.projection, dem.origin_xy, dem.resolution_m, dem.shape[0]),
        dem.row_cell_areas,
    )


# --------------------------------------------------------------------------- #
# The Phase 2 acceptance numbers
# --------------------------------------------------------------------------- #
def test_acceptance_mapped_area(dem):
    """PLAN Phase 2: 8.309 km^2."""
    assert dem.meta.mapped_area_m2 / 1e6 == pytest.approx(8.309, abs=0.005)


def test_acceptance_nodata_fraction(dem):
    """PLAN Phase 2: ~3%. The grid is a rectangle; the contour hull is not."""
    assert 0.01 < dem.meta.nodata_fraction < 0.05


def test_acceptance_elevation_range_is_exact(dem):
    """PLAN Phase 2: exactly 267-298 m, the contour range and nothing beyond it.

    The upper bound is one-sided and needs no tolerance: convexity forbids overshoot, so
    a failure here means the 357 m bug is back. The small undershoot is real and expected
   . Smoothing lowers a summit cell by about a micrometre.
    """
    lo, hi = dem.meta.elevation_range
    assert lo >= 267.0 and hi <= 298.0
    assert (lo, hi) == pytest.approx((267.0, 298.0), abs=1e-4)


def test_acceptance_grid_covers_the_sheet(dem):
    ny, nx = dem.shape
    assert (nx - 1) * dem.resolution_m == pytest.approx(3241.0, abs=dem.resolution_m)
    assert (ny - 1) * dem.resolution_m == pytest.approx(2626.0, abs=dem.resolution_m)


# --------------------------------------------------------------------------- #
# Structural variants, end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "builder", [mv.placemark_name, mv.z_coordinate, mv.polygon, mv.folder_name, mv.kmz]
)
def test_variants_build_a_dem(builder):
    dem = build_dem(parse_contours(builder()), resolution_m=5.0)
    assert dem.valid.any()
    assert dem.meta.mapped_area_m2 > 0


def test_collinear_contours_are_rejected_cleanly():
    """Degenerate geometry must produce a structured error, not a QhullError traceback."""
    document = (
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>flat</name>'
        "<Placemark><name>10</name><LineString><coordinates>"
        "81.0,21.0 81.001,21.0 81.002,21.0</coordinates></LineString></Placemark>"
        "<Placemark><name>20</name><LineString><coordinates>"
        "81.003,21.0 81.004,21.0</coordinates></LineString></Placemark>"
        "</Document></kml>"
    ).encode()
    with pytest.raises(DEMBuildError) as excinfo:
        build_dem(parse_contours(document))
    assert excinfo.value.code == "degenerate_geometry"
