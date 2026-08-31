"""Phase 3. Pit filling, D8 flow direction, flow accumulation and slope.

Flow routing is easy to get subtly wrong and hard to eyeball: a map of flow accumulation
looks like a river network whether or not the numbers are right. So most of what follows
runs on surfaces whose answer can be worked out on paper. A tilted plane, a single pit,
a flat plateau, and only then checks the sample sheet.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import TerrainConfig
from app.core.dem_builder import DEM, ContourSurface, DEMMetadata
from app.core.kml_parser import parse_contour_file
from app.core.projection import EquirectangularENU
from app.core.terrain import D8TerrainEngine, analyse_terrain

SAMPLE = "data/contours_1m.kml"
ENGINE = D8TerrainEngine()


# --------------------------------------------------------------------------- #
# Synthetic DEMs
# --------------------------------------------------------------------------- #
def synthetic_dem(z: np.ndarray, resolution_m: float = 10.0, nodata=None) -> DEM:
    """Wrap a bare elevation array as a DEM, bypassing contours entirely.

    Lets a test state a surface directly, so the expected flow can be derived by hand
    rather than inferred from an interpolation.
    """
    z = np.asarray(z, dtype=np.float64)
    if nodata is None:
        nodata = ~np.isfinite(z)
    ny, nx = z.shape
    projection = EquirectangularENU(lon0=81.3, lat0=21.25)
    meta = DEMMetadata(
        resolution_m=resolution_m, resolution_source="requested", smoothing_sigma_m=0.0,
        mean_contour_spacing_m=resolution_m * 4, total_contour_length_m=1.0,
        hull_area_m2=float(z.size) * resolution_m ** 2,
        mapped_area_m2=float((~nodata).sum()) * resolution_m ** 2,
        nodata_fraction=float(nodata.mean()), elevation_range=(float(np.nanmin(z)), float(np.nanmax(z))),
        max_smoothing_shift_m=0.0, smoothing_shift_p999_m=0.0, cells_over_interval=0,
        contour_interval_m=1.0, shape=(ny, nx),
    )
    return DEM(z=np.where(nodata, np.nan, z), nodata=nodata, raw_z=z,
               resolution_m=resolution_m, origin_xy=(0.0, 0.0),
               projection=projection, meta=meta)


def plane_east(ny=12, nx=15, drop=0.05, res=10.0) -> DEM:
    """A plane falling steadily towards +col. Every cell must drain due east."""
    col = np.arange(nx, dtype=np.float64)[None, :]
    return synthetic_dem(np.broadcast_to(100.0 - drop * col * res, (ny, nx)).copy(), res)


# --------------------------------------------------------------------------- #
# D8 direction
# --------------------------------------------------------------------------- #
def test_a_plane_drains_straight_downhill():
    """Not diagonally. The diagonal neighbour of a cell on an east-facing plane sits at
    exactly the same elevation as the eastern one but 41% further away, so the
    distance-weighted slope is shallower and the cardinal step must win."""
    dem = plane_east()
    ny, nx = dem.shape
    receivers = ENGINE.d8_receivers(dem.z, dem.nodata, dem.resolution_m).reshape(ny, nx)
    for row in range(ny):
        for col in range(nx - 1):
            assert receivers[row, col] == row * nx + (col + 1)
        assert receivers[row, nx - 1] == -1  # the downhill edge is the outlet


def test_a_diagonal_plane_drains_diagonally():
    """The converse: when the fall really is along the diagonal, the sqrt(2) weighting
    must not stop the router from taking it."""
    res = 10.0
    rows, cols = np.indices((10, 10))
    dem = synthetic_dem(100.0 - 0.05 * (rows + cols) * res, res)
    receivers = ENGINE.d8_receivers(dem.z, dem.nodata, res).reshape(10, 10)
    assert receivers[3, 3] == 4 * 10 + 4


def test_distance_weighting_changes_the_answer():
    """Make the diagonal drop 1.3x the cardinal one: bigger in absolute terms, but
    1.3 < sqrt(2), so the cardinal neighbour is still steeper and must still win."""
    z = np.array([[10.0, 10.0, 10.0],
                  [10.0, 10.0,  9.0],
                  [10.0, 10.0,  8.7]])
    receivers = ENGINE.d8_receivers(z, np.zeros_like(z, bool), 1.0).reshape(3, 3)
    assert receivers[1, 1] == 1 * 3 + 2       # east, drop 1.0 over 1.0
    z[2, 2] = 8.5                             # now 1.5 / sqrt(2) = 1.06 > 1.0
    receivers = ENGINE.d8_receivers(z, np.zeros_like(z, bool), 1.0).reshape(3, 3)
    assert receivers[1, 1] == 2 * 3 + 2       # south-east wins


def test_a_cell_with_nowhere_to_go_is_an_outlet():
    receivers = ENGINE.d8_receivers(np.zeros((3, 3)), np.zeros((3, 3), bool), 1.0)
    assert (receivers == -1).all()


def test_nodata_cells_have_no_receiver_and_receive_nothing():
    z = np.array([[3.0, 2.0, 1.0]] * 3)
    nodata = np.zeros((3, 3), dtype=bool)
    nodata[1, 1] = True
    receivers = ENGINE.d8_receivers(np.where(nodata, np.nan, z), nodata, 1.0)
    assert receivers[4] == -1
    assert 4 not in receivers.tolist()


def test_receivers_are_always_strictly_lower(flow):
    """The property that makes the flow graph acyclic and the descending-elevation sort a
    valid topological order. Everything downstream rests on it."""
    receivers = flow.receivers
    donors = np.flatnonzero(receivers >= 0)
    z = flow.filled.ravel()
    assert np.all(z[receivers[donors]] < z[donors])


# --------------------------------------------------------------------------- #
# Pit filling
# --------------------------------------------------------------------------- #
def test_a_single_pit_is_raised_to_its_lowest_rim():
    """Not to the level of its uphill neighbour: a pit spills over the *lowest* point of
    its rim, which on an east-facing plane is the cell immediately downhill."""
    dem = plane_east()
    rim = dem.z[5, 8]
    dem.z[5, 7] -= 5.0
    filled = ENGINE.fill_depressions(dem)
    assert filled[5, 7] > dem.z[5, 7]
    assert filled[5, 7] == pytest.approx(rim, abs=1e-3)
    assert filled[5, 7] < dem.z[5, 6]


def test_filling_never_lowers_the_ground():
    dem = plane_east()
    dem.z[4, 4] -= 3.0
    dem.z[8, 9] -= 1.0
    filled = ENGINE.fill_depressions(dem)
    assert np.all(filled[dem.valid] >= dem.z[dem.valid] - 1e-12)


def test_a_bowl_is_filled_to_its_spill_point():
    """A hollow inside a plateau, with a channel cut down to the map edge. The fill must
    stop at the channel, not rise to the plateau top."""
    z = np.full((9, 9), 20.0)
    z[3:6, 3:6] = 10.0
    z[6:9, 4] = 15.0  # a continuous channel from the bowl's rim to the border
    dem = synthetic_dem(z, 1.0)
    filled = ENGINE.fill_depressions(dem)
    assert filled[4, 4] == pytest.approx(15.0, abs=1e-2)
    assert filled[4, 4] < 20.0


def test_an_unreachable_notch_does_not_drain_a_bowl():
    """The converse, because it is the easy mistake: a low cell that the bowl cannot
    reach without first climbing the plateau sets no spill level at all."""
    z = np.full((9, 9), 20.0)
    z[3:6, 3:6] = 10.0
    z[8, 4] = 15.0  # low, on the border, but walled off by the 20 m plateau
    filled = ENGINE.fill_depressions(synthetic_dem(z, 1.0))
    assert filled[4, 4] == pytest.approx(20.0, abs=1e-2)


def test_filling_is_seeded_from_nodata_not_only_from_the_border():
    """PLAN Phase 3: 2.5% of the sample grid is outside the contour hull, so a basin can
    drain off the mapped area through an interior hole without ever touching row 0.

    Here a bowl sits in the middle of a plateau with no route to the border, but a no-data
    hole beside it. Seeded correctly the bowl drains into the hole and is barely touched;
    seeded from the border alone it would be flooded to the plateau top.
    """
    z = np.full((15, 15), 20.0)
    z[6:9, 6:9] = 10.0
    nodata = np.zeros((15, 15), dtype=bool)
    nodata[6:9, 9:12] = True
    dem = synthetic_dem(np.where(nodata, np.nan, z), 1.0, nodata)
    filled = ENGINE.fill_depressions(dem)
    assert filled[7, 7] < 11.0


def test_epsilon_breaks_ties_across_a_flat(flat_plateau):
    """Without the epsilon gradient a flat band has no downhill direction anywhere, and
    D8 turns every cell on it into its own outlet. The failure PLAN §2 Step 4 describes.

    Measured on the sample sheet: epsilon 0 gives 108,526 outlets and a maximum
    accumulation of 712 cells; epsilon 1e-4 gives 308 outlets and 158,262.
    """
    without = D8TerrainEngine(TerrainConfig(fill_epsilon_m=0.0)).analyse(flat_plateau)
    with_eps = D8TerrainEngine(TerrainConfig(fill_epsilon_m=1e-4)).analyse(flat_plateau)
    assert without.meta.outlet_count > with_eps.meta.outlet_count * 5
    assert with_eps.meta.max_accumulation > without.meta.max_accumulation * 5


def test_epsilon_ascent_stays_negligible(real_dem):
    """Priority-flood adds `eps` per cell along a flat run, so in principle a long flat
    could accumulate real height. 1e-4 over the sample's 332,365 cells would be 33 m if
    the whole map were one flat.

    It does not happen, because the flats are short: the largest per-cell ascent on the
    sample sheet is 32 mm, 3% of the contour interval. Worth pinning, because a change
    that lengthens the flats would show up here first.
    """
    off = D8TerrainEngine(TerrainConfig(fill_epsilon_m=0.0)).fill_depressions(real_dem)
    on = D8TerrainEngine(TerrainConfig(fill_epsilon_m=1e-4)).fill_depressions(real_dem)
    valid = real_dem.valid
    ascent = float(np.nanmax(on[valid] - off[valid]))
    assert 0.0 < ascent < real_dem.meta.contour_interval_m * 0.05


# --------------------------------------------------------------------------- #
# Accumulation
# --------------------------------------------------------------------------- #
def test_accumulation_on_a_plane_is_exactly_the_column_index():
    """Each row is an independent chain, so cell (r, c) collects the c cells west of it
    plus itself. An exact integer answer, checked exactly."""
    dem = plane_east(ny=12, nx=15)
    flow = ENGINE.analyse(dem)
    expected = np.broadcast_to(np.arange(1, 16, dtype=np.float64), (12, 15))
    np.testing.assert_array_equal(flow.accumulation, expected)


def test_every_cell_counts_itself():
    flow = ENGINE.analyse(plane_east())
    assert flow.accumulation.min() == 1.0


def test_total_accumulation_leaving_the_map_equals_the_cell_count(flow, real_dem):
    """Nothing is created or destroyed: what the outlets discharge is what fell on the
    map."""
    outlets = flow.receivers.reshape(flow.shape) < 0
    assert flow.accumulation[outlets & real_dem.valid].sum() == pytest.approx(
        real_dem.valid.sum()
    )


def test_weights_generalise_the_traversal():
    """Phase 7 may want to accumulate rainfall rather than cells; the traversal should not
    need to change. Doubling every weight must double every total."""
    dem = plane_east()
    flow = ENGINE.analyse(dem)
    doubled = ENGINE.flow_accumulation(
        flow.receivers, flow.order, dem.shape, np.full(dem.shape, 2.0)
    )
    np.testing.assert_allclose(doubled, flow.accumulation * 2)


def test_topological_order_processes_donors_before_receivers(flow):
    position = np.empty(flow.receivers.size, dtype=np.int64)
    position[flow.order] = np.arange(flow.order.size)
    donors = np.flatnonzero(flow.receivers >= 0)
    assert np.all(position[donors] < position[flow.receivers[donors]])


# --------------------------------------------------------------------------- #
# Mass balance. PLAN §3 Test B
# --------------------------------------------------------------------------- #
def test_every_cell_reaches_exactly_one_outlet(flow, real_dem):
    terminal = flow.terminal_outlets()
    assert np.all(terminal[real_dem.valid] >= 0)
    assert (flow.receivers.reshape(flow.shape)[real_dem.valid] < 0).sum() == len(
        np.unique(terminal[real_dem.valid])
    )


def test_basin_areas_sum_to_the_mapped_area(flow, real_dem):
    """PLAN §3 Test B. Every cell drains to exactly one outlet, so the basins tile the
    map and their areas must total the mapped area. Measured difference: 0.00000000%."""
    terminal = flow.terminal_outlets()
    areas = np.broadcast_to(real_dem.row_cell_areas[:, None], real_dem.shape)
    basins = np.bincount(terminal[real_dem.valid], weights=areas[real_dem.valid])
    assert basins.sum() == pytest.approx(real_dem.meta.mapped_area_m2, rel=1e-9)


@pytest.mark.parametrize("resolution", [5.0, 3.5, 2.5])
def test_mass_balance_holds_at_every_ensemble_resolution(surface, resolution):
    dem = surface.sample(resolution)
    flow = ENGINE.analyse(dem)
    terminal = flow.terminal_outlets()
    areas = np.broadcast_to(dem.row_cell_areas[:, None], dem.shape)
    basins = np.bincount(terminal[dem.valid], weights=areas[dem.valid])
    assert basins.sum() == pytest.approx(dem.meta.mapped_area_m2, rel=1e-9)


# --------------------------------------------------------------------------- #
# Slope
# --------------------------------------------------------------------------- #
def test_slope_of_a_known_plane():
    """Horn's estimator is exact on a plane. A 5% grade must read 0.05."""
    dem = plane_east(drop=0.05)
    slope = ENGINE.slope(dem)
    assert slope[3:-3, 3:-3] == pytest.approx(0.05, abs=1e-9)


def test_slope_of_flat_ground_is_zero():
    assert ENGINE.slope(synthetic_dem(np.full((8, 8), 42.0))) == pytest.approx(0.0)


def test_slope_is_direction_agnostic():
    res = 10.0
    rows = np.indices((10, 10))[0].astype(float)
    north = ENGINE.slope(synthetic_dem(rows * 0.05 * res, res))
    east = ENGINE.slope(plane_east(10, 10, 0.05, res))
    assert north[3:-3, 3:-3] == pytest.approx(east[3:-3, 3:-3], abs=1e-9)


def test_slope_is_finite_wherever_there_is_data(flow, real_dem):
    """No-data neighbours are replaced by the centre cell rather than propagating NaN,
    so a cell beside a hole still gets a usable slope instead of being unsiteable."""
    assert np.isfinite(flow.slope[real_dem.valid]).all()
    assert np.isnan(flow.slope[real_dem.nodata]).all()


# --------------------------------------------------------------------------- #
# The sample sheet. Phase 3 acceptance
# --------------------------------------------------------------------------- #
def test_acceptance_max_accumulation(flow):
    """PLAN Phase 3: ~157,766 cells at 5 m. That is 47.6% of the map, which is the same
    basin PLAN §3 reports as site 1 at 395.4 ha. Two independent routes to one number."""
    assert flow.meta.max_accumulation == pytest.approx(157_766, rel=0.02)


def test_acceptance_largest_basin_matches_the_published_site(flow, real_dem):
    area_ha = flow.meta.max_accumulation * real_dem.row_cell_areas.mean() / 1e4
    assert area_ha == pytest.approx(395.4, rel=0.02)


def test_pond_water_levels_are_quantised_to_the_contour_interval(flow, real_dem):
    """A limitation worth pinning rather than hiding (PLAN §10).

    A depression fills until it spills over its lowest rim, and on a contour-derived DEM
    that rim sits on a contour line. So a pool's water level, and with it the natural
    storage Phase 7 reports, can only take values a contour interval apart. On the sample,
    90% of pools of 20 cells or more have a spill elevation within 0.1 m of a whole metre,
    with a median offset of 2.6 mm.

    Note what is *not* quantised: the fill depth of an individual cell, because the cell's
    own elevation is interpolated between contours and takes any value. Only 39% of
    per-cell depths land near a whole metre. The distinction matters for how Phase 7
    reports storage.
    """
    from scipy import ndimage

    raised = np.where(real_dem.valid, flow.filled - real_dem.z, 0.0)
    labels, count = ndimage.label(raised > 0.01)
    pools = np.arange(1, count + 1)
    spill = np.asarray(ndimage.maximum(flow.filled, labels, pools))
    sizes = np.asarray(ndimage.sum(raised > 0.01, labels, pools))

    interval = real_dem.meta.contour_interval_m
    levels = spill[sizes >= 20] / interval
    offset = np.abs(levels - np.round(levels)) * interval
    assert (offset < 0.1).mean() > 0.8
    assert np.median(offset) < 0.05


def test_fill_is_reported_honestly(flow):
    m = flow.meta
    assert m.cells_raised > 0
    assert m.max_fill_m == pytest.approx(12.0, abs=0.05)
    assert m.fill_volume_m3 > 0
    assert m.cells_raised_over_interval > 0


def test_timings_are_reported(flow):
    assert set(flow.meta.timings_ms) == {"fill", "receivers", "accumulation", "slope"}
    assert all(v >= 0 for v in flow.meta.timings_ms.values())


def test_convenience_wrapper_matches_the_engine(real_dem):
    assert analyse_terrain(real_dem).meta.max_accumulation == ENGINE.analyse(
        real_dem
    ).meta.max_accumulation


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def surface():
    return ContourSurface(parse_contour_file(SAMPLE))


@pytest.fixture(scope="module")
def real_dem(surface):
    return surface.sample(5.0)


@pytest.fixture(scope="module")
def flow(real_dem):
    return ENGINE.analyse(real_dem)


@pytest.fixture
def flat_plateau():
    """A wide flat band with a single low outlet. The stair-step problem in miniature."""
    z = np.full((30, 30), 50.0)
    z[29, 15] = 49.0
    return synthetic_dem(z, 5.0)
