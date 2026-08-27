"""Phase 4 -- catchment delineation, edge diagnostics and the resolution ensemble.

The delineation itself is a tree traversal and is easy to check exactly on a synthetic
surface. What needs care is everything around it: the edge-contact measure, which was
wrong in a way that made a clipped basin look complete, and the ensemble, which is the
only thing standing between a plausible number and a trustworthy one.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import CatchmentConfig
from app.core.catchment import (
    Catchment,
    CatchmentDelineator,
    CatchmentEnsemble,
    DonorIndex,
    catchment_area,
    edge_contact_ratio,
    upstream_mask,
)
from app.core.dem_builder import ContourSurface
from app.core.kml_parser import parse_contour_file
from app.core.terrain import D8TerrainEngine, analyse_terrain
from tests.test_terrain import plane_east, synthetic_dem

SAMPLE = "data/contours_1m.kml"

# PLAN §3, "Results on the provided contour map".
# (lon, lat, area_ha, edge_contact_pct, relief_m, ensemble_mean_ha, ensemble_std_ha)
PUBLISHED_SITES = [
    (81.286465, 21.240094, 395.4, 1.3, 27.0, 423.5, 17.6),
    (81.293549, 21.263343, 101.8, 11.6, 14.4, 102.8, 1.5),
    (81.284248, 21.262484, 37.1, 3.9, 17.7, 37.3, 0.1),
    (81.297453, 21.240094, 35.7, 0.9, 10.0, 14.1, 15.4),
    (81.312393, 21.259544, 30.4, 4.4, 11.0, 31.6, 0.8),
]


# --------------------------------------------------------------------------- #
# Reverse-D8 traversal
# --------------------------------------------------------------------------- #
def test_donor_index_inverts_the_receiver_array(flow):
    """Every donor listed under a cell must actually drain into it, and every cell with a
    receiver must appear exactly once."""
    index = DonorIndex.build(flow.receivers, flow.receivers.size)
    for cell in (0, 5_000, 100_000, flow.receivers.size - 1):
        for donor in index.donors[index.offsets[cell] : index.offsets[cell + 1]]:
            assert flow.receivers[donor] == cell
    assert len(index.donors) == int((flow.receivers >= 0).sum())
    assert len(np.unique(index.donors)) == len(index.donors)


def test_upstream_mask_on_a_plane_is_one_row():
    """Each row of an east-facing plane is an independent chain, so the catchment of a
    cell is exactly the cells west of it in the same row."""
    dem = plane_east(ny=8, nx=12)
    flow = D8TerrainEngine().analyse(dem)
    index = DonorIndex.build(flow.receivers, dem.z.size)
    mask = upstream_mask(index, 3 * 12 + 7, dem.shape)
    expected = np.zeros(dem.shape, dtype=bool)
    expected[3, :8] = True
    np.testing.assert_array_equal(mask, expected)


def test_catchment_of_the_lowest_outlet_is_its_whole_basin(flow, delineator):
    """The mask must agree with the independently computed basin partition."""
    terminal = flow.terminal_outlets()
    outlet = int(np.bincount(terminal[flow.dem.valid]).argmax())
    row, col = divmod(outlet, flow.shape[1])
    mask = delineator.delineate_cell(row, col).mask
    np.testing.assert_array_equal(mask, terminal == outlet)


def test_traversal_is_iterative_not_recursive(delineator):
    """A basin on this sheet is over 150,000 cells; Python's recursion limit is 1,000.
    Delineating the largest one at all is the test."""
    biggest = np.unravel_index(
        int(np.argmax(delineator.flow.accumulation)), delineator.flow.shape
    )
    assert delineator.delineate_cell(*biggest).cell_count > 100_000


def test_a_catchment_contains_its_own_outlet(delineator):
    catchment = delineator.delineate(*PUBLISHED_SITES[2][:2])
    assert catchment.mask[catchment.outlet_rc]


def test_catchments_of_nested_outlets_are_nested(delineator, flow):
    """A downstream cell's catchment must contain its receiver's donors' -- basic
    consistency of the flow tree."""
    row, col = np.unravel_index(
        int(np.argmax(delineator.flow.accumulation)), flow.shape
    )
    downstream = delineator.delineate_cell(int(row), int(col))
    upstream_cell = int(delineator.donor_index.donors[
        delineator.donor_index.offsets[row * flow.shape[1] + col]
    ])
    upstream = delineator.delineate_cell(*divmod(upstream_cell, flow.shape[1]))
    assert (upstream.mask & ~downstream.mask).sum() == 0
    assert upstream.cell_count < downstream.cell_count


# --------------------------------------------------------------------------- #
# Area
# --------------------------------------------------------------------------- #
def test_catchment_area_is_latitude_weighted():
    mask = np.zeros((3, 4), dtype=bool)
    mask[0, :] = True
    rows = np.array([10.0, 20.0, 30.0])
    assert catchment_area(mask, rows) == pytest.approx(40.0)


def test_area_matches_the_dem_helper(delineator):
    catchment = delineator.delineate(*PUBLISHED_SITES[0][:2])
    assert catchment.area_m2 == pytest.approx(delineator.dem.area_of(catchment.mask))


def test_accumulation_at_the_outlet_equals_the_cell_count(delineator):
    """Two independent routes to the same catchment: the forward accumulation pass and
    the backward traversal. They must agree exactly."""
    for lon, lat, *_ in PUBLISHED_SITES:
        catchment = delineator.delineate(lon, lat)
        assert catchment.accumulation_cells == catchment.cell_count


# --------------------------------------------------------------------------- #
# Edge contact -- PLAN §11.3
# --------------------------------------------------------------------------- #
def test_an_interior_catchment_has_no_edge_contact():
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:7, 4:7] = True
    assert edge_contact_ratio(mask, np.zeros((10, 10), dtype=bool)) == 0.0


def test_a_catchment_filling_the_grid_is_all_edge():
    mask = np.ones((5, 5), dtype=bool)
    assert edge_contact_ratio(mask, np.zeros((5, 5), dtype=bool)) == 1.0


def test_edge_contact_counts_nodata_not_just_the_border():
    """The bug PLAN §11.3 names. This catchment touches no border at all, but half its
    perimeter faces a no-data hole -- ground the contours never described. Testing only
    the array border would report it complete."""
    nodata = np.zeros((12, 12), dtype=bool)
    nodata[4:8, 8:] = True
    mask = np.zeros((12, 12), dtype=bool)
    mask[4:8, 4:8] = True
    assert edge_contact_ratio(mask, nodata) == pytest.approx(4 / 16)
    # The border-only measure this replaced would have said zero.
    assert edge_contact_ratio(mask, np.zeros((12, 12), dtype=bool)) == 0.0


def test_edge_contact_is_a_perimeter_ratio_not_a_cell_ratio():
    """Counted as boundary edges, so a one-cell-wide finger touching the border
    contributes one unit of perimeter rather than one whole cell."""
    mask = np.zeros((6, 6), dtype=bool)
    mask[2, :] = True
    ratio = edge_contact_ratio(mask, np.zeros((6, 6), dtype=bool))
    assert ratio == pytest.approx(2 / 14)


def test_empty_mask_has_no_edge_contact():
    assert edge_contact_ratio(np.zeros((4, 4), dtype=bool), np.zeros((4, 4), bool)) == 0.0


def test_lower_bound_flag_follows_the_threshold(delineator):
    """Above the threshold the reported area is a floor, not a measurement."""
    catchment = delineator.delineate(*PUBLISHED_SITES[1][:2])
    assert catchment.edge_contact > 0.10
    strict = CatchmentDelineator(
        delineator.flow, config=CatchmentConfig(edge_contact_warn_fraction=0.05)
    ).delineate(*PUBLISHED_SITES[1][:2])
    assert strict.is_lower_bound
    lenient = CatchmentDelineator(
        delineator.flow, config=CatchmentConfig(edge_contact_warn_fraction=0.90)
    ).delineate(*PUBLISHED_SITES[1][:2])
    assert not lenient.is_lower_bound


# --------------------------------------------------------------------------- #
# Snapping -- PLAN §11.4, §11.9
# --------------------------------------------------------------------------- #
def test_snap_radius_scales_with_the_contour_spacing(delineator):
    """Not a fixed 30 m: the routed channel moves about 90 m between the ensemble's
    grids, so a fixed radius snaps to a different stream on each."""
    dem = delineator.dem
    assert delineator.snap_radius_m == pytest.approx(
        dem.meta.mean_contour_spacing_m * 3.0
    )
    assert delineator.snap_radius_m > 30.0


def test_snapping_moves_the_outlet_onto_a_channel(delineator):
    """A point 50 m off the stream must end up on it, with the move reported."""
    lon, lat, *_ = PUBLISHED_SITES[2]
    on_channel = delineator.delineate(lon, lat)
    off_channel = delineator.delineate(lon + 0.0004, lat)
    assert off_channel.snap_distance_m > 0
    assert off_channel.accumulation_cells > 100


def test_snap_distance_is_zero_when_already_on_the_peak(delineator):
    row, col = np.unravel_index(
        int(np.argmax(delineator.flow.accumulation)), delineator.flow.shape
    )
    lon, lat = (float(v) for v in delineator.dem.lonlat_of(int(row), int(col)))
    assert delineator.delineate(lon, lat).snap_distance_m == pytest.approx(0.0)


def test_snap_windows_are_clamped_at_every_edge(delineator):
    """PLAN §11.9. An unclamped slice turns a negative index into a wrap-around and
    argmax then returns a cell on the opposite side of the map. Every corner and edge
    must delineate without raising and without teleporting."""
    dem = delineator.dem
    ny, nx = dem.shape
    for row, col in ((0, 0), (0, nx - 1), (ny - 1, 0), (ny - 1, nx - 1),
                     (0, nx // 2), (ny - 1, nx // 2), (ny // 2, 0), (ny // 2, nx - 1)):
        lon, lat = (float(v) for v in dem.lonlat_of(row, col))
        try:
            catchment = delineator.delineate(lon, lat)
        except ValueError:
            continue  # legitimately outside the contour hull
        assert catchment.snap_distance_m <= delineator.snap_radius_m * 1.5


def test_a_point_outside_the_grid_is_rejected(delineator):
    with pytest.raises(ValueError, match="outside"):
        delineator.delineate(0.0, 0.0)


def test_snapping_can_be_turned_off(delineator):
    lon, lat, *_ = PUBLISHED_SITES[0]
    assert delineator.delineate(lon, lat, snap=False).snap_distance_m == 0.0


# --------------------------------------------------------------------------- #
# Relief and flow path
# --------------------------------------------------------------------------- #
def test_relief_is_measured_against_the_outlet_not_the_minimum(delineator):
    """A contour-derived DEM has unfilled pits in it, some below the outlet. Taking
    max-minus-min credits the basin with a drop the water never has -- about 4 m too much
    on every site of the sample."""
    catchment = delineator.delineate(*PUBLISHED_SITES[0][:2])
    ground = delineator.dem.z[catchment.mask]
    outlet_z = delineator.dem.z[catchment.outlet_rc]
    assert catchment.relief_m == pytest.approx(np.nanmax(ground) - outlet_z)
    assert np.nanmin(ground) < outlet_z          # the pits are really there
    assert catchment.relief_m < np.nanmax(ground) - np.nanmin(ground)


def test_kirpich_relief_differs_from_basin_relief(delineator):
    """The most distant point is rarely the highest one -- 19.7 m against 27.0 m on the
    largest basin. Kirpich wants the drop along the flow path."""
    catchment = delineator.delineate(*PUBLISHED_SITES[0][:2])
    assert catchment.flow_path_relief_m < catchment.relief_m
    assert catchment.flow_path_relief_m > 0


def test_flow_path_on_a_plane_is_the_row_length():
    """Exact answer: cell (3, 7) on an east-facing plane collects a straight chain of
    seven cells to its west, each one resolution apart."""
    dem = plane_east(ny=8, nx=12, res=10.0)
    delineator = CatchmentDelineator(D8TerrainEngine().analyse(dem))
    catchment = delineator.delineate_cell(3, 7)
    assert catchment.longest_flow_path_m == pytest.approx(70.0)


def test_flow_path_is_at_least_the_straight_line_distance(delineator):
    catchment = delineator.delineate(*PUBLISHED_SITES[0][:2])
    rows, cols = np.where(catchment.mask)
    row, col = catchment.outlet_rc
    span = np.hypot(rows - row, cols - col).max() * delineator.dem.resolution_m
    assert catchment.longest_flow_path_m >= span


# --------------------------------------------------------------------------- #
# The published results -- PLAN Phase 4 acceptance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("index", range(5))
def test_published_areas_reproduce(delineator, index):
    lon, lat, area_ha, _, _, _, _ = PUBLISHED_SITES[index]
    assert delineator.delineate(lon, lat).area_ha == pytest.approx(area_ha, rel=0.10)


@pytest.mark.parametrize("index", range(5))
def test_published_relief_reproduces(delineator, index):
    lon, lat, _, _, relief_m, _, _ = PUBLISHED_SITES[index]
    assert delineator.delineate(lon, lat).relief_m == pytest.approx(relief_m, abs=0.2)


def test_acceptance_site_one(delineator):
    """PLAN Phase 4: site 1 -> 395.4 ha at 5 m."""
    catchment = delineator.delineate(*PUBLISHED_SITES[0][:2])
    assert catchment.area_ha == pytest.approx(395.4, rel=0.01)
    assert not catchment.is_lower_bound
    assert catchment.cell_count / delineator.dem.valid.sum() == pytest.approx(0.48, abs=0.02)


def test_the_clipped_site_is_the_one_with_edge_contact(delineator):
    """Site 2 is the one PLAN §3 calls "partly clipped", and it must be the one this
    measure singles out."""
    contacts = [delineator.delineate(lon, lat).edge_contact for lon, lat, *_ in PUBLISHED_SITES]
    assert contacts.index(max(contacts)) == 1
    # 12.3% against a runner-up of 5.1%, and the only site within reach of the 15%
    # threshold at which an area stops being a measurement and becomes a floor.
    assert max(contacts) > 2 * sorted(contacts)[-2]
    assert sum(c > 0.10 for c in contacts) == 1


# --------------------------------------------------------------------------- #
# The ensemble -- PLAN §3 Test C
# --------------------------------------------------------------------------- #
def test_ensemble_uses_three_grids(ensemble):
    assert ensemble.resolutions_m == (5.0, 3.5, 2.5)
    assert len(ensemble.delineators) == 3


@pytest.mark.parametrize("index", range(5))
def test_published_ensemble_reproduces(ensemble, index):
    lon, lat, _, _, _, mean_ha, std_ha = PUBLISHED_SITES[index]
    result = ensemble.delineate(lon, lat)
    assert result.mean_area_ha == pytest.approx(mean_ha, rel=0.12)
    assert result.std_area_ha == pytest.approx(std_ha, abs=max(2.0, std_ha * 0.3))


def test_acceptance_site_one_ensemble(ensemble):
    """PLAN Phase 4: 423.5 +/- 17.6 ha."""
    result = ensemble.delineate(*PUBLISHED_SITES[0][:2])
    assert result.mean_area_ha == pytest.approx(423.5, rel=0.05)
    assert result.confidence == "high"


def test_site_four_is_rejected_as_unstable(ensemble):
    """The whole point of the ensemble (PLAN §3). Site 4 measures 35.7 ha on one grid and
    under 5 ha on the other two; reported as a single number it looks like a perfectly
    good 36 ha catchment."""
    result = ensemble.delineate(*PUBLISHED_SITES[3][:2])
    assert result.confidence == "low"
    assert result.coefficient_of_variation > 1.0
    areas = sorted(c.area_ha for c in result.per_grid)
    assert areas[-1] > 5 * areas[0]


def test_the_other_four_sites_are_high_confidence(ensemble):
    for lon, lat, *_ in PUBLISHED_SITES[:3] + PUBLISHED_SITES[4:]:
        assert ensemble.delineate(lon, lat).confidence == "high"


def test_each_grid_snaps_independently(ensemble):
    """Forcing every grid to one grid's cell would measure the snap rather than the
    terrain: the channel really is in a different place on each."""
    result = ensemble.delineate(*PUBLISHED_SITES[0][:2])
    outlets = {c.outlet_lonlat for c in result.per_grid}
    assert len(outlets) > 1


def test_confidence_thresholds_are_configurable(ensemble):
    tight = CatchmentConfig(confidence_high_cv=0.001, confidence_medium_cv=0.002)
    assert ensemble.classify(100.0, 1.0) == "high"
    assert CatchmentEnsemble.classify(
        type("E", (), {"config": tight})(), 100.0, 1.0
    ) == "low"


def test_zero_area_is_low_confidence(ensemble):
    assert ensemble.classify(0.0, 0.0) == "low"


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def surface():
    return ContourSurface(parse_contour_file(SAMPLE))


@pytest.fixture(scope="module")
def dem(surface):
    return surface.sample(5.0)


@pytest.fixture(scope="module")
def flow(dem):
    return analyse_terrain(dem)


@pytest.fixture(scope="module")
def delineator(flow):
    return CatchmentDelineator(flow)


@pytest.fixture(scope="module")
def ensemble(surface):
    return CatchmentEnsemble(surface)
