"""Phase 5 Test B -- mass balance.

The cheapest possible check on a flow network, and the strictest. Every cell drains to
exactly one outlet, so the basins tile the map: their areas must sum to the mapped area,
with nothing lost and nothing double-counted.

It is worth having because it fails loudly for almost any real defect in the routing. A
cycle in the flow graph, a cell that drains into no-data, an off-by-one in the neighbour
offsets, a catchment that claims a cell twice -- all of them break the sum. Measured on
the sample sheet the difference is 0.00000000%.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.catchment import CatchmentDelineator
from app.core.dem_builder import ContourSurface
from app.core.kml_parser import parse_contour_file, parse_contours
from app.core.terrain import analyse_terrain
from tests.fixtures import make_variants as mv
from tests.fixtures.make_synthetic import VALLEY

SAMPLE = "data/contours_1m.kml"
TOLERANCE = 1e-4  # PLAN §3 Test B: better than 0.01%


def basin_areas(flow, dem) -> np.ndarray:
    """Ground area of every basin on the map, indexed by its outlet."""
    terminal = flow.terminal_outlets()
    areas = np.broadcast_to(dem.row_cell_areas[:, None], dem.shape)
    return np.bincount(terminal[dem.valid], weights=areas[dem.valid])


# --------------------------------------------------------------------------- #
# The sample sheet
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("resolution", [5.0, 3.5, 2.5])
def test_basins_tile_the_sample_sheet(sample_surface, resolution):
    """PLAN §3 Test B, at each of the ensemble's grids."""
    dem = sample_surface.sample(resolution)
    flow = analyse_terrain(dem)
    assert basin_areas(flow, dem).sum() == pytest.approx(
        dem.meta.mapped_area_m2, rel=TOLERANCE
    )


def test_the_sample_mass_balance_is_exact(sample_flow, sample_dem):
    """Not merely within tolerance -- exact to floating point. Areas are summed the same
    way on both sides, so any discrepancy at all would be a real cell going missing."""
    total = basin_areas(sample_flow, sample_dem).sum()
    assert abs(total - sample_dem.meta.mapped_area_m2) / total < 1e-12


def test_every_valid_cell_belongs_to_a_basin(sample_flow, sample_dem):
    terminal = sample_flow.terminal_outlets()
    assert (terminal[sample_dem.valid] >= 0).all()
    assert (terminal[sample_dem.nodata] == -1).all()


def test_the_basin_count_matches_the_outlet_count(sample_flow, sample_dem):
    terminal = sample_flow.terminal_outlets()
    outlets = (sample_flow.receivers.reshape(sample_dem.shape) < 0) & sample_dem.valid
    assert len(np.unique(terminal[sample_dem.valid])) == int(outlets.sum())
    assert int(outlets.sum()) == sample_flow.meta.outlet_count


def test_no_basin_is_larger_than_the_map(sample_flow, sample_dem):
    assert basin_areas(sample_flow, sample_dem).max() <= sample_dem.meta.mapped_area_m2


# --------------------------------------------------------------------------- #
# The same property, from the delineation side
# --------------------------------------------------------------------------- #
def test_delineated_basins_partition_the_map(sample_flow, sample_dem):
    """Approach the same invariant through the code the API actually calls.

    `terminal_outlets` walks downstream; `upstream_mask` walks up. Delineating every
    outlet must reconstruct the map exactly once over -- no cell in two catchments, none
    in none. That the two independent traversals agree is the real content here.
    """
    delineator = CatchmentDelineator(sample_flow)
    outlets = np.flatnonzero(
        (sample_flow.receivers < 0) & sample_dem.valid.ravel()
    )
    coverage = np.zeros(sample_dem.shape, dtype=np.int32)
    for outlet in outlets:
        row, col = divmod(int(outlet), sample_dem.shape[1])
        coverage += delineator.delineate_cell(row, col).mask
    assert (coverage[sample_dem.valid] == 1).all()
    assert (coverage[sample_dem.nodata] == 0).all()


def test_catchment_areas_sum_to_the_mapped_area(sample_flow, sample_dem):
    delineator = CatchmentDelineator(sample_flow)
    outlets = np.flatnonzero((sample_flow.receivers < 0) & sample_dem.valid.ravel())
    total = sum(
        delineator.delineate_cell(*divmod(int(o), sample_dem.shape[1])).area_m2
        for o in outlets
    )
    assert total == pytest.approx(sample_dem.meta.mapped_area_m2, rel=TOLERANCE)


# --------------------------------------------------------------------------- #
# The analytic valley
# --------------------------------------------------------------------------- #
def test_the_valley_drains_to_a_single_outlet():
    """The whole point of the surface: it is one basin. If mass balance held but the
    valley came apart into several basins, the routing would be wrong in a way the sum
    alone could not see."""
    surface = ContourSurface(parse_contours(VALLEY.to_kml()))
    dem = surface.sample(5.0)
    flow = analyse_terrain(dem)
    areas = basin_areas(flow, dem)
    assert areas.sum() == pytest.approx(dem.meta.mapped_area_m2, rel=TOLERANCE)
    assert areas.max() > 0.98 * dem.meta.mapped_area_m2


# --------------------------------------------------------------------------- #
# Structural variants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name", ["placemark_name", "z_coordinate", "polygon", "multigeometry", "kmz"]
)
def test_mass_balance_on_every_structural_variant(name):
    surface = ContourSurface(parse_contours(mv.VARIANTS[name]()))
    dem = surface.sample(5.0)
    flow = analyse_terrain(dem)
    assert basin_areas(flow, dem).sum() == pytest.approx(
        dem.meta.mapped_area_m2, rel=TOLERANCE
    )


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample_surface():
    return ContourSurface(parse_contour_file(SAMPLE))


@pytest.fixture(scope="module")
def sample_dem(sample_surface):
    return sample_surface.sample(5.0)


@pytest.fixture(scope="module")
def sample_flow(sample_dem):
    return analyse_terrain(sample_dem)
