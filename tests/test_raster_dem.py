"""Phase 14. A drawn area to a DEM.

The raster path has to satisfy the same invariants as the contour path, or the claim
that `core/` cannot tell the two apart is only a claim. So the analytic valley and the mass
balance are re-run here through `RasterSurface`, alongside the checks that belong to this
path alone: the selection, the buffer, the adaptive resolution and the siting mask.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from app.config import settings
from app.core.catchment import CatchmentDelineator
from app.core.dem_builder import DEMBuildError
from app.core.projection import EquirectangularENU
from app.core.raster_dem import AreaError, AreaOfInterest, RasterSurface
from app.core.terrain import analyse_terrain
from app.providers.elevation import ElevationUnavailable, tile_range
from tests.fixtures.synthetic_tiles import AnalyticTileProvider, offline_config

SHEET_BBOX = (81.2814, 21.2398, 81.3126, 21.2636)
POLYGON = [[81.285, 21.241], [81.31, 21.243], [81.30, 21.262], [81.288, 21.255]]


@pytest.fixture(scope="module")
def sheet_surface() -> RasterSurface:
    return RasterSurface(AreaOfInterest.from_bbox(SHEET_BBOX))


@pytest.fixture(scope="module")
def sheet_dem(sheet_surface):
    return sheet_surface.sample()


# --------------------------------------------------------------------------- #
# The selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bbox", [(1, 2, 3), (81.3, 21.2, 81.2, 21.3), ("a", 1, 2, 3), None])
def test_a_malformed_bbox_is_a_named_error(bbox):
    with pytest.raises(AreaError) as err:
        AreaOfInterest.from_bbox(bbox)
    assert err.value.code == "invalid_aoi"


def test_a_latitude_off_the_map_is_caught():
    with pytest.raises(AreaError):
        AreaOfInterest.from_bbox((21.2, 86.2, 21.3, 95.3))


def test_a_polygon_needs_three_distinct_corners():
    with pytest.raises(AreaError):
        AreaOfInterest.from_polygon([[81.0, 21.0], [81.1, 21.0], [81.1, 21.0], [81.0, 21.0]])


def test_a_closed_ring_and_an_open_one_are_the_same_polygon():
    open_ring = AreaOfInterest.from_polygon(POLYGON)
    closed = AreaOfInterest.from_polygon(POLYGON + [POLYGON[0]])
    assert np.array_equal(open_ring.ring, closed.ring)


def test_either_winding_gives_a_positive_area():
    a = AreaOfInterest.from_polygon(POLYGON)
    b = AreaOfInterest.from_polygon(POLYGON[::-1])
    assert a.area_m2 == pytest.approx(b.area_m2) and a.area_m2 > 0


def test_geojson_feature_or_polygon_is_accepted():
    geometry = {"type": "Polygon", "coordinates": [POLYGON + [POLYGON[0]]]}
    feature = {"type": "Feature", "geometry": geometry, "properties": {}}
    assert AreaOfInterest.from_geojson(feature).area_m2 == pytest.approx(
        AreaOfInterest.from_geojson(geometry).area_m2
    )
    with pytest.raises(AreaError):
        AreaOfInterest.from_geojson({"type": "Point", "coordinates": [81, 21]})


def test_the_sheet_is_about_852_hectares():
    assert AreaOfInterest.from_bbox(SHEET_BBOX).area_m2 / 1e4 == pytest.approx(851.6, abs=1)


def test_a_huge_selection_is_a_named_422_not_an_oom():
    with pytest.raises(AreaError) as err:
        RasterSurface(AreaOfInterest.from_bbox((80.0, 20.0, 81.5, 21.5)))
    assert err.value.code == "aoi_too_large"


def test_a_tiny_selection_is_refused():
    with pytest.raises(AreaError) as err:
        RasterSurface(AreaOfInterest.from_bbox((81.29, 21.25, 81.2901, 21.2501)))
    assert err.value.code == "aoi_too_small"


def test_point_in_polygon():
    aoi = AreaOfInterest.from_polygon(POLYGON)
    assert aoi.contains(81.295, 21.25)
    assert not aoi.contains(81.28, 21.25)


# --------------------------------------------------------------------------- #
# The buffer
# --------------------------------------------------------------------------- #
def test_the_dem_covers_the_selection_plus_a_quarter_each_side(sheet_surface):
    min_lon, min_lat, max_lon, max_lat = SHEET_BBOX
    a = sheet_surface.analysed_bbox
    assert a[0] == pytest.approx(min_lon - 0.25 * (max_lon - min_lon))
    assert a[3] == pytest.approx(max_lat + 0.25 * (max_lat - min_lat))


def test_a_narrow_selection_still_gets_the_minimum_buffer():
    aoi = AreaOfInterest.from_bbox((81.29, 21.25, 81.30, 21.2505))
    b = aoi.buffered_bbox(0.25, 300.0)
    assert (21.25 - b[1]) * 110_540 == pytest.approx(300.0, rel=1e-6)


@pytest.mark.parametrize("make", [
    lambda: AreaOfInterest.from_bbox(SHEET_BBOX),
    lambda: AreaOfInterest.from_polygon(POLYGON),
])
def test_the_inside_mask_covers_the_drawn_area(make):
    aoi = make()
    surface = RasterSurface(aoi)
    dem = surface.sample()
    assert dem.area_of(surface.inside_mask(dem)) == pytest.approx(aoi.area_m2, rel=0.01)


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #
def test_the_sheet_grid_is_native_z13(sheet_dem):
    assert sheet_dem.resolution_m == pytest.approx(17.81, abs=0.01)
    assert sheet_dem.meta.resolution_source == "native"
    assert sheet_dem.nodata.sum() == 0


def test_the_raster_path_is_thirty_times_smaller_than_the_contour_path(sheet_dem):
    """PLAN3 §3.1: 886,000 cells on the contour path; tens of thousands here."""
    assert sheet_dem.shape[0] * sheet_dem.shape[1] < 70_000


def test_heights_over_the_sheet_match_the_survey_range(sheet_surface, sheet_dem):
    inside = sheet_surface.inside_mask(sheet_dem)
    raw = sheet_dem.raw_z[inside]
    assert 265 <= raw.min() <= 268 and 297 <= raw.max() <= 300


def test_smoothing_cannot_leave_the_sampled_range(sheet_dem):
    valid = sheet_dem.valid
    assert sheet_dem.z[valid].min() >= sheet_dem.raw_z[valid].min() - 1e-9
    assert sheet_dem.z[valid].max() <= sheet_dem.raw_z[valid].max() + 1e-9


def test_the_spacing_downstream_code_sees_is_the_source_posts(sheet_dem):
    """Snap radius and relative-elevation window scale with this; see RasterSurface."""
    assert sheet_dem.meta.mean_contour_spacing_m == settings.elevation.source_resolution_m
    assert sheet_dem.meta.contour_interval_m == 0.0


def test_a_requested_resolution_out_of_range_is_refused(sheet_surface):
    with pytest.raises(DEMBuildError) as err:
        sheet_surface.sample(1.0)
    assert err.value.code == "invalid_resolution"


def test_a_coarser_request_is_honoured(sheet_surface):
    dem = sheet_surface.sample(40.0)
    assert dem.resolution_m == 40.0 and dem.meta.resolution_source == "requested"


def test_a_big_selection_coarsens_to_the_cell_budget():
    cfg = offline_config(cell_budget=10_000)
    surface = RasterSurface(AreaOfInterest.from_bbox(SHEET_BBOX), config=cfg)
    dem = surface.sample()
    assert dem.shape[0] * dem.shape[1] <= 10_500
    assert dem.meta.resolution_source == "coarsened"
    assert any("cells" in w for w in dem.meta.warnings)


def test_too_much_missing_data_is_a_named_error():
    provider = AnalyticTileProvider()
    x0, y0, x1, y1 = tile_range(AreaOfInterest.from_bbox(SHEET_BBOX).buffered_bbox(0.25, 300), 13)
    provider.holes = {(13, x0, y0), (13, x1, y0), (13, x0, y1)}
    surface = RasterSurface(AreaOfInterest.from_bbox(SHEET_BBOX), provider=provider)
    with pytest.raises(ElevationUnavailable) as err:
        surface.sample()
    assert err.value.code == "elevation_unavailable"


# --------------------------------------------------------------------------- #
# Test A again: an analytic valley, through tiles
# --------------------------------------------------------------------------- #
def test_the_analytic_valley_through_tiles():
    """z = 0.05|x - c| + 0.01 y, built as terrarium tiles and read back.

    The channel is put on a column of the DEM so D8 has one channel, not two halves. Every
    cell flows across the valley and then down it, so the catchment of the channel cell in
    row R is every cell in rows R and above.
    """
    aoi = AreaOfInterest.from_bbox((81.295, 21.245, 81.305, 21.255))
    holder: dict = {}

    def height(lon, lat):
        projection: EquirectangularENU = holder["projection"]
        xy = projection.forward(np.stack([lon, lat], axis=-1))
        return 300.0 + 0.05 * np.abs(xy[..., 0] - holder["c"]) + 0.01 * xy[..., 1]

    provider = AnalyticTileProvider(height)
    surface = RasterSurface(aoi, provider=provider, config=offline_config(smoothing_sigma_m=0.0))
    ny, nx = surface.grid_shape(surface.auto_resolution_m)
    col = nx // 2
    holder["projection"] = surface.projection
    holder["c"] = surface.bounds_xy[0] + col * surface.auto_resolution_m

    dem = surface.sample()
    flow = analyse_terrain(dem)
    delineator = CatchmentDelineator(flow)
    for row in (ny // 4, ny // 2):
        catchment = delineator.delineate_cell(row, col)
        above = np.zeros(dem.shape, dtype=bool)
        above[row:, :] = True
        expected = dem.area_of(above)
        assert catchment.area_m2 == pytest.approx(expected, rel=0.02), row


# --------------------------------------------------------------------------- #
# Test B again: mass balance
# --------------------------------------------------------------------------- #
def test_basins_tile_the_raster_dem(sheet_dem):
    flow = analyse_terrain(sheet_dem)
    terminal = flow.terminal_outlets()
    areas = np.broadcast_to(sheet_dem.row_cell_areas[:, None], sheet_dem.shape)
    total = np.bincount(terminal[sheet_dem.valid], weights=areas[sheet_dem.valid]).sum()
    assert abs(total - sheet_dem.meta.mapped_area_m2) / total < 1e-12
