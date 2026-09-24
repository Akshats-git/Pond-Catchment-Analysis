"""Phase 13. The elevation provider.

What has to be true of tiles before a DEM is built from them: that the terrarium decoding
keeps its fractional metres, that the ground resolution carries the cosine, that the right
tiles are asked for, and that the cache layers answer in the right order and never ask the
network twice for a tile it already has.
"""

from __future__ import annotations

import dataclasses
import io
import math

import numpy as np
import pytest
from PIL import Image

from app.config import settings
from app.providers import elevation as el
from app.providers.elevation import (
    CachedProvider,
    ElevationUnavailable,
    GeographicGrid,
    TileMosaic,
    decode_terrarium,
    encode_terrarium,
    parse_aaigrid,
    tile_range,
    tile_resolution_m,
)
from tests.fixtures.synthetic_tiles import AnalyticTileProvider, offline_config

SHEET_BBOX = (81.2814, 21.2398, 81.3126, 21.2636)


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #
def test_terrarium_zero_is_128_0_0():
    assert decode_terrarium(np.array([[[128, 0, 0]]], dtype=np.uint8))[0, 0] == 0.0


def test_the_blue_channel_carries_fractions_of_a_metre():
    """PLAN3 §11.1. Dropping B/256 would quantise to whole metres."""
    z = decode_terrarium(np.array([[[129, 10, 128]]], dtype=np.uint8))[0, 0]
    assert z == pytest.approx(256 + 10 + 0.5)


def test_encoding_round_trips_to_a_256th_of_a_metre():
    heights = np.array([[-12.3, 0.0, 266.02], [298.37, 1234.5678, 8848.0]])
    assert np.abs(decode_terrarium(encode_terrarium(heights)) - heights).max() <= 1 / 512 + 1e-9


# --------------------------------------------------------------------------- #
# Web Mercator
# --------------------------------------------------------------------------- #
def test_z13_is_17_8_metres_at_the_sheet():
    assert tile_resolution_m(13, 21.25) == pytest.approx(17.81, abs=0.01)
    assert tile_resolution_m(12, 21.25) == pytest.approx(35.62, abs=0.01)


def test_the_equatorial_figure_would_overstate_resolution_by_seven_percent():
    """PLAN3 §11.2: the cosine is not optional at 21 N."""
    assert tile_resolution_m(13, 0.0) / tile_resolution_m(13, 21.25) == pytest.approx(1.073, abs=0.002)


def test_four_z13_tiles_cover_the_sheet():
    x0, y0, x1, y1 = tile_range(SHEET_BBOX, 13)
    assert (x1 - x0 + 1) * (y1 - y0 + 1) == 4


@pytest.mark.parametrize("wanted, zoom", [(17.9, 13), (30.0, 13), (36.0, 12), (60.0, 12), (5.0, 13), (5000.0, 8)])
def test_zoom_is_the_coarsest_at_least_as_fine_as_the_grid(wanted, zoom):
    provider = AnalyticTileProvider()
    assert provider.zoom_for(21.25, wanted) == zoom


# --------------------------------------------------------------------------- #
# Mosaics
# --------------------------------------------------------------------------- #
def test_a_mosaic_samples_the_surface_it_was_made_from():
    surface = lambda lon, lat: 200.0 + (lon - 81.29) * 1000.0 + (lat - 21.25) * 500.0  # noqa: E731
    provider = AnalyticTileProvider(surface)
    mosaic = provider.tiles_for(SHEET_BBOX, 13)
    lon, lat = np.meshgrid(np.linspace(81.285, 81.31, 20), np.linspace(21.242, 21.262, 20))
    assert np.abs(mosaic.sample(lon, lat) - surface(lon, lat)).max() < 0.01


def test_a_point_outside_the_mosaic_is_nan():
    mosaic = AnalyticTileProvider().tiles_for(SHEET_BBOX, 13)
    assert np.isnan(mosaic.sample(np.array([0.0]), np.array([0.0])))[0]


def test_a_missing_tile_is_a_hole_not_low_ground():
    provider = AnalyticTileProvider()
    x0, y0, _, _ = tile_range(SHEET_BBOX, 13)
    provider.holes = {(13, x0, y0)}
    mosaic = provider.tiles_for(SHEET_BBOX, 13)
    assert np.isnan(mosaic.z).sum() == 256 * 256
    assert mosaic.stats["tile_sources"] == {"network": 3, "missing": 1}


def test_no_tiles_at_all_is_a_named_error():
    provider = AnalyticTileProvider()
    x0, y0, x1, y1 = tile_range(SHEET_BBOX, 13)
    provider.holes = {(13, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
    with pytest.raises(ElevationUnavailable) as err:
        provider.tiles_for(SHEET_BBOX, 13)
    assert err.value.code == "elevation_unavailable"


def test_too_many_tiles_is_refused_before_any_are_fetched():
    provider = AnalyticTileProvider(config=offline_config(max_tiles=2))
    with pytest.raises(ElevationUnavailable) as err:
        provider.tiles_for(SHEET_BBOX, 13)
    assert err.value.code == "aoi_too_large"
    assert provider.calls == 0


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #
def test_the_committed_seed_tiles_serve_the_sheet_offline():
    provider = el.default_provider()
    mosaic = provider.tiles_for(SHEET_BBOX, 13)
    lon, lat = np.meshgrid(np.linspace(SHEET_BBOX[0], SHEET_BBOX[2], 60),
                           np.linspace(SHEET_BBOX[1], SHEET_BBOX[3], 60))
    z = mosaic.sample(lon, lat)
    # PLAN3 §3.1: SRTM over the sheet reads 266-299 m; the survey says 267-298 m.
    assert 265 <= np.nanmin(z) <= 268 and 297 <= np.nanmax(z) <= 300
    assert set(mosaic.stats["tile_sources"]) <= {"seed", "memory"}


def test_the_cache_fetches_each_tile_once_and_persists_it(tmp_path):
    cfg = offline_config(cache_dir=str(tmp_path), seed_dir=str(tmp_path / "none"))
    inner = AnalyticTileProvider(config=cfg)
    cached = CachedProvider(inner, cfg)

    first = cached.tiles_for(SHEET_BBOX, 13)
    assert first.stats["tile_sources"] == {"network": 4}
    assert inner.calls == 4
    assert len(list(tmp_path.rglob("*.png"))) == 4

    second = cached.tiles_for(SHEET_BBOX, 13)
    assert second.stats["tile_sources"] == {"memory": 4}
    assert inner.calls == 4

    # A fresh process reads the disk, not the network.
    again = CachedProvider(AnalyticTileProvider(config=cfg), cfg)
    assert again.tiles_for(SHEET_BBOX, 13).stats["tile_sources"] == {"disk": 4}
    assert np.array_equal(first.z, second.z)


def test_a_corrupt_cached_tile_is_refetched(tmp_path):
    cfg = offline_config(cache_dir=str(tmp_path), seed_dir=str(tmp_path / "none"))
    inner = AnalyticTileProvider(config=cfg)
    cached = CachedProvider(inner, cfg)
    x0, y0, _, _ = tile_range(SHEET_BBOX, 13)
    bad = tmp_path / "terrarium" / "13" / str(x0) / f"{y0}.png"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a png")
    heights, origin = cached.tile(13, x0, y0)
    assert heights is not None and origin == "network"


def test_a_different_config_gets_its_own_provider():
    other = dataclasses.replace(settings.elevation, max_zoom=12)
    assert el.default_provider(other) is not el.default_provider()
    assert el.default_provider() is el.default_provider()


# --------------------------------------------------------------------------- #
# OpenTopography's ASCII grid
# --------------------------------------------------------------------------- #
def test_an_ascii_grid_parses_and_samples():
    text = "\n".join([
        "ncols 3", "nrows 2", "xllcorner 81.0", "yllcorner 21.0", "cellsize 0.01",
        "NODATA_value -9999", "10 20 30", "40 50 -9999",
    ])
    grid = parse_aaigrid(text)
    assert isinstance(grid, GeographicGrid)
    assert grid.z.shape == (2, 3)
    assert np.isnan(grid.z[1, 2])
    # Row 0 is the north row: its centre is at 21.015.
    assert grid.sample(np.array([81.005]), np.array([21.015]))[0] == pytest.approx(10.0)
    assert grid.sample(np.array([81.015]), np.array([21.005]))[0] == pytest.approx(50.0)


def test_opentopography_needs_a_key():
    with pytest.raises(ElevationUnavailable):
        el.OpenTopographyProvider(offline_config(opentopo_key=""))
