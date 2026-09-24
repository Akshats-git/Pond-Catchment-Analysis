"""Elevation tiles made from a formula, for tests that must not touch the network.

`AnalyticTileProvider` answers every tile request by evaluating a function at each
pixel's lon/lat and round-tripping the result through a real terrarium PNG, so the tests
exercise the decoding, the mosaicking and the reprojection exactly as live tiles would,
against terrain whose answer is known.
"""

from __future__ import annotations

import dataclasses
import io
import math
from typing import Callable

import numpy as np
from PIL import Image

from app.config import settings
from app.providers.elevation import TILE_SIZE, TerrariumProvider, encode_terrarium

__all__ = ["AnalyticTileProvider", "offline_config"]


def offline_config(**overrides):
    return dataclasses.replace(settings.elevation, network_enabled=False, **overrides)


def pixel_lonlat(z: int, x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
    """lon/lat of the centre of every pixel in tile (z, x, y)."""
    world = TILE_SIZE * 2 ** z
    cols = x * TILE_SIZE + np.arange(TILE_SIZE) + 0.5
    rows = y * TILE_SIZE + np.arange(TILE_SIZE) + 0.5
    gx, gy = np.meshgrid(cols, rows)
    lon = gx / world * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(math.pi * (1.0 - 2.0 * gy / world))))
    return lon, lat


class AnalyticTileProvider(TerrariumProvider):
    """Tiles of `height(lon, lat)`, encoded as terrarium PNGs and decoded again."""

    name = "analytic"
    description = "analytic test surface"

    def __init__(
        self,
        height: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        *,
        holes: set[tuple[int, int, int]] | None = None,
        config=None,
    ) -> None:
        super().__init__(config or offline_config())
        self.height = height or (lambda lon, lat: np.full(np.shape(lon), 100.0))
        self.holes = holes or set()
        self.calls = 0

    def fetch(self, z: int, x: int, y: int) -> bytes | None:
        self.calls += 1
        if (z, x, y) in self.holes:
            return None
        lon, lat = pixel_lonlat(z, x, y)
        rgb = encode_terrarium(self.height(lon, lat))
        buffer = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(buffer, format="PNG")
        return buffer.getvalue()
