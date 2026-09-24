"""Satellite imagery, read for what the terrain cannot say (HLD FR-3 and §6.8).

The DEM says where water collects. It cannot say whether that ground is already a pond,
a house, or a grove, and no amount of elevation data will tell it. This module reads the
same Esri imagery the map shows and classifies every few metres of ground as one of:

    available   fields, fallow, bare ground: somewhere a pond could be dug
    water       an existing pond, tank or river
    built       roofs, yards and roads
    trees       closed tree cover

No OpenCV and no new dependency. The operations it would be used for here, colour indices,
local variance, morphological opening and connected components, are each a line of numpy
or scipy.ndimage, and a 60 MB wheel on a 512 MB container is a poor trade for them.

**How each class is recognised, and how honest that is.** These are spectral rules on
three visible bands, not a trained model:

* *Water* is dark, not green, and smooth. Village ponds on this imagery are dark
  olive-grey to blue-black; the texture test is what separates them from shadow, which is
  just as dark but sits beside something tall and bright.
* *Built* is bright, unsaturated and rough: concrete, tin and tile roofs, with edges.
* *Trees* are green, darker than crops, and rough: a canopy casts its own shadows.
* Everything else is *available*.

Blobs smaller than a few hundred square metres are dropped from every class, since at
2 m a pixel of shadow is not a building. The result is a screening layer that says "look
here before digging", and every response says so; it is not a cadastral survey.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from app.config import CVConfig, settings
from app.core.dem_builder import DEM, DEMMetadata
from app.core.projection import EquirectangularENU

__all__ = [
    "ImageryUnavailable", "Imagery", "LandClasses", "CLASSES", "fetch_imagery",
    "classify", "imagery_grid", "classes_for_area",
]

AVAILABLE, WATER, BUILT, TREES = 0, 1, 2, 3
CLASSES = {AVAILABLE: "available", WATER: "water", BUILT: "built", TREES: "trees"}
TILE = 256


class ImageryUnavailable(Exception):
    def __init__(self, code: str, detail: str, hint: str = "") -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.hint = hint


@dataclass(frozen=True)
class Imagery:
    """RGB resampled onto a square metric grid. `frame` is a `DEM` carrying only the
    geometry (no heights), so the same vectorising and area code as the terrain applies."""

    rgb: np.ndarray          # (ny, nx, 3) float32 in [0, 1]; NaN where no tile
    frame: DEM
    zoom: int
    tiles: int
    missing: int


@dataclass(frozen=True)
class LandClasses:
    classes: np.ndarray      # (ny, nx) uint8, one of CLASSES; 255 where no imagery
    frame: DEM
    imagery: Imagery

    def mask(self, cls: int) -> np.ndarray:
        return self.classes == cls

    def fractions(self) -> dict[str, float]:
        seen = self.classes != 255
        total = max(int(seen.sum()), 1)
        return {name: round(float((self.classes == c).sum()) / total, 4) for c, name in CLASSES.items()}


def _world(lon, lat):
    lat = np.clip(lat, -85.05112878, 85.05112878)
    s = np.sin(np.radians(lat))
    return (lon + 180.0) / 360.0, 0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)


def _frame(bbox, resolution_m: float) -> DEM:
    """An empty metric grid over `bbox`, as a `DEM` so the terrain's geometry code applies."""
    min_lon, min_lat, max_lon, max_lat = bbox
    projection = EquirectangularENU(lon0=(min_lon + max_lon) / 2, lat0=(min_lat + max_lat) / 2)
    (x0, y0), (x1, y1) = projection.forward(np.array([[min_lon, min_lat], [max_lon, max_lat]]))
    ny = int((y1 - y0) // resolution_m) + 1
    nx = int((x1 - x0) // resolution_m) + 1
    z = np.zeros((ny, nx))
    nodata = np.zeros((ny, nx), dtype=bool)
    meta = DEMMetadata(
        resolution_m=resolution_m, resolution_source="imagery", smoothing_sigma_m=0.0,
        mean_contour_spacing_m=resolution_m, total_contour_length_m=0.0,
        hull_area_m2=float((x1 - x0) * (y1 - y0)), mapped_area_m2=float((x1 - x0) * (y1 - y0)),
        nodata_fraction=0.0, elevation_range=(0.0, 0.0), max_smoothing_shift_m=0.0,
        smoothing_shift_p999_m=0.0, cells_over_interval=0, contour_interval_m=0.0,
        shape=(ny, nx),
    )
    return DEM(z=z, nodata=nodata, raw_z=z, resolution_m=resolution_m, origin_xy=(float(x0), float(y0)),
               projection=projection, meta=meta)


def _tile(url: str, key) -> bytes | None:
    # The renderer's tile cache and fetcher: one cache for the map picture and the CV.
    from app.core.render import _tile_bytes

    return _tile_bytes(url, key, settings.render)


def fetch_imagery(
    bbox: tuple[float, float, float, float],
    *,
    resolution_m: float | None = None,
    config: CVConfig | None = None,
    fetch=None,
) -> Imagery:
    """Esri World Imagery over `bbox`, resampled to a metric grid.

    `fetch(z, x, y) -> PNG/JPEG bytes | None` replaces the network in tests.
    """
    from io import BytesIO

    from PIL import Image

    cfg = config or settings.cv

    def span(z: int) -> tuple[int, int, int, int]:
        n = 2 ** z
        fx0, fy0 = _world(np.asarray(bbox[0]), np.asarray(bbox[3]))
        fx1, fy1 = _world(np.asarray(bbox[2]), np.asarray(bbox[1]))
        return (int(math.floor(float(fx0) * n)), int(math.floor(float(fy0) * n)),
                int(math.floor(float(fx1) * n)), int(math.floor(float(fy1) * n)))

    # The finest zoom whose mosaic fits the pixel budget. Memory, not detail, is the limit
    # on a 512 MB container: a bigger area is read coarser rather than refused.
    zoom = cfg.imagery_zoom
    while True:
        x0, y0, x1, y1 = span(zoom)
        count = (x1 - x0 + 1) * (y1 - y0 + 1)
        if count * TILE * TILE <= cfg.max_pixels or zoom <= cfg.min_imagery_zoom:
            break
        zoom -= 1
    if count * TILE * TILE > cfg.max_pixels:
        raise ImageryUnavailable(
            "aoi_too_large",
            f"The area needs {count} imagery tiles even at zoom {zoom}.",
            "Draw a smaller area for the imagery layers.",
        )
    n = 2 ** zoom
    lat0 = (bbox[1] + bbox[3]) / 2
    pixel_m = 2 * math.pi * 6378137.0 * math.cos(math.radians(lat0)) / (TILE * n)
    resolution_m = max(resolution_m or pixel_m, pixel_m)
    frame = _frame(bbox, resolution_m)

    wanted = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    template = settings.render.satellite_url
    get = fetch or (lambda z, x, y: _tile(template.format(z=z, x=x, y=y), ("satellite", z, x, y)))

    # uint8 and a separate hole mask: a quarter of the memory of float RGB.
    mosaic = np.zeros(((y1 - y0 + 1) * TILE, (x1 - x0 + 1) * TILE, 3), dtype=np.uint8)
    present = np.zeros(mosaic.shape[:2], dtype=bool)
    missing = 0
    with ThreadPoolExecutor(max_workers=settings.render.tile_workers) as pool:
        for (x, y), data in zip(wanted, pool.map(lambda xy: get(zoom, *xy), wanted)):
            try:
                tile = np.asarray(Image.open(BytesIO(data)).convert("RGB"), dtype=np.uint8) if data else None
            except Exception:  # noqa: BLE001, a bad tile is a hole
                tile = None
            if tile is None or tile.shape[:2] != (TILE, TILE):
                missing += 1
                continue
            r, c = (y - y0) * TILE, (x - x0) * TILE
            mosaic[r : r + TILE, c : c + TILE] = tile
            present[r : r + TILE, c : c + TILE] = True
    if missing == len(wanted):
        raise ImageryUnavailable(
            "imagery_unavailable",
            f"None of the {len(wanted)} satellite tiles for this area could be fetched.",
            "The imagery server may be unreachable from here. The terrain analysis does not "
            "need it; only the land-availability and existing-water layers do.",
        )

    ny, nx = frame.shape
    x0m, y0m = frame.origin_xy
    gx, gy = np.meshgrid(x0m + np.arange(nx) * resolution_m, y0m + np.arange(ny) * resolution_m)
    lonlat = frame.projection.inverse(np.stack([gx, gy], axis=-1))
    wx, wy = _world(lonlat[..., 0], lonlat[..., 1])
    cols = np.clip(np.rint(wx * n * TILE - x0 * TILE - 0.5).astype(int), 0, mosaic.shape[1] - 1)
    rows = np.clip(np.rint(wy * n * TILE - y0 * TILE - 0.5).astype(int), 0, mosaic.shape[0] - 1)
    rgb = mosaic[rows, cols].astype(np.float32) / 255.0
    rgb[~present[rows, cols]] = np.nan
    return Imagery(rgb=rgb, frame=frame, zoom=zoom, tiles=len(wanted), missing=missing)


def _local_std(values: np.ndarray, size: int) -> np.ndarray:
    mean = ndimage.uniform_filter(values, size)
    mean_sq = ndimage.uniform_filter(values * values, size)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def classify(imagery: Imagery, config: CVConfig | None = None) -> LandClasses:
    """Per-cell land class. See the module docstring for the rules and their limits."""
    cfg = config or settings.cv
    rgb = imagery.rgb
    seen = np.isfinite(rgb).all(axis=-1)
    rgb = np.where(seen[..., None], rgb, 0.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    total = r + g + b + 1e-6
    brightness = total / 3.0
    high = rgb.max(axis=-1)
    saturation = (high - rgb.min(axis=-1)) / (high + 1e-6)
    exg = (2 * g - r - b) / total
    # Scale the texture window to metres, so it means the same at any zoom.
    window = max(3, int(round(cfg.texture_window_px * 2.2 / imagery.frame.resolution_m)) | 1)
    texture = _local_std(ndimage.gaussian_filter(brightness, 0.8), window)

    blue_share = b / total
    looks_wet = (
        (blue_share > cfg.water_min_blue_share)
        & (brightness < cfg.water_max_brightness)
        & (exg < 0.32)
    )
    water = looks_wet & (texture < cfg.water_max_texture)
    # A village is not one roof but many close together: rough, unvegetated pixels. Scored
    # as a density over ~30 m, so the lanes and yards between roofs belong to it too.
    rough = (texture > cfg.builtup_min_texture) & (exg < 0.16) & (brightness > 0.24)
    rough |= (brightness > cfg.builtup_min_brightness) & (saturation < cfg.builtup_max_saturation)
    density_px = max(3, int(round(30.0 / imagery.frame.resolution_m)))
    built = ndimage.uniform_filter(rough.astype(np.float32), density_px) > cfg.builtup_min_density
    trees = (exg > cfg.vegetation_min_exg) & (texture > 0.05) & (brightness < 0.30)

    res = imagery.frame.resolution_m
    open_size = max(1, int(round(4.0 / res)))

    def clean(mask: np.ndarray, min_area_m2: float) -> np.ndarray:
        mask = ndimage.binary_opening(mask, iterations=open_size)
        mask = ndimage.binary_closing(mask, iterations=open_size)
        mask = ndimage.binary_fill_holes(mask)
        labels, count = ndimage.label(mask)
        if count == 0:
            return mask
        sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1)) * res * res
        keep = np.zeros(count + 1, dtype=bool)
        keep[1:] = sizes >= min_area_m2
        return keep[labels]

    water = clean(water, cfg.water_min_area_m2)
    # The texture window straddles every shoreline, so the smoothness test eats a band
    # half a window wide round each pond: a quarter of a 60 m pond's area. Grow the cores
    # back across that band, only into pixels that are spectrally water, and only from
    # cores already big enough to be ponds, so a flooded paddy cannot seed itself.
    water = ndimage.binary_dilation(water, iterations=window // 2 + 1, mask=looks_wet)
    built = clean(built & ~water, 600.0)
    trees = clean(trees & ~water & ~built, 400.0)

    classes = np.full(seen.shape, AVAILABLE, dtype=np.uint8)
    classes[trees] = TREES
    classes[built] = BUILT
    classes[water] = WATER
    classes[~seen] = 255
    return LandClasses(classes=classes, frame=imagery.frame, imagery=imagery)


def classes_for_area(bbox, *, config: CVConfig | None = None, fetch=None) -> LandClasses:
    return classify(fetch_imagery(bbox, config=config, fetch=fetch), config)


def imagery_grid(land: LandClasses, dem: DEM, cls_set=(WATER, BUILT, TREES), share: float = 0.5) -> np.ndarray:
    """(ny, nx) bool on `dem`'s grid: cells more than `share` covered by the given classes.

    Each DEM cell is tens of metres; the imagery is two. The DEM cell is unavailable when
    most of the ground inside it is, sampled on a 3 x 3 lattice within the cell.
    """
    ny, nx = dem.shape
    x0, y0 = dem.origin_xy
    res = dem.resolution_m
    offsets = (np.arange(3) - 1) * res / 3.0
    hits = np.zeros((ny, nx), dtype=np.float32)
    seen = np.zeros((ny, nx), dtype=np.float32)
    bad = np.isin(land.classes, cls_set)
    known = land.classes != 255
    fx0, fy0 = land.frame.origin_xy
    fres = land.frame.resolution_m
    fny, fnx = land.frame.shape
    for dy in offsets:
        for dx in offsets:
            gx, gy = np.meshgrid(x0 + np.arange(nx) * res + dx, y0 + np.arange(ny) * res + dy)
            lonlat = dem.projection.inverse(np.stack([gx, gy], axis=-1))
            fxy = land.frame.projection.forward(lonlat)
            c = np.rint((fxy[..., 0] - fx0) / fres).astype(int)
            r = np.rint((fxy[..., 1] - fy0) / fres).astype(int)
            inside = (r >= 0) & (r < fny) & (c >= 0) & (c < fnx)
            rc, cc = np.clip(r, 0, fny - 1), np.clip(c, 0, fnx - 1)
            ok = inside & known[rc, cc]
            seen += ok
            hits += ok & bad[rc, cc]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (seen > 0) & (hits / np.maximum(seen, 1) > share)
