"""Where the ground comes from when nobody uploaded a contour sheet.

Phase 3 lets a user draw an area on the map instead of uploading a survey. Something has
to supply the heights, and this module is that something. It mirrors `rainfall.py`: one
interface, a keyless primary source, an opt-in alternate, and a cache in front so the same
request never pays twice.

**The primary source is the AWS terrain-tile bucket in terrarium encoding.** Free, keyless,
reachable from the lab containers, and it answers in PNG tiles that Pillow already reads.
Each pixel is a height:

    elevation = (R * 256 + G + B / 256) - 32768

The `B / 256` term is not optional. Dropping it quantises every tile to whole metres and
flattens exactly the gentle slopes that decide where water goes (PLAN3 §11.1).

**Tiles are immutable**, so the cache never expires anything. It is layered: decoded tiles
in memory, then a read-only seed directory committed with the repository for the demo
region, then a writable disk cache, then the network. The seed layer is what makes the
demo work with the uplink down.

**No GDAL.** The alternate, OpenTopography, is asked for ASCII grid output, which numpy
reads. Adding GDAL to a 512 MB container with no sudo is not something to discover on
demo day.
"""

from __future__ import annotations

import io
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from app.config import ElevationConfig, settings

__all__ = [
    "ElevationUnavailable",
    "ElevationGrid",
    "TileMosaic",
    "GeographicGrid",
    "ElevationProvider",
    "TileProvider",
    "TerrariumProvider",
    "CachedProvider",
    "OpenTopographyProvider",
    "decode_terrarium",
    "encode_terrarium",
    "tile_resolution_m",
    "tile_range",
    "default_provider",
    "provider_for",
    "REPO_ROOT",
]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EARTH_CIRCUMFERENCE_M = 2 * math.pi * 6378137.0
"""Web Mercator's equatorial circumference. 156,543.03 m per pixel at z0 for 256 px tiles."""

TILE_SIZE = 256


class ElevationUnavailable(Exception):
    """The heights for the requested area could not be obtained.

    Carries the service-wide `(code, detail, hint)` triple so the route maps it the same
    way as every error in `app/core/`.
    """

    def __init__(self, code: str, detail: str, hint: str = "") -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.hint = hint


# --------------------------------------------------------------------------- #
# Web Mercator arithmetic
# --------------------------------------------------------------------------- #
def tile_resolution_m(zoom: int, lat: float) -> float:
    """Ground size of one tile pixel at this latitude.

    `156543.034 * cos(lat) / 2^z`. The cosine is the point: the equatorial figure
    overstates the ground resolution by 7% at 21 N, and every area downstream would
    inherit the error (PLAN3 §11.2).
    """
    return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / (TILE_SIZE * 2 ** zoom)


def _world_fraction(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """lon/lat -> [0, 1) Web Mercator, y increasing southward as tiles number."""
    lat = np.clip(lat, -85.05112878, 85.05112878)
    sin = np.sin(np.radians(lat))
    return (lon + 180.0) / 360.0, 0.5 - np.log((1 + sin) / (1 - sin)) / (4 * math.pi)


def tile_range(bbox: tuple[float, float, float, float], zoom: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1), inclusive, of the tiles covering a bbox at a zoom."""
    min_lon, min_lat, max_lon, max_lat = bbox
    n = 2 ** zoom
    fx0, fy0 = _world_fraction(np.asarray(min_lon), np.asarray(max_lat))
    fx1, fy1 = _world_fraction(np.asarray(max_lon), np.asarray(min_lat))
    clamp = lambda v: int(min(max(v, 0), n - 1))  # noqa: E731
    return (
        clamp(math.floor(float(fx0) * n)),
        clamp(math.floor(float(fy0) * n)),
        clamp(math.floor(float(fx1) * n)),
        clamp(math.floor(float(fy1) * n)),
    )


def decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    """(h, w, 3) uint8 -> (h, w) float32 metres. See the module docstring."""
    rgb = rgb.astype(np.float32)
    return (rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0) - 32768.0


def encode_terrarium(z: np.ndarray) -> np.ndarray:
    """The inverse, for tests and for building synthetic tiles. (h, w) -> (h, w, 3) uint8."""
    value = np.asarray(z, dtype=np.float64) + 32768.0
    r = np.floor(value / 256.0)
    g = np.floor(value - r * 256.0)
    b = np.round((value - r * 256.0 - g) * 256.0)
    # A fraction that rounds up to 256 carries into G.
    carry = b >= 256
    b[carry] -= 256
    g[carry] += 1
    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# What a provider hands back
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ElevationGrid(ABC):
    """Heights on some regular grid that can be read at any lon/lat.

    The raster DEM builder never looks inside: it asks for the height at each of its own
    cell centres, so a Web Mercator mosaic and a geographic ASCII grid are the same thing
    to it.
    """

    z: np.ndarray
    """(rows, cols) float32 metres, row 0 at the north edge. NaN where there is no data."""

    source: str
    resolution_m: float
    """Nominal ground size of one source pixel at the grid's centre."""

    stats: dict = field(default_factory=dict)
    """How the grid was assembled: tile counts by where they came from, and so on."""

    @abstractmethod
    def pixel_of(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fractional (row, col) of a lon/lat, in `z`'s own pixel frame."""

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Bilinear heights at the given points. NaN outside the grid or next to a hole.

        Bilinear rather than nearest: nearest would copy the source's 30 m posts onto a
        17 m grid as blocks, and a block is a flat, which is exactly what D8 cannot route
        across. A hole propagates as NaN rather than being interpolated across, because a
        missing tile is not low ground.
        """
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        rows, cols = self.pixel_of(lon, lat)
        ny, nx = self.z.shape
        outside = (rows < -0.5) | (rows > ny - 0.5) | (cols < -0.5) | (cols > nx - 0.5)
        coords = np.stack([rows.ravel(), cols.ravel()])
        filled = np.where(np.isfinite(self.z), self.z, 0.0).astype(np.float32)
        holes = (~np.isfinite(self.z)).astype(np.float32)
        values = map_coordinates(filled, coords, order=1, mode="nearest")
        touched = map_coordinates(holes, coords, order=1, mode="nearest")
        out = values.reshape(lon.shape).astype(np.float64)
        out[(touched.reshape(lon.shape) > 1e-6) | outside] = np.nan
        return out


@dataclass(frozen=True)
class TileMosaic(ElevationGrid):
    """Adjacent XYZ tiles pasted into one array, in Web Mercator pixels."""

    zoom: int = 0
    tile_x0: int = 0
    tile_y0: int = 0

    def pixel_of(self, lon, lat):
        fx, fy = _world_fraction(lon, lat)
        world = TILE_SIZE * 2 ** self.zoom
        # Pixel centres sit at i + 0.5, so the fractional index is shifted by half.
        cols = fx * world - self.tile_x0 * TILE_SIZE - 0.5
        rows = fy * world - self.tile_y0 * TILE_SIZE - 0.5
        return rows, cols


@dataclass(frozen=True)
class GeographicGrid(ElevationGrid):
    """A lon/lat raster, as OpenTopography's ASCII grid output describes one."""

    west: float = 0.0
    north: float = 0.0
    """lon/lat of the *centre* of pixel (0, 0)."""
    cell_deg: float = 1.0

    def pixel_of(self, lon, lat):
        return (self.north - lat) / self.cell_deg, (lon - self.west) / self.cell_deg


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class ElevationProvider(ABC):
    """The seam an elevation source plugs into."""

    name: str = "elevation"
    description: str = ""

    @abstractmethod
    def grid_for(
        self, bbox: tuple[float, float, float, float], resolution_m: float
    ) -> ElevationGrid:
        """Heights covering `bbox`, at about `resolution_m` or finer where the source allows."""

    @abstractmethod
    def native_resolution_m(self, lat: float) -> float:
        """The finest grid worth building from this source at this latitude."""


class TileProvider(ElevationProvider):
    """A source served as 256 px XYZ tiles. Subclasses say how to get and decode one."""

    def __init__(self, config: ElevationConfig | None = None) -> None:
        self.config = config or settings.elevation

    # ---- one tile ---- #
    @abstractmethod
    def tile(self, z: int, x: int, y: int) -> tuple[np.ndarray | None, str]:
        """(heights or None, where it came from). `None` means the tile is a hole."""

    # ---- zoom choice ---- #
    def zoom_for(self, lat: float, resolution_m: float) -> int:
        """The coarsest zoom whose pixels are at least as fine as the grid wanted.

        Coarsest, because every zoom level quadruples the tiles; at least as fine, so the
        grid never samples more coarsely than the tiles it is built from.
        """
        cfg = self.config
        base = EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / TILE_SIZE
        zoom = math.ceil(math.log2(base / max(resolution_m, 1e-3)) - 1e-9)
        return int(min(max(zoom, cfg.min_zoom), cfg.max_zoom))

    def native_resolution_m(self, lat: float) -> float:
        """The tile pixel at the maximum zoom, or the source's own posts if coarser.

        z13 is 17.8 m at 21 N, finer than SRTM's 30 m. The grid defaults to the tile pixel,
        since resampling bilinearly onto it costs little and keeps the divides where the
        tiles put them; the smoothing is scaled to the source instead.
        """
        return tile_resolution_m(self.config.max_zoom, lat)

    # ---- mosaic ---- #
    def tiles_for(self, bbox: tuple[float, float, float, float], zoom: int) -> TileMosaic:
        """Every tile covering `bbox` at `zoom`, pasted together.

        Clipping happens later, by sampling at the DEM's own cell centres. Never by
        rounding tile edges to the selection, which would move it (PLAN3 §11.3).
        """
        cfg = self.config
        x0, y0, x1, y1 = tile_range(bbox, zoom)
        wanted = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
        if len(wanted) > cfg.max_tiles:
            raise ElevationUnavailable(
                "aoi_too_large",
                f"The area needs {len(wanted)} elevation tiles at zoom {zoom}; the limit is "
                f"{cfg.max_tiles}.",
                "Draw a smaller area.",
            )

        mosaic = np.full(
            ((y1 - y0 + 1) * TILE_SIZE, (x1 - x0 + 1) * TILE_SIZE), np.nan, dtype=np.float32
        )
        origins: Counter = Counter()
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, cfg.tile_workers)) as pool:
            results = pool.map(lambda xy: self.tile(zoom, *xy), wanted)
            for (x, y), (heights, origin) in zip(wanted, results):
                origins[origin] += 1
                if heights is None:
                    continue
                r, c = (y - y0) * TILE_SIZE, (x - x0) * TILE_SIZE
                mosaic[r : r + TILE_SIZE, c : c + TILE_SIZE] = heights

        missing = origins.get("missing", 0)
        if missing == len(wanted):
            hint = (
                "Elevation tiles are fetched from the internet; check the service's outbound "
                "access, or try again."
                if cfg.network_enabled
                else "Network fetching is switched off (POND_ELEVATION_NETWORK_ENABLED=false) "
                "and this area is not in the tile cache."
            )
            raise ElevationUnavailable(
                "elevation_unavailable",
                f"None of the {len(wanted)} elevation tiles for this area could be read.",
                hint,
            )

        mid_lat = (bbox[1] + bbox[3]) / 2
        return TileMosaic(
            z=mosaic,
            source=self.description or self.name,
            resolution_m=tile_resolution_m(zoom, mid_lat),
            stats={
                "provider": self.name,
                "zoom": zoom,
                "tiles": len(wanted),
                "tile_sources": dict(origins),
                "fetch_ms": round((time.perf_counter() - started) * 1e3, 1),
            },
            zoom=zoom,
            tile_x0=x0,
            tile_y0=y0,
        )

    def grid_for(self, bbox, resolution_m):
        return self.tiles_for(bbox, self.zoom_for((bbox[1] + bbox[3]) / 2, resolution_m))


class TerrariumProvider(TileProvider):
    """AWS `elevation-tiles-prod`, terrarium encoding. Keyless; the primary source."""

    name = "terrarium"
    description = "AWS terrain tiles (terrarium), SRTM ~30 m over India"
    encoding = "terrarium"

    def fetch(self, z: int, x: int, y: int) -> bytes | None:
        """The PNG bytes, retried; `None` if it cannot be had. A 404 is not retried:
        a tile that does not exist will not exist on the second try either."""
        cfg = self.config
        if not cfg.network_enabled:
            return None
        url = cfg.terrarium_url.format(z=z, x=x, y=y)
        request = urllib.request.Request(url, headers={"User-Agent": cfg.user_agent})
        for attempt in range(cfg.tile_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=cfg.tile_timeout_s) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                if exc.code in (403, 404):
                    return None
            except (urllib.error.URLError, OSError, ValueError):
                pass
            # The lab uplink resets intermittently; a short backoff clears most of them.
            time.sleep(0.25 * (attempt + 1))
        return None

    @staticmethod
    def decode(data: bytes) -> np.ndarray | None:
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:  # noqa: BLE001, a corrupt tile is a hole, whatever the reason
            return None
        if image.size != (TILE_SIZE, TILE_SIZE):
            return None
        return decode_terrarium(np.asarray(image))

    def tile(self, z, x, y):
        data = self.fetch(z, x, y)
        if data is None:
            return None, "missing"
        heights = self.decode(data)
        return (heights, "network") if heights is not None else (None, "missing")


class CachedProvider(TileProvider):
    """Memory, then the committed seed tiles, then the disk cache, then the source.

    Content-addressed by `(source, z, x, y)`: a tile's name is its identity, and since
    tiles never change, a cached one is never stale. Stores the PNG bytes rather than the
    decoded floats, because the PNG is a quarter of the size and decoding is cheap.
    """

    def __init__(self, inner: TerrariumProvider, config: ElevationConfig | None = None) -> None:
        super().__init__(config or inner.config)
        self.inner = inner
        self.name = inner.name
        self.description = inner.description
        self._memory: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _resolve(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else REPO_ROOT / p

    def _path(self, root: str, z: int, x: int, y: int) -> Path:
        return self._resolve(root) / self.inner.encoding / str(z) / str(x) / f"{y}.png"

    def _remember(self, key, heights: np.ndarray) -> None:
        with self._lock:
            self._memory[key] = heights
            self._memory.move_to_end(key)
            while len(self._memory) > self.config.memory_cache_tiles:
                self._memory.popitem(last=False)

    def tile(self, z, x, y):
        key = (z, x, y)
        with self._lock:
            if key in self._memory:
                self._memory.move_to_end(key)
                return self._memory[key], "memory"

        for origin, root in (("seed", self.config.seed_dir), ("disk", self.config.cache_dir)):
            path = self._path(root, z, x, y)
            if path.is_file():
                heights = self.inner.decode(path.read_bytes())
                if heights is not None:
                    self._remember(key, heights)
                    return heights, origin

        data = self.inner.fetch(z, x, y)
        if data is None:
            return None, "missing"
        heights = self.inner.decode(data)
        if heights is None:
            return None, "missing"
        self._store(self._path(self.config.cache_dir, z, x, y), data)
        self._remember(key, heights)
        return heights, "network"

    @staticmethod
    def _store(path: Path, data: bytes) -> None:
        """Write atomically, so a worker killed mid-write leaves no half tile behind for
        the next request to decode as garbage. A read-only disk is not an error: the
        tile is simply fetched again next time."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
            partial.write_bytes(data)
            os.replace(partial, path)
        except OSError:
            pass

    def clear_memory(self) -> None:
        with self._lock:
            self._memory.clear()


class OpenTopographyProvider(ElevationProvider):
    """OpenTopography's global DEM API, opt-in with `POND_ELEVATION_OPENTOPO_KEY`.

    Asked for `AAIGrid`, which is plain ASCII and needs nothing but numpy to read. Returns
    the source's own one arc-second posts, so there is no tile pyramid and no zoom.
    """

    name = "opentopography"

    def __init__(self, config: ElevationConfig | None = None) -> None:
        self.config = config or settings.elevation
        if not self.config.opentopo_key:
            raise ElevationUnavailable(
                "elevation_unavailable",
                "OpenTopography needs an API key and none is configured.",
                "Set POND_ELEVATION_OPENTOPO_KEY, or use the keyless terrarium provider.",
            )
        self.description = f"OpenTopography {self.config.opentopo_dem_type}"

    def native_resolution_m(self, lat: float) -> float:
        return self.config.source_resolution_m

    def grid_for(self, bbox, resolution_m):
        cfg = self.config
        if not cfg.network_enabled:
            raise ElevationUnavailable(
                "elevation_unavailable", "Network fetching is switched off.", ""
            )
        min_lon, min_lat, max_lon, max_lat = bbox
        query = urllib.parse.urlencode(
            {
                "demtype": cfg.opentopo_dem_type,
                "south": min_lat, "north": max_lat, "west": min_lon, "east": max_lon,
                "outputFormat": "AAIGrid",
                "API_Key": cfg.opentopo_key,
            }
        )
        request = urllib.request.Request(
            f"{cfg.opentopo_url}?{query}", headers={"User-Agent": cfg.user_agent}
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=cfg.tile_timeout_s * 4) as response:
                text = response.read().decode("ascii", "replace")
        except Exception as exc:  # noqa: BLE001, any failure is the same failure
            raise ElevationUnavailable(
                "elevation_unavailable",
                f"OpenTopography could not be reached: {exc}",
                "Unset POND_ELEVATION_PROVIDER to use the keyless terrarium tiles.",
            ) from exc
        grid = parse_aaigrid(text, source=self.description)
        return GeographicGrid(
            z=grid.z,
            source=grid.source,
            resolution_m=cfg.source_resolution_m,
            stats={
                "provider": self.name,
                "tiles": 1,
                "tile_sources": {"network": 1},
                "fetch_ms": round((time.perf_counter() - started) * 1e3, 1),
            },
            west=grid.west,
            north=grid.north,
            cell_deg=grid.cell_deg,
        )


def parse_aaigrid(text: str, *, source: str = "AAIGrid") -> GeographicGrid:
    """ESRI ASCII grid -> `GeographicGrid`. Six header lines, then rows north to south."""
    lines = text.strip().splitlines()
    header: dict[str, float] = {}
    body_start = 0
    for index, line in enumerate(lines[:7]):
        parts = line.split()
        if len(parts) == 2 and parts[0].replace("_", "").isalpha():
            header[parts[0].lower()] = float(parts[1])
            body_start = index + 1
    try:
        ncols, nrows = int(header["ncols"]), int(header["nrows"])
        cell = header["cellsize"]
        west = header.get("xllcenter", header.get("xllcorner", 0.0) + cell / 2)
        south = header.get("yllcenter", header.get("yllcorner", 0.0) + cell / 2)
    except KeyError as exc:
        raise ElevationUnavailable(
            "elevation_unavailable", f"The elevation grid has no {exc} header.", ""
        ) from exc
    z = np.loadtxt(io.StringIO("\n".join(lines[body_start:])), dtype=np.float32, ndmin=2)
    if z.shape != (nrows, ncols):
        raise ElevationUnavailable(
            "elevation_unavailable",
            f"The elevation grid is {z.shape}, and its header says {(nrows, ncols)}.",
            "",
        )
    nodata = header.get("nodata_value")
    if nodata is not None:
        z[z == nodata] = np.nan
    return GeographicGrid(
        z=z, source=source, resolution_m=0.0,
        west=west, north=south + (nrows - 1) * cell, cell_deg=cell,
    )


# --------------------------------------------------------------------------- #
_DEFAULT: ElevationProvider | None = None
_DEFAULT_LOCK = threading.Lock()


def provider_for(config: ElevationConfig) -> ElevationProvider:
    if config.provider == "opentopography":
        return OpenTopographyProvider(config)
    return CachedProvider(TerrariumProvider(config), config)


def default_provider(config: ElevationConfig | None = None) -> ElevationProvider:
    """One process-wide provider, so its memory cache outlives a request. A config other
    than the service's own gets a provider of its own, so a test or a validation sweep
    that changes the zoom is not silently served the default one."""
    global _DEFAULT
    if config is not None and config != settings.elevation:
        return provider_for(config)
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = provider_for(settings.elevation)
        return _DEFAULT
