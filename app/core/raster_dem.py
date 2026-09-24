"""A drawn area -> the same `DEM` the contour path builds.

`ContourSurface` turns a contour sheet into a `DEM`. `RasterSurface` turns a selection on
the map into one, from elevation tiles. Everything downstream (flow, catchment, siting,
runoff, GeoJSON) takes a `DEM` and never learns which of the two made it. That is the seam
PLAN §8 promised and PLAN3 §2 tests, and this module is the only new thing on the far
side of it.

Three decisions here decide whether the answer is right, not just whether it runs.

**Analyse a buffer, recommend inside the selection** (PLAN3 §5.3). A drawn rectangle is an
arbitrary cut across the landscape. If the DEM stopped at it, every catchment touching the
line would be truncated and every volume understated, silently and plausibly. So the DEM
covers the selection grown by `aoi_buffer` on every side, flow is routed across all of it,
and `inside_mask` is what the siting step uses to keep the pond itself inside the drawn
line. The polygon is a *siting* mask, never a DEM mask, for the same reason: masking the
heights to the polygon would reintroduce exactly the truncation the buffer exists to stop.

**The resolution adapts so a big selection cannot OOM a 512 MB box.**

    resolution = max(native tile resolution, sqrt(analysed area / cell_budget))

A million cells is the envelope the Phase 2 sheet already proved fits.

**Whole-metre heights are a staircase.** SRTM stores integer metres, so gentle farmland is
a stack of flat terraces with one-metre risers, and on a flat D8 has no downhill to pick.
It is the contour path's stair-step problem again, and it gets the contour path's answer:
the same normalised, NaN-aware Gaussian, scaled to the source's 30 m posts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.config import ElevationConfig, settings
from app.core.dem_builder import DEM, ContourSurface, DEMBuildError, DEMMetadata, row_cell_areas
from app.core.projection import EquirectangularENU, Projection
from app.providers.elevation import ElevationProvider, ElevationUnavailable, default_provider

__all__ = ["AreaError", "AreaOfInterest", "RasterSurface"]


class AreaError(Exception):
    """A selection that cannot be analysed as drawn. Same `(code, detail, hint)` shape as
    every other structured error in the service."""

    def __init__(self, code: str, detail: str, hint: str = "") -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.hint = hint


_MAX_VERTICES = 2000


def _local_xy(ring: np.ndarray) -> np.ndarray:
    """A quick equirectangular frame about the ring's own centre, for areas and buffers."""
    lon0 = float((ring[:, 0].min() + ring[:, 0].max()) / 2)
    lat0 = float((ring[:, 1].min() + ring[:, 1].max()) / 2)
    return EquirectangularENU(lon0=lon0, lat0=lat0).forward(ring)


def _shoelace(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


# --------------------------------------------------------------------------- #
# The selection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AreaOfInterest:
    """What the user drew: a rectangle or a polygon, as one exterior ring in lon/lat."""

    ring: np.ndarray
    """(N, 2) lon/lat, open (the first vertex is not repeated at the end)."""

    kind: str
    """`bbox` or `polygon`."""

    # ---- construction ---- #
    @classmethod
    def from_bbox(cls, bbox) -> "AreaOfInterest":
        try:
            min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
        except (TypeError, ValueError) as exc:
            raise AreaError(
                "invalid_aoi",
                "bbox must be four numbers: [min_lon, min_lat, max_lon, max_lat].",
                "For example [81.2814, 21.2398, 81.3126, 21.2636].",
            ) from exc
        if not (min_lon < max_lon and min_lat < max_lat):
            raise AreaError(
                "invalid_aoi",
                f"bbox {list(bbox)} is empty or inverted: the minimum has to come first.",
                "Send [min_lon, min_lat, max_lon, max_lat], west-south-east-north.",
            )
        ring = np.array(
            [[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat]],
            dtype=np.float64,
        )
        return cls._checked(ring, "bbox")

    @classmethod
    def from_polygon(cls, coordinates) -> "AreaOfInterest":
        try:
            ring = np.asarray(coordinates, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise AreaError(
                "invalid_aoi", "polygon must be a list of [lon, lat] pairs.", ""
            ) from exc
        if ring.ndim != 2 or ring.shape[1] < 2:
            raise AreaError(
                "invalid_aoi",
                "polygon must be a list of [lon, lat] pairs.",
                "For example [[81.28, 21.24], [81.31, 21.24], [81.30, 21.26]].",
            )
        ring = ring[:, :2]
        if len(ring) > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        # Consecutive duplicates add nothing and break the edge arithmetic.
        keep = np.ones(len(ring), dtype=bool)
        keep[1:] = np.any(np.diff(ring, axis=0) != 0, axis=1)
        ring = ring[keep]
        if len(ring) < 3:
            raise AreaError(
                "invalid_aoi",
                "A polygon needs at least three distinct corners.",
                "Draw at least a triangle, or send a bbox.",
            )
        if len(ring) > _MAX_VERTICES:
            raise AreaError(
                "invalid_aoi",
                f"The polygon has {len(ring)} corners; the limit is {_MAX_VERTICES}.",
                "Simplify the outline; the analysis grid is tens of metres anyway.",
            )
        return cls._checked(ring, "polygon")

    @classmethod
    def from_geojson(cls, geometry: dict) -> "AreaOfInterest":
        """A GeoJSON Polygon, or a Feature wrapping one. The exterior ring only: a hole
        in a selection would still have water running through it, so it is not a hole
        in the analysis."""
        if not isinstance(geometry, dict):
            raise AreaError("invalid_aoi", "geometry must be a GeoJSON object.", "")
        if geometry.get("type") == "Feature":
            geometry = geometry.get("geometry") or {}
        if geometry.get("type") != "Polygon" or not geometry.get("coordinates"):
            raise AreaError(
                "invalid_aoi",
                f"geometry must be a GeoJSON Polygon; got {geometry.get('type')!r}.",
                "Send a Polygon, or use bbox / polygon instead.",
            )
        return cls.from_polygon(geometry["coordinates"][0])

    @classmethod
    def _checked(cls, ring: np.ndarray, kind: str) -> "AreaOfInterest":
        if not np.isfinite(ring).all():
            raise AreaError("invalid_aoi", "The selection has a non-numeric coordinate.", "")
        if (np.abs(ring[:, 0]) > 180).any() or (np.abs(ring[:, 1]) > 85).any():
            raise AreaError(
                "invalid_aoi",
                "The selection has a coordinate outside lon -180..180, lat -85..85.",
                "Coordinates are [lon, lat], longitude first.",
            )
        # Counter-clockwise, so the area comes out positive whatever order it was drawn in.
        if _shoelace(_local_xy(ring)) < 0:
            ring = ring[::-1].copy()
        return cls(ring=ring, kind=kind)

    # ---- geometry ---- #
    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            float(self.ring[:, 0].min()), float(self.ring[:, 1].min()),
            float(self.ring[:, 0].max()), float(self.ring[:, 1].max()),
        )

    @property
    def area_m2(self) -> float:
        return abs(_shoelace(_local_xy(self.ring)))

    @property
    def centre(self) -> tuple[float, float]:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return ((min_lon + max_lon) / 2, (min_lat + max_lat) / 2)

    def check_size(self, config: ElevationConfig | None = None) -> None:
        """Refuse a selection too small to hold a catchment or too big for the host.

        A named 422 rather than an attempt, for the reason Phase 2 established with
        `ensemble_unavailable`: an OOM takes every request on the worker down with it.
        """
        cfg = config or settings.elevation
        area_km2 = self.area_m2 / 1e6
        if area_km2 > cfg.max_aoi_area_km2:
            raise AreaError(
                "aoi_too_large",
                f"The selection is {area_km2:,.1f} km2; the limit is "
                f"{cfg.max_aoi_area_km2:g} km2.",
                "Draw a smaller area around the village. Catchments that cross the line "
                "are still counted in full.",
            )
        if self.area_m2 / 1e4 < cfg.min_aoi_area_ha:
            raise AreaError(
                "aoi_too_small",
                f"The selection is {self.area_m2 / 1e4:.2f} ha; the smallest area worth "
                f"analysing at a 30 m source resolution is {cfg.min_aoi_area_ha:g} ha.",
                "Draw a larger area.",
            )

    def buffered_bbox(
        self, fraction: float, min_buffer_m: float
    ) -> tuple[float, float, float, float]:
        """The bbox grown by `fraction` of its span on each side, and never by less than
        `min_buffer_m`. This is the ground the DEM covers."""
        min_lon, min_lat, max_lon, max_lat = self.bbox
        lat0 = math.radians((min_lat + max_lat) / 2)
        m_per_deg_lat = settings.projection.metres_per_degree_lat
        m_per_deg_lon = settings.projection.metres_per_degree_lon_equator * math.cos(lat0)
        pad_x = max((max_lon - min_lon) * fraction, min_buffer_m / m_per_deg_lon)
        pad_y = max((max_lat - min_lat) * fraction, min_buffer_m / m_per_deg_lat)
        return (min_lon - pad_x, min_lat - pad_y, max_lon + pad_x, max_lat + pad_y)

    def contains(self, lon: float, lat: float) -> bool:
        """Even-odd point in polygon, for one point (a pour point)."""
        x, y = lon, lat
        inside = False
        ring = self.ring
        j = len(ring) - 1
        for i in range(len(ring)):
            xi, yi = ring[i]
            xj, yj = ring[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        return inside

    def to_geojson(self, precision: int = 6) -> dict:
        ring = [[round(float(lon), precision), round(float(lat), precision)] for lon, lat in self.ring]
        return {"type": "Polygon", "coordinates": [ring + [ring[0]]]}


# --------------------------------------------------------------------------- #
# The surface
# --------------------------------------------------------------------------- #
class RasterSurface:
    """Elevation tiles over a buffered selection, sampleable onto a grid at any resolution.

    Plays the part `ContourSurface` plays on the contour path, down to the `sample()`
    signature, so `CatchmentEnsemble` can cross-check a site on several grids without
    knowing where the heights came from.
    """

    def __init__(
        self,
        aoi: AreaOfInterest,
        *,
        provider: ElevationProvider | None = None,
        config: ElevationConfig | None = None,
        projection: Projection | None = None,
    ) -> None:
        self.aoi = aoi
        self.config = config or settings.elevation
        self.provider = provider or default_provider(self.config)
        aoi.check_size(self.config)

        self.analysed_bbox = aoi.buffered_bbox(self.config.aoi_buffer, self.config.min_buffer_m)
        min_lon, min_lat, max_lon, max_lat = self.analysed_bbox
        self.projection = projection or EquirectangularENU(
            lon0=(min_lon + max_lon) / 2, lat0=(min_lat + max_lat) / 2
        )
        corners = self.projection.forward(
            np.array([[min_lon, min_lat], [max_lon, max_lat]], dtype=np.float64)
        )
        self.bounds_xy = (
            float(corners[0, 0]), float(corners[0, 1]),
            float(corners[1, 0]), float(corners[1, 1]),
        )
        self._grids: dict = {}
        self.fetches: list[dict] = []
        """One entry per distinct tile fetch this surface made, for the response."""

    # ---- derived parameters ---- #
    @property
    def analysed_area_m2(self) -> float:
        min_x, min_y, max_x, max_y = self.bounds_xy
        return (max_x - min_x) * (max_y - min_y)

    @property
    def native_resolution_m(self) -> float:
        return self.provider.native_resolution_m(self.aoi.centre[1])

    @property
    def budget_resolution_m(self) -> float:
        """The finest resolution that keeps the grid inside the cell budget."""
        return math.sqrt(self.analysed_area_m2 / self.config.cell_budget)

    @property
    def auto_resolution_m(self) -> float:
        return max(self.native_resolution_m, self.budget_resolution_m, self.config.min_resolution_m)

    @property
    def mean_spacing_m(self) -> float:
        """The spacing between independent height samples: the source's posts.

        Recorded in the DEM metadata's `mean_contour_spacing_m`, because that is the
        role the field plays downstream. The outlet snap radius, the relative-elevation
        window and the smallest resolvable pond all scale with how far apart the real
        measurements are, and on this path that is 30 m, not the grid's 17.8 m.
        """
        return max(self.config.source_resolution_m, self.budget_resolution_m)

    @property
    def smoothing_sigma_m(self) -> float:
        return self.config.smoothing_sigma_m

    def grid_shape(self, resolution_m: float) -> tuple[int, int]:
        min_x, min_y, max_x, max_y = self.bounds_xy
        return (
            int(np.floor((max_y - min_y) / resolution_m)) + 1,
            int(np.floor((max_x - min_x) / resolution_m)) + 1,
        )

    def _resolve_resolution(self, requested: float | None) -> tuple[float, str, list[str]]:
        cfg = self.config
        warnings: list[str] = []
        floor = max(self.budget_resolution_m, cfg.min_resolution_m)
        if requested is None:
            resolution = self.auto_resolution_m
            source = "native" if resolution <= self.native_resolution_m + 1e-9 else "coarsened"
            if source == "coarsened":
                warnings.append(
                    f"The grid is {resolution:.1f} m rather than the tiles' "
                    f"{self.native_resolution_m:.1f} m, to stay under "
                    f"{cfg.cell_budget:,} cells over the analysed area."
                )
            return resolution, source, warnings

        if not cfg.min_resolution_m <= requested <= cfg.max_resolution_m:
            raise DEMBuildError(
                "invalid_resolution",
                f"Requested grid resolution {requested:g} m is outside the allowed range "
                f"{cfg.min_resolution_m:g}-{cfg.max_resolution_m:g} m for a drawn area.",
                f"Omit it to use {self.auto_resolution_m:.1f} m, set by the elevation tiles.",
            )
        if requested < floor:
            warnings.append(
                f"Requested {requested:g} m would need more than {cfg.cell_budget:,} cells; "
                f"the grid is {floor:.1f} m instead."
            )
            return floor, "coarsened", warnings
        if requested < cfg.source_resolution_m / 2:
            warnings.append(
                f"{requested:g} m is much finer than the ~{cfg.source_resolution_m:g} m "
                "elevation source; the extra cells interpolate detail it does not contain."
            )
        return float(requested), "requested", warnings

    # ---- heights ---- #
    def _grid(self, resolution_m: float):
        """The provider's grid for this resolution, fetched once per zoom it maps to."""
        grid = self.provider.grid_for(self.analysed_bbox, resolution_m)
        key = (grid.stats.get("provider"), grid.stats.get("zoom"), grid.z.shape)
        if key not in self._grids:
            self._grids[key] = grid
            self.fetches.append(dict(grid.stats))
        return self._grids[key]

    def sample(self, resolution_m: float | None = None, *, smooth: bool = True) -> DEM:
        """Tiles onto a square metric grid over the analysed area. Returns a `DEM`
        indistinguishable in kind from one `ContourSurface.sample` builds."""
        cfg = self.config
        resolution, source, warnings = self._resolve_resolution(resolution_m)
        grid = self._grid(resolution)

        min_x, min_y, _, _ = self.bounds_xy
        ny, nx = self.grid_shape(resolution)
        xs = min_x + np.arange(nx) * resolution
        ys = min_y + np.arange(ny) * resolution
        gx, gy = np.meshgrid(xs, ys)
        lonlat = self.projection.inverse(np.stack([gx, gy], axis=-1))
        del gx, gy
        raw = grid.sample(lonlat[..., 0], lonlat[..., 1])
        del lonlat
        nodata = ~np.isfinite(raw)

        nodata_fraction = float(nodata.mean())
        if nodata_fraction > cfg.max_nodata_fraction:
            raise ElevationUnavailable(
                "elevation_unavailable",
                f"{nodata_fraction:.0%} of the analysed area has no elevation data "
                f"(limit {cfg.max_nodata_fraction:.0%}).",
                "Some elevation tiles could not be fetched. Try again, or draw the area "
                "away from the sea.",
            )
        if nodata_fraction > 0:
            warnings.append(
                f"{nodata_fraction:.1%} of the analysed area has no elevation data; "
                "flow stops at those cells as it would at the edge of the map."
            )

        sigma_m = self.smoothing_sigma_m if smooth else 0.0
        z = ContourSurface._smooth(raw, nodata, sigma_m / resolution) if smooth else raw.copy()

        valid = ~nodata
        lo, hi = float(raw[valid].min()), float(raw[valid].max())
        smoothed_lo, smoothed_hi = float(z[valid].min()), float(z[valid].max())
        if not (lo - 1e-6 <= smoothed_lo and smoothed_hi <= hi + 1e-6):
            raise DEMBuildError(
                "smoothing_out_of_range",
                f"Smoothed elevations span {smoothed_lo:.2f}-{smoothed_hi:.2f} m, outside "
                f"the sampled range {lo:.2f}-{hi:.2f} m.",
                "The NaN-aware Gaussian is not normalised correctly.",
            )
        shifts = np.abs(z[valid] - raw[valid]) if smooth else np.zeros(1)

        origin = (min_x, min_y)
        areas = row_cell_areas(self.projection, origin, resolution, ny)
        mapped_area = float((valid.sum(axis=1) * areas).sum())

        return DEM(
            z=z,
            nodata=nodata,
            raw_z=raw,
            resolution_m=resolution,
            origin_xy=origin,
            projection=self.projection,
            meta=DEMMetadata(
                resolution_m=resolution,
                resolution_source=source,
                smoothing_sigma_m=sigma_m,
                mean_contour_spacing_m=self.mean_spacing_m,
                total_contour_length_m=0.0,
                hull_area_m2=self.analysed_area_m2,
                mapped_area_m2=mapped_area,
                nodata_fraction=nodata_fraction,
                elevation_range=(smoothed_lo, smoothed_hi),
                max_smoothing_shift_m=float(shifts.max()),
                smoothing_shift_p999_m=float(np.percentile(shifts, 99.9)),
                cells_over_interval=0,
                # No contour interval on this path. Zero switches off the checks that
                # compare against one, rather than inventing an interval to pass them.
                contour_interval_m=0.0,
                shape=(ny, nx),
                warnings=tuple(warnings),
            ),
        )

    # ---- the selection on the grid ---- #
    def inside_mask(self, dem: DEM) -> np.ndarray:
        """(ny, nx) bool: cells whose centre lies inside what the user drew.

        A scanline fill at cell centres: each row's crossings with every edge, sorted and
        paired, even-odd. Rows times edges rather than cells times edges, so a million-cell
        grid and a 2,000-corner polygon cost a couple of million operations, not billions.
        Exact at centres, where Pillow's polygon fill includes both boundaries and
        overstates a rectangle by a cell on each side.
        """
        ny, nx = dem.shape
        xy = dem.projection.forward(self.aoi.ring)
        x0, y0 = dem.origin_xy
        cols = (xy[:, 0] - x0) / dem.resolution_m
        rows = (xy[:, 1] - y0) / dem.resolution_m
        ax, ay = cols, rows
        bx, by = np.roll(cols, -1), np.roll(rows, -1)

        mask = np.zeros((ny, nx), dtype=bool)
        lo = max(0, int(np.floor(rows.min())))
        hi = min(ny - 1, int(np.ceil(rows.max())))
        for r in range(lo, hi + 1):
            # Half-open in y, so a vertex exactly on the scanline is counted once.
            crosses = (ay > r) != (by > r)
            if not crosses.any():
                continue
            t = (r - ay[crosses]) / (by[crosses] - ay[crosses])
            xs = np.sort(ax[crosses] + t * (bx[crosses] - ax[crosses]))
            for start, end in zip(xs[0::2], xs[1::2]):
                c0 = max(0, int(np.ceil(start)))
                c1 = min(nx - 1, int(np.ceil(end)) - 1)
                if c1 >= c0:
                    mask[r, c0 : c1 + 1] = True
        return mask & dem.valid
