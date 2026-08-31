"""Contour lines -> a square metric grid of elevations.

Flow algorithms need a raster, so the gaps between contour lines have to be filled in
(PLAN §2 Step 2). Two things here are not obvious and both matter more than the
interpolation itself.

**The resolution is derived, not chosen.** A grid finer than the contours cannot invent
detail, and a grid coarser than them throws detail away. The mean spacing between
contour lines follows from an identity. Parallel lines of total length `L` spaced `w`
apart fill an area `A = L * w`, so

    mean contour spacing = mapped area / total contour length
    grid resolution      = spacing / 4

Four cells across the typical gap resolves the interpolated slope without pretending to
know more than the source does.

**Interpolating between contours builds a staircase, not a hillside** (PLAN §2 Step 3).
Linear interpolation makes the ground between two contour lines a flat band, and on a
flat band D8 has no downhill direction to pick, so water runs along the step instead of
down it. In the analytic validation this cost up to 12.79% of the catchment area. A
Gaussian of sigma = spacing / 8 turns the staircase back into a slope, moving the surface
by less than one contour interval.

The smoothing must be NaN-aware in the *normalised* form (PLAN §11.2): fill invalid cells
with zero, smooth, and divide by the smoothed validity mask. The obvious alternative is
to fill with the mean and divide by the valid-cell weight, and it inflated cells near the data
edge into a 357 m peak on a map whose true maximum is 298 m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull, Delaunay, QhullError

from app.config import DEMConfig, settings
from app.core.kml_parser import ContourSet
from app.core.projection import Projection, projection_for

__all__ = [
    "DEMBuildError", "DEM", "DEMMetadata", "ContourSurface", "build_dem",
    "contour_metrics", "row_cell_areas",
]


class DEMBuildError(Exception):
    """A contour set that cannot be turned into a usable grid."""

    def __init__(self, code: str, detail: str, hint: str = "") -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.hint = hint


def row_cell_areas(
    projection: Projection, origin_xy: tuple[float, float], resolution_m: float, ny: int
) -> np.ndarray:
    """(ny,) true ground area of one cell in each grid row, in m^2.

    `res^2 * cos(lat) / cos(lat0)`, which is the latitude weighting of PLAN §2 Step 6. It is a
    per-row quantity because the correction depends only on latitude. Free-standing
    rather than a method so the mapped area can be totalled while the DEM is still being
    assembled.
    """
    x0, y0 = origin_xy
    y = y0 + np.arange(ny) * resolution_m
    lat = projection.inverse(np.stack([np.full(ny, x0), y], axis=-1))[:, 1]
    return resolution_m ** 2 * np.asarray(projection.area_factor(lat))


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DEMMetadata:
    """Everything about *how* the grid was built, for the API response and the report."""

    resolution_m: float
    resolution_source: str
    """One of auto, requested or coarsened. `coarsened` means the memory ceiling bit."""

    smoothing_sigma_m: float
    mean_contour_spacing_m: float
    total_contour_length_m: float
    hull_area_m2: float
    """Analytic area of the convex hull of the contour vertices, in projected metres.
    Resolution-independent, so the resolution can be derived from it before any grid
    exists."""

    mapped_area_m2: float
    """Sum of the true ground areas of the valid cells. This is the number every
    downstream area is measured against, including the stream threshold and the Phase 5
    mass balance, so it has to come from the same cells the flow router will walk."""

    nodata_fraction: float
    elevation_range: tuple[float, float]

    max_smoothing_shift_m: float
    """Largest distance the smoothing moved any single cell."""

    smoothing_shift_p999_m: float
    """99.9th percentile of that shift, which is the honest summary.

    The maximum is set by a handful of cells on the convex hull, where Delaunay bridges
    two distant contour lines with a long sliver triangle and the raw surface steps by
    metres between adjacent cells. Smoothing across such a step is the correct behaviour,
    so the maximum says more about the triangulation than about the smoothing."""

    cells_over_interval: int
    """How many cells the smoothing moved by more than one contour interval. Reported
    rather than summarised away: on the sample it is 13 of 622,227."""

    contour_interval_m: float
    shape: tuple[int, int]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DEM:
    """A square metric grid of elevations, with the frame needed to locate a cell.

    Row 0 is the *south* edge: `y = y0 + row * resolution`. Keeping y increasing with the
    row index means no sign flips anywhere in the flow routing; the display layer flips
    it once, at the end.
    """

    z: np.ndarray
    """(ny, nx) float64. Metres, NaN outside the mapped area."""

    nodata: np.ndarray
    """(ny, nx) bool. True where `z` is NaN."""

    raw_z: np.ndarray
    """The unsmoothed interpolation, kept so the smoothing can be audited and so Phase 5
    can run the pipeline with smoothing off and show that it is worse."""

    resolution_m: float
    origin_xy: tuple[float, float]
    """Projected metres of the centre of cell (0, 0)."""

    projection: Projection
    meta: DEMMetadata

    # ---------------- geometry ---------------- #
    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape  # type: ignore[return-value]

    @property
    def valid(self) -> np.ndarray:
        return ~self.nodata

    def xy_of(self, row: np.ndarray | int, col: np.ndarray | int) -> tuple:
        """Grid indices -> projected metres (cell centres)."""
        x0, y0 = self.origin_xy
        return (x0 + np.asarray(col) * self.resolution_m,
                y0 + np.asarray(row) * self.resolution_m)

    def index_of(self, x: float, y: float) -> tuple[int, int]:
        """Projected metres -> (row, col), nearest cell centre.

        The division stays *inside* `round`. Rounding first and dividing after returns a
        neighbouring cell, and a catchment delineated from the wrong cell looks perfectly
        plausible. It is simply somebody else's basin (PLAN §11.8).
        """
        x0, y0 = self.origin_xy
        return (int(round((y - y0) / self.resolution_m)),
                int(round((x - x0) / self.resolution_m)))

    def lonlat_of(self, row: np.ndarray | int, col: np.ndarray | int) -> np.ndarray:
        x, y = self.xy_of(row, col)
        return self.projection.inverse(np.stack(np.broadcast_arrays(x, y), axis=-1))

    def index_of_lonlat(self, lon: float, lat: float) -> tuple[int, int]:
        xy = self.projection.forward(np.asarray([lon, lat], dtype=np.float64))
        return self.index_of(float(xy[0]), float(xy[1]))

    def contains(self, row: int, col: int) -> bool:
        ny, nx = self.shape
        return 0 <= row < ny and 0 <= col < nx

    # ---------------- areas ---------------- #
    @property
    def row_cell_areas(self) -> np.ndarray:
        """(ny,) true ground area of one cell in each row, in m^2."""
        return row_cell_areas(
            self.projection, self.origin_xy, self.resolution_m, self.shape[0]
        )

    def area_of(self, mask: np.ndarray) -> float:
        """Latitude-weighted ground area of a boolean cell mask, in m^2."""
        return float((mask.sum(axis=1) * self.row_cell_areas).sum())


# --------------------------------------------------------------------------- #
# Contour geometry metrics
# --------------------------------------------------------------------------- #
def contour_metrics(xy: np.ndarray, line_starts: np.ndarray) -> tuple[float, float]:
    """(total contour length, convex-hull area) in projected metres.

    Length is summed *per line*: the flat point array runs one contour straight into the
    next, so the segments that jump between lines have to be dropped. Leaving them in
    triples the total on the sample sheet (1,900 km against 568 km) and would shrink the
    derived resolution by the same factor.
    """
    if len(line_starts) < 2:
        raise DEMBuildError("no_geometry", "No contour lines to measure.", "")

    steps = np.diff(xy, axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    interior = np.ones(len(lengths), dtype=bool)
    interior[line_starts[1:-1] - 1] = False
    total_length = float(lengths[interior].sum())

    try:
        hull_area = float(ConvexHull(xy).volume)  # `volume` is area in 2D
    except QhullError as exc:
        raise DEMBuildError(
            "degenerate_geometry",
            f"The contour vertices are collinear or degenerate: {exc}",
            "The file may contain a single straight contour or duplicate geometry.",
        ) from exc

    if total_length <= 0.0 or hull_area <= 0.0:
        raise DEMBuildError(
            "degenerate_geometry",
            "The contour set has zero length or zero extent.",
            "Check that the coordinates are longitude/latitude in degrees.",
        )
    return total_length, hull_area


# --------------------------------------------------------------------------- #
# The surface
# --------------------------------------------------------------------------- #
class ContourSurface:
    """A triangulated contour set that can be sampled onto a grid at any resolution.

    The Delaunay triangulation of 159,113 vertices costs about a second and does not
    depend on the grid, so it is built once and reused. Phase 4's resolution ensemble
    samples the same surface three times (PLAN §3 Test C); rebuilding the triangulation
    for each would triple the cost of every request for no benefit.
    """

    def __init__(
        self,
        contours: ContourSet,
        *,
        projection: Projection | None = None,
        config: DEMConfig | None = None,
    ) -> None:
        self.contours = contours
        self.config = config or settings.dem
        self.projection = projection or projection_for(contours.points)
        self.xy = self.projection.forward(contours.points)
        self.z = contours.elevations

        self.total_length_m, self.hull_area_m2 = contour_metrics(
            self.xy, contours.line_starts
        )
        self.mean_spacing_m = self.hull_area_m2 / self.total_length_m

        try:
            self._triangulation = Delaunay(self.xy)
        except QhullError as exc:
            raise DEMBuildError(
                "degenerate_geometry",
                f"The contour vertices could not be triangulated: {exc}",
                "The contours may be collinear or all identical.",
            ) from exc
        self._interpolator = LinearNDInterpolator(self._triangulation, self.z)

    # ---------------- derived parameters ---------------- #
    @property
    def auto_resolution_m(self) -> float:
        """Grid resolution implied by the data, before clamping."""
        return self.mean_spacing_m / self.config.resolution_divisor

    @property
    def smoothing_sigma_m(self) -> float:
        """Gaussian sigma implied by the data. Tied to the contour spacing, not to the
        grid: the staircase it removes is an artefact of the *contours*, so its width is
        set by them and must not change when the grid does."""
        return self.mean_spacing_m / self.config.smoothing_sigma_divisor

    @property
    def bounds_xy(self) -> tuple[float, float, float, float]:
        return (
            float(self.xy[:, 0].min()), float(self.xy[:, 1].min()),
            float(self.xy[:, 0].max()), float(self.xy[:, 1].max()),
        )

    def grid_shape(self, resolution_m: float) -> tuple[int, int]:
        """(ny, nx) the grid at this resolution would have. One formula, used both to
        enforce the cell ceiling and to build the grid, so the two cannot disagree."""
        min_x, min_y, max_x, max_y = self.bounds_xy
        return (
            int(np.floor((max_y - min_y) / resolution_m)) + 1,
            int(np.floor((max_x - min_x) / resolution_m)) + 1,
        )

    def _resolve_resolution(self, requested: float | None) -> tuple[float, str, list[str]]:
        cfg = self.config
        warnings: list[str] = []

        if requested is None:
            resolution = self.auto_resolution_m
            source = "auto"
            clamped = float(np.clip(resolution, cfg.min_resolution_m, cfg.max_resolution_m))
            if clamped != resolution:
                warnings.append(
                    f"Derived resolution {resolution:.2f} m clamped to {clamped:.2f} m "
                    f"(allowed range {cfg.min_resolution_m:g}-{cfg.max_resolution_m:g} m)."
                )
            resolution = clamped
        else:
            if not cfg.min_resolution_m <= requested <= cfg.max_resolution_m:
                raise DEMBuildError(
                    "invalid_resolution",
                    f"Requested grid resolution {requested:g} m is outside the allowed "
                    f"range {cfg.min_resolution_m:g}-{cfg.max_resolution_m:g} m.",
                    f"Omit the parameter to use the data-derived "
                    f"{self.auto_resolution_m:.2f} m.",
                )
            resolution = float(requested)
            source = "requested"
            if resolution < self.mean_spacing_m / cfg.resolution_divisor / 2:
                warnings.append(
                    f"Requested resolution {resolution:.2f} m is much finer than the "
                    f"{self.mean_spacing_m:.1f} m contour spacing; the extra cells "
                    "interpolate detail the contours do not contain."
                )

        # Memory ceiling. Coarsen rather than fail: a usable coarse answer with a warning
        # beats a 500 on a large sheet (PLAN Phase 11).
        ny, nx = self.grid_shape(resolution)
        if ny * nx > cfg.max_grid_cells:
            original = resolution
            # Iterate rather than scaling once. The exact cell count floors the extent
            # before adding the fencepost cell, so a single analytic step lands just over
            # the limit, such as 50,298 cells against a 50,000 cap on the sample sheet.
            for _ in range(8):
                if ny * nx <= cfg.max_grid_cells or resolution >= cfg.max_resolution_m:
                    break
                scale = float(np.sqrt(ny * nx / cfg.max_grid_cells))
                resolution = min(resolution * scale * 1.001, cfg.max_resolution_m)
                ny, nx = self.grid_shape(resolution)

            if ny * nx > cfg.max_grid_cells:
                raise DEMBuildError(
                    "sheet_too_large",
                    f"The sheet needs {ny * nx:,} cells even at the coarsest allowed "
                    f"resolution ({cfg.max_resolution_m:g} m), over the "
                    f"{cfg.max_grid_cells:,}-cell limit.",
                    "Clip the contour sheet to the area of interest and retry.",
                )
            warnings.append(
                f"Resolution coarsened from {original:.2f} m to {resolution:.2f} m to "
                f"stay under the {cfg.max_grid_cells:,}-cell limit."
            )
            source = "coarsened"

        return resolution, source, warnings

    # ---------------- sampling ---------------- #
    def sample(
        self, resolution_m: float | None = None, *, smooth: bool = True
    ) -> DEM:
        """Interpolate the contours onto a grid.

        `smooth=False` exists for the Phase 5 validation, which has to show that the
        smoothing earns its place rather than merely asserting it.
        """
        cfg = self.config
        resolution, source, warnings = self._resolve_resolution(resolution_m)

        min_x, min_y, _, _ = self.bounds_xy
        ny, nx = self.grid_shape(resolution)
        xs = min_x + np.arange(nx) * resolution
        ys = min_y + np.arange(ny) * resolution

        grid_x, grid_y = np.meshgrid(xs, ys)
        raw = self._interpolator(np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1))
        raw = raw.reshape(ny, nx).astype(np.float64, copy=False)
        nodata = ~np.isfinite(raw)

        if nodata.all():
            raise DEMBuildError(
                "empty_grid",
                "Every grid cell fell outside the contour hull.",
                "The contour geometry may be degenerate.",
            )

        sigma_m = self.smoothing_sigma_m if smooth else 0.0
        z = self._smooth(raw, nodata, sigma_m / resolution) if smooth else raw.copy()

        interval = self.contours.metadata.interval_m or 0.0
        if smooth:
            shifts = np.abs(z[~nodata] - raw[~nodata])
            shift = float(shifts.max())
            shift_p999 = float(np.percentile(shifts, 99.9))
            over_interval = int((shifts > interval).sum()) if interval else 0
        else:
            shift = shift_p999 = 0.0
            over_interval = 0

        valid_raw = raw[~nodata]
        lo, hi = float(valid_raw.min()), float(valid_raw.max())
        smoothed_lo, smoothed_hi = float(z[~nodata].min()), float(z[~nodata].max())

        # A normalised Gaussian is a weighted average of valid neighbours, so the result
        # cannot leave the input range. That makes this a real invariant rather than a
        # tolerance, and it is exactly the invariant the mean-fill bug violated, by
        # 59 m. Assert it, cheaply, every time.
        if not (lo - 1e-6 <= smoothed_lo and smoothed_hi <= hi + 1e-6):
            raise DEMBuildError(
                "smoothing_out_of_range",
                f"Smoothed elevations span {smoothed_lo:.2f}-{smoothed_hi:.2f} m, outside "
                f"the interpolated range {lo:.2f}-{hi:.2f} m.",
                "The NaN-aware Gaussian is not normalised correctly.",
            )

        # Tested on the 99.9th percentile, not the maximum. The intent of the check is
        # "the smoothing did not reshape the terrain", and over hundreds of thousands of
        # cells a maximum is dominated by hull slivers rather than by the smoothing.
        if interval and shift_p999 > interval * cfg.max_smoothing_shift_intervals:
            warnings.append(
                f"Smoothing moved the surface by more than the {interval:g} m contour "
                f"interval across {over_interval:,} cells (99.9th percentile "
                f"{shift_p999:.2f} m). The DEM may be over-smoothed."
            )

        origin = (min_x, min_y)
        areas = row_cell_areas(self.projection, origin, resolution, ny)
        mapped_area = float(((~nodata).sum(axis=1) * areas).sum())

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
                total_contour_length_m=self.total_length_m,
                hull_area_m2=self.hull_area_m2,
                mapped_area_m2=mapped_area,
                nodata_fraction=float(nodata.mean()),
                elevation_range=(smoothed_lo, smoothed_hi),
                max_smoothing_shift_m=shift,
                smoothing_shift_p999_m=shift_p999,
                cells_over_interval=over_interval,
                contour_interval_m=interval,
                shape=(ny, nx),
                warnings=tuple(warnings),
            ),
        )

    @staticmethod
    def _smooth(raw: np.ndarray, nodata: np.ndarray, sigma_cells: float) -> np.ndarray:
        """Normalised, NaN-aware Gaussian (PLAN §11.2).

        Smoothing a grid with NaNs in it poisons everything within sigma of the hole. The
        fix is to smooth the data and the validity mask separately and divide:

            num = G * (z filled with 0)      den = G * valid
            out = num / den

        Filling with 0 rather than the mean is what makes this correct. `den` is the
        fraction of each kernel that landed on real data, so dividing by it renormalises
        the kernel; the zeros contribute nothing to `num` and are exactly cancelled in
        `den`. Filling with the *mean* leaves a phantom contribution in `num` that `den`
        does not account for, which is what produced a 357 m peak on a 298 m map.
        """
        if sigma_cells <= 0.0:
            return raw.copy()

        valid = (~nodata).astype(np.float64)
        filled = np.where(nodata, 0.0, raw)
        num = gaussian_filter(filled, sigma_cells, mode="nearest")
        den = gaussian_filter(valid, sigma_cells, mode="nearest")

        out = np.full(raw.shape, np.nan)
        usable = den > settings.dem.nodata_weight_floor
        np.divide(num, den, out=out, where=usable)

        # Restore the original footprint. The division above produces finite values in a
        # collar of previously-invalid cells just outside the hull; keeping them would
        # extrapolate terrain the contours never described, and would silently enlarge
        # the mapped area that every downstream figure is measured against.
        out[nodata] = np.nan
        return out


# --------------------------------------------------------------------------- #
def build_dem(
    contours: ContourSet,
    *,
    resolution_m: float | None = None,
    projection: Projection | None = None,
    smooth: bool = True,
    config: DEMConfig | None = None,
) -> DEM:
    """Contours -> DEM in one call. Use `ContourSurface` directly to sample repeatedly."""
    surface = ContourSurface(contours, projection=projection, config=config)
    return surface.sample(resolution_m, smooth=smooth)
