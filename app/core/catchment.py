"""Delineating the catchment of a point, and saying how much to trust it.

The catchment is every cell whose water eventually flows through the outlet (PLAN §2
Step 6). D8 gives each cell one receiver, so the flow field is a *forest*: following the
pointers backwards from the outlet collects its basin, and collects every cell exactly
once.

Three things around that turn a number into a trustworthy number.

**Snapping.** A pond location named in lon/lat almost never lands exactly on the routed
channel -- and the channel itself moves between grid resolutions, by around 90 m on this
sheet. So the outlet is snapped to the largest accumulation nearby, with the search
radius scaled to the contour spacing rather than fixed (PLAN §11.4).

**Edge contact.** A catchment that runs off the mapped area has been clipped, and its
area is a lower bound rather than an answer. The test has to be against *no-data or the
array border*, not the border alone: 2.5% of this grid lies outside the contour hull, so
a basin can leave the mapped area without ever reaching row 0 (PLAN §11.3). Testing only
the border wrongly labelled the 395 ha basin complete.

**The ensemble.** The same site is delineated on three independent grids. Agreement means
the answer is a property of the terrain; disagreement means it is a property of the grid,
and the site is flagged low-confidence rather than reported as fact. This is what
rejected site 4 of the sample, at 14.1 +/- 15.4 ha.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.config import CatchmentConfig, settings
from app.core.dem_builder import DEM, ContourSurface
from app.core.terrain import D8TerrainEngine, FlowField, TerrainEngine

__all__ = [
    "DonorIndex",
    "Catchment",
    "EnsembleResult",
    "CatchmentDelineator",
    "CatchmentEnsemble",
    "upstream_mask",
    "catchment_area",
    "edge_contact_ratio",
]


# --------------------------------------------------------------------------- #
# Reverse-D8 index
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DonorIndex:
    """Who drains *into* each cell, in CSR form.

    `receivers` answers "where does this cell send its water"; delineation needs the
    opposite. Built once per flow field with a single argsort, then every catchment costs
    time proportional to its own size rather than to the whole map -- which matters in
    Phase 6, where a dozen candidate sites are delineated against one flow field.
    """

    donors: np.ndarray
    """(n_donors,) flat cell indices, grouped by the cell they drain into."""

    offsets: np.ndarray
    """(n_cells + 1,) donors of cell `c` are `donors[offsets[c]:offsets[c + 1]]`."""

    @classmethod
    def build(cls, receivers: np.ndarray, n_cells: int) -> "DonorIndex":
        contributing = np.flatnonzero(receivers >= 0)
        targets = receivers[contributing]
        # Stable so the donor order is reproducible; the traversal does not depend on it,
        # but a deterministic order makes any future debugging repeatable.
        ordering = np.argsort(targets, kind="stable")
        donors = contributing[ordering]
        offsets = np.searchsorted(targets[ordering], np.arange(n_cells + 1))
        return cls(donors=donors, offsets=offsets.astype(np.int64))


def upstream_mask(
    donor_index: DonorIndex, outlet: int, shape: tuple[int, int]
) -> np.ndarray:
    """Every cell draining through `outlet`, as a (ny, nx) boolean mask.

    An explicit stack, not recursion: a catchment on this sheet is 150,000 cells deep in
    the worst case and Python's recursion limit is 1,000.

    No visited-set is needed. Each cell has exactly one receiver, so the flow graph is a
    forest and each cell appears in the donor list exactly once -- the traversal cannot
    reach the same cell twice.
    """
    donors = donor_index.donors.tolist()
    offsets = donor_index.offsets.tolist()

    collected: list[int] = []
    stack = [outlet]
    while stack:
        cell = stack.pop()
        collected.append(cell)
        stack.extend(donors[offsets[cell] : offsets[cell + 1]])

    mask = np.zeros(shape[0] * shape[1], dtype=bool)
    mask[collected] = True
    return mask.reshape(shape)


# --------------------------------------------------------------------------- #
# Measurements on a mask
# --------------------------------------------------------------------------- #
def catchment_area(mask: np.ndarray, row_areas: np.ndarray) -> float:
    """Latitude-weighted ground area of a cell mask, in m^2 (PLAN §2 Step 6)."""
    return float((mask.sum(axis=1) * row_areas).sum())


def edge_contact_ratio(mask: np.ndarray, nodata: np.ndarray) -> float:
    """Fraction of the catchment's perimeter that lies against the edge of valid data.

    Counted as boundary *edges* rather than cells: for each cell in the mask, each of its
    four sides that faces out of the mask is one unit of perimeter. That makes the measure
    a true length ratio and keeps it comparable between grid resolutions.

    A side is "edge contact" when what lies beyond it is no-data *or* off the array. Both
    mean the same thing -- the catchment continues into ground the contours never
    described -- and testing only the array border is the mistake that reported the 395 ha
    basin as complete (PLAN §11.3).
    """
    if not mask.any():
        return 0.0

    ny, nx = mask.shape
    # Pad with "outside", which counts as invalid on both tests.
    padded_mask = np.zeros((ny + 2, nx + 2), dtype=bool)
    padded_mask[1:-1, 1:-1] = mask
    padded_invalid = np.ones((ny + 2, nx + 2), dtype=bool)
    padded_invalid[1:-1, 1:-1] = nodata

    perimeter = 0
    against_edge = 0
    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        window = (slice(1 + d_row, 1 + d_row + ny), slice(1 + d_col, 1 + d_col + nx))
        outward = mask & ~padded_mask[window]
        perimeter += int(outward.sum())
        against_edge += int((outward & padded_invalid[window]).sum())

    return against_edge / perimeter if perimeter else 0.0


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Catchment:
    """One delineated catchment on one grid."""

    mask: np.ndarray
    outlet_rc: tuple[int, int]
    outlet_lonlat: tuple[float, float]
    requested_lonlat: tuple[float, float]
    snap_distance_m: float

    cell_count: int
    area_m2: float
    edge_contact: float
    is_lower_bound: bool
    """True when enough of the perimeter is against the edge of the data that the area
    should be read as a floor, not a measurement."""

    relief_m: float
    """Basin relief: the highest ground in the catchment minus the outlet elevation.

    Measured against the *outlet*, not the catchment minimum. A contour-derived DEM has
    unfilled pits scattered through it, some of them below the outlet, so max-minus-min
    reports a drop the water never has -- about 4 m too much on every site of the sample
    sheet.
    """

    longest_flow_path_m: float
    """Distance from the hydraulically most remote cell to the outlet, along the flow
    path -- the `L` in Phase 7's time of concentration."""

    flow_path_cell: int
    """Flat index of that most remote cell. Phase 8 walks the receivers down from it to
    draw the flow path; keeping the cell here means the traversal is not repeated."""

    flow_path_relief_m: float
    """Elevation of that most remote cell, minus the outlet's.

    Kirpich's `H`, and a different number from `relief_m`: the most distant point is
    rarely the highest one. On the sample's largest basin they are 19.7 m and 27.0 m, a
    36% difference in the slope term, so the distinction is worth keeping.
    """

    accumulation_cells: float
    resolution_m: float

    @property
    def area_ha(self) -> float:
        return self.area_m2 / 1e4


@dataclass(frozen=True)
class EnsembleResult:
    """The same site delineated on several grids (PLAN §3 Test C)."""

    per_grid: tuple[Catchment, ...]
    mean_area_m2: float
    std_area_m2: float
    confidence: str
    """high | medium | low, from the coefficient of variation across the grids."""

    @property
    def mean_area_ha(self) -> float:
        return self.mean_area_m2 / 1e4

    @property
    def std_area_ha(self) -> float:
        return self.std_area_m2 / 1e4

    @property
    def coefficient_of_variation(self) -> float:
        return self.std_area_m2 / self.mean_area_m2 if self.mean_area_m2 else 0.0

    @property
    def resolutions_m(self) -> tuple[float, ...]:
        return tuple(c.resolution_m for c in self.per_grid)


# --------------------------------------------------------------------------- #
# Delineation on one grid
# --------------------------------------------------------------------------- #
class CatchmentDelineator:
    """Delineates catchments against a single flow field.

    Holds the donor index, so delineating many candidate sites against one grid -- which
    is exactly what Phase 6 does -- pays the reverse-index cost once.
    """

    def __init__(self, flow: FlowField, *, config: CatchmentConfig | None = None) -> None:
        self.flow = flow
        self.dem: DEM = flow.dem
        self.config = config or settings.catchment
        self.donor_index = DonorIndex.build(flow.receivers, flow.filled.size)
        self._row_areas = self.dem.row_cell_areas

    # ------------------------------------------------------------------ #
    @property
    def snap_radius_m(self) -> float:
        """Scaled to the contour spacing, not fixed.

        The routed channel shifts by around 90 m between the ensemble's grids, so a fixed
        30 m radius snaps to a different stream on each and the ensemble reports
        disagreement that is really just a mis-snap (PLAN §11.4).
        """
        return (
            self.dem.meta.mean_contour_spacing_m
            * self.config.snap_radius_spacing_multiple
        )

    def snap_outlet(self, lon: float, lat: float) -> tuple[tuple[int, int], float]:
        """Move a requested point to the largest accumulation within the snap radius.

        Returns the cell and how far it moved, so the caller can see -- and report -- that
        the answer is not for the point that was asked about.
        """
        row, col = self.dem.index_of_lonlat(lon, lat)
        ny, nx = self.dem.shape
        if not (0 <= row < ny and 0 <= col < nx):
            raise ValueError(
                f"({lon}, {lat}) lies outside the mapped area "
                f"{self.dem.meta.shape} at {self.dem.resolution_m:g} m."
            )

        radius = max(1, int(round(self.snap_radius_m / self.dem.resolution_m)))
        # Clamp every bound. An unclamped window silently wraps a negative index to the
        # far side of the array, and argmax then returns a cell on the opposite edge of
        # the map (PLAN §11.9).
        rows = slice(max(0, row - radius), min(ny, row + radius + 1))
        cols = slice(max(0, col - radius), min(nx, col + radius + 1))

        window = np.where(
            self.dem.nodata[rows, cols], -1.0, self.flow.accumulation[rows, cols]
        )
        if window.max() < 0:
            raise ValueError(
                f"({lon}, {lat}) has no valid data within {self.snap_radius_m:.0f} m."
            )

        local = np.unravel_index(int(np.argmax(window)), window.shape)
        snapped = (rows.start + int(local[0]), cols.start + int(local[1]))

        x0, y0 = self.dem.xy_of(row, col)
        x1, y1 = self.dem.xy_of(*snapped)
        return snapped, float(math.hypot(float(x1 - x0), float(y1 - y0)))

    # ------------------------------------------------------------------ #
    def delineate(self, lon: float, lat: float, *, snap: bool = True) -> Catchment:
        """Full catchment of the point, with its diagnostics."""
        if snap:
            (row, col), distance = self.snap_outlet(lon, lat)
        else:
            row, col = self.dem.index_of_lonlat(lon, lat)
            distance = 0.0
            if self.dem.nodata[row, col]:
                raise ValueError(f"({lon}, {lat}) is outside the mapped area.")

        return self.delineate_cell(row, col, requested=(lon, lat), snap_distance_m=distance)

    def delineate_cell(
        self,
        row: int,
        col: int,
        *,
        requested: tuple[float, float] | None = None,
        snap_distance_m: float = 0.0,
    ) -> Catchment:
        """Delineate from a grid cell directly -- the entry point Phase 6 uses, since its
        candidates come from the accumulation grid rather than from a coordinate."""
        shape = self.dem.shape
        outlet = row * shape[1] + col
        mask = upstream_mask(self.donor_index, outlet, shape)

        lon, lat = (float(v) for v in self.dem.lonlat_of(row, col))
        contact = edge_contact_ratio(mask, self.dem.nodata)
        outlet_z = float(self.dem.z[row, col])
        path_length, remote_cell = self.longest_flow_path(outlet)

        return Catchment(
            mask=mask,
            outlet_rc=(row, col),
            outlet_lonlat=(lon, lat),
            requested_lonlat=requested or (lon, lat),
            snap_distance_m=snap_distance_m,
            cell_count=int(mask.sum()),
            area_m2=catchment_area(mask, self._row_areas),
            edge_contact=contact,
            is_lower_bound=contact > self.config.edge_contact_warn_fraction,
            relief_m=float(np.nanmax(self.dem.z[mask]) - outlet_z),
            longest_flow_path_m=path_length,
            flow_path_cell=int(remote_cell),
            flow_path_relief_m=float(self.dem.z.ravel()[remote_cell] - outlet_z),
            accumulation_cells=float(self.flow.accumulation[row, col]),
            resolution_m=self.dem.resolution_m,
        )

    # ------------------------------------------------------------------ #
    def longest_flow_path(self, outlet: int) -> tuple[float, int]:
        """(length in metres, flat index) of the hydraulically most remote cell.

        Measured on the way *up* the tree from the outlet, so it costs the size of this
        catchment rather than the size of the map: each cell's distance is its receiver's
        distance plus one step, and diagonal steps count sqrt(2) cells.

        The cell is returned as well as the distance because Kirpich needs the elevation
        drop along this same path, not the basin's overall relief.
        """
        nx = self.dem.shape[1]
        res = self.dem.resolution_m
        diagonal = res * math.sqrt(2.0)

        donors = self.donor_index.donors.tolist()
        offsets = self.donor_index.offsets.tolist()

        longest, remote = 0.0, outlet
        stack = [(outlet, 0.0)]
        while stack:
            cell, distance = stack.pop()
            if distance > longest:
                longest, remote = distance, cell
            row, col = divmod(cell, nx)
            for donor in donors[offsets[cell] : offsets[cell + 1]]:
                d_row, d_col = divmod(donor, nx)
                step = diagonal if (d_row != row and d_col != col) else res
                stack.append((donor, distance + step))
        return longest, remote


# --------------------------------------------------------------------------- #
# The resolution ensemble
# --------------------------------------------------------------------------- #
class CatchmentEnsemble:
    """One delineator per grid resolution, so a site can be cross-checked (Test C).

    The flow fields are built once, in the constructor, and reused for every site. Phase 6
    ranks many candidates; rebuilding three flow fields per candidate would make the
    ensemble unaffordable rather than merely slow.
    """

    def __init__(
        self,
        surface: ContourSurface,
        *,
        resolutions_m: tuple[float, ...] | None = None,
        engine: TerrainEngine | None = None,
        config: CatchmentConfig | None = None,
    ) -> None:
        self.config = config or settings.catchment
        self.engine = engine or D8TerrainEngine()
        self.resolutions_m = resolutions_m or self.config.ensemble_resolutions_m
        self.delineators = tuple(
            CatchmentDelineator(
                self.engine.analyse(surface.sample(resolution)), config=self.config
            )
            for resolution in self.resolutions_m
        )

    @property
    def primary(self) -> CatchmentDelineator:
        """The first grid. Its catchment is the one reported; the others are the error
        bar, not competing answers."""
        return self.delineators[0]

    def delineate(self, lon: float, lat: float) -> EnsembleResult:
        """Delineate the same point on every grid and summarise the spread.

        Each grid snaps independently, on purpose: the routed channel is in a slightly
        different place on each, and forcing them all to one grid's cell would measure the
        snap rather than the terrain.
        """
        results = tuple(d.delineate(lon, lat) for d in self.delineators)
        areas = np.array([c.area_m2 for c in results], dtype=np.float64)
        mean = float(areas.mean())
        # Population standard deviation: these three grids are the whole ensemble, not a
        # sample drawn from a larger one.
        std = float(areas.std())
        return EnsembleResult(
            per_grid=results,
            mean_area_m2=mean,
            std_area_m2=std,
            confidence=self.classify(mean, std),
        )

    def classify(self, mean: float, std: float) -> str:
        """Confidence from the coefficient of variation.

        A relative measure, so a 20 ha spread means something different on a 400 ha basin
        than on a 30 ha one. Site 4 of the sample scores 14.1 +/- 15.4 ha -- a CV above 1
        -- and is rejected on this rule alone.
        """
        if mean <= 0:
            return "low"
        cv = std / mean
        if cv <= self.config.confidence_high_cv:
            return "high"
        if cv <= self.config.confidence_medium_cv:
            return "medium"
        return "low"
