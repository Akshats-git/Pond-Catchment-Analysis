"""Pit filling, D8 flow direction, flow accumulation and slope.

The three steps that turn a surface into a drainage network (PLAN §2 steps 4-5).

**Fill the pits.** A DEM built from contours is full of one-cell hollows that exist only
because the interpolation is imperfect. Water routed on it gets stuck in them. Priority-
flood (Barnes, Lehman & Mulla 2014) floods inward from the edge of the data: pop the
lowest cell seen so far, and raise each of its unvisited neighbours to at least that
elevation plus a tiny epsilon. Every cell ends up with a strictly descending path to the
map edge, which also makes the resulting flow graph provably acyclic.

**D8 flow direction.** Each cell gives all its water to the steepest of its eight
neighbours, where steepness is distance-weighted:

    S_i = (z_c - z_i) / d_i        d_i = res (sides), res * sqrt(2) (diagonals)

Dividing by the diagonal distance is what stops the routing from drifting diagonally: a
diagonal neighbour is 41% further away, so an equal drop across it is a shallower slope.

**Flow accumulation.** Because every receiver is *strictly* lower than its donor, sorting
cells by descending elevation is a topological order of the flow graph. One pass down
that order, adding each cell's running total to its receiver, gives the number of cells
draining through every cell. The streams light up as ridges of large values.

`TerrainEngine` is an interface (PLAN §5, §8). The HLD names pysheds; we implement the
same D8 methodology directly so the service stays deployable on a free tier, and swapping
in pysheds later means writing one more subclass.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from heapq import heapify, heappop, heappush

import numpy as np
from scipy.ndimage import binary_dilation

from app.config import TerrainConfig, settings
from app.core.dem_builder import DEM

__all__ = [
    "FlowField",
    "FlowMetadata",
    "TerrainEngine",
    "D8TerrainEngine",
    "NEIGHBOUR_OFFSETS",
    "analyse_terrain",
]

NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
"""The eight D8 neighbours, in (row, col) deltas. Order fixes tie-breaking only."""

_MOORE = np.ones((3, 3), dtype=bool)


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FlowMetadata:
    """Diagnostics about the routing, for the API response and for spotting trouble."""

    cells_raised: int
    """How many cells the pit filling lifted. A large fraction means the DEM is noisy."""

    max_fill_m: float
    """Deepest single pit removed. Should be well under the contour interval; more than
    that means real terrain is being flooded, not an artefact."""

    fill_volume_m3: float
    """Total earth the filling notionally added: sum of (raise x cell area). A physical
    handle on how much of the surface is artefact."""

    cells_raised_over_interval: int
    """Cells lifted by more than one contour interval. These are the ones worth doubting:
    below the interval the DEM never claimed that much precision, but above it the fill is
    flooding something the contours actually described."""

    outlet_count: int
    """Cells with no downhill neighbour: the points where water leaves the mapped area.
    Every valid cell drains to exactly one of these, which is what the Phase 5 mass
    balance checks."""

    max_accumulation: float
    valid_cells: int
    resolution_m: float
    timings_ms: dict[str, float]


@dataclass(frozen=True)
class FlowField:
    """The drainage structure of a DEM."""

    filled: np.ndarray
    """(ny, nx) float64. Depression-filled elevations, NaN at no-data."""

    receivers: np.ndarray
    """(ny * nx,) int64. Flat index of the cell each cell drains into.

    -1 marks an outlet (water leaves the mapped area here) or a no-data cell. Flat rather
    than a direction code because Phase 4 walks these pointers backwards, and an index is
    what that traversal actually needs.
    """

    accumulation: np.ndarray
    """(ny, nx) float64. Number of cells draining through each cell, itself included."""

    order: np.ndarray
    """(n_valid,) int64. Valid cells by descending filled elevation. A topological
    order of the flow graph, reused by Phase 4 rather than recomputed."""

    slope: np.ndarray
    """(ny, nx) float64. Rise over run, dimensionless. NaN at no-data."""

    dem: DEM
    meta: FlowMetadata

    @property
    def shape(self) -> tuple[int, int]:
        return self.filled.shape  # type: ignore[return-value]

    def terminal_outlets(self) -> np.ndarray:
        """(ny, nx) int64. The flat index of the outlet each cell ultimately drains to.

        Every valid cell reaches exactly one outlet, so this partitions the map into
        basins. Phase 4 uses it to suppress a whole sub-basin after picking a site, and
        Phase 5's mass balance sums the basins and checks the total against the mapped
        area.

        Resolved by walking the topological order *backwards*: in ascending elevation
        every cell's receiver has already been resolved, so one pass suffices with no
        recursion and no repeated path walking.
        """
        receivers = self.receivers.tolist()
        terminal = [-1] * len(receivers)
        for cell in self.order.tolist()[::-1]:
            target = receivers[cell]
            terminal[cell] = cell if target < 0 else terminal[target]
        return np.asarray(terminal, dtype=np.int64).reshape(self.shape)

    def accumulated_area(self, row: int, col: int) -> float:
        """Upstream area draining through one cell, in m^2.

        Accumulation counts *cells*, so this uses the latitude-weighted area of the row
        the cell sits in. Across one sheet the row-to-row variation is under 0.01%, so
        the approximation sits far below the spread between grids. Phase 4 sums the real
        per-cell areas over the catchment mask when the number has to be exact.
        """
        return float(self.accumulation[row, col]) * float(
            self.dem.row_cell_areas[row]
        )


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class TerrainEngine(ABC):
    """Turns a DEM into a flow field.

    The seam PLAN §8 promises: the pipeline depends on this, not on how the routing is
    done, so a pysheds- or GPU-backed implementation drops in without touching Phase 4
    onwards.
    """

    @abstractmethod
    def fill_depressions(self, dem: DEM) -> np.ndarray: ...

    @abstractmethod
    def d8_receivers(self, filled: np.ndarray, nodata: np.ndarray, resolution_m: float) -> np.ndarray: ...

    @abstractmethod
    def flow_accumulation(self, receivers: np.ndarray, order: np.ndarray, shape: tuple[int, int], weights: np.ndarray | None = None) -> np.ndarray: ...

    @abstractmethod
    def slope(self, dem: DEM) -> np.ndarray: ...

    @abstractmethod
    def analyse(self, dem: DEM) -> FlowField: ...


# --------------------------------------------------------------------------- #
# Implementation
# --------------------------------------------------------------------------- #
class D8TerrainEngine(TerrainEngine):
    """Priority-flood + D8, in numpy and a little hand-written Python."""

    def __init__(self, config: TerrainConfig | None = None) -> None:
        self.config = config or settings.terrain

    # ------------------------------------------------------------------ #
    def fill_depressions(self, dem: DEM) -> np.ndarray:
        """Priority-flood with an epsilon gradient (Barnes et al. 2014, Algorithm 2).

        Seeded from the array border **and** every valid cell touching no-data. Seeding
        only the border would be wrong here: 2.5% of this grid falls outside the contour
        hull, so a basin can drain off the mapped area through an interior hole without
        ever reaching row 0.

        The heap loop runs on Python lists rather than numpy arrays on purpose. Indexing
        a numpy array with a scalar builds a boxed numpy scalar every time; over the ~2.7
        million neighbour visits this loop makes on the sample sheet, that dominates the
        runtime. Lists of plain floats are several times faster.
        """
        eps = self.config.fill_epsilon_m
        ny, nx = dem.shape
        width = nx + 2

        # Pad by one cell of no-data so the neighbour offsets need no bounds checks.
        padded = np.full((ny + 2, width), np.inf, dtype=np.float64)
        padded[1:-1, 1:-1] = np.where(dem.nodata, np.inf, dem.z)

        blocked = np.ones((ny + 2, width), dtype=bool)
        blocked[1:-1, 1:-1] = dem.nodata

        seeds = np.flatnonzero((binary_dilation(blocked, _MOORE) & ~blocked).ravel())
        if seeds.size == 0:
            raise ValueError("The DEM has no cell adjacent to its edge or to no-data.")

        z = padded.ravel().tolist()
        closed = blocked.ravel().tolist()
        offsets = (-width - 1, -width, -width + 1, -1, 1, width - 1, width, width + 1)

        heap = []
        for s in seeds.tolist():
            closed[s] = True
            heap.append((z[s], s))
        heapify(heap)

        push, pop = heappush, heappop
        while heap:
            z_c, c = pop(heap)
            floor = z_c + eps
            for offset in offsets:
                n = c + offset
                if closed[n]:
                    continue
                closed[n] = True
                z_n = z[n]
                if z_n < floor:
                    z_n = floor
                    z[n] = floor
                push(heap, (z_n, n))

        filled = np.asarray(z, dtype=np.float64).reshape(ny + 2, width)[1:-1, 1:-1]
        return np.where(dem.nodata, np.nan, filled)

    # ------------------------------------------------------------------ #
    def d8_receivers(
        self, filled: np.ndarray, nodata: np.ndarray, resolution_m: float
    ) -> np.ndarray:
        """Steepest-descent receiver for every cell, as a flat index. -1 = outlet.

        Vectorised across the eight neighbour shifts rather than looped per cell: eight
        whole-array comparisons instead of millions of Python iterations.
        """
        ny, nx = filled.shape
        valid = ~nodata

        padded_z = np.full((ny + 2, nx + 2), np.nan, dtype=np.float64)
        padded_z[1:-1, 1:-1] = filled
        padded_valid = np.zeros((ny + 2, nx + 2), dtype=bool)
        padded_valid[1:-1, 1:-1] = valid

        rows = np.arange(ny, dtype=np.int64)[:, None]
        cols = np.arange(nx, dtype=np.int64)[None, :]

        # Strictly positive: a receiver must be genuinely lower, so a cell with nowhere
        # downhill keeps -1 and becomes an outlet rather than pointing at a peer.
        best_slope = np.zeros((ny, nx), dtype=np.float64)
        best_index = np.full((ny, nx), -1, dtype=np.int64)

        for d_row, d_col in NEIGHBOUR_OFFSETS:
            window = (slice(1 + d_row, 1 + d_row + ny), slice(1 + d_col, 1 + d_col + nx))
            neighbour_z = padded_z[window]
            neighbour_ok = padded_valid[window] & valid

            distance = resolution_m * (
                self.config.diagonal_distance_factor if d_row and d_col else 1.0
            )
            with np.errstate(invalid="ignore"):
                gradient = (filled - neighbour_z) / distance

            better = neighbour_ok & (gradient > best_slope)
            best_slope = np.where(better, gradient, best_slope)
            best_index = np.where(
                better, (rows + d_row) * nx + (cols + d_col), best_index
            )

        best_index[nodata] = -1
        return best_index.ravel()

    # ------------------------------------------------------------------ #
    @staticmethod
    def topological_order(filled: np.ndarray, nodata: np.ndarray) -> np.ndarray:
        """Valid cells by descending filled elevation.

        Every receiver is strictly lower than its donor, so this ordering guarantees each
        cell is processed before the cell it drains into, and, along the way, that the
        flow graph has no cycles at all.
        """
        flat_valid = np.flatnonzero((~nodata).ravel())
        heights = filled.ravel()[flat_valid]
        return flat_valid[np.argsort(-heights, kind="stable")]

    def flow_accumulation(
        self,
        receivers: np.ndarray,
        order: np.ndarray,
        shape: tuple[int, int],
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Cells draining through each cell, itself included.

        `weights` lets a later phase accumulate something other than a cell count, such
        as per-cell rainfall, without changing the traversal.
        """
        total = np.zeros(shape[0] * shape[1], dtype=np.float64)
        if weights is None:
            total[order] = 1.0
        else:
            flat = np.asarray(weights, dtype=np.float64).ravel()
            total[order] = flat[order]

        # Sequential by nature: a cell's total must be complete before it is handed on.
        totals = total.tolist()
        receiver_list = receivers.tolist()
        for cell in order.tolist():
            target = receiver_list[cell]
            if target >= 0:
                totals[target] += totals[cell]

        return np.asarray(totals, dtype=np.float64).reshape(shape)

    # ------------------------------------------------------------------ #
    def slope(self, dem: DEM) -> np.ndarray:
        """Horn's 3x3 gradient, as rise over run.

        The same estimator GDAL and ArcGIS use: a weighted central difference that is far
        less noisy than a two-cell difference, which matters because Phase 6 rejects sites
        on a 3% slope and single-cell noise would reject good ground at random.

        No-data neighbours are replaced by the centre cell, the standard edge convention:
        it assumes the ground continues level rather than inventing a cliff at the data
        boundary.
        """
        ny, nx = dem.shape
        res = dem.resolution_m
        centre = np.where(dem.nodata, np.nan, dem.z)

        padded = np.empty((ny + 2, nx + 2), dtype=np.float64)
        padded[:] = np.nan
        padded[1:-1, 1:-1] = centre

        def neighbour(d_row: int, d_col: int) -> np.ndarray:
            block = padded[1 + d_row : 1 + d_row + ny, 1 + d_col : 1 + d_col + nx]
            return np.where(np.isfinite(block), block, centre)

        nw, n, ne = neighbour(-1, -1), neighbour(-1, 0), neighbour(-1, 1)
        w, e = neighbour(0, -1), neighbour(0, 1)
        sw, s, se = neighbour(1, -1), neighbour(1, 0), neighbour(1, 1)

        # Row 0 is the south edge, so +row is north; the y difference is signed
        # accordingly. Only the magnitude is used, but keeping the sign honest means the
        # same two arrays can serve an aspect calculation later.
        d_dx = ((ne + 2 * e + se) - (nw + 2 * w + sw)) / (8 * res)
        d_dy = ((nw + 2 * n + ne) - (sw + 2 * s + se)) / (8 * res)

        with np.errstate(invalid="ignore"):
            result = np.hypot(d_dx, d_dy)
        result[dem.nodata] = np.nan
        return result

    # ------------------------------------------------------------------ #
    def analyse(self, dem: DEM) -> FlowField:
        """Run the whole routing chain and collect its diagnostics."""
        timings: dict[str, float] = {}

        start = time.perf_counter()
        filled = self.fill_depressions(dem)
        timings["fill"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        receivers = self.d8_receivers(filled, dem.nodata, dem.resolution_m)
        timings["receivers"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        order = self.topological_order(filled, dem.nodata)
        accumulation = self.flow_accumulation(receivers, order, dem.shape)
        timings["accumulation"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        slope = self.slope(dem)
        timings["slope"] = (time.perf_counter() - start) * 1000

        valid = dem.valid
        raised = np.zeros(dem.shape, dtype=np.float64)
        raised[valid] = filled[valid] - dem.z[valid]
        was_raised = raised > 0
        interval = dem.meta.contour_interval_m

        return FlowField(
            filled=filled,
            receivers=receivers,
            accumulation=accumulation,
            order=order,
            slope=slope,
            dem=dem,
            meta=FlowMetadata(
                cells_raised=int(was_raised.sum()),
                max_fill_m=float(raised.max()),
                # Row-weighted, because a cell's ground area depends on its latitude.
                fill_volume_m3=float((raised.sum(axis=1) * dem.row_cell_areas).sum()),
                cells_raised_over_interval=(
                    int((raised > interval).sum()) if interval else 0
                ),
                outlet_count=int(((receivers < 0) & valid.ravel()).sum()),
                max_accumulation=float(accumulation[valid].max()),
                valid_cells=int(valid.sum()),
                resolution_m=dem.resolution_m,
                timings_ms=timings,
            ),
        )


# --------------------------------------------------------------------------- #
def analyse_terrain(dem: DEM, *, config: TerrainConfig | None = None) -> FlowField:
    """Convenience wrapper around the default engine."""
    return D8TerrainEngine(config).analyse(dem)
