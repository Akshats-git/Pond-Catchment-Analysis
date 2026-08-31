"""Choosing where the pond goes, catchment first.

A pond is only as good as the water that reaches it, so the search starts from the
drainage network rather than from the shape of the ground (PLAN §2, "Siting"):

1. **Stream network.** A cell is on a stream when at least 0.5% of the mapped area
   drains through it, which is 4.2 ha on the sample sheet. The threshold is absolute and
   derived from the input, never a percentile of flow accumulation: accumulation is so
   skewed that percentile ranking scored 0.7 ha hollows at 0.98 alongside a 320 ha
   valley (PLAN §11.6).
2. **Buildable ground.** Local slope under 3%, because steeper ground needs an
   embankment nobody will pay for, and at least 30 m inside the edge of the data, so
   neither the pond nor its catchment sits half off the map.
3. **Clear of the watercourse.** A channel that already drains more than 150 ha is a
   nala or a river, not a field drain, and a site must stand a pond depth above the one
   it runs into. See `clear_of_watercourse_mask` for why this rule exists and what it
   is measured against.
4. **Rank by catchment area.** The biggest basin first; ordering by flow accumulation is
   the same ordering, since accumulation counts exactly the cells the catchment mask
   collects.
5. **Suppress the whole catchment of each pick.** Not a square window. A square window
   returned five points strung along one stream at 391, 361, 215, 202 and 179 ha, every one
   of them nested inside the first (PLAN §11.7). Removing the entire upstream mask makes
   the alternatives independent sub-basins, which is the only way a list of five sites
   means five choices.

Descending accumulation plus catchment suppression cannot produce a nested pair, and the
argument is worth keeping: a cell downstream of a pick has *more* accumulation, so it was
considered earlier; had it been chosen, the pick would already have been suppressed as
part of its catchment.

Every site carries the five numbers that chose it: upstream area, slope, depression
depth, relative elevation and height above the watercourse. That is how the response can
say why and not just where. The ensemble is what makes the list honest: site 4 of the sample looks like an ordinary 35.7 ha basin
until three grids disagree about it by more than its own size, and it is returned flagged
rather than quietly recommended.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

from app.config import SitingConfig, settings
from app.core.catchment import (
    Catchment,
    CatchmentDelineator,
    CatchmentEnsemble,
    EnsembleResult,
)
from app.core.terrain import FlowField

__all__ = [
    "SitingError",
    "SiteScore",
    "PondSite",
    "SitingResult",
    "PondSiteSelector",
    "select_pond_sites",
]


class SitingError(Exception):
    """No ground on this sheet satisfies the siting rules."""

    def __init__(self, code: str, detail: str, hint: str = "") -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.hint = hint


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SiteScore:
    """The five measurements behind a pick, kept so the answer can be explained."""

    upstream_area_m2: float
    """Latitude-weighted area of the delineated catchment. This is the ranking key.

    Ranking is done on the accumulation grid, which is the same set of cells weighted by
    row rather than per cell; the two agree to under 0.01% on this sheet, and this is the
    exact one.
    """

    slope: float
    """Local slope at the outlet cell, rise over run (Horn's 3x3 gradient)."""

    depression_depth_m: float
    """How deep the natural hollow at this cell is: filled elevation minus surveyed.

    Zero on a channel, which is where catchment-first siting puts most sites. Those
    ponds are excavated, and Phase 7 computes their capacity from the target depth
    instead. Quantised to the contour interval, so read it as "about a metre", not as a
    survey.
    """

    relative_elevation_m: float
    """Outlet elevation minus the mean elevation of the ground around it.

    Negative means the site sits below its surroundings, which is what a pond wants:
    water arrives by gravity and the embankment has something to key into.
    """

    height_above_trunk_m: float
    """How far the site stands above the watercourse it drains into.

    `inf` on a sheet with no watercourse on it, which is the honest reading: there is
    nothing here for the pond to be too close to. See `clear_of_watercourse_mask`.
    """


@dataclass(frozen=True)
class PondSite:
    """One recommended location, with its catchment and its caveats."""

    rank: int
    catchment: Catchment
    score: SiteScore
    ensemble: EnsembleResult | None
    """The same site delineated on the other grids, when an ensemble was supplied."""

    confidence: str
    """high | medium | low from the ensemble spread, or `unassessed` without one."""

    warnings: tuple[str, ...] = ()

    @property
    def lonlat(self) -> tuple[float, float]:
        return self.catchment.outlet_lonlat

    @property
    def area_ha(self) -> float:
        return self.catchment.area_ha

    @property
    def is_recommended(self) -> bool:
        """False for a site the evidence does not support recommending.

        Only the ensemble can veto: a clipped catchment (high edge contact) is still a
        real place to put a pond, its area is simply a lower bound. A site the three
        grids cannot agree on is not a place at all.
        """
        return self.confidence != "low"


@dataclass(frozen=True)
class SitingResult:
    """The ranked sites and the search that produced them."""

    sites: tuple[PondSite, ...]
    mapped_area_m2: float
    stream_threshold_m2: float
    stream_cells: int
    buildable_cells: int
    clear_of_watercourse_cells: int
    candidate_cells: int
    trunk_cells: int
    trunk_threshold_m2: float
    resolution_m: float
    warnings: tuple[str, ...] = ()

    @property
    def recommended(self) -> tuple[PondSite, ...]:
        return tuple(s for s in self.sites if s.is_recommended)

    @property
    def best(self) -> PondSite | None:
        return self.recommended[0] if self.recommended else None


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
class PondSiteSelector:
    """Runs the five steps against one flow field.

    The delineator is shared, so the reverse-D8 index is built once and each candidate
    costs the size of its own catchment. That is what makes suppression affordable: the
    loop delineates only the cells it actually picks, never the whole candidate pool.
    """

    def __init__(
        self,
        flow: FlowField,
        *,
        config: SitingConfig | None = None,
        delineator: CatchmentDelineator | None = None,
        ensemble: CatchmentEnsemble | None = None,
    ) -> None:
        self.flow = flow
        self.dem = flow.dem
        self.config = config or settings.siting
        self.delineator = delineator or CatchmentDelineator(flow)
        self.ensemble = ensemble
        self._relative_elevation: np.ndarray | None = None
        self._height_above_trunk: np.ndarray | None = None

    @classmethod
    def from_ensemble(
        cls, ensemble: CatchmentEnsemble, *, config: SitingConfig | None = None
    ) -> "PondSiteSelector":
        """Site on the ensemble's primary grid and cross-check on the others.

        The primary grid's delineator is reused rather than rebuilt, so the ensemble adds
        two delineations per site and nothing else.
        """
        primary = ensemble.primary
        return cls(primary.flow, config=config, delineator=primary, ensemble=ensemble)

    # ------------------------------------------------------------------ #
    # The masks
    # ------------------------------------------------------------------ #
    @property
    def mapped_area_m2(self) -> float:
        """Ground area the contours actually cover. The denominator of step 1."""
        return self.dem.area_of(self.dem.valid)

    @property
    def stream_threshold_m2(self) -> float:
        return self.mapped_area_m2 * self.config.stream_threshold_fraction

    def upstream_area(self) -> np.ndarray:
        """(ny, nx) area draining through each cell, in m^2."""
        return self.flow.accumulation * self.dem.row_cell_areas[:, None]

    def stream_mask(self) -> np.ndarray:
        """Cells carrying at least the threshold area (PLAN §2 Siting step 1)."""
        return self.dem.valid & (self.upstream_area() >= self.stream_threshold_m2)

    def distance_to_edge_m(self) -> np.ndarray:
        """(ny, nx) distance from each cell to the nearest no-data cell or map border.

        The array is padded with no-data before the transform so the border counts as an
        edge of the data, which it is: ground beyond it was never surveyed either.
        """
        ny, nx = self.dem.shape
        padded = np.zeros((ny + 2, nx + 2), dtype=bool)
        padded[1:-1, 1:-1] = self.dem.valid
        distance = distance_transform_edt(padded)[1:-1, 1:-1]
        return np.asarray(distance) * self.dem.resolution_m

    def buildable_mask(self) -> np.ndarray:
        """Low-slope ground, far enough inside the mapped area to build on."""
        # No-data slopes are NaN; treating them as vertical keeps the comparison honest
        # and avoids a warning-laden NaN comparison.
        steepness = np.nan_to_num(self.flow.slope, nan=np.inf)
        gentle = steepness < self.config.max_slope_fraction
        return (
            self.dem.valid
            & gentle
            & (self.distance_to_edge_m() >= self.config.edge_buffer_m)
        )

    @property
    def trunk_threshold_m2(self) -> float:
        """Upstream area above which a channel counts as a watercourse."""
        return self.config.trunk_drainage_area_ha * 1e4

    def trunk_mask(self) -> np.ndarray:
        """Cells on a channel that already drains more than the trunk threshold.

        The threshold is absolute hectares rather than a share of the sheet, and that is
        the whole point. A share would scale with whatever was uploaded, so the biggest
        channel on a 20 ha farm map would be called a river and the rule would refuse to
        put a pond anywhere. A watercourse is a watercourse at 150 ha whether the sheet
        around it is 200 ha or 200 km^2.
        """
        return self.dem.valid & (self.upstream_area() >= self.trunk_threshold_m2)

    def height_above_trunk(self) -> np.ndarray:
        """(ny, nx) height of each cell above the watercourse it drains into.

        Height above nearest drainage, measured along the flow path rather than as a
        straight line: the number that matters is how far the pond bed stands above the
        channel that would flood it, and water gets there by flowing, not by the shortest
        route.

        Resolved in one pass up the topological order, the same trick
        `FlowField.terminal_outlets` uses. In ascending elevation every cell's receiver is
        already resolved, so a cell either is trunk (and stands zero above itself) or
        inherits whatever its receiver found.

        Two cases have no answer, and they are opposite ones. A sheet with no channel over
        the trunk threshold maps no watercourse at all, so every cell gets `inf`: there is
        nothing here to be too close to. On a sheet that does map one, a cell whose water
        leaves the sheet before reaching it gets `-inf` instead. Such a cell sits at the
        edge of the data with an unknown channel just beyond it, which is not the same as
        standing clear of one, and on the provided sheet those cells are exactly the strip
        along the near bank of the river.
        """
        trunk_mask = self.trunk_mask()
        if not trunk_mask.any():
            return np.full(self.dem.shape, np.inf)

        trunk = trunk_mask.ravel().tolist()
        receivers = self.flow.receivers.tolist()
        elevation = self.dem.z.ravel().tolist()

        # Lists rather than arrays: this is a serial pointer walk, and indexing a numpy
        # array one element at a time in a 600,000-iteration Python loop costs several
        # times what indexing a list does.
        channel_z = [float("nan")] * len(receivers)
        for cell in self.flow.order.tolist()[::-1]:
            if trunk[cell]:
                channel_z[cell] = elevation[cell]
            else:
                target = receivers[cell]
                channel_z[cell] = channel_z[target] if target >= 0 else float("inf")

        below = np.asarray(channel_z, dtype=np.float64).reshape(self.dem.shape)
        with np.errstate(invalid="ignore"):
            return self.dem.z - below

    @property
    def height_above_trunk_grid(self) -> np.ndarray:
        """`height_above_trunk()`, computed once. It is a serial pass over every valid
        cell, which is the most expensive thing in this module."""
        if self._height_above_trunk is None:
            self._height_above_trunk = self.height_above_trunk()
        return self._height_above_trunk

    def clear_of_watercourse_mask(self) -> np.ndarray:
        """Ground that stands far enough above the nearest watercourse to hold a pond.

        This is the rule that keeps site 1 out of the river. Ranking purely by catchment
        area asks for the cell that the most water passes through, and on any sheet that
        contains a river the answer is the river: on the provided map the top site landed
        in the Shivnath, 418 ha of a 831 ha sheet, and the "pond" was a 2.4 million m^3
        impoundment across a live channel.

        Two conditions, both needed. The cell must not be on the trunk itself, and it must
        stand `min_height_above_trunk_m` above the trunk cell it drains into, so the pond
        bed clears the channel and its floodplain. Checked against the OpenStreetMap water
        layer over the provided sheet: 310 of 2,413 candidate cells (12.8%) stood in the
        river before this rule and 9 of 754 (1.2%) after it.

        On a sheet with no channel over the trunk threshold every height is `inf` and the
        mask is simply the valid area, so a farm-scale map is unaffected.
        """
        return self.dem.valid & (
            self.height_above_trunk_grid >= self.config.min_height_above_trunk_m
        )

    def candidate_mask(self) -> np.ndarray:
        return (
            self.stream_mask() & self.buildable_mask() & self.clear_of_watercourse_mask()
        )

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def relative_elevation(self) -> np.ndarray:
        """(ny, nx) elevation of each cell minus the local mean around it.

        A NaN-aware box mean, computed the same normalised way as the DEM's Gaussian
        (PLAN §11.2): sum the valid elevations, divide by the count of valid cells. A
        plain `uniform_filter` over an array with NaN in it returns NaN for every cell
        within a window of the map edge, which is exactly the ground the sites sit near.
        """
        radius_m = (
            self.dem.meta.mean_contour_spacing_m
            * self.config.relative_elevation_radius_spacing_multiple
        )
        size = max(3, int(round(2 * radius_m / self.dem.resolution_m)) | 1)

        valid = self.dem.valid.astype(np.float64)
        filled = np.where(self.dem.nodata, 0.0, self.dem.z)
        total = uniform_filter(filled, size=size, mode="nearest")
        weight = uniform_filter(valid, size=size, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            local_mean = np.where(weight > 0, total / np.maximum(weight, 1e-6), np.nan)
        return self.dem.z - local_mean

    @property
    def relative_elevation_grid(self) -> np.ndarray:
        """`relative_elevation()`, computed once. Two box filters over the whole map is
        not much, but it is wasted work per site."""
        if self._relative_elevation is None:
            self._relative_elevation = self.relative_elevation()
        return self._relative_elevation

    def score_cell(self, row: int, col: int, catchment: Catchment) -> SiteScore:
        depression = float(self.flow.filled[row, col] - self.dem.z[row, col])
        return SiteScore(
            upstream_area_m2=catchment.area_m2,
            slope=float(self.flow.slope[row, col]),
            # Filling never lowers ground, so this is non-negative by construction; the
            # clamp is against float noise on cells the fill did not touch.
            depression_depth_m=max(0.0, depression),
            relative_elevation_m=float(self.relative_elevation_grid[row, col]),
            height_above_trunk_m=float(self.height_above_trunk_grid[row, col]),
        )

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #
    def select(self, top_n: int | None = None) -> SitingResult:
        """Rank the candidate cells and pick independent basins from the top down."""
        wanted = self._resolve_top_n(top_n)

        # The three masks are computed once here and carried through: each costs a pass
        # over the whole grid, and the diagnostics want the same arrays the loop used.
        stream = self.stream_mask()
        buildable = self.buildable_mask()
        clear = self.clear_of_watercourse_mask()
        candidates = stream & buildable & clear
        if not candidates.any():
            raise SitingError(*self._no_candidates_reason(stream, buildable, clear))

        available = candidates.copy()
        ranked = self._ranked_candidates(candidates)

        nx = self.dem.shape[1]
        # A view, not a copy: the in-place suppression below is visible through it.
        flat_available = available.ravel()
        sites: list[PondSite] = []
        for cell in ranked:
            if not flat_available[cell]:
                continue
            row, col = divmod(int(cell), nx)
            catchment = self.delineator.delineate_cell(row, col)
            sites.append(self._build_site(len(sites) + 1, row, col, catchment))
            # Step 4: this pick's whole catchment leaves the pool, so the next site is a
            # different basin rather than a different point on the same stream.
            available &= ~self._suppression_mask(row, col, catchment)
            if len(sites) == wanted:
                break

        warnings: list[str] = []
        if len(sites) < wanted:
            warnings.append(
                f"Only {len(sites)} independent basin(s) meet the siting rules; "
                f"{wanted} were requested."
            )

        return SitingResult(
            sites=tuple(sites),
            mapped_area_m2=self.mapped_area_m2,
            stream_threshold_m2=self.stream_threshold_m2,
            stream_cells=int(stream.sum()),
            buildable_cells=int(buildable.sum()),
            clear_of_watercourse_cells=int(clear.sum()),
            candidate_cells=int(candidates.sum()),
            trunk_cells=int(self.trunk_mask().sum()),
            trunk_threshold_m2=self.trunk_threshold_m2,
            resolution_m=self.dem.resolution_m,
            warnings=tuple(warnings),
        )

    def site_at(self, lon: float, lat: float, *, rank: int = 1) -> PondSite:
        """Score one caller-supplied pour point, bypassing the ranking entirely.

        Phase 9 lets a planner name a place they have already chosen. The point still
        snaps to the routed channel, because a catchment traced from a cell beside the
        stream is the hillside and not the basin. The site reports how far it moved,
        so the answer never silently describes somewhere else.

        Raises `ValueError` when the point lies off the mapped area or has no valid data
        within the snap radius; the caller turns that into a structured response.
        """
        catchment = self.delineator.delineate(lon, lat)
        return self._build_site(rank, *catchment.outlet_rc, catchment)

    # ------------------------------------------------------------------ #
    def _suppression_mask(self, row: int, col: int, catchment: Catchment) -> np.ndarray:
        """Ground this pick takes out of the pool (step 4).

        Normally the pick's entire upstream catchment. With
        `suppression_removes_catchment` off it is a square window instead. That is the
        rejected alternative, kept so PLAN §11.7 can be re-run rather than merely
        asserted: on the sample it returns five nested points on a single stream.
        """
        if self.config.suppression_removes_catchment:
            return catchment.mask

        ny, nx = self.dem.shape
        radius = max(1, int(round(self.delineator.snap_radius_m / self.dem.resolution_m)))
        window = np.zeros((ny, nx), dtype=bool)
        window[
            max(0, row - radius) : min(ny, row + radius + 1),
            max(0, col - radius) : min(nx, col + radius + 1),
        ] = True
        return window

    def _resolve_top_n(self, top_n: int | None) -> int:
        if top_n is None:
            return self.config.default_top_n
        return max(1, min(int(top_n), self.config.max_top_n))

    def _ranked_candidates(self, candidates: np.ndarray) -> np.ndarray:
        """Candidate cells as flat indices, most upstream area first.

        Sorted on accumulation rather than on delineated area so the ordering costs one
        argsort instead of one traversal per candidate. The two orderings are the same,
        since accumulation counts the cells the mask collects. `stable` keeps ties in
        index order, so the same sheet always produces the same list.
        """
        cells = np.flatnonzero(candidates.ravel())
        accumulation = self.flow.accumulation.ravel()[cells]
        return cells[np.argsort(-accumulation, kind="stable")]

    def _build_site(self, rank: int, row: int, col: int, catchment: Catchment) -> PondSite:
        ensemble = (
            self.ensemble.delineate(*catchment.outlet_lonlat) if self.ensemble else None
        )
        confidence = ensemble.confidence if ensemble else "unassessed"

        warnings: list[str] = []
        if catchment.is_lower_bound:
            warnings.append(
                f"{catchment.edge_contact:.1%} of the catchment perimeter is on the edge "
                "of the mapped data; the area is a lower bound."
            )
        if ensemble is not None and confidence == "low":
            warnings.append(
                f"The three grids disagree ({ensemble.mean_area_ha:.1f} +/- "
                f"{ensemble.std_area_ha:.1f} ha); this site is not recommended."
            )

        return PondSite(
            rank=rank,
            catchment=catchment,
            score=self.score_cell(row, col, catchment),
            ensemble=ensemble,
            confidence=confidence,
            warnings=tuple(warnings),
        )

    def _no_candidates_reason(
        self, stream: np.ndarray, buildable_mask: np.ndarray, clear_mask: np.ndarray
    ) -> tuple[str, str, str]:
        """Say which rule emptied the pool. That is the useful half of the error."""
        streams = int(stream.sum())
        buildable = int(buildable_mask.sum())
        threshold_ha = self.stream_threshold_m2 / 1e4
        if streams == 0:
            return (
                "no_stream_network",
                f"No cell drains more than {threshold_ha:.1f} ha "
                f"({self.config.stream_threshold_fraction:.1%} of the mapped area).",
                "The sheet may be too small or too flat to carry a channel.",
            )
        if buildable == 0:
            return (
                "no_buildable_ground",
                f"No ground is both under {self.config.max_slope_fraction:.0%} slope and "
                f"more than {self.config.edge_buffer_m:.0f} m inside the mapped area.",
                "The terrain is too steep, or the mapped area too narrow, for a pond.",
            )
        if not (stream & buildable_mask & clear_mask).any() and (
            stream & buildable_mask
        ).any():
            return (
                "no_ground_clear_of_watercourse",
                f"Every buildable channel on this sheet lies within "
                f"{self.config.min_height_above_trunk_m:.0f} m of a watercourse "
                f"draining more than {self.config.trunk_drainage_area_ha:.0f} ha. A pond "
                "there would sit in the channel or its floodplain.",
                "The sheet may be all river floodplain. Raise "
                "POND_SITING_TRUNK_DRAINAGE_AREA_HA, or lower "
                "POND_SITING_MIN_HEIGHT_ABOVE_TRUNK_M, if the watercourse here is small "
                "enough to build across.",
            )
        return (
            "no_site_found",
            f"{streams} stream cells and {buildable} buildable cells exist, but none "
            "coincide.",
            "Every channel on this sheet runs through ground too steep to dam.",
        )


# --------------------------------------------------------------------------- #
def select_pond_sites(
    flow: FlowField,
    *,
    top_n: int | None = None,
    ensemble: CatchmentEnsemble | None = None,
    config: SitingConfig | None = None,
) -> SitingResult:
    """Convenience wrapper: site on `flow`, optionally cross-checked by `ensemble`."""
    if ensemble is not None:
        selector = PondSiteSelector.from_ensemble(ensemble, config=config)
    else:
        selector = PondSiteSelector(flow, config=config)
    return selector.select(top_n)
