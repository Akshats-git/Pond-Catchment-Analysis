"""How much water arrives, how much the pond can hold, and how long it takes to get there.

**Runoff.** SCS-CN is an *event* model. Applied to a year's rainfall as a single 1200 mm
storm it returns a 92% runoff coefficient, which no catchment on earth produces; applied
per rain day and summed it returns 16%, which is what this terrain does (PLAN §4). Both
numbers are computed here -- the wrong one deliberately, because a report that only shows
the right answer cannot show why it is right.

**Storage.** The stage-storage curve is integrated from the DEM, not assumed from a
shape. At each stage the pond is every cell the water reaches from the outlet without
crossing ground that stands above it, and the depth of each of those cells is measured
against the surveyed surface -- so a hollow inside the pond counts as the water it holds.
At stage zero that integral is the water the untouched ground keeps; every stage above it
is held by works.

The curve is then read for its shape as much as for its endpoint. Where one step
multiplies the water surface, the pond has topped a divide, and the volume below that step
is the pond the site really holds: on the sample's best site, 8,000 m^3 over 0.8 ha at
2.75 m against 616,000 m^3 over 29 ha a quarter of a metre higher. The frustum formula a
spreadsheet would use is reported beside the integral as a cross-check, and underestimates
it by 26-65% across the sample's five sites -- real ground widens as it rises faster than
straight sides do, which is the whole argument for integrating.

The pond's depth comes from the *target depth* and the terrain, never from the depression
depth: a site on a channel has no natural storage at all, and reporting zero capacity for
the best site on the sheet would be an artefact of the model rather than a fact about the
ground (PLAN §11 / Phase 7).

**Time of concentration.** Kirpich, on the longest flow path and the drop along *that*
path -- not the basin's overall relief, which belongs to a different, usually steeper,
line down the hill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from app.config import HydrologyConfig, settings
from app.core.catchment import Catchment
from app.core.terrain import FlowField
from app.providers.rainfall import RainfallSeries

__all__ = [
    "HydrologyError",
    "RunoffResult",
    "StageStorage",
    "WaterBalance",
    "retention_mm",
    "scs_cn_runoff",
    "runoff_volume_m3",
    "time_of_concentration_min",
    "stage_storage",
    "water_balance",
]

_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
"""Water crosses a diagonal gap between two cells; a 4-connected pool would be split in
two by a single diagonal rim that does not exist on the ground."""


class HydrologyError(Exception):
    """A hydrological parameter that has no defined answer."""

    def __init__(self, code: str, detail: str, hint: str = "") -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.hint = hint


# --------------------------------------------------------------------------- #
# SCS-CN runoff
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunoffResult:
    """Annual runoff from a daily rainfall series, and the mistake not made."""

    curve_number: float
    retention_mm: float
    """S = 25400/CN - 254: the depth of water the soil can still absorb when dry."""

    initial_abstraction_mm: float
    """Ia = 0.2 S -- rain that wets the ground before any of it runs off."""

    rainfall_mm: float
    rain_days: int
    runoff_depth_mm: float
    """Sum of the per-day runoff depths. The number that is right."""

    runoff_coefficient: float
    contributing_days: int
    """Days that produced any runoff at all -- the rest never exceeded Ia."""

    single_event_depth_mm: float
    single_event_coefficient: float
    """The same model applied to the annual total as one storm. Kept because it is the
    error worth showing: 1104 mm and 92% against 192 mm and 16% (PLAN §4)."""

    method: str = "SCS-CN, applied per rain day and summed"

    @property
    def overestimate_factor(self) -> float:
        """How many times too much runoff the single-event mistake produces."""
        return (
            self.single_event_depth_mm / self.runoff_depth_mm
            if self.runoff_depth_mm > 0
            else float("inf")
        )


def retention_mm(curve_number: float, *, config: HydrologyConfig | None = None) -> float:
    """S = 25400/CN - 254, in millimetres."""
    cfg = config or settings.hydrology
    low, high = cfg.curve_number_range
    if not low <= curve_number <= high:
        raise HydrologyError(
            "curve_number_out_of_range",
            f"Curve number {curve_number} is outside the defined range {low}-{high}.",
            "CN 30 is dry sand under forest; CN 98 is impervious pavement.",
        )
    return 25400.0 / curve_number - 254.0


def scs_cn_runoff(
    daily_rainfall_mm: np.ndarray | RainfallSeries,
    curve_number: float | None = None,
    *,
    config: HydrologyConfig | None = None,
) -> RunoffResult:
    """Runoff depth from a daily rainfall series (PLAN §4).

        Q = (P - Ia)^2 / (P - Ia + S)     for P > Ia, else 0

    Applied to each day and summed. The quadratic numerator is why the aggregation order
    matters so much: runoff grows faster than rainfall, so one 100 mm day yields far more
    than ten 10 mm days, and a year treated as one storm yields more than either.
    """
    cfg = config or settings.hydrology
    cn = cfg.default_curve_number if curve_number is None else float(curve_number)
    series = (
        daily_rainfall_mm.daily_mm
        if isinstance(daily_rainfall_mm, RainfallSeries)
        else np.asarray(daily_rainfall_mm, dtype=np.float64)
    )
    if series.ndim != 1:
        raise HydrologyError(
            "bad_rainfall_series",
            "Daily rainfall must be a one-dimensional series of depths in mm.",
        )
    if series.size and series.min() < 0:
        raise HydrologyError(
            "bad_rainfall_series", "Daily rainfall cannot be negative."
        )

    retention = retention_mm(cn, config=cfg)
    abstraction = cfg.initial_abstraction_ratio * retention

    def runoff(depths: np.ndarray) -> np.ndarray:
        excess = depths - abstraction
        # Only days that get past the initial abstraction produce anything; the formula
        # is undefined below it, not merely small.
        return np.where(excess > 0, excess ** 2 / (excess + retention), 0.0)

    per_day = runoff(series)
    total_rain = float(series.sum())
    depth = float(per_day.sum())
    single = float(runoff(np.array([total_rain]))[0])

    return RunoffResult(
        curve_number=cn,
        retention_mm=retention,
        initial_abstraction_mm=abstraction,
        rainfall_mm=total_rain,
        rain_days=int(series.size),
        runoff_depth_mm=depth,
        runoff_coefficient=depth / total_rain if total_rain else 0.0,
        contributing_days=int((per_day > 0).sum()),
        single_event_depth_mm=single,
        single_event_coefficient=single / total_rain if total_rain else 0.0,
    )


def runoff_volume_m3(runoff_depth_mm: float, area_m2: float) -> float:
    """A depth of water over a catchment, as a volume: V = (Q/1000) * A."""
    return runoff_depth_mm / 1000.0 * area_m2


# --------------------------------------------------------------------------- #
# Time of concentration
# --------------------------------------------------------------------------- #
def time_of_concentration_min(
    length_m: float, relief_m: float, *, config: HydrologyConfig | None = None
) -> float:
    """Kirpich (1940): Tc = 0.01947 * L^0.77 * S^-0.385, in minutes.

    `length_m` and `relief_m` are the longest flow path and the drop *along it* -- the
    pair `Catchment` reports together for exactly this reason. Using the basin's overall
    relief instead would shorten Tc by about a fifth on the sample's largest basin, since
    the highest ground is not the most distant.
    """
    cfg = config or settings.hydrology
    if length_m <= 0:
        raise HydrologyError(
            "degenerate_flow_path",
            "Time of concentration needs a flow path of non-zero length.",
        )
    # A flow path that a contour-derived DEM reports as perfectly level is an artefact of
    # the contour interval, not a real horizontal channel. Clamping the slope keeps Tc
    # finite and errs long, which is the conservative direction for a spillway.
    slope = max(relief_m / length_m, cfg.kirpich_min_slope)
    return (
        cfg.kirpich_coefficient
        * length_m ** cfg.kirpich_length_exponent
        * slope ** cfg.kirpich_slope_exponent
    )


# --------------------------------------------------------------------------- #
# Stage-storage
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageStorage:
    """The pond, measured off the terrain rather than assumed."""

    stages_m: tuple[float, ...]
    """Water level above the outlet's ground surface, from 0 to the target depth."""

    areas_m2: tuple[float, ...]
    volumes_m3: tuple[float, ...]

    capacity_m3: float
    """Volume at the target depth: what the pond holds when full."""

    surface_area_m2: float
    max_depth_m: float

    natural_storage_m3: float
    """Volume at stage 0: the water the site holds with nothing built, counting only the
    ground above the outlet -- the half of any hollow a structure at the outlet keeps.
    Zero on a channel, which is where catchment-first siting puts most sites; those ponds
    are dug or bunded, and every stage above 0 comes from the target depth against the
    surrounding ground, never from the depression depth (PLAN §11)."""

    natural_storage_area_m2: float

    site_depression_m3: float
    """The whole natural hollow the site sits in, both sides of the outlet, filled to its
    spill level. `natural_storage_m3` counts only the upstream half -- the half a bund at
    the outlet would keep -- but the hollow is what a surveyor standing there would see,
    and PLAN §3 reports it for site 3 of the sample (7,400 m^3)."""

    site_depression_area_m2: float
    is_excavated: bool
    """True when the site has no natural hollow worth the name, so all of the capacity has
    to be dug or bunded. Judged on `site_depression_m3`: a hollow that lies below the
    outlet is still a hollow, even though the pond does not keep it."""

    frustum_estimate_m3: float
    """V = (d/3)(A_top + A_bot + sqrt(A_top * A_bot)), the textbook cross-check."""

    frustum_error: float
    """(frustum - integrated) / integrated. Positive: the straight-sided idealisation
    overestimates, which is what a concave basin does to it."""

    water_spread_fraction: float
    """Surface area at capacity as a fraction of the catchment. A pond that covers a
    large share of the ground that feeds it is a reservoir, and is flagged as one."""

    pond_mask: np.ndarray
    """(ny, nx) bool -- the water surface of the pond at its usable stage, which is what
    Phase 8 draws as the pond footprint."""

    spill_stage_index: int | None
    """Index of the stage at which the water first tops a divide and spreads -- the step
    where the surface area jumps by more than `spill_area_jump_factor`. None when the pond
    fills to the target depth without spilling."""

    is_reservoir: bool
    """True when a structure of the target depth would flood a large share of its own
    catchment. The capacity above it is still the volume the ground would hold; it is just
    no longer the volume of a village pond."""

    warnings: tuple[str, ...] = ()

    @property
    def spill_stage_m(self) -> float | None:
        """Depth of the deepest pond the site holds before the water spreads."""
        index = self.spill_stage_index
        return None if index is None else self.stages_m[index]

    @property
    def usable_capacity_m3(self) -> float:
        """Capacity below the spill: the pond a village would actually build here.

        Equal to `capacity_m3` where the water never tops a divide. Where it does, the
        difference is large -- 4,600 m^3 against 616,000 m^3 on the sample's best site --
        and reporting only the second would describe a reservoir the site cannot hold.
        """
        index = -1 if self.spill_stage_index is None else self.spill_stage_index
        return self.volumes_m3[index]

    @property
    def usable_area_m2(self) -> float:
        index = -1 if self.spill_stage_index is None else self.spill_stage_index
        return self.areas_m2[index]

    @property
    def triples(self) -> tuple[tuple[float, float, float], ...]:
        """(stage, area, volume) rows, the form the API returns."""
        return tuple(zip(self.stages_m, self.areas_m2, self.volumes_m3))

    @property
    def capacity_ha_m(self) -> float:
        """Capacity in hectare-metres, the unit Indian irrigation practice uses."""
        return self.capacity_m3 / 1e4


def _component(member: np.ndarray, outlet: tuple[int, int]) -> np.ndarray:
    """The connected patch of `member` containing the outlet, or nothing."""
    if not member[outlet]:
        return np.zeros_like(member)
    labels, _ = label(member, structure=_EIGHT_CONNECTED)
    return labels == labels[outlet]


def _pool_at(
    bed: np.ndarray, mask: np.ndarray, level: float, outlet: tuple[int, int]
) -> np.ndarray:
    """Cells of `mask` that water at `level` reaches from the outlet without crossing
    ground above it.

    Tested on the surveyed surface rather than the depression-filled one. The filled
    surface would be the tidier choice -- it cannot be cut in two by a one-cell artefact
    rim -- but priority-flood raises each step across a flat by an epsilon, so a filled
    depression's surface is a staircase, and thresholding it puts the pond at stage 0 in
    the outlet cell alone. The surveyed surface has no such staircase, and "water does not
    climb over ground that stands above it" is the physical rule anyway.
    """
    return _component(mask & (bed <= level), outlet)


def stage_storage(
    flow: FlowField,
    mask: np.ndarray,
    outlet: tuple[int, int],
    *,
    target_depth_m: float | None = None,
    config: HydrologyConfig | None = None,
) -> StageStorage:
    """Integrate the pond's (stage, area, volume) curve off the DEM.

    `mask` is the site's catchment: the water backs up into the ground that drains to the
    pond and nowhere else, so bounding the pool by the catchment is a statement about
    where water can physically be, not a convenience.
    """
    cfg = config or settings.hydrology
    depth = cfg.default_target_depth_m if target_depth_m is None else float(target_depth_m)
    if depth <= 0:
        raise HydrologyError(
            "bad_target_depth", "The pond's target depth must be greater than zero."
        )

    dem = flow.dem
    # Datum: the level at which the site's own hollow is exactly full. On a channel that
    # is the ground surface; inside a depression it is the spill elevation, which is what
    # priority-flood raised the outlet to. Stages are measured from there, so stage 0 is
    # the water the ground holds with nothing built and every stage above it is works.
    datum = float(flow.filled[outlet])
    bed = np.where(dem.nodata, np.inf, dem.z)
    row_areas = dem.row_cell_areas[:, None]

    steps = max(1, int(cfg.stage_storage_steps))
    stages, areas, volumes = [], [], []
    for step in range(steps + 1):
        stage = depth * step / steps
        pool = _pool_at(bed, mask, datum + stage, outlet)
        water = np.where(pool, np.maximum(datum + stage - bed, 0.0), 0.0)
        stages.append(stage)
        areas.append(float((pool * row_areas).sum()))
        volumes.append(float((water * row_areas).sum()))

    capacity = volumes[-1]
    surface = areas[-1]
    natural = volumes[0]
    catchment_area = dem.area_of(mask)

    # The site's hollow, unbounded by the catchment: the same water body, but counting the
    # half below the outlet as well as the half above it. Defined as the ground that
    # priority-flood had to raise, which is self-bounding -- flooding "everything below the
    # spill level" instead would run away down the valley, where the ground is lower still.
    hollow = _component((flow.filled > dem.z) & dem.valid, outlet)
    hollow_water = np.where(hollow, flow.filled - np.where(dem.nodata, 0.0, dem.z), 0.0)

    # Straight sides from bed to water surface -- the shape a spreadsheet assumes.
    frustum = (
        depth / 3.0 * (surface + areas[0] + math.sqrt(max(surface * areas[0], 0.0)))
    )
    spread = surface / catchment_area if catchment_area else 0.0

    # The first step that multiplies the water surface by more than the documented factor
    # is the pond topping a divide; below it the water is held by the basin itself. A pool
    # still smaller than one contour spacing squared is a puddle in a couple of cells, and
    # every step multiplies one of those -- so a jump only counts once the pond is at least
    # as large as the ground the DEM can resolve.
    resolvable_m2 = dem.meta.mean_contour_spacing_m ** 2
    spill_index: int | None = None
    for step in range(1, len(areas)):
        previous = areas[step - 1]
        jumped = areas[step] > cfg.spill_area_jump_factor * previous
        if previous >= resolvable_m2 and jumped:
            spill_index = step - 1
            break

    # The pond itself: the pool at the stage the site can actually hold, recomputed once
    # rather than keeping every stage's mask alive through the loop.
    pond_stage = stages[-1 if spill_index is None else spill_index]
    pond = _pool_at(bed, mask, datum + pond_stage, outlet)

    depression_volume = float((hollow_water * row_areas).sum())
    excavated = depression_volume <= cfg.natural_storage_floor_m3

    warnings: list[str] = []
    if excavated:
        warnings.append(
            "The site has no natural depression; the whole capacity has to be excavated "
            f"or bunded to the {depth:.1f} m target depth."
        )
    elif natural <= cfg.natural_storage_floor_m3:
        warnings.append(
            f"The {depression_volume:,.0f} m3 hollow at this site lies below the outlet "
            "cell, so a structure here retains almost none of it."
        )
    if spill_index is not None:
        warnings.append(
            f"The water tops a divide between {stages[spill_index]:.2f} m and "
            f"{stages[spill_index + 1]:.2f} m, spreading from "
            f"{areas[spill_index] / 1e4:.2f} ha to {areas[spill_index + 1] / 1e4:.1f} ha. "
            f"The pond this site holds is the {volumes[spill_index]:,.0f} m3 below that."
        )
    if spread > cfg.water_spread_warn_fraction:
        warnings.append(
            f"At {depth:.1f} m the water spreads over {spread:.0%} of the catchment "
            f"({surface / 1e4:.1f} ha). That is a reservoir rather than a pond -- a "
            "shallower structure, or a smaller excavated pond, suits this site better."
        )
    if dem.meta.contour_interval_m:
        warnings.append(
            f"Depths are quantised to the {dem.meta.contour_interval_m:.1f} m contour "
            "interval of the source map."
        )

    return StageStorage(
        stages_m=tuple(stages),
        areas_m2=tuple(areas),
        volumes_m3=tuple(volumes),
        capacity_m3=capacity,
        surface_area_m2=surface,
        max_depth_m=depth,
        natural_storage_m3=natural,
        natural_storage_area_m2=areas[0],
        site_depression_m3=depression_volume,
        site_depression_area_m2=float((hollow * row_areas).sum()),
        is_excavated=excavated,
        frustum_estimate_m3=frustum,
        frustum_error=(frustum - capacity) / capacity if capacity else 0.0,
        water_spread_fraction=spread,
        pond_mask=pond,
        spill_stage_index=spill_index,
        is_reservoir=spread > cfg.water_spread_warn_fraction,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# The whole water balance for one site
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WaterBalance:
    """Runoff in, storage available, and whether the two match."""

    runoff: RunoffResult
    storage: StageStorage
    catchment_area_m2: float
    annual_runoff_m3: float
    fill_ratio: float
    """Annual runoff divided by the pond's usable capacity -- the volume below the stage
    at which the water tops a divide, since that is the pond the site actually holds.
    Above 1 the pond fills and spills in an average year; below 1 it never fills."""

    assessment: str
    time_of_concentration_min: float
    rainfall_source: str
    warnings: tuple[str, ...] = ()

    @property
    def fills(self) -> bool:
        return self.fill_ratio >= 1.0


def _assess(ratio: float, cfg: HydrologyConfig) -> str:
    """The fill ratio in plain English -- the sentence a village officer reads."""
    dry, marginal, ample = cfg.fill_ratio_bands
    if ratio < dry:
        return (
            "The catchment cannot fill the pond in an average year; a smaller pond, or a "
            "site with more upstream area, would hold water more reliably."
        )
    if ratio < marginal:
        return (
            "The catchment just fills the pond in an average year, with little margin in "
            "a dry one."
        )
    if ratio < ample:
        return "The catchment comfortably fills the pond in an average year."
    return (
        "The catchment yields far more runoff than the pond can hold -- it fills early in "
        "the monsoon and spills, so the spillway matters more than the capacity."
    )


def water_balance(
    flow: FlowField,
    catchment: Catchment,
    rainfall: RainfallSeries,
    *,
    curve_number: float | None = None,
    target_depth_m: float | None = None,
    config: HydrologyConfig | None = None,
) -> WaterBalance:
    """Everything Phase 9 reports about the water at one site, in one call."""
    cfg = config or settings.hydrology

    runoff = scs_cn_runoff(rainfall, curve_number, config=cfg)
    storage = stage_storage(
        flow, catchment.mask, catchment.outlet_rc, target_depth_m=target_depth_m, config=cfg
    )
    volume = runoff_volume_m3(runoff.runoff_depth_mm, catchment.area_m2)
    capacity = storage.usable_capacity_m3
    ratio = volume / capacity if capacity > 0 else float("inf")

    warnings = list(rainfall.warnings) + list(storage.warnings)
    low, high = cfg.expected_runoff_coefficient_range
    if not low <= runoff.runoff_coefficient <= high:
        warnings.append(
            f"Runoff coefficient {runoff.runoff_coefficient:.0%} is outside the "
            f"{low:.0%}-{high:.0%} range expected for this terrain; check the curve "
            "number and the rainfall series."
        )
    if catchment.is_lower_bound:
        warnings.append(
            "The catchment area is a lower bound, so the runoff volume is too."
        )

    return WaterBalance(
        runoff=runoff,
        storage=storage,
        catchment_area_m2=catchment.area_m2,
        annual_runoff_m3=volume,
        fill_ratio=ratio,
        assessment=_assess(ratio, cfg),
        time_of_concentration_min=time_of_concentration_min(
            catchment.longest_flow_path_m, catchment.flow_path_relief_m, config=cfg
        ),
        rainfall_source=rainfall.source,
        warnings=tuple(warnings),
    )
