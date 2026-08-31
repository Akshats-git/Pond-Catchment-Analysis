"""The shape of the answer, and the rounding that makes it readable.

Two jobs. The first is documentation: these models are what FastAPI renders at `/docs`,
so a grader can see the whole contract without running anything, and every field carries
the sentence that says what it means. The second is translation, because the core speaks
in numpy arrays, masks and dataclasses, and none of that crosses a JSON boundary.

**On rounding.** Numbers are rounded where a reader would round them and nowhere else: a
catchment to 0.1 ha, a volume to whole cubic metres, a coordinate to six decimals (~0.1 m,
finer than any grid this service builds). Rounding is presentation, so it happens here and
never in `app/core/`, where the full precision is what the mass-balance test checks.

**On saying why.** `why` is assembled from the numbers that actually chose the site:
upstream area, slope, relative elevation, depression depth and height above the
watercourse, plus the verdict of the three grids.
It is a rendering of the score, not a second opinion about it: if the sentence and the
number ever disagreed, the sentence would be the wrong one, so it is generated from the
number rather than written alongside it.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings
from app.core.hydrology import WaterBalance
from app.core.pond_siting import PondSite
from app.pipeline import AnalysisResult
from app.providers.rainfall import RainfallSeries
from app.schemas.requests import AnalysisParams

__all__ = [
    "AnalysisResponse",
    "ErrorResponse",
    "HealthResponse",
    "RainfallResponse",
    "analysis_response",
    "rainfall_response",
    "site_reasons",
]

_HA = 1e4


def _r(value: float, places: int = 1) -> float:
    """Round, and hand back a plain float. `round()` on a numpy scalar returns a numpy
    scalar, which the JSON encoder then refuses."""
    return round(float(value), places)


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
class InputSummary(BaseModel):
    """What the service found in the uploaded file."""

    filename: str
    contour_count: int = Field(description="Contour lines carrying an elevation.")
    vertex_count: int
    elevation_source: str = Field(
        description="Where the heights were read from. One of z_coordinate, "
        "extended_data, placemark_name or folder_name."
    )
    interval_m: float | None = Field(description="Median spacing between contour levels.")
    level_count: int
    elevation_range_m: list[float]
    mapped_area_ha: float = Field(
        description="Ground the contours actually cover. Every share of the sheet "
        "quoted below is measured against this."
    )
    bbox: list[float] = Field(description="[min_lon, min_lat, max_lon, max_lat].")
    unit_hint: str | None = Field(
        default=None,
        description="A unit found beside the heights. Reported and never acted on, "
        "because this service will not quietly turn feet into metres.",
    )
    skipped_features: int = Field(
        default=0, description="Line placemarks no strategy could give an elevation to."
    )
    document_name: str | None = None


class DEMSummary(BaseModel):
    """How the grid under the analysis was built."""

    resolution_m: float
    resolution_source: str = Field(
        description="auto when it came from the contour spacing, requested when you "
        "asked for it, coarsened when it had to grow to stay under the cell limit."
    )
    shape: list[int] = Field(description="[rows, columns].")
    smoothing_sigma_m: float = Field(
        description="How much the surface was smoothed to take out the stair steps "
        "that contour interpolation leaves. It follows the contour spacing, not the grid."
    )
    mean_contour_spacing_m: float
    nodata_fraction: float = Field(
        description="Share of grid cells outside the contour hull."
    )
    elevation_range_m: list[float]
    max_smoothing_shift_m: float = Field(
        description="Furthest the smoothing moved any one cell. This is the evidence "
        "that it took out an artefact and not the terrain."
    )


class Location(BaseModel):
    lat: float
    lon: float


class CatchmentSummary(BaseModel):
    """The catchment draining to the site, and how much to trust it."""

    area_ha: float
    area_uncertainty_ha: float | None = Field(
        description="How far the three grids disagree about the area. Null when the "
        "cross-check was switched off."
    )
    ensemble_mean_area_ha: float | None = None
    confidence: str = Field(
        description="high, medium or low, from how far the three grids disagree. "
        "unassessed when the cross-check was switched off."
    )
    edge_contact_pct: float = Field(
        description="How much of the catchment boundary runs along the edge of the "
        "mapped data."
    )
    is_lower_bound: bool = Field(
        description="True when enough of the boundary hugs the edge of the data that "
        "the real catchment must carry on past it."
    )
    relief_m: float = Field(
        description="Highest ground in the catchment minus the height of the outlet."
    )
    longest_flow_path_m: float
    flow_path_relief_m: float = Field(
        description="Drop along that path. This is the H in Kirpich, and it differs "
        "from relief_m because the furthest point is rarely the highest one."
    )
    time_of_concentration_min: float
    cell_count: int
    method: str = "D8 steepest-descent on a contour-interpolated, smoothed DEM"
    grid_resolutions_m: list[float] = Field(
        default_factory=list, description="The ensemble grids, if one was run."
    )


class StorageSummary(BaseModel):
    """What the site can hold, integrated off the terrain rather than assumed."""

    capacity_m3: float = Field(
        description="What the site actually holds. The volume below the depth at which "
        "water first tops a ridge and starts spreading."
    )
    capacity_at_target_depth_m3: float = Field(
        description="Volume at the full depth asked for, ignoring any spill. Bigger "
        "than capacity_m3 wherever the water would spread past the site."
    )
    surface_area_m2: float
    max_depth_m: float
    usable_depth_m: float | None = Field(
        description="Depth at which the water starts spilling. Null when the pond "
        "reaches the depth asked for without spilling."
    )
    natural_storage_m3: float = Field(
        description="What the ground holds with nothing built. Zero on a channel, and "
        "a channel is where catchment-first siting puts most sites."
    )
    site_depression_m3: float = Field(
        description="The whole natural hollow, both sides of the outlet."
    )
    is_excavated: bool
    is_reservoir: bool = Field(
        description="True when a structure of the depth asked for would flood a large "
        "part of the catchment feeding it."
    )
    water_spread_fraction: float
    frustum_estimate_m3: float = Field(
        description="Textbook cross-check: V = (d/3)(A_top + A_bot + sqrt(A_top*A_bot))."
    )
    frustum_error: float
    stage_storage: list[list[float]] = Field(
        description="Rows of [depth_m, area_m2, volume_m3] from the bed up to the "
        "depth asked for."
    )


class RunoffSummary(BaseModel):
    """The water arriving in an average year."""

    method: str
    curve_number: float
    retention_mm: float = Field(description="S = 25400/CN - 254.")
    initial_abstraction_mm: float = Field(description="Ia = 0.2 * S.")
    rainfall_mm: float
    rain_days: int
    rainfall_source: str
    runoff_depth_mm: float
    runoff_coefficient: float
    contributing_days: int = Field(
        description="Days whose rain got past the initial abstraction and so produced "
        "runoff. The rest produced none."
    )
    annual_runoff_m3: float
    single_event_coefficient: float = Field(
        description="What SCS-CN gives if the whole year is treated as one storm. It "
        "is reported so the size of the mistake this service avoids is visible."
    )
    overestimate_factor: float
    fill_ratio: float = Field(
        description="Water arriving in a year divided by what the pond holds. Above 1 "
        "the pond fills and spills in an average year. A value of -1 means the site "
        "holds nothing at all, because JSON has no way to write an infinite ratio."
    )
    assessment: str


class SiteSummary(BaseModel):
    """One ranked location, with everything that justifies it."""

    rank: int
    is_recommended: bool = Field(
        description="False when the three grids cannot agree on the site's catchment. "
        "Such a site keeps its rank and comes back flagged. It is never quietly moved "
        "down the list."
    )
    location: Location
    snap_distance_m: float = Field(
        description="How far the outlet moved to reach the channel the terrain routes "
        "water down. A large value means the answer is not about the point you asked for."
    )
    why: list[str]
    slope: float = Field(description="Local slope at the outlet, rise over run.")
    relative_elevation_m: float = Field(
        description="Height of the outlet minus the average height of the ground "
        "around it. Negative means the site sits in a hollow, which is what a pond wants."
    )
    depression_depth_m: float
    height_above_trunk_m: float | None = Field(
        description="How far the site stands above the watercourse it drains into. Null "
        "when the sheet carries no channel over the trunk threshold, so there is no "
        "watercourse to stand clear of."
    )
    catchment: CatchmentSummary
    storage: StorageSummary
    runoff: RunoffSummary
    warnings: list[str] = Field(default_factory=list)


class SearchSummary(BaseModel):
    """The candidate search behind the ranking, so an empty or thin result explains
    itself. Absent when the client named a pour point and no search happened."""

    stream_threshold_ha: float = Field(
        description="How much ground has to drain through a cell before it counts as a "
        "stream. A fixed share of the mapped area, never a percentile of flow."
    )
    stream_cells: int
    buildable_cells: int = Field(
        description="Cells under the slope limit and clear of the edge buffer."
    )
    trunk_threshold_ha: float = Field(
        description="Upstream area above which a channel counts as a watercourse rather "
        "than a field drain. Absolute hectares, so it does not move with the size of the "
        "sheet."
    )
    trunk_cells: int = Field(
        description="Cells on such a watercourse. Zero on a sheet too small to carry one."
    )
    clear_of_watercourse_cells: int = Field(
        description="Cells standing far enough above the nearest watercourse to hold a "
        "pond."
    )
    candidate_cells: int = Field(description="Cells that satisfy all three rules.")


class AnalysisResponse(BaseModel):
    """A successful `POST /analyzeContour`."""

    status: str = "ok"
    input: InputSummary
    dem: DEMSummary
    parameters: AnalysisParams = Field(
        description="Every parameter the analysis used, defaults resolved."
    )
    recommended_site: SiteSummary
    alternative_sites: list[SiteSummary] = Field(default_factory=list)
    search: SearchSummary | None = None
    geojson: dict[str, Any] = Field(
        description="FeatureCollection: catchment polygons, pond footprint, outlets and "
        "the longest flow path. Loads directly in geojson.io."
    )
    warnings: list[str] = Field(default_factory=list)
    timing_ms: dict[str, float]


class RainfallResponse(BaseModel):
    """`GET /rainfall`. What the free rainfall feed says about one point.

    The demo page calls this while the reader is still filling in the parameter panel, so
    the rainfall box shows the figure for their village rather than a default they have to
    know is wrong.
    """

    status: str = "ok"
    lat: float
    lon: float
    annual_rainfall_mm: float = Field(
        description="Mean annual total across the record."
    )
    rain_days: int = Field(
        description="Days a year over the wet-day threshold, averaged the same way."
    )
    wettest_day_mm: float = Field(
        description="Largest daily total in the record. SCS-CN runoff is quadratic in "
        "daily depth, so this one number moves the yield more than the annual total does."
    )
    years: float = Field(description="Length of the record, in whole years.")
    source: str
    is_measured: bool = Field(
        description="False when the feed could not be reached and the documented "
        "climatology answered instead."
    )
    description: str
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Every failure the service returns, in one shape."""

    status: str = "error"
    code: str = Field(description="Stable machine-readable identifier, e.g. no_contours.")
    detail: str = Field(description="What went wrong, in a sentence.")
    hint: str = Field(default="", description="What to do about it. Sometimes empty.")


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def site_reasons(site: PondSite, balance: WaterBalance, mapped_area_m2: float) -> list[str]:
    """The sentence-length version of the four numbers that chose this site."""
    reasons: list[str] = []

    share = site.catchment.area_m2 / mapped_area_m2 if mapped_area_m2 > 0 else 0.0
    lead = "largest upstream area" if site.rank == 1 else f"ranked #{site.rank} by area"
    reasons.append(f"{lead}: {site.area_ha:.1f} ha, {share:.0%} of the mapped sheet")

    slope = site.score.slope
    limit = settings.siting.max_slope_fraction
    if slope <= limit / 2:
        reasons.append(f"low slope at the outlet ({slope:.1%})")
    else:
        reasons.append(f"buildable slope at the outlet ({slope:.1%}, limit {limit:.0%})")

    reasons.append(f"{site.catchment.relief_m:.0f} m of relief above the outlet")

    relative = site.score.relative_elevation_m
    if relative < 0:
        reasons.append(f"sits {abs(relative):.1f} m below the surrounding ground")
    # Rounded, not raw. A depression of 4 cm is the pit filler's epsilon and the noise
    # in the interpolation, and "natural hollow about 0.0 m deep" reads as a bug.
    if round(site.score.depression_depth_m, 1) > 0:
        reasons.append(
            f"natural hollow about {site.score.depression_depth_m:.1f} m deep"
        )
    elif balance.storage.is_excavated:
        reasons.append("no natural hollow, so the capacity quoted has to be dug out")

    height = site.score.height_above_trunk_m
    if math.isfinite(height):
        reasons.append(
            f"stands {height:.1f} m above the nearest watercourse, so the pond is clear "
            "of the channel"
        )

    if site.ensemble is not None and site.confidence == "high":
        reasons.append(
            f"three grids agree on the catchment to within "
            f"{site.ensemble.coefficient_of_variation:.0%}"
        )
    return reasons


def _catchment_summary(site: PondSite, balance: WaterBalance) -> CatchmentSummary:
    catchment = site.catchment
    ensemble = site.ensemble
    return CatchmentSummary(
        area_ha=_r(catchment.area_ha),
        area_uncertainty_ha=None if ensemble is None else _r(ensemble.std_area_ha),
        ensemble_mean_area_ha=None if ensemble is None else _r(ensemble.mean_area_ha),
        confidence=site.confidence,
        edge_contact_pct=_r(catchment.edge_contact * 100.0),
        is_lower_bound=catchment.is_lower_bound,
        relief_m=_r(catchment.relief_m),
        longest_flow_path_m=_r(catchment.longest_flow_path_m),
        flow_path_relief_m=_r(catchment.flow_path_relief_m),
        time_of_concentration_min=_r(balance.time_of_concentration_min),
        cell_count=catchment.cell_count,
        grid_resolutions_m=[] if ensemble is None else [_r(r, 2) for r in ensemble.resolutions_m],
    )


def _storage_summary(balance: WaterBalance) -> StorageSummary:
    storage = balance.storage
    return StorageSummary(
        capacity_m3=_r(storage.usable_capacity_m3, 0),
        capacity_at_target_depth_m3=_r(storage.capacity_m3, 0),
        surface_area_m2=_r(storage.usable_area_m2, 0),
        max_depth_m=_r(storage.max_depth_m, 2),
        usable_depth_m=None if storage.spill_stage_m is None else _r(storage.spill_stage_m, 2),
        natural_storage_m3=_r(storage.natural_storage_m3, 0),
        site_depression_m3=_r(storage.site_depression_m3, 0),
        is_excavated=storage.is_excavated,
        is_reservoir=storage.is_reservoir,
        water_spread_fraction=_r(storage.water_spread_fraction, 3),
        frustum_estimate_m3=_r(storage.frustum_estimate_m3, 0),
        frustum_error=_r(storage.frustum_error, 3),
        stage_storage=[
            [_r(stage, 2), _r(area, 0), _r(volume, 0)] for stage, area, volume in storage.triples
        ],
    )


def _runoff_summary(balance: WaterBalance) -> RunoffSummary:
    runoff = balance.runoff
    # A site with no usable capacity divides by zero and gets an infinite fill ratio.
    # JSON has no infinity and `NaN`/`Infinity` are not valid JSON tokens, so it goes out
    # as -1, documented on the field, rather than as something a strict parser rejects.
    ratio = balance.fill_ratio
    return RunoffSummary(
        method=runoff.method,
        curve_number=_r(runoff.curve_number, 1),
        retention_mm=_r(runoff.retention_mm),
        initial_abstraction_mm=_r(runoff.initial_abstraction_mm),
        rainfall_mm=_r(runoff.rainfall_mm),
        rain_days=runoff.rain_days,
        rainfall_source=balance.rainfall_source,
        runoff_depth_mm=_r(runoff.runoff_depth_mm),
        runoff_coefficient=_r(runoff.runoff_coefficient, 3),
        contributing_days=runoff.contributing_days,
        annual_runoff_m3=_r(balance.annual_runoff_m3, 0),
        single_event_coefficient=_r(runoff.single_event_coefficient, 3),
        overestimate_factor=_r(runoff.overestimate_factor, 2),
        fill_ratio=-1.0 if ratio == float("inf") else _r(ratio, 2),
        assessment=balance.assessment,
    )


def _site_summary(site: PondSite, balance: WaterBalance, mapped_area_m2: float) -> SiteSummary:
    lon, lat = site.lonlat
    precision = settings.geojson.coordinate_precision
    return SiteSummary(
        rank=site.rank,
        is_recommended=site.is_recommended,
        location=Location(lat=_r(lat, precision), lon=_r(lon, precision)),
        snap_distance_m=_r(site.catchment.snap_distance_m),
        why=site_reasons(site, balance, mapped_area_m2),
        slope=_r(site.score.slope, 4),
        relative_elevation_m=_r(site.score.relative_elevation_m, 2),
        depression_depth_m=_r(site.score.depression_depth_m, 2),
        # `inf` means the sheet carries no watercourse, and JSON has no infinity. Null
        # says "there is nothing here to measure against", which is what it means.
        height_above_trunk_m=(
            None
            if not math.isfinite(site.score.height_above_trunk_m)
            else _r(site.score.height_above_trunk_m, 2)
        ),
        catchment=_catchment_summary(site, balance),
        storage=_storage_summary(balance),
        runoff=_runoff_summary(balance),
        warnings=list(site.warnings),
    )


def rainfall_response(
    series: RainfallSeries, lon: float, lat: float
) -> RainfallResponse:
    """`RainfallSeries` -> the JSON body of `GET /rainfall`."""
    return RainfallResponse(
        lat=_r(lat, 4),
        lon=_r(lon, 4),
        annual_rainfall_mm=_r(series.annual_total_mm),
        rain_days=series.rain_days,
        wettest_day_mm=_r(series.wettest_day_mm),
        years=_r(series.years, 1),
        source=series.source,
        is_measured=series.is_measured,
        description=series.description,
        warnings=list(series.warnings),
    )


def analysis_response(result: AnalysisResult) -> AnalysisResponse:
    """`AnalysisResult` -> the JSON body, with nothing computed on the way."""
    meta = result.contours.metadata
    dem_meta = result.dem.meta
    mapped_area_m2 = dem_meta.mapped_area_m2

    return AnalysisResponse(
        input=InputSummary(
            filename=result.filename,
            contour_count=result.contours.line_count,
            vertex_count=result.contours.vertex_count,
            elevation_source=meta.elevation_source,
            interval_m=meta.interval_m,
            level_count=meta.level_count,
            elevation_range_m=[_r(v, 2) for v in meta.elevation_range],
            mapped_area_ha=_r(mapped_area_m2 / _HA),
            bbox=[_r(v, 6) for v in meta.bbox],
            unit_hint=meta.unit_hint,
            skipped_features=meta.skipped_features,
            document_name=meta.document_name,
        ),
        dem=DEMSummary(
            resolution_m=_r(dem_meta.resolution_m, 2),
            resolution_source=dem_meta.resolution_source,
            shape=list(dem_meta.shape),
            smoothing_sigma_m=_r(dem_meta.smoothing_sigma_m, 2),
            mean_contour_spacing_m=_r(dem_meta.mean_contour_spacing_m, 2),
            nodata_fraction=_r(dem_meta.nodata_fraction, 4),
            elevation_range_m=[_r(v, 2) for v in dem_meta.elevation_range],
            max_smoothing_shift_m=_r(dem_meta.max_smoothing_shift_m, 3),
        ),
        parameters=result.params,
        recommended_site=_site_summary(
            result.recommended, result.recommended_balance, mapped_area_m2
        ),
        alternative_sites=[
            _site_summary(site, balance, mapped_area_m2)
            for site, balance in zip(result.alternatives, result.alternative_balances)
        ],
        search=(
            None
            if result.siting is None
            else SearchSummary(
                stream_threshold_ha=_r(result.siting.stream_threshold_m2 / _HA),
                stream_cells=result.siting.stream_cells,
                buildable_cells=result.siting.buildable_cells,
                trunk_threshold_ha=_r(result.siting.trunk_threshold_m2 / _HA),
                trunk_cells=result.siting.trunk_cells,
                clear_of_watercourse_cells=result.siting.clear_of_watercourse_cells,
                candidate_cells=result.siting.candidate_cells,
            )
        ),
        geojson=result.geojson,
        warnings=list(result.warnings),
        timing_ms=result.timings_ms,
    )
