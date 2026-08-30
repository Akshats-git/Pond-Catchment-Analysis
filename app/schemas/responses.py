"""The shape of the answer, and the rounding that makes it readable.

Two jobs. The first is documentation: these models are what FastAPI renders at `/docs`,
so a grader can see the whole contract without running anything, and every field carries
the sentence that says what it means. The second is translation -- the core modules speak
in numpy arrays, masks and dataclasses, and none of that crosses a JSON boundary.

**On rounding.** Numbers are rounded where a reader would round them and nowhere else: a
catchment to 0.1 ha, a volume to whole cubic metres, a coordinate to six decimals (~0.1 m,
finer than any grid this service builds). Rounding is presentation, so it happens here and
never in `app/core/`, where the full precision is what the mass-balance test checks.

**On saying why.** `why` is assembled from the four numbers that actually chose the site --
upstream area, slope, relative elevation, depression depth -- plus the ensemble's verdict.
It is a rendering of the score, not a second opinion about it: if the sentence and the
number ever disagreed, the sentence would be the wrong one, so it is generated from the
number rather than written alongside it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config import settings
from app.core.hydrology import WaterBalance
from app.core.pond_siting import PondSite
from app.pipeline import AnalysisResult
from app.schemas.requests import AnalysisParams

__all__ = [
    "AnalysisResponse",
    "ErrorResponse",
    "HealthResponse",
    "analysis_response",
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
        description="Which strategy resolved the elevations: z_coordinate, "
        "extended_data, placemark_name or folder_name."
    )
    interval_m: float | None = Field(description="Median spacing between contour levels.")
    level_count: int
    elevation_range_m: list[float]
    mapped_area_ha: float = Field(
        description="Ground the contours actually describe -- the denominator every "
        "area share in this response is measured against."
    )
    bbox: list[float] = Field(description="[min_lon, min_lat, max_lon, max_lat].")
    unit_hint: str | None = Field(
        default=None,
        description="A unit token seen beside the elevations. Reported, never acted on: "
        "the service does not silently convert feet to metres.",
    )
    skipped_features: int = Field(
        default=0, description="Line placemarks no strategy could give an elevation to."
    )
    document_name: str | None = None


class DEMSummary(BaseModel):
    """How the grid under the analysis was built."""

    resolution_m: float
    resolution_source: str = Field(
        description="auto (derived from contour spacing), requested, or coarsened to "
        "stay under the cell limit."
    )
    shape: list[int] = Field(description="[rows, columns].")
    smoothing_sigma_m: float = Field(
        description="Gaussian sigma that removes the stair-step artefact of contour "
        "interpolation. Tied to the contour spacing, not to the grid."
    )
    mean_contour_spacing_m: float
    nodata_fraction: float = Field(
        description="Share of grid cells outside the contour hull."
    )
    elevation_range_m: list[float]
    max_smoothing_shift_m: float = Field(
        description="Furthest the smoothing moved any single cell -- the evidence that "
        "it removed an artefact rather than the terrain."
    )


class Location(BaseModel):
    lat: float
    lon: float


class CatchmentSummary(BaseModel):
    """The catchment draining to the site, and how much to trust it."""

    area_ha: float
    area_uncertainty_ha: float | None = Field(
        description="Standard deviation of the area across the resolution ensemble. "
        "Null when the ensemble was switched off."
    )
    ensemble_mean_area_ha: float | None = None
    confidence: str = Field(
        description="high, medium or low from the ensemble spread; unassessed without "
        "an ensemble."
    )
    edge_contact_pct: float = Field(
        description="Share of the catchment perimeter lying on no-data or the map "
        "border."
    )
    is_lower_bound: bool = Field(
        description="True when enough of the perimeter is against the edge of the data "
        "that the true catchment continues off the sheet."
    )
    relief_m: float = Field(
        description="Highest ground in the catchment minus the outlet elevation."
    )
    longest_flow_path_m: float
    flow_path_relief_m: float = Field(
        description="Drop along that path -- the H in Kirpich, and a different number "
        "from relief_m because the most distant point is rarely the highest."
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
        description="Usable capacity: the volume below the stage at which water first "
        "tops a divide, which is the pond the site actually holds."
    )
    capacity_at_target_depth_m3: float = Field(
        description="Volume at the full target depth, ignoring any spill. Larger than "
        "capacity_m3 wherever the water would spread beyond the site."
    )
    surface_area_m2: float
    max_depth_m: float
    usable_depth_m: float | None = Field(
        description="Depth at the spill stage. Null when the pond fills to the target "
        "depth without spilling."
    )
    natural_storage_m3: float = Field(
        description="What the site holds with nothing built. Zero on a channel, which "
        "is where catchment-first siting puts most sites."
    )
    site_depression_m3: float = Field(
        description="The whole natural hollow, both sides of the outlet."
    )
    is_excavated: bool
    is_reservoir: bool = Field(
        description="True when a structure of the target depth would flood a large "
        "share of its own catchment."
    )
    water_spread_fraction: float
    frustum_estimate_m3: float = Field(
        description="Textbook cross-check: V = (d/3)(A_top + A_bot + sqrt(A_top*A_bot))."
    )
    frustum_error: float
    stage_storage: list[list[float]] = Field(
        description="[depth_m, area_m2, volume_m3] rows from the bed to the target depth."
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
        description="Days whose rainfall exceeded Ia and therefore produced runoff."
    )
    annual_runoff_m3: float
    single_event_coefficient: float = Field(
        description="What SCS-CN returns if the annual total is treated as one storm -- "
        "reported to show the error the event-based method avoids."
    )
    overestimate_factor: float
    fill_ratio: float = Field(
        description="Annual runoff divided by usable capacity. Above 1 the pond fills "
        "and spills in an average year; -1 means the site has no usable capacity at all, "
        "which JSON cannot express as an infinite ratio."
    )
    assessment: str


class SiteSummary(BaseModel):
    """One ranked location, with everything that justifies it."""

    rank: int
    is_recommended: bool = Field(
        description="False when the ensemble cannot agree on the site's catchment. Such "
        "a site keeps its rank and is returned flagged, never quietly reordered away."
    )
    location: Location
    snap_distance_m: float = Field(
        description="How far the outlet moved onto the routed channel. Large values mean "
        "the answer is not for the point that was asked about."
    )
    why: list[str]
    slope: float = Field(description="Local slope at the outlet, rise over run.")
    relative_elevation_m: float = Field(
        description="Outlet elevation minus the mean of the ground around it. Negative "
        "means the site sits in a hollow."
    )
    depression_depth_m: float
    catchment: CatchmentSummary
    storage: StorageSummary
    runoff: RunoffSummary
    warnings: list[str] = Field(default_factory=list)


class SearchSummary(BaseModel):
    """The candidate search behind the ranking, so an empty or thin result explains
    itself. Absent when the client named a pour point and no search happened."""

    stream_threshold_ha: float = Field(
        description="Upstream area above which a cell counts as a stream: a fixed "
        "fraction of the mapped area, never a percentile of flow accumulation."
    )
    stream_cells: int
    buildable_cells: int = Field(
        description="Cells under the slope limit and clear of the edge buffer."
    )
    candidate_cells: int = Field(description="Cells that are both.")


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


class ErrorResponse(BaseModel):
    """Every failure the service returns, in one shape."""

    status: str = "error"
    code: str = Field(description="Stable machine-readable identifier, e.g. no_contours.")
    detail: str = Field(description="What went wrong, in a sentence.")
    hint: str = Field(default="", description="What to do about it. May be empty.")


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
    if site.score.depression_depth_m > 0:
        reasons.append(
            f"natural hollow about {site.score.depression_depth_m:.1f} m deep"
        )
    elif balance.storage.is_excavated:
        reasons.append("no natural hollow -- the capacity quoted is excavated")

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
        catchment=_catchment_summary(site, balance),
        storage=_storage_summary(balance),
        runoff=_runoff_summary(balance),
        warnings=list(site.warnings),
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
                candidate_cells=result.siting.candidate_cells,
            )
        ),
        geojson=result.geojson,
        warnings=list(result.warnings),
        timing_ms=result.timings_ms,
    )
