"""The one place the stages are wired together.

`app/core/` holds seven modules that each do one thing and know nothing about HTTP. This
is where they meet: bytes in, an `AnalysisResult` out. The route above it does validation
and error mapping; the schemas beside it do presentation. Neither contains a step of the
analysis, and nothing here knows what a status code is. That is what makes the whole
pipeline runnable from a test, a notebook or a future CLI without a server (PLAN §6).

Two decisions in here are worth the reader's attention.

**The primary grid and the ensemble are not the same grid.** The reported catchment is
delineated on the data-derived grid (3.1 m on the sample sheet); the error bar comes from
three further grids at 5.0 / 3.5 / 2.5 m. Siting on the ensemble's coarsest member instead
would save a flow field and cost the resolution the methodology was validated at, so the
extra field is paid for deliberately (PLAN §3 Test C).

**The triangulation is built once.** `ContourSurface` costs about a second on 159,113
vertices and does not depend on the grid, so every grid samples the same surface: the
primary one and all three ensemble members. Rebuilding it per grid would quadruple the
most expensive step in the request for no change in the answer.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from app.config import Settings, settings
from app.core.catchment import CatchmentEnsemble
from app.core.dem_builder import DEM, ContourSurface
from app.core.geojson import build_geojson
from app.core.hydrology import WaterBalance, water_balance
from app.core.kml_parser import ContourSet, parse_contours
from app.core.network import stream_network
from app.core.pond_siting import PondSite, PondSiteSelector, SitingError, SitingResult
from app.core.raster_dem import AreaOfInterest, RasterSurface
from app.core.terrain import D8TerrainEngine, FlowField, TerrainEngine
from app.providers.elevation import ElevationProvider
from app.providers.rainfall import RainfallProvider, RainfallSeries, rainfall_for
from app.schemas.requests import AnalysisParams

__all__ = ["AnalysisError", "AnalysisResult", "Stopwatch", "analyse", "analyse_area", "ProgressFn"]

ProgressFn = Callable[[str], None]
"""Told the name of each stage as it starts. The job API turns this into a progress bar
that reports what the analysis is actually doing rather than animating a guess."""


class AnalysisError(Exception):
    """A request that cannot be answered for a reason the core modules do not raise.

    Carries the same `(code, detail, hint)` triple as `ContourParseError` and its
    siblings, so the route maps every failure in the service through one path.
    """

    def __init__(self, code: str, detail: str, hint: str = "") -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.hint = hint


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
class Stopwatch:
    """Per-stage wall-clock timings, reported so a slow request can be diagnosed.

    Wall clock rather than CPU time on purpose: what a client waited for is the number
    that matters, and the numpy inside these stages is threaded.
    """

    def __init__(self, on_stage: ProgressFn | None = None) -> None:
        self.timings_ms: dict[str, float] = {}
        self._started = time.perf_counter()
        self._on_stage = on_stage

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self._on_stage is not None:
            self._on_stage(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            # Recorded even when the stage raises, so a timeout or a crash still says
            # which step consumed the request.
            self.timings_ms[name] = round((time.perf_counter() - start) * 1e3, 1)

    def finish(self) -> dict[str, float]:
        elapsed = (time.perf_counter() - self._started) * 1e3
        return {**self.timings_ms, "total": round(elapsed, 1)}


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnalysisResult:
    """Everything one analysis produced, still as core objects.

    Deliberately not JSON: the masks and grids in here are what Phase 10's map and the
    validation suite want, and flattening them to numbers is the response schema's job,
    not the pipeline's.
    """

    filename: str
    params: AnalysisParams
    contours: ContourSet | None
    """The parsed sheet on the contour path; `None` when the area was drawn on a map."""

    surface: ContourSurface | RasterSurface
    dem: DEM
    flow: FlowField
    rainfall: RainfallSeries
    sites: tuple[PondSite, ...]
    balances: tuple[WaterBalance, ...]
    """One water balance per site, index-aligned with `sites`."""

    geojson: dict
    siting: SitingResult | None
    """The search that produced the sites, or `None` when the client named a pour point
    and there was no search."""

    ensemble_resolutions_m: tuple[float, ...]
    """Empty when the ensemble was switched off. Every site's confidence then
    reads `unassessed`, which is the honest answer rather than a default of `high`."""

    warnings: tuple[str, ...]
    timings_ms: dict[str, float]

    aoi: AreaOfInterest | None = None
    """What the user drew, on the map path. `None` for an uploaded sheet."""

    network: dict | None = None
    """The drainage network as a FeatureCollection, for the map's flow layer."""

    @property
    def source(self) -> str:
        return "contour_file" if self.aoi is None else "elevation_tiles"

    @property
    def extent_bbox(self) -> tuple[float, float, float, float]:
        """The ground the analysis covered: the sheet, or the buffered selection."""
        if self.contours is not None:
            return tuple(self.contours.metadata.bbox)  # type: ignore[return-value]
        return tuple(self.surface.analysed_bbox)  # type: ignore[union-attr]

    @property
    def recommended(self) -> PondSite:
        """Rank 1, which is not the same as recommendable. Check `is_recommended`: a site the
        ensemble rejects keeps its rank and is returned flagged, never reordered away."""
        return self.sites[0]

    @property
    def recommended_balance(self) -> WaterBalance:
        return self.balances[0]

    @property
    def alternatives(self) -> tuple[PondSite, ...]:
        return self.sites[1:]

    @property
    def alternative_balances(self) -> tuple[WaterBalance, ...]:
        return self.balances[1:]


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
def _dedupe(messages: list[str]) -> tuple[str, ...]:
    """Order-preserving unique. Several stages warn about the same clipped edge, and a
    response repeating one sentence four times reads like a bug."""
    seen: set[str] = set()
    out: list[str] = []
    for message in messages:
        if message not in seen:
            seen.add(message)
            out.append(message)
    return tuple(out)


def _check_pour_point(lon: float, lat: float, contours: ContourSet) -> None:
    """Reject a point outside the sheet before the DEM is built.

    The delineator catches this too, but only after the grid and the flow field have been
    computed. Checking against the contour bounding box costs nothing and turns fifteen
    seconds of work into an immediate answer.
    """
    min_lon, min_lat, max_lon, max_lat = contours.metadata.bbox
    if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
        raise AnalysisError(
            "pour_point_outside_map",
            f"({lon:g}, {lat:g}) is outside the mapped sheet, which spans "
            f"{min_lon:.6f}..{max_lon:.6f} E and {min_lat:.6f}..{max_lat:.6f} N.",
            "Omit lat and lon to let the service choose a site from the terrain.",
        )


def analyse(
    data: bytes,
    filename: str = "upload.kml",
    params: AnalysisParams | None = None,
    *,
    rainfall_provider: RainfallProvider | None = None,
    engine: TerrainEngine | None = None,
    config: Settings | None = None,
    progress: ProgressFn | None = None,
) -> AnalysisResult:
    """Contour bytes to a complete answer.

    `rainfall_provider` replaces the live feed rather than the fallback, so a test can
    hand in a fixed series without reaching the network, and a caller-stated rainfall
    figure still takes precedence over both.

    Raises the core modules' structured errors unchanged. Those are `ContourParseError`,
    `DEMBuildError`, `SitingError`, `HydrologyError` and `GeoJSONError`, plus
    `AnalysisError` for the one failure that belongs to the wiring rather than to a
    stage. The route turns each into a status code; none of them are caught here,
    because a partial analysis is not a useful thing to return.
    """
    cfg = config or settings
    params = params or AnalysisParams()
    engine = engine or D8TerrainEngine(cfg.terrain)
    watch = Stopwatch(progress)

    # ---- 1. Contours ------------------------------------------------- #
    with watch.stage("parse"):
        contours = parse_contours(data, filename, config=cfg.parser)

    pour_point = params.pour_point
    if pour_point is not None:
        _check_pour_point(*pour_point, contours)

    # ---- 2. DEM ------------------------------------------------------ #
    with watch.stage("dem"):
        surface = ContourSurface(contours, config=cfg.dem)
        dem = surface.sample(params.grid_resolution)

    # ---- 3-7. Everything downstream of the DEM ------------------------ #
    tail = _downstream(
        surface, dem, params, watch, cfg, engine, rainfall_provider,
        ensemble_resolutions=None if not params.ensemble else (),
    )
    return AnalysisResult(
        filename=filename,
        params=params,
        contours=contours,
        surface=surface,
        dem=dem,
        warnings=_collect_warnings(
            contours.metadata.warnings, dem, tail["siting"], tail["sites"], tail["balances"]
        ),
        timings_ms=watch.finish(),
        **_result_fields(tail),
    )


def analyse_area(
    aoi: AreaOfInterest,
    params: AnalysisParams | None = None,
    *,
    provider: ElevationProvider | None = None,
    exclusion_mask: Callable[[DEM], np.ndarray | None] | None = None,
    rainfall_provider: RainfallProvider | None = None,
    engine: TerrainEngine | None = None,
    config: Settings | None = None,
    progress: ProgressFn | None = None,
) -> AnalysisResult:
    """A selection drawn on the map to a complete answer (PLAN3 §5).

    Only the first two stages differ from `analyse`: elevation tiles instead of a parsed
    sheet, `RasterSurface` instead of `ContourSurface`. The DEM they produce goes through
    the same flow, siting, runoff and GeoJSON code, untouched.

    The DEM covers the selection plus a buffer, and the pond is kept inside the selection
    by handing the siting step the outside as excluded ground. `exclusion_mask`, when
    given, adds more ground to leave alone (the land-availability layer), as a function of
    the DEM because the grid is not known until the tiles are in.
    """
    cfg = config or settings
    params = params or AnalysisParams()
    engine = engine or D8TerrainEngine(cfg.terrain)
    watch = Stopwatch(progress)

    # ---- 1-2. Elevation --------------------------------------------- #
    with watch.stage("elevation"):
        surface = RasterSurface(aoi, provider=provider, config=cfg.elevation)
        dem = surface.sample(params.grid_resolution)
        inside = surface.inside_mask(dem)

    extra_warnings: list[str] = [
        f"Elevation is {surface.fetches[0].get('provider', 'tile')} data at about "
        f"{cfg.elevation.source_resolution_m:.0f} m, resampled to a {dem.resolution_m:.1f} m "
        "grid. Heights are to the nearest metre, so storage depths and small catchments "
        "are approximate; upload a surveyed contour sheet for a design-grade answer."
    ]

    pour_point = params.pour_point
    if pour_point is not None:
        min_lon, min_lat, max_lon, max_lat = surface.analysed_bbox
        lon, lat = pour_point
        if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
            raise AnalysisError(
                "pour_point_outside_map",
                f"({lon:g}, {lat:g}) is outside the analysed area, which spans "
                f"{min_lon:.6f}..{max_lon:.6f} E and {min_lat:.6f}..{max_lat:.6f} N.",
                "Put the point inside the area you drew, or omit lat and lon.",
            )
        if not aoi.contains(lon, lat):
            extra_warnings.append(
                "The pour point you named lies outside the area you drew, in the buffer "
                "around it. It was analysed anyway."
            )

    exclusion = ~inside
    if exclusion_mask is not None:
        with watch.stage("land"):
            unavailable = exclusion_mask(dem)
        if unavailable is not None:
            exclusion = exclusion | unavailable

    # The ensemble costs three more flow fields. On this path they are small, so it runs
    # whenever the grid is under the budget, even on a host where the contour path's
    # ensemble is refused; above the budget it is skipped with a warning, not refused.
    ensemble_resolutions: tuple[float, ...] | None = None
    if params.ensemble:
        cells = dem.shape[0] * dem.shape[1]
        if cells <= cfg.elevation.ensemble_max_cells:
            ensemble_resolutions = tuple(
                min(max(dem.resolution_m * f, cfg.elevation.min_resolution_m),
                    cfg.elevation.max_resolution_m)
                for f in cfg.elevation.ensemble_factors
            )
        else:
            extra_warnings.append(
                f"The resolution ensemble was skipped: the grid has {cells:,} cells, over "
                f"the {cfg.elevation.ensemble_max_cells:,} this host cross-checks. Draw a "
                "smaller area for an error bar on the catchment."
            )

    try:
        tail = _downstream(
            surface, dem, params, watch, cfg, engine, rainfall_provider,
            ensemble_resolutions=ensemble_resolutions, exclusion=exclusion,
        )
    except SitingError as exc:
        if exc.code == "no_available_ground":
            raise AnalysisError(
                "no_site_in_selection",
                "The terrain has pond sites nearby, but none inside the area you drew"
                + (" that is also available land." if exclusion_mask is not None else "."),
                "Draw a larger area, or move it along the valley. Sites need a channel "
                "draining at least 0.5% of the analysed ground, on gentle slope.",
            ) from exc
        raise

    return AnalysisResult(
        filename="selected area",
        params=params,
        contours=None,
        surface=surface,
        dem=dem,
        warnings=_collect_warnings(
            tuple(extra_warnings), dem, tail["siting"], tail["sites"], tail["balances"]
        ),
        timings_ms=watch.finish(),
        aoi=aoi,
        **_result_fields(tail),
    )


def _downstream(
    surface,
    dem: DEM,
    params: AnalysisParams,
    watch: Stopwatch,
    cfg: Settings,
    engine: TerrainEngine,
    rainfall_provider: RainfallProvider | None,
    *,
    ensemble_resolutions: tuple[float, ...] | None,
    exclusion: np.ndarray | None = None,
) -> dict:
    """Stages 3 to 7, shared by both entry points: whatever made the DEM, this is the
    analysis. `ensemble_resolutions` is `None` for no ensemble, `()` for the configured
    grids, or explicit resolutions."""
    # ---- 3. Flow routing --------------------------------------------- #
    with watch.stage("flow"):
        flow = engine.analyse(dem)

    # ---- 4. The error bar -------------------------------------------- #
    ensemble = None
    if ensemble_resolutions is not None:
        with watch.stage("ensemble"):
            ensemble = CatchmentEnsemble(
                surface,
                resolutions_m=ensemble_resolutions or None,
                engine=engine,
                config=cfg.catchment,
            )

    # ---- 5. Siting --------------------------------------------------- #
    selector = PondSiteSelector(
        flow, config=cfg.siting, ensemble=ensemble, exclusion_mask=exclusion
    )
    siting: SitingResult | None = None
    pour_point = params.pour_point
    with watch.stage("siting"):
        if pour_point is not None:
            sites = (_site_at(selector, *pour_point),)
        else:
            siting = selector.select(params.top_n)
            sites = siting.sites

    # ---- 6. Water ---------------------------------------------------- #
    with watch.stage("hydrology"):
        # Rainfall is fetched for the *chosen site*, which is why this stage comes after
        # siting rather than before it. A figure the caller stated wins; otherwise ten
        # years of Open-Meteo records for that point answer, and the documented
        # climatology stands behind both in case the service cannot be reached.
        rainfall = rainfall_for(
            *sites[0].lonlat,
            annual_total_mm=params.rainfall_mm,
            rain_days=params.rain_days,
            live=rainfall_provider,
            config=cfg.hydrology,
        )
        balances = tuple(
            water_balance(
                flow,
                site.catchment,
                rainfall,
                curve_number=params.curve_number,
                target_depth_m=params.target_depth_m,
                config=cfg.hydrology,
            )
            for site in sites
        )

    # ---- 7. Geometry ------------------------------------------------- #
    with watch.stage("geojson"):
        geojson = build_geojson(flow, sites, balances, config=cfg.geojson)
        network = stream_network(
            flow,
            selector.stream_threshold_m2,
            trunk_threshold_m2=selector.trunk_threshold_m2,
            config=cfg.geojson,
        )

    return {
        "flow": flow,
        "rainfall": rainfall,
        "sites": sites,
        "balances": balances,
        "geojson": geojson,
        "siting": siting,
        "network": network,
        "ensemble_resolutions_m": (
            tuple(ensemble.resolutions_m) if ensemble is not None else ()
        ),
    }


def _result_fields(tail: dict) -> dict:
    return {
        key: tail[key]
        for key in (
            "flow", "rainfall", "sites", "balances", "geojson", "siting", "network",
            "ensemble_resolutions_m",
        )
    }


def _site_at(selector: PondSiteSelector, lon: float, lat: float) -> PondSite:
    """Delineate a client-named pour point, converting the delineator's `ValueError`.

    The bounding-box check above catches a point off the sheet; this catches the subtler
    case of a point inside the box but on no data, such as a corner the contour hull
    does not reach. That is only knowable once the grid exists.
    """
    try:
        return selector.site_at(lon, lat)
    except ValueError as exc:
        raise AnalysisError(
            "pour_point_unusable",
            str(exc),
            "The point falls on ground the contours do not cover. Move it towards the "
            "middle of the sheet, or omit lat and lon to let the service choose.",
        ) from exc


def _collect_warnings(
    source_warnings: tuple[str, ...],
    dem: DEM,
    siting: SitingResult | None,
    sites: tuple[PondSite, ...],
    balances: tuple[WaterBalance, ...],
) -> tuple[str, ...]:
    """The caveats that apply to the answer as a whole.

    Per-site caveats stay on their site, because a clipped catchment on alternative 3
    says nothing about the recommendation. Only the recommended site's warnings are
    promoted here, alongside those of the file, the grid and the search.
    """
    messages: list[str] = []
    messages.extend(source_warnings)
    messages.extend(dem.meta.warnings)
    if siting is not None:
        messages.extend(siting.warnings)
    messages.extend(sites[0].warnings)
    messages.extend(balances[0].warnings)

    if not sites[0].is_recommended:
        messages.append(
            "The highest-ranked site is not recommended: the resolution ensemble does "
            "not agree on its catchment. Read the alternatives before acting."
        )
    return _dedupe(messages)
