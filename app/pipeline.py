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
from typing import Iterator

from app.config import Settings, settings
from app.core.catchment import CatchmentEnsemble
from app.core.dem_builder import DEM, ContourSurface
from app.core.geojson import build_geojson
from app.core.hydrology import WaterBalance, water_balance
from app.core.kml_parser import ContourSet, parse_contours
from app.core.pond_siting import PondSite, PondSiteSelector, SitingResult
from app.core.terrain import D8TerrainEngine, FlowField, TerrainEngine
from app.providers.rainfall import RainfallProvider, RainfallSeries, rainfall_for
from app.schemas.requests import AnalysisParams

__all__ = ["AnalysisError", "AnalysisResult", "Stopwatch", "analyse"]


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

    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
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
    contours: ContourSet
    surface: ContourSurface
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
    watch = Stopwatch()

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

    # ---- 3. Flow routing --------------------------------------------- #
    with watch.stage("flow"):
        flow = engine.analyse(dem)

    # ---- 4. The error bar -------------------------------------------- #
    ensemble = None
    if params.ensemble:
        with watch.stage("ensemble"):
            ensemble = CatchmentEnsemble(surface, engine=engine, config=cfg.catchment)

    # ---- 5. Siting --------------------------------------------------- #
    selector = PondSiteSelector(flow, config=cfg.siting, ensemble=ensemble)
    siting: SitingResult | None = None
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

    return AnalysisResult(
        filename=filename,
        params=params,
        contours=contours,
        surface=surface,
        dem=dem,
        flow=flow,
        rainfall=rainfall,
        sites=sites,
        balances=balances,
        geojson=geojson,
        siting=siting,
        ensemble_resolutions_m=(
            tuple(ensemble.resolutions_m) if ensemble is not None else ()
        ),
        warnings=_collect_warnings(contours, dem, siting, sites, balances),
        timings_ms=watch.finish(),
    )


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
    contours: ContourSet,
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
    messages.extend(contours.metadata.warnings)
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
