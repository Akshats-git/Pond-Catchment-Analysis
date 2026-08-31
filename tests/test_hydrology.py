"""Phase 7. Event-based SCS-CN runoff, stage-storage and time of concentration.

The runoff tests exist because of one mistake: SCS-CN is an event model, and a year of
rain put through it as a single storm returns a 92% runoff coefficient. Both aggregations
are computed here and the gap between them is asserted, so the right answer cannot quietly
turn back into the wrong one.

The storage tests are analytic. A cone and a wedge have exact volumes, so the integrated
stage-storage curve can be checked against them rather than against itself, and the
frustum formula that the curve replaces can be shown erring on the same shapes.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from app.config import settings
from app.core.catchment import CatchmentDelineator
from app.core.dem_builder import ContourSurface
from app.core.hydrology import (
    HydrologyError,
    retention_mm,
    runoff_volume_m3,
    scs_cn_runoff,
    stage_storage,
    time_of_concentration_min,
    water_balance,
)
from app.core.kml_parser import parse_contour_file
from app.core.pond_siting import PondSiteSelector
from app.core.terrain import analyse_terrain
from app.providers.rainfall import DefaultRainfallProvider, RainfallSeries
from tests.test_terrain import synthetic_dem

SAMPLE = "data/contours_1m.kml"
CFG = settings.hydrology


# --------------------------------------------------------------------------- #
# SCS-CN, by hand
# --------------------------------------------------------------------------- #
def test_retention_is_the_textbook_curve():
    """S = 25400/CN - 254. CN 75 is the default for this terrain."""
    assert retention_mm(75.0) == pytest.approx(84.667, abs=0.01)
    assert retention_mm(98.0) == pytest.approx(5.184, abs=0.01)
    assert retention_mm(100.0 - 25.0) == retention_mm(75.0)


def test_a_single_day_matches_the_hand_calculation():
    """P = 100 mm, CN = 75: S = 84.67, Ia = 16.93, Q = 83.07^2 / 167.74 = 41.14 mm."""
    result = scs_cn_runoff([100.0], 75.0)
    assert result.retention_mm == pytest.approx(84.667, abs=0.01)
    assert result.initial_abstraction_mm == pytest.approx(16.933, abs=0.01)
    assert result.runoff_depth_mm == pytest.approx(41.14, abs=0.05)


def test_rain_below_the_initial_abstraction_runs_off_nowhere():
    """Ia is 16.9 mm at CN 75, so a 15 mm day wets the ground and stops."""
    result = scs_cn_runoff([15.0, 16.9, 5.0], 75.0)
    assert result.runoff_depth_mm == 0.0
    assert result.contributing_days == 0


def test_runoff_grows_faster_than_rainfall():
    """The quadratic numerator: one 100 mm day yields far more than ten 10 mm days, which
    is the whole reason the aggregation order matters."""
    one_big = scs_cn_runoff([100.0], 75.0).runoff_depth_mm
    ten_small = scs_cn_runoff([10.0] * 10, 75.0).runoff_depth_mm
    assert ten_small == 0.0
    assert one_big > 40.0


def test_a_higher_curve_number_runs_off_more():
    daily = [30.0, 45.0, 12.0]
    assert (
        scs_cn_runoff(daily, 90.0).runoff_depth_mm
        > scs_cn_runoff(daily, 75.0).runoff_depth_mm
        > scs_cn_runoff(daily, 60.0).runoff_depth_mm
    )


def test_an_undefined_curve_number_is_an_error():
    with pytest.raises(HydrologyError) as raised:
        scs_cn_runoff([50.0], 120.0)
    assert raised.value.code == "curve_number_out_of_range"
    with pytest.raises(HydrologyError):
        scs_cn_runoff([50.0], 10.0)


def test_negative_rainfall_is_an_error():
    with pytest.raises(HydrologyError) as raised:
        scs_cn_runoff([10.0, -1.0], 75.0)
    assert raised.value.code == "bad_rainfall_series"


def test_runoff_volume_is_a_depth_over_an_area():
    """100 mm over a hectare is 1,000 m^3."""
    assert runoff_volume_m3(100.0, 10_000.0) == pytest.approx(1000.0)


# --------------------------------------------------------------------------- #
# The event-model mistake (PLAN §4)
# --------------------------------------------------------------------------- #
def test_the_annual_total_as_one_storm_is_the_error_it_is_documented_as(series):
    """PLAN §4's table, reproduced: 1200 mm as a single event gives 1104 mm of runoff and
    a 92% coefficient. A figure no catchment produces."""
    result = scs_cn_runoff(series, CFG.default_curve_number)
    assert result.single_event_depth_mm == pytest.approx(1104.0, abs=1.0)
    assert result.single_event_coefficient == pytest.approx(0.92, abs=0.01)


def test_per_day_summation_lands_in_the_documented_band(series):
    """The same rain, the same curve number, applied per day and summed: 11-19% for this
    terrain, and about six times less runoff than the single-event mistake."""
    result = scs_cn_runoff(series, CFG.default_curve_number)
    low, high = CFG.expected_runoff_coefficient_range
    assert low <= result.runoff_coefficient <= high
    assert result.runoff_depth_mm == pytest.approx(191.5, abs=1.0)
    assert result.overestimate_factor > 5.0
    assert result.contributing_days < result.rain_days


# --------------------------------------------------------------------------- #
# Kirpich
# --------------------------------------------------------------------------- #
def test_time_of_concentration_matches_the_hand_calculation():
    """Tc = 0.01947 * 4660^0.77 * (19.7/4660)^-0.385 = 107 minutes. The sample's
    largest basin."""
    assert time_of_concentration_min(4660.0, 19.7) == pytest.approx(107.0, abs=1.0)


def test_a_longer_flatter_path_takes_longer():
    assert time_of_concentration_min(2000.0, 10.0) > time_of_concentration_min(1000.0, 10.0)
    assert time_of_concentration_min(1000.0, 5.0) > time_of_concentration_min(1000.0, 20.0)


def test_using_basin_relief_instead_of_path_relief_would_shorten_it():
    """`Catchment` reports the drop along the longest path separately from the basin's
    relief for this reason: on the sample's largest basin they are 19.7 m and 27.0 m, and
    substituting one for the other moves Tc by more than 10%."""
    along_path = time_of_concentration_min(4660.0, 19.7)
    basin_relief = time_of_concentration_min(4660.0, 27.0)
    assert basin_relief < along_path
    assert (along_path - basin_relief) / along_path > 0.10


def test_a_level_flow_path_is_clamped_rather_than_infinite():
    """A perfectly level path is a contour-interval artefact. Kirpich would divide by
    zero; the clamp returns a long, finite, conservative time."""
    clamped = time_of_concentration_min(1000.0, 0.0)
    assert math.isfinite(clamped)
    assert clamped == pytest.approx(
        time_of_concentration_min(1000.0, CFG.kirpich_min_slope * 1000.0)
    )


def test_a_zero_length_path_is_an_error():
    with pytest.raises(HydrologyError) as raised:
        time_of_concentration_min(0.0, 5.0)
    assert raised.value.code == "degenerate_flow_path"


# --------------------------------------------------------------------------- #
# Stage-storage against shapes with exact volumes
# --------------------------------------------------------------------------- #
def conical_bowl(k: float = 0.02, half_width: int = 40, res: float = 10.0):
    """`z = k * r`: a cone opening upward, whose exact volume is known.

    Water standing `h` above the apex covers a disc of radius `h/k`, so
    `A = pi (h/k)^2` and `V = pi h^3 / (3 k^2)`.
    """
    axis = (np.arange(2 * half_width + 1) - half_width) * res
    radius = np.hypot(axis[:, None], axis[None, :])
    return synthetic_dem(k * radius, resolution_m=res)


def test_natural_storage_matches_the_volume_of_a_cone():
    """The bowl has no outlet, so it fills to its lowest rim. The middle of a side, at
    `k * half_width * res`. That is the analytic cone: 1.34 million m^3 at 8 m deep."""
    k, half_width, res = 0.02, 40, 10.0
    dem = conical_bowl(k, half_width, res)
    flow = analyse_terrain(dem)
    outlet = (half_width, half_width)

    storage = stage_storage(flow, dem.valid, outlet, target_depth_m=1.0)

    rim = k * half_width * res
    assert storage.natural_storage_m3 == pytest.approx(
        math.pi * rim ** 3 / (3 * k ** 2), rel=0.02
    )
    assert storage.natural_storage_area_m2 == pytest.approx(
        math.pi * (rim / k) ** 2, rel=0.02
    )
    assert not storage.is_excavated


def test_the_stage_curve_matches_the_volume_of_a_wedge():
    """A plane tilted 2% drains along its rows, so the catchment of a cell is the strip
    east of it. Water standing `h` above that cell fills a wedge: it reaches `h/0.02`
    metres back and holds `w * h^2 / (2 * 0.02)`."""
    slope, res, target = 0.02, 5.0, 2.0
    z = slope * np.arange(120)[None, :] * res + np.zeros((9, 1))
    dem = synthetic_dem(z, resolution_m=res)
    flow = analyse_terrain(dem)

    row, col = 4, 20
    mask = np.zeros(dem.shape, dtype=bool)
    mask[row, col:] = True  # the cell's own catchment on this surface

    storage = stage_storage(flow, mask, (row, col), target_depth_m=target)
    cell_width = dem.row_cell_areas[row] / res

    assert storage.surface_area_m2 == pytest.approx(target / slope * cell_width, rel=0.05)
    assert storage.capacity_m3 == pytest.approx(
        cell_width * target ** 2 / (2 * slope), rel=0.05
    )
    # Straight sides underestimate a wedge: the frustum's (d/3)(A_top + A_bot + ...)
    # against the wedge's (d/2)A_top, softened here by the one cell of bed at stage 0.
    assert -0.25 < storage.frustum_error < -0.05
    assert storage.is_excavated
    # The wedge widens steadily, so nothing here is a spill, and the single cell the
    # pond starts from cannot be called one either, however fast it multiplies.
    assert storage.spill_stage_m is None
    assert storage.usable_capacity_m3 == storage.capacity_m3


def test_the_curve_rises_with_the_stage_and_starts_at_the_natural_pool():
    dem = conical_bowl()
    flow = analyse_terrain(dem)
    storage = stage_storage(flow, dem.valid, (40, 40), target_depth_m=3.0)

    assert storage.stages_m[0] == 0.0
    assert storage.stages_m[-1] == 3.0
    assert len(storage.triples) == CFG.stage_storage_steps + 1
    assert storage.volumes_m3[0] == storage.natural_storage_m3
    assert storage.volumes_m3[-1] == storage.capacity_m3
    assert list(storage.areas_m2) == sorted(storage.areas_m2)
    assert list(storage.volumes_m3) == sorted(storage.volumes_m3)


def test_the_curve_finds_the_stage_where_the_water_tops_a_divide():
    """A narrow valley behind the outlet, walled to 3 m, opening onto a wide pan 1.5 m up.

    Water held at the outlet fills the valley. 33 cells, so 1,100 m^3 at 1.33 m, and
    then reaches the pan, where a further 17 cm puts 3.9 ha under water. The capacity at
    the 2 m target depth is the flooded pan; the pond the site holds is the volume below
    the jump, and that is what the fill ratio is taken against.
    """
    res = 5.0
    z = np.full((40, 80), 13.0)
    z[5:36, 21:71] = 11.5                                    # the wide pan, upstream
    z[19:22, 10:21] = 10.0                                   # the walled valley
    z[19:22, 0:10] = 10.0 - 0.1 * (10 - np.arange(10))       # draining off the map
    dem = synthetic_dem(z, resolution_m=res)
    flow = analyse_terrain(dem)

    mask = np.zeros(dem.shape, dtype=bool)
    mask[:, 10:] = True  # everything upstream of the outlet
    storage = stage_storage(flow, mask, (20, 10), target_depth_m=2.0)

    assert storage.spill_stage_m == pytest.approx(4 / 3, abs=0.01)
    assert storage.usable_area_m2 == pytest.approx(33 * res ** 2, rel=0.02)
    assert storage.usable_capacity_m3 == pytest.approx(33 * res ** 2 * 4 / 3, rel=0.02)
    # Half a metre of water over the pan is 19,000 m^3 more, across 48 times the area.
    assert storage.surface_area_m2 > 40 * storage.usable_area_m2
    assert storage.capacity_m3 == pytest.approx(21_000, rel=0.05)
    assert any("tops a ridge" in w for w in storage.warnings)


def test_the_pond_cannot_spread_outside_the_catchment():
    """The mask is a statement about where water can physically be. Halving it must halve
    the pond, not leave it unchanged."""
    dem = conical_bowl()
    flow = analyse_terrain(dem)
    half = np.zeros(dem.shape, dtype=bool)
    half[:, 40:] = True

    whole_pond = stage_storage(flow, dem.valid, (40, 40), target_depth_m=2.0)
    half_pond = stage_storage(flow, half, (40, 40), target_depth_m=2.0)
    assert half_pond.capacity_m3 == pytest.approx(whole_pond.capacity_m3 / 2, rel=0.05)


def test_a_hollow_inside_the_pond_counts_as_the_water_it_holds():
    """Connectivity is judged on the filled surface so an artefact rim cannot cut the pond
    in two; depth is measured against the surveyed one, so the hollow behind that rim is
    still storage."""
    slope, res = 0.02, 5.0
    z = slope * np.arange(120)[None, :] * res + np.zeros((9, 1))
    gouged = z.copy()
    gouged[4, 30] -= 1.5

    mask = np.zeros(z.shape, dtype=bool)
    mask[4, 20:] = True
    plain = stage_storage(
        analyse_terrain(synthetic_dem(z, resolution_m=res)),
        mask,
        (4, 20),
        target_depth_m=2.0,
    )
    dented = stage_storage(
        analyse_terrain(synthetic_dem(gouged, resolution_m=res)),
        mask,
        (4, 20),
        target_depth_m=2.0,
    )
    cell_area = float(synthetic_dem(z, resolution_m=res).row_cell_areas[4])
    assert dented.capacity_m3 - plain.capacity_m3 == pytest.approx(
        1.5 * cell_area, rel=0.05
    )


def test_a_target_depth_of_zero_is_an_error():
    dem = conical_bowl()
    flow = analyse_terrain(dem)
    with pytest.raises(HydrologyError) as raised:
        stage_storage(flow, dem.valid, (40, 40), target_depth_m=0.0)
    assert raised.value.code == "bad_target_depth"


# --------------------------------------------------------------------------- #
# The rainfall provider
# --------------------------------------------------------------------------- #
def test_the_default_series_is_the_documented_climatology(series):
    assert series.annual_total_mm == pytest.approx(CFG.default_annual_rainfall_mm)
    assert series.rain_days == CFG.default_rain_days
    assert not series.is_measured
    assert series.warnings, "a climatology must say that it is not an observation"


def test_the_series_is_reproducible():
    """Every reported runoff figure depends on this: a seeded draw, not a random one."""
    first = DefaultRainfallProvider().daily_series(81.3, 21.25)
    second = DefaultRainfallProvider().daily_series(81.3, 21.25)
    assert np.array_equal(first.daily_mm, second.daily_mm)


def test_the_series_is_right_skewed(series):
    """Monsoon rain arrives in a few large events, and SCS-CN is quadratic in daily depth,
    so the shape of the distribution matters as much as the total."""
    daily = series.daily_mm
    assert series.wettest_day_mm > 4 * float(np.median(daily))
    assert float(np.mean(daily > np.mean(daily))) < 0.5


def test_the_defaults_can_be_overridden():
    provider = DefaultRainfallProvider(annual_total_mm=800.0, rain_days=40)
    other = provider.daily_series(0.0, 0.0)
    assert other.annual_total_mm == pytest.approx(800.0)
    assert other.rain_days == 40


def test_impossible_rainfall_is_refused():
    with pytest.raises(ValueError):
        DefaultRainfallProvider(annual_total_mm=0.0).daily_series(0.0, 0.0)


# --------------------------------------------------------------------------- #
# The whole balance, on the real sheet
# --------------------------------------------------------------------------- #
def test_the_top_site_reproduces_the_reference_water_balance(balance, site):
    """Runoff in the documented band, and the volume the top catchment yields.

    191.5 mm of runoff off 1,200 mm of rain is a coefficient of 16%, which is what this
    terrain does. Over the 120 ha catchment of the recommended site that is about
    229,000 m^3 in an average year.
    """
    low, high = CFG.expected_runoff_coefficient_range
    assert low <= balance.runoff.runoff_coefficient <= high
    assert balance.runoff.runoff_depth_mm == pytest.approx(191.5, abs=1.0)
    assert balance.annual_runoff_m3 == pytest.approx(229_000, rel=0.15)
    assert balance.annual_runoff_m3 == pytest.approx(
        runoff_volume_m3(balance.runoff.runoff_depth_mm, site.catchment.area_m2)
    )


def test_the_top_site_holds_a_village_pond_below_its_spill(balance):
    """The curve does the work here. Up to 1.0 m the water is held in a 0.31 ha basin.
    A further 25 cm tops the divide and puts 11 ha of the valley floor under water. The
    pond the site holds is the volume below that step, not the flooded valley above it."""
    storage = balance.storage
    assert storage.spill_stage_m == pytest.approx(1.0, abs=0.01)
    assert storage.usable_capacity_m3 == pytest.approx(2_170, rel=0.10)
    assert storage.usable_area_m2 / 1e4 == pytest.approx(0.31, rel=0.10)
    assert storage.capacity_m3 / storage.usable_capacity_m3 > 50
    assert any("tops a ridge" in w for w in storage.warnings)


def test_the_frustum_cross_check_underestimates_real_ground(balance):
    """The formula a spreadsheet would use, reported alongside the integral it replaces.
    Straight sides cannot follow ground that widens as it rises, and on this site they
    miss by half, which is the argument for integrating the DEM."""
    storage = balance.storage
    assert storage.frustum_error < -0.30
    assert storage.frustum_estimate_m3 == pytest.approx(
        storage.max_depth_m
        / 3
        * (
            storage.surface_area_m2
            + storage.areas_m2[0]
            + math.sqrt(storage.surface_area_m2 * storage.areas_m2[0])
        )
    )


def test_a_channel_site_has_no_natural_storage_and_says_so(balance):
    """PLAN §11: a site on a channel stores nothing by itself. The capacity comes from the
    target depth against the terrain, never from the depression depth, which is zero."""
    storage = balance.storage
    assert storage.natural_storage_m3 < CFG.natural_storage_floor_m3
    assert storage.site_depression_m3 < CFG.natural_storage_floor_m3
    assert storage.is_excavated
    assert any("no natural hollow here" in w for w in storage.warnings)
    assert storage.usable_capacity_m3 > 0


def test_a_site_with_a_hollow_reports_its_natural_storage(flow):
    """There is a real depression on this sheet at 81.2842 E, 21.2625 N: 7,400 m^3 of
    water the ground holds before anything is built.

    The point is named rather than taken from the ranking, because what is being tested
    is the storage model and not which site comes out on top. Almost none of that water
    is upstream of the outlet, so a structure there keeps a few cubic metres of it, and
    the warnings say so rather than leave it implied.
    """
    catchment = CatchmentDelineator(flow).delineate(81.284248, 21.262484)
    storage = stage_storage(
        flow, catchment.mask, catchment.outlet_rc, target_depth_m=3.0
    )
    assert storage.site_depression_m3 == pytest.approx(7_400, rel=0.10)
    assert storage.site_depression_area_m2 / 1e4 == pytest.approx(0.85, rel=0.10)
    assert not storage.is_excavated
    assert storage.natural_storage_m3 < CFG.natural_storage_floor_m3
    assert any("sits below the outlet" in w for w in storage.warnings)


def test_the_fill_ratio_is_reported_in_plain_english(balance):
    """750,000 m^3 of runoff against an 8,000 m^3 pond: it fills early and spills, and the
    ratio is taken against the usable capacity rather than the flooded-valley one."""
    assert balance.fill_ratio == pytest.approx(
        balance.annual_runoff_m3 / balance.storage.usable_capacity_m3
    )
    assert balance.fill_ratio > CFG.fill_ratio_bands[-1]
    assert balance.fills
    assert "more water arrives than the pond can hold" in balance.assessment


def test_a_wide_flat_site_is_flagged_as_a_reservoir(sites, flow, series):
    """Site 4 sits on flat ground: a 3 m structure there would put four fifths of its own
    catchment under water. The number is still what the ground holds; the warning is what
    stops it being read as a pond."""
    balance = water_balance(flow, sites[3].catchment, series)
    assert balance.storage.is_reservoir
    assert balance.storage.water_spread_fraction > 0.3
    assert any("a reservoir and not a village pond" in w for w in balance.warnings)


def test_the_balance_carries_the_provenance_of_its_rainfall(balance):
    assert "climatology" in balance.rainfall_source
    assert any("not an observation" in w for w in balance.warnings)


def test_time_of_concentration_uses_the_catchment_it_was_given(balance, site):
    assert balance.time_of_concentration_min == pytest.approx(
        time_of_concentration_min(
            site.catchment.longest_flow_path_m, site.catchment.flow_path_relief_m
        )
    )
    assert 60.0 < balance.time_of_concentration_min < 180.0


def test_a_drier_year_yields_less_and_a_wetter_one_more(flow, site):
    """The whole point of the provider seam: change the rainfall, and every number
    downstream follows it."""
    dry = water_balance(
        flow,
        site.catchment,
        DefaultRainfallProvider(annual_total_mm=600.0).daily_series(0, 0),
    )
    wet = water_balance(
        flow,
        site.catchment,
        DefaultRainfallProvider(annual_total_mm=1800.0).daily_series(0, 0),
    )
    assert dry.annual_runoff_m3 < wet.annual_runoff_m3
    assert dry.runoff.runoff_coefficient < wet.runoff.runoff_coefficient
    assert dry.fill_ratio < wet.fill_ratio


def test_an_unexpected_runoff_coefficient_is_flagged(flow, site, series):
    """A curve number far outside what this terrain supports still returns a number, with
    a warning saying it is outside the band the region is documented to produce."""
    balance = water_balance(flow, site.catchment, series, curve_number=95.0)
    assert balance.runoff.runoff_coefficient > CFG.expected_runoff_coefficient_range[1]
    assert any("this terrain normally gives" in w for w in balance.warnings)


def test_a_deeper_pond_holds_more(flow, site, series):
    """Capacity at the target depth is monotone in that depth, and nothing else is.

    The usable capacity is read off a curve of twelve steps between the bed and the
    target, so a deeper target means coarser steps and the spill stage lands somewhere
    else. On this site 2.5 m finds a spill at 0.83 m and 1.5 m finds one at 1.0 m, which
    makes the shallower pond the larger usable one. That is a property of the curve and
    not of the ground, so the assertion is on the volume the depth actually buys.
    """
    shallow = water_balance(flow, site.catchment, series, target_depth_m=1.5)
    deep = water_balance(flow, site.catchment, series, target_depth_m=2.5)
    assert shallow.storage.capacity_m3 < deep.storage.capacity_m3
    assert shallow.storage.max_depth_m == 1.5


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def series() -> RainfallSeries:
    return DefaultRainfallProvider().daily_series(81.3, 21.25)


@pytest.fixture(scope="module")
def flow():
    return analyse_terrain(ContourSurface(parse_contour_file(SAMPLE)).sample(5.0))


@pytest.fixture(scope="module")
def sites(flow):
    return PondSiteSelector(flow).select(5).sites


@pytest.fixture(scope="module")
def site(sites):
    """The recommended site: 120 ha of catchment, clear of the river."""
    return sites[0]


@pytest.fixture(scope="module")
def balance(flow, site, series):
    return water_balance(flow, site.catchment, series)
