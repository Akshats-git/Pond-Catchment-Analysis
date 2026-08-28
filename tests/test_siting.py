"""Phase 6 -- catchment-first pond siting.

Two kinds of evidence here. On synthetic surfaces the right answer is derivable by hand:
a symmetric valley has exactly one buildable channel, and the best site on it is the
furthest-downstream cell the edge buffer allows -- so the selector can be checked cell by
cell rather than plausibility-checked.

On the real sheet the tests are about the two mistakes this phase exists to avoid. Both
are reproduced rather than asserted: that percentile-ranked accumulation would put a
sub-threshold hollow in the top 2% (PLAN §11.6), and that square-window suppression
returns five nested points on one stream (PLAN §11.7). A test that only checks the right
answer would still pass after somebody reintroduced either.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.config import settings
from app.core.catchment import CatchmentEnsemble
from app.core.dem_builder import ContourSurface
from app.core.kml_parser import parse_contour_file
from app.core.pond_siting import (
    PondSiteSelector,
    SitingError,
    select_pond_sites,
)
from app.core.terrain import D8TerrainEngine, analyse_terrain
from tests.test_terrain import synthetic_dem

SAMPLE = "data/contours_1m.kml"

# PLAN §3, "Results on the provided contour map": the prototype run this phase has to
# reproduce. (area_ha, ensemble_mean_ha, ensemble_std_ha, confidence)
PUBLISHED_SITES = [
    (395.4, 423.5, 17.6, "high"),
    (101.8, 102.8, 1.5, "high"),
    (37.1, 37.3, 0.1, "high"),
    (35.7, 14.1, 15.4, "low"),
    (30.4, 31.6, 0.8, "high"),
]


# --------------------------------------------------------------------------- #
# Synthetic surfaces with a derivable answer
# --------------------------------------------------------------------------- #
def valley(
    channels: tuple[int, ...] = (20,),
    *,
    ny: int = 40,
    nx: int = 41,
    res: float = 10.0,
    cross: float = 0.05,
    along: float = 0.01,
):
    """A V-shaped valley (or several) draining south, as a bare DEM.

    `z = cross * (distance to the nearest channel) + along * y`. The cross slope is the
    steeper one, so every hillslope cell runs to its channel and then down it: the
    catchment of a channel cell at row `r` is provably every cell at row >= r that is
    nearer to this channel than to any other.
    """
    rows = np.arange(ny)[:, None] * res
    cols = np.arange(nx)[None, :] * res
    distance = np.min([np.abs(cols - c * res) for c in channels], axis=0)
    return synthetic_dem(cross * distance + along * rows, resolution_m=res)


def selector_for(dem, **overrides) -> PondSiteSelector:
    config = replace(settings.siting, **overrides) if overrides else None
    return PondSiteSelector(analyse_terrain(dem), config=config)


def test_only_the_channel_is_a_candidate():
    """The hillslopes carry water but stand at 5%; the channel is the only ground that
    is both a stream and buildable."""
    selector = selector_for(valley())
    candidates = selector.candidate_mask()

    assert candidates.any()
    assert set(np.flatnonzero(candidates.any(axis=0))) == {20}
    # The hillslope cells next to the channel do carry more than the 0.5% threshold --
    # it is the slope rule, not the stream rule, that excludes them.
    assert selector.stream_mask()[:, 19].any()
    assert not selector.buildable_mask()[:, 19].any()


def test_the_best_site_is_the_lowest_channel_cell_the_buffer_allows():
    """Ranking is by catchment area, so the pick is as far downstream as it may go: the
    30 m buffer keeps it two cells (2 x 10 m, plus the border itself) off the edge."""
    dem = valley()
    selector = selector_for(dem)
    site = selector.select(1).sites[0]

    assert site.catchment.outlet_rc == (2, 20)
    # Everything from row 2 up: 38 rows of 41 cells.
    assert site.catchment.cell_count == 38 * 41
    assert site.catchment.area_m2 == pytest.approx(dem.area_of(site.catchment.mask))


def test_no_site_sits_inside_the_edge_buffer():
    dem = valley()
    selector = selector_for(dem)
    distance = selector.distance_to_edge_m()

    for site in selector.select(3).sites:
        assert distance[site.catchment.outlet_rc] >= settings.siting.edge_buffer_m


def test_two_valleys_give_two_independent_basins():
    """The point of suppression: alternatives are different basins, not different points
    on one stream."""
    selector = selector_for(valley(channels=(10, 30)))
    sites = selector.select(2).sites

    assert [s.catchment.outlet_rc for s in sites] == [(2, 10), (2, 30)]
    assert not (sites[0].catchment.mask & sites[1].catchment.mask).any()
    # Not exactly equal: with 41 columns the divide falls on a cell, which one valley
    # keeps and the other does not.
    assert sites[0].catchment.area_m2 == pytest.approx(sites[1].catchment.area_m2, rel=0.10)


def test_ranking_is_by_descending_catchment_area():
    selector = selector_for(valley(channels=(6, 20, 34)))
    areas = [s.catchment.area_m2 for s in selector.select(3).sites]
    assert areas == sorted(areas, reverse=True)


def test_asking_for_more_basins_than_exist_says_so():
    """One valley holds one independent site; the rest of its channel is its own
    catchment. That is a warning, not an error."""
    result = selector_for(valley()).select(3)

    assert len(result.sites) == 1
    assert result.warnings and "independent basin" in result.warnings[0]


def test_top_n_is_clamped_to_the_configured_maximum():
    selector = selector_for(valley(channels=tuple(range(2, 40, 4))))
    assert len(selector.select(50).sites) <= settings.siting.max_top_n


def test_selection_is_deterministic():
    """Equal-area basins are broken by cell index, so the same sheet always gives the
    same list -- a report that changes between runs cannot be checked."""
    dem = valley(channels=(10, 30))
    first = selector_for(dem).select(2)
    second = selector_for(dem).select(2)
    assert [s.catchment.outlet_rc for s in first.sites] == [
        s.catchment.outlet_rc for s in second.sites
    ]


def test_the_score_describes_the_cell_that_was_chosen():
    selector = selector_for(valley())
    site = selector.select(1).sites[0]
    row, col = site.catchment.outlet_rc

    assert site.score.upstream_area_m2 == pytest.approx(site.catchment.area_m2)
    assert site.score.slope == pytest.approx(selector.flow.slope[row, col])
    # A monotone surface has no hollows to fill, so the depression depth is zero rather
    # than the priority-flood epsilon.
    assert site.score.depression_depth_m == pytest.approx(0.0, abs=1e-3)
    # And a channel sits below the ground on either side of it.
    assert site.score.relative_elevation_m < 0


def test_a_depression_is_measured_to_its_spill_point():
    """A hollow gouged 2 m into the channel holds water to the level it spills at, not to
    the level it was dug from: the cell below it is already 0.1 m lower, so the depression
    is 1.9 m deep. That is the depth Phase 7's natural storage follows from."""
    dem = valley()
    z = dem.z.copy()
    z[20, 20] -= 2.0
    selector = selector_for(synthetic_dem(z, resolution_m=dem.resolution_m))
    catchment = selector.delineator.delineate_cell(20, 20)

    assert selector.score_cell(20, 20, catchment).depression_depth_m == pytest.approx(
        1.9, abs=0.01
    )


def test_a_hillside_with_no_stream_is_an_explicit_error():
    """A long featureless slope: every cell drains east on its own, nothing collects
    0.5% of the map, and the answer is a named error rather than an empty list."""
    res = 10.0
    z = 0.05 * np.arange(6)[None, :] * res + np.zeros((300, 1))
    with pytest.raises(SitingError) as raised:
        selector_for(synthetic_dem(z, resolution_m=res)).select()
    assert raised.value.code == "no_stream_network"


def test_terrain_too_steep_to_dam_is_an_explicit_error():
    with pytest.raises(SitingError) as raised:
        selector_for(valley(cross=0.5, along=0.2)).select()
    assert raised.value.code == "no_buildable_ground"


# --------------------------------------------------------------------------- #
# The two pitfalls, reproduced on the real sheet
# --------------------------------------------------------------------------- #
def test_the_stream_threshold_is_half_a_percent_of_the_mapped_area(selector):
    """Derived from the input, not hard-coded (PLAN §9)."""
    assert selector.stream_threshold_m2 == pytest.approx(
        selector.mapped_area_m2 * settings.siting.stream_threshold_fraction
    )
    assert selector.stream_threshold_m2 / 1e4 == pytest.approx(4.15, abs=0.05)


def test_percentile_ranked_accumulation_would_be_degenerate(selector):
    """PLAN §11.6, reproduced rather than asserted.

    Flow accumulation is so skewed that the 98th percentile of it is a hollow draining
    under half the stream threshold -- ranking on percentiles scores such a hollow
    alongside a 320 ha valley. The absolute threshold is what stops that.
    """
    upstream = selector.upstream_area()[selector.dem.valid]
    assert np.quantile(upstream, 0.98) < 0.5 * selector.stream_threshold_m2
    # And the threshold itself is far out in the tail: streams are rare.
    on_stream = float((upstream >= selector.stream_threshold_m2).mean())
    assert on_stream < 0.02


def test_square_window_suppression_returns_one_stream_five_times(flow):
    """PLAN §11.7. With catchment suppression off, the selector reproduces the failure
    that motivated it: five nested points strung along the main valley."""
    windowed = PondSiteSelector(
        flow, config=replace(settings.siting, suppression_removes_catchment=False)
    ).select(5)
    masks = [s.catchment.mask for s in windowed.sites]

    nested = [
        (i, j)
        for i in range(len(masks))
        for j in range(i + 1, len(masks))
        if (masks[i] & masks[j]).any()
    ]
    assert len(nested) == 10, "every pair should overlap -- that is the bug"
    assert all(s.catchment.area_ha > 300 for s in windowed.sites)


def test_catchment_suppression_returns_independent_basins(sites):
    """The fix, on the same sheet: no cell belongs to two recommended catchments."""
    masks = [s.catchment.mask for s in sites]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            assert not (masks[i] & masks[j]).any()


# --------------------------------------------------------------------------- #
# The published results table
# --------------------------------------------------------------------------- #
def test_reproduces_the_published_results_table(sites):
    """PLAN §3, "Results on the provided contour map".

    Sites 1, 3 and 4 come back on the published cell. Sites 2 and 5 come back one or two
    cells upstream of it: the published outlets sit 20-25 m from the edge of the data and
    the 30 m buffer of `SitingConfig.edge_buffer_m` excludes them, so the selector takes
    the next cell up the same channel. The basins are the same basins -- each published
    outlet's catchment contains the selected one -- which is why the tolerance below is
    on the area rather than on the cell.
    """
    assert len(sites) == len(PUBLISHED_SITES)
    for site, (area_ha, mean_ha, std_ha, confidence) in zip(sites, PUBLISHED_SITES):
        assert site.catchment.area_ha == pytest.approx(area_ha, rel=0.20)
        assert site.ensemble.mean_area_ha == pytest.approx(mean_ha, rel=0.15)
        assert site.ensemble.std_area_ha == pytest.approx(std_ha, abs=1.5)
        assert site.confidence == confidence


def test_the_published_sites_are_the_same_basins(sites, delineator, flow):
    """Each selected site drains the basin the prototype reported.

    Two independent checks: the selected outlet and the published one ultimately leave
    the map at the same cell, and their catchments share at least 80% of their area.
    Where the selector differs it has moved a cell or two along the same channel, not
    found somewhere else."""
    terminal = flow.terminal_outlets()
    published_lonlat = [
        (81.286465, 21.240094),
        (81.293549, 21.263343),
        (81.284248, 21.262484),
        (81.297453, 21.240094),
        (81.312393, 21.259544),
    ]
    for site, (lon, lat) in zip(sites, published_lonlat):
        reference = delineator.delineate(lon, lat, snap=False)
        assert terminal[site.catchment.outlet_rc] == terminal[reference.outlet_rc]
        shared = int((site.catchment.mask & reference.mask).sum())
        assert shared / max(site.catchment.cell_count, reference.cell_count) >= 0.80


def test_the_top_site_drains_almost_half_the_sheet(sites, selector):
    best = sites[0]
    assert best.rank == 1
    assert best.catchment.area_m2 / selector.mapped_area_m2 == pytest.approx(0.47, abs=0.03)
    assert best.confidence == "high" and best.is_recommended


def test_site_four_is_rejected_by_the_ensemble(sites):
    """The acceptance criterion of Phase 6. On the primary grid site 4 is an ordinary
    35.7 ha basin; the three grids put it at 14.1 +/- 15.4 ha, so it is returned flagged
    rather than recommended."""
    site = sites[3]
    assert site.catchment.area_ha == pytest.approx(35.7, rel=0.05)
    assert site.ensemble.coefficient_of_variation > 1.0
    assert site.confidence == "low"
    assert not site.is_recommended
    assert any("disagree" in w for w in site.warnings)


def test_edge_contact_is_reported_as_a_warning_not_a_rejection(sites):
    """A clipped catchment is still a real place for a pond; its area is a lower bound.
    Only the ensemble vetoes."""
    for site in sites:
        assert site.catchment.is_lower_bound == any(
            "lower bound" in w for w in site.warnings
        )
        if site.catchment.is_lower_bound and site.confidence != "low":
            assert site.is_recommended


def test_every_selected_site_obeys_the_siting_rules(sites, selector):
    distance = selector.distance_to_edge_m()
    for site in sites:
        row, col = site.catchment.outlet_rc
        assert site.score.slope < settings.siting.max_slope_fraction
        assert distance[row, col] >= settings.siting.edge_buffer_m
        assert site.score.upstream_area_m2 >= selector.stream_threshold_m2
        assert not selector.dem.nodata[row, col]


def test_the_convenience_wrapper_matches_the_selector(flow):
    result = select_pond_sites(flow, top_n=2)
    assert [s.catchment.outlet_rc for s in result.sites] == [
        s.catchment.outlet_rc for s in PondSiteSelector(flow).select(2).sites
    ]
    assert all(s.confidence == "unassessed" for s in result.sites)
    assert result.resolution_m == flow.dem.resolution_m


# --------------------------------------------------------------------------- #
# Fixtures -- the sample sheet is parsed once for the module
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def surface():
    return ContourSurface(parse_contour_file(SAMPLE))


@pytest.fixture(scope="module")
def ensemble(surface):
    return CatchmentEnsemble(surface)


@pytest.fixture(scope="module")
def delineator(ensemble):
    return ensemble.primary


@pytest.fixture(scope="module")
def flow(delineator):
    return delineator.flow


@pytest.fixture(scope="module")
def selector(ensemble):
    return PondSiteSelector.from_ensemble(ensemble)


@pytest.fixture(scope="module")
def sites(selector):
    """The five sites of PLAN §3, delineated on the 5 m grid and cross-checked on the
    2.5 m and 3.5 m ones."""
    return selector.select(5).sites
