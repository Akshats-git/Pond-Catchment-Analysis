"""Phase 6. Catchment-first pond siting.

Two kinds of evidence here. On synthetic surfaces the right answer is derivable by hand:
a symmetric valley has exactly one buildable channel, and the best site on it is the
furthest-downstream cell the edge buffer allows, so the selector can be checked cell by
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

# The five sites the sample sheet returns on the 5 m ensemble grid, as
# (area_ha, ensemble_mean_ha, ensemble_std_ha, confidence).
#
# These replaced the table in PLAN §3. That table was produced before the watercourse
# rule existed, so its first site was the Shivnath itself: 395 ha of a 831 ha sheet,
# outlet in the middle of the river. Every entry below stands clear of the river, which
# is the whole reason the numbers moved.
REFERENCE_SITES = [
    (119.7, 120.3, 0.7, "high"),
    (37.5, 63.7, 17.9, "medium"),
    (18.4, 21.3, 2.0, "high"),
    (18.1, 18.3, 0.1, "high"),
    (12.9, 13.1, 1.2, "high"),
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
    # The hillslope cells next to the channel do carry more than the 0.5% threshold.
    # It is the slope rule, not the stream rule, that excludes them.
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


# --------------------------------------------------------------------------- #
# Ground the terrain cannot rule out
#
# Contours say where water collects, never whether that spot is a house or a field
# somebody owns. `exclusion_mask` is where that answer arrives from outside, and these
# tests pin it on the valley whose right answer is derivable: the channel is one column,
# so what excluding part of it should do is knowable cell by cell.
# --------------------------------------------------------------------------- #
def test_no_exclusion_mask_changes_nothing():
    """The seam has to be free when unused, or it is a cost paid by every caller."""
    dem = valley()
    plain = selector_for(dem).select(1).sites[0]
    passed_none = PondSiteSelector(analyse_terrain(dem), exclusion_mask=None)

    assert passed_none.select(1).sites[0].catchment.outlet_rc == plain.catchment.outlet_rc
    assert passed_none.available_mask().all()


def test_excluded_ground_is_not_sited_on():
    """Rule out the cell the selector would otherwise pick and it picks another. The
    catchment is still delineated on the real terrain: only the choice of outlet moves."""
    dem = valley()
    chosen = selector_for(dem).select(1).sites[0].catchment.outlet_rc

    mask = np.zeros(dem.shape, dtype=bool)
    mask[chosen] = True
    site = PondSiteSelector(analyse_terrain(dem), exclusion_mask=mask).select(1).sites[0]

    assert site.catchment.outlet_rc != chosen
    assert not mask[site.catchment.outlet_rc]


def test_excluding_the_whole_channel_says_which_rule_emptied_the_pool():
    """The failure has to name the exclusion rather than blame the terrain, because the
    terrain is unchanged and only the caller can act on it."""
    dem = valley()
    selector = selector_for(dem)
    mask = selector.candidate_mask().copy()

    with pytest.raises(SitingError) as caught:
        PondSiteSelector(analyse_terrain(dem), exclusion_mask=mask).select(1)

    assert caught.value.code == "no_available_ground"
    assert caught.value.hint


def test_a_mask_of_the_wrong_shape_is_refused_with_both_shapes():
    """A caller bug numpy would otherwise broadcast into a quietly wrong site."""
    dem = valley()
    with pytest.raises(SitingError) as caught:
        PondSiteSelector(analyse_terrain(dem), exclusion_mask=np.zeros((3, 3), bool))

    assert caught.value.code == "exclusion_mask_shape"
    assert "(3, 3)" in caught.value.detail and str(dem.shape) in caught.value.detail


def test_the_convenience_wrapper_carries_the_mask_through():
    """`select_pond_sites` is the entry point the pipeline uses, so the seam is only
    real if it survives that hop."""
    dem = valley()
    flow = analyse_terrain(dem)
    chosen = select_pond_sites(flow, top_n=1).sites[0].catchment.outlet_rc

    mask = np.zeros(dem.shape, dtype=bool)
    mask[chosen] = True

    assert select_pond_sites(flow, top_n=1, exclusion_mask=mask).sites[
        0
    ].catchment.outlet_rc != chosen


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
    same list. A report that changes between runs cannot be checked."""
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
    under half the stream threshold. Ranking on percentiles scores such a hollow
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
    assert len(nested) == 10, "every pair should overlap, which is the bug"
    assert all(s.catchment.area_ha > 90 for s in windowed.sites)


def test_catchment_suppression_returns_independent_basins(sites):
    """The fix, on the same sheet: no cell belongs to two recommended catchments."""
    masks = [s.catchment.mask for s in sites]
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            assert not (masks[i] & masks[j]).any()


# --------------------------------------------------------------------------- #
# The reference results
# --------------------------------------------------------------------------- #
def test_reproduces_the_reference_results(sites):
    """The five sites the sample sheet returns, area and error bar together.

    The tolerance is on the area rather than on the cell. An outlet can move a cell or
    two up the same channel when anything about the grid changes, and that is not a
    different answer; a basin that changes size by a fifth is.
    """
    assert len(sites) == len(REFERENCE_SITES)
    for site, (area_ha, mean_ha, std_ha, confidence) in zip(sites, REFERENCE_SITES):
        assert site.catchment.area_ha == pytest.approx(area_ha, rel=0.20)
        assert site.ensemble.mean_area_ha == pytest.approx(mean_ha, rel=0.15)
        assert site.ensemble.std_area_ha == pytest.approx(std_ha, abs=1.5)
        assert site.confidence == confidence


def test_no_site_stands_in_the_river(sites, selector):
    """The bug this rule exists for.

    Ranking by catchment area alone asks for the cell that the most water passes through,
    and on a sheet with a river across it that cell is the river. The old top site was
    395 ha of a 831 ha sheet with its outlet in the Shivnath. Every site now stands clear
    of the trunk and a pond depth above it.
    """
    trunk = selector.trunk_mask()
    assert trunk.any(), "the sample sheet does carry a watercourse over the threshold"

    for site in sites:
        row, col = site.catchment.outlet_rc
        assert not trunk[row, col]
        assert (
            site.score.height_above_trunk_m
            >= settings.siting.min_height_above_trunk_m
        )
        # A cell on the trunk is one the rule already excluded, so nothing that survives
        # it can drain more than the trunk threshold.
        assert site.catchment.area_m2 < selector.trunk_threshold_m2


def test_the_trunk_is_the_river_and_only_the_river(selector):
    """291 cells of a 133,000-cell grid: one channel, not a network.

    The threshold is 150 ha of drainage, and on this sheet exactly one line crosses it.
    That line is the Shivnath, which is what makes the exclusion narrow enough to be
    safe. It removes a river, not the drainage network the ponds are meant to sit on.
    """
    trunk = selector.trunk_mask()
    stream = selector.stream_mask()
    assert 0 < trunk.sum() < 0.10 * stream.sum()
    # Every trunk cell is on a stream by construction: the trunk threshold is 36 times
    # the stream threshold.
    assert not (trunk & ~stream).any()


def test_a_sheet_with_no_watercourse_is_left_alone(selector):
    """A farm-scale map has no channel over 150 ha, so the rule has nothing to exclude.

    The threshold is absolute hectares for exactly this reason. A share of the sheet
    would call the biggest gully on a 20 ha map a river and refuse to site anything.
    """
    small = selector_for(valley())
    assert not small.trunk_mask().any()
    assert np.isinf(small.height_above_trunk()).all()
    assert (small.clear_of_watercourse_mask() == small.dem.valid).all()


def test_ground_that_drains_off_the_sheet_is_not_assumed_clear(selector):
    """A cell whose water leaves the map before reaching the trunk gets no clearance.

    Its channel continues past the edge of the data and nothing here says how far below
    it runs. On this sheet those cells are the strip along the near bank of the river,
    which is precisely where a site must not go.
    """
    heights = selector.height_above_trunk()
    off_map = np.isneginf(heights)

    assert off_map.any(), "the river leaves this sheet, so some ground drains past it"
    assert not (off_map & selector.clear_of_watercourse_mask()).any()
    # And nothing on this sheet is marked "no watercourse below me", which is the value
    # reserved for a map that carries no trunk at all.
    assert not np.isposinf(heights).any()


def test_the_ensemble_and_not_the_area_decides_confidence(sites):
    """A site is flagged on the spread between grids, never on how big it is.

    Site 2 is the demonstration: 37.5 ha on the primary grid, 63.7 +/- 17.9 ha across the
    three, a coefficient of variation of 0.28. That is a medium, and it is what a reader
    needs to know before acting on the number.
    """
    for site in sites:
        cv = site.ensemble.coefficient_of_variation
        if cv <= settings.catchment.confidence_high_cv:
            assert site.confidence == "high"
        elif cv <= settings.catchment.confidence_medium_cv:
            assert site.confidence == "medium"
        else:
            assert site.confidence == "low"
            assert not site.is_recommended
            assert any("disagree" in w for w in site.warnings)

    spread = sites[1]
    assert spread.confidence == "medium"
    assert spread.ensemble.coefficient_of_variation > settings.catchment.confidence_high_cv


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
        assert (
            site.score.height_above_trunk_m
            >= settings.siting.min_height_above_trunk_m
        )
        assert not selector.dem.nodata[row, col]


def test_the_convenience_wrapper_matches_the_selector(flow):
    result = select_pond_sites(flow, top_n=2)
    assert [s.catchment.outlet_rc for s in result.sites] == [
        s.catchment.outlet_rc for s in PondSiteSelector(flow).select(2).sites
    ]
    assert all(s.confidence == "unassessed" for s in result.sites)
    assert result.resolution_m == flow.dem.resolution_m


# --------------------------------------------------------------------------- #
# Fixtures. The sample sheet is parsed once for the module
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
