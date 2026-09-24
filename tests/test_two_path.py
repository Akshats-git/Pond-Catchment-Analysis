"""Phase 17. The SRTM path validated against the 1 m contour survey.

`docs/validate_two_path.py` produces the published table (docs/VALIDATION.md). These tests
pin its findings, so a change that makes the free-data path disagree with the survey fails
here rather than quietly changing a number in a document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline import analyse
from app.schemas.requests import AnalysisParams
from docs import validate_two_path as v


@pytest.fixture(scope="module")
def contour():
    return analyse(Path(v.SHEET).read_bytes(), "contours_1m.kml", AnalysisParams(ensemble=False, top_n=5))


@pytest.fixture(scope="module")
def raster():
    result, _ = v.run_raster(8.0, 13)
    return result


def test_the_two_dems_agree_to_under_a_metre(contour, raster):
    agreement = v.dem_agreement(contour, raster)
    assert agreement["rmse_m"] < 1.0
    assert abs(agreement["bias_m"]) < 0.5
    assert agreement["corr"] > 0.99


def test_the_elevation_ranges_agree_to_two_metres(contour, raster):
    lo, hi = contour.dem.meta.elevation_range
    raw = raster.dem.raw_z[raster.surface.inside_mask(raster.dem)]
    assert abs(raw.min() - lo) <= 2 and abs(raw.max() - hi) <= 2


def test_the_surveys_catchment_is_found_on_the_free_data(contour, raster):
    """Some map-path site traces substantially the same ground as the survey's site 1."""
    iou, rank = v.best_overlap(contour, raster)
    assert iou > 0.4 and rank <= 3


def test_the_recommendation_is_in_the_same_valley(contour, raster):
    assert v.metres_between(contour.sites[0].lonlat, raster.sites[0].lonlat) < 600


def test_the_catchment_areas_are_within_a_factor_of_two(contour, raster):
    ratio = raster.sites[0].area_ha / contour.sites[0].area_ha
    assert 0.5 < ratio < 2.0


def test_z12_is_worse_than_z13(contour, raster):
    """The reason the default zoom is 13: at z12 the site moves to another valley."""
    coarse, _ = v.run_raster(8.0, 12)
    assert v.metres_between(contour.sites[0].lonlat, coarse.sites[0].lonlat) > v.metres_between(
        contour.sites[0].lonlat, raster.sites[0].lonlat
    )
