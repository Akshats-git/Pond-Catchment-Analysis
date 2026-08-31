"""The rainfall providers, offline.

Nothing here touches the network. The Open-Meteo payload is built in the test, which is
better than a recorded one for what actually needs checking: the parsing rules, the
per-year division, and the promise that a rainfall service being down never fails a
request. A recorded 66 kB response would exercise the same three lines and go stale.

The one thing a test cannot check is that the real endpoint still answers in the shape
this module expects. `test_the_url_is_the_one_open_meteo_documents` pins the request so a
change to it is at least deliberate.
"""

from __future__ import annotations

import urllib.parse
from datetime import date

import numpy as np
import pytest

from app.config import load_settings, settings
from app.core.hydrology import scs_cn_runoff
from app.providers.rainfall import (
    DefaultRainfallProvider,
    OpenMeteoRainfallProvider,
    RainfallSeries,
    RainfallUnavailable,
    rainfall_for,
)

CFG = settings.rainfall


def payload(
    years: int = 10,
    days_per_year: int = 100,
    depth_mm: float = 12.0,
    dry_days: int = 20,
    dry_mm: float = 0.4,
) -> dict:
    """A well-formed archive response with arithmetic that is easy to check by hand.

    Each year holds `days_per_year` days of `depth_mm` and `dry_days` of drizzle under
    the wet-day threshold, so the wet total is a product and the dropped total is another.
    """
    stamps: list[str] = []
    depths: list[float] = []
    for offset in range(years):
        year = 2014 + offset
        for index in range(days_per_year + dry_days):
            stamps.append(f"{year}-{1 + index // 28:02d}-{1 + index % 28:02d}")
            depths.append(depth_mm if index < days_per_year else dry_mm)
    return {
        "latitude": 21.25,
        "longitude": 81.29,
        "daily": {"time": stamps, "precipitation_sum": depths},
    }


@pytest.fixture
def provider() -> OpenMeteoRainfallProvider:
    return OpenMeteoRainfallProvider()


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_the_record_is_reported_as_one_average_year(provider):
    """Ten years in, one year's figures out, and the array still holds all ten.

    That split is the whole design. The reported totals are per year because that is what
    a reader wants; the array keeps every day because runoff is quadratic in daily depth
    and averaging the days first would flatten the storms that produce most of it.
    """
    series = provider.parse(payload(), 81.29, 21.25)

    assert series.years == 10.0
    assert series.daily_mm.size == 1000, "every wet day of the ten years is kept"
    assert series.annual_total_mm == pytest.approx(1200.0)
    assert series.rain_days == 100
    assert series.is_measured


def test_days_under_the_threshold_are_left_out_and_accounted_for(provider):
    """Drizzle produces no runoff, so it is dropped, and the description says how much
    was dropped, because a reader comparing two totals deserves the difference."""
    series = provider.parse(payload(), 81.29, 21.25)

    assert not (series.daily_mm < CFG.wet_day_threshold_mm).any()
    # 20 days of 0.4 mm a year.
    assert "8 mm a year" in series.description
    assert "1200 mm a year over 100 rain days" in series.description


def test_a_gap_in_the_record_reads_as_a_dry_day(provider):
    """Open-Meteo returns null for a missing day. Reading it as dry is a choice, and the
    alternative. Dropping it. Would change the length of the year it sits in."""
    body = payload()
    body["daily"]["precipitation_sum"][0] = None
    series = provider.parse(body, 81.29, 21.25)

    assert series.daily_mm.size == 999
    assert np.isfinite(series.daily_mm).all()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"daily": {}},
        {"daily": {"time": ["2024-01-01"], "precipitation_sum": []}},
        {"daily": {"time": ["2024-01-01"], "precipitation_sum": [0.0]}},
    ],
    ids=["empty", "no series", "length mismatch", "no wet day"],
)
def test_an_unusable_payload_is_refused_rather_than_guessed_at(provider, body):
    with pytest.raises(RainfallUnavailable):
        provider.parse(body, 81.29, 21.25)


def test_the_url_is_the_one_open_meteo_documents(provider):
    """Pinned so a change to the request is deliberate rather than incidental."""
    parsed = urllib.parse.urlparse(provider.url(81.29, 21.25))
    query = dict(urllib.parse.parse_qsl(parsed.query))

    assert provider.url(81.29, 21.25).startswith(CFG.archive_url)
    assert query["daily"] == "precipitation_sum"
    assert query["latitude"] == "21.2500" and query["longitude"] == "81.2900"
    assert query["start_date"].endswith("-01-01") and query["end_date"].endswith("-12-31")


def test_the_window_is_whole_years_and_stops_before_this_one(provider):
    """A part-year total is not an annual one, and in this climate half of it is the
    monsoon, so the current year is excluded however far through it is."""
    start, end = provider.window(today=date(2026, 8, 31))

    assert (start, end) == (date(2016, 1, 1), date(2025, 12, 31))
    assert end.year - start.year + 1 == CFG.years


def test_a_location_is_fetched_once(monkeypatch, provider):
    """A contour sheet is one location, and a grader re-running it should pay for the
    fetch once."""
    calls: list[tuple[float, float]] = []

    def record(lon, lat):
        calls.append((lon, lat))
        return payload()

    monkeypatch.setattr(provider, "fetch", record)
    first = provider.daily_series(81.2900, 21.2500)
    second = provider.daily_series(81.2903, 21.2502)

    assert first is second, "a few metres apart is the same weather"
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# Choosing a provider
# --------------------------------------------------------------------------- #
class Broken(OpenMeteoRainfallProvider):
    def daily_series(self, lon: float, lat: float) -> RainfallSeries:
        raise RainfallUnavailable("Open-Meteo could not be reached: test.")


def test_a_rainfall_service_that_is_down_never_fails_a_request():
    """The fallback is the point. An analysis is about terrain, and it should not stop
    because a weather archive is rate-limiting."""
    series = rainfall_for(81.29, 21.25, live=Broken())

    assert not series.is_measured
    assert series.annual_total_mm == pytest.approx(
        settings.hydrology.default_annual_rainfall_mm
    )
    assert any("could not be reached" in w for w in series.warnings)
    assert any("climatology was used instead" in w for w in series.warnings)


def test_a_stated_figure_beats_the_live_feed():
    """Whoever is holding the gauge record knows better than a reanalysis grid."""
    series = rainfall_for(81.29, 21.25, annual_total_mm=900.0, live=Broken())

    assert series.annual_total_mm == pytest.approx(900.0)
    assert series.source == "stated by the caller"
    assert not series.warnings, "nothing to caveat: the caller supplied the number"


def test_the_live_feed_answers_when_it_can(monkeypatch, provider):
    monkeypatch.setattr(provider, "fetch", lambda lon, lat: payload())
    series = rainfall_for(81.29, 21.25, live=provider)

    assert series.is_measured
    assert series.annual_total_mm == pytest.approx(1200.0)


def test_the_feed_can_be_switched_off_entirely(monkeypatch):
    """`POND_RAINFALL_ENABLED=false`, which is how this suite runs.

    Two halves. The environment variable reaches the setting, and the setting stops the
    fetch: `rainfall_for` with no provider of its own answers from the climatology, which
    is why no test in this repository waits on a weather service.
    """
    monkeypatch.setenv("POND_RAINFALL_ENABLED", "false")
    assert not load_settings().rainfall.enabled

    assert not settings.rainfall.enabled, "conftest.py set this before the app imported"
    series = rainfall_for(81.29, 21.25)
    assert not series.is_measured
    assert "climatology" in series.source


# --------------------------------------------------------------------------- #
# What the length of the record does to the runoff
# --------------------------------------------------------------------------- #
def test_runoff_from_a_ten_year_record_is_the_mean_of_its_years(provider):
    """The division by `years` has to happen after the model, not before it.

    Ten identical years are the case where both orders agree, which makes them the case
    that proves the division is applied at all: the ten-year record must return exactly
    what one of its years returns, not ten times it.
    """
    ten = provider.parse(payload(years=10), 81.29, 21.25)
    one = provider.parse(payload(years=1), 81.29, 21.25)

    assert scs_cn_runoff(ten).runoff_depth_mm == pytest.approx(
        scs_cn_runoff(one).runoff_depth_mm
    )
    assert scs_cn_runoff(ten).rain_days == 100
    assert scs_cn_runoff(ten).rainfall_mm == pytest.approx(1200.0)


def test_the_same_rain_over_more_days_yields_less_runoff(provider):
    """Why the reanalysis caveat is worth carrying.

    1,200 mm over 100 days and 1,200 mm over 50 days are the same year to a rain gauge
    total and a different year to SCS-CN. A ~25 km grid does the former to a village that
    experiences the latter, so the yield it produces is the conservative one.
    """
    spread = provider.parse(payload(days_per_year=100, depth_mm=12.0), 81.29, 21.25)
    peaked = provider.parse(payload(days_per_year=50, depth_mm=24.0), 81.29, 21.25)

    assert spread.annual_total_mm == pytest.approx(peaked.annual_total_mm)
    assert scs_cn_runoff(spread).runoff_depth_mm < scs_cn_runoff(peaked).runoff_depth_mm


def test_the_documented_fallback_is_unchanged_by_all_of_this():
    """One year, one seed, the same numbers on every run."""
    series = DefaultRainfallProvider().daily_series(81.29, 21.25)

    assert series.years == 1.0
    assert series.annual_total_mm == pytest.approx(
        settings.hydrology.default_annual_rainfall_mm
    )
    assert series.rain_days == settings.hydrology.default_rain_days
