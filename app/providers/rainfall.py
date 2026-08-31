"""Where the rain comes from.

The runoff calculation needs a *daily* rainfall series, not an annual total. SCS-CN is an
event model, and applying it to a year's rain in one lump overestimates runoff about six
times over (PLAN §4).

Two providers sit behind one interface. `OpenMeteoRainfallProvider` fetches ten years of
daily records for the site from Open-Meteo, which is free, needs no key and covers any
point on land. `DefaultRainfallProvider` states a documented climatology and labels it as
one; it is what answers when the service cannot be reached, and what answers when the
caller names a rainfall figure of their own.

**Why the observed series is ten years long and not one.** Rainfall varies more between
years than a pond design can absorb. On the sample location the annual total runs from
1,057 mm to 1,858 mm, so a single year is off by a third either way. The provider hands
back every wet day of the ten and says so in `years`; the runoff model divides by that,
which makes the reported figure the mean of the ten annual runoffs rather than a number
built from an invented average year. Averaging the years and then computing runoff would
be the wrong order, because runoff is quadratic in daily depth.

**Why the default series is reproducible rather than random.** A fixed seed, a gamma shape
chosen because monsoon rainfall is right-skewed, and a total rescaled to exactly the
documented annual figure. Every runoff number the fallback produces comes out the same on
every run, which is the only way a reported figure can be checked.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from app.config import HydrologyConfig, RainfallConfig, settings

__all__ = [
    "RainfallSeries",
    "RainfallProvider",
    "DefaultRainfallProvider",
    "OpenMeteoRainfallProvider",
    "RainfallUnavailable",
    "rainfall_for",
]


class RainfallUnavailable(Exception):
    """The rainfall service could not be reached, or answered with nothing usable.

    Never fatal. The caller falls back to the documented climatology and carries this
    message into the response, so a reader can tell a fallback from an observation.
    """


@dataclass(frozen=True)
class RainfallSeries:
    """Daily rainfall depths in millimetres, covering `years` whole years."""

    daily_mm: np.ndarray
    """(n_days,) rainfall on each *rain* day across the whole record. Dry days are left
    out rather than stored as zeros: SCS-CN returns zero runoff for them, so they change
    nothing but the length of the array."""

    source: str
    """Where the numbers came from. Carried into the response so a reader can tell a
    climatology from an observation."""

    is_measured: bool
    """False for the documented default. The response says so, because a number that
    looks like data and is not is worse than no number at all."""

    description: str = ""
    warnings: tuple[str, ...] = ()

    years: float = 1.0
    """How many years the record spans.

    Every annual figure below is a total divided by this. The alternative is to build one
    average year out of ten and run the model on that, which would smooth away the big
    days, and the big days are most of the runoff.
    """

    @property
    def annual_total_mm(self) -> float:
        return float(self.daily_mm.sum()) / self.years

    @property
    def rain_days(self) -> int:
        return int(round(self.daily_mm.size / self.years))

    @property
    def wettest_day_mm(self) -> float:
        return float(self.daily_mm.max()) if self.daily_mm.size else 0.0


class RainfallProvider(ABC):
    """The seam a real feed plugs into (PLAN §8).

    Takes a location because real rainfall depends on one. The fallback implementation
    ignores it and says so, rather than pretending the sample sheet's climate is
    universal.
    """

    @abstractmethod
    def daily_series(self, lon: float, lat: float) -> RainfallSeries:
        """Daily rainfall for the point, in millimetres."""


# --------------------------------------------------------------------------- #
# The documented fallback
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DefaultRainfallProvider(RainfallProvider):
    """A documented climatology for the region of the sample sheet.

    1,200 mm across 55 rain days is the published annual normal for Raipur
    (Chhattisgarh), where the provided contour map is. The spread across those days is a
    seeded gamma draw rescaled so the total is exact. The shape of the monsoon matters to
    SCS-CN, since runoff is quadratic in daily depth and a few big days carry most of it,
    but the individual days here do not pretend to be real dates.
    """

    config: HydrologyConfig = field(default_factory=lambda: settings.hydrology)
    annual_total_mm: float | None = None
    rain_days: int | None = None
    """Per-request overrides for the two documented defaults."""

    reason: str = ""
    """Why this provider is answering instead of the live one. Empty when the caller
    asked for it directly."""

    def daily_series(
        self, lon: float | None = None, lat: float | None = None
    ) -> RainfallSeries:
        total = (
            self.annual_total_mm
            if self.annual_total_mm is not None
            else self.config.default_annual_rainfall_mm
        )
        days = (
            self.rain_days
            if self.rain_days is not None
            else self.config.default_rain_days
        )
        if total <= 0 or days <= 0:
            raise ValueError("Annual rainfall and rain days must both be positive.")

        # Seeded, so the same request always returns the same series.
        rng = np.random.default_rng(self.config.rainfall_seed)
        draw = rng.gamma(self.config.rainfall_gamma_shape, 1.0, int(days))
        daily = total * draw / draw.sum()

        stated = self.annual_total_mm is not None or self.rain_days is not None
        warnings = []
        if self.reason:
            warnings.append(self.reason)
        if not stated:
            warnings.append(
                "Rainfall is a documented regional climatology, not an observation for "
                "this location."
            )

        return RainfallSeries(
            daily_mm=daily,
            source=(
                "stated by the caller"
                if stated
                else "documented climatology (Raipur, Chhattisgarh)"
            ),
            is_measured=False,
            description=(
                f"{total:.0f} mm across {int(days)} rain days, spread by a seeded "
                f"gamma(shape={self.config.rainfall_gamma_shape}) draw."
            ),
            warnings=tuple(warnings),
        )


# --------------------------------------------------------------------------- #
# The live feed
# --------------------------------------------------------------------------- #
class OpenMeteoRainfallProvider(RainfallProvider):
    """Daily rainfall for any point on land, from Open-Meteo's ERA5 archive.

    Free, no key, no registration. The request asks for `precipitation_sum` over the last
    ten complete calendar years and keeps the days that cleared the wet-day threshold.

    One caveat travels with every series this returns, and it is not a small one. ERA5 is
    a reanalysis on a roughly 25 km grid, so it spreads a village's rain over more days
    than a rain gauge in that village would record. Runoff is quadratic in daily depth, so
    a flatter series yields less runoff: the same place reads 16% of rainfall as runoff on
    the seeded climatology and 8% on ten years of ERA5. The observed figure is the more
    defensible starting point and the more conservative one, but a design being costed
    should be checked against the nearest gauge record.
    """

    def __init__(self, config: RainfallConfig | None = None) -> None:
        self.config = config or settings.rainfall
        self._cache: OrderedDict[tuple[float, float], RainfallSeries] = OrderedDict()

    # ------------------------------------------------------------------ #
    def window(self, today: date | None = None) -> tuple[date, date]:
        """The last N complete calendar years. The current year is left out because a
        part-year total is not an annual one, and the monsoon is half of it."""
        year = (today or date.today()).year - 1
        return date(year - self.config.years + 1, 1, 1), date(year, 12, 31)

    def url(self, lon: float, lat: float) -> str:
        start, end = self.window()
        query = urllib.parse.urlencode(
            {
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": "precipitation_sum",
                "timezone": "auto",
            }
        )
        return f"{self.config.archive_url}?{query}"

    def fetch(self, lon: float, lat: float) -> dict:
        """One GET. Any failure at all becomes `RainfallUnavailable`.

        Deliberately broad. A DNS failure, a 429, a proxy returning HTML and a body that
        is not JSON are four different problems to whoever runs the service and exactly
        one to the caller: there is no observed rainfall for this request, so the
        documented climatology answers instead.
        """
        request = urllib.request.Request(
            self.url(lon, lat),
            headers={"Accept": "application/json", "User-Agent": "pond-catchment/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RainfallUnavailable(
                f"Open-Meteo returned HTTP {exc.code}."
            ) from exc
        except Exception as exc:  # noqa: BLE001, deliberate: see the docstring
            raise RainfallUnavailable(f"Open-Meteo could not be reached: {exc}") from exc

    # ------------------------------------------------------------------ #
    def parse(self, payload: dict, lon: float, lat: float) -> RainfallSeries:
        """Payload to series. Raises `RainfallUnavailable` on anything unusable."""
        daily = (payload or {}).get("daily") or {}
        stamps = daily.get("time") or []
        depths = daily.get("precipitation_sum") or []
        if not stamps or len(stamps) != len(depths):
            raise RainfallUnavailable("Open-Meteo returned no daily rainfall.")

        # Gaps in the record come back as null and are read as dry days. They are rare and
        # scattered; treating them as missing would mean guessing at the year length.
        values = np.array([0.0 if d is None else float(d) for d in depths])
        years = len({stamp[:4] for stamp in stamps})
        if years == 0 or not np.isfinite(values).all():
            raise RainfallUnavailable("Open-Meteo returned an unusable record.")

        wet = values[values >= self.config.wet_day_threshold_mm]
        if wet.size == 0:
            raise RainfallUnavailable(
                f"Open-Meteo records no day over "
                f"{self.config.wet_day_threshold_mm:g} mm at this location."
            )

        # Every figure reported is the wet-day total, because that is the series the
        # runoff model is given. Saying so, and saying what the dropped drizzle came to,
        # is cheaper than leaving a reader to wonder why two totals differ by 1%.
        threshold = self.config.wet_day_threshold_mm
        dropped = (float(values.sum()) - float(wet.sum())) / years
        return RainfallSeries(
            daily_mm=wet,
            years=float(years),
            source=f"Open-Meteo ERA5 daily records, {years} years",
            is_measured=True,
            description=(
                f"{wet.sum() / years:.0f} mm a year over {wet.size / years:.0f} rain "
                f"days, averaged across {years} years of daily records at "
                f"{payload.get('latitude', lat):.3f} N, "
                f"{payload.get('longitude', lon):.3f} E. Days under {threshold:g} mm are "
                f"left out; they come to {dropped:.0f} mm a year and produce no runoff."
            ),
            warnings=(
                "Rainfall is ERA5 reanalysis on a ~25 km grid, which spreads rain over "
                "more days than a village rain gauge would record. Runoff grows faster "
                "than rainfall, so the yield below is on the conservative side. Check it "
                "against the nearest gauge before costing a design.",
            ),
        )

    def daily_series(self, lon: float, lat: float) -> RainfallSeries:
        """Ten years of daily rainfall for the point, cached by rounded location."""
        precision = self.config.coordinate_precision
        key = (round(lon, precision), round(lat, precision))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        series = self.parse(self.fetch(lon, lat), lon, lat)
        self._cache[key] = series
        while len(self._cache) > self.config.cache_size:
            self._cache.popitem(last=False)
        return series


_LIVE = OpenMeteoRainfallProvider()
"""One process-wide instance, so its cache outlives a single request."""


def rainfall_for(
    lon: float,
    lat: float,
    *,
    annual_total_mm: float | None = None,
    rain_days: int | None = None,
    live: RainfallProvider | None = None,
    config: HydrologyConfig | None = None,
) -> RainfallSeries:
    """The rainfall one analysis should use, and the rule for choosing it.

    A figure the caller stated wins, because they know their gauge and this service does
    not. Otherwise the live feed answers. If the live feed cannot be reached the
    documented climatology answers and says why, so a response is never blocked by the
    weather service being down.
    """
    cfg = config or settings.hydrology
    if annual_total_mm is not None or rain_days is not None:
        return DefaultRainfallProvider(
            config=cfg, annual_total_mm=annual_total_mm, rain_days=rain_days
        ).daily_series(lon, lat)

    if live is None and not settings.rainfall.enabled:
        return DefaultRainfallProvider(config=cfg).daily_series(lon, lat)

    try:
        return (live or _LIVE).daily_series(lon, lat)
    except RainfallUnavailable as exc:
        return DefaultRainfallProvider(
            config=cfg,
            reason=(
                f"{exc} The documented regional climatology was used instead: "
                f"{cfg.default_annual_rainfall_mm:.0f} mm over "
                f"{cfg.default_rain_days} rain days."
            ),
        ).daily_series(lon, lat)
