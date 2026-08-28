"""Where the rain comes from.

The runoff calculation needs a *daily* rainfall series, not an annual total -- SCS-CN is
an event model and applying it to a year's rain in one lump overestimates runoff about
six-fold (PLAN §4). Phase 2 has no rainfall feed, so this module states a documented
climatology and labels it as one; Phase 3 replaces the class behind `RainfallProvider`
with Open-Meteo's daily observations and nothing downstream changes.

The default series is deliberately *reproducible rather than random*: a fixed seed, a
gamma shape chosen because monsoon rainfall is right-skewed (many small days, a few very
large ones), and a total rescaled to exactly the documented annual figure. Every runoff
number in the response therefore comes out the same on every run, which is the only way
a reported figure can be checked.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from app.config import HydrologyConfig, settings

__all__ = ["RainfallSeries", "RainfallProvider", "DefaultRainfallProvider"]


@dataclass(frozen=True)
class RainfallSeries:
    """One year of daily rainfall depths, in millimetres."""

    daily_mm: np.ndarray
    """(n_days,) rainfall on each *rain* day. Dry days are omitted rather than stored as
    zeros: SCS-CN returns zero runoff for them, so they change nothing but the length of
    the array."""

    source: str
    """Where the numbers came from -- carried into the response so a reader can tell a
    climatology from an observation."""

    is_measured: bool
    """False for the documented default. The response says so; a number that looks like
    data and is not is worse than no number."""

    description: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def annual_total_mm(self) -> float:
        return float(self.daily_mm.sum())

    @property
    def rain_days(self) -> int:
        return int(self.daily_mm.size)

    @property
    def wettest_day_mm(self) -> float:
        return float(self.daily_mm.max()) if self.daily_mm.size else 0.0


class RainfallProvider(ABC):
    """The seam Phase 3 replaces (PLAN §8).

    Takes a location because real rainfall depends on one; the default implementation
    ignores it, and says so, rather than pretending the sample sheet's climate is
    universal.
    """

    @abstractmethod
    def daily_series(self, lon: float, lat: float) -> RainfallSeries:
        """One year of daily rainfall for the point, in millimetres."""


@dataclass(frozen=True)
class DefaultRainfallProvider(RainfallProvider):
    """A documented climatology for the region of the sample sheet.

    1200 mm across 55 rain days is the published annual normal for Raipur
    (Chhattisgarh), which is where the provided contour map is. The distribution across
    those days is a seeded gamma draw, rescaled so the total is exact -- the shape of the
    monsoon matters to SCS-CN (runoff is quadratic in daily depth, so a few big days
    dominate the yield) but the individual days do not pretend to be real dates.
    """

    config: HydrologyConfig = field(default_factory=lambda: settings.hydrology)
    annual_total_mm: float | None = None
    rain_days: int | None = None
    """Per-request overrides for the two documented defaults (Phase 9 exposes both)."""

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

        return RainfallSeries(
            daily_mm=daily,
            source="documented climatology (Raipur, Chhattisgarh)",
            is_measured=False,
            description=(
                f"{total:.0f} mm across {int(days)} rain days, distributed by a seeded "
                f"gamma(shape={self.config.rainfall_gamma_shape}) draw."
            ),
            warnings=(
                "Rainfall is a documented regional climatology, not an observation for "
                "this location; Phase 3 replaces it with daily Open-Meteo data.",
            ),
        )
