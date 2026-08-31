"""What a client may ask for, and what counts as a valid ask.

Every knob the route exposes is validated here, once, before any terrain work starts.
That ordering is the point: delineating three grids on the sample sheet costs about
twelve seconds, and a curve number of 200 is knowable as wrong in microseconds. A
parameter rejected at the door returns a `422` immediately instead of after the analysis
that was never going to be usable.

The bounds are read from `app.config` at validation time rather than baked into the field
definitions. A `POND_*` environment override, or a test that narrows a range, therefore moves
the API's contract with it. Nothing in this module invents a limit of its own; the two
exceptions are the geographic bounds of lon/lat, which are facts about the planet rather
than tunables.

Defaults come from the same place and are resolved *eagerly*: by the time the pipeline
sees an `AnalysisParams`, every field holds the value the analysis will actually use, so
the response can echo the parameters back without guessing which of them were implied.

Rainfall is the exception, and it has to be. `rainfall_mm` and `rain_days` stay null when
the caller leaves them out, because what fills them is ten years of records for a location
that is not known until the site is chosen. What the analysis actually used comes back
under `recommended_site.runoff`, alongside the source it came from.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import settings

__all__ = ["AnalysisParams"]


class AnalysisParams(BaseModel):
    """The optional half of `POST /analyzeContour`, which is everything except the file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grid_resolution: float | None = Field(
        default=None,
        description=(
            "DEM cell size in metres. Omit to derive it from the mean contour spacing "
            "(spacing / 4), which is what the validated methodology uses."
        ),
    )
    top_n: int = Field(
        default_factory=lambda: settings.siting.default_top_n,
        description="How many independent basins to return, best first.",
    )
    lat: float | None = Field(
        default=None,
        description="Explicit pour point latitude. Overrides automatic siting; "
        "must be given together with `lon`.",
    )
    lon: float | None = Field(
        default=None, description="Explicit pour point longitude. See `lat`."
    )
    curve_number: float = Field(
        default_factory=lambda: settings.hydrology.default_curve_number,
        description="SCS curve number for the catchment's soil and land cover.",
    )
    rainfall_mm: float | None = Field(
        default=None,
        description=(
            "Annual rainfall total in millimetres. Leave it out and the service fetches "
            "ten years of daily records for the site from Open-Meteo, which is free and "
            "needs no key. Set it and your figure is used instead."
        ),
    )
    rain_days: int | None = Field(
        default=None,
        description=(
            "Number of days that rain falls on. SCS-CN is an event model: the total is "
            "spread over these days and runoff is summed per day, never applied to the "
            "annual figure as one storm. Left out, it comes from the same records as "
            "`rainfall_mm`."
        ),
    )
    target_depth_m: float = Field(
        default_factory=lambda: settings.hydrology.default_target_depth_m,
        description="Depth of the pond to be excavated or bunded at the site.",
    )
    ensemble: bool = Field(
        default_factory=lambda: settings.api.default_ensemble,
        description=(
            "Cross-check every site on three independent grids to put an error bar on "
            "the area. Roughly triples the analysis time; the error bar is the "
            "difference between a number and a number you can act on."
        ),
    )

    # ------------------------------------------------------------------ #
    # Bounds. Each reads its limit from config so there is one source of truth.
    # ------------------------------------------------------------------ #
    @field_validator("grid_resolution")
    @classmethod
    def _check_resolution(cls, value: float | None) -> float | None:
        # Only positivity here. Whether 1.5 m is *allowed* is the DEM builder's rule and
        # depends on the sheet, so it stays there rather than being restated.
        if value is not None and value <= 0:
            raise ValueError("grid_resolution must be greater than zero metres.")
        return value

    @field_validator("top_n")
    @classmethod
    def _check_top_n(cls, value: int) -> int:
        limit = settings.siting.max_top_n
        if not 1 <= value <= limit:
            raise ValueError(f"top_n must be between 1 and {limit}.")
        return value

    @field_validator("curve_number")
    @classmethod
    def _check_curve_number(cls, value: float) -> float:
        low, high = settings.hydrology.curve_number_range
        if not low <= value <= high:
            raise ValueError(
                f"curve_number must be between {low:g} and {high:g}; outside that range "
                "the SCS-CN relation is not defined."
            )
        return value

    @field_validator("rainfall_mm")
    @classmethod
    def _check_rainfall(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("rainfall_mm must be greater than zero.")
        return value

    @field_validator("rain_days")
    @classmethod
    def _check_rain_days(cls, value: int | None) -> int | None:
        if value is not None and not 1 <= value <= 366:
            raise ValueError("rain_days must be between 1 and 366.")
        return value

    @field_validator("target_depth_m")
    @classmethod
    def _check_depth(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("target_depth_m must be greater than zero metres.")
        return value

    @field_validator("lat")
    @classmethod
    def _check_lat(cls, value: float | None) -> float | None:
        if value is not None and not -90.0 <= value <= 90.0:
            raise ValueError("lat must be between -90 and 90 degrees.")
        return value

    @field_validator("lon")
    @classmethod
    def _check_lon(cls, value: float | None) -> float | None:
        if value is not None and not -180.0 <= value <= 180.0:
            raise ValueError("lon must be between -180 and 180 degrees.")
        return value

    @model_validator(mode="after")
    def _check_pour_point(self) -> "AnalysisParams":
        if (self.lat is None) != (self.lon is None):
            missing = "lon" if self.lon is None else "lat"
            raise ValueError(
                f"An explicit pour point needs both lat and lon; {missing} is missing."
            )
        return self

    # ------------------------------------------------------------------ #
    @property
    def pour_point(self) -> tuple[float, float] | None:
        """(lon, lat) if the client named a site, else `None`. This is the flag the pipeline
        branches on. Returned lon-first, the order every core function takes."""
        if self.lat is None or self.lon is None:
            return None
        return (self.lon, self.lat)
