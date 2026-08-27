"""A valley whose catchment area can be worked out on paper.

Every other check in this repository compares the pipeline against itself or against the
sample sheet. This one compares it against arithmetic.

The surface is

    z = 0.05 * |x| + 0.01 * y        over x in [-500, 500], y in [0, 1000]

-- a V-shaped valley draining north to south, with walls twenty times steeper than the
floor. Two facts follow, and together they give an exact answer.

**Every cell flows straight at the channel.** From any cell the slope towards the channel
is 0.05 and the slope down-valley is 0.01. The diagonal between them drops
(0.05 + 0.01) / sqrt(2) = 0.042 per metre, which is *less* than 0.05, so D8 always picks
the pure cross-valley step. Water reaches x = 0 at the same y it started from, then turns
and runs down the channel.

**So the catchment of the channel point at y = Y is everything above it:**

    A(Y) = 1000 * (1000 - Y)   square metres

The valley is written out as a contour map and read back through the identical pipeline --
parser, projection, interpolation, smoothing, fill, D8, accumulation, delineation -- so
what is being tested is the whole chain, not a component in isolation.

The contours are exact straight lines: level `c` is `y = 100c - 5|x|`, a tent clipped to
the domain. The endpoints are where the tent leaves the rectangle, so the vertices reach
all four corners and the interpolated surface covers the whole domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.projection import EquirectangularENU

__all__ = ["SyntheticValley", "VALLEY", "clip_contours"]


@dataclass(frozen=True)
class SyntheticValley:
    """The analytic valley, and everything derived from it."""

    x_half: float = 500.0
    """Half-width of the domain in metres; x runs from -x_half to +x_half."""

    y_max: float = 1000.0
    slope_x: float = 0.05
    slope_y: float = 0.01

    interval_m: float = 1.0
    """Contour interval. With |grad z| = 0.051 this puts the contour lines 19.6 m apart,
    so the pipeline derives a 4.9 m grid and a 2.45 m smoothing sigma -- the 5 m / 2.5 m
    regime PLAN §3 Test A reports."""

    vertex_spacing_m: float = 5.0
    """How finely each contour line is sampled. The lines are straight, so this only has
    to be fine enough that the triangulation does not cut corners."""

    origin: tuple[float, float] = (81.3, 21.25)
    """Where to place the valley on the globe. Arbitrary -- and deliberately not where the
    sample sheet is, so a result that secretly depends on the sample's coordinates comes
    out visibly wrong rather than plausibly right."""

    # ---------------- the surface ---------------- #
    def elevation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.slope_x * np.abs(x) + self.slope_y * y

    @property
    def z_max(self) -> float:
        return float(self.elevation(np.array(self.x_half), np.array(self.y_max)))

    @property
    def area_m2(self) -> float:
        return 2 * self.x_half * self.y_max

    def analytic_catchment_area(self, y: float) -> float:
        """A(Y) = full width x the distance from Y to the top of the valley."""
        return 2 * self.x_half * (self.y_max - y)

    # ---------------- contours ---------------- #
    def contour_lines(self) -> list[tuple[float, np.ndarray]]:
        """(level, (n, 2) array of x/y vertices) for every contour, both branches.

        Level `c` is the tent `y = 100c - 5|x|`. On the right branch it enters the domain
        where the tent crosses `y = y_max` and leaves where it crosses `y = 0`; outside
        that span the contour is not in the rectangle at all.
        """
        lines: list[tuple[float, np.ndarray]] = []
        levels = np.arange(0.0, self.z_max + self.interval_m / 2, self.interval_m)

        for level in levels:
            # y = level/slope_y - (slope_x/slope_y) * |x|
            apex_y = level / self.slope_y
            ratio = self.slope_x / self.slope_y
            x_at_top = (apex_y - self.y_max) / ratio   # where the tent crosses y = y_max
            x_at_bottom = apex_y / ratio               # where it crosses y = 0

            lo = max(0.0, x_at_top)
            hi = min(self.x_half, x_at_bottom)
            if hi < lo:
                continue

            count = max(2, int(np.ceil((hi - lo) / self.vertex_spacing_m)) + 1)
            xs = np.linspace(lo, hi, count)
            ys = apex_y - ratio * xs
            for sign in (1.0, -1.0):
                if sign < 0 and lo == 0.0:
                    # The tent's apex is inside the domain, so the two branches are one
                    # connected line; emitting the left one separately would duplicate
                    # the vertex at x = 0.
                    xs_side, ys_side = -xs[1:], ys[1:]
                else:
                    xs_side, ys_side = sign * xs, ys
                if len(xs_side) >= 2:
                    lines.append((float(level), np.column_stack([xs_side, ys_side])))
        return lines

    # ---------------- output ---------------- #
    def projection(self) -> EquirectangularENU:
        return EquirectangularENU(lon0=self.origin[0], lat0=self.origin[1])

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        return self.projection().inverse(np.asarray(xy, dtype=np.float64))

    def to_kml(self) -> bytes:
        """Write the valley out as a contour KML, exactly as a real one would arrive."""
        placemarks = []
        for level, xy in self.contour_lines():
            lonlat = self.to_lonlat(xy)
            coordinates = " ".join(f"{lon:.9f},{lat:.9f}" for lon, lat in lonlat)
            placemarks.append(
                f"<Placemark><name>{level:g}</name><LineString><coordinates>"
                f"{coordinates}</coordinates></LineString></Placemark>"
            )
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            "<name>synthetic valley</name>\n" + "\n".join(placemarks) +
            "\n</Document></kml>"
        ).encode("utf-8")

    def channel_point(self, y: float) -> tuple[float, float]:
        """The lon/lat of the channel at this distance up the valley."""
        lonlat = self.to_lonlat(np.array([0.0, y]))
        return float(lonlat[0]), float(lonlat[1])


VALLEY = SyntheticValley()


# --------------------------------------------------------------------------- #
def clip_contours(contours, bbox: tuple[float, float, float, float]):
    """Cut a contour set down to a lon/lat window, for the sub-tile test.

    Each line is walked vertex by vertex and every unbroken run of vertices inside the
    window becomes a line of its own, so a contour that leaves and re-enters produces two
    lines -- which is exactly what a real clipped export looks like.
    """
    from app.core.kml_parser import ContourSet

    min_lon, min_lat, max_lon, max_lat = bbox
    chunks: list[np.ndarray] = []
    elevations: list[float] = []
    starts: list[int] = [0]

    for index in range(contours.line_count):
        coords = contours.line_coords(index)
        inside = (
            (coords[:, 0] >= min_lon) & (coords[:, 0] <= max_lon)
            & (coords[:, 1] >= min_lat) & (coords[:, 1] <= max_lat)
        )
        if not inside.any():
            continue
        # Split the vertex sequence into maximal runs of "inside".
        breaks = np.flatnonzero(np.diff(inside.astype(np.int8)) != 0) + 1
        for run in np.split(np.arange(len(coords)), breaks):
            if len(run) >= 2 and inside[run[0]]:
                chunks.append(coords[run])
                elevations.append(float(contours.line_elevations[index]))
                starts.append(starts[-1] + len(run))

    if len(chunks) < 2:
        raise ValueError("The clip window contains too few contour lines.")

    import dataclasses

    line_starts = np.asarray(starts, dtype=np.int64)
    points = np.concatenate(chunks)
    levels = tuple(sorted(set(elevations)))
    # Refresh the metadata that the clip invalidates -- a stale bbox or level list would
    # let the sub-tile test pass on numbers describing the whole sheet.
    metadata = dataclasses.replace(
        contours.metadata,
        levels=levels,
        elevation_range=(levels[0], levels[-1]),
        bbox=(float(points[:, 0].min()), float(points[:, 1].min()),
              float(points[:, 0].max()), float(points[:, 1].max())),
        line_count=len(chunks),
        vertex_count=len(points),
    )
    return ContourSet(
        points=points,
        elevations=np.repeat(elevations, np.diff(line_starts)),
        line_starts=line_starts,
        metadata=metadata,
    )
