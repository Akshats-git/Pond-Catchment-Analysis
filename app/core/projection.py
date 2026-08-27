"""Geographic <-> local metric coordinates.

Flow routing is a metric operation: D8 compares slopes, so the grid must be square in
*metres*, not in degrees. At the latitude of the sample sheet a degree of longitude is
about 7% shorter than a degree of latitude, so routing on raw lon/lat would tilt every
slope in the map.

`EquirectangularENU` is deliberately the simplest projection that is good enough here:
one cosine, evaluated once at the dataset centroid.

    x = (lon - lon0) * 111320 * cos(lat0)
    y = (lat - lat0) * 110540

Accuracy, stated honestly. Those two constants are mid-latitude values, so at the 21 N of
the sample sheet the frame carries a small *scale* error against WGS-84: -0.04%
east-west and -0.16% north-south. That is a ~4 m stretch across the 2.6 km sheet and a
-0.2% bias on every reported area -- 0.8 ha on a 395 ha catchment, twenty times smaller
than the +/-4% spread the resolution ensemble reports for that same catchment. The
*shape* distortion, which is the part flow routing could actually notice, is 0.12%: a
uniform scale error cancels when D8 compares one neighbour against another.

The frame is exactly invertible and has no dependencies, and both scale constants live in
`config.py`, so a deployment at a very different latitude can correct them without
touching code. The `Projection` interface exists so Phase 3 can drop in pyproj/UTM for
larger regions without touching anything downstream (PLAN §8); that is also the clean
place to remove the scale bias entirely, should a later phase want it.

The one place the flat-earth assumption has to be paid back is area. Because the scale
factor is the *constant* cos(lat0), a square cell of side `res` in projected space covers

    res^2 * cos(lat_cell) / cos(lat0)

square metres on the ground -- the latitude weighting of PLAN §2 Step 6.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.config import ProjectionConfig, settings

__all__ = ["Projection", "EquirectangularENU", "projection_for"]


class Projection(ABC):
    """Maps lon/lat in degrees to a local metric frame and back.

    Implementations must be exactly invertible: `inverse(forward(p)) == p` to within
    floating-point noise. Phase 4 rounds real-world coordinates onto grid indices and
    back, and a lossy round trip there silently returns the wrong cell (PLAN §11.8).
    """

    @property
    @abstractmethod
    def origin(self) -> tuple[float, float]:
        """(lon0, lat0) in degrees -- the point that maps to (0, 0) metres."""

    @abstractmethod
    def forward(self, lonlat: np.ndarray) -> np.ndarray:
        """(..., 2) degrees -> (..., 2) metres."""

    @abstractmethod
    def inverse(self, xy: np.ndarray) -> np.ndarray:
        """(..., 2) metres -> (..., 2) degrees."""

    @abstractmethod
    def area_factor(self, lat: np.ndarray | float) -> np.ndarray | float:
        """Ground area of a unit projected cell at this latitude.

        Multiply `resolution^2` by this to get true square metres.
        """

    def forward_xy(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convenience for separate lon/lat arrays."""
        out = self.forward(np.stack(np.broadcast_arrays(lon, lat), axis=-1))
        return out[..., 0], out[..., 1]


@dataclass(frozen=True)
class EquirectangularENU(Projection):
    """Equirectangular east-north-up frame about a fixed origin."""

    lon0: float
    lat0: float
    config: ProjectionConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            object.__setattr__(self, "config", settings.projection)
        if not -90.0 <= self.lat0 <= 90.0:
            raise ValueError(f"Origin latitude {self.lat0} is outside [-90, 90].")
        # Guard the poles: cos(lat0) -> 0 makes the longitude scale degenerate and the
        # inverse transform unstable. Village-scale contour sheets never live there.
        if abs(self.lat0) > 85.0:
            raise ValueError(
                f"Origin latitude {self.lat0} is within 5 degrees of a pole; the "
                "equirectangular frame degenerates there. Use a polar projection."
            )
        object.__setattr__(self, "_cos_lat0", math.cos(math.radians(self.lat0)))

    @property
    def origin(self) -> tuple[float, float]:
        return (self.lon0, self.lat0)

    @property
    def metres_per_degree_lon(self) -> float:
        return self.config.metres_per_degree_lon_equator * self._cos_lat0  # type: ignore[attr-defined]

    @property
    def metres_per_degree_lat(self) -> float:
        return self.config.metres_per_degree_lat

    def forward(self, lonlat: np.ndarray) -> np.ndarray:
        lonlat = np.asarray(lonlat, dtype=np.float64)
        out = np.empty_like(lonlat)
        out[..., 0] = (lonlat[..., 0] - self.lon0) * self.metres_per_degree_lon
        out[..., 1] = (lonlat[..., 1] - self.lat0) * self.metres_per_degree_lat
        return out

    def inverse(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        out = np.empty_like(xy)
        out[..., 0] = xy[..., 0] / self.metres_per_degree_lon + self.lon0
        out[..., 1] = xy[..., 1] / self.metres_per_degree_lat + self.lat0
        return out

    def area_factor(self, lat: np.ndarray | float) -> np.ndarray | float:
        """cos(lat) / cos(lat0) -- see the module docstring."""
        return np.cos(np.radians(lat)) / self._cos_lat0  # type: ignore[attr-defined]


def projection_for(
    lonlat: np.ndarray, *, config: ProjectionConfig | None = None
) -> EquirectangularENU:
    """Build a projection centred on a point cloud.

    The origin is the *bounding-box centre*, not the centroid of the vertices: contour
    vertices bunch up where the ground is steep, so a vertex centroid would pull the
    origin towards the hills and put the largest projection error over the flat ground
    where the pond is going to go.
    """
    lonlat = np.asarray(lonlat, dtype=np.float64)
    if lonlat.ndim != 2 or lonlat.shape[1] != 2 or len(lonlat) == 0:
        raise ValueError("Expected a non-empty (N, 2) array of lon/lat pairs.")
    lon0 = float((lonlat[:, 0].min() + lonlat[:, 0].max()) * 0.5)
    lat0 = float((lonlat[:, 1].min() + lonlat[:, 1].max()) * 0.5)
    return EquirectangularENU(lon0=lon0, lat0=lat0, config=config or settings.projection)
