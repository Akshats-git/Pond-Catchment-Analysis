"""Phase 2. Local metric projection.

The projection is three lines of arithmetic, which is exactly why it is worth testing:
everything downstream assumes the grid is square in metres and exactly invertible, and
neither property announces itself when it breaks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.core.projection import EquirectangularENU, projection_for

RAIPUR = (81.297, 21.252)


@pytest.fixture
def proj():
    return EquirectangularENU(lon0=RAIPUR[0], lat0=RAIPUR[1])


def test_origin_maps_to_zero(proj):
    np.testing.assert_allclose(proj.forward(np.array(RAIPUR)), [0.0, 0.0], atol=1e-9)


def test_round_trip_is_exact(proj):
    """Phase 4 converts real-world coordinates to grid indices and back. A lossy round
    trip there returns a neighbouring cell, and somebody else's catchment."""
    rng = np.random.default_rng(0)
    lonlat = np.column_stack([
        rng.uniform(RAIPUR[0] - 0.02, RAIPUR[0] + 0.02, 5000),
        rng.uniform(RAIPUR[1] - 0.02, RAIPUR[1] + 0.02, 5000),
    ])
    np.testing.assert_allclose(proj.inverse(proj.forward(lonlat)), lonlat, atol=1e-12)


def test_axes_have_the_right_scale(proj):
    """A degree of longitude at 21 N is about 7% shorter than a degree of latitude.
    Routing on raw degrees would tilt every slope in the map by that much."""
    east = proj.forward(np.array([RAIPUR[0] + 1.0, RAIPUR[1]]))[0]
    north = proj.forward(np.array([RAIPUR[0], RAIPUR[1] + 1.0]))[1]
    assert east == pytest.approx(111_320.0 * math.cos(math.radians(RAIPUR[1])), rel=1e-9)
    assert north == pytest.approx(110_540.0, rel=1e-9)
    assert east / north == pytest.approx(0.938, abs=0.002)


def test_a_known_distance(proj):
    """0.01 deg of latitude is 1105.4 m; the same step east is shorter by cos(lat)."""
    north = proj.forward(np.array([RAIPUR[0], RAIPUR[1] + 0.01]))
    assert north[1] == pytest.approx(1105.4, rel=1e-6)
    east = proj.forward(np.array([RAIPUR[0] + 0.01, RAIPUR[1]]))
    assert east[0] == pytest.approx(1037.5, rel=1e-3)


def test_forward_preserves_shape(proj):
    assert proj.forward(np.zeros((7, 3, 2))).shape == (7, 3, 2)


def test_area_factor_is_one_at_the_origin(proj):
    assert proj.area_factor(RAIPUR[1]) == pytest.approx(1.0)


def test_area_factor_shrinks_polewards(proj):
    """PLAN §2 Step 6: a projected cell covers res^2 * cos(lat)/cos(lat0) on the ground."""
    assert proj.area_factor(RAIPUR[1] + 1.0) < 1.0 < proj.area_factor(RAIPUR[1] - 1.0)


def test_area_factor_correction_is_small_over_one_sheet(proj):
    """Worth applying, not worth worrying about: 0.01% across 2.6 km of latitude."""
    factors = proj.area_factor(np.array([RAIPUR[1] - 0.012, RAIPUR[1] + 0.012]))
    assert np.abs(factors - 1.0).max() < 1e-4


def test_projection_for_centres_on_the_bounding_box():
    """The origin is the bbox centre, not the vertex centroid: contour vertices bunch up
    on steep ground, which would drag the origin into the hills."""
    lonlat = np.array([[80.0, 20.0], [80.0, 20.0], [80.0, 20.0], [82.0, 22.0]])
    proj = projection_for(lonlat)
    assert proj.origin == pytest.approx((81.0, 21.0))


def test_projection_for_rejects_empty_input():
    with pytest.raises(ValueError):
        projection_for(np.empty((0, 2)))


def test_polar_origin_is_rejected():
    """cos(lat0) -> 0 makes the longitude scale degenerate and the inverse unstable."""
    with pytest.raises(ValueError, match="pole"):
        EquirectangularENU(lon0=0.0, lat0=89.0)


# --------------------------------------------------------------------------- #
# How accurate is the flat-earth frame, really?
# --------------------------------------------------------------------------- #
WGS84_A = 6_378_137.0
WGS84_E2 = 0.00669437999014


def _wgs84_degree_lengths(lat_deg: float) -> tuple[float, float]:
    """(metres per degree of longitude, per degree of latitude) on the WGS-84 ellipsoid.

    The honest yardstick. A haversine with a mean spherical radius is *not*: at 21 N it
    overstates the meridian degree by 0.4%, which is larger than the error being measured.
    """
    lat = math.radians(lat_deg)
    sin2 = math.sin(lat) ** 2
    meridional = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * sin2) ** 1.5
    normal = WGS84_A / math.sqrt(1 - WGS84_E2 * sin2)
    return (normal * math.cos(lat) * math.pi / 180, meridional * math.pi / 180)


def test_scale_error_against_wgs84_is_small(proj):
    """The frame's fixed constants (111320, 110540) are mid-latitude values, so at 21 N
    the scale is slightly off: -0.04% east-west and -0.16% north-south.

    That is a ~4 m stretch across the 2.6 km sheet and a -0.2% bias on every reported
    area. 0.8 ha on a 395 ha catchment. Twenty times smaller than the +/-4% spread the
    resolution ensemble reports for the same catchment, so it is a documented bias rather
    than a correction worth making.
    """
    true_lon, true_lat = _wgs84_degree_lengths(proj.lat0)
    assert abs(proj.metres_per_degree_lon / true_lon - 1) < 0.0025
    assert abs(proj.metres_per_degree_lat / true_lat - 1) < 0.0025


def test_shape_distortion_is_smaller_than_scale_error(proj):
    """What flow routing actually cares about. D8 compares a step north against a step
    east, so a *uniform* scale error cancels and only the anisotropy can bias a slope."""
    true_lon, true_lat = _wgs84_degree_lengths(proj.lat0)
    anisotropy = (proj.metres_per_degree_lon / true_lon) / (
        proj.metres_per_degree_lat / true_lat
    )
    assert abs(anisotropy - 1) < 0.0015


def test_the_frame_is_locally_flat(proj):
    """Curvature, as distinct from scale: the residual after the best uniform scale is
    divided out. This is what would make the projection unusable over a large sheet, and
    over one contour sheet it is well under a millimetre."""
    true_lon, true_lat = _wgs84_degree_lengths(proj.lat0)
    for distance in (100.0, 800.0, 1600.0):
        target = proj.inverse(np.array([0.0, distance]))
        modelled = (target[1] - proj.lat0) * true_lat
        assert abs(modelled - distance * (true_lat / proj.metres_per_degree_lat)) < 1e-3
