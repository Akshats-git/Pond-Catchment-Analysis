"""Turning masks and cells into GeoJSON a map can draw.

Three problems, in order.

**Tracing.** A catchment is a set of cells; a map wants a ring. The trace here walks cell
*edges* rather than interpolating a contour through cell centres: for every cell in the
mask, each side facing out of the mask is a directed segment, oriented so the interior
lies to its left. Chaining those segments end to end gives closed rings whose enclosed
area is exactly the mask's area, so no half-cell is gained or lost before
simplification. The orientation falls out right for free: exteriors counter-clockwise and
holes clockwise, which is what RFC 7946 asks for.

**Simplification.** The traced ring is a staircase with one vertex per cell edge, 40,000
of them on the sample's largest basin, which no browser wants. Douglas-Peucker in
projected metres removes the staircase. The tolerance is kept below one cell on purpose:
under that, a moved vertex cannot cross into a neighbouring cell, so the area the ring
encloses is preserved to a fraction of a percent (PLAN Phase 8). If the ring is still too
large for the vertex budget the tolerance is doubled until it fits, and the response says
so rather than silently shipping a different shape.

**Projection.** Rings are traced and simplified in projected metres, where distances mean
something, and converted to lon/lat once at the end. Coordinates are rounded to six
decimal places, about 0.1 m, which is finer than any grid this service builds.

Everything emitted is a plain `dict` of Python scalars: the numbers cross a JSON boundary
next, and a numpy float32 does not survive that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.config import GeoJSONConfig, settings
from app.core.catchment import Catchment
from app.core.dem_builder import DEM
from app.core.hydrology import StageStorage, WaterBalance
from app.core.kml_parser import ContourSet
from app.core.pond_siting import PondSite
from app.core.projection import projection_for
from app.core.terrain import FlowField

__all__ = [
    "GeoJSONError",
    "TracedRings",
    "trace_rings",
    "simplify",
    "mask_geometry",
    "flow_path_geometry",
    "catchment_feature",
    "pond_feature",
    "outlet_feature",
    "flow_path_feature",
    "site_features",
    "ContourDrawing",
    "contour_drawing",
    "feature_collection",
    "build_geojson",
]


_MAX_TOLERANCE_DOUBLINGS = 12
"""Ceiling on the search for a tolerance that fits the vertex budget. Twelve doublings is
a factor of 4,096, past the width of any sheet this service accepts. Bounding the loop
matters more than the exact figure: a mask of many small pieces cannot be thinned
below four vertices each, so the budget is sometimes simply unreachable."""


class GeoJSONError(Exception):
    """A geometry that cannot be expressed as GeoJSON."""

    def __init__(self, code: str, detail: str, hint: str = "") -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.hint = hint


# --------------------------------------------------------------------------- #
# Tracing a cell mask
# --------------------------------------------------------------------------- #
# Directed sides of a cell, in corner coordinates where corner (row, col) sits at the
# south-west corner of cell (row, col). Each entry is (neighbour offset, start, end): the
# side is emitted when that neighbour is outside the mask, and is directed so that the
# inside of the mask is on the left.
_SIDES = (
    ((-1, 0), (0, 0), (0, 1)),  # south side, walked east
    ((0, 1), (0, 1), (1, 1)),   # east side, walked north
    ((1, 0), (1, 1), (1, 0)),   # north side, walked west
    ((0, -1), (1, 0), (0, 0)),  # west side, walked south
)


@dataclass(frozen=True)
class TracedRings:
    """Closed rings in *corner* coordinates, before any projection."""

    exteriors: tuple[tuple[tuple[int, int], ...], ...]
    holes: tuple[tuple[tuple[int, int], ...], ...]

    @property
    def ring_count(self) -> int:
        return len(self.exteriors) + len(self.holes)


def _ring_area(ring) -> float:
    """Twice the signed area of a closed ring: positive when counter-clockwise."""
    total = 0.0
    for (y0, x0), (y1, x1) in zip(ring, ring[1:]):
        total += x0 * y1 - x1 * y0
    return total


def _point_in_ring(point, ring) -> bool:
    """Even-odd ray cast, used only to decide which exterior a hole belongs to."""
    y, x = point
    inside = False
    for (y0, x0), (y1, x1) in zip(ring, ring[1:]):
        if (y0 > y) != (y1 > y):
            crossing = x0 + (y - y0) / (y1 - y0) * (x1 - x0)
            if crossing > x:
                inside = not inside
    return inside


def trace_rings(mask: np.ndarray) -> TracedRings:
    """Closed rings around a boolean cell mask, in corner coordinates.

    Rings are exact: every vertex is a cell corner, so the enclosed area is the mask's
    area to the last square metre. Exteriors come back counter-clockwise and holes
    clockwise, which is the winding RFC 7946 requires and also the cheapest way to tell
    one from the other, by the sign of the enclosed area.
    """
    if not mask.any():
        raise GeoJSONError(
            "empty_mask", "There is nothing to draw: the cell mask is empty."
        )

    ny, nx = mask.shape
    padded = np.zeros((ny + 2, nx + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask

    # One pass over the four neighbour directions collects every boundary side, vectorised
    # rather than looped per cell: on the sample's largest basin that is 150,000 cells.
    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for (d_row, d_col), start, end in _SIDES:
        window = padded[1 + d_row : 1 + d_row + ny, 1 + d_col : 1 + d_col + nx]
        rows, cols = np.nonzero(mask & ~window)
        for row, col in zip(rows.tolist(), cols.tolist()):
            tail = (row + start[0], col + start[1])
            head = (row + end[0], col + end[1])
            outgoing.setdefault(tail, []).append(head)

    exteriors: list[tuple] = []
    holes: list[tuple] = []
    for origin in list(outgoing):
        while outgoing.get(origin):
            ring = [origin]
            current = origin
            previous = None
            while True:
                nxt = _next_step(outgoing, current, previous)
                ring.append(nxt)
                if nxt == origin:
                    break
                previous, current = current, nxt
            (exteriors if _ring_area(ring) > 0 else holes).append(tuple(ring))

    return TracedRings(exteriors=tuple(exteriors), holes=tuple(holes))


def _next_step(outgoing, current, previous):
    """Leave `current` along one of its unused sides, turning as sharply left as possible.

    Only a pinch matters here: where two cells of the mask touch at a corner and nowhere
    else, four sides meet at one vertex and the walk has a choice. The interior is on the
    left, so the leftmost turn stays on the cell being traced and closes its ring; the
    other cell gets a ring of its own that touches this one at a point. Turning right
    instead threads the two together into a figure-eight, which is not a valid polygon and
    which renderers fill wrongly.
    """
    options = outgoing[current]
    if len(options) == 1:
        return options.pop()

    if previous is None:
        return options.pop()

    incoming = (current[0] - previous[0], current[1] - previous[1])
    # Counter-clockwise from the incoming direction: left, straight on, right, back.
    order = {
        (0, 1): ((1, 0), (0, 1), (-1, 0), (0, -1)),
        (0, -1): ((-1, 0), (0, -1), (1, 0), (0, 1)),
        (1, 0): ((0, -1), (1, 0), (0, 1), (-1, 0)),
        (-1, 0): ((0, 1), (-1, 0), (0, -1), (1, 0)),
    }[incoming]
    for turn in order:
        candidate = (current[0] + turn[0], current[1] + turn[1])
        if candidate in options:
            options.remove(candidate)
            return candidate
    return options.pop()


# --------------------------------------------------------------------------- #
# Simplification
# --------------------------------------------------------------------------- #
def simplify(
    points: list[tuple[float, float]], tolerance_m: float
) -> list[tuple[float, float]]:
    """Douglas-Peucker, iteratively rather than recursively.

    A traced ring can carry tens of thousands of vertices and Python's recursion limit is
    1,000; an explicit stack of index ranges does the same work without that ceiling.
    """
    if tolerance_m <= 0 or len(points) < 3:
        return list(points)

    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        x0, y0 = points[first]
        x1, y1 = points[last]
        dx, dy = x1 - x0, y1 - y0
        span = math.hypot(dx, dy)

        best_index, best_distance = -1, -1.0
        for index in range(first + 1, last):
            x, y = points[index]
            if span == 0.0:
                # A closed ring's endpoints coincide; fall back to radial distance so the
                # far side of the ring is never discarded wholesale.
                distance = math.hypot(x - x0, y - y0)
            else:
                distance = abs(dy * (x - x0) - dx * (y - y0)) / span
            if distance > best_distance:
                best_index, best_distance = index, distance

        if best_distance > tolerance_m:
            keep[best_index] = True
            stack.append((first, best_index))
            stack.append((best_index, last))

    return [point for point, kept in zip(points, keep.tolist()) if kept]


def _corners_to_xy(ring, dem: DEM) -> list[tuple[float, float]]:
    """Corner indices to projected metres. Corner (row, col) is the south-west corner of
    cell (row, col), so it sits half a cell below and left of that cell's centre."""
    x0, y0 = dem.origin_xy
    res = dem.resolution_m
    return [
        (x0 + (col - 0.5) * res, y0 + (row - 0.5) * res) for row, col in ring
    ]


def _xy_to_lonlat(ring, dem: DEM, precision: int) -> list[list[float]]:
    xy = np.asarray(ring, dtype=np.float64)
    lonlat = dem.projection.inverse(xy)
    return [
        [round(float(lon), precision), round(float(lat), precision)]
        for lon, lat in lonlat
    ]


def _close(ring: list) -> list:
    """GeoJSON rings are explicitly closed; simplification can drop the repeat."""
    if ring and ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    return ring


def _thin(ring: list, tolerance_m: float) -> list:
    """Simplify a closed ring, keeping it a ring.

    A pond only a few cells across can be thinned below the three distinct points a
    polygon needs. Dropping it would lose the geometry that matters most at a village
    scale, so the traced staircase is kept as it is: blocky, but true.
    """
    thinned = _close(simplify(ring, tolerance_m))
    return thinned if len(thinned) >= 4 else _close(list(ring))


def mask_geometry(
    mask: np.ndarray, dem: DEM, *, config: GeoJSONConfig | None = None
) -> tuple[dict, tuple[str, ...]]:
    """A cell mask as a Polygon or MultiPolygon, plus any warnings about it.

    Holes are placed inside the exterior that contains them, because a catchment can
    genuinely have one where a closed depression drains to itself. Disjoint components become
    a MultiPolygon rather than being silently dropped.
    """
    cfg = config or settings.geojson
    rings = trace_rings(mask)

    traced_exteriors = [_corners_to_xy(r, dem) for r in rings.exteriors]
    traced_holes = [_corners_to_xy(r, dem) for r in rings.holes]

    start_tolerance = cfg.simplify_tolerance_cells * dem.resolution_m
    tolerance = start_tolerance
    for _ in range(_MAX_TOLERANCE_DOUBLINGS):
        exteriors = [_thin(r, tolerance) for r in traced_exteriors]
        holes = [_thin(r, tolerance) for r in traced_holes]
        total = sum(len(r) for r in exteriors) + sum(len(r) for r in holes)
        if total <= cfg.max_polygon_vertices:
            break
        # Doubling rather than nudging: the vertex count falls roughly as 1/tolerance, so
        # a handful of doublings settles any ring this service can produce.
        tolerance *= 2

    warnings: list[str] = []
    if tolerance > start_tolerance:
        warnings.append(
            f"Boundary simplified at {tolerance:.1f} m rather than "
            f"{start_tolerance:.1f} m to stay under {cfg.max_polygon_vertices} vertices."
        )
    if total > cfg.max_polygon_vertices:
        # Reachable: a mask of many small components has a floor of four vertices each,
        # which no tolerance goes under. Say so rather than loop forever chasing it.
        warnings.append(
            f"The boundary needs {total} vertices, above the {cfg.max_polygon_vertices} "
            "budget: it is made of many separate pieces, each of which needs a ring."
        )

    if not exteriors:
        raise GeoJSONError(
            "degenerate_geometry", "The traced boundary enclosed nothing."
        )

    polygons: list[list[list[list[float]]]] = []
    for exterior in exteriors:
        polygon = [_xy_to_lonlat(exterior, dem, cfg.coordinate_precision)]
        polygons.append(polygon)

    for hole in holes:
        # Smallest containing exterior wins: nested rings would otherwise attach a hole to
        # the outermost ring, punching through ground that is genuinely inside.
        best, best_area = None, math.inf
        for index, exterior in enumerate(exteriors):
            if _point_in_ring(hole[0][::-1], [(y, x) for x, y in exterior]):
                area = abs(_ring_area([(y, x) for x, y in exterior]))
                if area < best_area:
                    best, best_area = index, area
        if best is not None:
            polygons[best].append(_xy_to_lonlat(hole, dem, cfg.coordinate_precision))

    if len(polygons) == 1:
        geometry = {"type": "Polygon", "coordinates": polygons[0]}
    else:
        geometry = {"type": "MultiPolygon", "coordinates": polygons}
    return geometry, tuple(dict.fromkeys(warnings))


def flow_path_geometry(flow: FlowField, catchment: Catchment, dem: DEM) -> dict:
    """The longest flow path as a LineString, from the most remote cell to the outlet.

    Walked down the receiver pointers, which is the same path `Catchment` measured, so
    the drawn line and the reported length are the same object, not two estimates of one.
    """
    receivers = flow.receivers
    nx = dem.shape[1]
    outlet = catchment.outlet_rc[0] * nx + catchment.outlet_rc[1]

    cells = [catchment.flow_path_cell]
    while cells[-1] != outlet:
        nxt = int(receivers[cells[-1]])
        if nxt < 0:
            break
        cells.append(nxt)

    rows, cols = np.divmod(np.asarray(cells, dtype=np.int64), nx)
    lonlat = dem.lonlat_of(rows, cols)
    precision = settings.geojson.coordinate_precision
    return {
        "type": "LineString",
        "coordinates": [
            [round(float(lon), precision), round(float(lat), precision)]
            for lon, lat in lonlat
        ],
    }


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def _feature(geometry: dict, properties: dict) -> dict:
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def catchment_feature(
    site: PondSite, dem: DEM, *, config: GeoJSONConfig | None = None
) -> dict:
    """The catchment polygon, carrying the numbers that justify it."""
    geometry, warnings = mask_geometry(site.catchment.mask, dem, config=config)
    catchment = site.catchment
    properties = {
        "role": "catchment",
        "rank": site.rank,
        "area_ha": round(catchment.area_ha, 2),
        "confidence": site.confidence,
        "edge_contact_pct": round(catchment.edge_contact * 100, 2),
        "is_lower_bound": catchment.is_lower_bound,
        "relief_m": round(catchment.relief_m, 2),
        "longest_flow_path_m": round(catchment.longest_flow_path_m, 1),
        "resolution_m": catchment.resolution_m,
        # simplestyle-spec, which geojson.io and Leaflet plugins both understand. The
        # accept test for this phase is that the output *looks* right when dropped on a map.
        "stroke": "#1f78b4",
        "fill": "#a6cee3",
        "fill-opacity": 0.25,
    }
    if site.ensemble is not None:
        properties["area_uncertainty_ha"] = round(site.ensemble.std_area_ha, 2)
    if warnings:
        properties["warnings"] = list(warnings)
    return _feature(geometry, properties)


def pond_feature(
    storage: StageStorage, dem: DEM, *, rank: int = 1, config: GeoJSONConfig | None = None
) -> dict | None:
    """The water surface of the pond, at the stage the site can hold.

    Drawn for every site, not just the chosen one: a candidate catchment without its
    water body shows the ground but not the thing being built on it.

    None when the pond is a single cell: a 25 m^2 square drawn on a satellite image says
    less than nothing, and the storage numbers carry the fact anyway.
    """
    if storage.pond_mask.sum() < 2:
        return None
    primary = rank == 1
    geometry, _ = mask_geometry(storage.pond_mask, dem, config=config)
    return _feature(
        geometry,
        {
            "role": "pond",
            "rank": rank,
            "capacity_m3": round(storage.usable_capacity_m3),
            "surface_area_m2": round(storage.usable_area_m2),
            "depth_m": round(
                storage.spill_stage_m
                if storage.spill_stage_m is not None
                else storage.max_depth_m,
                2,
            ),
            # Paler for the alternates, the same way their markers are, so one glance
            # separates the pond that was recommended from the ponds that were offered.
            "stroke": "#08519c" if primary else "#3182bd",
            "fill": "#3182bd" if primary else "#9ecae1",
            "fill-opacity": 0.6 if primary else 0.45,
        },
    )


def outlet_feature(site: PondSite, balance: WaterBalance | None = None) -> dict:
    """The pond location itself, with the reasons it was chosen."""
    lon, lat = site.lonlat
    precision = settings.geojson.coordinate_precision
    properties = {
        "role": "pond_site",
        "rank": site.rank,
        "recommended": site.is_recommended,
        "catchment_area_ha": round(site.catchment.area_ha, 2),
        "confidence": site.confidence,
        "slope_pct": round(site.score.slope * 100, 2),
        "relative_elevation_m": round(site.score.relative_elevation_m, 2),
        "height_above_watercourse_m": (
            round(site.score.height_above_trunk_m, 2)
            if math.isfinite(site.score.height_above_trunk_m)
            else None
        ),
        "snap_distance_m": round(site.catchment.snap_distance_m, 1),
        "marker-color": "#08519c" if site.rank == 1 else "#6baed6",
        "marker-symbol": "water",
    }
    if site.warnings:
        properties["warnings"] = list(site.warnings)
    if balance is not None:
        properties["annual_runoff_m3"] = round(balance.annual_runoff_m3)
        properties["capacity_m3"] = round(balance.storage.usable_capacity_m3)
        properties["fill_ratio"] = (
            round(balance.fill_ratio, 2) if math.isfinite(balance.fill_ratio) else None
        )
        properties["time_of_concentration_min"] = round(
            balance.time_of_concentration_min, 1
        )
    return _feature(
        {
            "type": "Point",
            "coordinates": [round(lon, precision), round(lat, precision)],
        },
        properties,
    )


def flow_path_feature(flow: FlowField, site: PondSite, dem: DEM) -> dict:
    """The line Kirpich's time of concentration is measured along."""
    return _feature(
        flow_path_geometry(flow, site.catchment, dem),
        {
            "role": "flow_path",
            "rank": site.rank,
            "length_m": round(site.catchment.longest_flow_path_m, 1),
            "relief_m": round(site.catchment.flow_path_relief_m, 2),
            "stroke": "#e31a1c",
            "stroke-width": 2,
        },
    )


def site_features(
    flow: FlowField,
    site: PondSite,
    balance: WaterBalance | None = None,
    *,
    detailed: bool = True,
    config: GeoJSONConfig | None = None,
) -> list[dict]:
    """Everything drawable about one site.

    `detailed` is off for the alternates: the longest flow path belongs to the site being
    costed, and five flow paths over one sheet is noise rather than information. The
    catchment, the pond and the outlet are drawn for every site, because the question a
    reader asks of an alternate is where its pond would sit.
    """
    dem = flow.dem
    features = [catchment_feature(site, dem, config=config)]
    if detailed:
        features.append(flow_path_feature(flow, site, dem))
    if balance is not None:
        pond = pond_feature(balance.storage, dem, rank=site.rank, config=config)
        if pond is not None:
            features.append(pond)
    features.append(outlet_feature(site, balance))
    return features


# --------------------------------------------------------------------------- #
# Contours
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContourDrawing:
    """The input contours as something a map can draw over the answer.

    The point of handing these back is a check no number can make for the reader: a
    catchment boundary is right when it runs along the ridges, and that is visible in one
    glance with the contours underneath it and invisible without them.
    """

    geojson: dict
    tolerance_m: float
    """Simplification actually applied, which is the requested one unless the vertex
    budget forced it up."""

    vertex_count: int
    warnings: tuple[str, ...] = ()


def _ramp_colour(fraction: float, ramp: tuple[str, str, str]) -> str:
    """A colour off the three-stop elevation ramp, linearly interpolated in sRGB.

    sRGB rather than a perceptual space because the stops are already close together in
    lightness: the ramp says "how high", and the *interval* between the lines says how
    steep, so nothing here depends on the ramp being perceptually uniform.
    """
    fraction = min(max(fraction, 0.0), 1.0)
    lower = fraction < 0.5
    low, high = (ramp[0], ramp[1]) if lower else (ramp[1], ramp[2])
    t = fraction * 2.0 if lower else fraction * 2.0 - 1.0

    def channel(index: int) -> int:
        a = int(low[1 + 2 * index : 3 + 2 * index], 16)
        b = int(high[1 + 2 * index : 3 + 2 * index], 16)
        return round(a + (b - a) * t)

    return "#%02x%02x%02x" % (channel(0), channel(1), channel(2))


def _index_levels(levels: tuple[float, ...], interval: float | None, every: int) -> set:
    """Which contour levels print heavy.

    Every nth level counted off a round multiple of the interval, not off the lowest line
    in the file: a sheet clipped to start at 267 m should still make 270 and 275 the heavy
    lines, because those are the numbers a reader expects to be heavy.
    """
    if interval is None or interval <= 0.0 or every <= 1:
        return set()
    step = interval * every
    return {
        level
        for level in levels
        if abs(level - round(level / step) * step) < interval * 0.25
    }


def contour_drawing(
    contours: ContourSet,
    *,
    tolerance_m: float | None = None,
    config: GeoJSONConfig | None = None,
) -> ContourDrawing:
    """Every contour line in the file as a styled `LineString`, thinned for a browser.

    Thinning happens in projected metres, and the default tolerance is three quarters of
    the finest grid this service will ever build, so what is drawn and what was analysed
    are the same lines to within a fraction of a cell. A client that wants the file
    exactly can ask for no thinning at all.

    Colour carries the elevation, because a line on its own cannot: without it a reader
    sees nested loops and has no way to tell a hill from a hollow. Every nth line is drawn
    heavy, which is how a printed sheet says the same thing.
    """
    cfg = config or settings.geojson
    if contours.line_count == 0:
        raise GeoJSONError(
            "no_contours",
            "There are no contour lines to draw.",
            "Upload a sheet with contour geometry in it.",
        )

    requested = cfg.contour_simplify_tolerance_m if tolerance_m is None else tolerance_m
    requested = max(0.0, float(requested))
    tolerance = requested
    precision = cfg.coordinate_precision
    metadata = contours.metadata
    low, high = metadata.elevation_range
    span = high - low
    index_levels = _index_levels(
        metadata.levels, metadata.interval_m, cfg.contour_index_every
    )

    # One projection for the whole sheet, built off every vertex, so simplification is in
    # metres everywhere rather than in degrees that mean different distances by latitude.
    projection = projection_for(contours.points)
    xy = projection.forward(contours.points)
    elevations = contours.line_elevations

    warnings: list[str] = []
    for attempt in range(_MAX_TOLERANCE_DOUBLINGS + 1):
        # Elevation travels with its line rather than being looked up by index later: a
        # line thinned below two points is dropped, and a positional lookup would then
        # hand every line after it the wrong height.
        lines: list[tuple[list, float]] = []
        total = 0
        for index in range(contours.line_count):
            start, end = contours.line_starts[index], contours.line_starts[index + 1]
            points = [(float(x), float(y)) for x, y in xy[start:end]]
            thinned = simplify(points, tolerance) if tolerance > 0.0 else points
            # Two points is still a line; anything less is not drawable.
            if len(thinned) < 2:
                continue
            lines.append((thinned, float(elevations[index])))
            total += len(thinned)
        if total <= cfg.contour_max_vertices or attempt == _MAX_TOLERANCE_DOUBLINGS:
            break
        tolerance = tolerance * 2.0 if tolerance > 0.0 else 1.0
    if tolerance > requested:
        warnings.append(
            f"The contour overlay was thinned to {tolerance:.1f} m to stay under "
            f"{cfg.contour_max_vertices:,} vertices. The lines are the ones in the file; "
            "they are drawn with fewer points."
        )

    features: list[dict] = []
    for thinned, elevation in lines:
        is_index = elevation in index_levels
        features.append(
            _feature(
                {
                    "type": "LineString",
                    "coordinates": [
                        [round(float(lon), precision), round(float(lat), precision)]
                        for lon, lat in projection.inverse(np.asarray(thinned))
                    ],
                },
                {
                    "role": "contour",
                    "elevation_m": round(float(elevation), 2),
                    "index": is_index,
                    # simplestyle again, so the overlay draws the same in this service's
                    # demo page and in geojson.io. Index lines are heavier and more solid,
                    # which is the convention a printed sheet uses to say the same thing.
                    "stroke": _ramp_colour(
                        (elevation - low) / span if span > 0.0 else 0.5, cfg.contour_ramp
                    ),
                    "stroke-width": 1.6 if is_index else 0.8,
                    "stroke-opacity": 0.95 if is_index else 0.7,
                },
            )
        )

    return ContourDrawing(
        geojson=feature_collection(features, bbox=metadata.bbox),
        tolerance_m=round(tolerance, 2),
        vertex_count=sum(len(line) for line, _ in lines),
        warnings=tuple(warnings),
    )


def feature_collection(features: list[dict], *, bbox: tuple | None = None) -> dict:
    collection: dict = {"type": "FeatureCollection", "features": features}
    if bbox is not None:
        precision = settings.geojson.coordinate_precision
        collection["bbox"] = [round(float(v), precision) for v in bbox]
    return collection


def build_geojson(
    flow: FlowField,
    sites,
    balances=None,
    *,
    config: GeoJSONConfig | None = None,
) -> dict:
    """The whole response geometry: every site's catchment, pond and outlet, and the
    longest flow path of the recommended one.

    Features come out in draw order, with catchments first, then lines, then the points
    that sit on top of them, because most renderers honour it. geojson.io does.
    """
    sites = list(sites)
    if not sites:
        raise GeoJSONError("no_sites", "There are no sites to draw.")
    balances = list(balances) if balances is not None else [None] * len(sites)
    if len(balances) < len(sites):
        balances = balances + [None] * (len(sites) - len(balances))

    areas: list[dict] = []
    lines: list[dict] = []
    points: list[dict] = []
    for index, (site, balance) in enumerate(zip(sites, balances)):
        for feature in site_features(
            flow, site, balance, detailed=index == 0, config=config
        ):
            geometry_type = feature["geometry"]["type"]
            if geometry_type == "Point":
                points.append(feature)
            elif geometry_type == "LineString":
                lines.append(feature)
            else:
                areas.append(feature)

    return feature_collection(areas + lines + points, bbox=_bbox(areas + lines + points))


def _bbox(features: list[dict]) -> tuple[float, float, float, float]:
    """(min lon, min lat, max lon, max lat) over every coordinate emitted."""
    lons: list[float] = []
    lats: list[float] = []

    def walk(coordinates) -> None:
        if isinstance(coordinates[0], (int, float)):
            lons.append(coordinates[0])
            lats.append(coordinates[1])
            return
        for part in coordinates:
            walk(part)

    for feature in features:
        walk(feature["geometry"]["coordinates"])
    return (min(lons), min(lats), max(lons), max(lats))
