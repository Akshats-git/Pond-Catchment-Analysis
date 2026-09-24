"""Contour lines from a DEM (HLD FR-2).

The contour path starts from contour lines; the map path starts from a grid and has none.
This draws them, so a drawn area gets the same contour overlay an uploaded sheet does, and
the reader can check a catchment boundary against the ridges either way.

Marching squares, in numpy, with no new dependency: each grid cell's four corners are
classified above or below the level, the sixteen cases say which cell edges the line
crosses, and the crossings are joined into polylines by the edges they share. Edges are
identified by integer ids, not by floating-point positions, so two segments meet exactly
or not at all.

The output is a `ContourSet`, the same object the KML parser produces, so everything
downstream of the parser (styling, thinning, drawing, rendering) is reused unchanged.
"""

from __future__ import annotations

import math

import numpy as np

from app.core.dem_builder import DEM
from app.core.kml_parser import ContourMetadata, ContourSet

__all__ = ["ContourError", "nice_interval", "contour_levels", "contours_from_dem"]


class ContourError(Exception):
    def __init__(self, code: str, detail: str, hint: str = "") -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.hint = hint


_TARGET_LEVELS = 25
_MAX_LEVELS = 400

# Edges of a cell: 0 bottom (v0-v1), 1 right (v1-v2), 2 top (v3-v2), 3 left (v0-v3), with
# v0 = (r, c), v1 = (r, c+1), v2 = (r+1, c+1), v3 = (r+1, c). Row 0 is south, as in `DEM`.
_SEGMENTS: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((3, 0),), 2: ((0, 1),), 3: ((3, 1),), 4: ((1, 2),),
    6: ((0, 2),), 7: ((3, 2),), 8: ((2, 3),), 9: ((0, 2),),
    11: ((1, 2),), 12: ((1, 3),), 13: ((0, 1),), 14: ((3, 0),),
}
# The two saddles, resolved by the cell-centre average: (centre above, centre below).
_SADDLES: dict[int, tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = {
    5: (((0, 1), (2, 3)), ((3, 0), (1, 2))),
    10: (((3, 0), (1, 2)), ((0, 1), (2, 3))),
}


def nice_interval(span_m: float, target: int = _TARGET_LEVELS) -> float:
    """A round contour interval, 1, 2, 2.5 or 5 times a power of ten, giving about
    `target` lines over the span. Never under half a metre, and on SRTM's whole-metre heights
    never usefully under one."""
    if span_m <= 0:
        return 1.0
    raw = span_m / target
    power = 10 ** math.floor(math.log10(raw))
    for step in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= step * power:
            return max(0.5, step * power)
    return max(0.5, 10.0 * power)


def contour_levels(z_min: float, z_max: float, interval: float) -> np.ndarray:
    """Every multiple of `interval` strictly inside the range."""
    first = math.floor(z_min / interval) + 1
    last = math.ceil(z_max / interval) - 1
    return np.arange(first, last + 1) * interval


def _edge_point(edge: np.ndarray, r: np.ndarray, c: np.ndarray, z: np.ndarray, level: float):
    """Crossing position (col, row) in grid units, and a unique integer id, per edge."""
    ny, nx = z.shape
    # Endpoints of each edge.
    ra = np.where(edge == 2, r + 1, r)
    ca = np.where(edge == 1, c + 1, c)
    rb = np.where((edge == 1) | (edge == 2) | (edge == 3), r + 1, r)
    cb = np.where((edge == 0) | (edge == 1) | (edge == 2), c + 1, c)
    za, zb = z[ra, ca], z[rb, cb]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.clip((level - za) / (zb - za), 0.0, 1.0)
    t = np.nan_to_num(t, nan=0.5)
    col = ca + t * (cb - ca)
    row = ra + t * (rb - ra)
    horizontal = (edge == 0) | (edge == 2)
    ids = np.where(horizontal, ra * nx + ca, ny * nx + ra * nx + ca)
    return col, row, ids


def _segments(z: np.ndarray, level: float):
    """All crossing segments of one level: (ids_a, ids_b, pts_a, pts_b)."""
    v0, v1 = z[:-1, :-1], z[:-1, 1:]
    v2, v3 = z[1:, 1:], z[1:, :-1]
    finite = np.isfinite(v0) & np.isfinite(v1) & np.isfinite(v2) & np.isfinite(v3)
    with np.errstate(invalid="ignore"):
        case = (
            (v0 >= level).astype(np.int8)
            | ((v1 >= level).astype(np.int8) << 1)
            | ((v2 >= level).astype(np.int8) << 2)
            | ((v3 >= level).astype(np.int8) << 3)
        )
    case[~finite] = 0
    centre_above = (v0 + v1 + v2 + v3) / 4.0 >= level

    edges_a: list[np.ndarray] = []
    edges_b: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []

    def emit(mask: np.ndarray, pairs) -> None:
        r, c = np.nonzero(mask)
        if r.size == 0:
            return
        for a, b in pairs:
            edges_a.append(np.full(r.size, a))
            edges_b.append(np.full(r.size, b))
            rows.append(r)
            cols.append(c)

    for value, pairs in _SEGMENTS.items():
        emit(case == value, pairs)
    for value, (above, below) in _SADDLES.items():
        at = case == value
        emit(at & centre_above, above)
        emit(at & ~centre_above, below)

    if not rows:
        return None
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    ca, ra, ida = _edge_point(np.concatenate(edges_a), r, c, z, level)
    cb, rb, idb = _edge_point(np.concatenate(edges_b), r, c, z, level)
    return ida, idb, np.stack([ca, ra], axis=1), np.stack([cb, rb], axis=1)


def _join(ida: np.ndarray, idb: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> list[np.ndarray]:
    """Chain segments sharing an edge id into polylines. Each edge is shared by at most two
    segments (the two cells either side of it), so every id has at most two neighbours."""
    n = ida.size
    touching: dict[int, list[int]] = {}
    for index, (a, b) in enumerate(zip(ida.tolist(), idb.tolist())):
        touching.setdefault(a, []).append(index)
        touching.setdefault(b, []).append(index)
    point_of: dict[int, tuple[float, float]] = {}
    for ids, pts in ((ida, pa), (idb, pb)):
        for key, (x, y) in zip(ids.tolist(), pts.tolist()):
            point_of[key] = (x, y)
    ends = list(zip(ida.tolist(), idb.tolist()))

    used = bytearray(n)
    lines: list[np.ndarray] = []
    for start in range(n):
        if used[start]:
            continue
        used[start] = 1
        a, b = ends[start]
        chain = [a, b]
        # Walk forward from b, then backward from a.
        for direction in (1, -1):
            tip = chain[-1] if direction == 1 else chain[0]
            while True:
                nxt = next((s for s in touching[tip] if not used[s]), None)
                if nxt is None:
                    break
                used[nxt] = 1
                s_a, s_b = ends[nxt]
                tip = s_b if s_a == tip else s_a
                if direction == 1:
                    chain.append(tip)
                else:
                    chain.insert(0, tip)
        lines.append(np.array([point_of[k] for k in chain], dtype=np.float64))
    return lines


def contours_from_dem(
    dem: DEM,
    *,
    interval_m: float | None = None,
    mask: np.ndarray | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> ContourSet:
    """Contour lines of `dem`, as a `ContourSet` indistinguishable from a parsed sheet's.

    `interval_m` defaults to a round figure giving about 25 lines. `mask` limits the lines
    to some cells (they are traced as if the rest were no-data); `bbox` is what the
    metadata reports as the extent.
    """
    z = np.where(dem.valid if mask is None else (mask & dem.valid), dem.z, np.nan)
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        raise ContourError("no_elevation", "There is no elevation data to contour.", "")
    z_min, z_max = float(finite.min()), float(finite.max())
    span = z_max - z_min

    if interval_m is None:
        interval = nice_interval(span)
    else:
        if interval_m <= 0:
            raise ContourError("invalid_interval", "interval_m must be greater than zero.", "")
        interval = float(interval_m)
    levels = contour_levels(z_min, z_max, interval)
    if levels.size > _MAX_LEVELS:
        raise ContourError(
            "invalid_interval",
            f"A {interval:g} m interval over {span:.0f} m of relief is {levels.size} lines; "
            f"the limit is {_MAX_LEVELS}.",
            f"Ask for {nice_interval(span, _MAX_LEVELS // 2):g} m or more, or omit it.",
        )

    x0, y0 = dem.origin_xy
    res = dem.resolution_m
    pieces: list[np.ndarray] = []
    heights: list[float] = []
    for level in levels.tolist():
        found = _segments(z, level)
        if found is None:
            continue
        for line in _join(*found):
            if len(line) < 2:
                continue
            xy = np.stack([x0 + line[:, 0] * res, y0 + line[:, 1] * res], axis=1)
            pieces.append(dem.projection.inverse(xy))
            heights.append(level)

    if not pieces:
        raise ContourError(
            "no_contours",
            f"The ground spans only {span:.2f} m, less than one {interval:g} m interval.",
            "Ask for a smaller interval_m.",
        )

    counts = np.array([len(p) for p in pieces])
    points = np.concatenate(pieces)
    elevations = np.repeat(np.asarray(heights), counts)
    line_starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    if bbox is None:
        bbox = (
            float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max()),
        )
    present = tuple(sorted(set(heights)))
    return ContourSet(
        points=points,
        elevations=elevations,
        line_starts=line_starts,
        metadata=ContourMetadata(
            elevation_source="dem",
            interval_m=interval,
            levels=present,
            elevation_range=(min(present), max(present)),
            bbox=tuple(bbox),
            line_count=len(pieces),
            vertex_count=int(points.shape[0]),
            warnings=(
                f"Contours generated from the elevation grid at a {interval:g} m interval. "
                "They are drawn from ~30 m data, so read them for the shape of the ground, "
                "not for heights to the metre.",
            ),
        ),
    )


def total_length_m(contours: ContourSet, dem: DEM) -> float:
    """Total contour length in metres, per line, for the spacing the renderer needs."""
    xy = dem.projection.forward(contours.points)
    total = 0.0
    for i in range(contours.line_count):
        seg = xy[contours.line_starts[i] : contours.line_starts[i + 1]]
        total += float(np.hypot(*np.diff(seg, axis=0).T).sum())
    return total
