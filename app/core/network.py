"""The drainage network as lines a map can draw.

The catchment polygon says what drains to the pond. The network says how everything else
on the map drains, and it is the layer that lets a reader check the router against the
imagery: the blue lines should run down the nalas visible in the satellite picture, and
where they do not, the DEM is wrong there and the answer should be trusted less.

Every stream cell links to the cell it drains into. Chains are broken at confluences, so
each line is one reach with one upstream area, and wider lines carry more water.
"""

from __future__ import annotations

import math

import numpy as np

from app.config import GeoJSONConfig, settings
from app.core.geojson import feature_collection, simplify
from app.core.terrain import FlowField

__all__ = ["stream_network"]


def stream_network(
    flow: FlowField,
    threshold_m2: float,
    *,
    trunk_threshold_m2: float | None = None,
    max_vertices: int = 60_000,
    config: GeoJSONConfig | None = None,
) -> dict:
    """FeatureCollection of `LineString` reaches, one per stretch between confluences.

    `threshold_m2` is the same stream threshold the siting step used, so the drawn network
    is exactly the set of channels a pond was allowed to sit on. Reaches above
    `trunk_threshold_m2` are flagged `watercourse`, the ones a pond must stand clear of.
    """
    cfg = config or settings.geojson
    dem = flow.dem
    ny, nx = dem.shape
    upstream = flow.accumulation * dem.row_cell_areas[:, None]
    stream = (dem.valid & (upstream >= threshold_m2)).ravel()
    receivers = flow.receivers
    stream_idx = np.flatnonzero(stream)
    if stream_idx.size == 0:
        return feature_collection([])

    # How many stream cells drain into each cell. A chain runs through cells with exactly
    # one stream donor; anything else starts a new reach.
    targets = receivers[stream_idx]
    donors = np.zeros(ny * nx, dtype=np.int32)
    ok = targets >= 0
    np.add.at(donors, targets[ok], 1)

    heads = stream_idx[donors[stream_idx] != 1]
    precision = cfg.coordinate_precision
    area_flat = upstream.ravel()
    receivers_list = receivers.tolist()
    stream_list = stream.tolist()
    donors_list = donors.tolist()

    reaches: list[tuple[list[int], float]] = []
    for head in heads.tolist():
        chain = [head]
        cell = head
        while True:
            nxt = receivers_list[cell]
            if nxt < 0 or not stream_list[nxt]:
                break
            chain.append(nxt)
            if donors_list[nxt] != 1:
                break  # A confluence: it ends this reach and heads the next.
            cell = nxt
        if len(chain) >= 2:
            reaches.append((chain, float(area_flat[chain[-1]])))

    x0, y0 = dem.origin_xy
    res = dem.resolution_m
    tolerance = res * 0.5
    features: list[dict] = []
    total = 0
    # Largest first, so a vertex cap drops the smallest rills and never the main channel.
    for chain, area in sorted(reaches, key=lambda item: -item[1]):
        rows, cols = np.divmod(np.asarray(chain), nx)
        points = list(zip((x0 + cols * res).tolist(), (y0 + rows * res).tolist()))
        points = simplify(points, tolerance)
        if total + len(points) > max_vertices:
            break
        total += len(points)
        lonlat = dem.projection.inverse(np.asarray(points))
        area_ha = area / 1e4
        watercourse = trunk_threshold_m2 is not None and area >= trunk_threshold_m2
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [round(float(lon), precision), round(float(lat), precision)]
                        for lon, lat in lonlat
                    ],
                },
                "properties": {
                    "role": "stream",
                    "upstream_area_ha": round(area_ha, 1),
                    "watercourse": bool(watercourse),
                    "stroke": "#1d6fb8" if watercourse else "#4aa3df",
                    "stroke-width": round(min(4.0, 0.8 + 0.6 * math.log10(max(area_ha, 1.0))), 2),
                    "stroke-opacity": 0.9,
                },
            }
        )
    return feature_collection(features)
