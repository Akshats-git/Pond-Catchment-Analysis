"""Place search, proxied to Nominatim (the HLD's `GET /api/villages`).

Proxied rather than called from the browser for three reasons: one cache serves every
user, the service sends the identifying User-Agent Nominatim's policy asks for, and the
one-request-a-second limit is enforced in one place instead of per browser tab.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict

from fastapi import APIRouter, Query, status
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.errors import APIError
from app.schemas.responses import ErrorResponse

router = APIRouter(tags=["places"])

_CACHE: OrderedDict[str, list[dict]] = OrderedDict()
_LOCK = threading.Lock()
_LAST = [0.0]


class PlacesUnavailable(Exception):
    pass


def search(query: str) -> list[dict]:
    cfg = settings.places
    key = query.strip().lower()
    with _LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    if not cfg.enabled:
        raise PlacesUnavailable("Place search is switched off on this server.")

    params = {"q": query, "format": "jsonv2", "limit": cfg.max_results, "addressdetails": 0}
    if cfg.country_codes:
        params["countrycodes"] = cfg.country_codes
    request = urllib.request.Request(
        f"{cfg.url}?{urllib.parse.urlencode(params)}",
        headers={"User-Agent": cfg.user_agent, "Accept": "application/json"},
    )
    # Nominatim allows one request a second from one client. Serialise and space them.
    with _LOCK:
        wait = _LAST[0] + cfg.min_interval_s - time.time()
        if wait > 0:
            time.sleep(wait)
        _LAST[0] = time.time()
        try:
            with urllib.request.urlopen(request, timeout=cfg.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise PlacesUnavailable(f"Nominatim could not be reached: {exc}") from exc

    results = []
    for item in payload if isinstance(payload, list) else []:
        try:
            south, north, west, east = (float(v) for v in item.get("boundingbox", []))
            bbox = [west, south, east, north]
        except (TypeError, ValueError):
            bbox = None
        try:
            results.append({
                "name": item.get("name") or item.get("display_name", "").split(",")[0],
                "display_name": item.get("display_name", ""),
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "type": item.get("type") or item.get("category"),
                "bbox": bbox,
            })
        except (KeyError, TypeError, ValueError):
            continue
    with _LOCK:
        _CACHE[key] = results
        while len(_CACHE) > cfg.cache_size:
            _CACHE.popitem(last=False)
    return results


@router.get(
    "/places",
    responses={503: {"model": ErrorResponse, "description": "The place search is unavailable."}},
    summary="Find a village or town by name",
)
async def places(q: str = Query(..., min_length=2, max_length=200, description="Place name.")) -> dict:
    """Up to eight matches from OpenStreetMap's Nominatim, each with a point and a bounding
    box to fly the map to. Cached; a repeated search does not leave the server."""
    try:
        return {"query": q, "results": await run_in_threadpool(search, q)}
    except PlacesUnavailable as exc:
        raise APIError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "places_unavailable",
            str(exc),
            "Pan and zoom the map to the village yourself; the analysis does not need the search.",
        ) from exc
