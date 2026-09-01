"""The answer as a picture: satellite tiles, the contour sheet, and the catchment on top.

Everything here is downstream of the analysis and changes none of it. The GeoJSON is the
answer; this draws it, in the same colours, so a reader who cannot run a map client can
still do the one check that matters — does the catchment boundary follow the ridges the
contours show? A number cannot answer that and a picture answers it in a glance.

**Web Mercator, because the tiles are.** The analysis runs in a local equirectangular
frame (`app.core.projection`), which is the right frame for comparing slopes and the
wrong one for compositing tiles. So the drawing happens in Web Mercator pixels: lon/lat
in, screen pixels out, one integer zoom chosen so the extent fits the canvas. The two
frames disagree by a fraction of a pixel over a 3 km sheet, and only the picture depends
on the second one.

**The overlay is drawn oversized and scaled down.** Pillow's draw primitives have no
anti-aliasing, and a staircased catchment boundary is not a cosmetic problem when the
boundary is the thing being checked. `RenderConfig.supersample` sets the factor.

**The basemap is allowed to fail.** Tiles need outbound network, and a service that
returns a 502 because a tile server rate-limited it has turned somebody else's outage
into its own. Past `RenderConfig.tile_failure_ratio` the render falls back to a hillshade
of the DEM the analysis actually ran on — which needs no network, cannot fail, and shows
more of what the flow router saw than imagery does — and says so in a warning. This is
the same bargain `app.providers.rainfall` makes with its climatology.

**Attribution is burnt into the image.** A PNG travels without the response that carried
it. Pasted into a report it has to carry its own credit, so the credit is drawn on it.
"""

from __future__ import annotations

import io
import math
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.config import RenderConfig, settings
from app.core.dem_builder import DEM

__all__ = ["RenderError", "MapView", "render_png"]

_TILE_ORIGIN_SHIFT = 85.05112878
"""Web Mercator is undefined at the poles and every tile scheme clips here."""

_HILLSHADE_ROWS = 128
"""Canvas rows warped per pass. Small enough that the float64 temporaries stay under a
few megabytes, large enough that the per-block overhead disappears."""


class RenderError(Exception):
    """A render that cannot be produced. Same `(code, detail, hint)` triple as every
    other core module, so the route's status table needs no special case."""

    def __init__(self, code: str, detail: str, hint: str = "") -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.hint = hint


# --------------------------------------------------------------------------- #
# Web Mercator
# --------------------------------------------------------------------------- #
def _project(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """lon/lat degrees -> the unit square, y increasing southward as tiles number."""
    lat = np.clip(lat, -_TILE_ORIGIN_SHIFT, _TILE_ORIGIN_SHIFT)
    sin = np.sin(np.radians(lat))
    return (
        (lon + 180.0) / 360.0,
        0.5 - np.log((1.0 + sin) / (1.0 - sin)) / (4.0 * math.pi),
    )


def _unproject(fx: np.ndarray, fy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The inverse, needed to ask "what is at this pixel" when warping the hillshade."""
    return (
        fx * 360.0 - 180.0,
        np.degrees(np.arctan(np.sinh(math.pi * (1.0 - 2.0 * fy)))),
    )


@dataclass(frozen=True)
class MapView:
    """A canvas with a place on the Earth: zoom, and where its top-left corner sits.

    `origin_x/y` are world pixels at `zoom`, so a lon/lat becomes a canvas pixel by one
    scale and one subtraction. Everything drawn goes through `to_px`, which is what keeps
    the contours, the catchment and the tiles in register.
    """

    zoom: float
    """Fractional on purpose. Tiles exist only at integer zooms, but framing does not
    have to be quantised to them: an integer-zoom fit leaves up to half the canvas empty
    whenever the extent falls just past a power of two, and on a 4:3 image that is a
    catchment drawn at half the size it could have been. `tile_zoom` is the integer the
    basemap is actually fetched at; the composite is scaled from there to here."""

    origin_x: float
    origin_y: float
    width: int
    height: int
    tile_size: int
    max_zoom: int

    @property
    def tile_zoom(self) -> int:
        """The integer zoom to fetch. Rounded *up*, so the basemap is downsampled into
        the canvas rather than stretched: sharper, and never invents detail."""
        return int(min(self.max_zoom, math.ceil(self.zoom - 1e-9)))

    @property
    def world_px(self) -> float:
        return float(self.tile_size) * (2.0 ** self.zoom)

    def to_px(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        fx, fy = _project(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
        scale = self.world_px
        return fx * scale - self.origin_x, fy * scale - self.origin_y

    @property
    def centre_lonlat(self) -> tuple[float, float]:
        scale = self.world_px
        lon, lat = _unproject(
            np.asarray((self.origin_x + self.width / 2) / scale),
            np.asarray((self.origin_y + self.height / 2) / scale),
        )
        return float(lon), float(lat)

    @property
    def metres_per_px(self) -> float:
        """Ground distance one pixel covers, at the centre latitude.

        Mercator's scale factor is 1/cos(lat), so this is a function of where on the
        canvas you measure. At the centre of a 3 km sheet the variation across the image
        is under 0.05%, which is well inside the width of the scale bar's own line."""
        _, lat = self.centre_lonlat
        return 2 * math.pi * 6378137.0 * math.cos(math.radians(lat)) / self.world_px


def fit_view(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    cfg: RenderConfig,
) -> MapView:
    """The scale that fits `bbox` in the canvas exactly, with padding, centred.

    Fractional, so the drawn extent fills the frame whichever side is binding. Capped at
    `max_zoom`, which only bites on an extent small enough that no imagery has the detail
    for it anyway.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    fx0, fy0 = _project(np.asarray(min_lon), np.asarray(max_lat))  # north-west
    fx1, fy1 = _project(np.asarray(max_lon), np.asarray(min_lat))  # south-east
    span_x = max(float(fx1 - fx0), 1e-12)
    span_y = max(float(fy1 - fy0), 1e-12)

    usable_w = max(width - 2 * cfg.padding_px, 32)
    usable_h = max(height - 2 * cfg.padding_px, 32)

    # World pixels the extent may occupy, then the zoom that produces exactly that.
    world = min(usable_w / span_x, usable_h / span_y)
    zoom = min(float(cfg.max_zoom), math.log2(max(world, 1.0) / cfg.tile_size_px))
    zoom = max(zoom, 0.0)

    scale = float(cfg.tile_size_px) * (2.0 ** zoom)
    centre_x = (float(fx0) + float(fx1)) / 2 * scale
    centre_y = (float(fy0) + float(fy1)) / 2 * scale
    return MapView(
        zoom=zoom,
        origin_x=centre_x - width / 2,
        origin_y=centre_y - height / 2,
        width=width,
        height=height,
        tile_size=cfg.tile_size_px,
        max_zoom=cfg.max_zoom,
    )


# --------------------------------------------------------------------------- #
# Basemap
# --------------------------------------------------------------------------- #
_TILE_CACHE: "OrderedDict[tuple[str, int, int, int], bytes | None]" = OrderedDict()
_TILE_LOCK = threading.Lock()
"""Shared across requests and across the threadpool's workers, so a re-render of the same
sheet pays for the fetch once. `None` is cached too: a tile that 404s at this zoom will
404 on every retry, and re-asking is how a service gets itself blocked."""


def _cached_tile(key, fetch) -> bytes | None:
    with _TILE_LOCK:
        if key in _TILE_CACHE:
            _TILE_CACHE.move_to_end(key)
            return _TILE_CACHE[key]
    data = fetch()
    with _TILE_LOCK:
        _TILE_CACHE[key] = data
        _TILE_CACHE.move_to_end(key)
        while len(_TILE_CACHE) > settings.render.tile_cache_size:
            _TILE_CACHE.popitem(last=False)
    return data


def _tile_bytes(url: str, key, cfg: RenderConfig) -> bytes | None:
    def fetch() -> bytes | None:
        request = urllib.request.Request(url, headers={"User-Agent": cfg.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=cfg.tile_timeout_s) as response:
                return response.read()
        except (urllib.error.URLError, OSError, ValueError):
            # Every failure is the same failure here: this tile is a hole. Which of a
            # timeout, a 429 and a DNS miss it was belongs in the warning the caller
            # writes once, not in a per-tile exception that would abort the render.
            return None

    return _cached_tile(key, fetch)


def _tiled_basemap(
    view: MapView, layer: str, cfg: RenderConfig
) -> tuple[Image.Image, int, int]:
    """The canvas filled with map tiles, plus (fetched, attempted) so the caller can
    decide whether what came back is a basemap or a rumour of one."""
    template = cfg.satellite_url if layer == "satellite" else cfg.street_url
    size = view.tile_size
    zoom = view.tile_zoom

    # The canvas rectangle expressed in the tile zoom's own pixels. `k` is 1 when the fit
    # landed on an integer zoom and at most 2 otherwise, so the oversized composite is
    # never more than four times the target's pixels.
    k = (2.0 ** zoom) * size / view.world_px
    left = view.origin_x * k
    top = view.origin_y * k
    big_w = max(1, int(math.ceil(view.width * k)))
    big_h = max(1, int(math.ceil(view.height * k)))

    x0 = math.floor(left / size)
    y0 = math.floor(top / size)
    x1 = math.floor((left + big_w) / size)
    y1 = math.floor((top + big_h) / size)
    span = 1 << zoom

    wanted = []
    for ty in range(y0, y1 + 1):
        if not 0 <= ty < span:
            continue  # Above the north pole or below the south one. No tile exists.
        for tx in range(x0, x1 + 1):
            wanted.append((tx % span, ty, tx))

    if len(wanted) > cfg.max_tiles:
        raise RenderError(
            "render_too_large",
            f"This view needs {len(wanted)} map tiles and the limit is {cfg.max_tiles}.",
            "Ask for a smaller width and height, or basemap=hillshade.",
        )

    canvas = Image.new("RGB", (big_w, big_h), (26, 30, 34))
    fetched = 0
    with ThreadPoolExecutor(max_workers=cfg.tile_workers) as pool:
        jobs = {
            pool.submit(
                _tile_bytes,
                template.format(z=zoom, x=tx, y=ty),
                (layer, zoom, tx, ty),
                cfg,
            ): (tx, ty, raw_x)
            for tx, ty, raw_x in wanted
        }
        for job, (tx, ty, raw_x) in jobs.items():
            data = job.result()
            if data is None:
                continue
            try:
                tile = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                continue
            # `raw_x` rather than the wrapped `tx`: the wrap is for *which* tile to ask
            # for, the un-wrapped column is where it goes on a canvas that may straddle
            # the antimeridian.
            canvas.paste(
                tile,
                (int(round(raw_x * size - left)), int(round(ty * size - top))),
            )
            fetched += 1

    if (big_w, big_h) != (view.width, view.height):
        canvas = canvas.resize((view.width, view.height), Image.LANCZOS)
    return canvas, fetched, len(wanted)


def _hillshade_basemap(view: MapView, dem: DEM) -> Image.Image:
    """The DEM the analysis ran on, lit from the north-west and warped to the canvas.

    Sampled per output pixel rather than warped as an image because the analysis frame is
    a local equirectangular one and the canvas is Mercator: the mapping is exact, cheap
    at a million points, and needs no resampling library.
    """
    z = dem.z
    res = dem.resolution_m
    dy, dx = np.gradient(np.nan_to_num(z, nan=float(np.nanmin(z))), res)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = math.radians(315.0), math.radians(45.0)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0.0, 1.0)

    # Elevation supplies the hue, the shading supplies the form. Terrain alone is flat to
    # look at; shading alone loses which end of the valley is the top.
    low, high = float(np.nanmin(z)), float(np.nanmax(z))
    height = (z - low) / (high - low) if high > low else np.zeros_like(z)

    x0, y0 = dem.origin_xy
    ny, nx = z.shape
    cols = np.arange(view.width) + 0.5
    world = view.world_px
    out = np.empty((view.height, view.width, 3), dtype=np.uint8)

    # Row-blocked so the warp never holds the whole canvas as float64. At the maximum
    # image size the unblocked version wants roughly 200 MB of temporaries, which is
    # most of the headroom the analysis leaves on a 512 MB host.
    for start in range(0, view.height, _HILLSHADE_ROWS):
        stop = min(start + _HILLSHADE_ROWS, view.height)
        rows = np.arange(start, stop) + 0.5
        lon, lat = _unproject(
            (view.origin_x + cols[None, :]) / world,
            (view.origin_y + rows[:, None]) / world,
        )
        block = (stop - start, view.width)
        x, y = dem.projection.forward_xy(
            np.broadcast_to(lon, block), np.broadcast_to(lat, block)
        )
        col = np.rint((x - x0) / res).astype(np.int64)
        row = np.rint((y - y0) / res).astype(np.int64)
        inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
        np.clip(col, 0, nx - 1, out=col)
        np.clip(row, 0, ny - 1, out=row)

        valid = inside & ~dem.nodata[row, col]
        shaded = shade[row, col]
        warm = height[row, col]

        # A dark green-brown at the valley floor to a pale tan on the ridges, multiplied
        # by the shading. Warm on purpose, for the same reason the contour ramp is: the
        # answer drawn over it is blue and red.
        rgb = np.empty((*block, 3), dtype=np.float64)
        rgb[..., 0] = 0.28 + 0.52 * warm
        rgb[..., 1] = 0.30 + 0.44 * warm
        rgb[..., 2] = 0.22 + 0.34 * warm
        rgb *= (0.35 + 0.75 * shaded)[..., None]
        rgb[~valid] = 0.12
        out[start:stop] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    return Image.fromarray(out, mode="RGB")


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)
_BOLD_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """A scalable font, whatever the host has.

    Pillow ships a usable fallback face, so this never raises and the render never
    depends on a font package being installed in the container.
    """
    for path in (_BOLD_CANDIDATES if bold else _FONT_CANDIDATES):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _rgba(hex_colour: str, alpha: float) -> tuple[int, int, int, int]:
    value = hex_colour.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
        int(round(max(0.0, min(1.0, alpha)) * 255)),
    )


# --------------------------------------------------------------------------- #
# Vector overlay
# --------------------------------------------------------------------------- #
def _rings(geometry: dict) -> list[list]:
    """Every ring of a Polygon or MultiPolygon, exteriors and holes alike, flat."""
    kind = geometry.get("type")
    if kind == "Polygon":
        return list(geometry["coordinates"])
    if kind == "MultiPolygon":
        return [ring for polygon in geometry["coordinates"] for ring in polygon]
    return []


def _to_screen(view: MapView, coordinates, scale: int) -> list[tuple[float, float]]:
    array = np.asarray(coordinates, dtype=np.float64)
    px, py = view.to_px(array[:, 0], array[:, 1])
    return list(zip((px * scale).tolist(), (py * scale).tolist()))


@dataclass
class _Overlay:
    """The transparent sheet everything vector is drawn on, at supersampled size."""

    image: Image.Image
    draw: ImageDraw.ImageDraw
    view: MapView
    scale: int
    scratch: Image.Image = field(init=False)

    def __post_init__(self) -> None:
        # One reusable layer for filled polygons. Holes have to be punched in *replace*
        # mode, which cannot be done on a sheet that already has other features on it,
        # and allocating one of these per feature would be six full-canvas RGBA buffers.
        self.scratch = Image.new("RGBA", self.image.size, (0, 0, 0, 0))

    def polygon(self, geometry: dict, fill: str, opacity: float, stroke: str, width: float) -> None:
        rings = _rings(geometry)
        if not rings:
            return
        self.scratch.paste((0, 0, 0, 0), (0, 0, *self.scratch.size))
        scratch_draw = ImageDraw.Draw(self.scratch)
        solid = _rgba(fill, 1.0)
        for index, ring in enumerate(rings):
            points = _to_screen(self.view, ring, self.scale)
            if len(points) < 3:
                continue
            # Ring 0 is the exterior; the rest are holes, cleared rather than painted so
            # a catchment with a nodata island shows the ground through it.
            scratch_draw.polygon(points, fill=solid if index == 0 else (0, 0, 0, 0))
        # The opacity is applied to the alpha channel alone. Blending the whole image
        # toward transparent black would darken the fill as well as thin it, which on a
        # 25%-opacity catchment over bright imagery is the difference between a tint and
        # a smudge.
        self.scratch.putalpha(
            self.scratch.getchannel("A").point(lambda v: int(v * opacity))
        )
        self.image.alpha_composite(self.scratch)
        for ring in rings:
            points = _to_screen(self.view, ring, self.scale)
            if len(points) < 2:
                continue
            self.draw.line(
                points + [points[0]],
                fill=_rgba(stroke, 1.0),
                width=max(1, int(round(width * self.scale))),
                joint="curve",
            )

    def line(self, geometry: dict, stroke: str, width: float, opacity: float = 1.0) -> None:
        if geometry.get("type") != "LineString":
            return
        points = _to_screen(self.view, geometry["coordinates"], self.scale)
        if len(points) < 2:
            return
        self.draw.line(
            points,
            fill=_rgba(stroke, opacity),
            width=max(1, int(round(width * self.scale))),
            joint="curve",
        )


def _visible(view: MapView, geometry: dict, margin: float = 64.0) -> bool:
    """Whether anything of this feature lands on the canvas.

    Worth the check for contours: a sheet framed on one catchment can have most of its
    1,355 lines entirely off-screen, and Pillow will happily rasterise every one of them.
    """
    coordinates = geometry.get("coordinates")
    if not coordinates:
        return False
    flat: list = []

    def walk(node) -> None:
        if isinstance(node[0], (int, float)):
            flat.append(node)
            return
        for part in node:
            walk(part)

    walk(coordinates)
    array = np.asarray(flat, dtype=np.float64)
    px, py = view.to_px(array[:, 0], array[:, 1])
    return bool(
        (px.max() >= -margin)
        and (px.min() <= view.width + margin)
        and (py.max() >= -margin)
        and (py.min() <= view.height + margin)
    )


# --------------------------------------------------------------------------- #
# Furniture
# --------------------------------------------------------------------------- #
_SCALE_STEPS = (10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000)


def _draw_scale_bar(draw: ImageDraw.ImageDraw, view: MapView, scale: int) -> None:
    """A bar of a round ground distance. Without one, "66.3 ha" is a number the reader
    has no way to sanity-check against the picture."""
    mpp = view.metres_per_px
    target_px = view.width * 0.18
    metres = next(
        (s for s in _SCALE_STEPS if s / mpp >= target_px * 0.6), _SCALE_STEPS[-1]
    )
    length = int(round(metres / mpp)) * scale
    x = 16 * scale
    y = view.height * scale - 22 * scale
    label = f"{metres} m" if metres < 1000 else f"{metres / 1000:g} km"
    font = _font(11 * scale, bold=True)

    draw.rectangle([x - 6 * scale, y - 17 * scale, x + length + 8 * scale, y + 7 * scale],
                   fill=(0, 0, 0, 120))
    for offset in (0, 1):
        # Drawn twice, black then white, so the bar reads on bright imagery and on dark
        # water alike without needing to know which is underneath it.
        colour = (0, 0, 0, 190) if offset == 0 else (255, 255, 255, 235)
        w = 4 * scale if offset == 0 else 2 * scale
        draw.line([(x, y), (x + length, y)], fill=colour, width=w)
        draw.line([(x, y - 5 * scale), (x, y + 3 * scale)], fill=colour, width=w)
        draw.line([(x + length, y - 5 * scale), (x + length, y + 3 * scale)], fill=colour, width=w)
    draw.text((x, y - 15 * scale), label, font=font, fill=(255, 255, 255, 245))


def _draw_credit(draw: ImageDraw.ImageDraw, view: MapView, scale: int, credit: str) -> None:
    font = _font(10 * scale)
    right = view.width * scale - 8 * scale
    bottom = view.height * scale - 8 * scale
    box = draw.textbbox((0, 0), credit, font=font, anchor="ls")
    draw.rectangle(
        [right - (box[2] - box[0]) - 6 * scale, bottom - (box[3] - box[1]) - 5 * scale,
         right + 3 * scale, bottom + 4 * scale],
        fill=(0, 0, 0, 120),
    )
    draw.text((right, bottom), credit, font=font, fill=(255, 255, 255, 220), anchor="rs")


def _marker(draw: ImageDraw.ImageDraw, x: float, y: float, rank: int, colour: str, scale: int) -> None:
    radius = 11 * scale
    draw.ellipse([x - radius - 2 * scale, y - radius - 2 * scale,
                  x + radius + 2 * scale, y + radius + 2 * scale], fill=(255, 255, 255, 235))
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=_rgba(colour, 1.0))
    draw.text((x, y), str(rank), font=_font(12 * scale, bold=True),
              fill=(255, 255, 255, 255), anchor="mm")


def _pill(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, scale: int,
          fill: tuple[int, int, int, int] = (31, 120, 180, 235)) -> None:
    font = _font(12 * scale, bold=True)
    box = draw.textbbox((x, y), text, font=font, anchor="mm")
    pad_x, pad_y = 8 * scale, 5 * scale
    draw.rounded_rectangle(
        [box[0] - pad_x, box[1] - pad_y, box[2] + pad_x, box[3] + pad_y],
        radius=9 * scale, fill=fill,
    )
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255), anchor="mm")


def _draw_legend(draw: ImageDraw.ImageDraw, scale: int, rows: list[tuple[str, str]], title: str) -> None:
    """The recommended site's headline numbers, so the image answers on its own."""
    title_font = _font(13 * scale, bold=True)
    key_font = _font(11 * scale)
    value_font = _font(11 * scale, bold=True)

    pad = 12 * scale
    line_h = 16 * scale
    width = max(
        draw.textlength(title, font=title_font),
        max(
            (draw.textlength(k, font=key_font) + 14 * scale
             + draw.textlength(v, font=value_font) for k, v in rows),
            default=0,
        ),
    ) + 2 * pad
    height = 2 * pad + 20 * scale + len(rows) * line_h

    x, y = 14 * scale, 14 * scale
    draw.rounded_rectangle([x, y, x + width, y + height], radius=8 * scale,
                           fill=(16, 20, 26, 205))
    draw.text((x + pad, y + pad), title, font=title_font, fill=(255, 255, 255, 250))
    cursor = y + pad + 22 * scale
    for key, value in rows:
        draw.text((x + pad, cursor), key, font=key_font, fill=(186, 196, 208, 240))
        draw.text((x + width - pad, cursor), value, font=value_font,
                  fill=(255, 255, 255, 250), anchor="ra")
        cursor += line_h


def _label_anchor(view: MapView, geometry: dict, scale: int) -> tuple[float, float] | None:
    """Somewhere inside the polygon and on the canvas, for the area pill.

    The centroid of the exterior ring, clamped to the visible area. A true
    pole-of-inaccessibility would place it better on a horseshoe-shaped catchment, and is
    a great deal of code for a label nudged by a few tens of pixels.
    """
    rings = _rings(geometry)
    if not rings:
        return None
    points = _to_screen(view, rings[0], scale)
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (
        min(max(sum(xs) / len(xs), 60 * scale), (view.width - 60) * scale),
        min(max(sum(ys) / len(ys), 40 * scale), (view.height - 40) * scale),
    )


# --------------------------------------------------------------------------- #
# The render
# --------------------------------------------------------------------------- #
def _number(value: float, unit: str = "") -> str:
    text = f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{text} {unit}".strip()


def render_png(
    *,
    analysis: dict,
    dem: DEM,
    contours: dict | None = None,
    legend_rows: list[tuple[str, str]] | None = None,
    legend_title: str = "Recommended site",
    width: int | None = None,
    height: int | None = None,
    basemap: str | None = None,
    frame_bbox: tuple[float, float, float, float] | None = None,
    cfg: RenderConfig | None = None,
) -> tuple[bytes, list[str]]:
    """The map as PNG bytes, and whatever had to be said about producing it.

    `analysis` is the FeatureCollection `/analyzeContour` returns and `contours` the one
    `/contours` returns; both are drawn exactly as their simplestyle properties ask, so
    this picture and the one a client draws from the same response are the same picture.
    """
    cfg = cfg or settings.render
    width = int(width or cfg.default_width)
    height = int(height or cfg.default_height)
    basemap = (basemap or cfg.default_basemap).lower()
    warnings: list[str] = []

    if not cfg.min_size_px <= width <= cfg.max_size_px or not cfg.min_size_px <= height <= cfg.max_size_px:
        raise RenderError(
            "invalid_image_size",
            f"width and height must each be between {cfg.min_size_px} and "
            f"{cfg.max_size_px} pixels; got {width}x{height}.",
            "Leave them out for the default "
            f"{cfg.default_width}x{cfg.default_height}.",
        )
    if basemap not in cfg.basemaps:
        raise RenderError(
            "invalid_basemap",
            f"basemap is {basemap!r}; it has to be one of {', '.join(cfg.basemaps)}.",
            "satellite is the default. hillshade needs no network and always works.",
        )

    features = list(analysis.get("features") or [])
    if not features:
        raise RenderError(
            "nothing_to_draw",
            "The analysis produced no geometry to draw.",
            "This is a bug if /analyzeContour returned a site; report the file.",
        )

    bbox = frame_bbox or analysis.get("bbox")
    if bbox is None:
        raise RenderError("nothing_to_draw", "The analysis carries no bounding box.")
    view = fit_view(tuple(bbox), width, height, cfg=cfg)

    # ---- basemap ---- #
    credit = cfg.hillshade_credit
    if basemap in ("satellite", "street"):
        canvas, fetched, attempted = _tiled_basemap(view, basemap, cfg)
        if attempted and fetched / attempted < (1.0 - cfg.tile_failure_ratio):
            warnings.append(
                f"Only {fetched} of {attempted} {basemap} tiles could be fetched, so the "
                "hillshade of the uploaded sheet was drawn instead. The overlay is "
                "unaffected."
            )
            canvas = _hillshade_basemap(view, dem)
        else:
            credit = cfg.satellite_credit if basemap == "satellite" else cfg.street_credit
            if fetched < attempted:
                warnings.append(
                    f"{attempted - fetched} of {attempted} basemap tiles were missing and "
                    "are drawn as blank ground."
                )
    elif basemap == "hillshade":
        canvas = _hillshade_basemap(view, dem)
    else:
        canvas = Image.new("RGB", (width, height), (18, 22, 27))
        credit = ""

    # ---- overlay ---- #
    scale = max(1, int(cfg.supersample))
    sheet = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
    overlay = _Overlay(image=sheet, draw=ImageDraw.Draw(sheet, "RGBA"), view=view, scale=scale)

    if contours:
        # How close together the lines land on *this* image decides how many of them are
        # worth drawing. See `RenderConfig.contour_min_spacing_px`.
        spacing_px = dem.meta.mean_contour_spacing_m / view.metres_per_px
        every = settings.geojson.contour_index_every
        index_only = spacing_px < cfg.contour_min_spacing_px
        drop_all = spacing_px * every < cfg.contour_min_spacing_px
        if drop_all:
            warnings.append(
                f"The contour lines fall {spacing_px:.1f} pixels apart at this size and "
                "were left off the image, where they would have been a wash rather than "
                "lines. Ask for a larger width, or frame=sites."
            )
        elif index_only:
            warnings.append(
                f"The contour lines fall {spacing_px:.1f} pixels apart at this size, so "
                f"only every {every}th is drawn. The interval on the image is "
                f"{dem.meta.contour_interval_m * every:g} m, not "
                f"{dem.meta.contour_interval_m:g} m."
            )
        if not drop_all:
            for feature in contours.get("features") or []:
                props = feature.get("properties", {})
                if index_only and not props.get("index"):
                    continue
                geometry = feature["geometry"]
                if not _visible(view, geometry):
                    continue
                overlay.line(
                    geometry,
                    props.get("stroke", "#e08b1e"),
                    float(props.get("stroke-width", 0.8)),
                    float(props.get("stroke-opacity", 0.7)),
                )

    # Draw order follows the collection's own, which `build_geojson` already sorts into
    # areas, then lines, then points, for exactly this reason.
    points: list[dict] = []
    label_target: dict | None = None
    for feature in features:
        geometry = feature["geometry"]
        props = feature.get("properties", {})
        role = props.get("role")
        if geometry["type"] == "Point":
            points.append(feature)
            continue
        if not _visible(view, geometry):
            continue
        if geometry["type"] == "LineString":
            overlay.line(geometry, props.get("stroke", "#e31a1c"),
                         float(props.get("stroke-width", 2)))
            continue
        overlay.polygon(
            geometry,
            props.get("fill", "#a6cee3"),
            float(props.get("fill-opacity", 0.25)),
            props.get("stroke", "#1f78b4"),
            3.0 if role == "catchment" else 2.0,
        )
        if role == "catchment" and props.get("rank") == 1:
            label_target = feature

    if label_target is not None:
        anchor = _label_anchor(view, label_target["geometry"], scale)
        if anchor is not None:
            area = label_target["properties"].get("area_ha")
            uncertainty = label_target["properties"].get("area_uncertainty_ha")
            # One decimal, matching how every other surface reports an area. The GeoJSON
            # keeps two because it is data; a label on a picture is not read to that
            # precision and 66.34 only makes the pill wider.
            text = (
                f"{area:,.1f} ha"
                if uncertainty is None
                else f"{area:,.1f} +/- {uncertainty:,.1f} ha"
            )
            _pill(overlay.draw, anchor[0], anchor[1], text, scale)

    for feature in points:
        px, py = view.to_px(*feature["geometry"]["coordinates"])
        props = feature.get("properties", {})
        _marker(overlay.draw, float(px) * scale, float(py) * scale,
                int(props.get("rank", 1)), props.get("marker-color", "#08519c"), scale)

    if legend_rows:
        _draw_legend(overlay.draw, scale, legend_rows, legend_title)
    _draw_scale_bar(overlay.draw, view, scale)
    if credit:
        _draw_credit(overlay.draw, view, scale, credit)

    if scale > 1:
        sheet = sheet.resize((width, height), Image.LANCZOS)
    canvas = canvas.convert("RGBA")
    canvas.alpha_composite(sheet)

    buffer = io.BytesIO()
    canvas.convert("RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), warnings
