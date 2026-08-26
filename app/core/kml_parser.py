"""KML/KMZ contour parsing.

This is the generalisation surface of the service: everything downstream works on
`ContourSet`, so supporting a new flavour of contour file means changing only this
module.

The hard problem is not the geometry -- it is finding the *elevation*. KML has no
standard place to put a contour's height, so producers put it wherever they like. The
parser therefore resolves elevation by a **cascade** (PLAN Phase 1): four strategies are
tried in order and the first one that explains almost every contour wins.

    1. z_coordinate   -- the third ordinate of a 3D <coordinates> tuple
    2. extended_data  -- <ExtendedData><SimpleData name="elev">270</SimpleData>
    3. placemark_name -- <Placemark><name>270.0</name>          <- the provided sample
    4. folder_name    -- an enclosing <Folder><name>270 m</name>

A strategy is accepted only when it resolves an elevation for at least
`strategy_min_coverage` of the geometries *and* produces at least
`min_elevation_levels` distinct values. Both guards matter on the sample sheet: it
contains one stray `land` boundary polygon whose coordinates are 3D at z=30, which
would otherwise let strategy 1 win with a single absurd elevation.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from typing import Iterator, Sequence
from xml.etree import ElementTree as ET

import numpy as np

from app.config import ParserConfig, settings

__all__ = [
    "ContourParseError",
    "ContourMetadata",
    "ContourSet",
    "parse_contours",
    "parse_contour_file",
]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ContourParseError(Exception):
    """A contour file that cannot be turned into a usable surface.

    Carries the structured `(code, detail, hint)` triple the API returns (PLAN Phase 9),
    so the route layer maps the code to a status without re-deriving the diagnosis.
    """

    def __init__(self, code: str, detail: str, hint: str = "") -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.hint = hint


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContourMetadata:
    """What the parser learned about the file, reported back in the API response."""

    elevation_source: str
    """Which cascade strategy won: z_coordinate | extended_data | placemark_name |
    folder_name."""

    interval_m: float | None
    """Median spacing between distinct contour levels. `None` if fewer than two."""

    levels: tuple[float, ...]
    """Every distinct contour elevation, ascending."""

    elevation_range: tuple[float, float]
    bbox: tuple[float, float, float, float]
    """(min_lon, min_lat, max_lon, max_lat) in degrees."""

    line_count: int
    vertex_count: int

    unit_hint: str | None = None
    """A unit token seen alongside the elevations ("m", "ft", ...), if any. Reported,
    never acted on -- the parser does not convert."""

    skipped_features: int = 0
    """Placemarks carrying line geometry that no strategy could give an elevation to.
    On the sample this is exactly 1: the `land` map-boundary polygon."""

    label_placemark_count: int = 0
    """Point placemarks in ignored folders. The sample's `labels` folder duplicates all
    1,355 contour elevations as points; they add no surface information (PLAN §11.1)."""

    labels_consistent: bool | None = None
    """Whether those label elevations are a subset of the contour levels -- a free
    cross-check that the cascade picked the right field."""

    document_name: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def level_count(self) -> int:
        return len(self.levels)


@dataclass(frozen=True)
class ContourSet:
    """Contour vertices as a flat point cloud, plus the line structure that made it.

    Every vertex of every contour line is a known `(lon, lat, z)`: a contour is a line of
    constant height, so each of its points has that height (PLAN §2 Step 1). The DEM
    builder wants the flat cloud; the resolution heuristic needs total contour *length*,
    which needs the per-line grouping -- hence `line_starts`, CSR-style offsets into
    `points`.
    """

    points: np.ndarray
    """(N, 2) float64 -- lon, lat in degrees."""

    elevations: np.ndarray
    """(N,) float64 -- the parent contour's elevation, repeated per vertex."""

    line_starts: np.ndarray
    """(L + 1,) int64 -- line `i` occupies points[line_starts[i]:line_starts[i + 1]]."""

    metadata: ContourMetadata

    @property
    def line_count(self) -> int:
        return len(self.line_starts) - 1

    @property
    def vertex_count(self) -> int:
        return len(self.points)

    @property
    def line_elevations(self) -> np.ndarray:
        """(L,) float64 -- one elevation per contour line."""
        return self.elevations[self.line_starts[:-1]]

    def line_coords(self, index: int) -> np.ndarray:
        """(n, 2) view of the vertices of contour line `index`."""
        return self.points[self.line_starts[index] : self.line_starts[index + 1]]

    def iter_lines(self) -> Iterator[np.ndarray]:
        for i in range(self.line_count):
            yield self.line_coords(i)


# --------------------------------------------------------------------------- #
# XML helpers -- namespace-agnostic throughout
# --------------------------------------------------------------------------- #
def _localname(tag: object) -> str:
    """`{http://www.opengis.net/kml/2.2}Placemark` -> `Placemark`.

    KML in the wild is written against 2.0, 2.1, 2.2, the Google extension namespace, or
    no namespace at all. Matching on the local name makes all of them equivalent.
    """
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _child_text(elem: ET.Element, name: str) -> str | None:
    """Text of the first *direct* child with this local name.

    Direct, not descendant: a Placemark's own <name> must not be confused with the
    <name> of a Style or a nested feature.
    """
    for child in elem:
        if _localname(child.tag) == name:
            return (child.text or "").strip()
    return None


_LINE_TAGS = {"LineString", "LinearRing"}
_CONTAINER_TAGS = {"Folder", "Document"}


# --------------------------------------------------------------------------- #
# Coordinate parsing
# --------------------------------------------------------------------------- #
_COMMA_WS = re.compile(r"\s*,\s*")


def _parse_coordinates(text: str) -> np.ndarray:
    """A KML <coordinates> body -> (n, 3) float64, z = NaN where absent.

    The body is whitespace-separated `lon,lat[,alt]` tuples, but producers sprinkle
    spaces after the commas, so those are normalised away first. When every tuple has
    the same arity -- the overwhelmingly common case -- the whole block is converted in
    one numpy call; mixed arity falls back to a per-tuple loop.
    """
    text = _COMMA_WS.sub(",", text.strip())
    if not text:
        return np.empty((0, 3), dtype=np.float64)

    tokens = text.split()
    arities = {t.count(",") for t in tokens}

    if len(arities) == 1:
        ncols = arities.pop() + 1
        if ncols not in (2, 3):
            raise ContourParseError(
                "unparseable_geometry",
                f"Coordinate tuples have {ncols} ordinates; expected 2 or 3.",
                "Each tuple must be lon,lat or lon,lat,altitude.",
            )
        try:
            flat = np.fromstring(text.replace(",", " "), dtype=np.float64, sep=" ")
        except ValueError as exc:  # pragma: no cover - numpy is lenient here
            raise ContourParseError(
                "unparseable_geometry", f"Non-numeric coordinate: {exc}", ""
            ) from exc
        if flat.size != len(tokens) * ncols:
            return _parse_coordinates_slow(tokens)
        coords = flat.reshape(-1, ncols)
        if ncols == 2:
            coords = np.column_stack([coords, np.full(len(coords), np.nan)])
        return coords

    return _parse_coordinates_slow(tokens)


def _parse_coordinates_slow(tokens: Sequence[str]) -> np.ndarray:
    out = np.empty((len(tokens), 3), dtype=np.float64)
    out[:] = np.nan
    for i, token in enumerate(tokens):
        parts = token.split(",")
        if len(parts) < 2:
            raise ContourParseError(
                "unparseable_geometry",
                f"Coordinate tuple {token!r} has fewer than two ordinates.",
                "Each tuple must be lon,lat or lon,lat,altitude.",
            )
        try:
            out[i, 0] = float(parts[0])
            out[i, 1] = float(parts[1])
            if len(parts) > 2 and parts[2].strip():
                out[i, 2] = float(parts[2])
        except ValueError as exc:
            raise ContourParseError(
                "unparseable_geometry", f"Non-numeric coordinate {token!r}.", ""
            ) from exc
    return out


# --------------------------------------------------------------------------- #
# Number parsing
# --------------------------------------------------------------------------- #
_NUMBER_RE = re.compile(
    r"""^(?:[^\d+\-]*[\s:=])?                    # optional label prefix ("Elevation: ")
        ([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)   # the number itself
        \s*(m|meter|meters|metre|metres|ft|foot|feet|')?  # optional unit
        \s*$""",
    re.VERBOSE | re.IGNORECASE,
)

_UNIT_CANON = {
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "ft": "ft", "foot": "ft", "feet": "ft", "'": "ft",
}


def _parse_number(text: str | None) -> tuple[float, str | None] | None:
    """Read a label that *is* a number, optionally with a unit. Anchored on purpose.

    Anchoring is what lets `land` and `sources` -- real placemark names in the sample --
    fail cleanly, while `277.0`, `270 m` and `Elevation: 270 m` all succeed. A loose
    search would happily return 1 for `contours_1.0m`.

    A label prefix is allowed only when it ends at a genuine separator (whitespace, `:`
    or `=`). Without that restriction the leading `-` of a name like `contour-12` is read
    as a minus sign and the contour lands 12 m below sea level.
    """
    if text is None:
        return None
    match = _NUMBER_RE.match(text.strip())
    if match is None:
        return None
    unit = match.group(2)
    return float(match.group(1)), _UNIT_CANON.get(unit.lower()) if unit else None


# --------------------------------------------------------------------------- #
# Intermediate representation
# --------------------------------------------------------------------------- #
@dataclass
class _Feature:
    """One Placemark's worth of line geometry, with everything the cascade might use."""

    lines: list[np.ndarray] = field(default_factory=list)
    name: str | None = None
    extended: dict[str, str] = field(default_factory=dict)
    folders: tuple[str, ...] = ()
    ignored: bool = False
    """Inside a folder matching `ignore_folder_pattern` -- carried for the label check."""

    point_names: list[str] = field(default_factory=list)
    """Names of Point geometries. Points never contribute surface information."""


def _read_placemark(elem: ET.Element, folders: tuple[str, ...], ignored: bool) -> _Feature:
    feature = _Feature(name=_child_text(elem, "name"), folders=folders, ignored=ignored)

    for node in elem.iter():
        tag = _localname(node.tag)
        if tag in _LINE_TAGS:
            # Reaching LinearRing by descent covers Polygon outer/inner rings and
            # MultiGeometry without special-casing either.
            text = _child_text(node, "coordinates")
            if text:
                coords = _parse_coordinates(text)
                if len(coords) >= 2:
                    feature.lines.append(coords)
        elif tag == "Point":
            if feature.name:
                feature.point_names.append(feature.name)
        elif tag in ("SimpleData", "Data"):
            key = node.get("name")
            if not key:
                continue
            value = (node.text or "").strip()
            if tag == "Data" and not value:
                value = _child_text(node, "value") or ""
            if value:
                feature.extended[key] = value

    return feature


def _walk(
    elem: ET.Element,
    folders: tuple[str, ...],
    ignored: bool,
    out: list[_Feature],
    ignore_re: re.Pattern[str],
) -> None:
    tag = _localname(elem.tag)

    if tag == "Placemark":
        out.append(_read_placemark(elem, folders, ignored))
        return

    if tag in _CONTAINER_TAGS:
        name = _child_text(elem, "name")
        if name:
            folders = folders + (name,)
            ignored = ignored or bool(ignore_re.search(name))

    for child in elem:
        _walk(child, folders, ignored, out, ignore_re)


# --------------------------------------------------------------------------- #
# The elevation cascade
# --------------------------------------------------------------------------- #
def _elev_from_z(feature: _Feature) -> tuple[float, str | None] | None:
    """Median z of the feature's vertices, if they are 3D and effectively constant.

    A contour is a line of constant height, so a 3D contour whose z varies by more than
    a metre along its length is not a contour -- it is a boundary or a track, and this
    strategy declines it.
    """
    zs = np.concatenate([line[:, 2] for line in feature.lines])
    if not np.isfinite(zs).all():
        return None
    if float(zs.max() - zs.min()) > 1.0:
        return None
    return float(np.median(zs)), None


def _elev_from_extended(
    feature: _Feature, field_re: re.Pattern[str]
) -> tuple[float, str | None] | None:
    for key, value in feature.extended.items():
        if field_re.match(key.strip()):
            parsed = _parse_number(value)
            if parsed is not None:
                return parsed
    return None


def _elev_from_name(feature: _Feature) -> tuple[float, str | None] | None:
    return _parse_number(feature.name)


def _elev_from_folder(feature: _Feature) -> tuple[float, str | None] | None:
    # Nearest enclosing folder first: contours grouped per level sit in the innermost
    # folder, while outer folders name the document or the interval.
    for name in reversed(feature.folders):
        parsed = _parse_number(name)
        if parsed is not None:
            return parsed
    return None


def _resolve_elevations(
    features: Sequence[_Feature], cfg: ParserConfig
) -> tuple[str, list[tuple[float, str | None] | None]]:
    """Run the cascade; return the winning strategy name and its per-feature result."""
    field_re = re.compile(cfg.elevation_field_pattern, re.IGNORECASE)
    resolvers = {
        "z_coordinate": _elev_from_z,
        "extended_data": lambda f: _elev_from_extended(f, field_re),
        "placemark_name": _elev_from_name,
        "folder_name": _elev_from_folder,
    }

    attempts: list[str] = []
    flat_levels: list[tuple[str, set[float]]] = []
    for strategy in cfg.elevation_strategies:
        resolve = resolvers.get(strategy)
        if resolve is None:
            continue

        results = [resolve(f) for f in features]
        resolved = [r for r in results if r is not None]
        coverage = len(resolved) / len(features)
        levels = {round(value, 6) for value, _ in resolved}

        if coverage >= cfg.strategy_min_coverage:
            if len(levels) >= cfg.min_elevation_levels:
                return strategy, results
            # The field was found and read; it just is not a height. Almost always an
            # identifier, so say so rather than reporting a generic failure.
            flat_levels.append((strategy, levels))

        attempts.append(
            f"{strategy}: {coverage:.0%} of contours, {len(levels)} distinct level(s)"
        )

    if flat_levels:
        strategy, levels = flat_levels[0]
        raise ContourParseError(
            "too_few_levels",
            f"Strategy {strategy!r} gave every contour one of {len(levels)} elevation "
            f"value(s); a surface needs at least {cfg.min_elevation_levels}.",
            "The field read as an elevation may be an identifier or a contour interval "
            "rather than a height.",
        )

    raise ContourParseError(
        "no_elevations",
        "No elevation strategy explained the contours. Tried -- " + "; ".join(attempts),
        "Each contour needs a height: 3D coordinates, an <ExtendedData> field named "
        "elev/elevation/level/height, a numeric <Placemark><name>, or a numeric "
        "enclosing <Folder><name>.",
    )


# --------------------------------------------------------------------------- #
# Container handling
# --------------------------------------------------------------------------- #
def _kmz_member(data: bytes) -> bytes:
    """Extract the KML payload from a KMZ archive.

    OGC says the document is the first `.kml` at the root of the archive; producers are
    inconsistent, so prefer `doc.kml`, then the shallowest path, then alphabetical.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ContourParseError(
            "unreadable_file", f"Not a readable KMZ archive: {exc}", "Upload a .kml or .kmz."
        ) from exc

    with archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".kml")]
        if not members:
            raise ContourParseError(
                "unreadable_file",
                "The KMZ archive contains no .kml member.",
                "A KMZ must contain the KML document, conventionally doc.kml.",
            )
        members.sort(key=lambda n: (n.lower() != "doc.kml", n.count("/"), n.lower()))
        return archive.read(members[0])


def _parse_xml(data: bytes) -> ET.Element:
    try:
        return ET.fromstring(data.lstrip())
    except ET.ParseError as exc:
        raise ContourParseError(
            "unparseable_xml",
            f"The file is not well-formed XML: {exc}",
            "Upload a valid KML document or a KMZ archive containing one.",
        ) from exc


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def parse_contours(
    data: bytes,
    filename: str | None = None,
    *,
    config: ParserConfig | None = None,
) -> ContourSet:
    """Parse KML or KMZ bytes into a `ContourSet`.

    Raises `ContourParseError` -- never returns a partially valid result.
    """
    cfg = config or settings.parser

    if len(data) > cfg.max_upload_bytes:
        raise ContourParseError(
            "file_too_large",
            f"{len(data) / 1e6:.1f} MB exceeds the {cfg.max_upload_bytes / 1e6:.0f} MB limit.",
            "Clip the contour sheet to the area of interest and retry.",
        )
    if not data.strip():
        raise ContourParseError("unreadable_file", "The uploaded file is empty.", "")

    is_kmz = data[:4] == b"PK\x03\x04" or (filename or "").lower().endswith(".kmz")
    payload = _kmz_member(data) if is_kmz else data
    root = _parse_xml(payload)

    ignore_re = re.compile(cfg.ignore_folder_pattern, re.IGNORECASE)
    features: list[_Feature] = []
    _walk(root, (), False, features, ignore_re)

    label_names = [n for f in features if f.ignored for n in f.point_names]
    candidates = [f for f in features if f.lines and not f.ignored]

    if not candidates:
        raise ContourParseError(
            "no_contours",
            "No contour line geometry found (looked for LineString, LinearRing, "
            "Polygon rings and MultiGeometry).",
            "The file may contain only points or overlays. Export the contour *lines*.",
        )

    strategy, resolved = _resolve_elevations(candidates, cfg)
    return _assemble(
        candidates, resolved, strategy, label_names, _child_text(root, "name"), cfg
    )


def parse_contour_file(path: str, *, config: ParserConfig | None = None) -> ContourSet:
    """Convenience wrapper: read a path from disk and parse it."""
    with open(path, "rb") as handle:
        return parse_contours(handle.read(), filename=path, config=config)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _assemble(
    features: Sequence[_Feature],
    resolved: Sequence[tuple[float, str | None] | None],
    strategy: str,
    label_names: Sequence[str],
    document_name: str | None,
    cfg: ParserConfig,
) -> ContourSet:
    chunks: list[np.ndarray] = []
    line_elevations: list[float] = []
    starts: list[int] = [0]
    units: set[str] = set()
    skipped = 0
    cursor = 0

    for feature, elevation in zip(features, resolved):
        if elevation is None:
            skipped += 1
            continue
        value, unit = elevation
        if unit:
            units.add(unit)
        for line in feature.lines:
            chunks.append(line[:, :2])
            line_elevations.append(value)
            cursor += len(line)
            starts.append(cursor)

    if len(starts) - 1 < cfg.min_contour_lines:
        raise ContourParseError(
            "no_contours",
            f"Only {len(starts) - 1} contour line(s) carried an elevation; "
            f"at least {cfg.min_contour_lines} are needed to build a surface.",
            "Check that the contour lines have a numeric label.",
        )

    points = np.concatenate(chunks).astype(np.float64, copy=False)
    per_line = np.asarray(line_elevations, dtype=np.float64)
    line_starts = np.asarray(starts, dtype=np.int64)
    elevations = np.repeat(per_line, np.diff(line_starts))

    levels = np.unique(per_line)
    if len(levels) < cfg.min_elevation_levels:
        raise ContourParseError(
            "too_few_levels",
            f"All contours share {len(levels)} elevation level(s); a surface needs at "
            f"least {cfg.min_elevation_levels}.",
            "The elevation field may be an identifier rather than a height.",
        )

    diffs = np.diff(levels)
    interval = float(np.median(diffs))

    warnings: list[str] = []
    unit_hint = sorted(units)[0] if units else None

    # Feet-vs-metres: flag, never convert (PLAN Phase 1). A 5-unit interval is far more
    # likely to be 5 ft than 5 m, but guessing wrong would silently rescale the terrain.
    low, high = cfg.feet_interval_range
    if low <= interval <= high:
        warnings.append(
            f"Contour interval is {interval:g} units. Intervals in the "
            f"{low:g}-{high:g} range are commonly feet, not metres. Elevations are "
            "being used as given, unconverted -- verify the source units."
        )
    if unit_hint == "ft":
        warnings.append(
            "Elevation labels carry a feet unit. Values are used as given, "
            "unconverted -- verify the source units."
        )
    if skipped:
        warnings.append(
            f"{skipped} placemark(s) with line geometry had no resolvable elevation "
            f"and were excluded (strategy: {strategy})."
        )

    labels_consistent: bool | None = None
    label_values = {
        round(parsed[0], 6)
        for parsed in (_parse_number(name) for name in label_names)
        if parsed is not None
    }
    if label_values:
        contour_values = {round(v, 6) for v in levels.tolist()}
        labels_consistent = label_values <= contour_values
        if not labels_consistent:
            warnings.append(
                "Label placemarks carry elevations that are not among the contour "
                "levels; the elevation field may have been misidentified."
            )

    metadata = ContourMetadata(
        elevation_source=strategy,
        interval_m=interval,
        levels=tuple(float(v) for v in levels),
        elevation_range=(float(levels[0]), float(levels[-1])),
        bbox=(
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        ),
        line_count=len(per_line),
        vertex_count=len(points),
        unit_hint=unit_hint,
        skipped_features=skipped,
        label_placemark_count=len(label_names),
        labels_consistent=labels_consistent,
        document_name=document_name,
        warnings=tuple(warnings),
    )

    return ContourSet(
        points=points, elevations=elevations, line_starts=line_starts, metadata=metadata
    )
