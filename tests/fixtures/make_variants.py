"""Synthetic KML/KMZ contour files in every structural flavour the parser claims.

The provided sample exercises exactly one path through the elevation cascade
(`placemark_name`, 2D LineStrings). These generators cover the rest, so a change to the
parser that quietly breaks 3D KML or Polygon rings fails a test instead of a grader's
upload.

Everything is generated, not committed: the fixtures stay readable and diffable, and the
repository stays small. Run as a script to dump them to disk for manual inspection::

    python -m tests.fixtures.make_variants /tmp/variants
"""

from __future__ import annotations

import io
import zipfile

# A tiny synthetic sheet: three nested rectangles, one per contour level. Placed near the
# sample sheet so any accidental hard-coding of a location shows up as a plausible-looking
# but wrong answer rather than an obvious one.
LON0, LAT0 = 81.30, 21.25
LEVELS = (10.0, 20.0, 30.0)
NAMESPACE = 'xmlns="http://www.opengis.net/kml/2.2"'


def _ring(level: float, *, closed: bool = True) -> list[tuple[float, float]]:
    """A rectangle whose size shrinks as the level rises. A simple hill."""
    half = 0.004 * (1.0 - LEVELS.index(level) * 0.25)
    corners = [
        (LON0 - half, LAT0 - half),
        (LON0 + half, LAT0 - half),
        (LON0 + half, LAT0 + half),
        (LON0 - half, LAT0 + half),
    ]
    return corners + [corners[0]] if closed else corners


def _coords(level: float, *, z: bool = False, closed: bool = True, sep: str = " ") -> str:
    suffix = f",{level}" if z else ""
    return sep.join(f"{lon},{lat}{suffix}" for lon, lat in _ring(level, closed=closed))


def _document(body: str, *, namespace: str = NAMESPACE, name: str = "Variant") -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<kml {namespace}><Document><name>{name}</name>\n{body}\n</Document></kml>"
    ).encode("utf-8")


# --------------------------------------------------------------------------- #
# Cascade strategy 1. Z coordinate
# --------------------------------------------------------------------------- #
def z_coordinate() -> bytes:
    """3D LineStrings with deliberately non-numeric names, so only z can win."""
    body = "\n".join(
        f"<Placemark><name>contour-{i}</name>"
        f"<LineString><coordinates>{_coords(lv, z=True, closed=False)}</coordinates>"
        f"</LineString></Placemark>"
        for i, lv in enumerate(LEVELS)
    )
    return _document(body)


# --------------------------------------------------------------------------- #
# Cascade strategy 2. ExtendedData
# --------------------------------------------------------------------------- #
def extended_data(field_name: str = "elevation", *, numeric_names: bool = False) -> bytes:
    """A schema field carries the height; names and coordinates give nothing away.

    `numeric_names` additionally labels the placemarks, so a test can check which field
    the cascade reaches for when both are present.
    """
    body = "\n".join(
        f"<Placemark><name>{lv if numeric_names else f'contour-{i}'}</name>"
        f"<ExtendedData><SchemaData schemaUrl=\"#c\">"
        f'<SimpleData name="ID">{i}</SimpleData>'
        f'<SimpleData name="{field_name}">{lv}</SimpleData>'
        f"</SchemaData></ExtendedData>"
        f"<LineString><coordinates>{_coords(lv, closed=False)}</coordinates></LineString>"
        f"</Placemark>"
        for i, lv in enumerate(LEVELS)
    )
    return _document(body)


def extended_data_untyped() -> bytes:
    """The other ExtendedData spelling: <Data name=..><value>..</value></Data>."""
    body = "\n".join(
        f"<Placemark><name>contour-{i}</name>"
        f'<ExtendedData><Data name="Elev"><value>{lv}</value></Data></ExtendedData>'
        f"<LineString><coordinates>{_coords(lv, closed=False)}</coordinates></LineString>"
        f"</Placemark>"
        for i, lv in enumerate(LEVELS)
    )
    return _document(body)


# --------------------------------------------------------------------------- #
# Cascade strategy 3. Placemark name (the provided sample's shape)
# --------------------------------------------------------------------------- #
def placemark_name(suffix: str = "", levels: tuple[float, ...] = LEVELS) -> bytes:
    body = "\n".join(
        f"<Placemark><name>{lv}{suffix}</name>"
        f"<LineString><coordinates>{_coords(LEVELS[i % len(LEVELS)], closed=False)}</coordinates>"
        f"</LineString></Placemark>"
        for i, lv in enumerate(levels)
    )
    return _document(body)


# --------------------------------------------------------------------------- #
# Cascade strategy 4. Enclosing Folder name
# --------------------------------------------------------------------------- #
def folder_name() -> bytes:
    """One folder per level. The outer folder name parses as a number on purpose,
    it names the *interval*, not a height, to prove the nearest folder wins."""
    body = "<Folder><name>Contours 1.0</name>" + "".join(
        f"<Folder><name>{lv:g} m</name>"
        f"<Placemark><name>contour-{i}</name>"
        f"<LineString><coordinates>{_coords(lv, closed=False)}</coordinates></LineString>"
        f"</Placemark></Folder>"
        for i, lv in enumerate(LEVELS)
    ) + "</Folder>"
    return _document(body)


# --------------------------------------------------------------------------- #
# Geometry flavours
# --------------------------------------------------------------------------- #
def polygon() -> bytes:
    """Closed contours exported as Polygons; the innermost also carries a hole, so the
    inner ring must be picked up as its own line."""
    parts = []
    for i, lv in enumerate(LEVELS):
        inner = ""
        if i == len(LEVELS) - 1:
            inner = (
                "<innerBoundaryIs><LinearRing><coordinates>"
                f"{_coords(LEVELS[-1])}"
                "</coordinates></LinearRing></innerBoundaryIs>"
            )
        parts.append(
            f"<Placemark><name>{lv}</name><Polygon>"
            f"<outerBoundaryIs><LinearRing><coordinates>{_coords(lv)}</coordinates>"
            f"</LinearRing></outerBoundaryIs>{inner}</Polygon></Placemark>"
        )
    return _document("\n".join(parts))


def multigeometry() -> bytes:
    """One Placemark, one elevation, several disjoint line segments. How a contour
    that leaves and re-enters the sheet is usually exported."""
    parts = []
    for lv in LEVELS:
        segments = "".join(
            f"<LineString><coordinates>{a[0]},{a[1]} {b[0]},{b[1]}</coordinates></LineString>"
            for a, b in zip(_ring(lv), _ring(lv)[1:])
        )
        parts.append(
            f"<Placemark><name>{lv}</name><MultiGeometry>{segments}</MultiGeometry></Placemark>"
        )
    return _document("\n".join(parts))


def no_namespace() -> bytes:
    """Namespace-free KML, as several desktop tools emit."""
    return _document(_body_of(placemark_name()), namespace="")


def spaced_coordinates() -> bytes:
    """Coordinates with spaces after the commas. Legal, and it breaks naive splitting."""
    body = "\n".join(
        f"<Placemark><name>{lv}</name><LineString><coordinates>"
        + _coords(lv, closed=False).replace(",", ", ")
        + "</coordinates></LineString></Placemark>"
        for lv in LEVELS
    )
    return _document(body)


def with_labels_folder() -> bytes:
    """A `labels` folder of duplicate Point placemarks, as the provided sample has.
    They must be ignored as geometry and used only as a consistency check (PLAN §11.1)."""
    lines = _body_of(placemark_name())
    labels = "<Folder><name>labels</name>" + "".join(
        f"<Placemark><name>{lv:g}</name><Point><coordinates>{LON0},{LAT0}</coordinates>"
        f"</Point></Placemark>"
        for lv in LEVELS
    ) + "</Folder>"
    return _document(f"<Folder><name>lines</name>{lines}</Folder>{labels}")


STRAY_LEVELS = tuple(float(v) for v in range(10, 210, 10))
"""Twenty contours, so a single unlabelled placemark is 5% of the file. That sits inside
the default coverage tolerance, as the sample's 1 in 1,356 does. Three contours plus one
stray would be 25%, which the guard is right to reject."""


def stray_3d_polygon() -> bytes:
    """The sample's real trap: 2D contours plus one 3D boundary polygon at a nonsense
    height. The z strategy must be rejected on coverage, and the polygon dropped."""
    lines = _body_of(placemark_name(levels=STRAY_LEVELS))
    stray = (
        "<Placemark><name>land</name><Polygon><outerBoundaryIs><LinearRing>"
        "<coordinates>"
        + " ".join(f"{lon},{lat},30" for lon, lat in _ring(LEVELS[0]))
        + "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>"
    )
    return _document(lines + stray)


def feet_interval() -> bytes:
    """A 5-unit contour interval. Almost certainly feet. Must warn, never convert."""
    return placemark_name(levels=(100.0, 105.0, 110.0))


def kmz(payload: bytes | None = None, member: str = "doc.kml") -> bytes:
    """Zip a KML document into a KMZ archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("files/overlay.png", b"\x89PNG not-a-real-image")
        archive.writestr(member, payload if payload is not None else placemark_name())
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Files that must be rejected
# --------------------------------------------------------------------------- #
def malformed_xml() -> bytes:
    return b"<kml><Document><Placemark><name>10</name></Document></kml>"


def points_only() -> bytes:
    body = "\n".join(
        f"<Placemark><name>{lv}</name><Point><coordinates>{LON0},{LAT0}</coordinates>"
        f"</Point></Placemark>"
        for lv in LEVELS
    )
    return _document(body)


def unlabelled_contours() -> bytes:
    body = "\n".join(
        f"<Placemark><name>contour {chr(97 + i)}</name>"
        f"<LineString><coordinates>{_coords(lv, closed=False)}</coordinates></LineString>"
        f"</Placemark>"
        for i, lv in enumerate(LEVELS)
    )
    return _document(body)


def single_level() -> bytes:
    body = "\n".join(
        f"<Placemark><name>10</name>"
        f"<LineString><coordinates>{_coords(lv, closed=False)}</coordinates></LineString>"
        f"</Placemark>"
        for lv in LEVELS
    )
    return _document(body)


# --------------------------------------------------------------------------- #
def _body_of(document: bytes) -> str:
    """Strip the <kml><Document> wrapper so a document can be nested inside another."""
    text = document.decode("utf-8")
    start = text.index("</name>") + len("</name>")
    return text[start : text.rindex("</Document>")]


VARIANTS = {
    "z_coordinate": z_coordinate,
    "extended_data": extended_data,
    "extended_data_untyped": extended_data_untyped,
    "placemark_name": placemark_name,
    "folder_name": folder_name,
    "polygon": polygon,
    "multigeometry": multigeometry,
    "no_namespace": no_namespace,
    "spaced_coordinates": spaced_coordinates,
    "with_labels_folder": with_labels_folder,
    "stray_3d_polygon": stray_3d_polygon,
    "feet_interval": feet_interval,
    "kmz": kmz,
    "malformed_xml": malformed_xml,
    "points_only": points_only,
    "unlabelled_contours": unlabelled_contours,
    "single_level": single_level,
}


if __name__ == "__main__":  # pragma: no cover
    import pathlib
    import sys

    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "variants")
    out.mkdir(parents=True, exist_ok=True)
    for name, build in VARIANTS.items():
        suffix = ".kmz" if name == "kmz" else ".kml"
        path = out / f"{name}{suffix}"
        path.write_bytes(build())
        print(f"{path}  {path.stat().st_size:,} bytes")
