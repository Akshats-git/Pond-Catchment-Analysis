"""Phase 1. KML/KMZ contour parser.

Two kinds of test here. The first block pins the parser against the *provided* sample
sheet: those numbers are the Phase 1 acceptance criteria and must not drift. The second
block runs the structural variants, which is what makes the claim "this works on contour
files other than the one we were given" checkable rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import ParserConfig
from app.core.kml_parser import ContourParseError, parse_contour_file, parse_contours
from tests.fixtures import make_variants as mv

SAMPLE = "data/contours_1m.kml"


# --------------------------------------------------------------------------- #
# The provided sheet
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample():
    return parse_contour_file(SAMPLE)


def test_sample_acceptance_numbers(sample):
    """PLAN Phase 1: 1,355 lines, 159,113 vertices, 32 levels, 267-298 m."""
    m = sample.metadata
    assert m.line_count == 1355
    assert m.vertex_count == 159113
    assert m.level_count == 32
    assert m.elevation_range == (267.0, 298.0)
    assert m.interval_m == pytest.approx(1.0)


def test_sample_uses_the_placemark_name_strategy(sample):
    assert sample.metadata.elevation_source == "placemark_name"


def test_sample_levels_are_a_complete_one_metre_ladder(sample):
    assert sample.metadata.levels == tuple(float(v) for v in range(267, 299))


def test_sample_bbox_is_the_raipur_sheet(sample):
    min_lon, min_lat, max_lon, max_lat = sample.metadata.bbox
    assert (min_lon, max_lon) == pytest.approx((81.281404, 81.312647), abs=1e-5)
    assert (min_lat, max_lat) == pytest.approx((21.239822, 21.263581), abs=1e-5)


def test_stray_land_polygon_is_excluded(sample):
    """The sheet carries one `land` boundary polygon at z=30 with a non-numeric name.

    Letting it through would drag the elevation range from 267-298 down to 30 and put a
    30 m cliff in the middle of the DEM.
    """
    assert sample.metadata.skipped_features == 1
    assert sample.elevations.min() == 267.0
    assert len(sample.metadata.warnings) == 1
    assert "had no height that could be read" in sample.metadata.warnings[0]


def test_labels_folder_is_ignored_but_cross_checked(sample):
    """1,355 duplicate Point placemarks: no geometry contributed, used as a check."""
    assert sample.metadata.label_placemark_count == 1355
    assert sample.metadata.labels_consistent is True


def test_sample_arrays_are_structurally_consistent(sample):
    assert sample.points.shape == (sample.vertex_count, 2)
    assert sample.elevations.shape == (sample.vertex_count,)
    assert sample.line_starts[0] == 0
    assert sample.line_starts[-1] == sample.vertex_count
    assert len(sample.line_starts) == sample.line_count + 1
    # Every line has at least two vertices, or it is not a line.
    assert np.all(np.diff(sample.line_starts) >= 2)


def test_every_vertex_carries_its_own_contour_elevation(sample):
    """PLAN §2 Step 1. The whole basis of the DEM."""
    for i in (0, 1, 700, sample.line_count - 1):
        block = sample.elevations[sample.line_starts[i] : sample.line_starts[i + 1]]
        assert np.all(block == sample.line_elevations[i])
    assert len(sample.line_elevations) == sample.line_count


def test_no_elevation_is_off_the_ladder(sample):
    assert set(np.unique(sample.elevations)) <= set(sample.metadata.levels)


# --------------------------------------------------------------------------- #
# The elevation cascade
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "builder, expected_source",
    [
        (mv.z_coordinate, "z_coordinate"),
        (mv.extended_data, "extended_data"),
        (mv.extended_data_untyped, "extended_data"),
        (mv.placemark_name, "placemark_name"),
        (mv.folder_name, "folder_name"),
    ],
)
def test_each_cascade_strategy_resolves_its_own_flavour(builder, expected_source):
    cs = parse_contours(builder())
    assert cs.metadata.elevation_source == expected_source
    assert cs.metadata.levels == mv.LEVELS


def test_nearest_folder_wins_over_a_numeric_outer_folder():
    """The outer folder name parses as a number, but it names the contour *interval*,
    not a height. The nearest enclosing folder is the one that means anything."""
    cs = parse_contours(mv.folder_name())
    assert cs.metadata.levels == mv.LEVELS
    assert 1.0 not in cs.metadata.levels


def test_cascade_prefers_z_over_a_numeric_name():
    """Both are present and both are valid; order decides. z is the most direct."""
    cs = parse_contours(mv.z_coordinate().replace(b"contour-0", b"999"))
    assert cs.metadata.elevation_source == "z_coordinate"


def test_a_single_3d_placemark_cannot_hijack_the_cascade():
    """The coverage guard, in miniature: this is the sample's `land` polygon trap."""
    cs = parse_contours(mv.stray_3d_polygon())
    assert cs.metadata.elevation_source == "placemark_name"
    assert cs.metadata.levels == mv.STRAY_LEVELS
    assert cs.metadata.skipped_features == 1


def test_id_like_extended_fields_are_not_mistaken_for_elevations():
    """The sample's only SimpleData field is `ID`, running 0..1354. A plausible-looking
    set of "elevations" that would silently destroy the terrain. The cascade must skip
    it and fall through to the placemark name."""
    cs = parse_contours(mv.extended_data(field_name="ID", numeric_names=True))
    assert cs.metadata.elevation_source == "placemark_name"
    assert cs.metadata.levels == mv.LEVELS


def test_an_id_field_alone_is_not_enough_to_parse():
    """Same file without numeric names: nothing resolves, rather than `ID` winning."""
    with pytest.raises(ContourParseError) as excinfo:
        parse_contours(mv.extended_data(field_name="ID"))
    assert excinfo.value.code == "no_elevations"


def test_a_hyphen_in_a_name_is_not_read_as_a_minus_sign():
    """`contour-12` must not resolve to -12 m. A label prefix is only honoured when it
    ends at a real separator, so identifiers fall through to the next strategy."""
    cs = parse_contours(mv.folder_name())
    assert cs.metadata.elevation_source == "folder_name"
    assert min(cs.metadata.levels) > 0


def test_coverage_threshold_is_configurable():
    """A stricter threshold rejects a file the default accepts. Proving the guard is
    the thing doing the work, not a coincidence of ordering."""
    document = mv.stray_3d_polygon()
    strict = ParserConfig(strategy_min_coverage=1.0)
    with pytest.raises(ContourParseError) as excinfo:
        parse_contours(document, config=strict)
    assert excinfo.value.code == "no_elevations"


# --------------------------------------------------------------------------- #
# Geometry flavours
# --------------------------------------------------------------------------- #
def test_polygon_rings_are_read_as_contours():
    """Outer boundary plus one inner ring -> four lines from three placemarks."""
    cs = parse_contours(mv.polygon())
    assert cs.line_count == 4
    assert cs.metadata.levels == mv.LEVELS


def test_multigeometry_segments_share_one_elevation():
    cs = parse_contours(mv.multigeometry())
    assert cs.line_count == 4 * len(mv.LEVELS)
    assert set(cs.line_elevations.tolist()) == set(mv.LEVELS)


def test_namespace_agnostic():
    """KML 2.0/2.1/2.2 and namespace-free files must all parse identically."""
    plain = parse_contours(mv.no_namespace())
    namespaced = parse_contours(mv.placemark_name())
    assert plain.metadata.levels == namespaced.metadata.levels
    assert plain.vertex_count == namespaced.vertex_count


def test_coordinates_may_carry_spaces_after_commas():
    spaced = parse_contours(mv.spaced_coordinates())
    tight = parse_contours(mv.placemark_name())
    np.testing.assert_allclose(spaced.points, tight.points)


def test_labels_folder_contributes_no_geometry():
    with_labels = parse_contours(mv.with_labels_folder())
    without = parse_contours(mv.placemark_name())
    assert with_labels.vertex_count == without.vertex_count
    assert with_labels.metadata.label_placemark_count == len(mv.LEVELS)
    assert with_labels.metadata.labels_consistent is True


def test_kmz_archive():
    kmz = parse_contours(mv.kmz(), filename="contours.kmz")
    kml = parse_contours(mv.placemark_name())
    assert kmz.metadata.levels == kml.metadata.levels
    np.testing.assert_allclose(kmz.points, kml.points)


def test_kmz_detected_by_magic_bytes_without_a_filename():
    assert parse_contours(mv.kmz()).metadata.levels == mv.LEVELS


def test_kmz_with_a_non_standard_member_name():
    assert parse_contours(mv.kmz(member="export/contours.kml")).metadata.levels == mv.LEVELS


# --------------------------------------------------------------------------- #
# Units
# --------------------------------------------------------------------------- #
def test_feet_interval_is_flagged_and_not_converted():
    """PLAN Phase 1: detect feet-vs-metres, flag it, never silently convert."""
    cs = parse_contours(mv.feet_interval())
    assert cs.metadata.interval_m == pytest.approx(5.0)
    assert cs.metadata.elevation_range == (100.0, 110.0)  # unchanged
    assert any("feet" in w for w in cs.metadata.warnings)


def test_unit_suffixes_are_parsed_and_reported():
    cs = parse_contours(mv.placemark_name(suffix=" m"))
    assert cs.metadata.levels == mv.LEVELS
    assert cs.metadata.unit_hint == "m"


def test_a_one_metre_interval_raises_no_unit_warning():
    cs = parse_contours(mv.placemark_name(levels=(10.0, 11.0, 12.0)))
    assert not any("feet" in w for w in cs.metadata.warnings)


# --------------------------------------------------------------------------- #
# Rejections. Structured, with a code the API can map to a status
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "builder, code",
    [
        (mv.malformed_xml, "unparseable_xml"),
        (mv.points_only, "no_contours"),
        (mv.unlabelled_contours, "no_elevations"),
        (mv.single_level, "too_few_levels"),
    ],
)
def test_bad_input_raises_a_structured_error(builder, code):
    with pytest.raises(ContourParseError) as excinfo:
        parse_contours(builder())
    assert excinfo.value.code == code
    assert excinfo.value.detail


def test_empty_upload_is_rejected():
    with pytest.raises(ContourParseError) as excinfo:
        parse_contours(b"   ")
    assert excinfo.value.code == "unreadable_file"


def test_oversized_upload_is_rejected_before_parsing():
    tiny = ParserConfig(max_upload_bytes=10)
    with pytest.raises(ContourParseError) as excinfo:
        parse_contours(mv.placemark_name(), config=tiny)
    assert excinfo.value.code == "file_too_large"


def test_a_kmz_that_is_not_a_zip_is_rejected():
    with pytest.raises(ContourParseError) as excinfo:
        parse_contours(b"PK\x03\x04 truncated garbage", filename="x.kmz")
    assert excinfo.value.code == "unreadable_file"


def test_a_kmz_without_a_kml_member_is_rejected():
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", b"no kml here")
    with pytest.raises(ContourParseError) as excinfo:
        parse_contours(buffer.getvalue(), filename="x.kmz")
    assert excinfo.value.code == "unreadable_file"
