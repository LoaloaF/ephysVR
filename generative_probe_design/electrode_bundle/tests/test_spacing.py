"""Site layouts (spacing.Uniform / spacing.Segments) -> real electrode positions."""
import pytest

from electrode_bundle.bundle import build_bundle
from electrode_bundle.config import BundleConfig
from electrode_bundle.spacing import Segments, Uniform


def _contact_ys(layout, n_channels=64):
    cfg = BundleConfig(num_channels=n_channels, **layout.delta_y_kwargs(n_channels))
    return build_bundle(cfg).electrode_locs[:, 1]


def test_uniform_spans_exactly_what_it_says():
    ys = _contact_ys(Uniform(span=1200.0))
    assert ys[-1] - ys[0] == pytest.approx(1200.0)


def test_uniform_pitch_is_span_over_gaps():
    #span is the handle; pitch is derived, which is how the design docs are written
    assert Uniform(span=5790).delta_y_kwargs(64)["delta_y"] == pytest.approx(5790 / 63)


def test_segments_shared_boundary_decides_gap_ownership():
    #"Ch 1-42: 90 um, Ch 42-43: 970 um, Ch 43-64: 95 um" -- 42 and 43 each appear twice,
    #and that shared label is what assigns the gap across the boundary
    overrides = Segments([(1, 42, 90.0), (42, 43, 970.0), (43, 64, 95.0)]).overrides(64)
    assert len(overrides) == 63
    assert all(overrides[g] == 90.0 for g in range(1, 42))
    assert overrides[42] == 970.0
    assert all(overrides[g] == 95.0 for g in range(43, 64))


def test_segments_produce_those_positions_for_real():
    layout = Segments([(1, 42, 90.0), (42, 43, 970.0), (43, 64, 95.0)])
    ys = _contact_ys(layout)
    assert ys[-1] - ys[0] == pytest.approx(41 * 90.0 + 970.0 + 21 * 95.0)
    assert ys[-1] - ys[0] == pytest.approx(layout.span_um(64))


def test_segments_rest_pitch_fills_the_remainder():
    #"Ch 1-40: 40 um, rest 120 um pitch"
    overrides = Segments([(1, 40, 40.0)], rest_pitch=120.0).overrides(64)
    assert len(overrides) == 63
    assert all(overrides[g] == 40.0 for g in range(1, 40))
    assert all(overrides[g] == 120.0 for g in range(40, 64))


def test_segments_must_cover_every_gap():
    with pytest.raises(ValueError):
        Segments([(1, 40, 40.0)]).overrides(64)


def test_layouts_agree_when_they_describe_the_same_probe():
    """A Uniform and the Segments spelling of it must land on identical positions."""
    uniform = _contact_ys(Uniform(span=63 * 50.0))
    segmented = _contact_ys(Segments([(1, 64, 50.0)]))
    assert segmented == pytest.approx(uniform)
