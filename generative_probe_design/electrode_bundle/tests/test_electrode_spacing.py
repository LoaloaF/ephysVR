"""Per-gap electrode spacing (delta_y_overrides) in BundleConfig / build_bundle.

Electrode depth is a cumulative-gap staircase: the gap BEFORE channel i defaults to
delta_y, overridable via delta_y_overrides. Positions accumulate, so changing one gap
shifts every electrode above it and leaves those below untouched. We assert on the
contact-centre y-values (build_bundle's electrode_locs, channel order); the pad shape is
identical between configs, so y differences isolate the cy shift exactly.
"""
import numpy as np
import pytest

from electrode_bundle.bundle import build_bundle
from electrode_bundle.config import BundleConfig


def _contact_ys(cfg):
    return build_bundle(cfg).electrode_locs[:, 1]


def _base_cfg():
    """Override-free baseline: the pure delta_y staircase these tests measure against.

    Deliberately NOT a bare BundleConfig() -- the shipped default carries a dense band
    (delta_y_overrides={33..48: 30.0}), so a bare config is already compressed and would
    contaminate every base-vs-override comparison below.
    """
    return BundleConfig(delta_y_overrides=None)


def test_no_overrides_is_uniform_delta_y_staircase():
    cfg = _base_cfg()
    ys = _contact_ys(cfg)
    # consecutive electrodes are exactly delta_y apart (uniform staircase, no overrides)
    assert np.allclose(np.diff(ys), cfg.delta_y)


def test_gap_override_shifts_only_electrodes_above():
    base = _base_cfg()
    dy = base.delta_y
    y_base = _contact_ys(base)
    y_over = _contact_ys(BundleConfig(delta_y_overrides={5: 80.0}))

    # channels below the widened gap are untouched...
    assert np.allclose(y_over[:5], y_base[:5])
    # ...and channels 5.. all shift up by exactly (new_gap - delta_y).
    assert np.allclose(y_over[5:] - y_base[5:], 80.0 - dy)


def test_multiple_gap_overrides_accumulate():
    base = _base_cfg()
    dy = base.delta_y
    y_base = _contact_ys(base)
    y_over = _contact_ys(BundleConfig(delta_y_overrides={5: 80.0, 6: 80.0}))

    assert np.allclose(y_over[:5], y_base[:5])
    # channel 5 gains one widened gap; channels 6.. gain two.
    assert np.allclose(y_over[5] - y_base[5], 80.0 - dy)
    assert np.allclose(y_over[6:] - y_base[6:], 2 * (80.0 - dy))


def test_gap_override_out_of_range_raises():
    with pytest.raises(AssertionError):
        build_bundle(BundleConfig(delta_y_overrides={0: 80.0}))   # ch0 has no gap before it
    with pytest.raises(AssertionError):
        build_bundle(BundleConfig(delta_y_overrides={64: 80.0}))  # >= num_channels
