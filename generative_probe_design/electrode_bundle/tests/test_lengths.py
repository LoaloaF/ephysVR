"""ribbon_length (config) and the bundle-length / loop-distance solves (lengths.py).

The spec's numbers are `fiber_length` (loop -> shoulder) and `loop_offset` (loop ->
deepest site); these pin
that solve_lengths lands on them exactly and that the ribbon behaves as a fixed connector
run rather than a stretchy one.
"""
import pytest

from electrode_bundle.bundle import build_bundle
from electrode_bundle.config import BundleConfig
from electrode_bundle.lengths import measure, solve_lengths


def test_default_ribbon_reproduces_historical_pad_first_y():
    #pad_first_y used to be a literal 15800; it is now derived and must not have moved
    assert BundleConfig().pad_first_y == 15800.0


def test_pads_slide_with_the_fiber_instead_of_stretching_the_ribbon():
    base, short = BundleConfig(), BundleConfig(bottom_elec=8000)
    assert base.pad_first_y - base.l_max == short.pad_first_y - short.l_max
    #a 4000 um shorter fiber moves the pads down by exactly 4000 um
    assert base.pad_first_y - short.pad_first_y == pytest.approx(4000.0)


def test_ribbon_length_moves_only_the_pads():
    base = BundleConfig()
    longer = BundleConfig(ribbon_length=base.ribbon_length + 500)
    assert longer.pad_first_y - base.pad_first_y == pytest.approx(500.0)
    assert longer.l_max == base.l_max          # the shoulder does not move


def test_solve_hits_fiber_length():
    cfg = solve_lengths(BundleConfig(delta_y=90.0, delta_y_overrides=None), fiber_length=9000)
    _, got, _ = measure(build_bundle(cfg))
    assert got == pytest.approx(9000.0, abs=0.05)


def test_solve_hits_loop_offset():
    cfg = solve_lengths(BundleConfig(delta_y=90.0, delta_y_overrides=None),
                        loop_offset=445)
    _, _, got = measure(build_bundle(cfg))
    assert got == pytest.approx(445.0, abs=0.05)


def test_solve_hits_both_targets_together():
    #the loop distance is solved first because moving the loop also moves the shoulder
    #measurement; both must still be on target afterwards
    cfg = solve_lengths(BundleConfig(delta_y=85.0, delta_y_overrides=None),
                        fiber_length=10600, loop_offset=345)
    _, bundle, loop = measure(build_bundle(cfg))
    assert bundle == pytest.approx(10600.0, abs=0.05)
    assert loop == pytest.approx(345.0, abs=0.05)


def test_solve_leaves_the_site_span_alone():
    """Lengths and spacing are independent axes: solving one must not disturb the other."""
    base = BundleConfig(delta_y=90.0, delta_y_overrides=None)
    span_before, _, _ = measure(build_bundle(base))
    cfg = solve_lengths(base, fiber_length=9000, loop_offset=445)
    span_after, _, _ = measure(build_bundle(cfg))
    assert span_after == pytest.approx(span_before)


def test_overall_length_makes_the_ribbon_absorb_fiber_differences():
    """Two designs with different fibers but one overall_length end at the same connector."""
    from electrode_bundle.batch import config_for
    from electrode_bundle.design_sets import DesignSpec
    from electrode_bundle.lengths import measure_overall
    from electrode_bundle.spacing import Uniform

    short = DesignSpec(name="short", n_channels=64, sites=Uniform(span=980),
                       fiber_length=6500, loop_offset=550, overall_length=14406.5)
    long_ = DesignSpec(name="long", n_channels=64, sites=Uniform(span=5590),
                       fiber_length=10600, loop_offset=345, overall_length=14406.5)

    cfg_s, cfg_l = config_for(short), config_for(long_)
    assert measure_overall(build_bundle(cfg_s), cfg_s) == pytest.approx(14406.5, abs=0.05)
    assert measure_overall(build_bundle(cfg_l), cfg_l) == pytest.approx(14406.5, abs=0.05)
    #the 4100 um of fiber the long one uses comes straight out of its ribbon
    assert cfg_s.ribbon_length - cfg_l.ribbon_length == pytest.approx(10600 - 6500)


def test_lengths_are_additive_as_the_docstring_claims():
    """loop_offset + span + free = fiber_length, and fiber_length + ribbon = overall."""
    from electrode_bundle.batch import config_for
    from electrode_bundle.design_sets import DESIGNS
    from electrode_bundle.lengths import measure_overall

    for spec in DESIGNS:
        cfg = config_for(spec)
        result = build_bundle(cfg)
        span, fiber, loop = measure(result)
        free = fiber - loop - span
        assert free > 0, f"{spec.name}: sites do not fit below the shoulder"
        assert loop + span + free == pytest.approx(fiber)
        assert fiber + cfg.ribbon_length == pytest.approx(
            measure_overall(result, cfg), abs=0.05)


def test_overall_length_too_short_for_the_fanout_raises():
    from electrode_bundle.batch import config_for
    from electrode_bundle.design_sets import DesignSpec
    from electrode_bundle.spacing import Uniform

    #only 500 um of ribbon left for a 2000 um fanout
    spec = DesignSpec(name="cramped", n_channels=64, sites=Uniform(span=980),
                      fiber_length=6500, loop_offset=550, overall_length=7000)
    with pytest.raises(ValueError, match="ribbon"):
        config_for(spec)
