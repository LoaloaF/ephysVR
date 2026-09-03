import numpy as np
import pytest

from electrode_bundle.config import (
    IonpConfig, BundleConfig, ID_TOP_ANCHOR_CENTER, ID_SLOT_CENTERS, id_codeword,
)
from electrode_bundle.bundle import build_bundle
from electrode_bundle.ionp import (
    build_id_pattern, drop_wells_near_contacts, calc_ionp_well_locs, effective_y_top,
)


@pytest.fixture(scope="module")
def bundle():
    return build_bundle(BundleConfig())


def _centers(pattern):
    """Auto-sized bands are symmetric, so each band's midpoint is its stripe center."""
    return [round((s + e) / 2, 3) for s, e in pattern]


def _band_counts(el_locs, cfg, pattern):
    """Real well count inside each band of `pattern`, after the contact keep-out."""
    y_top, yb = effective_y_top(el_locs, cfg), cfg.y_bottom
    yr = y_top - yb
    xs, ys = calc_ionp_well_locs(el_locs, pattern, y_top, yb,
                                 cfg.ionp_well_distance, cfg.ionp_firstwell_distance)
    xs, ys = drop_wells_near_contacts(xs, ys, el_locs, cfg.keepout_radius)
    return [int(np.sum((ys >= s * yr + yb) & (ys <= e * yr + yb))) for s, e in pattern]


# --- pattern structure (geometry-aware) ---

def test_id_codeword_min_hamming_distance_is_3():
    # the headline guarantee: every pair of the 16 IDs differs in >= 3 data stripes
    cws = [id_codeword(i) for i in range(16)]
    assert len(set(cws)) == 16
    dmin = min(sum(a != b for a, b in zip(cws[i], cws[j]))
               for i in range(16) for j in range(i + 1, 16))
    assert dmin == 3


def test_id0_is_two_anchors_only(bundle):
    # codeword 0 -> no data slots; just the tip anchor + top anchor
    p = build_id_pattern(0, bundle.electrode_locs, IonpConfig())
    assert len(p) == 2
    assert p[0][0] == 0.0                                   # bottom anchor pinned at the tip
    assert round((p[1][0] + p[1][1]) / 2, 2) == ID_TOP_ANCHOR_CENTER  # top anchor centered


def test_id15_has_nine_stripes(bundle):
    # codeword of 15 is all-ones (7 data slots) + 2 anchors
    assert len(build_id_pattern(15, bundle.electrode_locs, IonpConfig())) == 9


def test_both_anchors_present_in_all_16(bundle):
    for i in range(16):
        p = build_id_pattern(i, bundle.electrode_locs, IonpConfig())
        assert p[0][0] == 0.0                                        # tip anchor at the bottom
        assert round((p[-1][0] + p[-1][1]) / 2, 2) == ID_TOP_ANCHOR_CENTER  # top anchor


def test_present_slots_match_codeword(bundle):
    for i in range(16):
        cs = _centers(build_id_pattern(i, bundle.electrode_locs, IonpConfig()))
        expected = {round(c, 3) for bit, c in zip(id_codeword(i), ID_SLOT_CENTERS) if bit}
        assert expected.issubset(set(cs))


def test_patterns_sorted_ascending_by_start(bundle):
    for i in range(16):
        starts = build_id_pattern(i, bundle.electrode_locs, IonpConfig())[:, 0]
        assert np.all(np.diff(starts) > 0)


def test_all_16_patterns_distinct(bundle):
    seen = {frozenset(_centers(build_id_pattern(i, bundle.electrode_locs, IonpConfig())))
            for i in range(16)}
    assert len(seen) == 16


# --- the MRI guarantee: every stripe of every ID clears the 700-well floor ---

def test_every_stripe_meets_700(bundle):
    el = bundle.electrode_locs
    for i in range(16):
        cfg = IonpConfig(pattern_i=i)
        pat = build_id_pattern(i, el, cfg)
        counts = _band_counts(el, cfg, pat)
        for n in counts:
            assert cfg.min_n_wells <= n <= cfg.max_n_wells, f"ID {i}: strip {n} wells out of range"
        yr = effective_y_top(el, cfg) - cfg.y_bottom
        for k in range(1, len(pat)):
            gap = (pat[k][0] - pat[k - 1][1]) * yr
            assert gap >= cfg.min_strip_distance, f"ID {i}: strips too close ({gap:.0f}um)"


# --- config ---

def test_config_min_n_wells_is_700():
    assert IonpConfig().min_n_wells == 700


def test_config_anchor_heights_default():
    cfg = IonpConfig()
    assert cfg.id_bot_anchor_h_um == 550.0 and cfg.id_top_anchor_h_um == 350.0


def test_config_default_pattern_is_none():
    # heights are geometry-dependent -> pattern is built at runtime, not resolved in the config
    assert IonpConfig().pattern is None


def test_config_explicit_pattern_preserved():
    custom = np.array([[0.1, 0.2], [0.9, 1.0]])
    cfg = IonpConfig(pattern_i=3, pattern=custom)
    assert np.array_equal(cfg.pattern, custom)


def test_config_keepout_radius_default():
    assert IonpConfig().keepout_radius == 30.0


def test_config_id_top_margin_default():
    assert IonpConfig().id_top_margin == 220.0


# --- keep-out filter ---

def test_keepout_removes_wells_within_radius():
    xs = np.array([0.0, 0.0, 100.0])
    ys = np.array([10.0, 40.0, 40.0])          # first well 10um from a contact at (0,0)
    contacts = np.array([[0.0, 0.0]])
    fx, fy = drop_wells_near_contacts(xs, ys, contacts, radius=30.0)
    assert set(zip(fx.tolist(), fy.tolist())) == {(0.0, 40.0), (100.0, 40.0)}


def test_keepout_radius_zero_is_noop():
    xs = np.array([0.0, 5.0])
    ys = np.array([0.0, 0.0])
    contacts = np.array([[0.0, 0.0]])
    fx, fy = drop_wells_near_contacts(xs, ys, contacts, radius=0.0)
    assert len(fx) == 2 and len(fy) == 2


# --- effective top (id_top_margin) ---

def test_effective_y_top_uses_highest_electrode_plus_margin():
    el_locs = np.array([[0.0, 100.0], [10.0, 1968.0], [20.0, 500.0]])  # max y = 1968
    cfg = IonpConfig(id_top_margin=220.0)
    assert effective_y_top(el_locs, cfg) == 1968.0 + 220.0


def test_effective_y_top_none_falls_back_to_y_top():
    el_locs = np.array([[0.0, 100.0], [10.0, 1968.0]])
    cfg = IonpConfig(id_top_margin=None)
    assert effective_y_top(el_locs, cfg) == cfg.y_top


# --- legend ---

def test_legend_renders_16_panels_and_writes_png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from electrode_bundle.id_legend import render_id_legend
    out = tmp_path / "legend.png"
    fig = render_id_legend(str(out))
    assert len(fig.axes) == 16
    assert out.exists() and out.stat().st_size > 0


# --- per-ID output filenames + CLI ---

def test_config_filenames_encode_pattern_i():
    cfg = IonpConfig(pattern_i=7)
    assert "id07" in cfg.dxf_file
    assert "id07" in cfg.json_file
    assert IonpConfig(pattern_i=3).dxf_file != IonpConfig(pattern_i=4).dxf_file
    assert IonpConfig(pattern_i=3).json_file != IonpConfig(pattern_i=4).json_file


def test_config_explicit_filenames_preserved():
    cfg = IonpConfig(pattern_i=9, dxf_file="/tmp/custom.dxf", json_file="/tmp/custom.json")
    assert cfg.dxf_file == "/tmp/custom.dxf"
    assert cfg.json_file == "/tmp/custom.json"


def test_cli_accepts_pattern_i():
    from electrode_bundle.main import _build_parser
    args = _build_parser().parse_args(["all", "--pattern-i", "11"])
    assert args.pattern_i == 11


def test_cli_rejects_out_of_range_pattern_i():
    from electrode_bundle.main import _build_parser
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["all", "--pattern-i", "16"])
