"""The generator: build the whole electrode-bundle DXF in memory.

`build_bundle(cfg)` is the old hook_bundle_generator ``__main__`` turned into a function.
It returns a `BundleResult` carrying the ezdxf doc AND the in-memory data the two
consumers need -- the per-channel contact centres (`electrode_locs`, for IONP) and the
channel->pad assignment (`chan_pad`, for the flex mapping) -- so neither consumer has to
re-read the DXF or re-derive the routing order.

Rendering is decoupled: geometry emission collects (xs, ys, colour) polylines into
`result.render`, and `render_bundle(result, ax)` draws them, replacing the inline
matplotlib calls that used to be interleaved with the geometry in __main__.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import dxf_io as io
from . import shapes as sh
from .config import BundleConfig
from .geometry import convert_rectangle_to_polyline, create_ellipse_polygon, polygon_centroid


@dataclass
class BundleResult:
    doc: object
    msp: object
    electrode_locs: np.ndarray            # Nx2 contact centres, channel order (for IONP)
    top_points: List[Tuple[float, float]]  # (x_src, wire_top) per channel
    chan_pad: Dict[int, tuple]            # channel -> (column, pad_row) (for the mapping)
    wire_top: float
    pads_left: List[Tuple[float, float]]
    pads_right: List[Tuple[float, float]]
    num_channels: int
    render: List[tuple] = field(default_factory=list)  # (xs, ys, colour) debug polylines


def _closed(xy):
    """(x_arr, y_arr) -> closed (xs, ys) lists for plotting (first point repeated)."""
    xs, ys = list(xy[0]), list(xy[1])
    return xs + [xs[0]], ys + [ys[0]]


def build_bundle(cfg: BundleConfig) -> BundleResult:
    import ezdxf

    #If a reference DXF is given, drive the per-channel centres from the real electrode
    #centroids; otherwise fall back to the parametric grid.
    centroids = None
    num_channels = cfg.num_channels
    if cfg.ref_dxf is not None:
        centroids = io.extract_centroids_from_dxf(cfg.ref_dxf, cfg.ref_layer)
        num_channels = len(centroids)
        print(f"Loaded {num_channels} electrode centroids from {cfg.ref_dxf} (layer '{cfg.ref_layer}')")

    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    io.ensure_layer(doc, 'Metal', color=4)
    io.ensure_layer(doc, 'Polyimide', color=3)
    io.ensure_layer(doc, 'Electrodes', color=2)
    io.ensure_layer(doc, 'Etching', color=1)
    io.ensure_layer(doc, 'Polyimide_Negative', color=5)

    render: List[tuple] = []
    l = cfg.l

    #Two columns of bond pads on the Metal layer. The bottom pad of each column is REF/GND
    #(left unconnected); the other 32 per column are wired to the channels below.
    pads_left, pads_right = sh.build_pad_columns(
        cfg.pad_row_pitch, cfg.pad_first_y, cfg.pad_pitch, cfg.n_pads_per_column)
    for cx, cy in pads_left + pads_right:
        circle_poly = sh.create_polygon_circle(cx, cy, radius=cfg.pad_diam / 2, resolution=64)
        msp.add_lwpolyline(circle_poly, close=True, dxfattribs={'layer': 'Metal'})
        render.append((circle_poly[:, 0], circle_poly[:, 1], 'k'))

    #Wires run upward from each pad to a common top. In parametric mode this is l_max (so
    #output is unchanged); in reference mode it must clear the deepest electrode.
    wire_top = cfg.l_max
    if centroids is not None and len(centroids):
        wire_top = max(cfg.l_max, float(centroids[:, 1].max()) + cfg.l_max)

    #Resolve per-channel wire lengths (parametric mode only; reference mode gets cy from the
    #DXF centroids). Base = a cumulative per-gap staircase: gap before channel i defaults to
    #delta_y (overridable via delta_y_overrides), positions accumulate from cy_0 = -delta_y, so
    #uniform gaps reproduce the original cy_i = (i-1)*delta_y exactly.
    electrode_lengths = None
    if centroids is None:
        if cfg.electrode_lengths_profile is None:
            if not cfg.delta_y_overrides:
                #Uniform staircase: keep the exact single-multiply form so the default output is
                #bit-identical (repeated addition would round differently in the last digit).
                electrode_lengths = [wire_top - (i - 1) * cfg.delta_y for i in range(num_channels)]
            else:
                gaps = {i: cfg.delta_y for i in range(num_channels)}
                for ch, g in cfg.delta_y_overrides.items():
                    assert 1 <= ch < num_channels, \
                        f"delta_y_overrides channel {ch} out of range 1..{num_channels - 1}"
                    gaps[ch] = float(g)
                electrode_lengths, cy_stack = [], -cfg.delta_y
                for i in range(num_channels):
                    if i > 0:
                        cy_stack += gaps[i]
                    electrode_lengths.append(wire_top - cy_stack)
        else:
            electrode_lengths = [float(cfg.electrode_lengths_profile(i)) for i in range(num_channels)]
        for ch, length in (cfg.length_overrides or {}).items():
            assert 0 <= ch < num_channels, \
                f"length_overrides channel {ch} out of range 0..{num_channels - 1}"
            electrode_lengths[ch] = float(length)
        if any(length <= l for length in electrode_lengths):
            print(f"WARNING: some electrode lengths <= pad size l={l}; "
                  "those wires will have non-positive height (degenerate geometry).")

    #Parametric x of every channel, placed so the gap between neighbouring polyimide traces is
    #constant even with a widened channel (the wide trace pushes only the channels beyond it).
    wire_hw = cfg.wire_hw
    half_widths = {i: (cfg.wide_wire_hw if i == cfg.wide_channel else wire_hw)
                   for i in range(num_channels)}
    chan_x = sh.channel_x_positions(num_channels, cfg.delta_x, half_widths, wire_hw)

    #One continuous polyimide body behind the bond pads AND the fanout, drawn once (NOT per
    #channel): narrow bottom on the wire bundle -> diagonal fan-out -> vertical stem ->
    #tangent-fillet flare into the wide block over the pad columns.
    poly_block_top = pads_left[-1][1] + 500
    poly_top_hw = cfg.polyimide_width / 2
    poly_bot_hw = max(abs(x) for x in chan_x.values()) + wire_hw + cfg.polyimide_bundle_margin
    y_fan_top = wire_top + cfg.polyimide_fan_height
    y_curve = cfg.pad_first_y - cfg.polyimide_curve_offset
    polyimide_body = sh.build_polyimide_fanout_body(
        poly_bot_hw, cfg.polyimide_fan_hw, poly_top_hw,
        y_bottom=wire_top, y_fan_top=y_fan_top, y_curve=y_curve,
        y_top=poly_block_top, R=cfg.polyimide_curve_r,
    )

    electrode_pads: Dict[int, tuple] = {}
    contact_pads: Dict[int, tuple] = {}
    polyimide_outlines: Dict[int, tuple] = {}
    polyimide_parts = []                # (x,y) per polyimide piece (traces + hooks) to merge
    polyimide_tip_ys = []               # per-channel tip y; shallowest = negative-wedge start
    top_points = []                     # (x_src, wire_top) per channel, fed to the fanout
    electrode_locs = []                 # (x, y) contact centre per channel (for IONP)
    hook_etch = None
    rescue_etch = None

    for i in range(num_channels):
        #Per-channel centre: (bx = unsigned x magnitude, cy = y, side = +/-1).
        if centroids is not None:
            cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
            side = -1 if cx < 0 else 1
            bx = abs(cx)
        else:
            xc_signed = chan_x[i]
            side = -1 if xc_signed < 0 else 1
            bx = abs(xc_signed)
            cy = wire_top - electrode_lengths[i]

        xc = side * bx
        top_points.append((xc, wire_top))       # wire top, where the fanout neck begins

        #Electrode PAD + WIRE as ONE merged metal polygon.
        electrode_pads[i] = sh.create_metal_pad_wire_outline(
            xc, cy, pad_hw=l / 2, wire_hw=cfg.wire_width / 2, wire_top_y=wire_top,
            pad_R=cfg.metal_pad_arc_radius, shoulder_R=cfg.metal_pad_arc_radius)
        render.append((*_closed(electrode_pads[i]), 'b'))

        #Etched contact area above the electrode pad, with rounded corners.
        contact_pads[i] = sh.create_polyimide_pads_with_arcs(
            side * (bx - cfg.l_contact / 2), cy - cfg.l_contact / 2,
            side * cfg.l_contact, cfg.l_contact, cfg.contact_pad_arc_radius)
        render.append((*_closed(contact_pads[i]), 'r'))

        #Contact centre = mean of the contact polyline EXACTLY as written to the DXF (the closed
        #ring repeats its first vertex), so IONP wells match reading the Electrodes layer back.
        contact_ring = convert_rectangle_to_polyline(contact_pads[i])
        electrode_locs.append(tuple(contact_ring.mean(axis=0)))

        #Polyimide as ONE merged polygon: flat-topped upper trace, wider pad with convex
        #rounded corners, lower trace + sharp tip, concave fillets smoothing the pad<->trace steps.
        pad_hw = l / 2 + cfg.padding          # =10.5
        if i == cfg.wide_channel:
            this_wire_hw = cfg.wide_wire_hw
            this_R = 0.0
            this_tipy = (cy - pad_hw) - 250
        elif i == cfg.second_channel:
            this_wire_hw = wire_hw
            this_R = cfg.polyimide_arc_radius
            this_tipy = (cy - pad_hw) - 250
        else:
            this_wire_hw = wire_hw
            this_R = cfg.polyimide_arc_radius
            this_tipy = (cy - pad_hw) - 250 - 20
        polyimide_tip_ys.append(this_tipy)
        polyimide_outlines[i] = sh.create_polyimide_outline(
            xc, cy, pad_hw, this_wire_hw,
            wire_top_y=(cy + pad_hw) + (wire_top - cy),
            wire_bot_y=(cy - pad_hw) - 250,
            tip_y=this_tipy,
            R=this_R,
        )
        polyimide_parts.append(polyimide_outlines[i])

        #Hook anchor at the bottom of the wide channel's lower trace, + inner etch ellipse.
        if cfg.hook_on_wide_channel and i == cfg.wide_channel:
            y_attach = (cy - pad_hw) - 250
            hx, hy = sh.create_hook_polygon(xc, y_attach, scale=cfg.hook_scale, drop=cfg.hook_drop)
            #Centre the etch on the BARB BODY (drop=0), then shift down by the drop, so the thin
            #drop-stem riser doesn't skew the area centroid upward as hook_drop grows.
            hbx, hby = sh.create_hook_polygon(xc, y_attach, scale=cfg.hook_scale, drop=0.0)
            ecx, ecy = polygon_centroid(hbx, hby)
            ecy -= cfg.hook_drop
            hook_etch = create_ellipse_polygon(ecx + cfg.hook_etch_dx, ecy + cfg.hook_etch_dy,
                                               cfg.hook_etch_rx, cfg.hook_etch_ry)
            polyimide_parts.append((hx, hy))
            render.append((hook_etch[:, 0], hook_etch[:, 1], 'r'))

        #Circular 'rescue' hook on the second channel: drop-stem + tapered neck -> plain circle.
        if cfg.hook_on_second_channel and i == cfg.second_channel:
            y_attach = (cy - pad_hw) - 250
            rhx, rhy = sh.create_rescue_hook_polygon(
                xc, y_attach,
                angle_deg=cfg.rescue_hook_angle_deg,
                arm_len=cfg.rescue_hook_arm_len,
                wire_end_width=2 * this_wire_hw,
                circle_end_width=cfg.rescue_hook_circle_width,
                circle_radius=cfg.rescue_hook_circle_r,
                drop=cfg.rescue_hook_drop,
                scale=cfg.hook_scale,
            )
            if cfg.rescue_hook_etch_r > 0:
                ccx, ccy = sh.rescue_hook_circle_centre(
                    xc, y_attach, cfg.rescue_hook_angle_deg, cfg.rescue_hook_arm_len,
                    drop=cfg.rescue_hook_drop, scale=cfg.hook_scale)
                rescue_etch = create_ellipse_polygon(ccx, ccy, cfg.rescue_hook_etch_r,
                                                     cfg.rescue_hook_etch_r)
            polyimide_parts.append((rhx, rhy))
            if rescue_etch is not None:
                render.append((rescue_etch[:, 0], rescue_etch[:, 1], 'r'))

    #Metal pad+wire and etched contact area per channel.
    for i in range(num_channels):
        msp.add_lwpolyline(convert_rectangle_to_polyline(electrode_pads[i]),
                           close=True, dxfattribs={'layer': 'Metal'})
        msp.add_lwpolyline(convert_rectangle_to_polyline(contact_pads[i]),
                           close=True, dxfattribs={'layer': 'Electrodes'})

    #Merge the fanout body + every per-channel trace + the hooks into ONE polygon with rounded
    #trace<->body junctions (band-restricted, radius polyimide_join_r). One Polyimide ring each.
    pad_hw = l / 2 + cfg.padding
    polyimide_merged = sh.merge_polyimide_with_fillets(
        polyimide_body, polyimide_parts,
        band_ylo=wire_top - 3 * cfg.polyimide_join_r,
        band_yhi=wire_top + pad_hw + 3 * cfg.polyimide_join_r,
        R=cfg.polyimide_join_r,
    )
    for ring in polyimide_merged:
        io.add_ring(msp, ring, 'Polyimide')
        rx = [p[0] for p in ring]; ry = [p[1] for p in ring]
        render.append((rx + [rx[0]], ry + [ry[0]], 'g'))

    #NEGATIVE of the polyimide: a constant-width MOAT hugging the polyimide, with the polyimide
    #punched out as a hole; below the shallowest tip it becomes a symmetric straight wedge to a
    #single tip under the deepest wire (the off-centre hook).
    from shapely.geometry import Polygon as _Polygon, box as _box
    from shapely.ops import unary_union as _unary_union
    _poly = _unary_union([_Polygon(r).buffer(0) for r in polyimide_merged])
    _outer = _poly.buffer(cfg.polyimide_negative_margin, join_style=2)
    _y_sh = max(polyimide_tip_ys)
    _xs = max(abs(x) for x in chan_x.values()) + wire_hw + cfg.polyimide_negative_margin
    _ob = _outer.bounds
    _apex_y = _poly.bounds[1] - cfg.polyimide_negative_margin
    _apex_x = chan_x[cfg.wide_channel] if cfg.wide_channel is not None else 0.0
    _body = _outer.difference(_box(_ob[0] - 1, _ob[1] - 1, _ob[2] + 1, wire_top))
    _rect = _box(-_xs, _y_sh, _xs, wire_top)
    _wedge = _Polygon([(-_xs, _y_sh), (_xs, _y_sh), (_apex_x, _apex_y)])
    _outer = _unary_union([_body, _rect, _wedge])
    _negative = _outer.difference(_poly)
    for _g in (_negative.geoms if _negative.geom_type == "MultiPolygon" else [_negative]):
        _coords = sh.polygon_to_seamed_ring(_g)
        io.add_ring(msp, _coords, 'Polyimide_Negative')
        render.append(([p[0] for p in _coords], [p[1] for p in _coords], 'm'))

    #Hook/rescue inner etch holes on the Etching layer.
    if hook_etch is not None:
        io.add_ring(msp, hook_etch, 'Etching')
    if rescue_etch is not None:
        io.add_ring(msp, rescue_etch, 'Etching')

    #Route every electrode wire up into its bond pad: one tapered Metal polygon per channel.
    route_polys = sh.build_pad_routes(
        top_points, pads_left, pads_right,
        neck_tw=cfg.fan_neck_tw, top_tw=cfg.fan_top_tw,
        neck_len=cfg.fan_neck_len, fan_len=cfg.fan_fan_len,
        bundle_pitch=cfg.bundle_pitch, approach_deg=cfg.pad_approach_deg,
    )
    for i in range(num_channels):
        ring = route_polys.get(i)
        if ring is None:
            continue
        io.add_ring(msp, ring, 'Metal')
        rx = [p[0] for p in ring]; ry = [p[1] for p in ring]
        render.append((rx + [rx[0]], ry + [ry[0]], 'b'))

    chan_pad = sh.assign_channels_to_pads(top_points)
    return BundleResult(
        doc=doc, msp=msp,
        electrode_locs=np.asarray(electrode_locs, dtype=float),
        top_points=top_points, chan_pad=chan_pad, wire_top=wire_top,
        pads_left=pads_left, pads_right=pads_right,
        num_channels=num_channels, render=render,
    )


def render_bundle(result: BundleResult, ax) -> None:
    """Draw the collected debug polylines onto a matplotlib axis (equal aspect)."""
    for xs, ys, colour in result.render:
        ax.plot(xs, ys, colour)
    ax.set_aspect('equal')
