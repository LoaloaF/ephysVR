# Electrode Generator — see USER SETTINGS below for parameters. ###MET 22/06/26
# authors: Mario Togni & Baran Yasar

import numpy as np
from matplotlib import pyplot as plt
import ezdxf
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon
from shapely.geometry import LineString
from shapely.strtree import STRtree
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
from scipy.spatial import ConvexHull
from collections import defaultdict
import time

# Timing infrastructure
_timing_log: list[tuple[str, float]] = []
_t_script_start = time.perf_counter()

def print_timing_summary():
    total_script = time.perf_counter() - _t_script_start
    if not _timing_log:
        print("[timing] No phases recorded.")
        return
    max_label = max(len(l) for l, _ in _timing_log)
    col = max(max_label, 30)
    print()
    print("=" * (col + 22))
    print(f"  {'Phase':<{col}}  {'Time (s)':>8}  {'% total':>7}")
    print("-" * (col + 22))
    for label, elapsed in _timing_log:
        pct = 100.0 * elapsed / total_script if total_script > 0 else 0.0
        print(f"  {label:<{col}}  {elapsed:>8.3f}  {pct:>6.1f}%")
    measured_sum = sum(e for _, e in _timing_log)
    print("-" * (col + 22))
    print(f"  {'Sum of measured phases':<{col}}  {measured_sum:>8.3f}")
    print(f"  {'Total script wall time':<{col}}  {total_script:>8.3f}")
    print("=" * (col + 22))

# ===========================================================
# USER SETTINGS
# ===========================================================

# Presets: "4x Xiaomin 16", "1x Xiaomin 64", "4x Xiaomin 64",
# "4x Xiaomin 64 v2", "50mu pitch", "50mu pitch v2", "32 800 mu span"

def run_design(Type):
    _timing_log.clear()
    global _t_script_start
    _t_script_start = time.perf_counter()
    if Type == "4x Xiaomin 16":
        num_instances    = 4            
        num_channels     = 16
        wire_width_scale = 4.0
        padding          = 4
        padding2         = 13
        delta_x          = 24       ##space between electrodes
        shaft_length_below_electrode = -250
        bottom_elec      = 6000
        target_length    = 1000
        connector_height = 0
        roof_angle       = np.deg2rad(80)

    if Type == "1x Xiaomin 64":
        num_instances    = 1
        num_channels     = 64
        wire_width_scale = 4.0
        padding          = 4
        padding2         = 13
        delta_x          = 24
        shaft_length_below_electrode = -250
        bottom_elec      = 6000
        target_length    = 1000
        connector_height = 0
        roof_angle       = np.deg2rad(80)

    if Type == "4x Xiaomin 64 v2":
        num_instances    = 4
        num_channels     = 64
        wire_width_scale = 4.0
        padding          = 4
        padding2         = 13
        delta_x          = 22
        shaft_length_below_electrode = -20
        bottom_elec      = 6000
        target_length    = 1000
        connector_height = 0
        roof_angle       = np.deg2rad(80)

    if Type == "4x Xiaomin 64":
        num_instances    = 4
        num_channels     = 64
        wire_width_scale = 4.0
        padding          = 4
        padding2         = 13
        delta_x          = 24
        shaft_length_below_electrode = -250
        bottom_elec      = 6000
        target_length    = 1000
        connector_height = 0
        roof_angle       = np.deg2rad(80)

    if Type == "50mu pitch":
        num_instances    = 1
        num_channels     = 64
        wire_width_scale = 4.0
        padding          = 4.5
        padding2         = 130
        delta_x          = 24
        shaft_length_below_electrode = -250
        bottom_elec      = 8500
        target_length    = 3000
        connector_height = 0
        roof_angle       = np.deg2rad(60)

    if Type == "50mu pitch v2":
        num_instances    = 1
        num_channels     = 64
        wire_width_scale = 10.0
        padding          = 4.5
        padding2         = 130
        delta_x          = 24
        shaft_length_below_electrode = -250
        bottom_elec      = 8500
        target_length    = 3000
        connector_height = 0
        roof_angle       = np.deg2rad(60)

    if Type == "32 800 mu span":
        num_instances    = 1
        num_channels     = 32
        wire_width_scale = 4.0
        padding          = 4.5
        padding2         = 71
        delta_x          = 24
        shaft_length_below_electrode = -250
        bottom_elec      = 4000
        target_length    = 800
        connector_height = 0
        roof_angle       = np.deg2rad(80)

    # Wire / contact pad sizes
    wire_width       = 2                                   # width of metal wires
    wire_width_scale2 = 10.0                               #scaling of vertical metal wires at contact pads
    wire_width_scale3 = 15.0                               #scaling of vertical metal wires at contact pads

    l                = 13                                  # metal pad under electrode contact size
    l_contact        = 11                                  # electrode contact size

    # Loop anchor placement
    loop_index       = 0                                    # location of loop tentacle
    loop_spacing     = 300                                  #spacing between electrode loops
    center_loop_length  = -295                              #distance between botom contact pad, loop center

    # Solder pad / wire routing
    wire_spacing_solderpad = 20                             #wire spacing between solderpads
    solder_pad_offset = 50                                  #vertical offset 

    # Top structure heights
    top_rect_height        = 22000
    inverted_roof_height   = 2000

    roof_angle2                  = np.deg2rad(40)
    second_inset                 = 0.0
    second_roof_height_offset    = None

    # Outline
    outline_offset = 120        #Outline thickness
    outline_smooth = 30

    # Silicon layer
    silicon_outline_offset = 50   # outward buffer applied to silicon stadium shape
    silicon_outline_smooth = 30   # inward erosion to smooth the outer hull

    silicon_circle_radius  = 30   #radius of circle above loop centers
    silicon_rect_width     = 1000
    silicon_rect_height    = 400
    silicon_bottom_diameter = 500

    silicon_bridge_width  = 20   # thickness of each etch-resist bridge
    silicon_bridge_margin = 50   # how far each bridge reaches past the ring

    # Aluminium
    aluminium_margin = 30   # safety margin around the polyimide/silicon overlap

    # Anchor marker
    ANCHOR_MARKER_W = 35
    ANCHOR_MARKER_H = 5

    # Contact Pads
    stack_w       = 820     
    stack_h       = 120
    stack_spacing = 80

    # Loop-anchor shape: corner radii / arc resolution
    ANCHOR_CORNER_R  = 10
    ANCHOR_CORNER_R1 = 20
    ANCHOR_CORNER_R2 = 30
    ANCHOR_ARC_PTS   = 16

    # Corner rounding
    ROUND_POLYIMIDE  = True
    ROUND_METAL      = True
    ROUND_ELECTRODES = True

    CORNER_RADII = [(60, 4), (300, 16), (float('inf'), 40)]

    # Collision solver (instance placement)
    max_iterations = 5000
    angle_step     = np.deg2rad(0.25)

    # Basic helpers

    def plot_rectangle(rect, color):
        x = rect[0]; y = rect[1]
        for i in range(4):
            if i < 3:
                plt.plot([x[i], x[i+1]], [y[i], y[i+1]], color)
            else:
                plt.plot([x[i], x[0]], [y[i], y[0]], color)

    def plot_polyline(poly, color):
        plt.plot(poly[0], poly[1], color)

    def create_rectangle(x0, y0, w, l):
        x = [x0, x0+w, x0+w, x0]
        y = [y0, y0, y0+l, y0+l]
        return (x, y)

    def create_circle(cx, cy, r, num_points=120):
        theta = np.linspace(0, 2*np.pi, num_points)
        return (cx + r*np.cos(theta), cy + r*np.sin(theta))

    def create_semicircle(cx, cy, r, start_angle, end_angle, num_points=60):
        theta = np.linspace(start_angle, end_angle, num_points)
        return (cx + r*np.cos(theta), cy + r*np.sin(theta))

    # Loop-anchor shape

    _ANCHOR_COORDS_RAW = [
        (-10.5,  80),
        (-10.5,  70),
        (-10.5,  40),
        (-25,    32),
        (-25,     5),
        (-42,    48),           
        (  0,   -20),
        ( 25,   -10),
        ( 42,    48),           
        ( 25,     5),
        ( 25,    32),
        ( 10.5,  40),
        ( 10.5,  70),
        (10.5,    80),
    ]

    ANCHOR_COORDS = [(x, y - 10) for x, y in _ANCHOR_COORDS_RAW]

    ANCHOR_RADII = [
        0,
        0,
        ANCHOR_CORNER_R2,
        ANCHOR_CORNER_R1,
        0,
        0,
        ANCHOR_CORNER_R1,
        ANCHOR_CORNER_R2,
        ANCHOR_CORNER_R1,
        0,
        0,
        ANCHOR_CORNER_R,
        ANCHOR_CORNER_R1,
        0,
        0,
    ]

    def _arc_between(p0, p1, cx, cy, r, n=16):
        """Circular arc of radius r centred at (cx, cy) from p0 to p1."""
        a0 = np.arctan2(p0[1] - cy, p0[0] - cx)
        a1 = np.arctan2(p1[1] - cy, p1[0] - cx)
        da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        angles = np.linspace(a0, a0 + da, n)
        return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]


    def build_anchor_polygon(cx, cy, coords=None, radii=None, arc_pts=None):
        if coords   is None: coords   = ANCHOR_COORDS
        if radii    is None: radii    = ANCHOR_RADII
        if arc_pts  is None: arc_pts  = ANCHOR_ARC_PTS

        if len(coords) != len(radii):
            raise ValueError(
                f"build_anchor_polygon: coords has {len(coords)} points but "
                f"radii has {len(radii)} entries — they must be the same length."
            )

        n   = len(coords)
        pts = list(coords)
        rad = [0.0 if (r is None or r <= 0) else float(r) for r in radii]

        result = []
        for i in range(n):
            r = rad[i]
            if r == 0.0:
                result.append(pts[i])
                continue

            prev_pt = pts[(i - 1) % n]
            curr_pt = pts[i]
            next_pt = pts[(i + 1) % n]

            d_in  = np.array(curr_pt) - np.array(prev_pt)
            d_out = np.array(next_pt) - np.array(curr_pt)
            l_in  = np.hypot(*d_in);  l_out = np.hypot(*d_out)
            if l_in < 1e-12 or l_out < 1e-12:
                result.append(curr_pt); continue

            u_back = -(d_in  / l_in)
            u_out  =   d_out / l_out

            bisector = u_back + u_out
            bis_len  = np.hypot(*bisector)
            if bis_len < 1e-9:
                result.append(curr_pt); continue
            bis_norm = bisector / bis_len

            cos_h = np.clip(np.dot(bis_norm, u_out), -1.0, 1.0)
            sin_h = np.sqrt(max(0.0, 1.0 - cos_h ** 2))
            if sin_h < 1e-9:
                result.append(curr_pt); continue

            setback = r * cos_h / sin_h
            max_sb  = min(l_in, l_out) * 0.45
            if setback > max_sb:
                r       = max_sb * sin_h / max(cos_h, 1e-12)
                setback = max_sb

            t_in_pt  = np.array(curr_pt) + setback * u_back
            t_out_pt = np.array(curr_pt) + setback * u_out
            centre   = np.array(curr_pt) + (r / sin_h) * bis_norm

            arc = _arc_between(t_in_pt, t_out_pt, centre[0], centre[1], r, n=arc_pts)
            result.extend(arc)

        result.append(result[0])
        xs = [p[0] + cx for p in result]
        ys = [p[1] + cy for p in result]
        return (xs, ys)


    # Rotation helpers

    def rotate_points(xs, ys, cx, cy, theta):
        ct = np.cos(theta); st = np.sin(theta)
        xs_a = np.asarray(xs, dtype=np.float64)
        ys_a = np.asarray(ys, dtype=np.float64)
        dx = xs_a - cx; dy = ys_a - cy
        xs2 = cx + dx*ct - dy*st
        ys2 = cy + dx*st + dy*ct
        return xs2.tolist(), ys2.tolist()

    # Corner-rounding helpers

    _ARC_N    = 30
    _T_FRAC   = np.linspace(0.0, 1.0, _ARC_N)
    _ARC_OFFS = np.arange(_ARC_N, dtype=np.int32)
    _TWO_PI   = 2.0 * np.pi
    _FLAT_COS = -0.9994


    def _remove_collinear(pts):
        if len(pts) < 4:
            return pts
        keep = np.ones(len(pts), dtype=bool)
        changed = True
        while changed:
            changed = False
            idx = np.where(keep)[0]
            m   = len(idx)
            if m < 3:
                break
            p      = pts[idx]
            prev_p = p[np.arange(m) - 1]
            next_p = p[(np.arange(m) + 1) % m]
            d0 = p      - prev_p
            d1 = next_p - p
            cross_abs = np.abs(d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0])
            to_remove = cross_abs < 1e-6
            if to_remove.any():
                keep[idx[to_remove]] = False
                changed = True
        return pts[keep]


    def round_polygon_corners(coords, radii_thresholds):
        if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.dtype == np.float64:
            pts = coords
        else:
            pts = np.asarray(coords, dtype=np.float64).reshape(-1, 2)

        if len(pts) > 1 and pts[0, 0] == pts[-1, 0] and pts[0, 1] == pts[-1, 1]:
            pts = pts[:-1]

        pts = _remove_collinear(pts)

        n = len(pts)
        if n < 3:
            return pts.copy()

        v_in  = pts - np.concatenate((pts[-1:], pts[:-1]))
        v_out = np.concatenate((pts[1:], pts[:1])) - pts

        sq_in  = v_in[:,0]*v_in[:,0]   + v_in[:,1]*v_in[:,1]
        sq_out = v_out[:,0]*v_out[:,0] + v_out[:,1]*v_out[:,1]
        safe   = (sq_in > 1e-18) & (sq_out > 1e-18)

        len_in  = np.sqrt(np.where(safe, sq_in,  1.0))
        len_out = np.sqrt(np.where(safe, sq_out, 1.0))
        u_back  = -v_in  * np.where(safe, 1.0/len_in,  0.0)[:,None]
        u_fwd   =  v_out * np.where(safe, 1.0/len_out, 0.0)[:,None]

        bisector = u_back + u_fwd
        bis_sq   = bisector[:,0]*bisector[:,0] + bisector[:,1]*bisector[:,1]
        bis_ok   = bis_sq > 1e-18
        inv_bis  = np.where(bis_ok, 1.0/np.sqrt(np.where(bis_ok, bis_sq, 1.0)), 0.0)
        bis_norm = bisector * inv_bis[:,None]

        cos_half = (bis_norm[:,0]*u_fwd[:,0] + bis_norm[:,1]*u_fwd[:,1]).clip(-1.0, 1.0)
        sin_half = np.sqrt(np.maximum(0.0, 1.0 - cos_half*cos_half))

        shorter = np.where(safe, np.minimum(len_in, len_out), 0.0)

        sorted_thresholds = sorted(radii_thresholds, key=lambda x: x[0])
        r_arr = np.zeros(n)
        for thresh, rad in reversed(sorted_thresholds):
            r_arr = np.where(shorter < thresh, rad, r_arr)

        r_arr = np.minimum(r_arr, shorter * 0.45)

        do_arc = safe & bis_ok & (sin_half > 1e-9) & (r_arr > 1e-6) & (cos_half > _FLAT_COS)
        idx    = np.where(do_arc)[0]
        n_arc  = len(idx)

        ri  = r_arr[idx];  ch = cos_half[idx];  sh = sin_half[idx]
        sb  = ri*ch/sh;    dc = ri/sh
        pi_ = pts[idx];    ub = u_back[idx];    uf = u_fwd[idx]
        t_in   = pi_ + sb[:,None]*ub
        t_out  = pi_ + sb[:,None]*uf
        centre = pi_ + dc[:,None]*bis_norm[idx]

        cross = (-ub[:,0])*uf[:,1] - (-ub[:,1])*uf[:,0]
        a_s   = np.arctan2(t_in[:,1]-centre[:,1],  t_in[:,0]-centre[:,0])
        a_e   = np.arctan2(t_out[:,1]-centre[:,1], t_out[:,0]-centre[:,0])
        ccw   = cross > 0
        a_e   = np.where( ccw & (a_e < a_s), a_e + _TWO_PI, a_e)
        a_e   = np.where(~ccw & (a_e > a_s), a_e - _TWO_PI, a_e)

        a_mat  = a_s[:,None] + (a_e - a_s)[:,None] * _T_FRAC
        arc_xy = np.empty((n_arc, _ARC_N, 2))
        arc_xy[:,:,0] = centre[:,0:1] + ri[:,None]*np.cos(a_mat)
        arc_xy[:,:,1] = centre[:,1:2] + ri[:,None]*np.sin(a_mat)

        slot_sizes      = np.empty(n, dtype=np.int32); slot_sizes[:] = 1
        slot_sizes[idx] = _ARC_N
        slot_start      = np.empty(n+1, dtype=np.int32); slot_start[0] = 0
        np.cumsum(slot_sizes, out=slot_start[1:])
        out             = np.empty((int(slot_start[-1]), 2))
        pass_idx        = np.where(~do_arc)[0]
        out[slot_start[pass_idx]] = pts[pass_idx]
        out[(slot_start[idx][:,None] + _ARC_OFFS[None,:]).ravel()] = arc_xy.reshape(-1, 2)
        return out


    def _round_shapely_polygon(poly, radii_thresholds):
        if poly is None or poly.is_empty:
            return poly
        rounded_ext = round_polygon_corners(
            np.array(poly.exterior.coords, dtype=np.float64), radii_thresholds)
        if len(rounded_ext) < 3:
            return poly
        rounded_holes = []
        for interior in poly.interiors:
            rh = round_polygon_corners(
                np.array(interior.coords, dtype=np.float64), radii_thresholds)
            if len(rh) >= 3:
                rounded_holes.append(rh)
        try:
            result = ShapelyPolygon(rounded_ext, rounded_holes)
            if not result.is_valid:
                result = result.buffer(0)
            return result if not result.is_empty else poly
        except Exception:
            return poly


    def _round_multi_or_poly(geom, radii_thresholds):
        if geom.geom_type == 'MultiPolygon':
            parts = [_round_shapely_polygon(g, radii_thresholds) for g in geom.geoms]
            parts = [p for p in parts if p is not None and not p.is_empty]
            return unary_union(parts) if parts else geom
        return _round_shapely_polygon(geom, radii_thresholds)

    def build_rect_with_semicircles(cx, cy, w, h, num_pts=64):
        r = w / 2.0
        t_bot = np.linspace(np.pi, 2*np.pi, num_pts//2, endpoint=False)
        t_top = np.linspace(0, np.pi, num_pts//2, endpoint=False)
        xb = cx + r * np.cos(t_bot)
        yb = cy - h/2 + r * np.sin(t_bot)
        xt = cx + r * np.cos(t_top)
        yt = cy + h/2 + r * np.sin(t_top)
        xs = np.concatenate([xb, xt, [xb[0]]])
        ys = np.concatenate([yb, yt, [yb[0]]])
        return xs.tolist(), ys.tolist()

    # Derived parameters

    l_max            = bottom_elec - l / 2
    delta_y          = target_length / (num_channels - 1)   

    delta_x_anchor = delta_x + (l - wire_width) / 2

    def _x_counter_pos(x_counter):
        if x_counter == 0:
            return 0.0
        return delta_x_anchor + (x_counter - 1) * delta_x

    upper_width = num_channels * wire_width* wire_width_scale + (num_channels - 1) * wire_spacing_solderpad + 2 * padding2
    lower_width = (num_channels-1)*delta_x + wire_width + 2 * padding
    print(lower_width, upper_width)
    roof_target_height     = connector_height + ((upper_width - lower_width) /2) * np.tan(roof_angle)

    # Storage

    electrode_wires        = {}
    electrode_pads         = {}
    contact_pads           = {}
    polyimide_pads         = {}
    polyimide_wires        = {}
    polyimide_wires_below  = {}
    polyimide_donuts       = {}
    anchor_placements      = {}
    top_connectors         = {}
    roof_points_left       = []
    roof_points_right      = []
    instance_roofs         = {}
    roof_top_left          = {}
    roof_top_right         = {}
    metal_slanted          = {}
    metal_vertical         = {}
    anchor_markers         = {}

    # Main geometry

    plt.figure(figsize=[20, 20])

    _t0 = time.perf_counter()
    for inst in range(num_instances):
        print(f"  [geometry] instance {inst+1}/{num_instances} ...")

        x_offset = inst * loop_spacing
        odd = True
        x_counter = 0
        instance_min_x = +1e9
        instance_max_x = -1e9

        def shift_rect(rect):
            return ([x + x_offset for x in rect[0]], rect[1])

        for i in range(num_channels):
            if i % 16 == 0:
                print(f"    channel {i+1}/{num_channels}  (+{time.perf_counter()-_t0:.2f}s)")

            side = -1 if odd else +1
            odd  = not odd
            if side == +1:
                x_counter += 1

            r = create_rectangle(
                side * (_x_counter_pos(x_counter) - wire_width / 2),
                (i - 1) * delta_y + l / 2,
                side * wire_width,
                l_max - ((i - 1) * delta_y - connector_height)
            )
            r = shift_rect(r)
            electrode_wires[(inst, i)] = r
            plot_rectangle(r, 'b')

            r = create_rectangle(
                side * (_x_counter_pos(x_counter) - l / 2),
                (i - 1) * delta_y - l / 2,
                side * l, l
            )
            r = shift_rect(r)
            electrode_pads[(inst, i)] = r
            plot_rectangle(r, 'b')

            r = create_rectangle(
                side * (_x_counter_pos(x_counter) - l_contact / 2),
                (i - 1) * delta_y - l_contact / 2,
                side * l_contact, l_contact
            )
            r = shift_rect(r)
            contact_pads[(inst, i)] = r
            plot_rectangle(r, 'r')

            r = create_rectangle(
                side * (_x_counter_pos(x_counter) - l / 2 - padding),
                (i - 1) * delta_y - l / 2 - padding,
                side * (l + 2 * padding), (l + 2 * padding)
            )
            r = shift_rect(r)
            polyimide_pads[(inst, i)] = r
            plot_rectangle(r, 'g')

            if i != loop_index:
                r = create_rectangle(
                        side * (_x_counter_pos(x_counter) - wire_width / 2 - padding),
                        (i-1) * delta_y + l / 2,
                        side * (wire_width + 2 * padding),
                        l_max - ((i - 1) * delta_y)
                    )
            else:
                r = create_rectangle(
                    side * (_x_counter_pos(x_counter) - (l + 2 * padding) / 2),
                    (i - 1) * delta_y + l / 2,
                    side * (l + 2 * padding),
                    l_max - ((i - 1) * delta_y)
                )
            r = shift_rect(r)
            polyimide_wires[(inst, i)] = r
            plot_rectangle(r, 'g')

            if i != loop_index:
                shaft_w = abs(wire_width + 2 * padding)
                shaft_h = shaft_length_below_electrode
                raw_x0  = side * (_x_counter_pos(x_counter) - wire_width / 2 - padding)
                x_left  = min(raw_x0, raw_x0 + side * (wire_width + 2 * padding))
                x_right = x_left + shaft_w
                y_top   = (i - 1) * delta_y - l / 2
                y_bot   = y_top + shaft_h
                x_mid   = 0.5 * (x_left + x_right)
                tip_len = shaft_w * 0.75
                raw_x = [x_left,  x_right, x_right, x_mid,           x_left,  x_left]
                raw_y = [y_top,   y_top,   y_bot,   y_bot - tip_len,  y_bot,   y_top]
                xs_shifted = [x + x_offset for x in raw_x]
                r = (xs_shifted, raw_y)
            else:
                r = create_rectangle(
                    side * (_x_counter_pos(x_counter) - (l + 2 * padding) / 2),
                    (i - 1) * delta_y - l / 2,
                    side * (l + 2 * padding), center_loop_length
                )
                r = shift_rect(r)

            polyimide_wires_below[(inst, i)] = r
            plot_polyline(r, 'g')

            if i == loop_index:
                anch_cx = x_offset
                anch_cy = (i - 1) * delta_y - l / 2 + center_loop_length
                anchor_poly = build_anchor_polygon(anch_cx, anch_cy)
                polyimide_donuts[(inst, i)]        = anchor_poly
                anchor_placements[(inst, i)]       = (anch_cx, anch_cy)
                plot_polyline(anchor_poly, 'g')

            instance_min_x = min(instance_min_x, min(r[0]))
            instance_max_x = max(instance_max_x, max(r[0]))

        top_y = l_max + connector_height + l / 2
        roof_points_left.append((instance_min_x, top_y))
        roof_points_right.append((instance_max_x, top_y))

        top_rect = create_rectangle(
            instance_min_x, l_max,
            instance_max_x - instance_min_x, connector_height + l / 2
        )
        top_connectors[inst] = top_rect
        plot_rectangle(top_rect, 'g')

    _timing_log.append(("Main geometry (wires / pads / anchors)", time.perf_counter() - _t0))
    print(f"  [timing] Main geometry: {_timing_log[-1][1]:.3f}s")

    # Anchor centers

    donut_centers = {}
    for inst in range(num_instances):
        cx, cy = anchor_placements[(inst, loop_index)]
        donut_centers[inst] = (cx, cy)

    # First roofs (per instance) — individual to each instance

    _t0 = time.perf_counter()
    for inst in range(num_instances):
        x0L, y0L = roof_points_left[inst]
        dy = roof_target_height + bottom_elec - y0L
        dx = dy / np.tan(roof_angle) if abs(np.tan(roof_angle)) > 1e-9 else 0.0
        x1L = x0L - dx;  y1L = roof_target_height + bottom_elec

        x0R, y0R = roof_points_right[inst]
        dy = roof_target_height + bottom_elec - y0R
        dx = dy / np.tan(roof_angle) if abs(np.tan(roof_angle)) > 1e-9 else 0.0
        x1R = x0R + dx;  y1R = roof_target_height + bottom_elec

        roof_x = [x0L, x1L, x1R, x0R]
        roof_y = [y0L, y1L, y1R, y0R]
        instance_roofs[inst]  = (roof_x, roof_y)
        roof_top_left[inst]   = (x1L, y1L)
        roof_top_right[inst]  = (x1R, y1R)
        plot_polyline((roof_x, roof_y), 'k')

    _timing_log.append(("First roofs (per instance)", time.perf_counter() - _t0))
    print(f"  [timing] First roofs: {_timing_log[-1][1]:.3f}s")

    # Sort wires 

    sorted_indices = []
    for i in range(num_channels):
        rect = electrode_wires[(0, i)]
        xc = 0.5 * (rect[0][0] + rect[0][1])
        sorted_indices.append((i, xc))
    sorted_indices.sort(key=lambda p: p[1])
    sorted_channel_order = [p[0] for p in sorted_indices]

    # Slanted wires

    _t0 = time.perf_counter()
    for inst in range(num_instances):
        xL, yL = roof_top_left[inst]
        xR, yR = roof_top_right[inst]
        LR     = np.hypot(xR - xL, yR - yL)
        ux     = (xR - xL) / LR;  uy = (yR - yL) / LR
        pad    = padding2+wire_width*wire_width_scale/2
        usable = LR - 2 * pad
        w1     = wire_width_scale * wire_width

        for rank, i in enumerate(sorted_channel_order):
            rect  = electrode_wires[(inst, i)]
            xc    = 0.5 * (rect[0][0] + rect[0][1])
            y_top = max(rect[1])
            t     = rank / (num_channels - 1)
            dist  = pad + t * usable
            xt    = xL + dist * ux;  yt = yL + dist * uy
            poly_x = [xc - wire_width/2, xc + wire_width/2, xt + w1/2, xt - w1/2, xc - wire_width/2]
            poly_y = [y_top, y_top, yt, yt, y_top]
            metal_slanted[(inst, i)] = (poly_x, poly_y)
            plot_polyline((poly_x, poly_y), 'b')

    _timing_log.append(("Slanted wires (per instance)", time.perf_counter() - _t0))
    print(f"  [timing] Slanted wires: {_timing_log[-1][1]:.3f}s")

    # Anchor markers

    _t0 = time.perf_counter()
    for inst in range(num_instances):
        cx, cy = anchor_placements[(inst, loop_index)]
        marker_xs, marker_ys = build_rect_with_semicircles(
            cx, cy, ANCHOR_MARKER_W, ANCHOR_MARKER_H)
        anchor_markers[inst] = (marker_xs, marker_ys)

    _timing_log.append(("Anchor markers", time.perf_counter() - _t0))
    print(f"  [timing] Anchor markers: {_timing_log[-1][1]:.3f}s")

    def _collect_instance_pts(inst):
        xs, ys = [], []
        for i in range(num_channels):
            for d in [electrode_wires, electrode_pads, contact_pads,
                    polyimide_pads, polyimide_wires, polyimide_wires_below]:
                xs.extend(d[(inst, i)][0]); ys.extend(d[(inst, i)][1])
            if (inst, i) in polyimide_donuts:
                xs.extend(polyimide_donuts[(inst, i)][0])
                ys.extend(polyimide_donuts[(inst, i)][1])
            if (inst, i) in metal_slanted:
                xs.extend(metal_slanted[(inst, i)][0]); ys.extend(metal_slanted[(inst, i)][1])
        for d in [top_connectors, instance_roofs]:
            if inst in d:
                xs.extend(d[inst][0]); ys.extend(d[inst][1])
        return xs, ys


    def get_instance_hull(inst, buffer_dist=4.0, buffer_resolution=16):
        xs, ys = _collect_instance_pts(inst)
        if len(xs) < 3:
            return np.empty((0, 2))
        pts  = np.column_stack([xs, ys])
        hull = ConvexHull(pts)
        poly = ShapelyPolygon(pts[hull.vertices])
        if buffer_dist:
            poly = poly.buffer(buffer_dist, resolution=buffer_resolution)
        if poly.is_empty:
            return np.empty((0, 2))
        if poly.geom_type == 'MultiPolygon':
            poly = max(poly.geoms, key=lambda g: g.area)
        return np.array(list(poly.exterior.coords))


    def polygons_intersect(poly1, poly2):
        if poly1.size == 0 or poly2.size == 0:
            return False
        def axes(p):
            result = []
            for i in range(len(p)):
                e = p[(i+1) % len(p)] - p[i]
                n = np.hypot(e[0], e[1])
                if n > 1e-12:
                    result.append(np.array([-e[1], e[0]]) / n)
            return result
        def proj(p, ax):
            d = p @ ax; return d.min(), d.max()
        for ax in axes(poly1) + axes(poly2):
            a1, b1 = proj(poly1, ax); a2, b2 = proj(poly2, ax)
            if b1 < a2 or b2 < a1:
                return False
        return True


    def rotate_instance(inst, theta):
        cx, cy = donut_centers[inst]
        for d in [electrode_wires, electrode_pads, contact_pads,
                polyimide_pads, polyimide_wires, polyimide_wires_below]:
            for i in range(num_channels):
                d[(inst, i)] = rotate_points(*d[(inst, i)], cx, cy, theta)
        if (inst, loop_index) in polyimide_donuts:
            polyimide_donuts[(inst, loop_index)] = rotate_points(
                *polyimide_donuts[(inst, loop_index)], cx, cy, theta)
        if inst in anchor_markers:
            anchor_markers[inst] = rotate_points(*anchor_markers[inst], cx, cy, theta)
        if (inst, loop_index) in anchor_placements:
            ax, ay = anchor_placements[(inst, loop_index)]
            rx, ry = rotate_points([ax], [ay], cx, cy, theta)
            anchor_placements[(inst, loop_index)] = (rx[0], ry[0])
            donut_centers[inst] = (rx[0], ry[0])
        for d in [top_connectors, instance_roofs]:
            if inst in d:
                d[inst] = rotate_points(*d[inst], cx, cy, theta)
        for i in range(num_channels):
            if (inst, i) in metal_slanted:
                metal_slanted[(inst, i)] = rotate_points(*metal_slanted[(inst, i)], cx, cy, theta)
        rx, ry = rotate_points([roof_top_left[inst][0]], [roof_top_left[inst][1]], cx, cy, theta)
        roof_top_left[inst] = (rx[0], ry[0])
        rx, ry = rotate_points([roof_top_right[inst][0]], [roof_top_right[inst][1]], cx, cy, theta)
        roof_top_right[inst] = (rx[0], ry[0])

    hull_cache = [None] * num_instances
    dirty      = [True]  * num_instances   

    def _get_hull(inst):
        if dirty[inst]:
            hull_cache[inst] = get_instance_hull(inst)
            dirty[inst] = False
        return hull_cache[inst]


    _t0 = time.perf_counter()
    _collision_iters = 0
    for it in range(max_iterations):
        if it % 100 == 0 and it > 0:
            print(f"  [collision] iter {it}  (+{time.perf_counter()-_t0:.2f}s)")
        rotate_dir = [0.0] * num_instances
        for k in range(num_instances):
            if k > 0 and polygons_intersect(_get_hull(k), _get_hull(k-1)):
                rotate_dir[k] -= 1.0
            if k < num_instances - 1 and polygons_intersect(_get_hull(k), _get_hull(k+1)):
                rotate_dir[k] += 1.0

        if all(abs(d) < 1e-9 for d in rotate_dir):
            _collision_iters = it
            print(f"  [collision] converged after {it} iters")
            break

        for inst in range(num_instances):
            if abs(rotate_dir[inst]) > 1e-9:
                rotate_instance(inst, rotate_dir[inst] * angle_step)
                dirty[inst] = True
    else:
        _collision_iters = max_iterations

    _t_collision = time.perf_counter() - _t0
    _timing_log.append((f"Collision solver ({_collision_iters} iters)", _t_collision))
    print(f"  [timing] Collision solver ({_collision_iters} iters): {_t_collision:.3f}s")

    # Combined top structure 

    _t0 = time.perf_counter()

    total_channels = num_instances * num_channels

    xL0, yL0 = roof_top_left[0]
    xRn, yRn = roof_top_right[num_instances - 1]
    rect_top_y = max(yL0, yRn) + top_rect_height

    y_transition_top = max(yL0, yRn) + inverted_roof_height

    wire_pitch = wire_width_scale2 * wire_width + wire_width_scale * wire_width


    x_center_combined = 0.5 * (xL0 + xRn)
    x_out_positions = [
        x_center_combined - 0.5 * (total_channels - 1) * wire_pitch + rank * wire_pitch
        for rank in range(total_channels)
    ]

    # Combined wire ordering 

    def _slant_top_x(pair):
        poly_x = metal_slanted[pair][0]
        return 0.5 * (poly_x[2] + poly_x[3])

    all_pairs_sorted = sorted(
        [(inst, i) for inst in range(num_instances) for i in range(num_channels)],
        key=_slant_top_x
    )

    # Rank range covered by each instance 

    inst_rank_range = {}
    for rank, (inst, i) in enumerate(all_pairs_sorted):
        if inst not in inst_rank_range:
            inst_rank_range[inst] = [rank, rank]
        else:
            inst_rank_range[inst][1] = rank

    w1 = wire_width_scale * wire_width
    inv_roof_margin = w1 / 2 + padding2

    # Per-instance inverted roofs 

    inverted_roofs = {}
    for inst in range(num_instances):
        xbL, ybL = roof_top_left[inst]
        xbR, ybR = roof_top_right[inst]
        r0, r1   = inst_rank_range[inst]
        xtL = x_out_positions[r0] - inv_roof_margin
        xtR = x_out_positions[r1] + inv_roof_margin
        inverted_roofs[inst] = (
            [xbL, xbR, xtR, xtL, xbL],
            [ybL, ybR, y_transition_top, y_transition_top, ybL],
        )
        plot_polyline(inverted_roofs[inst], 'k')

    # Top rectangle 

    x_rect_left  = x_out_positions[0]  - inv_roof_margin
    x_rect_right = x_out_positions[-1] + inv_roof_margin
    combined_top_rect = create_rectangle(x_rect_left, y_transition_top,
                                        x_rect_right - x_rect_left, rect_top_y - y_transition_top)
    plot_rectangle(combined_top_rect, 'k')

    # Second roof + second top rectangle

    top_xs, top_ys = combined_top_rect
    top_left   = min(top_xs[0], top_xs[3])
    top_right  = max(top_xs[1], top_xs[2])
    baseline_y = max(top_ys)

    beta = roof_angle2
    a = wire_width * wire_width_scale
    b = wire_width * wire_width_scale2
    inner_vertical_extension = wire_width_scale * wire_width * (b/a-1) * np.tan(beta)
    slanted_wire_width        = b* np.sin(beta)
    center_rank  = (total_channels - 1) / 2.0

    
    desired_slant_spacing = wire_width_scale2 * wire_width+ slanted_wire_width

    delta_y_step = wire_pitch *(np.tan(beta))*((desired_slant_spacing)/(wire_pitch*np.sin(beta))-1)

    wire_extension = solder_pad_offset + 10 + center_rank * delta_y_step
    print(wire_extension)
    second_top_rect_width  = abs(top_left - top_right) + stack_w + 2 * wire_extension / np.tan(beta)
    second_top_rect_height = total_channels * (stack_h + stack_spacing) + 600

    left_edge  = top_left  + second_inset
    right_edge = top_right - second_inset
    if left_edge >= right_edge:
        left_edge = top_left; right_edge = top_right

    if second_roof_height_offset is not None:
        second_top_y = baseline_y + second_roof_height_offset
    else:
        second_top_y = baseline_y + np.tan(roof_angle2) * (second_top_rect_width - abs(top_left - top_right)) / 2

    sec_w = float(second_top_rect_width)
    base_center = 0.5 * (left_edge + right_edge)
    x1L_second  = base_center - sec_w / 2.0
    x1R_second  = base_center + sec_w / 2.0

    second_roof = ([left_edge, x1L_second, x1R_second, right_edge],
                    [baseline_y, second_top_y, second_top_y, baseline_y])
    plot_polyline(second_roof, 'm')

    sec_x0 = x1L_second
    sec_h  = second_top_rect_height
    if sec_w <= 0:
        sec_w = right_edge - left_edge; sec_x0 = left_edge

    second_top_rect = create_rectangle(sec_x0, second_top_y, sec_w, sec_h)
    plot_rectangle(second_top_rect, 'm')

    _timing_log.append(("Combined top rectangle + second roof", time.perf_counter() - _t0))
    print(f"  [timing] Combined top rect + second roof: {_timing_log[-1][1]:.3f}s")

    # Transition + vertical wires:

    metal_transition = {}

    for rank, (inst, i) in enumerate(all_pairs_sorted):
        poly_x, poly_y = metal_slanted[(inst, i)]
        x2, y2 = poly_x[2], poly_y[2]
        x3, y3 = poly_x[3], poly_y[3]
        x_out  = x_out_positions[rank]

        t_poly_x = [x3, x2, x_out + w1/2, x_out - w1/2, x3]
        t_poly_y = [y3, y2, y_transition_top, y_transition_top, y3]
        metal_transition[(inst, i)] = (t_poly_x, t_poly_y)
        plot_polyline((t_poly_x, t_poly_y), 'b')

        v_poly_x = [x_out - w1/2, x_out + w1/2, x_out + w1/2, x_out - w1/2, x_out - w1/2]
        v_poly_y = [y_transition_top, y_transition_top, rect_top_y, rect_top_y, y_transition_top]
        metal_vertical[(inst, i)] = (v_poly_x, v_poly_y)
        plot_polyline((v_poly_x, v_poly_y), 'b')

    # Metal stacks

    metal_stack_rects = {}
    stack_pitch = stack_h + stack_spacing

    def add_stack_in_rectangle(rect, prefix, count):
        xs, ys     = rect
        top_width  = xs[1] - xs[0]
        top_height = ys[3] - ys[0]
        x_left     = xs[0] + top_width / 2 - stack_w / 2
        y_start    = ys[0]

        for k in range(count):
            y_bottom = y_start + k * stack_pitch
            if y_bottom + stack_h > ys[0] + top_height:
                break
            rect_x = [x_left, x_left + stack_w, x_left + stack_w, x_left]
            rect_y = [y_bottom, y_bottom, y_bottom + stack_h, y_bottom + stack_h]
            metal_stack_rects[(prefix, k)] = (rect_x, rect_y)
            plot_rectangle((rect_x, rect_y), 'c')

    add_stack_in_rectangle(second_top_rect, "top2", total_channels)

    # Vertical wire extensions into the second roof 

    metal_vertical_extended = {}

    s_rx, s_ry  = second_roof
    s_base_y    = min(s_ry);  s_apex_y = max(s_ry)
    apex_x_left = s_rx[1];   apex_x_right = s_rx[2]
    apex_cx     = 0.5 * (apex_x_left + apex_x_right)

    def _slant(bx, ax):
        dx = ax - bx; dy = s_apex_y - s_base_y
        L  = np.hypot(dx, dy)
        return dx / L, dy / L

    l_ux, l_uy = _slant(s_rx[0], apex_x_left)
    r_ux, r_uy = _slant(s_rx[3], apex_x_right)

    middle_rank = int(total_channels / 2) - 1

    for rank, (inst, i) in enumerate(all_pairs_sorted):
        v_poly_x, v_poly_y = metal_vertical[(inst, i)]
        xs_base = 0.5 * (v_poly_x[0] + v_poly_x[1])

        dist_from_center = abs(rank - center_rank) / center_rank if center_rank > 0 else 0.0
        is_middle        = (rank == middle_rank)

        if is_middle:
            poly_x = [xs_base - w1/2, xs_base + w1/2,
                    xs_base + w1/2,  xs_base - w1/2, xs_base - w1/2]
            poly_y = [s_base_y, s_base_y, s_apex_y, s_apex_y, s_base_y]
        else:
            extension = solder_pad_offset + (1.0 - dist_from_center)*(wire_extension - solder_pad_offset - 10)

            y_knee    = s_base_y + extension

            if xs_base < apex_cx:
                su_x, su_y = l_ux, l_uy
                x_inner    = xs_base + w1/2
                x_outer    = xs_base - w1/2
            else:
                su_x, su_y = r_ux, r_uy
                x_inner    = xs_base - w1/2
                x_outer    = xs_base + w1/2

            y_knee_outer = y_knee
            y_knee_inner = y_knee + inner_vertical_extension

            slant_t_outer = (s_apex_y - y_knee_outer) / su_y if abs(su_y) > 1e-9 else 0.0
            slant_t_inner = (s_apex_y - y_knee_inner) / su_y if abs(su_y) > 1e-9 else 0.0

            x_top_outer = x_outer + slant_t_outer * su_x
            y_top_outer = y_knee_outer + slant_t_outer * su_y

            x_top_inner = x_inner + slant_t_inner * su_x
            y_top_inner = y_knee_inner + slant_t_inner * su_y

            if xs_base < apex_cx:
                poly_x = [x_outer, x_inner, x_inner, x_top_inner, x_top_outer, x_outer, x_outer]
                poly_y = [s_base_y, s_base_y, y_knee_inner, y_top_inner,
                        y_top_outer, y_knee_outer, s_base_y]
            else:
                poly_x = [x_inner, x_outer, x_outer, x_top_outer, x_top_inner, x_inner, x_inner]
                poly_y = [s_base_y, s_base_y, y_knee_outer, y_top_outer,
                        y_top_inner, y_knee_inner, s_base_y]

        metal_vertical_extended[(inst, i)] = (poly_x, poly_y)
        plot_polyline((poly_x, poly_y), 'b')

    # Vertical wire extensions above the second roof 

    metal_vertical_above_roof = {}

    stack_ys = sorted(
        0.5 * (min(metal_stack_rects[("top2", k)][1]) +
            max(metal_stack_rects[("top2", k)][1]))
        for k in range(total_channels)
        if ("top2", k) in metal_stack_rects
    )

    non_mid  = sorted([r for r in range(total_channels) if r != middle_rank],
                    key=lambda r: abs(r - center_rank))
    avail_si = list(range(1, len(stack_ys)))

    rank_to_stack_y = {}
    for wo, rank in enumerate(non_mid):
        rank_to_stack_y[rank] = stack_ys[avail_si[wo]] if wo < len(avail_si) else stack_ys[-1]

    for rank, (inst, i) in enumerate(all_pairs_sorted):
        if (inst, i) not in metal_vertical_extended:
            continue

        ppx, ppy  = metal_vertical_extended[(inst, i)]
        is_middle = (rank == middle_rank)

        if is_middle:
            x_top_l  = ppx[3]; x_top_r = ppx[2]
            y_top    = max(ppy)
            target_y = stack_ys[0] if stack_ys else y_top
        else:
            x_top_r  = ppx[3]; x_top_l = ppx[4]
            y_top    = max(ppy)
            target_y = rank_to_stack_y.get(rank, stack_ys[-1] if stack_ys else y_top)

        if target_y <= y_top:
            continue

        metal_vertical_above_roof[(inst, i)] = (
            [x_top_l, x_top_r, x_top_r, x_top_l, x_top_l],
            [y_top, y_top,
            target_y + wire_width*wire_width_scale3/2,
            target_y + wire_width*wire_width_scale3/2, y_top]
        )
        plot_polyline(metal_vertical_above_roof[(inst, i)], 'b')

    # Horizontal wire extensions 

    metal_horizontal_to_stack = {}

    w1h = wire_width_scale3 * wire_width

    stack_info = sorted(
        [
            (0.5*(min(metal_stack_rects[("top2",k)][1])+max(metal_stack_rects[("top2",k)][1])),
            0.5*(min(metal_stack_rects[("top2",k)][0])+max(metal_stack_rects[("top2",k)][0])),
            min(metal_stack_rects[("top2",k)][0]),
            max(metal_stack_rects[("top2",k)][0]))
            for k in range(total_channels)
            if ("top2",k) in metal_stack_rects
        ],
        key=lambda s: s[0]
    )

    avail_si = list(range(1, len(stack_info)))

    rank_to_si = {}
    for wo, rank in enumerate(non_mid):
        rank_to_si[rank] = stack_info[avail_si[wo]] if wo < len(avail_si) else stack_info[-1]
    if stack_info:
        rank_to_si[middle_rank] = stack_info[0]

    for rank, (inst, i) in enumerate(all_pairs_sorted):
        if (inst, i) not in metal_vertical_above_roof:
            continue

        ppx, ppy      = metal_vertical_above_roof[(inst, i)]
        y_top         = max(ppy) - wire_width * wire_width_scale3 / 2
        x_wire_center = 0.5 * (min(ppx) + max(ppx))

        if rank not in rank_to_si:
            continue
        s_y, s_xc, s_xl, s_xr = rank_to_si[rank]
        x_end = s_xl if x_wire_center <= s_xc else s_xr

        metal_horizontal_to_stack[(inst, i)] = (
            [x_wire_center, x_end,   x_end,   x_wire_center, x_wire_center],
            [y_top - w1h/2,  y_top - w1h/2, y_top + w1h/2, y_top + w1h/2, y_top - w1h/2]
        )
        plot_polyline(metal_horizontal_to_stack[(inst, i)], 'b')

    # Polyimide stack rectangles

    polyimide_stack_rects = {}
    stack_poly_w = 800
    stack_poly_h = 100

    for (prefix, k), rect in metal_stack_rects.items():
        rx, ry   = rect
        x_center = 0.5 * (min(rx) + max(rx))
        y_center = 0.5 * (min(ry) + max(ry))
        r = create_rectangle(x_center - stack_poly_w/2, y_center - stack_poly_h/2,
                            stack_poly_w, stack_poly_h)
        polyimide_stack_rects[(prefix, k)] = r
        plot_rectangle(r, 'g')

    _timing_log.append(("Combined wiring/stacks/extensions", time.perf_counter() - _t0))
    print(f"  [timing] Combined wiring/stacks/extensions: {_timing_log[-1][1]:.3f}s")

    # Silicon layer

    loop_cx, loop_cy = [], []

    for inst in range(num_instances):
        cx, cy = donut_centers[inst]
        loop_cx.append(cx); loop_cy.append(cy)
        plot_polyline(create_circle(cx, cy, silicon_circle_radius), 'y')

    center_x = 0.5 * (min(loop_cx) + max(loop_cx))
    center_y = 0.5 * (min(loop_cy) + max(loop_cy))
    left_x   = center_x - silicon_rect_width / 2
    right_x  = center_x + silicon_rect_width / 2
    top_y    = center_y + silicon_rect_height / 2
    bot_y    = center_y - silicon_rect_height / 2
    side_r   = silicon_rect_height / 2
    bottom_r = silicon_bottom_diameter / 2

    sil_x, sil_y = [], []
    def _add(px, py): sil_x.extend(px); sil_y.extend(py)

    _add([left_x, right_x], [top_y, top_y])
    rx, ry = create_semicircle(right_x, center_y, side_r, np.pi/2, -np.pi/2)
    _add(rx[1:], ry[1:])
    _add([center_x + bottom_r], [bot_y])
    bx, by = create_semicircle(center_x, bot_y, bottom_r, 0.0, -np.pi)
    _add(bx[1:], by[1:])
    _add([left_x], [bot_y])
    lx, ly = create_semicircle(left_x, center_y, side_r, 3*np.pi/2, np.pi/2)
    _add(lx[1:], ly[1:])
    plot_polyline((np.append(sil_x, sil_x[0]), np.append(sil_y, sil_y[0])), 'y')

    silicon_filled = ShapelyPolygon(list(zip(sil_x, sil_y)))
    if not silicon_filled.is_valid:
        silicon_filled = silicon_filled.buffer(0)
    silicon_outer = (silicon_filled
                    .buffer(silicon_outline_offset + silicon_outline_smooth, resolution=64)
                    .buffer(-silicon_outline_smooth, resolution=64))

    def _to_shapely(xs, ys):
        coords = list(zip(xs, ys))
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if len(coords) < 3:
            return None
        try:
            p = ShapelyPolygon(coords)
            if not p.is_valid:
                p = p.buffer(0)
            return None if p.is_empty else p
        except Exception:
            return None

    _shapely_cache = {}

    def _cached(name, key, xs, ys):
        k = (name, key)
        if k not in _shapely_cache:
            _shapely_cache[k] = _to_shapely(xs, ys)
        return _shapely_cache[k]

    # Merge polyimide shapes

    _t0 = time.perf_counter()
    _pim_polys    = []
    _anchor_polys = []

    for inst in range(num_instances):
        for i in range(num_channels):
            key = (inst, i)
            _pim_polys.append(_cached('polyimide_pads',        key, *polyimide_pads[key]))
            _pim_polys.append(_cached('polyimide_wires',       key, *polyimide_wires[key]))
            _pim_polys.append(_cached('polyimide_wires_below', key, *polyimide_wires_below[key]))
            if key in polyimide_donuts:
                p = _cached('polyimide_donuts', key, *polyimide_donuts[key])
                if p is not None and not p.is_empty:
                    _anchor_polys.append(p)
        if inst in top_connectors:
            _pim_polys.append(_cached('top_connectors', inst, *top_connectors[inst]))
        if inst in instance_roofs:
            _pim_polys.append(_cached('instance_roofs', inst, *instance_roofs[inst]))

    # Combined top structures
    for inst in range(num_instances):
        _pim_polys.append(_cached('inverted_roof', inst, *inverted_roofs[inst]))
    _pim_polys.append(_cached('combined_top_rect', 0, *combined_top_rect))
    _pim_polys.append(_cached('second_roof', 0, *second_roof))
    _pim_polys.append(_cached('second_top_rect', 0, *second_top_rect))

    _pim_polys      = [p for p in _pim_polys if p is not None and not p.is_empty]
    polyimide_merged = unary_union(_pim_polys)
    _timing_log.append(("Polyimide union", time.perf_counter() - _t0))
    print(f"  [timing] Polyimide union: {_timing_log[-1][1]:.3f}s  ({polyimide_merged.geom_type})")

    # Round polyimide corners

    _t0 = time.perf_counter()
    if ROUND_POLYIMIDE:
        rounded_pim = _round_multi_or_poly(polyimide_merged, CORNER_RADII)
    else:
        rounded_pim = polyimide_merged

    if _anchor_polys:
        rounded_pim = unary_union([rounded_pim] + _anchor_polys)

    if rounded_pim.geom_type == 'Polygon':
        _pim_filled = ShapelyPolygon(rounded_pim.exterior)
    else:
        _pim_filled = unary_union([ShapelyPolygon(g.exterior) for g in rounded_pim.geoms])

    global_outer = (_pim_filled
                    .buffer(outline_offset + outline_smooth, resolution=64)
                    .buffer(-outline_smooth, resolution=64))
    outline_ring = global_outer.difference(_pim_filled)
    _timing_log.append(("Polyimide rounding + outline buffer", time.perf_counter() - _t0))
    print(f"  [timing] Polyimide rounding + outline: {_timing_log[-1][1]:.3f}s  ({outline_ring.geom_type})")

    # Merge metal shapes 

    _t0 = time.perf_counter()
    _TOUCH_EPS = 0.01

    all_metal_shapes = []

    for inst in range(num_instances):
        for i in range(num_channels):
            key = (inst, i)
            for dname, d in [('electrode_wires', electrode_wires),
                            ('electrode_pads',  electrode_pads),
                            ('metal_slanted',   metal_slanted),
                            ('metal_transition', metal_transition),
                            ('metal_vertical',  metal_vertical),
                            ('metal_vertical_extended',    metal_vertical_extended),
                            ('metal_vertical_above_roof',  metal_vertical_above_roof),
                            ('metal_horizontal_to_stack',  metal_horizontal_to_stack)]:
                if key in d:
                    p = _cached(dname, key, *d[key])
                    if p is not None:
                        all_metal_shapes.append((p, key))

    for (prefix, k), rect in metal_stack_rects.items():
        p = _cached('metal_stack_rects', (prefix, k), *rect)
        if p is not None:
            all_metal_shapes.append((p, (prefix, k)))

    n      = len(all_metal_shapes)
    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def _union_uf(a, b):
        a, b = _find(a), _find(b)
        if a != b:
            parent[b] = a

    geom_list     = [s for s, _ in all_metal_shapes]
    geom_list_buf = [s.buffer(_TOUCH_EPS) for s in geom_list]
    tree          = STRtree(geom_list_buf)

    try:
        pairs = tree.query_pairs(geom_list_buf)          # Shapely >= 2.0 API
        for i, j in pairs:
            if not geom_list_buf[i].disjoint(geom_list_buf[j]):
                _union_uf(i, j)
    except (AttributeError, TypeError):
        for i, si_buf in enumerate(geom_list_buf):
            candidates = tree.query(si_buf)
            for j in candidates:
                if j <= i:
                    continue
                if not geom_list_buf[i].disjoint(geom_list_buf[j]):
                    _union_uf(i, j)

    groups = defaultdict(list)
    for idx in range(n):
        groups[_find(idx)].append(idx)

    print(f"  [metal] connected groups: {len(groups)}")

    merged_metal_groups = {}
    _n_groups = len(groups)
    for _gi, (root, indices) in enumerate(groups.items()):
        if _gi % 10 == 0:
            print(f"  [metal union] group {_gi+1}/{_n_groups}  (+{time.perf_counter()-_t0:.2f}s)")
        polys  = [geom_list[idx] for idx in indices]
        merged = (unary_union(polys)
                .buffer(0.05).buffer(-0.05)
                .simplify(0.05, preserve_topology=True))
        if not merged.is_empty:
            merged_metal_groups[root] = merged

    _timing_log.append((f"Metal union ({len(merged_metal_groups)} bodies)", time.perf_counter() - _t0))
    print(f"  [timing] Metal union ({len(merged_metal_groups)} bodies): {_timing_log[-1][1]:.3f}s")

    # Optional: round merged metal bodies

    _t0 = time.perf_counter()
    if ROUND_METAL:
        rounded_metal_groups = {}
        _n_metal = len(merged_metal_groups)
        for _mi, (root, geom) in enumerate(merged_metal_groups.items()):
            if _mi % 10 == 0:
                print(f"  [metal rounding] body {_mi+1}/{_n_metal}  (+{time.perf_counter()-_t0:.2f}s)")
            rounded_metal_groups[root] = _round_multi_or_poly(geom, CORNER_RADII)
    else:
        rounded_metal_groups = merged_metal_groups

    _timing_log.append(("Metal corner rounding", time.perf_counter() - _t0))
    print(f"  [timing] Metal corner rounding: {_timing_log[-1][1]:.3f}s")

    # Optional: round contact pad 

    def _round_xy_poly(xs, ys):
        p = _to_shapely(xs, ys)
        if p is None:
            return None
        return _round_shapely_polygon(p, CORNER_RADII)

    # DXF export

    _t0 = time.perf_counter()
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    doc.layers.new(name='Metal',      dxfattribs={'linetype': 'CONTINUOUS', 'color': 4})
    doc.layers.new(name='Polyimide',  dxfattribs={'linetype': 'CONTINUOUS', 'color': 3})
    doc.layers.new(name='Electrodes', dxfattribs={'linetype': 'CONTINUOUS', 'color': 2})
    doc.layers.new(name='Silicon',    dxfattribs={'linetype': 'CONTINUOUS', 'color': 8})
    doc.layers.new(name='Outline',    dxfattribs={'linetype': 'CONTINUOUS', 'color': 1})
    doc.layers.new(name='Aluminium',  dxfattribs={'linetype': 'CONTINUOUS', 'color': 5})

    def export_xy_closed(xs, ys, layer):
        coords = [(float(x), float(y)) for x, y in zip(xs, ys)]
        msp.add_lwpolyline(coords, dxfattribs={'layer': layer, 'closed': True})

    def _find_bridge(hole_pts, ext_pts, poly, top_k=20):
        """Shortest valid bridge from any hole vertex to any exterior vertex:
        vectorised H×E distance matrix, test the top_k nearest pairs for
        containment, else fall back to the nearest pair unconditionally."""
        dx   = hole_pts[:, 0:1] - ext_pts[:, 0]   
        dy   = hole_pts[:, 1:2] - ext_pts[:, 1]
        dist = dx*dx + dy*dy                       

        flat_idx = np.argpartition(dist.ravel(), min(top_k, dist.size) - 1)
        flat_idx = flat_idx[:min(top_k, dist.size)]
        flat_idx = flat_idx[np.argsort(dist.ravel()[flat_idx])]  

        for fi in flat_idx:
            j, i = divmod(int(fi), ext_pts.shape[0])
            bridge = LineString([tuple(hole_pts[j]), tuple(ext_pts[i])])
            if poly.contains(bridge) or poly.covers(bridge):
                return j, i

        j, i = divmod(int(np.argmin(dist)), ext_pts.shape[0])
        return j, i


    def export_shapely_with_holes_as_single_polygon(poly, layer):
        if poly is None or poly.is_empty:
            return
        if poly.geom_type == 'MultiPolygon':
            for geom in poly.geoms:
                export_shapely_with_holes_as_single_polygon(geom, layer)
            return
        if poly.geom_type != 'Polygon':
            return

        poly = orient(poly, sign=1.0)
        ext_coords = list(poly.exterior.coords)

        if not list(poly.interiors):
            msp.add_lwpolyline(
                [(float(x), float(y)) for x, y in ext_coords],
                dxfattribs={'layer': layer, 'closed': True})
            return

        sorted_interiors = sorted(
            poly.interiors,
            key=lambda ring: -max(y for x, y in ring.coords)
        )

        current_ring = ext_coords

        for interior in sorted_interiors:
            hole_coords = list(interior.coords)
            hole_pts    = np.array(hole_coords[:-1])
            ext_pts     = np.array(current_ring[:-1])

            best_j, best_i = _find_bridge(hole_pts, ext_pts, poly)

            best_hole_pt = tuple(hole_pts[best_j])
            best_ext_pt  = tuple(ext_pts[best_i])
            hole_rotated = hole_coords[best_j:] + hole_coords[:best_j + 1]

            current_ring = (
                current_ring[:best_i + 1] +
                [best_ext_pt, best_hole_pt] +
                hole_rotated +
                [best_hole_pt, best_ext_pt] +
                current_ring[best_i + 1:]
            )

        msp.add_lwpolyline(
            [(float(x), float(y)) for x, y in current_ring],
            dxfattribs={'layer': layer, 'closed': True})

    def export_shapely_closed(poly, layer):
        export_shapely_with_holes_as_single_polygon(poly, layer)

    # Outline
    print(f"  [export] Outline  (+{time.perf_counter()-_t0:.2f}s)")
    if outline_ring is not None and not outline_ring.is_empty:
        export_shapely_closed(global_outer.union(silicon_outer).difference(_pim_filled), 'Outline')

    # Metal (optionally rounded)
    print(f"  [export] Metal ({len(rounded_metal_groups)} bodies)  (+{time.perf_counter()-_t0:.2f}s)")
    for root, merged in rounded_metal_groups.items():
        export_shapely_closed(merged, 'Metal')

    # Electrodes (contact pads, optionally rounded)
    print(f"  [export] Electrodes  (+{time.perf_counter()-_t0:.2f}s)")
    for inst in range(num_instances):
        for i in range(num_channels):
            if (inst, i) in contact_pads:
                if ROUND_ELECTRODES:
                    p = _round_xy_poly(*contact_pads[(inst, i)])
                    if p is not None:
                        export_shapely_closed(p, 'Electrodes')
                        continue
                export_xy_closed(*contact_pads[(inst, i)], 'Electrodes')

    # Polyimide (optionally rounded)
    print(f"  [export] Polyimide  (+{time.perf_counter()-_t0:.2f}s)")
    _polyimide_layer_polys = []
    if isinstance(rounded_pim, ShapelyMultiPolygon):
        for geom in rounded_pim.geoms:
            export_shapely_closed(geom, 'Polyimide')
            _polyimide_layer_polys.append(geom)
    else:
        export_shapely_closed(rounded_pim, 'Polyimide')
        _polyimide_layer_polys.append(rounded_pim)

    # Second top rectangle 
    p = _cached('second_top_rect_pim', 0, *second_top_rect)
    if p is not None:
        if ROUND_POLYIMIDE:
            p = _round_shapely_polygon(p, CORNER_RADII)
        export_shapely_closed(p, 'Polyimide')
        _polyimide_layer_polys.append(p)

    for key, rect in polyimide_stack_rects.items():
        p = _cached('polyimide_stack_rects', key, *rect)
        if p is not None:
            if ROUND_POLYIMIDE:
                export_shapely_closed(_round_shapely_polygon(p, CORNER_RADII), 'Outline')
            else:
                export_shapely_closed(p, 'Outline')

    # Anchor markers
    for inst in range(num_instances):
        p = _cached('anchor_markers', inst, *anchor_markers[inst])
        if p is not None:
            export_shapely_closed(p, 'Outline')

    # Silicon 
    print(f"  [export] Silicon  (+{time.perf_counter()-_t0:.2f}s)")
    silicon_ring = silicon_outer.difference(silicon_filled)

    left_tip  = left_x  - side_r
    right_tip = right_x + side_r
    bridge_reach = silicon_outline_offset + silicon_bridge_margin
    left_bridge = create_rectangle(
        left_tip - bridge_reach, center_y - silicon_bridge_width / 2,
        bridge_reach + silicon_bridge_margin, silicon_bridge_width
    )
    right_bridge = create_rectangle(
        right_tip - silicon_bridge_margin, center_y - silicon_bridge_width / 2,
        bridge_reach + silicon_bridge_margin, silicon_bridge_width
    )
    plot_rectangle(left_bridge, 'y')
    plot_rectangle(right_bridge, 'y')

    silicon_bridges = unary_union([
        ShapelyPolygon(zip(*left_bridge)),
        ShapelyPolygon(zip(*right_bridge)),
    ])
    silicon_ring = silicon_ring.difference(silicon_bridges)

    export_shapely_with_holes_as_single_polygon(silicon_ring, 'Silicon')
    _silicon_layer_polys = [silicon_ring]

    for inst in range(num_instances):
        cx_h, cy_h = donut_centers[inst]
        circ_x, circ_y = create_circle(cx_h, cy_h, silicon_circle_radius, 120)
        circle_poly = ShapelyPolygon(list(zip(circ_x, circ_y)))
        if not circle_poly.is_valid:
            circle_poly = circle_poly.buffer(0)
        export_shapely_with_holes_as_single_polygon(circle_poly, 'Silicon')
        _silicon_layer_polys.append(circle_poly)

    # Aluminium
    print(f"  [export] Aluminium  (+{time.perf_counter()-_t0:.2f}s)")
    _polyimide_union = unary_union(_polyimide_layer_polys)
    _silicon_union   = unary_union(_silicon_layer_polys)
    aluminium_overlap = _polyimide_union.intersection(_silicon_union)
    if not aluminium_overlap.is_empty:
        aluminium_shape = aluminium_overlap.buffer(aluminium_margin, resolution=32)
        loop_holes = []
        for inst in range(num_instances):
            p = _cached('anchor_markers', inst, *anchor_markers[inst])
            if p is not None and not p.is_empty:
                loop_holes.append(p)
        if loop_holes:
            aluminium_shape = aluminium_shape.difference(unary_union(loop_holes))
        export_shapely_closed(aluminium_shape, 'Aluminium')

    print(f"  [export] saving DXF  (+{time.perf_counter()-_t0:.2f}s)")
    doc.saveas('electrode_%s.dxf' % Type)
    _timing_log.append(("DXF export", time.perf_counter() - _t0))
    print(f"  [timing] DXF export: {_timing_log[-1][1]:.3f}s")

    plt.axis('equal')
    # plt.show()

    print_timing_summary()


# Presets: "4x Xiaomin 16", "1x Xiaomin 64", "4x Xiaomin 64",
# "4x Xiaomin 64 v2", "50mu pitch", "50mu pitch v2", "32 800 mu span"

TYPES_TO_RUN = [
    "32 800 mu span",
    "1x Xiaomin 64",
    "4x Xiaomin 16",
    "50mu pitch",
]

for Type in TYPES_TO_RUN:
    print(f"\n{'#'*60}\n# Running design: {Type}\n{'#'*60}")
    run_design(Type)
    plt.close('all')   