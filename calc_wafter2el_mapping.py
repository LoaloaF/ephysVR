import os
import sys
import json
import cv2
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata

from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import napari
from itertools import product as iproduct


# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

## ── main functions ─────────────────────────────────────────────────────────────
## 1. detect_interconnect_pads: detect wafer pads from the wafer image, 
# and allow manual adjustment in Napari
def detect_interconnect_pads(path, precomputed=False, save=False):
    fname_base = os.path.basename(path)
    circles_fullfname = f"{path}/detected_pads_{fname_base}.csv"
    if not precomputed:
        full_fname = f'{path}/waferpic_{fname_base}.png'
        if not os.path.exists(full_fname):
            Logger().logger.error(f"Wafer image not found at {full_fname}")
            exit(1)
        image = cv2.imread(f'{path}/waferpic_{fname_base}.png')
        print("Image shape: ", image.shape)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Use the Hough Circle Transform to detect circles
        # smaller wafer pads
        circles = cv2.HoughCircles(blurred, 
                                cv2.HOUGH_GRADIENT, 
                                dp=1.4, 
                                minDist=45, 
                                param1=60, 
                                param2=10, 
                                minRadius=13, 
                                maxRadius=27)
        # large wafer pads
        # circles = cv2.HoughCircles(blurred, 
        #                         cv2.HOUGH_GRADIENT, 
        #                         dp=1.4, 
        #                         minDist=65, 
        #                         param1=50, 
        #                         param2=20, 
        #                         minRadius=17, 
        #                         maxRadius=33)

        circles = np.round(circles[0, :]).astype("int")
        circles = np.array([(y, x) for (x, y, r) in circles])
        print("\n", len(circles), "circle centers detected")
    
        print('Update the circle centers using Napari\n\n')
        viewer = napari.Viewer()
        viewer.add_image(blurred, name='Original Image', rgb=False)
        points_layer = viewer.add_points(circles, size=25, border_color='green', 
                                    face_color='white', name='Circle Centers')
        napari.run()
        
        if save:
            circles = points_layer.data  # This contains the updated coordinates
            np.savetxt(circles_fullfname.replace('.csv', '_new.csv'), circles, 
                       delimiter=',', fmt='%d')
            print(circles_fullfname.replace('.csv', '_new.csv'), "saved")
    else:
        # load the circle centers from the CSV file, adjusted ones
        circles = np.loadtxt(circles_fullfname, delimiter=',', dtype=int)
        print(circles.shape)
    return circles

## 2. shank_ordered_interconnect_pads_fromtrace: BFS flood fill along the snake-trace, 
# capturing wafer pads in order they are routed to polyimide elctrodes
def shank_ordered_interconnect_pads_fromtrace(path, device_name, interconnect_pads, start_point, step_size=10, visualize=True):
    """
    BFS flood fill along a snake-trace, capturing wafer pads in order.
    
    Args:
        path:        path to device folder (same convention as shank_ordered_interconnect_pads)
        interconnect_pads:  (N,2) array of pad coordinates [row, col]
        start_point: (row, col) starting coordinate
        step_size:   BFS expansion steps per "wave" before checking for new pads
        visualize:   show intermediate imshow plots (first 10 steps only) + final HSV order plot
    
    Returns:
        DataFrame with columns [y, x, metal, depth, r, g, b, a], index=pad_id (capture order)
    """

    def detect_pad_trace(fname):
        assert os.path.exists(fname), f"Trace image not found at {fname}"
        trace_image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        return (trace_image > 128).astype(bool)
    
    def detect_shorted_pads(fname):
        assert os.path.exists(fname), f"Shorted pads image not found at {fname}"
        trace_image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        return (trace_image > 128).astype(bool)

    def detect_metallization(fname):
        assert os.path.exists(fname), f"Metalization mask image not found at {fname}"
        image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        metal_mask = ~image.astype(bool)  # True = layer 1, False = layer 2
        return metal_mask
    
    def generate_unique_colors(n: int, seed: int = 42, cmap: str = "hsv",
                           cmap_lims: tuple = (0.0, 0.95)) -> np.ndarray:
        np.random.seed(seed)

        colormap = plt.get_cmap(cmap, n)
        raw = colormap(np.linspace(*cmap_lims, n))
        colors = (raw[:, :3] * 255).round().astype(np.uint8)

        def _make_unique(arr: np.ndarray, max_val: int = 255) -> np.ndarray:
            arr = arr.copy().astype(np.int16)   # int16 to avoid overflow arithmetic

            # Pre-build offsets sorted by L1 distance (nearest first).
            # ±1 on each channel = 6 neighbours at dist 1, then dist 2, etc.
            MAX_SEARCH = 12
            offsets = sorted(
                (o for o in iproduct(range(-MAX_SEARCH, MAX_SEARCH + 1), repeat=3)
                if o != (0, 0, 0)),
                key=lambda o: abs(o[0]) + abs(o[1]) + abs(o[2])
            )

            # "taken" maps color tuple -> index of owner; keepers own their slot.
            taken: dict[tuple, int] = {}
            _, unique_idx = np.unique(arr, axis=0, return_index=True)
            for i in unique_idx:
                taken[tuple(arr[i])] = i

            dup_idx = np.setdiff1d(np.arange(len(arr)), unique_idx)
            print(f"  Duplicates to resolve: {len(dup_idx)}")

            for idx in dup_idx:
                orig = tuple(arr[idx])
                for (dr, dg, db) in offsets:
                    candidate = (
                        int(orig[0]) + dr,
                        int(orig[1]) + dg,
                        int(orig[2]) + db,
                    )
                    if all(0 <= c <= max_val for c in candidate) and candidate not in taken:
                        taken[candidate] = idx
                        arr[idx] = candidate
                        break
                else:
                    raise RuntimeError(f"Could not place color at index {idx} "
                                    f"within search radius {MAX_SEARCH}.")

            return arr.astype(np.uint8)

        return _make_unique(colors)

    # ── helper: check pads newly covered ──────────────────────────────────────
    def collect_new_pads():
        new = []
        for pidx, (r, c) in enumerate(pad_rcs):
            if pidx not in captured and 0 <= r < H and 0 <= c < W and visited[r, c]:
                new.append(pidx)
                captured.add(pidx)
        return new

    # ── helper: one BFS wave of `step_size` pixel expansions ──────────────────
    def bfs_wave(frontier):
        for _ in range(step_size):
            if not frontier:
                break
            next_frontier = deque()
            while frontier:
                r, c = frontier.popleft()
                for dr, dc in neighbours:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and trace_mask[nr, nc]:
                        visited[nr, nc] = True
                        if visualize:
                            vis_canvas[nr, nc] = 128
                        next_frontier.append((nr, nc))
            frontier = next_frontier
        return frontier

    # fname_base = os.path.basename(path)
    trace_mask = detect_pad_trace(f'{path}/routingorder_{device_name}.png')
    shorted_pads_mask = detect_shorted_pads(f'{path}/routingshorts_{device_name}.png')
    metal_mask = detect_metallization(f'{path}/metal_mask_{device_name}.png')

    H, W = trace_mask.shape
    print(f"Trace mask shape: {trace_mask.shape}, metal mask shape: {metal_mask.shape}, shorted pads mask shape: {shorted_pads_mask.shape}")

    # coordinate convention: start_point and interconnect_pads are (row, col)
    sr, sc = start_point
    start_rc = (sr, sc)

    if not trace_mask[start_rc]:
        raise ValueError(f"start_point {start_point} is not on the trace (mask is False there).")

    # pad_rcs: list of (row, col) for each pad
    pad_rcs = [(int(p[0]), int(p[1])) for p in interconnect_pads]

    # Fast lookup: pixel (row,col) -> pad index
    pad_pixel_map = {}
    for idx, (r, c) in enumerate(pad_rcs):
        if 0 <= r < H and 0 <= c < W:
            pad_pixel_map[(r, c)] = idx

    # ── BFS state ──────────────────────────────────────────────────────────────
    visited = np.zeros((H, W), dtype=bool)
    visited[start_rc] = True

    frontier = deque([start_rc])
    ordered_pad_indices = []
    captured = set()

    if visualize:
        vis_canvas = np.zeros((H, W), dtype=np.uint8)
        vis_canvas[start_rc] = 128

    step_num = 0
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity

    # ── main loop ─────────────────────────────────────────────────────────────
    while frontier:
        frontier = bfs_wave(frontier)
        step_num += 1
        new_pads = collect_new_pads()

        if new_pads:
            ordered_pad_indices.extend(new_pads)
            if len(new_pads) > 1:
                print(f"  ⚠  Step {step_num}: captured {len(new_pads)} pads at once "
                      f"(indices {new_pads}). Consider reducing step_size.")

            if visualize and step_num <= 100:
                fig, ax = plt.subplots(figsize=(14, 4))
                ax.imshow(vis_canvas, cmap='gray', origin='upper')
                ax.scatter(interconnect_pads[:, 1], interconnect_pads[:, 0], color='white', s=5, label='All Pads')

                for pidx in ordered_pad_indices[:-len(new_pads)]:
                    r, c = pad_rcs[pidx]
                    ax.scatter(c, r, color='lime', s=20, zorder=3)

                for pidx in new_pads:
                    r, c = pad_rcs[pidx]
                    ax.scatter(c, r, color='red', s=60, zorder=4)
                    ax.annotate(str(len(ordered_pad_indices) - len(new_pads) +
                                    new_pads.index(pidx) + 1),
                                (c, r), color='yellow', fontsize=6, ha='center')

                ax.scatter(sc, sr, color='cyan', s=80, marker='*', zorder=5, label='Start')
                ax.set_title(f"Step {step_num} — pad #{len(ordered_pad_indices)} captured "
                             f"(wafer_pad index {new_pads[0]})")
                ax.legend(fontsize=7)
                plt.tight_layout()
                plt.show()

    # Pads never reached (off-trace or disconnected)
    missed = [i for i in range(len(interconnect_pads)) if i not in captured]
    if missed:
        print(f"⚠  {len(missed)} pad(s) never reached by flood fill: indices {missed}")
    print(f"\nDone. {len(ordered_pad_indices)} pads ordered in {step_num} BFS waves.")

    # ── build output DataFrame (same format as shank_ordered_interconnect_pads) ──────
    n = len(ordered_pad_indices)
    rows_list, chain_head_list, cols_list, metal_list, shorted_list = [], [], [], [], []
    
    prv_metal_i = {1: None, 2: None}
    chain_head  = {1: None, 2: None}  # tracks root of current chain per metal layer
    for i, pidx in enumerate(ordered_pad_indices):
        r, c = pad_rcs[pidx]
        rows_list.append(r)
        cols_list.append(c)
        metal_val = 1 if metal_mask[r, c] else 2
        metal_list.append(metal_val)

        if shorted_pads_mask[r, c]:
            shorted_list.append(prv_metal_i[metal_val])   # previous same-metal pad (linked list)
            chain_head_list.append(chain_head[metal_val]) # root of this chain
            # chain_head does NOT update — we're still in the same chain
        else:
            shorted_list.append(-1)
            chain_head_list.append(-1)        # this pad IS the root, no head needed
            chain_head[metal_val] = i         # start a new chain rooted here
        prv_metal_i[metal_val] = i           # always update previous
    
    # get colors unique over both devices, may be bonded toegther
    colors = generate_unique_colors(n=n*2, seed=42, cmap='hsv', cmap_lims=(0.0, 0.95))
    if device_name == 'S0844pad8shank':
        colors = colors[:n]
        ids = np.arange(n)
    elif device_name == 'S0844pad6shank':
        colors = colors[n:]
        ids = np.arange(n, 2*n)
    
    ordered_interconnect_pads = pd.DataFrame({
        'pad_y':     pd.Series(rows_list,              name='y'),
        'pad_x':     pd.Series(cols_list,              name='x'),
        'pad_id':    pd.Series(ids,                    name='pad_id'),
        'pad_metal': pd.Series(metal_list,             name='metal'),
        'pad_r':     pd.Series(colors[:, 0],  name='r'),
        'pad_g':     pd.Series(colors[:, 1],  name='g'),
        'pad_b':     pd.Series(colors[:, 2],  name='b'),
        'pad_a':     pd.Series(np.ones(n, dtype=int)*255,  name='a'),
        'shorted_to_prv_pad': pd.Series(shorted_list) != -1,
        'shorted_to_pad':     pd.Series(shorted_list),      # points to previous (natural linked list)
        'chain_head_pad':     pd.Series(chain_head_list),   # points to root (-1 if root itself)
    })
    print(ordered_interconnect_pads)
    print(f"N pads: {len(ordered_interconnect_pads)}")
    print(f"Unique non-shorted pads: {(ordered_interconnect_pads['shorted_to_pad'] == -1).sum()}")
    print(f"Shorted pads: {(ordered_interconnect_pads['shorted_to_pad'] != -1).sum()}")
    print("--" * 20, "\n\n")
    
    # important - mirror the x values over midline because devices id bonded upside down
    midline = W // 2
    ordered_interconnect_pads['pad_x'] = midline - (ordered_interconnect_pads['pad_x'] - midline)
    
    # ── final HSV order visualisation ─────────────────────────────────────────
    if visualize and n > 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.imshow(trace_mask, cmap='gray', origin='upper', alpha=0.4)

        if missed:
            for pidx in missed:
                r, c = pad_rcs[pidx]
                ax.scatter(c, r, color='grey', s=10, zorder=2)

        # Pad-coloured dots using assigned RGB values
        pad_rgb = ordered_interconnect_pads[['pad_r', 'pad_g', 'pad_b']].to_numpy(dtype=float) / 255.0
        ax.scatter(cols_list, rows_list, c=pad_rgb, s=30, zorder=3)

        # Metal layer boxes: green=layer1(mask=1), purple=layer2(mask=2)
        metal_arr = np.array(metal_list)
        layer1_cols = [c for c, m in zip(cols_list, metal_arr) if m == 1]
        layer1_rows = [r for r, m in zip(rows_list, metal_arr) if m == 1]
        layer2_cols = [c for c, m in zip(cols_list, metal_arr) if m == 2]
        layer2_rows = [r for r, m in zip(rows_list, metal_arr) if m == 2]

        ax.scatter(layer1_cols, layer1_rows, facecolors='none', edgecolors='green',
                   s=80, linewidths=1.2, zorder=4, label=f'Metal layer 1 (n={len(layer1_cols)})')
        ax.scatter(layer2_cols, layer2_rows, facecolors='none', edgecolors='purple',
                   s=80, linewidths=1.2, zorder=4, label=f'Metal layer 2 (n={len(layer2_cols)})')
        
        # Shorted pads: red crosses
        shorted_pads = ordered_interconnect_pads[(ordered_interconnect_pads['shorted_to_pad'] != -1).values]
        ax.scatter(shorted_pads['pad_x'], shorted_pads['pad_y'],
                   color='k', marker='x', s=50, zorder=5, label=f'Shorted pads (n={(ordered_interconnect_pads["shorted_to_pad"] != -1).sum()})')
        # connect shorted pads with red lines
        for idx, row in shorted_pads.iterrows():
            shorted_idx = row['shorted_to_pad']
            if shorted_idx != -1:
                ax.plot([row['pad_x'], ordered_interconnect_pads.loc[shorted_idx, 'pad_x']],
                        [row['pad_y'], ordered_interconnect_pads.loc[shorted_idx, 'pad_y']],
                        color='k', linewidth=0.8, zorder=4)

        ax.scatter(sc, sr, color='cyan', s=100, marker='*', zorder=5, label='Start')
        ax.set_title(f"Final pad capture order — {n} pads")
        # add annotation for padid between 415 and 425
        # for i in range(426):
        #     r, c = pad_rcs[ordered_pad_indices[i]]
        #     ax.annotate(str(i), (c, r), color='k', fontsize=10, ha='center', zorder=6)
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.show()
    return ordered_interconnect_pads

# helper to load device info json, which has the shank geometries info we need for the next step
def get_device_info(path):
    assert os.path.exists(path), f"Device path not found at {path}"
    fname_base = os.path.basename(path)
    with open(f'{path}/info_{fname_base}.json', 'r') as f:
        device_info = json.load(f, object_pairs_hook=OrderedDict)
    return device_info
        
# 3. connect_pad2polyimide_shanks: use the shank geometry info to connect 
# the ordered wafer pads to the polyimide electrodes,
def connect_pad2polyimide_shanks(interconnect_pads, device_info, visualize=True):
    # get shank routing info, which has the shank_id, shank_name, ionp_pattern, 
    # colormap and routed_order (left to right)
    shank_routing_info = device_info['wafer_view_shanks_down_left2right']
    # make sure shanks are ordered by their routing (wrt interconnect left to right)
    shank_order = sorted(shank_routing_info.items(), 
                        key=lambda x: x[1]['routed_order'])
    shank_order = [item[1]['shank_id'] for item in shank_order]
    shank_routing_info = {str(int(info['shank_id'])): info 
                          for _, info in sorted(shank_routing_info.items(), 
                                        key=lambda x: x[1]['routed_order'])}
    
    from_interc_pad_id = 0
    pad2polyim_el, shank_ionp_pattern = [], []
    for shank_info in shank_routing_info.values():
        shank_id = shank_info['shank_id']
        shank_name = shank_info['shank_name']
        ionp_pattern_str = shank_info['ionp_pattern']
        shank_cmap = shank_info['colormap']
        shank_mirror = shank_info.get('mirror_el_leftright', False)
        
        # load ionp pattern info, and electrode infos
        with open(f'{path}/shanks/{ionp_pattern_str}.json', 'r') as f:
            detailed_shank_info = json.load(f)
        n_electrodes = detailed_shank_info['electrodes']['n_electrodes']
        # get the ionp strip pattern for drawing later
        ionp_strips = detailed_shank_info['ionp_pattern']['strips']
        shank_ionp_pattern.append([
            (st['real_y_um']['start']*-1, st['real_y_um']['end']*-1) for st in ionp_strips.values()
        ])
        
        el_locs = np.array([list(el.values()) for el 
                            in detailed_shank_info['electrodes']['locations_um']])
        # shank can be mirrored left-right, if so, invert x coordinates
        if shank_mirror:
            el_locs[:, 0] *= -1
        # sort based on x, left to right, how they are routed
        el_locs = el_locs[np.argsort(el_locs[:, 0])]
        # depth is inverted, subtract max value
        el_locs[:, 1] -= el_locs[:, 1].max()
        
        # map electrode depth using colormap
        cmap = plt.get_cmap(shank_cmap)
        el_colors = cmap(np.abs(el_locs[:, 1]) / np.abs(el_locs[:, 1]).max())
        
        pads_left = interconnect_pads.loc[from_interc_pad_id:]
        # Simply skip ALL shorted pads — cross-shank shorts are handled by copy-back
        pad_ids = pads_left[~pads_left['shorted_to_prv_pad']].index[:n_electrodes]
        from_interc_pad_id = pad_ids[-1] + 1

        print()
        print(f"Shank {shank_id} ({shank_name}): assigning pads {pad_ids[0]} to {pad_ids[-1]} ")
        print(f"  Pads left for next shanks: {len(interconnect_pads) - from_interc_pad_id}, non-shorted ones: {(interconnect_pads.loc[from_interc_pad_id:,'shorted_to_prv_pad'] == False).sum()}")
        print(f"  Electrodes shape: {el_locs.shape}, pad_ids n: {len(pad_ids)}")
        
        if len(pad_ids) < n_electrodes:
            Logger().logger.warning(f"Not enough pads left to assign for shank {shank_id} ({shank_name}). "
                                    f"Pads needed: {n_electrodes}, pads available: {len(pad_ids)}. "
                                    f"Assuming that these electrodes are not routed!")
            el_locs = el_locs[:len(pad_ids)]
            el_colors = el_colors[:len(pad_ids)]
        
        pad2polyim_el.append(pd.DataFrame(
            {'shank_id': shank_id,
             'shank_name': shank_name,
             'shank_side': np.where(el_locs[:, 0] < 0, 'left', 'right'),
             'el_depth': el_locs[:, 1],
             'el_wafer_x': el_locs[:, 0],
             'el_id': np.arange(el_locs.shape[0]) +shank_id*1000,
             'el_r': el_colors[:, 0],
             'el_g': el_colors[:, 1],
             'el_b': el_colors[:, 2],
            },
            index=pad_ids,
        ).rename_axis('pad_id'))

    # put together the shanks again
    pad2polyim_el = pd.concat(pad2polyim_el, axis=0)
    
    # add the shorted pads too, above just adds the first pad of the two+ shorted ones
    # use it to find the second,third,... one and add a copy of the electrode
    shorted_pads = interconnect_pads.loc[pad2polyim_el.index, 'shorted_to_pad']
    shorted_pads = shorted_pads[shorted_pads != -1]
    # All shorted pads not yet in pad2polyim_el (skipped during assignment, not cross-shank)
    unassigned_shorted = interconnect_pads[
        interconnect_pads['shorted_to_prv_pad'] &
        ~interconnect_pads.index.isin(pad2polyim_el.index)
    ]
    # Their chain_head_pad tells us exactly which electrode assignment to replicate
    missing_entries = pad2polyim_el.loc[unassigned_shorted['chain_head_pad']].copy()
    missing_entries.index = unassigned_shorted.index

    pad2polyim_el = pd.concat([pad2polyim_el, missing_entries], axis=0).sort_index()
    pad2polyim_el.index.name = 'pad_id'
    # join the two based on index (pad_id) fully overlaps
    pad2polyim_el = interconnect_pads.join(pad2polyim_el, how='inner')
    
    if visualize:
        # draw interconnect pads and polyimide electrodes in the same plot to check the mapping, draw with el_color
        plt.figure(figsize=(14, 10))
        plt.scatter(interconnect_pads.loc[pad2polyim_el.index,'pad_x'], 
                    interconnect_pads.loc[pad2polyim_el.index,'pad_y'], 
                    color=pad2polyim_el.apply(lambda row: (row.el_r, row.el_g, row.el_b), axis=1), s=65)
        # add annotation for pad_id between 415 and 425
        plt.gca().invert_yaxis()
        # plt.axis('off')
        plt.axis('equal')
        plt.savefig(f'{path}/wafer_pads.svg')
        plt.show()
        
        shanks = pad2polyim_el['shank_id'].unique()
        n_shanks = len(shanks)
        fig, axs = plt.subplots(1, n_shanks, figsize=(.5 * n_shanks, 3), sharey=True)
        for i, shank_id in enumerate(shanks):
            ax = axs[i]
            shank_df = pad2polyim_el[pad2polyim_el['shank_id'] == shank_id]
            
            xs = (shank_df['el_wafer_x']).values
            ys = shank_df['el_depth'].values
            rs = shank_df['el_r'].values
            gs = shank_df['el_g'].values
            bs = shank_df['el_b'].values

            # ── filled gradient interior ───────────────────────────────────────────
            # dense grid over bounding box
            x_grid, y_grid = np.meshgrid(
                np.linspace(xs.min(), xs.max(), 300),
                np.linspace(ys.min(), ys.max(), 300)
            )
            points = np.column_stack([xs, ys])

            # interpolate each channel onto the grid
            r_grid = griddata(points, rs, (x_grid, y_grid), method='linear')
            g_grid = griddata(points, gs, (x_grid, y_grid), method='linear')
            b_grid = griddata(points, bs, (x_grid, y_grid), method='linear')

            # mask outside convex hull
            hull = ConvexHull(points)
            from matplotlib.path import Path
            hull_path = Path(points[hull.vertices])
            grid_pts  = np.column_stack([x_grid.ravel(), y_grid.ravel()])
            inside    = hull_path.contains_points(grid_pts).reshape(x_grid.shape)

            # build RGBA image, NaN outside hull → transparent
            img = np.ones((*x_grid.shape, 4))
            img[..., 0] = np.where(inside, r_grid, np.nan)
            img[..., 1] = np.where(inside, g_grid, np.nan)
            img[..., 2] = np.where(inside, b_grid, np.nan)
            img[..., 3] = np.where(inside, 1.0,    0.0)
            img = np.clip(img, 0, 1)

            ax.imshow(img,
                    extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                    origin='lower', aspect='auto', zorder=1)

            # ── electrode dots on top ──────────────────────────────────────────────
            ax.scatter(xs, ys,
                    color=shank_df.apply(lambda row: (row.el_r, row.el_g, row.el_b), axis=1),
                    s=10, zorder=2)

            ax.set_title(f"{shank_id}")
            # ax.xaxis.set_visible(False)
            ax.set_yticks(np.array([-2000,-4000,-8000]))
            ax.set_yticklabels(np.array([2,5,8]))
            # y grid
            ax.yaxis.grid(True, which='major', linestyle='--', alpha=0.5)
            ax.set_xlim(-3000, 3000)
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            # turn off spines
            [ax.spines[side].set_visible(False) for side in ['top', 'right', 'bottom', 'left']]
            
            # draw the ionp strip pattern as faint box in gray
            ionp_strips = shank_ionp_pattern[i]
            for start, end in ionp_strips:
                patch = plt.Rectangle((-3000, start), 6000, end-start, facecolor='white', alpha=0.5, 
                                      edgecolor='none', zorder=3)
                ax.add_patch(patch)

        plt.suptitle("Shank")
        plt.ylim(-9000, 0)
        plt.savefig(f'{path}/shank_geometries.svg')
        plt.show()

        
    return pad2polyim_el

# 4. finalize_and_save_pad2el: save the final mapping as csv and png, 
# png is used for alignment of bonding
def finalize_and_save_pad2el(path, wafer_pad2el, device_info, visualize=True):
    print("==="*20)
    print(wafer_pad2el)
    print("==="*20)
    print(wafer_pad2el)
    
    width = 3850
    height = 2100//2
    pad_diam = 16
    canvas = np.zeros([height, width, 3], dtype=np.uint8)
    for i, row in wafer_pad2el.iterrows():
        y, x = int(row['pad_y']), int(row['pad_x'])
        y += 10 # so it fits well on canvas, a constant offsset is irrelevant for alignemnt
        col = np.array((row['pad_r'], row['pad_g'], row['pad_b'])).astype(float)
        cv2.circle(canvas, (x, y), pad_diam, col, -1)
        if visualize:
            offset = 14 if i < 1000 else 19
            cv2.putText(canvas, f"{i}", (x-offset,y+5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (128,128,128,255), 1)

    if visualize:
        viewer = napari.Viewer()
        viewer.add_image(canvas, name='waferpad2electrode')
        napari.run()
    
    # Convert RGBA to BGR before saving
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGRA)
    fname_base = os.path.basename(path) 
    cv2.imwrite(f"{path}/device_mapping_{fname_base}.png", canvas_bgr)
    wafer_pad2el.sort_index().to_csv(f"{path}/device_mapping_{fname_base}.csv")
    print(f"{path}/device_mapping_{fname_base}.csv")
    

if __name__ == "__main__":
    nas_dir = device_paths()[0]
    device_name = "S0844pad8shank" 
    # device_name = "S0844pad6shank" 
    start_point = (85, 855) # the y,x coordinates of the first pad in the routing order, can be found in the routingorder pngs
    path = os.path.join(nas_dir, "devices", "electrode_devices", device_name)

    # detect the pads using cv2.HoughCircles on um-scale image
    detected_interconnect_pads = detect_interconnect_pads(path, precomputed=True, save=False)
    # use the snake-trace and BFS flood fill to order the pads in the routing order, 
    # and also get info on metal layer and shorts    
    interconnect_pads = shank_ordered_interconnect_pads_fromtrace(
                path=path,
                device_name=device_name,
                interconnect_pads=detected_interconnect_pads,   # your (844,2) array
                start_point=start_point,
                step_size=6,
                visualize=False
            )
    
    # # get the device info from the json file with shank info (n electrodes etc)
    device_info = get_device_info(path)
    
    wafer_pad2el = connect_pad2polyimide_shanks(interconnect_pads, device_info, 
                                                visualize=True)
    
    # # # save the final pad2el mapping as csv and png
    finalize_and_save_pad2el(path, wafer_pad2el, device_info, visualize=False)