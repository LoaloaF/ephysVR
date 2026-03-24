import os
import sys
import json
import cv2
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import napari

# to import logger, VR-wide constants and device paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from baseVR.base_logger import CustomLogger as Logger
from baseVR.base_functionality import device_paths

def detect_wafer_pads(path, precomputed=False, save=False):
    fname_base = os.path.basename(path)
    circles_fullfname = f"{path}/detected_pads_{fname_base}.csv"
    if not precomputed:
        image = cv2.imread(f'{path}/waferpic_{fname_base}.png')
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Use the Hough Circle Transform to detect circles
        # smaller wafer pads
        # circles = cv2.HoughCircles(blurred, 
        #                         cv2.HOUGH_GRADIENT, 
        #                         dp=1.4, 
        #                         minDist=45, 
        #                         param1=60, 
        #                         param2=10, 
        #                         minRadius=13, 
        #                         maxRadius=27)
        # large wafer pads
        circles = cv2.HoughCircles(blurred, 
                                cv2.HOUGH_GRADIENT, 
                                dp=1.4, 
                                minDist=65, 
                                param1=50, 
                                param2=20, 
                                minRadius=17, 
                                maxRadius=33)

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


def shank_ordered_wafer_pads(path, wafer_pads, visualize=True):
    def detect_columns(fname):
        # image has has little black ticks on the top of the png
        image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        row_mask = image[0,:] != 255
        print("Found n columns: ", row_mask.sum())
        locs = np.where(row_mask)[0]
        return np.array([0, *locs, image.shape[1]])

    def detect_rows(fname):
        # image has has little black ticks on the left of the png
        image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        row_mask = image[:,0] != 255
        print("Found n rows: ", row_mask.sum())
        ylocs = [0, *np.where(row_mask)[0]]
        return np.array(ylocs)

    def detect_zones(fname):
        # image has has diffferent gray tones in 8 different zones, border are black
        zones_image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        values = np.unique(zones_image)
        print("Found n zones: ", len(values), values)
        
        zone_mask = np.zeros_like(zones_image, dtype=np.int16)
        # go over unique colors in image mas
        for i, value in enumerate(sorted(values)):
            mask_val = i if value != 255 else -1
            zone_mask[zones_image == value] = mask_val
        return zone_mask

    def detect_metallization(fname):
        # image is black and white with black being the metalization layer 1
        image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        metal_mask = ~image.astype(bool) # green metalization layer 1, mask=1 
        return metal_mask                # purple metalization layer 2, mask=0
        
    def get_layout_dims(fname):
        # simply the layout (+mask) image dim, 1 pixel == 1 um
        image = cv2.imread(fname)
        return image.shape[1], image.shape[0]
    
    fname_base = os.path.basename(path)
    width, height = get_layout_dims(f'{path}/waferpic_{fname_base}.png')
    layout_zone_mask = detect_zones(f"{path}/layout_zone_mask_{fname_base}.png")
    metal_mask = detect_metallization(f"{path}/metal_mask_{fname_base}.png")
    columns = detect_columns(f"{path}/columns_{fname_base}.png")
    rows = detect_rows(f"{path}/rows_{fname_base}.png")

    # define the layout zones and how pads are routed in each zone
    layout_zone_map = OrderedDict({0: "bottom_left", 1: "bottom_left_side", # bottom left quadrant
                                   2: "top_left_side", 3: "top_left",     # top left quadrant
                                   4: "bottom_right", 5: "bottom_right_side",  # bottom right quadrant
                                   6: "top_right_side", 7: "top_right"}) # top right quadrant
    layout_zone_col_order = {"bottom_left": -1, "bottom_left_side": 1, # left to right, right to left
                             "top_left_side": -1, "top_left": 1, # left to right, right to left
                             "bottom_right": 1, "bottom_right_side": -1,  # right to left, left to right
                             "top_right_side": 1, "top_right": -1} # right to left, left to right
    
    ordered_wafer_pads = []
    # iterate over zones, columns, rows in the correct orders, and append the corresponding pads
    for zone_id, zone_name in layout_zone_map.items():
        print('--'*20)
        ordered_zone_pads = []
        
        # get the wafer pads in the current zone
        zone_mask = layout_zone_mask == zone_id
        pads_mask = zone_mask[wafer_pads[:, 0], wafer_pads[:, 1]]
        zone_circles = wafer_pads[pads_mask]
        
        # make a list of column starts and stop x-coordinates according to routing in that zome
        if layout_zone_col_order[zone_name] == 1:
            columns_idcs = np.arange(len(columns)-1)
            iter_columns = zip(columns_idcs, zip(columns[:-1], columns[1:]))
        elif layout_zone_col_order[zone_name] == -1:
            columns_idcs = np.arange(len(columns)-1)[::-1]
            iter_columns = zip(columns_idcs, (zip(reversed(columns[:-1]), reversed(columns[1:]))))
        
        # iter over columns in the zone
        for col_i, (from_x, to_x) in iter_columns:
            # get the circles in the current column
            pads_in_col = zone_circles[(zone_circles[:, 1] > from_x) 
                                        & (zone_circles[:, 1] < to_x)]
            
            # reverse the order for everything in top zones (routed bottom-up)
            if 'side' not in zone_name:
                # big blocks top and bottom left and roght aloways count top-down
                sorted_indices = np.argsort(pads_in_col[:, 0])
            else:
                # side blocks always count bottom-up (reverse)
                sorted_indices = np.argsort(pads_in_col[:, 0])[::-1]
            pads_in_col = pads_in_col[sorted_indices]
                
            # get the metalization layer of the pads in the column
            circles_metal_layer = metal_mask[pads_in_col[:, 0], pads_in_col[:, 1]].astype(int)
            circles_metal_layer[circles_metal_layer == 0] = 2
            
            print(f'Zone {zone_name}, Column {col_i},  n={len(pads_in_col)},'
                  f' y_order={np.arange(len(pads_in_col))} metal:{circles_metal_layer}')
            # many columns are empty, skip them
            if len(pads_in_col) > 0:
                # collect characterisitcs of the pads in the column
                n = len(pads_in_col)
                y = pd.Series(pads_in_col[:, 0], name='y')
                x_order = pd.Series(np.full(n, col_i), name='x_order')
                y_order = pd.Series(np.arange(n), name='y_order')
                x = pd.Series(pads_in_col[:, 1], name='x')
                metal = pd.Series(circles_metal_layer, name='metal')
                zone = pd.Series(np.full(n, zone_id), name='zone')
                depth = pd.Series(np.full(n, 1), name='depth')
                r = pd.Series(np.full(n, 1), name='r')
                g = pd.Series(np.full(n, 1), name='g')
                b = pd.Series(np.full(n, 1), name='b')
                a = pd.Series(np.full(n, 1), name='a')
                
                ordered_zone_pads.append(pd.concat([y, x, metal, zone, y_order, 
                                                    x_order, depth, r, g, b, a], 
                                                    axis=1))
        ordered_zone_pads = pd.concat(ordered_zone_pads, axis=0).reset_index(drop=True)
        
        # side zones are not column ordered, but row ordered - fix here
        if 'side' in zone_name:
            iter_rows = zip(reversed(rows[:-1]), reversed(rows[1:]))

            reordered_zone_pads = []
            # sample principle as for columns, but for rows
            for (from_y, to_y) in iter_rows:
                within_row = (ordered_zone_pads.y > from_y) & (ordered_zone_pads.y < to_y)
                print(f"{zone_name} - Row {from_y} to {to_y}, n={within_row.sum()}")
                reordered_zone_pads.append(ordered_zone_pads[within_row])
                
            # replace the column ordered pads with the row ordered pads
            ordered_zone_pads = pd.concat(reordered_zone_pads, axis=0).reset_index(drop=True)
        ordered_wafer_pads.append(ordered_zone_pads)
    # concatenate all the ordered pads in zone order
    ordered_wafer_pads = pd.concat(ordered_wafer_pads, axis=0).reset_index(drop=True)
    ordered_wafer_pads.index.name = 'pad_id'
    print(ordered_wafer_pads.groupby('metal').size())
    print(ordered_wafer_pads)
    print("--"*20, "\n\n")
    
    if visualize:
        # Create n hdv colors for the pads (one for each pad)
        colormap = plt.get_cmap('hsv', len(ordered_wafer_pads))
        colors = [np.array((col[0]*255, col[1]*255, col[2]*255, 255)).round()
                for col in colormap(np.linspace(0, 1, len(ordered_wafer_pads)))]
        
        # Create a canvas to draw the circles on to visualize the pad order
        canvas = np.zeros([height, width, 4], dtype=np.uint8)
        for i, row in ordered_wafer_pads.iterrows():
            y, x = int(row['y']), int(row['x'])
            # draw the pad with its hsv color
            cv2.circle(canvas, (x, y), 20, colors[i].astype(float), -1)
            offset = 14 if i < 1000 else 19
            cv2.putText(canvas, f"{i}", (x-offset,y+5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (235,235,235,255), 1)
            if i != 0:
                # connxect y,x with previous y,x
                cv2.arrowedLine(canvas, (prev_x, prev_y), (x, y), (255,255,255,255), 1)
            prev_x, prev_y = x, y
            
        # draw the full lines of columns and rows, not just indices
        col_row_canvas = np.zeros([height,width,4], dtype=np.uint8)
        for row_i in rows:
            col_row_canvas[row_i-1:row_i+1, :] = 255
        for col_i in columns:
            col_row_canvas[:, col_i-1:col_i+1] = 255

        viewer = napari.Viewer()
        viewer.add_image(metal_mask, colormap='PiYG', opacity=.3,
                         name='metallization layer (Green=1, Purple=2)',)
        viewer.add_image(col_row_canvas, name='Columns and Rows', opacity=.2)
        viewer.add_image(layout_zone_mask, name='Layout Zone', opacity=.25)
        viewer.add_image(canvas, name='Detected Circles')
        napari.run()
    return ordered_wafer_pads


def get_device_info(path):
    fname_base = os.path.basename(path)
    print(f'{path}/info_{fname_base}.json')
    with open(f'{path}/info_{fname_base}.json', 'r') as f:
        device_info = json.load(f)
    print(json.dumps(device_info, indent=4))
    return device_info


def make_unique(colors, max_col):
    while True:
        _, unique_idx = np.unique(colors, axis=0, return_index=True)
        non_unique_idx = np.setdiff1d(np.arange(len(colors)), unique_idx)

        print(f"Unique (of {len(colors)}): ", len(unique_idx), 
               "Non-unique: ", len(non_unique_idx))
        if len(unique_idx) == len(colors):
            print("All unique")
            break

        print("shifting non-unique")
        non_u = colors[non_unique_idx]
        shift_rgb_mask = (non_u!=max_col)
        non_u[shift_rgb_mask] = non_u[shift_rgb_mask] + 1
        colors[non_unique_idx] = non_u
    return colors

def get_unique_grays(n, min_col, max_col):
    grays = np.array([np.arange(min_col, max_col)]*3).T
    if n > grays.shape[0]:
        
        incr_needed = (n // grays.shape[0]) +1
        new_grays = []
        for gr in grays:
            for i in range(incr_needed):
                idx, incr_by = i%3, i//3 +1
                shifted_gr = gr.copy()
                shifted_gr[idx] += incr_by
                new_grays.append(shifted_gr)
        grays = np.array(new_grays)
    return grays[:n]
        
def integr_shank_geometries(wafer_pad2el, device_info):
    def one_side_one_metal_el_distances(start, shank_info, right=False):
        distances = []
        print(f"\nSpacings {"right" if right else "left"} side:\n"
              f"{shank_info['pad_um_center_distances']}")

        # iterate groups of electrodes in the shank with the same spacing
        for group_info in shank_info['pad_um_center_distances']:
            group_n_els = group_info['n_els']
            el_dist = group_info['el_dist']
            group_gap_left = group_info['gap_left']
            group_gap_right = group_info['gap_right']
            
            if right:
                group_top_gap = group_gap_right
            else:
                group_top_gap = group_gap_left
            
            start = distances[-1]+group_top_gap if len(distances) else start
            end = start + (group_n_els//4) * (el_dist*4)

            # append the spacings for the electrodes in the group, *4 because 
            # we only look at one side and one metal
            group_distances = list(np.arange(start, end, el_dist*4))
            print("Group top gap: ", group_top_gap, "Start: ", start, "End: ", 
                  end, 'n:', len(group_distances), ", last one: ", group_distances[-1])
            distances.extend(group_distances)
            
            # excpetion for specifically long shanks, group 2, they need 20 um spacer at the bottom
            if (all((right, group_n_els==72, el_dist==20, group_gap_left==140, group_gap_right==100),) or
               all((not right, group_n_els==72, el_dist==20, group_gap_left==100, group_gap_right==140),)):
                distances[-1] += 20
            
            
        if not right and shank_info['n_electrode_pairs']%2 != 0:
            # append the last el pair if the number of pairs is uneven
            tip_dist = distances[-1]+el_dist*4 if shank_id<3 else distances[-1]+el_dist*2
            print("Append one more for deepest electrode: ", tip_dist)
            distances.append(tip_dist)
        return np.array(distances)
 
 
    shank_infos = device_info['shank_top_wafer_view_left2right']
    shank_order = device_info['shank_routed_order']
        
    from_pad_m1, from_pad_m2 = 0, 0
    # construct a new dataframe with added columns depth, shank, el_pair, color
    updated_wafer_pad2el = []
    # iterate over the shanks in the order they are routed
    for i, shank_id in enumerate(shank_order):
        shank_info = shank_infos[str(int(shank_id))]
        shank_name = shank_info['shank_name']
        n_pairs = shank_info['n_electrode_pairs']
        print(f"\n\nShank {shank_id}: {shank_name}, n fibers: {n_pairs}")
        
        # shank 1 and 2 are on the left, then 3 starts bottom right again 
        # -> therefore reset to half of all pads 
        if shank_id == 3:
            from_pad_m2 = len(wafer_pad2el[wafer_pad2el.metal==2])//2
            from_pad_m1 = len(wafer_pad2el[wafer_pad2el.metal==1])//2
            
        # critical: slice n_pairs (n_fibers) for each metal layer starting
        # where the previous shank left off. The routing order (row order in 
        # dataframe) holds for both metal layers: they are too routed in the 
        # same order. So iloc gives the correct order of pads for each metal
        m1_pads = wafer_pad2el[wafer_pad2el.metal==1].iloc[from_pad_m1:from_pad_m1+n_pairs]
        m2_pads = wafer_pad2el[wafer_pad2el.metal==2].iloc[from_pad_m2:from_pad_m2+n_pairs]

        # assign the electrode pair number to each pad, egual for both metal layers
        m1_pads['el_pair'] = np.arange(n_pairs)
        m2_pads['el_pair'] = np.arange(n_pairs)
        
        # populate columns with constant value
        m1_pads['shank_id'] = shank_info['shank_id']
        m1_pads['shank_name'] = shank_info['shank_name']
        m2_pads['shank_id'] = shank_info['shank_id']
        m2_pads['shank_name'] = shank_info['shank_name']
        
        # first pad can be left or right, depending on the shank
        first_el_dist = shank_info['pad_um_center_distances'][0]['el_dist']
        leftstart = 0 if shank_info['least_deep_electrode_side'] == 'left' else first_el_dist*2
        rightstart = 0 if shank_info['least_deep_electrode_side'] == 'right' else first_el_dist*2


        # calc the spacings between polyimide electrodes along one side of the shank
        left_side_depth = one_side_one_metal_el_distances(leftstart, shank_info)
        print("One metal left n: ", len(left_side_depth))
        # right side is the same as left but reversed and without appending tip for uneven
        right_side_depth = one_side_one_metal_el_distances(rightstart, shank_info, right=True)[::-1]
        print("One metal right n: ", len(right_side_depth))
        left_right_el_pairs = [['left']*len(left_side_depth), ['right']*len(right_side_depth)]
        # upper layer is the metallization starting at depth=0 (first)
        upper_metal_depth = np.append(left_side_depth, right_side_depth)
        print("One metal (upper) n: ", len(upper_metal_depth))


        # the 2nd metallization layer is shifted by the spacings of the polyimide electrodes
        # but spacings differ for each group of electrodes
        print("\nCalculate spacing delta for lower metal layer")
        metal_el_shift_dist = []
        for group_info in shank_info['pad_um_center_distances']:
            group_n_els = group_info['n_els']
            el_dist = group_info['el_dist']
            # append the el_dist n times (n=group_n_els//4), 
            # /4 because we only look at one side one metal
            print(f"{el_dist} um x {group_n_els//4} times")
            metal_el_shift_dist.extend([el_dist]*(group_n_els//4))

        if shank_info['n_electrode_pairs']%2 != 0:
            print("Append one more for deepest electrode")
            tip = [el_dist]
        else:
            tip = []

        twosided_metal_el_shift_dist = np.array((*metal_el_shift_dist, *tip, 
                                                 *metal_el_shift_dist[::-1]))
        # shift the lower metal layer by the spacings of the groups
        lower_metal_depth = upper_metal_depth.copy() + twosided_metal_el_shift_dist
        print("Lower metal n: ", len(lower_metal_depth))

        m1_pads['shank_side'] = np.append(*left_right_el_pairs)
        m2_pads['shank_side'] = np.append(*left_right_el_pairs)
        # concat two metal layers and sort by depth
        if shank_info['least_deep_electrode_metal'] == 1:
            m1_pads['depth'] = upper_metal_depth
            m2_pads['depth'] = lower_metal_depth
        else:
            m2_pads['depth'] = upper_metal_depth
            m1_pads['depth'] = lower_metal_depth
        shank_electrodes = pd.concat([m1_pads, m2_pads], axis=0).sort_values('depth')
        print("---"*20)
        print(shank_electrodes)
        
        updated_wafer_pad2el.append(shank_electrodes)
        # shift the pad layout indices for the next shank
        from_pad_m1 += n_pairs
        from_pad_m2 += n_pairs
    updated_wafer_pad2el = pd.concat(updated_wafer_pad2el, axis=0)
    # add not-connected pads to the dataframe
    updated_wafer_pad2el = pd.concat([updated_wafer_pad2el, 
                                      wafer_pad2el.drop(updated_wafer_pad2el.index)])
    return updated_wafer_pad2el.sort_index()
        
        
def assign_unqiue_colors(wafer_pad2el, device_info, visualize=True,
                         save_visualization=None):
    def draw_shank(ax, shank_electrodes, shank_info):
        shank_name = shank_info['shank_name']
        n_pairs = shank_info['n_electrode_pairs']
        # ax.set_aspect('equal')
        
        # get the electrode pads for each metal layer
        m1_pads = shank_electrodes[shank_electrodes.metal==1].sort_values('el_pair')
        m2_pads = shank_electrodes[shank_electrodes.metal==2].sort_values('el_pair')
        # the unique colors for the pads we assigned before
        m1_colors = m1_pads.loc[:, ['r', 'g', 'b']].values/255
        m2_colors = m1_pads.loc[:, ['r', 'g', 'b']].values/255
        
        # Plot the depth of electrodes on the shank in the assgiend color
        rects = []
        for _, row in m1_pads.iterrows():
            x = row['el_pair']
            y = -row['depth']
            rects.append(plt.Rectangle((x-11/2, y-11/2), 11, 11, zorder=20,
                                       color=row[['r', 'g', 'b']].values/255, lw=1))
            rects.append(plt.Rectangle((x-11, y-11/2), 11/4, 11, zorder=20,
                                       color='green', lw=1) )
            ax.hlines(y, -30, 200, color='black', lw=1, alpha=.2)
        for _, row in m2_pads.iterrows():
            x = row['el_pair']
            y = -row['depth']
            rects.append(plt.Rectangle((x-11/2, y-11/2), 11, 11, zorder=20,
                                       color=row[['r', 'g', 'b']].values/255, lw=1))
            rects.append(plt.Rectangle((x-11, y-11/2), 11/4, 11, zorder=20,
                                       color='purple', lw=1) )
            ax.hlines(y, -30, 200, color='black', lw=1, alpha=.2)
        [ax.add_patch(rect) for rect in rects]
        
        # plot the hook
        hook_depth = shank_electrodes.depth.max() + shank_info['hook_to_deepest_el_distance_um']
        ax.scatter(n_pairs//2, -hook_depth, marker='o', edgecolor='gray', 
                   facecolor='none', alpha=.8, s=40)

        ax.set_xlim(-30, 200)
        ax.set_title(f"Shank {shank_id}:\n{shank_name}")
        ax.set_xticks([])
        # ax.(axis='y', zorder=0)
        [sp.set_visible(False) for sp in ax.spines.values()]

        # anntotions
        annot = f'{shank_info["n_electrode_pairs"]} electrode pairs'
        # number of electrode pairs (metal1+2)
        if shank_info['n_electrode_pairs'] % 2 != 0:
            annot += f' (uneven)'
        else:
            annot += f' (even)'
            
        # for debugging, the spacings between the electrode pairs, both sides, both metals
        spacings = np.unique(np.diff(shank_electrodes.depth), return_counts=True)
        spacings_str = '\n'.join([f"{c}x{s}um" for s, c in zip(*spacings)])
        # Set the x-axis label with the annotation
        annot += f"\nSpacings: {spacings_str}"
        ax.set_xlabel(annot, fontsize=7, loc='left')
        
        if shank_id == 1:
            ax.set_ylabel('Depth [um]')
        if shank_info['least_deep_electrode_side'] == 'left':
            txt = f"Left-Start: metal {shank_info['least_deep_electrode_metal']}"
            ax.text(0, 140, txt, fontsize=7)
        else:
            txt = f"Right-Start: metal {shank_info['least_deep_electrode_metal']}"
            ax.text(n_pairs, 140, txt, ha='right', fontsize=7)
    
    def make_unique_shank_colors(info, visualize=False):
        shank_infos = [(shank_info['shank_id'], shank_info['n_electrodes'], 
                        shank_info['colormap'], shank_info['colormap_lims']) 
                        for shank_info in info.values()]

        all_cols = []
        all_cols_16 = []
        all_cols_8 = []
        for _, (_, n_el, cmap, cmap_lims) in enumerate(shank_infos):
            colormap = plt.get_cmap(cmap, n_el)
            colors = colormap(np.linspace(*cmap_lims, n_el))
            
            # make the colors initial colors
            sc = 255
            colors_uint8 = np.array([(col[0]*sc, col[1]*sc, col[2]*sc)
                                    for col in colors]).round().astype(np.uint8)
            sc = (2**16)-1
            colors_uint16 = np.array([(col[0]*sc, col[1]*sc, col[2]*sc)
                                        for col in colors]).round().astype(np.uint16)
            all_cols.extend(colors)
            all_cols_8.extend(colors_uint8) 
            all_cols_16.extend(colors_uint16)
        
        # original colors in differnt formats
        all_cols_16 = np.array(all_cols_16)
        all_cols_8 = np.array(all_cols_8)
        all_cols = np.array(all_cols)
        
        # make colors unique by shifting the non-unique ones by 1 until all are unique
        all_cols_8 = make_unique(all_cols_8, max_col=255)
        # all_cols_16 = make_unqiue(all_cols_16, max_col=65535)
        
        if visualize:
            # check the colors of uint8, do they look unique
            all_cols = all_cols_8.astype(float)/255
            # Create the grid coordinates
            x_coords, y_coords = np.meshgrid(np.arange(35), np.arange(35))
            x_coords = x_coords.flatten()[:len(all_cols)]
            y_coords = y_coords.flatten()[:len(all_cols)]
            # Create the scatter plot with the grid pattern
            plt.figure(figsize=(10, 10))
            plt.scatter(x_coords, y_coords, c=all_cols, s=100)
            plt.show()

        # create a pandas dataframe with the colors with shank_id as index        
        shank_ids, n_els = zip(*[(shank_info['shank_id'], shank_info['n_electrodes']) 
                                    for shank_info in info.values()])
        index = [[shank_id]*n_el for shank_id, n_el in zip(shank_ids, n_els)]
        index = pd.Index(np.concatenate(index), name='shank_id')
        colors = pd.DataFrame(all_cols_8, columns=['r', 'g', 'b'], index=index)
        return colors

    
    # prepare the unique pad/polyimide electrode colors for later alignment
    el_colors = make_unique_shank_colors(device_info['shank_top_wafer_view_left2right'], 
                                        #  visualize=visualize)
                                         visualize=False)

    if visualize:
        fig, axes = plt.subplots(ncols=4, figsize=(12,10), sharey=True, sharex=True)
    for shank_id in wafer_pad2el.shank_id.unique():
        if pd.isna(shank_id): continue
        shank_electrodes = wafer_pad2el[wafer_pad2el.shank_id==shank_id].sort_values('depth')
        shank_electrodes.loc[:, ('r', 'g', 'b')] = el_colors.loc[shank_id].values

        if visualize:
            shank_info = device_info['shank_top_wafer_view_left2right'][str(int(shank_id))]
            print(shank_id)
            print(shank_info)
            draw_shank(axes[int(shank_id)-1], shank_electrodes, shank_info)
    
        # update the wafer_pad2el dataframe with the new colors
        wafer_pad2el.loc[shank_electrodes.index] = shank_electrodes
        
    # assign non-connected pads unique colors too (grays)
    routed_mask = wafer_pad2el.shank_id.isna()
    grays = get_unique_grays(routed_mask.sum(), min_col=3, max_col=252)
    wafer_pad2el.loc[routed_mask, ('r', 'g', 'b')] = np.array(grays)
    wafer_pad2el['pad_diameter_um'] = device_info['pad_diameter_um']
    
    if visualize:
        if save_visualization is not None:
            fname_base = os.path.basename(save_visualization)
            plt.savefig(f"{save_visualization}/shank_geometry_{fname_base}.svg")
        plt.show()
    return wafer_pad2el
    
        
def finalize_and_save_pad2el(path, wafer_pad2el, device_info, visualize=True):
    print("==="*20)
    print(wafer_pad2el)
    print("==="*20)
    
    width = device_info['padlayout_width_um']
    height = device_info['padlayout_height_um']
    pad_diam = device_info['pad_diameter_um']
    canvas = np.zeros([height, width, 3], dtype=np.uint8)
    for i, row in wafer_pad2el.iterrows():
        y, x = int(row['y']), int(row['x'])
        col = np.array((row['r'], row['g'], row['b'])).astype(float)
        cv2.circle(canvas, (x, y), int(pad_diam), col, -1)
        if visualize:
            offset = 14 if i < 1000 else 19
            cv2.putText(canvas, f"{i}", (x-offset,y+5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255,255,255,255), 1)

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
    device_name = "H1278pad4shank"
    # device_name = "H1628pad1shank"
    # device_name = "H1384pad4shank" # not done yet
    path = os.path.join(nas_dir, "devices", "electrode_devices", device_name)
    print(path)

    # detect the pads using cv2.HoughCircles on um-scale image
    detected_wafer_pads = detect_wafer_pads(path, precomputed=True, save=False)
    # # order the pads x,y according in which order they are routed to the shanks
    wafer_pad2el = shank_ordered_wafer_pads(path, detected_wafer_pads, visualize=False)
    # # get the device info from the json file with ank info (n electrodes etc)
    device_info = get_device_info(path)
    # # add shank depth and metalization pairings to pad/polyimide electrode 
    wafer_pad2el = integr_shank_geometries(wafer_pad2el, device_info)
    # # assign unique colors to the pads/polyimide electrodes, 
    wafer_pad2el = assign_unqiue_colors(wafer_pad2el, device_info, visualize=True,
                                        save_visualization=path)
    # # save the final pad2el mapping as csv and png
    finalize_and_save_pad2el(path, wafer_pad2el, device_info, visualize=True)