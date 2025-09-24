import cv2
import json
import matplotlib.pyplot as plt
import napari
from collections import OrderedDict

import numpy as np
import pandas as pd


def get_device_info(fname_base):
    device_46pad_4shanks_1276electrode_info = {
        'shank_top_wafer_view_left2right': {
            "1": {
                'shank_id': 1,
                'routed_order': 1,
                'shank_name': 'anterior_short_shank',
                'colormap': 'PuBu',
                'colormap_lims': (.1,.9),
                'hook_to_deepest_el_distance_um': 500,
                'n_electrodes': 200,
                'n_electrode_pairs': 100,
                'least_deep_electrode_side': 'right',
                'least_deep_electrode_metal': 2,
                'pad_size_um': 13,
                'pad_um_center_distances': (
                    {'n_els': 200, 'el_dist': 30, 'gap_left': 0, 'gap_right': 0},
                )
            },
            "2": {
                'shank_id': 2,
                'routed_order': 0,
                'shank_name': 'anterior_long_shank',
                'colormap': 'YlGn',
                'colormap_lims': (.1,.9),
                'hook_to_deepest_el_distance_um': 265,
                'n_electrodes': 354,
                'n_electrode_pairs': 177,
                'least_deep_electrode_side': 'left',
                'least_deep_electrode_metal': 2,
                'pad_size_um': 13,
                'pad_um_center_distances': (
                    {'n_els': 44, 'el_dist': 40, 'gap_left': 0, 'gap_right': 100},
                    {'n_els': 72, 'el_dist': 20, 'gap_left': 140, 'gap_right': 100},
                    {'n_els': 60, 'el_dist': 40, 'gap_left': 140, 'gap_right': 160},
                    {'n_els': 116, 'el_dist': 20, 'gap_left': 190, 'gap_right': 150},
                    {'n_els': 60, 'el_dist': 40, 'gap_left': 110, 'gap_right': 150},
                )
            },
            "3": {
                'shank_id': 3,
                'routed_order': 2,
                'shank_name': 'posterior_long_shank',
                'colormap': 'RdPu',
                'colormap_lims': (.1,.9),
                'hook_to_deepest_el_distance_um': 250,
                'n_electrodes': 354,
                'n_electrode_pairs': 177,
                'least_deep_electrode_side': 'right',
                'least_deep_electrode_metal': 2,
                'pad_size_um': 13,
                'pad_um_center_distances': (
                    {'n_els': 44, 'el_dist': 40, 'gap_left': 100, 'gap_right': 0},
                    {'n_els': 72, 'el_dist': 20, 'gap_left': 100, 'gap_right': 140},
                    {'n_els': 60, 'el_dist': 40, 'gap_left': 160, 'gap_right': 140},
                    {'n_els': 116, 'el_dist': 20, 'gap_left': 150, 'gap_right': 150},
                    {'n_els': 60, 'el_dist': 40, 'gap_left': 150, 'gap_right': 110},
                )
            },
                
            "4": {
                'shank_id': 4,
                'routed_order': 3,
                'shank_name': 'posterior_short_shank',
                'colormap': 'Oranges',
                'colormap_lims': (.1,.9),
                'hook_to_deepest_el_distance_um': 500,
                'n_electrodes': 250,
                'n_electrode_pairs': 125,
                'least_deep_electrode_side': 'right',
                'least_deep_electrode_metal': 2,
                'pad_size_um': 13,
                'pad_um_center_distances': (
                    {'n_els': 250, 'el_dist': 25, 'gap_left': 0, 'gap_right': 0},
                    # 250: 25,
                )
            },
        },
        'shank_routed_order': (2,1,3,4),
        'n_pads': 1278,
        'n_electrodes': 1158,
        'n_electrode_pairs': 579,
        'n_metallization_1': 662,
        'n_metallization_2': 616,
    }
    
    # save to json
    import json
    with open('./assets/device_46pads_4shank.json', 'w') as f:
        json.dump(device_46pad_4shanks_1276electrode_info,  f, indent=4)
    
    with open(f'./assets/device_{fname_base}_4shank.json') as f:
        device_46pad_4shanks = json.load(f)
    return device_46pad_4shanks


















def detect_columns(fname):
    image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    row_mask = image[0,:] != 255
    print("Found n columns: ", row_mask.sum())
    locs = np.where(row_mask)[0]
    return np.array([0, *locs, image.shape[1]])

def detect_rows(fname):
    image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    row_mask = image[:,0] != 255
    print("Found n rows: ", row_mask.sum())
    ylocs = [0, *np.where(row_mask)[0]]
    return np.array(ylocs)

def detect_zones(fname):
    zones_image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    values = np.unique(zones_image)
    print("Found n zones: ", len(values), values)
    
    zone_mask = np.zeros_like(zones_image, dtype=np.int16)
    for i, value in enumerate(sorted(values)):
        mask_val = i if value != 255 else -1
        zone_mask[zones_image == value] = mask_val
    return zone_mask

def detect_metallization(fname):
    image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    metal_mask = ~image.astype(bool) # green metalization layer 1, mask=1 
                                     # purple metalization layer 2, mask=0
    return metal_mask

def get_layout_dims(fname):
    image = cv2.imread(fname)
    return image.shape[1], image.shape[0]

def detect_pads(fname_base, precomputed=False, save=False):
    path = './assets'
    if not precomputed:
        image = cv2.imread(f"{path}/waferpic_{fname_base}.png")
        canvas = np.zeros((*image.shape[:2],4), dtype=np.uint16)
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Use the Hough Circle Transform to detect circles
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
            np.savetxt(circles_fname.replace('.csv', '_new.csv'), circles, delimiter=',', fmt='%d')
    else:
        # load the circle centers from the CSV file, adjusted ones
        circles_fname = f"{path}/circle_centers_{fname_base}.csv"
        circles = np.loadtxt(circles_fname, delimiter=',', dtype=int)
    return circles

def get_shank_ordered_pads(fname_base, circles, visualize=True):
    path = './assets'
    
    width, height = get_layout_dims(f'{path}/waferpic_{fname_base}.png')
    layout_zone_mask = detect_zones(f"{path}/layout_zone_mask_{fname_base}.png")
    metal_mask = detect_metallization(f"{path}/metal_mask_{fname_base}.png")
    columns = detect_columns(f"{path}/columns_{fname_base}.png")
    rows = detect_rows(f"{path}/rows_{fname_base}.png")

    layout_zone_map = OrderedDict({0: "bottom_left", 1: "bottom_left_side", # bottom left quadrant
                                   2: "top_left_side", 3: "top_left",     # top left quadrant
                                   4: "bottom_right", 5: "bottom_right_side",  # bottom right quadrant
                                   6: "top_right_side", 7: "top_right"}) # top right quadrant
    layout_zone_col_order = {"bottom_left": -1, "bottom_left_side": 1, # left to right, right to left
                             "top_left_side": -1, "top_left": 1, # left to right, right to left
                             "bottom_right": 1, "bottom_right_side": -1,  # right to left, left to right
                             "top_right_side": 1, "top_right": -1} # right to left, left to right
    
    circles_sorted = []
    for zone_id, zone_name in layout_zone_map.items():
        print('--'*20)
        sorted_zone_circles = []
        
        zone_mask = layout_zone_mask == zone_id
        # zone_circles = circles[zone_mask]
        # print(len(circles))
        circles_mask = zone_mask[circles[:, 0], circles[:, 1]]
        zone_circles = circles[circles_mask]
        # print(len(zone_circles))
        # cirlces_np.zeros_like(zone_mask, dtype=bool)
        # plt.imshow(zone_mask)
        
        if layout_zone_col_order[zone_name] == 1:
            columns_idcs = np.arange(len(columns)-1)
            iter_columns = zip(columns_idcs, zip(columns[:-1], columns[1:]))
        elif layout_zone_col_order[zone_name] == -1:
            columns_idcs = np.arange(len(columns)-1)[::-1]
            iter_columns = zip(columns_idcs, (zip(reversed(columns[:-1]), reversed(columns[1:]))))
        
        for col_i, (from_x, to_x) in iter_columns:
            circles_in_col = zone_circles[(zone_circles[:, 1] > from_x) 
                                        & (zone_circles[:, 1] < to_x)]
            # reverse the order for everything in top zones
            if 'side' not in zone_name:
                # big blocks top and bottom left and roght aloways count top-down
                sorted_indices = np.argsort(circles_in_col[:, 0])
            else:
                # side blocks always count bottom-up (reverse)
                sorted_indices = np.argsort(circles_in_col[:, 0])[::-1]
            circles_in_col = circles_in_col[sorted_indices]
                
            circles_metal_layer = metal_mask[circles_in_col[:, 0], circles_in_col[:, 1]].astype(int)
            circles_metal_layer[circles_metal_layer == 0] = 2
            
            print(f'Zone {zone_name}, Column {col_i},  n={len(circles_in_col)},'
                  f' y_order={np.arange(len(circles_in_col))} metal:{circles_metal_layer}')
            if len(circles_in_col) > 0:
                n = len(circles_in_col)
                y = pd.Series(circles_in_col[:, 0], name='y')
                x_order = pd.Series(np.full(n, col_i), name='x_order')
                y_order = pd.Series(np.arange(n), name='y_order')
                x = pd.Series(circles_in_col[:, 1], name='x')
                metal = pd.Series(circles_metal_layer, name='metal')
                zone = pd.Series(np.full(n, zone_id), name='zone')
                depth = pd.Series(np.full(n, 1), name='depth')
                r = pd.Series(np.full(n, 1), name='r')
                g = pd.Series(np.full(n, 1), name='g')
                b = pd.Series(np.full(n, 1), name='b')
                a = pd.Series(np.full(n, 1), name='a')
                # a = pd.Series(circles_metal_layer/2, name='a')
                
                sorted_zone_circles.append(pd.concat([y, x, metal, zone, y_order, 
                                                      x_order, depth, r, g, b, a], 
                                                     axis=1))
                
                # colors = circles_sorted[-1].metal.map({1: 'green', 2: 'purple'})
                # plt.scatter(circles_in_col[:, 1], circles_in_col[:, 0], s=10, c=colors)
                # add text labevls for each circle
                # for i, (y, x) in enumerate(circles_in_col):
                #     plt.text(x, y, f'{i}', fontsize=8, color='r')
                # plt.show()
        sorted_zone_circles = pd.concat(sorted_zone_circles, axis=0).reset_index(drop=True)
        if 'side' in zone_name:
            # rows_idcs = np.arange(len(rows)-1)[::-1]
            iter_rows = zip(reversed(rows[:-1]), reversed(rows[1:]))

            resorted_zone_cirles = []
            for (from_y, to_y) in iter_rows:
                print(from_y, to_y)                
                within_row = (sorted_zone_circles.y > from_y) & (sorted_zone_circles.y < to_y)
                if 'top' in zone_name:
                    # within_row = reversed(within_row)
                    # pass
                    # print(sorted_zone_circles[within_row])
                    # print(sorted_zone_circles[within_row].iloc[::-1])
                    # exit()
                    
                    resorted_zone_cirles.append((sorted_zone_circles[within_row]))
                    print(sorted_zone_circles[within_row])
                else:
                    resorted_zone_cirles.append(sorted_zone_circles[within_row])
            sorted_zone_circles = pd.concat(resorted_zone_cirles, axis=0).reset_index(drop=True)
            print("--")
            print(sorted_zone_circles)
            
        circles_sorted.append(sorted_zone_circles)
        # if zone_id == 3:
        #     break
    pad_layout = pd.concat(circles_sorted, axis=0).reset_index(drop=True)
    pad_layout.index.name = 'pad_id'
    
    
    if visualize:
        # Create a colormap for the labeled pads
        colormap = plt.get_cmap('hsv', len(pad_layout))
        colors = [np.array((col[0]*255, col[1]*255, col[2]*255, 255)).round()
                for col in colormap(np.linspace(0, 1, len(pad_layout)))]
        
        print(colors[421])
        
        canvas = np.zeros([height, width, 4], dtype=np.uint8)
        for i, row in pad_layout.iterrows():
            y, x = row['y'], row['x']
            x = int(x)
            y = int(y)
            cv2.circle(canvas, (x, y), 26, colors[i].astype(float), -1)
            # print(f"{i}: {x}, {y}")
            offset = 14 if i < 1000 else 19
            cv2.putText(canvas, f"{i}", (x-offset,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255,255), 1)
        # print(canvas[15,15])

        col_row_canvas = np.zeros([height,width,4], dtype=np.uint8)
        for row_i in rows:
            col_row_canvas[row_i-1:row_i+1, :] = 255
        for col_i in columns:
            col_row_canvas[:, col_i-1:col_i+1] = 255

        # Launch Napari viewer to visualize the image and the circle points
        # canvas = canvas[:-60]
        # shift_canvas = np.zeros([2099, 3849, 4], dtype=np.uint16)
        # shift_canvas[85:canvas.shape[0]+85, 95:canvas.shape[1]+95] = canvas
        
        viewer = napari.Viewer()
        viewer.add_image(metal_mask, colormap='PiYG', name='metallization layer (Green=1, Purple=2)', visible=False)
        viewer.add_image(col_row_canvas, name='Columns and Rows', opacity=.2, visible=False)
        viewer.add_image(layout_zone_mask, name='Layout Zone', opacity=.25, visible=False)
        viewer.add_image(canvas, name='Detected Circles')
        # viewer.add_image(shift_canvas, name='Detected Circles')
        napari.run()
        
        # print(shift_canvas.shape)
        # cv2.imwrite(f"{path}/padlayout_aligned.png", shift_canvas)
        
        
        
        
        
        
        print(canvas[15,15])
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGRA)
        # print(canvas_bgr[115,115])
        # convert to 16 bit
        canvas_bgr = cv2.normalize(canvas_bgr, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
        print(canvas_bgr[15,15])
        # cv2.imwrite(f"{path}/lbled_padlayout_{fname_base}.png", canvas_bgr)
        # cv2.imwrite(f"{path}/padlayout_{fname_base}.png", canvas_bgr)
        cv2.imwrite(f"{path}/padlayout_{fname_base}.png", canvas_bgr)
    return pad_layout

def make_unqiue(colors, max_col):
    while True:
        _, unique_idx = np.unique(colors, axis=0, return_index=True)
        non_unique_idx = np.setdiff1d(np.arange(len(colors)), unique_idx)

        print(f"Unique (of {len(colors)}): ", len(unique_idx), "Non-unique: ", len(non_unique_idx))
        if len(unique_idx) == len(colors):
            print("All unique")
            break

        print("shifting non-unique")
        non_u = colors[non_unique_idx]
        shift_rgb_mask = (non_u!=max_col)
        non_u[shift_rgb_mask] = non_u[shift_rgb_mask] + 1
        colors[non_unique_idx] = non_u
    return colors

def make_unique_shank_colors(info, vis=False):
    # make colors unique
    shank_infos = [(shank_info['shank_id'], shank_info['n_electrodes'], 
                    shank_info['colormap'], shank_info['colormap_lims']) 
                    for shank_info in info.values()]

    all_cols = []
    all_cols_16 = []
    all_cols_8 = []
    for _, (_, n_el, cmap, cmap_lims) in enumerate(shank_infos):
        colormap = plt.get_cmap(cmap, n_el)
        colors = colormap(np.linspace(*cmap_lims, n_el))
        
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
    
    # all_cols_16 = make_unqiue(all_cols_16, max_col=65535)
    all_cols_8 = make_unqiue(all_cols_8, max_col=255)
    
    if vis:
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


        
def calculate_shank_geometries(fname_base, pad_layout, device_info, el_colors):
    
    def draw_shank(axes, shank_electrodes):
        m1_pads = shank_electrodes[shank_electrodes.metal==1].sort_values('el_pair')
        m2_pads = shank_electrodes[shank_electrodes.metal==2].sort_values('el_pair')
        m1_colors = m1_pads.loc[:, ['r', 'g', 'b']].values/255
        m2_colors = m1_pads.loc[:, ['r', 'g', 'b']].values/255
        # Plot the depth of electrodes on the shank
        axes[i].scatter(np.arange(len(m1_pads)), -m1_pads.depth, color='green', marker='s', alpha=.8, s=3)
        axes[i].scatter(np.arange(len(m2_pads)), -m2_pads.depth, color='purple', marker='s', alpha=.8, s=3)
        # axes[i].scatter(np.arange(len(m1_pads)), -m1_pads.depth, color=m1_colors, marker='s', alpha=.8, s=3)
        # axes[i].scatter(np.arange(len(m2_pads)), -m2_pads.depth, color=m2_colors, marker='s', alpha=.8, s=3)
        
        # plot the hook
        # max_depth = max(max(upper_depth), max(lower_depth))
        axes[i].scatter([len(left_side_depth)], -(max(lower_depth)+shank_info['hook_to_deepest_el_distance_um']), 
                    marker='o', edgecolor='gray', facecolor='none', alpha=.8, s=40)

        # Set the title of the subplot to indicate the shank ID and name
        axes[i].set_title(f"Shank {shank_id}:\n{shank_name}")
        # Remove x-axis ticks
        axes[i].set_xticks([])
        # Add grid lines along the y-axis
        axes[i].grid(axis='y')
        # Hide all the spines (borders) of the subplot
        [sp.set_visible(False) for sp in axes[i].spines.values()]

        # anntotions
        annot = f'{shank_info["n_electrode_pairs"]} electrode pairs'
        # number of electrode pairs (metal1+2)
        if shank_info['n_electrode_pairs'] % 2 != 0:
            annot += f' (uneven)'
        else:
            annot += f' (even)'
        # Set the x-axis label with the annotation
        axes[i].set_xlabel(annot, fontsize=7)
        if i == 0:
            axes[i].set_ylabel('Depth [um]')
        if shank_info['least_deep_electrode_side'] == 'left':
            axes[i].text(0, 140, f"Left-Start: metal {shank_info['least_deep_electrode_metal']}", fontsize=7)
        else:
            axes[i].text(n_pairs, 140, f"Right-Start: metal {shank_info['least_deep_electrode_metal']}", ha='right', fontsize=7)
    
    

    print('---------------------------------------------------------------')
    
        
    
    def one_side_one_metal_el_distances(start, shank_info, right=False):
        distances = []
        print(shank_info['pad_um_center_distances'])
        # shank_order = shank_info['routed_order']
        for group_info in shank_info['pad_um_center_distances']:
            group_n_els = group_info['n_els']
            dist = group_info['el_dist']
            group_gap_left = group_info['gap_left']
            group_gap_right = group_info['gap_right']
            
        # for group_n_els, (dist,group_gap_left,group_gap_right) in shank_info['pad_um_center_distances'].items():
            if right:
                # group_gap = group_gap_right
                group_top_gap = group_gap_right
                # group_bottom_gap = group_gap_right-dist*2
                group_bottom_gap = 0
                if group_info == {'n_els': 72, 'el_dist': 20, 'gap_left': 140, 'gap_right': 100}:
                    group_bottom_gap = 20
            else:
                group_top_gap = group_gap_left
                group_bottom_gap = 0
            
            print("Gaps: ", group_top_gap, group_bottom_gap)
            start = distances[-1]+group_top_gap if len(distances) else start
            end = start + (group_n_els//4) * (dist*4)
            print("start ", start, "    end ", end, 'last: ', distances[-1] if len(distances) else "Nooo")
            # print(group_n_els, start, end, dist)
            distances.extend(list(np.arange(start, end, dist*4)))
            distances[-1] += group_bottom_gap
            # print(distances)
            print(len(list(np.arange(start, end, dist*4))))
            # print((distances), '\n', (list(np.arange(start, end, dist*4))))
            print()
            # if not right:
            
        # distances = distances[:-1]
            
            
        if not right and shank_info['n_electrode_pairs']%2 != 0:
            tip_dist = distances[-1]+dist*4 if shank_id<3 else distances[-1]+dist*2
            distances.append(tip_dist)
        print("Final: ", len(distances))
        print()
        return np.array(distances)
    
    
    
    
    
    
    
    
    
    print(pad_layout[pad_layout.metal==1])
    print(pad_layout[pad_layout.metal==2])
    
    shank_infos = device_info['shank_top_wafer_view_left2right']
    shank_order = device_info['shank_routed_order']
    
    fig, axes = plt.subplots(ncols=4, figsize=(10,10), sharey=True)
    
    from_pad_m1, from_pad_m2 = 0, 0
    updated_pad_layout = []
    # for shank_info in enumerate(info):
    #     shank_id = shank_info['shank_id']
    
    for i, shank_id in enumerate(shank_order):
        shank_info = shank_infos[str(shank_id)]
        # shank_id = shank_info['shank_id']
    
    
        
        
        
        shank_name = shank_info['shank_name']
        print(f"\n\n\n\n\nShank {shank_id}: {shank_name}")
        

        # if shank_id != 4: continue
        
        
        n_pairs = shank_info['n_electrode_pairs']
        if shank_id == 3:
            from_pad = 1278//4 if fname_base == '46pads' else print('Not implemented')
            from_pad = 639
            from_pad_m2 = len(pad_layout[pad_layout.metal==2])//2
            from_pad_m1 = len(pad_layout[pad_layout.metal==1])//2
            
        # to_pad += n_pairs
        # print(pad_layout[pad_layout.metal==2])
        # print(pad_layout[pad_layout.metal==1])
        print(pad_layout[pad_layout.metal==2].iloc[from_pad_m2:])
        # print(len(range(from_pad, from_pad+n_pairs)))
        # print(from_pad)
        m1_pads = pad_layout[pad_layout.metal==1].iloc[from_pad_m1:from_pad_m1+n_pairs]
        m2_pads = pad_layout[pad_layout.metal==2].iloc[from_pad_m2:from_pad_m2+n_pairs]
        print(m2_pads)
        m1_pads['el_pair'] = np.arange(n_pairs)
        m2_pads['el_pair'] = np.arange(n_pairs)
        m1_pads['shank_id'] = shank_info['shank_id']
        m1_pads['shank_name'] = shank_info['shank_name']
        m2_pads['shank_id'] = shank_info['shank_id']
        m2_pads['shank_name'] = shank_info['shank_name']
        
        # print(m2_pads)
        
        # cumulative_group_n_els = np.cumsum(info["n_els"] for info in (shank_info['pad_um_center_distances']))
        first_el_dist = shank_info['pad_um_center_distances'][0]['el_dist']
        leftstart = 0 if shank_info['least_deep_electrode_side'] == 'left' else first_el_dist*2
        rightstart = 0 if shank_info['least_deep_electrode_side'] == 'right' else first_el_dist*2

        # calc_electrode_distance
        left_side_depth = one_side_one_metal_el_distances(leftstart, shank_info)
        right_side_depth = one_side_one_metal_el_distances(rightstart, shank_info, right=True)[::-1]
        # exit()

        print('\n\n\n====================')
        print(len(left_side_depth))
        print(len(right_side_depth))
        
        upper_depth = np.append(left_side_depth, right_side_depth)
        lower_depth = upper_depth.copy()
        # print(upper_depth)
        # print(cumulative_group_n_els)
        print("-----------")
        metal_el_shift_dist = []
        # for group_n_els, (el_dist,_,_) in shank_info['pad_um_center_distances'].items():
        for group_info in shank_info['pad_um_center_distances']:
            group_n_els = group_info['n_els']
            el_dist = group_info['el_dist']

            # if i == 0:
            #     start = 0
            print((group_n_els//4))
            metal_el_shift_dist.extend([el_dist]*(group_n_els//4))
            print(metal_el_shift_dist)
            # lower_depth[start:end] += el_dist
            # print(lower_depth)
            # start = end
        # exit()
        # lower_depth = upper_depth+first_el_dist # offset by single electr. dist
        # if unenven
        print()
        print(len(metal_el_shift_dist))
        print(n_pairs)
        twosided_metal_el_shift_dist = [*metal_el_shift_dist]
        # print(twosided_metal_el_shift_dist)
        # print(shank_info['n_electrode_pairs']%2)
        print(len(twosided_metal_el_shift_dist))
        if shank_info['n_electrode_pairs']%2 != 0:
            twosided_metal_el_shift_dist.append(el_dist)
        print(len(twosided_metal_el_shift_dist))
        # print(twosided_metal_el_shift_dist)
        twosided_metal_el_shift_dist.extend(metal_el_shift_dist[::-1])
        print(len(twosided_metal_el_shift_dist))
        # print(twosided_metal_el_shift_dist)
            
        # print(len(metal_el_shift_dist))
        twosided_metal_el_shift_dist = np.array(twosided_metal_el_shift_dist)
        
        print(twosided_metal_el_shift_dist.shape)
        
        # # lower_depth += np.zeros_like(lower_depth)
        # metal_el_shift_dist_ = np.zeros_like(lower_depth)
        # metal_el_shift_dist_[:len(metal_el_shift_dist)] = metal_el_shift_dist
        # metal_el_shift_dist = metal_el_shift_dist_
        print(lower_depth.shape)
        lower_depth += twosided_metal_el_shift_dist

        
        if shank_info['least_deep_electrode_metal'] == 1:
            
            m1_pads['depth'] = upper_depth
            m2_pads['depth'] = lower_depth
        else:
            m2_pads['depth'] = upper_depth
            m1_pads['depth'] = lower_depth
            print('-----------------')
        print(m2_pads)
        print(m1_pads)
        shank_electrodes = pd.concat([m1_pads, m2_pads], axis=0).sort_values('depth')
        
        print(shank_electrodes)
        print(shank_electrodes.depth)

        # colormap = plt.get_cmap(shank_info['colormap'], len(shank_electrodes))
        # colors = [np.array((col[0]*255, col[1]*255, col[2]*255, 255)).round()
        #             for col in colormap(np.linspace(0, 1, len(shank_electrodes['depth'])))]
        # print(len(colors))
        
        # m1_pads.loc['r']
        
        # print(shank_electrodes.loc[:, ('r', 'g', 'b', 'a')].shape)
        # print(np.array(colors).T.shape)
        # # print(colors)
        # print(shank_electrodes)
        
        
        # print(shank_electrodes)
        print("n colors: ", len(el_colors.loc[shank_id]))
        print()
        print()
        print()
        print()
        # if shank_id != 2 and shank_id != 3:
        shank_electrodes.loc[:, ('r', 'g', 'b')] = el_colors.loc[shank_id].values
            # exit()
        # exit()

        
        draw_shank(axes, shank_electrodes)
        
        updated_pad_layout.append(shank_electrodes)
        
        from_pad_m1 += n_pairs
        from_pad_m2 += n_pairs
    plt.show()
    
    
    updated_pad_layout = pd.concat(updated_pad_layout, axis=0)
    print(updated_pad_layout)
    non_connected_pad_layout = pad_layout.drop(updated_pad_layout.index)
    
    gray_colormap = plt.get_cmap('gray', len(non_connected_pad_layout))
    grays = [np.array((col[0]*255, col[1]*255, col[2]*255)).round()
            #  for col in gray_colormap(np.linspace(.8,1, len(non_connected_pad_layout)))]
             for col in gray_colormap(np.linspace(.4,.95, len(non_connected_pad_layout)))]
    grays = make_unqiue(np.array(grays), max_col=255)
    non_connected_pad_layout.loc[:, ('r', 'g', 'b')] = np.array(grays)
    
    pad_layout = pd.concat([updated_pad_layout, 
                            non_connected_pad_layout], axis=0)
    print(pad_layout)
    
    # save the updated pad layout
    path = './assets'
    pad_layout.to_csv(f"{path}/device_mapping_{fname_base}.csv")
    # updated_pad_layout
    
    
    
    # # HPC
    # shank1_info = {"n_electrode_pairs": 358, 'deepest_el_pair': 88}
    # m1 = circles_sorted[circles_sorted.metal==1][:178]
    # depth = np.arange(len(m1), dtype=float) -shank1_info['deepest_el_pair']
    # depth[depth<0] -= 1.5
    # depth = abs(depth)
    # print("m1: ", depth)
    # m1['depth'] = depth
    # print()
    
    # m2 = circles_sorted[circles_sorted.metal==2][:178]
    # depth = np.arange(len(m1), dtype=float) -shank1_info['deepest_el_pair']
    # depth[depth<0] -= 1.5
    # depth = abs(depth)
    # depth += .2
    # print("m2: ", depth)
    # m2['depth'] = depth
    
    # # m1[88]
    # # m2 = circles_sorted[circles_sorted.metal==2][:178]
    # # print(m2)
    # # print(m2[88:91])
    # # exit()
    # # # exit()
    # circles_sorted = pd.concat([m1, m2], axis=0).sort_index()
    
    # # print(depth)
    # # # circles_sorted['depth'] = depth
    # # print(circles_sorted.sort_values(by=['depth']))
    # # print(circles_sorted.depth)
    # # exit()
    
    # circles_sorted = circles_sorted.sort_values(by='depth')
    # print(circles_sorted)
    
    
    
    # shank_cmaps = {'lShank1': 'YlGn', 'lShank2': 'PuBu', 'lShank3': 'RdPu', 'lShank4': 'Oranges'}
    
    
     
    # # Create a colormap for the labeled pads
    # colormap = plt.get_cmap('coolwarm', len(circles_sorted))
    # colors = [np.array((col[0]*255, col[1]*255, col[2]*255, 255)).round()
    #           for col in colormap(np.linspace(0, 1, len(circles_sorted)))]
    # # print(colors)
    # # colors = [colors//3 if metal == 1 else colors for colors, metal in zip(colors, circles_sorted.metal)]
    
    # colors_df = pd.DataFrame(np.array(colors, dtype=int)[:,:3], columns=['r', 'g', 'b'], 
    #                          index=np.arange(len(circles_sorted)))
    # print(colors_df)
    # # check if all values (rows) are unique 
    # # print(colors_df[:29])
    # # assert colors_df.duplicated().sum() == 0, f"There are duplicate colors: {colors_df.duplicated().sum()}"
    # colors_df.to_csv(f"{path}/padcolors_{fname_base}.csv", index=False)    
    
    # print(colors[:5])
    
    # # [1276 rows x 3 columns]
    # # [array([127.,   0.,   0., 127.]), array([255.,   1.,   0., 255.]), array([127.,   1.,   0., 127.]), array([255.,   4.,   0., 255.]), array([127.,   2.,   0., 127.])]
    # # exit()
    
    height = 2099
    width = 3849
    canvas = np.zeros([height, width, 3], dtype=np.uint8)

    # for i, row in circles_sorted.iterrows():
    for i, row in pad_layout.iterrows():
        y, x = row['y'], row['x']
    # for i, (y, x) in enumerate(circles):
        # print(i, colors[i])
        # print(i, colors[i].dtype)
        # print(x,y)
        x = int(x)
        y = int(y)
        col = np.array((row['r'], row['g'], row['b'])).astype(float)
        # print(col)
        # print(colors[i])
        # cv2.circle(canvas, (x, y), 26, colors[i].astype(float), -1)
        # labelled version
        # if i>=len(colors):
        #     continue
        cv2.circle(canvas, (x, y), 26, col, -1)
        offset = 14 if i < 1000 else 19
        cv2.putText(canvas, f"{i}", (x-offset,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255,255), 1)

    # add vertical lines to separate columns
    # for col_i in columns:
    #     blurred[:, col_i-1:col_i+1] = 0
        
    # col_row_canvas = np.zeros([*blurred.shape[:2],4], dtype=np.uint8)
    # for row_i in rows:
    #     col_row_canvas[row_i-1:row_i+1, :] = 255
    # for col_i in columns:
    #     col_row_canvas[:, col_i-1:col_i+1] = 255

    # Launch Napari viewer to visualize the image and the circle points
    viewer = napari.Viewer()
    # viewer.add_image(blurred, name='wafer pictures', rgb=False, colormap='gray_r', opacity=.2)
    #  use cv2 to to convert uint16 canvas to uint8
    # Normalize the canvas to the range [0, 255] before converting to uint8
    # canvas = 
    # viewer.add_shapes(data=layout_zone_mask, shape_type='polygon', edge_color='red', face_color='red', opacity=0.2, name='Layout Zone')
    # viewer.add_image(metal_mask, colormap='PiYG', name='metallization layer (Green=1, Purple=2)')
    # viewer.add_image(col_row_canvas, name='Columns and Rows', opacity=.2)
    # viewer.add_image(layout_zone_mask, name='Layout Zone', opacity=.25)
    # viewer.add_image(cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), 
    #                  name='Detected Circles')
    viewer.add_image(canvas, name='Detected Circles')
    napari.run()
    
    # Convert RGBA to BGR before saving
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGRA)
    # black to transparent
    canvas_bgr[canvas_bgr[:,:,0]==0] = 0
    path = './assets'
    cv2.imwrite(f"{path}/padlayout_hollow_{fname_base}.png", canvas_bgr)
    # cv2.imwrite(f"{path}/padlayout_{fname_base}.png", canvas_bgr)
    # cv2.imwrite(f"{path}/lbled_padlayout_{fname_base}.png", canvas_bgr)
    
    
fname_base = '46pads'
# detect_pads(fname_base)
# calculate_device_layout(fname_base)
circles = detect_pads(fname_base, precomputed=True)
pad_layout = get_shank_ordered_pads(fname_base, circles, visualize=False)
device_info = get_device_info(fname_base=fname_base)
# print(device_info)
el_colors = make_unique_shank_colors(device_info['shank_top_wafer_view_left2right'], vis=False)
calculate_shank_geometries(fname_base, pad_layout, device_info, el_colors)

# fname_base = '48pads'
# detect_pads(fname_base)