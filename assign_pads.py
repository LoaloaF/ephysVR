import os
from glob import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import scipy.signal as signal
from scipy.signal import butter, filtfilt
from scipy.signal import hilbert

from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch

import multiprocessing as mp
from functools import partial

import pandas as pd
import cv2

from analyze_connectivity import draw_mea1k
    
from ephys_constants import DEVICE_NAME, MEA1K_EL_2D_TABLE_PIXEL, MEA1K_EL_TABLE_PIXEL_YX_IDX, NAS_DIR, MEA1K_EL_CENTER_TABLE_PIXEL, MEA1K_EL_CENTER_TABLE_MICROMETER, MEA1K_EL_WIDTH_MICROMETER, MEA1K_EL_HEIGHT_MICROMETER
import napari
from napari.utils import DirectLabelColormap
    

def align_pads(path):
    # Load the image with 16-bit depth
    # pads = cv2.imread(f"{path}/padlayout_46pads.png", -1)
    # print(pads[15,15])
    
    pads = cv2.imread(f"{path}/padlayout_aligned.png", -1)
    pads = cv2.cvtColor(pads, cv2.COLOR_BGR2RGB)
    scaler = ((2**16)-1)
    scaler = 255
    
    # convert to 16 bit
    # cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), 
    # #                  name='Detected Circles'
    # pads = cv2.normalize(pads, None, 0, scaler, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    print(pads.shape)
    # print(pads[115,115])
    # exit()
    
    
    # print(pads[20:, 20:, 0])
    # print(pads.shape)
    # this is BGR but should be RGB
    # pads = pads[:,:,::-1]
    # turn the black background to white
    # black_pixels = np.all(pads == [0, 0, 0], axis=-1)
    # pads[black_pixels] = [255, 255, 255]    # pads[pads==[0,0,0]] = 255
    padfig = plt.figure()
    plt.imshow(pads[:,:,0])

    plt.figure()
    mea1k = cv2.imread(f"{path}/connectivity_ampl.png")
    mea1k = mea1k[:,:,0] # black and white
    plt.imshow(mea1k, cmap='gray_r')
    plt.xlim()
    
    which_layout = "46pads"
    padcolor = pd.read_csv(f"./assets/padcolors_{which_layout}.csv")
    # print(padcolor)
    # print(padcolor.iloc[421:422])
    
    device_mapping = pd.read_csv(f"./assets/device_mapping_46pads.csv")
    print(device_mapping)
    
    connectivity = pd.read_pickle(f"{path}/extracted_signal.pkl")
    print(connectivity)
    
    mea1k_xy = []
    assignments = []
    i = 0
    
    dupl_col_i = 0
    for y in np.arange(17.5/4, 2100, 17.5):
        for x in np.arange(17.5/4, 3850, 17.5):
            # if not (x < 350 and y<100):
            #     continue
            um_x, um_y = round(x+4.5), round(y+4.5)
            mea1k_xy.append((um_x,um_y))
            
            
            col = (pads[um_y,um_x])
            if np.all(np.array(col == [0,0,0])) or i not in connectivity.index:
                i += 1
                continue
            
            # print(i, col, end=' ' if np.all(col == [0,0,0]) else '\n')
            # print(f"\ncol: {col}\n======")
            # pad = padcolor[(padcolor['r'] == col[0]) & (padcolor['g'] == col[1]) & (padcolor['b'] == col[2])]
            # pad = device_mapping[device_mapping['r'] == col[0] & (device_mapping['g'] == col[1]) & (device_mapping['b'] == col[2])]
            pad = device_mapping[(device_mapping['r'] == col[0]) & 
                        (device_mapping['g'] == col[1]) & 
                        (device_mapping['b'] == col[2])]
            # print(pad.loc[:,['pad_id', 'r', 'g', 'b']])
            # print(pad)
            
            # print(pad)
            el_info = connectivity.loc[i].to_frame().T.reset_index().rename(columns={"index":"el"})
            
            assignm = pad.copy()
            assignm['el'] = i
            assignm['ampl'] = el_info['ampl'].values[0]
            assignm['power'] = el_info['power'].values[0]
            # print(assignm)
            # assignm
            # assignm = connectivity.loc[i].copy()
            # assignm.loc['pad_id'] = pad['pad_id'].values[0]
            # print(assignm)
            # exit()
            # print all infos
            # padfig.gca().scatter(um_x, um_y, c=(np.array(col/255))[np.newaxis,:], s=10, edgecolors='black')
            
            # print(i, x, y, end='\n')
            # if not pad.empty:
                # print(i, end='..')
                # print(col, pad, end='\n\n')
                
                # pad = pad.copy().reset_index()
                # if pad.shape[0] > 1:
                #     # print(f"Duplicate color: {col}, len: {pad.shape[0]}")
                #     # dupl color issue, get first the first one, then the second one, and reset the index
                #     pad = pad.iloc[dupl_col_i:dupl_col_i+1]
                #     # print(pad, dupl_col_i, "next:", 1 if dupl_col_i == 0 else 0)
                #     dupl_col_i = 1 if dupl_col_i == 0 else 0
                # pad.loc[:,'el'] = i
            assignments.append(assignm)
            
            rect = plt.Rectangle((x, y), 9, 9, facecolor=col/scaler, edgecolor=None)
            plt.gca().add_patch(rect)
            i += 1
    
    
    assignments = pd.concat(assignments, axis=0)
    
    
    print(assignments.shape)
    print(device_mapping)
    
    missing_pads = device_mapping.drop(assignments.pad_id)
    print(missing_pads)
    
    assignments = pd.concat([assignments, missing_pads], axis=0).sort_values('pad_id').reset_index(drop=True)
    print(assignments)
    
    print(connectivity)
    
    
    
    
    
    print()
    ranks = assignments[['pad_id', 'ampl']].sort_values(['pad_id', 'ampl'], ascending=[True, False])
    # print(ranks)
    ranks = assignments[['pad_id', 'ampl']].sort_values(['pad_id', 'ampl'], ascending=[True, False]).groupby('pad_id').rank(ascending=False).sort_index()
    # print(ranks)
    
    assignments['ampl_order'] = ranks['ampl'].astype(int)
    assignments['routed'] = ((assignments['ampl']>20) & (assignments['ampl_order'] == 1))
    print(assignments[assignments['routed']])
    
    
    
    
    missing_els = connectivity.drop(assignments.el).reset_index().rename(columns={"index":"el"})
    print(missing_els)
    
    assignments = pd.concat([assignments, missing_els], axis=0).reset_index(drop=True)
    print(assignments)

    
    
    
    
    
    
    # print(assignments.groupby("pad_id").max("ampl"))
    # assignments.sort_values("pad_id", inplace=True)
    # assignments.to_pickle(f"{path}/pad_assignment.pkl")     
    assignments.to_csv(f"{path}/bonding_electrode_map.csv")     
    
    assignments = pd.read_csv(f"{path}/pad_assignment.csv", index_col=[0])
    print(assignments)
            
    # plt.scatter(*zip(*mea1k_xy), c='red', s=1)
            
    # plt.show()
    
def get_pad_alignment_canvas(PATH, hollow=False, white_bg=False, remove_alpha=False, incl_colors=None):
    fname = f"{PATH}/padlayout_aligned.png"
    if hollow:
        fname = f"{PATH}/padlayout_aligned_hollow.png"
    pad_alignment = cv2.imread(fname, -1)
    pad_alignment = cv2.cvtColor(pad_alignment, cv2.COLOR_BGR2RGB)

    if white_bg:    
        pad_alignment = pad_alignment[:,:,:3]
        black_pixels = np.all(pad_alignment == [0, 0, 0], axis=-1)
        pad_alignment[black_pixels] = [255, 255, 255]
    elif remove_alpha:
        pad_alignment = pad_alignment[:,:,:3]
        
    if incl_colors is not None:
        # Convert incl_colors to a set of tuples for fast lookup
        incl_colors_set = {tuple(color) for color in incl_colors}
        # Apply the filter to pad_alignment
        pad_alignment = np.array([
            [color if tuple(color) in incl_colors_set else [255, 255, 255] for color in row]
            for row in pad_alignment
        ])
    return pad_alignment        
    
def make_mea1k_config(PATH, min_ampl=20):
    connected_els = [11704, 15775, 9689, 10962, 14305, 10735, 11942, 17559, 17358, 12571, 14894, 12551, 9650, 15678, 12726, 8819, 9672, 7533, 15827, 8815, 11619, 12580, 16675, 11467, 11610, 10579, 14500, 7732, 12962, 12795, 16484, 16312, 8999, 12392, 15770, 11669, 10971, 14803, 11648, 10178, 8770, 15444, 11862, 7939, 14258, 8413, 14727, 12815, 8131, 14535, 19331, 10596, 15375, 16519, 12730, 6864, 12719, 16328, 12734, 9502, 6355, 16522, 16475, 10795, 12054, 15409, 17109, 10791, 12039, 15422, 12751, 11962, 14714, 11706, 18662, 10827, 9849, 11698, 21111, 8115, 15211, 16678, 8418, 6803, 15659, 15366, 10602, 7020, 16507, 16487, 10598, 12713, 16698, 8827, 6367, 12582, 19757, 10582, 10103, 16314, 7657, 8831, 9658, 17419, 17566, 3969, 9011, 17385, 19322, 7529, 8107, 17182, 17375, 6834, 6777, 17393, 17367, 8585, 10539, 15647, 11635, 6980, 10173, 20637, 10754, 20495, 19571, 3332, 4171, 20266, 23065, 5104, 6359, 21375, 3974, 8766, 18499, 17363, 5234, 23958, 5959, 8763, 21150, 23966, 5902, 15693, 22423, 4854, 7010, 19396, 18864, 1118, 5258, 14326, 24153, 7947, 5043, 17414, 18231, 4851, 3261, 16553, 21102, 10551, 16526, 21098, 6626, 1294, 25745, 5730, 5935, 15186, 14502, 10367, 4810, 15413, 21297, 5726, 11630, 21795, 17343, 8614, 7854, 18528, 5743, 4141, 21142, 10590, 5889, 14115, 22198, 6825, 6811, 15227, 20425, 6829, 9002, 14344, 20444, 6807, 19374, 16686, 11470, 8534, 17171, 19338, 11474, 7255, 18510, 21326, 4384, 17887, 12545, 7687, 14551, 19335, 14246, 20035, 19326, 7490, 7695, 18063, 15591, 8774, 16295, 15382, 6653, 5268, 17398, 15378, 6850, 9395, 14347, 12710, 9702, 10543, 19163, 11859, 3763, 19563, 1278, 19576, 3291, 20228, 19379, 3502, 23316, 2879, 24654, 20456, 7707, 7005, 20439, 7934, 18495, 24183, 2222, 20263, 24196, 4370, 20247, 3239, 5055, 23773, 2594, 24867, 5254, 25076, 1114, 22216, 5932, 454, 9667, 5905, 21116, 20430, 1346, 3104, 3046, 22938, 25934, 5083, 10959, 19166, 20642, 5954, 25543, 18666, 9494, 1303, 24822, 5975, 1282, 23079, 9698, 6984, 25059, 6855, 2179, 22043, 22427, 7891, 22247, 22863, 3478, 23053, 4199, 8118, 22908, 18653, 7234, 24843, 5971, 5017, 20940, 23311, 7869, 15354, 8122, 22904, 23950, 3315, 1507, 24830, 23784, 18687, 5966, 1298, 18455, 6774, 23812, 24847, 11626, 2431, 4123, 20435, 5690, 23954, 3311, 4168, 23295, 4374, 24666, 2631, 22211, 19568, 5052, 21127, 23307, 7058, 3498, 18460, 5539, 1344, 3495, 18744, 22195, 7722, 5875, 23941, 5087, 10095, 21133, 20405, 2226, 11155, 22030, 21293, 4207, 10746, 18878, 7743, 9619, 21311, 3108, 3270, 19172, 23057, 2390, 22177, 5567, 2585, 19180, 15780, 3771, 9866, 20060, 20634, 7062, 5892, 24671, 24814, 1352, 2377, 15681, 25055, 4872, 8978, 23947, 4021, 9643, 24696, 25926, 3335, 6090, 17554, 2235, 6782, 19412, 14490, 5095, 5021, 17410, 21307, 622, 16540, 23905, 6877, 2359, 17648, 22411, 7734, 2149, 18286, 23929, 3779, 5003, 23799, 15569, 11383, 17187, 19749, 10413, 9414, 14131, 23070, 2900, 7881, 21521, 6001, 4365, 23149, 23281, 3141, 8519, 20050, 22165, 7519, 6970, 21165, 11623, 5100, 12019, 23803, 25045, 6639, 6099, 24683, 14475, 5078, 8529, 22067, 24818, 5793, 10289, 18294, 18429, 4238, 3254, 18079, 23920, 7538, 3430, 21196, 14890, 3123, 5634, 22933, 23025, 2067, 6768, 17451, 22155, 3127, 9375, 18753, 23035, 4682, 10269, 21408, 19700, 2277, 6117, 23171, 17989, 5777, 6079, 24703, 19725, 5796, 4774, 23823, 18625, 7781, 10259, 20296, 21288, 4026, 3465, 12387, 22145, 11291, 10689, 16573, 23039, 3153, 5000, 23828, 16470, 9521, 13354, 22925, 16883, 8641, 11600, 22278, 17530, 2917, 4340, 19422, 19534, 4896, 7640, 11077, 23923, 6897, 10685, 12588, 22149, 9526, 1265, 21183, 19514, 11286, 8735, 23818, 10505, 9292, 10264, 23140, 21274, 3995, 3450, 17422, 22405, 6869, 4353, 24693, 23269, 8623, 7863, 18538, 18415, 9490, 3248, 17877, 21314, 8627, 3909, 14575, 19524, 5108, 4789, 21393, 22401, 5992, 4979, 19638, 25244, 4005, 8730, 17432, 22174, 9322, 4334, 24689, 14263, 5113, 8743, 19207, 20408, 9486, 10714, 15451, 15578, 6666, 4138, 22047, 14690, 3367, 7630, 24707, 17539, 8668, 1494, 21169, 25019, 4908, 7844, 18332, 21490, 5548, 6754, 16558, 12926, 11522, 3889, 22071, 17754, 4257, 3225, 16874, 5131, 4769, 23831, 21280, 13053, 9390, 16781, 11145, 7541, 8720, 17661, 17099, 14807, 5190, 11316, 15582, 13292, 4984, 17683, 21505, 6452, 7818, 22061, 17514, 9531, 6105, 22941, 24808, 7791, 2560, 20522, 21485, 9327, 11343, 18762, 14235, 6908, 12450, 19628, 12029, 9518, 3220, 20087, 23249, 10202, 9360, 18327, 8433, 4083, 13928, 17524, 1181, 8479, 12407, 12034, 4917, 7835, 23166, 21259, 3378, 4319, 14153, 15785, 10622, 4079, 19652, 18849, 6663, 6725, 11569, 5143, 4754, 21422, 19729, 11321, 9355, 21177, 23234, 7553, 4105, 12176, 22135, 10418, 7595, 20311, 19520, 7766, 4955, 21193, 13339, 8663, 12024, 11982, 17084, 4267, 6940, 24933, 11579, 3392, 10254, 23183, 23900, 4948, 7807, 21459, 1411, 2301, 12435, 12225, 10662, 6691, 15686, 23219, 7578, 4739, 17499, 21250, 4287, 3198, 20579, 2087, 5823, 16635, 23239, 3181, 7605, 9365, 23214, 4722, 3832, 23875, 8467, 6734, 21454, 21206, 5582, 4073, 24948, 17509, 5614, 3408, 13277, 19715, 6916, 5619, 23188, 13108, 4043, 6729, 15921, 12014, 5597, 4282, 11547, 8458, 6959, 16620, 17713, 8705, 1234, 10679, 19704, 3382, 10446, 12827, 17504, 3193, 6923, 12440, 24104, 2296, 7802, 10218, 19280, 7568, 8463, 17678, 14880, 2957, 4309, 24760, 23003, 5152, 6927, 10647, 24774, 9332, 6935, 12847, 14209, 5812, 6965, 19624, 15749, 6047, 10720, 10627, 14205, 4938, 11554, 13308, 18839, 11278, 3424, 12428, 20383, 6027, 3875, 13943, 20380, 5148, 4965, 23177, 18395, 4277, 3214, 11308, 14215, 4031, 8494, 19438, 22140, 13282, 6085, 23230]
    assignm = pd.read_pickle(f"{PATH}/pad_assignment.pkl")
    print(assignm)
    # best_el_per_pad = assignm.reset_index().groupby("pad_id").max("ampl")
    best_el_per_pad = assignm[["pad_id", "ampl", ]].loc[
        assignm.groupby("pad_id")["ampl"].idxmax()
    ]
    
    # integrate best pad or even better pad order in final device mapping
    
    best_el_per_pad = best_el_per_pad[best_el_per_pad['ampl'] > min_ampl]
    best_el_per_pad = assignm.loc[best_el_per_pad.index]
    
    colors = best_el_per_pad[['r', 'g', 'b']].values
    pad_canvas = get_pad_alignment_canvas(PATH, hollow=True, white_bg=False, incl_colors=colors)
    
    # print(pad_alignment.shape)
    # pad_alignment[:, :600] = 255
    # plt.imshow(pad_alignment)
    # plt.show()
    
    # draw_mea1k()
    (fig, ax), els = draw_mea1k(bg=pad_canvas, el_color='#ffffff')
    ax.imshow(pad_canvas)
    grays_cmap = plt.get_cmap('gray_r', 100)
    vmax = 25
    text_patches = []
    for el_i, row in assignm.iterrows():
        # pad_id = row['pad_id']
        # x, y = els[pad_id]
        connectivity_col = grays_cmap(row['ampl']/vmax)
        els[el_i].set_facecolor(connectivity_col)
        # els[el_i].set_alpha(1)
        # if el_i > 1000:
        #     break
        col = row[['r', 'g', 'b']].values/255
        # if el_i in best_el_per_pad.el.values:
        if el_i in connected_els:
            els[el_i].set_edgecolor(connectivity_col)
            els[el_i].set_linewidth(2)
            
            if el_i in best_el_per_pad.index:
                els[el_i].set_linewidth(4)
            x,y = els[el_i].get_xy()
            ax.text(x, y-5, f"el{el_i:05d}", color='black', fontsize=6, 
                    transform=ax.transData, ha='center', va='center')
            
            # tp = TextPath((x,y), f"el{el_i:05d}", size=3, )
            # text_patches.append(PathPatch(tp, color="white"))

        # print(x, y)
        # ax.text(x, y, f"{i}", color='red', fontsize=6)
    [ax.add_patch(tp) for tp in text_patches]
    plt.show()
    
def align_pads2mea1k(ELECTRODE_DEVICE_NAME, connectivty_measures_path):
    path = os.path.join(NAS_DIR, "electrode_devices", ELECTRODE_DEVICE_NAME, 
                        f"device_mapping_{ELECTRODE_DEVICE_NAME}.csv")
    mapping = pd.read_csv(path, index_col=[0])
    print(mapping)
    
    path = os.path.join(NAS_DIR, connectivty_measures_path, 'connectivity_ampl.png')
    mea1k_connectivity_png = cv2.imread(path)[:,:,0] # black and white
    path = os.path.join(NAS_DIR, connectivty_measures_path, 'ext_signal_ampl.csv')
    mea1k_connectivity = pd.read_csv(path, index_col=[0]).sort_index()
    print(mea1k_connectivity)
    
    # draw circles (connectivity pads)
    pad_r = 26
    pad_colors = {-1: (0,0,0,0)}
    pads_canvas = np.ones_like(mea1k_connectivity_png, dtype=np.int16) * -1
    for pad_id, row in mapping.iterrows():
        x,y = row[['x', 'y']]
        cv2.circle(pads_canvas, (x,y), pad_r, pad_id, -1)
        pad_colors[pad_id] = (row[['r','g','b']].values.astype(float)/255.)
    pad_colors = DirectLabelColormap(color_dict=pad_colors)
        
    # align    
    viewer = napari.Viewer()
    mea1k = viewer.add_image(mea1k_connectivity_png, name='Connectivity', colormap='gray_r')
    alignment = viewer.add_labels(pads_canvas, name='Labels', colormap=pad_colors)
    napari.run()
    
    # parse the alligned pads and match them to mea1k electrodes under them
    pad_alignment = []
    for pad_id, row in mapping.iterrows():
        # get xy coordinates of the pad
        new_pad_yx_um = np.stack(np.where(alignment.data == pad_id)).T
        # transofrm to mea1k coordate system (world coo)
        new_pad_yx_um = [alignment.data_to_world(xy) for xy in new_pad_yx_um]
        # convert to pixel coordinates (integers for indexing)
        new_pad_yx_pixels = np.stack(new_pad_yx_um).round().astype(int) 
        # the new pad center is the median of all pixels in the circle        
        new_pad_yx = np.median(new_pad_yx_pixels, axis=0).round().astype(int)
        # save the new pad center after alignment
        aligned_pad = row.copy()
        aligned_pad["pad_id"] = pad_id
        aligned_pad["mea1k_el"] = np.nan    # update this later if pad over it
        aligned_pad["mea1k_connectivity"] = np.nan    # update this later if pad over it
        aligned_pad["y_aligned"] = new_pad_yx[0]
        aligned_pad["x_aligned"] = new_pad_yx[1]

        # get the mea1k electrodes under the pad
        mea1k_els_under_pad_mask = (MEA1K_EL_2D_TABLE_PIXEL.index.isin(new_pad_yx_pixels.tolist()))
        mea1k_el_underpad = MEA1K_EL_2D_TABLE_PIXEL[mea1k_els_under_pad_mask]
        mea1k_el_underpad = np.unique(mea1k_el_underpad.values)
        print(f"pad{pad_id} -> n el:{len(mea1k_el_underpad)}", end='...', flush=True)
        
        # append the pad n times, matching the number of electrodes under it
        # if no electrode under the pad, append the pad once, el=np.nan
        dat = [aligned_pad]*max(1, mea1k_el_underpad.shape[0])
        aligned_pad = pd.concat(dat, axis=1).T.reset_index(drop=True)
        if mea1k_el_underpad.shape[0] > 0:
            # add the mea1k electrodes under the pad
            aligned_pad.loc[:,"mea1k_el"] = mea1k_el_underpad
            aligned_pad.loc[:,"mea1k_connectivity"] = mea1k_connectivity.reindex(mea1k_el_underpad).values
        pad_alignment.append(aligned_pad)
        
        # if pad_id>100:
        #     break
    pad_alignment = pd.concat(pad_alignment, axis=0).reset_index(drop=True)


    # group by pad_id and rank the electrodes by connectivity
    ranks = pad_alignment[['pad_id', 'mea1k_connectivity']]
    ranks = ranks.sort_values(['pad_id', 'mea1k_connectivity'], 
                              ascending=[True, False])
    ranks = ranks.groupby('pad_id').rank(ascending=False).sort_index()
    pad_alignment['connectivity_order'] = ranks['mea1k_connectivity']
    # the best electrode is rounted if it has a connectivity > 80%
    pad_alignment['routed'] = ((pad_alignment['mea1k_connectivity']>20) & 
                               (pad_alignment['connectivity_order'] == 1.0)).fillna(False)
    
    # append the electrodes that are not under any pad    
    under_pad_els = mea1k_connectivity.index.intersection(pad_alignment.mea1k_el)
    mea1k_connectivity = mea1k_connectivity.drop(under_pad_els).reset_index()
    mea1k_connectivity.columns = ["mea1k_el", "mea1k_connectivity"]
    pad_alignment = pd.concat([pad_alignment, mea1k_connectivity], axis=0)
    
    order = ['pad_id', 'depth', 'shank_id',
             'y', 'x', 'metal', 'zone', 'y_order', 'x_order', 'r', 'g', 'b', 'a', 
             'shank_name', 'y_aligned', 'x_aligned', "el_pair", 'shank_side',
              'mea1k_el', 'mea1k_connectivity',  'connectivity_order','routed']
    pad_alignment = pad_alignment[order].sort_values(['pad_id','mea1k_el']).reset_index(drop=True)
    print('\n======\n', pad_alignment)
    
    
    # save the alignment
    #TODO should be in implante_device folder, together with connectivity measures
    #TODO add na for all 26400 el entries
    path = os.path.join(NAS_DIR, "electrode_devices", ELECTRODE_DEVICE_NAME, 
                        f'bonding_electrode_map_{ELECTRODE_DEVICE_NAME}.csv')
    pad_alignment.to_csv(path)
    
    
    
def plot_pad_alignment(implant_device_name):
    pad_alignment = pd.read_csv(f"{NAS_DIR}/implant_devices/{implant_device_name}"
                                f"/bonding_electrode_map_46pad4shank.csv", index_col=[0])
    (fig, ax), els = draw_mea1k()
    
    pad_circles = []
    for el_i, el_rec in enumerate(els):
        if el_i not in pad_alignment.mea1k_el.values:
            continue # not measured during connectivity analysis
        el_entry = pad_alignment[pad_alignment['mea1k_el'] == el_i].iloc[0]
        el_rec.set_alpha(min(1,el_entry.mea1k_connectivity/25))
        # print(el_entry)
        if pd.isna(el_entry.pad_id):
            continue
        col = pad_alignment[pad_alignment['mea1k_el'] == el_i][['r', 'g', 'b']].values[0]/255
        # el_rec.set_facecolor(col)
        el_rec.set_facecolor(np.array([1.,1.,1.])*min(1,el_entry.mea1k_connectivity/25))
        
        if el_entry.routed:
            # el_rec.set_linewidth(2)
            el_rec.set_edgecolor(col)
            
            pad_circles.append(plt.Circle((el_entry.x_aligned, el_entry.y_aligned), 
                                          26, color=col, fill=False, linewidth=.8))
    [ax.add_patch(pc) for pc in pad_circles]
    plt.show()
        
    
    
        

    
    
    
def main():
    # basepath = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/Simon/impedance/"
    # device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded_extCurrent_singleElAll'
    # device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded_extCurrent1024_rec2'
    # # device_name = 'device_headmount_new3EpoxyWalls/impedance_bonded2_D1_1KHz_1024_rec1'
    connectivty_measures_path = 'Simon/impedance/device_headmount_new3EpoxyWalls/impedance_bonded4_D0_1KHz_1024_rec1'
    ELECTRODE_DEVICE_NAME = '46pad4shank'
    # align_pads2mea1k(ELECTRODE_DEVICE_NAME, connectivty_measures_path)
    
    IMPLANT_DEVICE_NAME = '241016_headstage03_46pad4shank'
    plot_pad_alignment(IMPLANT_DEVICE_NAME)

    
    # PATH = basepath + device_name
    # print(PATH)
    # if not os.path.exists(PATH):
    #     print("Path does not exist: ", PATH)
    #     exit()

    # # align_pads(PATH)
    # make_mea1k_config(PATH)
    
    
    
if __name__ == '__main__':
    main()