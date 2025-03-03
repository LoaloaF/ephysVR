import os
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import napari
from napari.utils import DirectLabelColormap

import ephys_constants as C
from ephys_constants import device_paths
from mea1k_viz import draw_mea1k

def _align(mapping, mea1k_connectivity, alignment, split_el_device, alignment2=None):
    # parse the alligned pads and match them to mea1k electrodes under them
    pad_alignment = []
    for pad_id, row in mapping.iterrows():
        if not (split_el_device and pad_id > 638):
            # get xy coordinates of the pad
            new_pad_yx_um = np.stack(np.where(alignment.data == pad_id)).T
            # transofrm to mea1k coordate system (world coo)
            new_pad_yx_um = [alignment.data_to_world(xy) for xy in new_pad_yx_um]
        else:
            # same for the second half of the interconnect
            new_pad_yx_um = np.stack(np.where(alignment2.data == pad_id)).T
            new_pad_yx_um = [alignment2.data_to_world(xy) for xy in new_pad_yx_um]
        
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
        mea1k_els_under_pad_mask = (C.MEA1K_EL_2D_TABLE_PIXEL.index.isin(new_pad_yx_pixels.tolist()))
        mea1k_el_underpad = C.MEA1K_EL_2D_TABLE_PIXEL[mea1k_els_under_pad_mask]
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
    pad_alignment = pd.concat(pad_alignment, axis=0).reset_index(drop=True)
    return pad_alignment


def _add_pads_to_napari(viewer, mea1k_connectivity_png, mapping):
    # draw circles (connectivity pads)
    pad_colors = {-1: (0,0,0,0)}
    pads_canvas = np.ones_like(mea1k_connectivity_png, dtype=np.int16) * -1
    for pad_id, row in mapping.iterrows():
        x,y = row[['x', 'y']]
        # oversize pads by 70%
        pad_r = int(row.pad_diameter_um/2 * 1.7)
        cv2.circle(pads_canvas, (x,y), pad_r, pad_id, -1)
        pad_colors[pad_id] = (row[['r','g','b']].values.astype(float)/255.)
    pad_colors = DirectLabelColormap(color_dict=pad_colors)
        
    # align manually the pads to the mea1k electrodes
    alignment = viewer.add_labels(pads_canvas, colormap=pad_colors)
    return alignment
    
def align_pads2mea1k(electrode_device_name, IMPLANT_DEVICE_NAME, 
                     connectivty_measures_path, split_el_device=False):
    nas_dir = device_paths()[0]
    mapping_fullfname = os.path.join(nas_dir, 'devices', "electrode_devices", electrode_device_name, 
                                     f"device_mapping_{electrode_device_name}.csv")
    mapping = pd.read_csv(mapping_fullfname, index_col=[0])
    
    connectivity_fullfname = os.path.join(nas_dir, connectivty_measures_path, 
                                          'processed', 'extr_connectivity.png')
    mea1k_connectivity_png = cv2.imread(connectivity_fullfname)[:,:,0] # black and white
    
    # path = os.path.join(nas_dir, connectivty_measures_path, 'ext_signal_ampl.csv')
    path = connectivity_fullfname.replace(".png", ".csv")
    mea1k_connectivity = pd.read_csv(path, index_col=[0]).sort_index()
    mea1k_connectivity.rename(columns={'connectivity': 'mea1k_connectivity'}, inplace=True)
    mea1k_connectivity.drop("ampl", inplace=True, axis=1)
    mea1k_connectivity.set_index('el', inplace=True)
    
    # drop duplicates issue on one specific connectivity measurement...
    mea1k_connectivity = mea1k_connectivity[~mea1k_connectivity.index.duplicated(keep='first')]
    mea1k_connectivity = mea1k_connectivity.reindex(np.arange(26400))

    viewer = napari.Viewer()
    viewer.add_image(mea1k_connectivity_png, name='Connectivity', colormap='gray_r',
                     opacity=.5)
    
    if split_el_device:
        # two half s that need to be aligned seperately
        alignment = _add_pads_to_napari(viewer, mea1k_connectivity_png, mapping[mapping.index<=638])
        alignment2 = _add_pads_to_napari(viewer, mea1k_connectivity_png, mapping[mapping.index>638])
    else:
        alignment = _add_pads_to_napari(viewer, mea1k_connectivity_png, mapping)
        alignment2 = None
        
    # do the alignment manually
    napari.run()
    # done
    
    # assign the pads to the mea1k electrodes according to the placement of 
    # the interconnect/ bonding
    pad_alignment = _align(mapping, mea1k_connectivity, alignment, 
                           split_el_device, alignment2)
    
    # group by pad_id and rank the electrodes by connectivity
    ranks = pad_alignment[['pad_id', 'mea1k_connectivity']]
    ranks = ranks.sort_values(['pad_id', 'mea1k_connectivity'], 
                              ascending=[True, False])
    ranks = ranks.groupby('pad_id').rank(ascending=False).sort_index()
    pad_alignment['connectivity_order'] = ranks['mea1k_connectivity']
    
    # the best electrode is rounted if it has a connectivity > 80%
    # pad_alignment['routed'] = ((pad_alignment['mea1k_connectivity']>0.8) & 
    #                            (pad_alignment['connectivity_order'] == 1.0) &
    #                            (pad_alignment['shank_id'].notna())).fillna(False)
    
    # append the electrodes that are not under any pad
    under_pad_els = mea1k_connectivity.index.intersection(pad_alignment.mea1k_el)
    mea1k_connectivity = mea1k_connectivity.drop(under_pad_els).reset_index()
    mea1k_connectivity.columns = ["mea1k_el", "mea1k_connectivity", 'input_mV']
    pad_alignment = pd.concat([pad_alignment, mea1k_connectivity], axis=0)
    
    order = ['pad_id', 'depth', 'shank_id',
             'y', 'x', 'metal', 'zone', 'y_order', 'x_order', 'pad_diameter_um', 'r', 'g', 'b', 'a', 
             'shank_name', 'y_aligned', 'x_aligned', "el_pair", 'shank_side',
             'mea1k_el', 'mea1k_connectivity',  'connectivity_order']
    # missed electrodes during connectivity analysis
    missing_els = np.setdiff1d(np.arange(26400), pad_alignment.mea1k_el)
    pad_alignment = pd.concat([pad_alignment, pd.DataFrame({"mea1k_el":missing_els})], axis=0)
    pad_alignment = pad_alignment[order].sort_values(['pad_id','mea1k_el']).reset_index(drop=True)
    print('\n======\n', pad_alignment)
    
    # save the alignment
    fullfname = os.path.join(nas_dir, 'devices', "implant_devices", IMPLANT_DEVICE_NAME, 
                             "bonding", f'bonding_mapping_{IMPLANT_DEVICE_NAME}.csv')
    pad_alignment.to_csv(fullfname, index=False)
    
def plot_pad_alignment(IMPLANT_DEVICE_NAME):
    nas_dir = device_paths()[0]
    fullfname = os.path.join(nas_dir, "devices", "implant_devices", 
                             IMPLANT_DEVICE_NAME, "bonding",
                             f'bonding_mapping_{IMPLANT_DEVICE_NAME}.csv')
    pad_alignment = pd.read_csv(fullfname)
    
    (fig, ax), els = draw_mea1k()
    
    pad_circles = []
    for el_i, el_rec in enumerate(els):
        if el_i not in pad_alignment.mea1k_el.values:
            continue # not measured during connectivity analysis
        el_entry = pad_alignment[pad_alignment['mea1k_el'] == el_i].iloc[0]
        el_rec.set_alpha(min(1,el_entry.mea1k_connectivity/3))
        if pd.isna(el_entry.pad_id):
            continue
        
        col = pad_alignment[pad_alignment['mea1k_el'] == el_i][['r', 'g', 'b']].values[0]/255
        # el_rec.set_facecolor(col)

        whiteness = np.clip(el_entry.mea1k_connectivity*1, 0, 1)
        el_rec.set_facecolor((whiteness, whiteness, whiteness))
        
        if (el_entry.connectivity_order == 1) and (el_entry.mea1k_connectivity > .8) and el_entry.notna()["shank_id"]:
            el_rec.set_alpha(1)
            el_rec.set_facecolor(col)
            pad_circles.append(plt.Circle((el_entry.x_aligned, el_entry.y_aligned), 
                                          el_entry.pad_diameter_um/2 *1.7, color=col, 
                                          fill=False, linewidth=.8,
                                          alpha=.5))
    [ax.add_patch(pc) for pc in pad_circles]
    fig.savefig(fullfname.replace("csv", "png"), dpi=300, transparent=False, 
                bbox_inches='tight', pad_inches=0)
    plt.show()
    
def main():
    nas_dir = device_paths()[0]
    # rat 006
    # ELECTRODE_DEVICE_NAME = 'H1278pad4shank'
    # HEADSTAGE_DEVICE_NAME = 'MEA1K03'
    # date = '241016'
    # batch = 5
    # IMPLANT_DEVICE_NAME = f"{date}_{HEADSTAGE_DEVICE_NAME}_{ELECTRODE_DEVICE_NAME}B{batch}"
    # rec_dir_name = 'impedance_bonded4_D0_1KHz_1024_rec1'
    # connectivity_rec_path = os.path.join(nas_dir, 'implant_devices', 
    #                                      IMPLANT_DEVICE_NAME, 'recordings', 
    #                                      rec_dir_name)
    # # align_pads2mea1k(ELECTRODE_DEVICE_NAME, IMPLANT_DEVICE_NAME, connectivity_rec_path)
    # plot_pad_alignment(IMPLANT_DEVICE_NAME)
    
    # testing on high density interconnect
    # ELECTRODE_DEVICE_NAME = 'H1628pad1shank'
    # HEADSTAGE_DEVICE_NAME = 'MEA1K05'
    # date = '241129'
    # batch = 5
    # IMPLANT_DEVICE_NAME = f"{date}_{HEADSTAGE_DEVICE_NAME}_{ELECTRODE_DEVICE_NAME}B{batch}"
    # # rec_dir_name = f"{date}_ext25mV_1KHz_allels"
    # rec_dir_name = '25mVext_oneShankbatch2_press/processed'
    # connectivity_rec_path = os.path.join(nas_dir, 'implant_devices', 
    #                                      IMPLANT_DEVICE_NAME, 'recordings', 
    #                                      rec_dir_name)
    # align_pads2mea1k(ELECTRODE_DEVICE_NAME, IMPLANT_DEVICE_NAME, connectivity_rec_path)
    # plot_pad_alignment(IMPLANT_DEVICE_NAME)
    
    # # rat 011
    # ELECTRODE_DEVICE_NAME = 'H1278pad4shank'
    # HEADSTAGE_DEVICE_NAME = 'MEA1K06'
    # date = '241211'
    # batch = 5
    # IMPLANT_DEVICE_NAME = f"{date}_{HEADSTAGE_DEVICE_NAME}_{ELECTRODE_DEVICE_NAME}B{batch}"
    # rec_dir_name = 'bonding5_4shank_B6_241211_ext5mV1Khz_silk_rec3'
    # connectivity_rec_path = os.path.join(nas_dir, 'implant_devices', 
    #                                      IMPLANT_DEVICE_NAME, 'recordings', 
    #                                      rec_dir_name)
    # # align_pads2mea1k(ELECTRODE_DEVICE_NAME, IMPLANT_DEVICE_NAME, connectivity_rec_path)
    # plot_pad_alignment(IMPLANT_DEVICE_NAME)
    
    # rat 010
    ELECTRODE_DEVICE_NAME = 'H1278pad4shank'
    HEADSTAGE_DEVICE_NAME = 'MEA1K03'
    date = '250205'
    batch = 5
    IMPLANT_DEVICE_NAME = f"{date}_{HEADSTAGE_DEVICE_NAME}_{ELECTRODE_DEVICE_NAME}B{batch}"
    rec_dir_name = 'bonding2_250205_D9.2_25mVext_2_2Shankbatch5_silk'
    split_el_device = True
    connectivity_rec_path = os.path.join(nas_dir, 'devices', 'implant_devices', 
                                         IMPLANT_DEVICE_NAME, 'recordings', 
                                         rec_dir_name)
    align_pads2mea1k(ELECTRODE_DEVICE_NAME, IMPLANT_DEVICE_NAME, connectivity_rec_path,
                     split_el_device)
    # plot_pad_alignment(IMPLANT_DEVICE_NAME)
    
if __name__ == '__main__':
    main()