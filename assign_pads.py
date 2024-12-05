import os
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import napari
from napari.utils import DirectLabelColormap

import ephys_constants as C
from analyze_connectivity import draw_mea1k

def align_pads2mea1k(ELECTRODE_DEVICE_NAME, IMPLANT_DEVICE_NAME, connectivty_measures_path):
    path = os.path.join(C.NAS_DIR, "electrode_devices", ELECTRODE_DEVICE_NAME, 
                        f"device_mapping_{ELECTRODE_DEVICE_NAME}.csv")
    mapping = pd.read_csv(path, index_col=[0])
    print(mapping)
    
    # path = os.path.join(C.NAS_DIR, connectivty_measures_path, 'connectivity_ampl.png')
    path = os.path.join(C.NAS_DIR, connectivty_measures_path, 'extr_connectivity_25.png')
    print(path)
    mea1k_connectivity_png = cv2.imread(path)[:,:,0] # black and white
    # path = os.path.join(C.NAS_DIR, connectivty_measures_path, 'ext_signal_ampl.csv')
    path = os.path.join(C.NAS_DIR, connectivty_measures_path, 'extr_connectivity_25.csv')
    mea1k_connectivity = pd.read_csv(path, index_col=[0]).sort_index()
    mea1k_connectivity.rename(columns={'connectivity': 'mea1k_connectivity'}, inplace=True)
    mea1k_connectivity.drop("ampl", inplace=True, axis=1)
    mea1k_connectivity.set_index('el', inplace=True)
    print(mea1k_connectivity)
    
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
             'y', 'x', 'metal', 'zone', 'y_order', 'x_order', 'pad_diameter_um', 'r', 'g', 'b', 'a', 
             'shank_name', 'y_aligned', 'x_aligned', "el_pair", 'shank_side',
              'mea1k_el', 'mea1k_connectivity',  'connectivity_order','routed']
    # missed electrodes during connectivity analysis
    missing_els = np.setdiff1d(np.arange(26400), pad_alignment.mea1k_el)
    pad_alignment = pd.concat([pad_alignment, pd.DataFrame({"mea1k_el":missing_els})], axis=0)
    pad_alignment = pad_alignment[order].sort_values(['pad_id','mea1k_el']).reset_index(drop=True)
    print('\n======\n', pad_alignment)
    
    # save the alignment
    fullfname = os.path.join(C.NAS_DIR, "implant_devices", IMPLANT_DEVICE_NAME, "bonding",
                             f'bonding_mapping_{IMPLANT_DEVICE_NAME}.csv')
    pad_alignment.to_csv(fullfname, index=False)
    
def plot_pad_alignment(IMPLANT_DEVICE_NAME):
    fullfname = os.path.join(C.NAS_DIR, "implant_devices", IMPLANT_DEVICE_NAME, "bonding",
                             f'bonding_mapping_{IMPLANT_DEVICE_NAME}.csv')
    pad_alignment = pd.read_csv(fullfname)
    
    (fig, ax), els = draw_mea1k()
    
    pad_circles = []
    for el_i, el_rec in enumerate(els):
        if el_i not in pad_alignment.mea1k_el.values:
            continue # not measured during connectivity analysis
        el_entry = pad_alignment[pad_alignment['mea1k_el'] == el_i].iloc[0]
        el_rec.set_alpha(min(1,el_entry.mea1k_connectivity/25))
        if pd.isna(el_entry.pad_id):
            continue
        col = pad_alignment[pad_alignment['mea1k_el'] == el_i][['r', 'g', 'b']].values[0]/255
        el_rec.set_facecolor(col)
        
        if (el_entry.connectivity_order == 1) and (el_entry.mea1k_connectivity > .8):
            el_rec.set_linewidth(2)
            el_rec.set_edgecolor("white")
            
            pad_circles.append(plt.Circle((el_entry.x_aligned, el_entry.y_aligned), 
                                          el_entry.pad_diameter_um/2 *1.7, color=col, 
                                          fill=False, linewidth=.8,
                                          alpha=.75))
    [ax.add_patch(pc) for pc in pad_circles]
    fig.savefig(fullfname.replace("csv", "png"), dpi=300, transparent=True, 
                bbox_inches='tight', pad_inches=0)
    plt.show()
    
def main():
    # ELECTRODE_DEVICE_NAME = 'H1278pad4shank'
    # HEADSTAGE_DEVICE_NAME = 'MEA1K03'
    # date = '241016'
    # batch = 5
    # IMPLANT_DEVICE_NAME = f"{date}_{HEADSTAGE_DEVICE_NAME}_{ELECTRODE_DEVICE_NAME}B{batch}"
    # rec_dir_name = 'impedance_bonded4_D0_1KHz_1024_rec1'
    # connectivity_rec_path = os.path.join(C.NAS_DIR, 'implant_devices', 
    #                                      IMPLANT_DEVICE_NAME, 'recordings', 
    #                                      rec_dir_name)
    # # align_pads2mea1k(ELECTRODE_DEVICE_NAME, IMPLANT_DEVICE_NAME, connectivity_rec_path)
    # plot_pad_alignment(IMPLANT_DEVICE_NAME)
    
    
    ELECTRODE_DEVICE_NAME = 'H1628pad1shank'
    HEADSTAGE_DEVICE_NAME = 'MEA1K05'
    date = '241129'
    batch = 5
    IMPLANT_DEVICE_NAME = f"{date}_{HEADSTAGE_DEVICE_NAME}_{ELECTRODE_DEVICE_NAME}B{batch}"
    # rec_dir_name = f"{date}_ext25mV_1KHz_allels"
    rec_dir_name = '25mVext_oneShankbatch2_press/processed'
    connectivity_rec_path = os.path.join(C.NAS_DIR, 'implant_devices', 
                                         IMPLANT_DEVICE_NAME, 'recordings', 
                                         rec_dir_name)
    align_pads2mea1k(ELECTRODE_DEVICE_NAME, IMPLANT_DEVICE_NAME, connectivity_rec_path)
    plot_pad_alignment(IMPLANT_DEVICE_NAME)
    
if __name__ == '__main__':
    main()