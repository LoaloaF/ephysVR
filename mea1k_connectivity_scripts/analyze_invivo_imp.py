import os
import sys

# to import from parent dir
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from CustomLogger import CustomLogger as Logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import ephys_constants as C
from mea1k_modules.mea1k_raw_preproc import read_raw_data
from mea1k_modules.mea1k_raw_preproc import read_stim_DAC
from mea1k_modules.mea1k_raw_preproc import get_raw_implant_mapping
from mea1k_modules.mea1k_raw_preproc import get_recording_implant_mapping
from signal_helpers import estimate_frequency_power, calculate_phase_shift

from mea1k_modules.mea1k_visualizations import vis_shank_traces, viz_mea1k_config, draw_interconnect_pads
from mea1k_modules.mea1k_visualizations import draw_mea1k
from mea1k_modules.mea1k_visualizations import draw_mea1K_colorbar

def get_hdf5_fnames_from_dir(subdir):
    fnames, ids = [], []
    for fname in sorted(os.listdir(subdir)):
        if fname.endswith('raw.h5'):
            fnames.append(fname)
            # check 4 digit case...
            pruned_fname = fname.replace('.raw.h5', '')
            id_str = pruned_fname.split('_')[-1]
            ids.append(id_str)
            # try:
            #     ids.append(int(id_str))
            # except ValueError:
            #     print(f"Error: {id_str} is not a valid id. Full name: {fname}")
            #     exit(1)
    return fnames, ids

def save_output(subdir, data, fname):
    fullpath = os.path.join(subdir, "processed")
    if not os.path.exists(fullpath):
        print("creating processed output dir: ", fullpath)
        os.makedirs(fullpath)
    print("Saving to: ", os.path.join(fullpath, fname))
    data.to_csv(os.path.join(fullpath, fname))

def extract_impedance(subdir, implant_name, current_ampl_nA, debug=False, deepdebug=False):
    L = Logger()
    fnames, ids = get_hdf5_fnames_from_dir(subdir)
    aggr_imp_data = []
    for fname, i in zip(fnames, ids):
        L.logger.info(f"Config {i},{fname} of {len(fnames)}")
        
        if int(i[-4:]) <5:
            continue
        # get the config information about this configuration
        stimulated = pd.read_csv(os.path.join(subdir, fname.replace(".raw.h5", ".csv")))


        dac = read_stim_DAC(subdir, fname)
        # stim_sample_ids = np.where(dac != 512)[0]
        # shortcut, since we know the stim samples are between 20500 and 29500
        stim_sample_ids = (20500, 29500)
        data = read_raw_data(subdir, fname, convert2uV=True,
                            subtract_dc_offset=False, 
                            col_slice=slice(None, None))
                            # col_slice=slice(stim_sample_ids[0], stim_sample_ids[-1], None))
        # )
        
        # Calculate phase shifts
        phase_shifts = calculate_phase_shift(data[:, 20500:29500], dac[20500:29500]* -1, 
                                             sampling_rate=C.SAMPLING_RATE, 
                                             freq=1000, debug=deepdebug)
    
        mean_ampl = []
        for j,row in enumerate(data):
            _, m_ampl = estimate_frequency_power(row[stim_sample_ids[0]:stim_sample_ids[1]].astype(float), 
                                                 sampling_rate=C.SAMPLING_RATE, 
                                                 debug=deepdebug, 
                                                 min_band=960, max_band=1040)
            if stimulated.stim[j]:
                pass
            mean_ampl.append(m_ampl)
        mean_ampl = np.array(mean_ampl)
        
        stimulated['phse_shift'] = phase_shifts
        stimulated['imp_voltage_uV'] = mean_ampl
        stimulated['imp_kOhm'] = (mean_ampl / (current_ampl_nA * 1e-3)) / 1e3 #* stimulated.stim.astype(int)
        stimulated['imp_stim_ratio'] = mean_ampl/ mean_ampl[stimulated.stim].item()
        stimulated.drop(columns=['channel', 'x', 'y'], inplace=True)
        stimulated.index = pd.MultiIndex.from_product([[fname.replace(".raw.h5","")],
                                                        stimulated.index], names=['config', 'el'])
        L.logger.debug(stimulated)
        L.logger.info(f"Imp: {stimulated['imp_kOhm'][stimulated.stim].item():.2f} kOhm")
        aggr_imp_data.append(stimulated)
        
        if debug:
            mapping = get_recording_implant_mapping(subdir, fname, implant_name=implant_name,
                                                    drop_non_bonded=False)
            data = data[mapping.shank_id == 1] # or 1 
            mapping = mapping[mapping.shank_id == 1] # or 1
            viz_mea1k_config(mapping, stim_mea1k_el=stimulated.electrode[stimulated.stim].item(),
                             col_scaler=.6)
            # skip the first 100 samples
            vis_shank_traces(data[:, 200:]-data[:,0:1], mapping, stim_mea1k_el=stimulated.electrode[stimulated.stim].item(), 
                             scaler=1/3_000, uVrange=470_000, stimulated=stimulated)
            
            plt.figure(figsize=(18, 8))
            plt.xlabel("Time [ms]")
            plt.subplot(2, 1, 1)
            plt.plot(data.T/1000)
            plt.ylabel("Voltage [mV]")
            plt.title(f"PadID: {stimulated.pad_id[stimulated.stim].item()}")
            
            plt.subplot(2, 1, 2, sharex=plt.gca())              
            plt.plot(read_stim_DAC(subdir, fname)[200:])
        plt.show()
 
    
    aggr_imp_data = pd.concat(aggr_imp_data)
    save_output(subdir, aggr_imp_data, "extracted_imp_voltages.csv")



    
def vis_impedance(subdir, implant_name):
    def draw_bounary_lines(stim_el_imp_data, cmap, norm):
        # draw stim unit boundaries
        for i, stim_unit in enumerate(stim_el_imp_data.stim_unit.unique()):
            plt.axvline(np.where(stim_el_imp_data.stim_unit == stim_unit)[0][0], color='k', 
                        linestyle='dashed', alpha=.4)
            plt.text(np.where(stim_el_imp_data.stim_unit == stim_unit)[0][0]+5, 1e4, 
                    f"Stim Unit: {int(stim_unit):2d}", color=cmap(norm(shuffle_remapping[stim_unit])),
                    rotation=90, fontsize=11, ha='left', va='top')
                    
                    
                    
    all_els_imp_data = pd.read_csv(os.path.join(subdir, "processed", "extracted_imp_voltages.csv"))
    all_els_imp_data = all_els_imp_data.drop(columns=['Unnamed: 0']).set_index('electrode', drop=True)
    
    # for a normal recorindg, implant name will get the correct mapping
    implant_mapping = get_raw_implant_mapping(implant_name=implant_name).set_index("mea1k_el", drop=True)
    # replacement if there is no real mapping, use the single shank config mapping, even if in a differnt folder 
    # implant_mapping = get_raw_implant_mapping(implant_name="250308_MEA1K07_H1628pad1shankB6").set_index("mea1k_el", drop=True)

    dupl_columns = all_els_imp_data.columns.intersection(implant_mapping.columns)
    all_els_imp_data = pd.merge(all_els_imp_data.drop(columns=dupl_columns),  
                               implant_mapping, how='left', left_index=True, right_index=True)

    # electrode index is not unique, keep as column (called mea1k_el)
    all_els_imp_data.reset_index(inplace=True, drop=True)
    
    # subset to only the stimulated els
    stim_el_imp_data = all_els_imp_data[all_els_imp_data.stim_unit.notna()].copy().reset_index(drop=True)

    # aggreate over Mea1k electrodes to form pad data
    all_pads_imp_data = all_els_imp_data.groupby(['config', 'pad_id']).agg(
        {'imp_kOhm': ['mean', 'std'],      # Mean and variance of impedance
         'stim_unit': 'max',
         'phse_shift': 'mean',
         'imp_stim_ratio': 'mean',         # Mean stimulation ratio
         'imp_voltage_uV': 'mean',         # Mean voltage
         'depth': 'first',                 # Mean depth
        #  'shank': 'first',                 # First shank
         'shank_id': 'first',              # First shank ID
         'metal': 'first',                 # First metal type
         'el_pair': 'first',               # First electrode pair
         'shank_side': 'first',            # First shank side
         'r': 'first',                     # Mean red color value
         'g': 'first',                     # Mean green color value
         'b': 'first',                     # Mean blue color value
         'a': 'first',                     # Mean alpha value
         'mea1k_connectivity': 'mean',     # Mean connectivity
         })
    # variance across els within pad
    all_pads_imp_data[('imp_kOhm_std', '_')] = all_pads_imp_data[('imp_kOhm', 'std')]
    all_pads_imp_data.columns = all_pads_imp_data.columns.droplevel(1)
    
    # impedance color 
    cmap_plasma = plt.get_cmap('plasma')
    norm_imp_col = plt.Normalize(vmin=0, vmax=6.5*1e2)
    stim_el_imp_data['imp_color'] = stim_el_imp_data.apply(
        lambda x: cmap_plasma(norm_imp_col(x.imp_kOhm))[:3], axis=1).values

    # stim unit color
    cmap_tab = plt.get_cmap('tab20')
    norm_stim_col = plt.Normalize(vmin=0, vmax=32)
    shuffle_remapping = np.random.permutation(32)
    shuffle_remapping = {old:new for old, new in zip(np.arange(32), shuffle_remapping)}
    stim_el_imp_data['stimunit_color'] = stim_el_imp_data.apply(
        lambda x: cmap_tab(norm_stim_col(shuffle_remapping[x.stim_unit]))[:3], axis=1).values
    
    
    draw_interconnect_imp = True
    draw_interconnect_stim_unit = True
    
    if draw_interconnect_imp:
        (fig, ax), el_rects = draw_mea1k(mapping=implant_mapping)
        imp_color_dict = {r.pad_id: r.imp_color for _, r in stim_el_imp_data.iterrows()}
        draw_interconnect_pads(implant_mapping, edgecolor=imp_color_dict, 
                               draw_on_ax=ax, pad_alpha=0, add_pad_label=False)
        draw_mea1K_colorbar(cmap_plasma, norm_imp_col, 'Impedance [kOhm]', orientation='vertical')
        plt.savefig(os.path.join(subdir, "processed", "imp_interconnect.png"))
        
    if draw_interconnect_stim_unit:
        (fig, ax), el_rects = draw_mea1k(mapping=implant_mapping)
        stim_unit_color_dict = {r.pad_id: r.stimunit_color for _, r in stim_el_imp_data.iterrows()}
        draw_interconnect_pads(implant_mapping, edgecolor=stim_unit_color_dict, 
                               draw_on_ax=ax, pad_alpha=.7, add_pad_label=True)
        draw_mea1K_colorbar(cmap_tab, norm_stim_col, 'Stim Unit', orientation='vertical')
        plt.savefig(os.path.join(subdir, "processed", "stimunit_interconnect.png"))
        

    
    
    stim_el_imp_data = stim_el_imp_data.sort_values(['stim_unit', 'imp_kOhm'])
    
    # Reorder that data by the highest average impedance per stim unit
    stimunit_imp = stim_el_imp_data.groupby('stim_unit').agg({'imp_kOhm': 'mean'}).sort_values(by='imp_kOhm', ascending=False)
    stimunit_rank = stimunit_imp.reset_index().reset_index().set_index('stim_unit')['index']
    stim_el_imp_data['stim_unit_rank'] = stim_el_imp_data['stim_unit'].map(stimunit_rank)
    reorder = stim_el_imp_data.sort_values(by=['stim_unit_rank', 'imp_kOhm']).index
    # reorder
    stim_el_imp_data = stim_el_imp_data.loc[reorder]
    print("Post reorder stim unit impedance data")
    print(stim_el_imp_data.stim_unit)
    print(list(stim_el_imp_data.stim_unit.unique()))
    

    draw_pad_id_ordered_imp = True
    draw_stim_unit_ordered_imp = False
    
    if draw_stim_unit_ordered_imp:
        fig = plt.figure(figsize=(18, 8))
        plt.xlabel("Pad ID on MEA1K  |  Polyimide el. ID in brain")
        plt.ylabel("Impedance [kOhm]")
        plt.yscale('log')
        # ax.set_ylim(10, 1e4)
        
        plt.ylim(1e1, 1e4)
        plt.yscale('log')
        plt.axhspan(200, 400, color='gray', alpha=.06, zorder=0)
        
        plt.title("Impedance by Stim Unit, stim-unit order")
        plt.scatter(np.arange(len(stim_el_imp_data)), 
                    stim_el_imp_data.imp_kOhm,
                    color=stim_el_imp_data.stimunit_color, marker='o', s=40, alpha=.3)
        draw_bounary_lines(stim_el_imp_data, cmap_tab, norm_stim_col)
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, "processed", "stim_unit_ordered_imp.png"))
        
        
    if draw_pad_id_ordered_imp:
        fig = plt.figure(figsize=(18, 8))
        plt.xlabel("Pad ID on MEA1K  |  Polyimide el. ID in brain")
        plt.ylabel("Impedance [kOhm]")
        plt.yscale('log')
        
        plt.ylim(1e1, 1e4)
        plt.yscale('log')
        plt.axhspan(200, 400, color='gray', alpha=.06, zorder=0)
        
        plt.title("Impedance by Stim Unit, pad order")
        plt.scatter(stim_el_imp_data.pad_id, stim_el_imp_data.imp_kOhm,
                    color=stim_el_imp_data.stimunit_color, marker='o', s=40, alpha=.3)
        plt.tight_layout()
        plt.savefig(os.path.join(subdir, "processed", "pad_ordered_imp.png"))
        

    # exclude not routed pads
    # stim_el_imp_data = stim_el_imp_data[stim_el_imp_data.shank_id.notna()]
    # reorder again back to pad order
    # stim_el_imp_data = stim_el_imp_data.sort_values(['pad_id', 'stim_unit'])
    
       
    draw_stim_pad_var = True     
    if draw_stim_pad_var:
        fig = plt.figure(figsize=(18, 8))
        plt.xlabel("Pad ID on MEA1K  |  Polyimide el. ID in brain")
        plt.ylabel("Impedance [kOhm]")
        
        plt.ylim(1e1, 1e4)
        plt.yscale('log')
        plt.axhspan(200, 400, color='gray', alpha=.06, zorder=0)
        
        draw_bounary_lines(stim_el_imp_data, cmap_tab, norm_stim_col)
        ticks = []
        for i, row in stim_el_imp_data.reset_index().iterrows():
            print(i, end='...')
            # slice to the config where the el was stimulued, get other pad data
            all_els_dat = all_els_imp_data[all_els_imp_data.config == row.config]
            ticks.append(row.pad_id)
            depth_alpha = np.clip(row.depth/6000, 0, 1)
            
            if pd.isna(row.pad_id):
                continue
            
            # if row.pad_id == 0:
            #     print(all_els_dat.loc[:, "imp_voltage_uV", 'pad_id'])
            #     print(all_els_dat.loc[:, "imp_kOhm", 'pad_id'])
            # else:
            #     continue

            
            # calculate the mean and std of the impedance for all pads in the config, across mea1k els
            config_pad_imp_std = all_els_dat.groupby('pad_id').agg({'imp_kOhm': 'std'}).fillna(0)
            config_pad_imp_mean = all_els_dat.groupby('pad_id').agg({'imp_kOhm': 'mean', 
                                                                     'el_pair': 'first',
                                                                     'depth': 'first',})
            # seperate out the stimulated pad
            config_stim_pad_std = config_pad_imp_std.loc[row.pad_id].imp_kOhm
            config_stim_pad_mean = config_pad_imp_mean.loc[row.pad_id].imp_kOhm
            config_pad_imp_std.drop(row.pad_id, inplace=True)
            config_pad_imp_mean.drop(row.pad_id, inplace=True)
            
            # # draw stimulated pad average over mea1k els
            # plt.scatter([i], config_stim_pad_mean, color=row.stimunit_color,
            #             marker='o', s=25, alpha=.7)
            # plt.errorbar([i], config_stim_pad_mean, yerr=config_stim_pad_std,
            #              color=row.stimunit_color, alpha=.7)
            
            # draw other pads, only if they have > 20% of the stim pad impednace
            low_imp_mask = config_pad_imp_mean.imp_kOhm.values/config_stim_pad_mean > 0.2
            config_pad_imp_mean = config_pad_imp_mean[low_imp_mask]
            config_pad_imp_std = config_pad_imp_std[low_imp_mask]
            # green, * replacement 
            el_pair_color = ['blue' if row.el_pair == el_pair else 'red' for el_pair in config_pad_imp_mean.el_pair]
            el_pair_marker = ['*' if row.el_pair == el_pair else 'x' for el_pair in config_pad_imp_mean.el_pair]

            # Loop through each pad and plot individually
            for j, (mean, color, marker) in enumerate(zip(config_pad_imp_mean.imp_kOhm, el_pair_color, el_pair_marker)):
                plt.scatter([i], [mean], color='k', alpha=0.7, s=25, marker='o')
                # plt.scatter([i], [mean], color=color, alpha=0.7, s=25, marker=marker)
            plt.errorbar([i]*len(config_pad_imp_mean), config_pad_imp_mean.imp_kOhm.values.flatten(), yerr=config_pad_imp_std.values.flatten(), color='gray', alpha=.3)
            
            # onlly stimulated mea1k el
            plt.scatter([i], [row.imp_kOhm],
                        color='k', marker='o', s=6, alpha=.7)
            
        plt.xticks(np.arange(len(ticks)), ticks, rotation=90, fontsize=7)
        
    plt.tight_layout()
    
def compare_impedance(subdirs, implant_name):
    for subdir in subdirs:
        data = pd.read_csv(os.path.join(subdir, "processed", "extracted_imp_voltages.csv"))
        data = data[data.stim_unit.notna()]
        data.sort_values(by='pad_id', inplace=True)
        imp = np.clip(data.imp_kOhm.values, 0, 6.5*1e3)
        [plt.axvline(x, color='k', linestyle='dashed') for x in [0, 106, 814, 920]]
        plt.scatter(data.pad_id, imp, alpha=.6, label=subdir.split('/')[-1])
    plt.legend()
    plt.show()
    
def main(): 
    L = Logger()
    L.init_logger(None, None, "DEBUG")
    L.logger.debug("Starting in vivo impedance analysis")
    nas_dir = C.device_paths()[0]
    
    # fix seed
    np.random.seed(42)
    
    # implant_name = "241016_MEA1K03_H1278pad4shankB5"
    # implant_name = "4983"
    implant_name = "250308_MEA1K07_H1628pad1shankB6"
    # implant_name = "241211_MEA1K06_H1278pad4shankB5"
    # implant_name = "250205_MEA1K03_H1278pad4shankB5"
    current_ampl_nA = 20 
    
    subdirs = [
        # f"devices/implant_devices/{implant_name}/recordings/2025-03-25_15.32.21_invivo_imp_mode='small_current'_stimpulse='sine'_amplitude=10",
        
        # 1 shank device
        # well device (just using shank1 configs)
        f"devices/well_devices/{4983}/recordings/11.17.35_singleshankConfigs_test_mode='small_current'_stimpulse='sine'2_amplitude=10",
        # last in nitro imp measurement
        # f"devices/implant_devices/{implant_name}/recordings/13.36.27_Ringer_postwelldevice_mode='small_current'_stimpulse='sine'2_amplitude=10",
        # after el removal
        # f"devices/implant_devices/{implant_name}/recordings/13.37.39_Ringer_postElectrRemoved_GNDREF_mode='small_current'_stimpulse='sine'2_amplitude=10",
    ]
    
    
    # el_config_S1D1650.raw.h5
    extract_impedance(os.path.join(nas_dir, subdirs[0]), implant_name=implant_name, 
                      current_ampl_nA=current_ampl_nA, debug=True)
    # print(os.path.join(nas_dir, subdirs[0]))
    # vis_impedance(os.path.join(nas_dir, subdirs[0]), implant_name=implant_name)
    # compare_impedance([os.path.join(nas_dir, subdir) for subdir in subdirs], implant_name=implant_name)

    plt.show()
    
if __name__ == "__main__":
    main()
    
# # quick comparison of stim unit ordering between chips
# # with electrode mea1k7
# mea1k7_with_els =     [7.0, 1.0, 4.0, 6.0, 9.0, 8.0, 23.0, 3.0, 0.0, 29.0, 19.0, 5.0, 2.0, 16.0, 28.0, 21.0, 10.0, 22.0, 13.0, 26.0, 30.0, 20.0, 17.0, 27.0, 24.0, 18.0, 12.0, 25.0, 11.0, 14.0, 15.0, 31.0]
# # post el removal mea1k7, but ligquid dried up half way    
# mea1k7_post_removal = [4.0, 6.0, 7.0, 1.0, 2.0, 9.0, 16.0, 0.0, 19.0, 3.0, 29.0, 23.0, 8.0, 5.0, 22.0, 21.0, 17.0, 10.0, 28.0, 26.0, 13.0, 30.0, 20.0, 18.0, 27.0, 24.0, 12.0, 25.0, 11.0, 14.0, 15.0, 31.0]
# # well device
# well_device = [26.0, 2.0, 20.0, 0.0, 17.0, 6.0, 21.0, 8.0, 19.0, 27.0, 25.0, 18.0, 24.0, 7.0, 4.0, 23.0, 22.0, 16.0, 1.0, 11.0, 10.0, 9.0, 3.0, 5.0, 29.0, 12.0, 28.0, 13.0, 15.0, 14.0, 31.0, 30.0]
# # mea1k03 animal10
# mea1k03_animal10 = [3.0, 22.0, 25.0, 23.0, 17.0, 6.0, 21.0, 10.0, 19.0, 7.0, 9.0, 24.0, 0.0, 8.0, 11.0, 20.0, 1.0, 26.0, 4.0, 2.0, 27.0, 16.0, 12.0, 5.0, 18.0, 15.0, 13.0, 28.0, 31.0, 14.0, 30.0, 29.0]

# plt.plot(mea1k7_post_removal, color='red', alpha=.5)
# plt.plot(mea1k7_with_els, color='red', alpha=.9)
# plt.plot(well_device)
# plt.plot(mea1k03_animal10)
# plt.show()