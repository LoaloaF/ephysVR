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

def extract_impedance(subdir, implant_name, current_ampl_nA, debug=False):
    L = Logger()
    fnames, ids = get_hdf5_fnames_from_dir(subdir)
    aggr_imp_data = []
    for fname, i in zip(fnames, ids):
        L.logger.info(f"Config {i},{fname} of {len(fnames)}")
        
        # if int(i[-4:]) != 1200:
        #     continue
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
                                             freq=1000, debug=debug)
    
        mean_ampl = []
        for j,row in enumerate(data):
            _, m_ampl = estimate_frequency_power(row[stim_sample_ids[0]:stim_sample_ids[1]].astype(float), 
                                                 sampling_rate=C.SAMPLING_RATE, 
                                                 debug=debug, 
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
            data = data[mapping.shank_id == 2]
            mapping = mapping[mapping.shank_id == 2]
            # viz_mea1k_config(mapping, stim_mea1k_el=stimulated.electrode[stimulated.stim].item(),
            #                  col_scaler=.6)
            vis_shank_traces(data-data[:,0:1], mapping, stim_mea1k_el=stimulated.electrode[stimulated.stim].item(), 
                             scaler=1/3_000, uVrange=470_000, stimulated=stimulated)
            
            plt.subplot(2, 1, 1)
            plt.plot(data.T/1000)
            plt.ylabel("Voltage [mV]")
            
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
                    plt.text(np.where(stim_el_imp_data.stim_unit == stim_unit)[0][0], 1e4, 
                            f"Stim Unit: {int(stim_unit):2d}", color=cmap(norm(shuffle_remapping[stim_unit])),
                            rotation=90, fontsize=11, ha='left', va='top')
                    
                    
                    
    all_els_imp_data = pd.read_csv(os.path.join(subdir, "processed", "extracted_imp_voltages.csv"))
    all_els_imp_data = all_els_imp_data.drop(columns=['Unnamed: 0']).set_index('electrode', drop=True)
    implant_mapping = get_raw_implant_mapping(implant_name=implant_name).set_index("mea1k_el", drop=True)
    
    # merge metadata with impedance data
    all_els_imp_data = pd.merge(all_els_imp_data.drop(columns=['depth', 'pad_id']), 
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
         'shank': 'first',                 # First shank
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
                               draw_on_ax=ax, pad_alpha=.7)
        draw_mea1K_colorbar(cmap_plasma, norm_imp_col, 'Impedance [kOhm]', orientation='vertical')
        plt.savefig(os.path.join(subdir, "processed", "imp_interconnect.png"))
        
    if draw_interconnect_stim_unit:
        (fig, ax), el_rects = draw_mea1k(mapping=implant_mapping)
        stim_unit_color_dict = {r.pad_id: r.stimunit_color for _, r in stim_el_imp_data.iterrows()}
        draw_interconnect_pads(implant_mapping, edgecolor=stim_unit_color_dict, 
                               draw_on_ax=ax, pad_alpha=.7)
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
    print(stim_el_imp_data)
    

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
            # slice to the config where the el was stimulued, get other pad data
            all_els_dat = all_els_imp_data[all_els_imp_data.config == row.config]
            ticks.append(row.pad_id)
            
            # calculate the mean and std of the impedance for all pads in the config, across mea1k els
            config_pad_imp_std = all_els_dat.groupby('pad_id').agg({'imp_kOhm': 'std'}).fillna(0)
            config_pad_imp_mean = all_els_dat.groupby('pad_id').agg({'imp_kOhm': 'mean', 'el_pair': 'first'})
            # seperate out the stimulated pad
            config_stim_pad_std = config_pad_imp_std.loc[row.pad_id].imp_kOhm
            config_stim_pad_mean = config_pad_imp_mean.loc[row.pad_id].imp_kOhm
            config_pad_imp_std.drop(row.pad_id, inplace=True)
            config_pad_imp_mean.drop(row.pad_id, inplace=True)
            
            # draw stimulated pad average over mea1k els
            plt.scatter([i], config_stim_pad_mean, color=row.stimunit_color,
                        marker='o', s=25, alpha=.7)
            plt.errorbar([i], config_stim_pad_mean, yerr=config_stim_pad_std,
                         color=row.stimunit_color, alpha=.7)
            
            # draw other pads, only if they have > 20% of the stim pad impednace
            low_imp_mask = config_pad_imp_mean.imp_kOhm.values/config_stim_pad_mean > 0.2
            config_pad_imp_mean = config_pad_imp_mean[low_imp_mask]
            config_pad_imp_std = config_pad_imp_std[low_imp_mask]
            el_pair_color = ['green' if row.el_pair == el_pair else 'k' for el_pair in config_pad_imp_mean.el_pair]
            el_pair_marker = ['*' if row.el_pair == el_pair else 'o' for el_pair in config_pad_imp_mean.el_pair]

            # Loop through each pad and plot individually
            for j, (mean, color, marker) in enumerate(zip(config_pad_imp_mean.imp_kOhm, el_pair_color, el_pair_marker)):
                plt.scatter([i], [mean], color=color, alpha=0.7, s=25, marker=marker)
            plt.errorbar([i]*len(config_pad_imp_mean), config_pad_imp_mean.imp_kOhm.values.flatten(), yerr=config_pad_imp_std.values.flatten(), color='gray', alpha=.3)
            
            # onlly stimulated mea1k el
            plt.scatter([i], [row.imp_kOhm],
                        color='k', marker='o', s=6, alpha=.6)
            
        plt.xticks(ticks, stim_el_imp_data.pad_id.astype(int).values, rotation=90, fontsize=7)
        
    plt.tight_layout()
    plt.show()
    exit()
    
    
    
    
    # # scatter plot
    # data = data.reset_index(drop=True)
    
    # # reorder = data.sort_values(by=['shank', 'depth']).index
    # reorder = data.sort_values(by=['stim_unit', 'imp_kOhm']).index
    # # rank what stim_unit has highest impedance
    # stim_order = data.groupby('stim_unit').agg({'imp_kOhm': 'mean'}).sort_values(by='imp_kOhm', ascending=False)
    # print(stim_order)
    # print(data)
    
    
    # # Map stim_unit to its rank based on stim_order
    # stim_unit_rank = stim_order.reset_index().reset_index().set_index('stim_unit')['index']
    # data['stim_unit_rank'] = data['stim_unit'].map(stim_unit_rank)

    # # Sort data by stim_unit_rank and imp_kOhm
    # reorder = data.sort_values(by=['stim_unit_rank', 'imp_kOhm']).index
    # # print(data)
    # # exit()
    
    # # draw_interconnect_pads(mapping, draw_on_ax=ax, pad_alpha=.6, edgecolor=stimunit_color)
    

    # plt.figure()
    # metal_col = ['green' if m==1 else 'purple' for m in data.iloc[reorder].metal.values]
    # plt.scatter(np.arange(len(imp)), imp[reorder], 
    #             # color=metal_col, marker='o', s=100, alpha=.3)
    #             color='k', marker='_', s=100, alpha=.8)
    #             # color=np.array(list(stimunit_color.values()))[reorder])
    # print(data.stim_unit[reorder])
    
    print(imp[reorder])
    j = 0
    for _, dat in data.iloc[reorder].iterrows():
        # print(dat)
        stim_on_fiber = dat.el_pair
    
        other_pads = all_data[(all_data.config == dat.config)]
        # print(other_pads.iloc[:3].T)
        
        pad_averages = other_pads.loc[:,['shank', 'depth', "imp_stim_ratio", 'pad_id']].groupby('pad_id').mean()
        pad_averages = pad_averages[pad_averages.imp_stim_ratio>.1]
        
        stim_pad_id = other_pads.pad_id[other_pads.stim].item()
        
        # pop out the stim pad
        stim_pad_data = pad_averages.loc[stim_pad_id]
        plt.scatter(j, stim_pad_data.imp_stim_ratio*imp[reorder][j],
                    color='k', marker='o', s=20, alpha=.3)
        
        print()
        pad_averages.drop(stim_pad_id, inplace=True)
        
        other_pads = pad_averages.index
        # print(stim_on_fiber)
        other_pads_elpair = mapping[mapping.pad_id.isin(other_pads)].loc[:, ['pad_id', 'el_pair']].set_index('pad_id', drop=True).drop_duplicates()
        print(other_pads_elpair, stim_on_fiber)
        
        el_pair_color = ['green' if stim_on_fiber == el_pair else 'k' for el_pair in other_pads_elpair.el_pair]
        
        
        
        
        # print(pad_averages[pad_averages.el_pair == stim_pad_data.el_pair])
        # print(pad_averages)
        
        plt.scatter([j]*len(pad_averages), pad_averages.imp_stim_ratio.values*imp[reorder][j],
                    color=el_pair_color, marker='o', s=100, alpha=.4)
        j += 1
        # print(pad_averages)
        # print(pad_averages.iloc[:3].T)
        
        # exit()
    
    # slice to always the first unqiue stim unit
    unique_stim_units = data.stim_unit.unique()
    for i, stim_unit in enumerate(unique_stim_units):
        plt.axvline(np.where(data.stim_unit[reorder] == stim_unit)[0][0], color='k', 
                    linestyle='dashed', alpha=.4)
        # label
        plt.text(np.where(data.stim_unit[reorder] == stim_unit)[0][0], 1e4, 
                 f"Stim Unit: {int(stim_unit):2d}", color=cmap(norm(shuffle_remapping[stim_unit])),
                 rotation=90, fontsize=11, ha='left', va='top')
        
    plt.yscale('log')
    plt.ylim(10, 1e4)
    # log scale y
    plt.ylabel("Impedance [kOhm]")
    plt.xlabel("Pad ID on MEA1K  |  Polyimide el. ID in brain")
    plt.xticks(np.arange(len(imp))[::5], (data.pad_id[reorder][::5]).astype(int), rotation=90, fontsize=5)
    plt.title("Order Impedance by Stim Unit")
    
    # plt.ylim(0, 2500)
    # vertical line at 0, 125, 1628/2+125, 1628/2+125+125
    # plt.axvline(0, color='k', linestyle='dashed')
    # plt.axvline(106, color='k', linestyle='dashed')
    # plt.axvline(814, color='k', linestyle='dashed')
    # plt.axvline(920, color='k', linestyle='dashed')
    #save
    
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
    np.random.seed(0)
    
    # implant_name = "241016_MEA1K03_H1278pad4shankB5"
    implant_name = "4983"
    # implant_name = "250308_MEA1K07_H1628pad1shankB6"
    # implant_name = "241211_MEA1K06_H1278pad4shankB5"
    # implant_name = "250205_MEA1K03_H1278pad4shankB5"
    current_ampl_nA = 20 
    
    subdirs = [
        # f"devices/well_devices/{implant_name}/recordings/18.27.08_invivo_localstim_mode='small_current'_stimpulse='sine'2_amplitude=20",
        # f"devices/implant_devices/{implant_name}/recordings/17.07.54_GND_REF_cable_imp5_localstim_mode='small_current'_stimpulse='sine'2_amplitude=10",
        # f"devices/implant_devices/{implant_name}/recordings/00_17.07.54_GND_REF_cable_imp5_localstim_mode='small_current'_stimpulse='sine'2_amplitude=10",
        # f"devices/implant_devices/{implant_name}/recordings/2025-03-25_15.32.21_invivo_imp_mode='small_current'_stimpulse='sine'_amplitude=10",
        # f"devices/implant_devices/{implant_name}/recordings/02_14.06.56_noSilk_GND_REF_cable_imp7_localstim_mode='small_current'_stimpulse='sine'2_amplitude=10",
        # f"devices/implant_devices/{implant_name}/recordings/03_16.01.21_noSilk_tapwater_GND_REF_cable_imp7_localstim_mode='small_current'_stimpulse='sine'2_amplitude=10",
        f"devices/well_devices/{implant_name}/recordings/11.17.35_singleshankConfigs_test_mode='small_current'_stimpulse='sine'2_amplitude=10",
    ]
    # el_config_S1D1650.raw.h5
    extract_impedance(os.path.join(nas_dir, subdirs[0]), implant_name=implant_name, 
                      current_ampl_nA=current_ampl_nA, debug=False)
    # print(os.path.join(nas_dir, subdirs[0]))
    # vis_impedance(os.path.join(nas_dir, subdirs[0]), implant_name=implant_name)
    # compare_impedance([os.path.join(nas_dir, subdir) for subdir in subdirs], implant_name=implant_name)
    plt.show()
    
    
    # subdir = "/Volumes/large/BMI/VirtualReality/SpatialSequenceLearning/RUN_rYL006/rYL006_P1100/2025-01-27_13-39_rYL006_P1100_LinearTrackStop_73min"
    # data = read_raw_data(subdir, "ephys_output.raw.h5", convert2mV_float16=True, col_slice=slice(20_000, 30_000), to_df=True, subtract_dc_offset=True)
    # mapping = get_recording_implant_mapping(subdir, "ephys_output.raw.h5", animal_name="rYL006", exclude_shanks=[2,3,4])
    # mapping = mapping[mapping.depth < 2000]
    # mapping = mapping[mapping.shank_side == 'left']
    # data = data.loc[mapping.mea1k_el]
    # vis_shank_traces(data.values, mapping, scaler=180)
    
if __name__ == "__main__":
    main()