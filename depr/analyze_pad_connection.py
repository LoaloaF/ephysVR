import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt

import mea1k_ephys as mea1k
from mea1k_viz import plot_trace, correlate_traces, plot_implanted_connectivity_freq_bands, plot_connectivity_parallel_ext_signal
from mea1k_viz import plot_connectivity_neighbour_corr
from ephys_constants import SAMPLING_RATE, DEVICE_NAME, NAS_DIR, LOCAL_DATA_DIR

def _get_config_names(path):
    recording_fullfnames = sorted(glob(os.path.join(path, "*.h5")))
    if not recording_fullfnames:
        raise ValueError(f"No recordings found in {path}")
    first = os.path.basename(recording_fullfnames[0])
    last = os.path.basename(recording_fullfnames[-1])
    print(f"Found {len(recording_fullfnames)} recordings, {first} ... {last}")
    return recording_fullfnames

def process_sine(trace, lowcut, highcut, name, btype, debug=False, 
                 implant_mapping=None, max_ampl=.2):
    flt_trace = mea1k.filter_trace(trace, SAMPLING_RATE, lowcut=lowcut, 
                            highcut=highcut, btype=btype)
    avg_ampl, _ = mea1k.extract_average_amplitude(flt_trace)
    agg_power, pf, pps = mea1k.estimate_frequency_power(trace, min_band=lowcut, 
                                                        max_band=highcut)

    if debug:
        positive_freqs, positive_power_spectrum = pf, pps
        plot_trace(trace, positive_freqs, positive_power_spectrum, agg_power, 
                    flt_trace, avg_ampl, implant_mapping=implant_mapping,
                    label='Implanted - one electrode trace', 
                    filter_label=name, max_ampl=max_ampl)
    return agg_power, avg_ampl
        
def implanted_connectivity_freq_bands(path, implant_mapping, offset_samples=12000, 
                                      debug=False):
    rec_fullfnames = _get_config_names(path)
    print(implant_mapping)
    print("------------------------------------")

    all_powers = []
    all_traces = []
    for i, rec_fullfname in enumerate(rec_fullfnames):
        data = mea1k.read_raw_data(*os.path.split(rec_fullfname), convert2vol=True, to_df=True,
                                   subtract_dc_offset=True, 
                                   col_slice=slice(offset_samples,None,None))
        
        if debug:
            correlate_traces(data, implant_mapping)
        
        filter_params = {"300-3000Hz": (300, 3000, 'band'), "<300Hz": (300, None, 'low')}
        config_powers = []
        for el_i, trace in data.iterrows():
            for name, (lowcut, highcut, btype) in filter_params.items():
                agg_power, _ = process_sine(trace, lowcut, highcut, name, btype, 
                                            debug=debug, implant_mapping=implant_mapping)
                config_powers.append(pd.Series([agg_power], index=[name], name=el_i))

        config_powers = pd.concat(config_powers, axis=1).T
        config_powers_lp = config_powers.copy()['<300Hz'].dropna()
        config_powers_bp = config_powers.copy()['300-3000Hz'].dropna()
        config_powers = pd.concat([config_powers_lp, config_powers_bp], axis=1)
        config_powers['config_i'] = i
        config_powers['outside_brain_ampl'] = implant_mapping.reindex(data.index).ampl
        all_powers.append(config_powers)
        
        data['config_i'] = i
        all_traces.append(data)

    all_powers = pd.concat(all_powers)
    all_powers.to_csv(os.path.join(path, 'connectivity.csv'))
    all_traces = pd.concat(all_traces)
    all_traces.to_pickle(os.path.join(path, 'traces.pkl'))
    print(all_traces)

def connectivity_parallel_ext_signal(path, unit, unit_scale, input_freq=1000, 
                                     offset_samples=12000, debug=True):
    rec_fullfnames = _get_config_names(path)

    all_amplitudes = []
    for i, rec_fullfname in enumerate(rec_fullfnames):
        data = mea1k.read_raw_data(*os.path.split(rec_fullfname), convert2vol=True, 
                                   to_df=True, subtract_dc_offset=True, 
                                   col_slice=slice(offset_samples,None,None))
        
        for el_i, trace in data.iterrows():
            _, ampl = process_sine(trace, lowcut=input_freq-20, highcut=input_freq+20, 
                                   name=f"{input_freq}Hz", btype='band', debug=debug,
                                   max_ampl=4)
            all_amplitudes.append(pd.Series([ampl], index=['ampl'], name=el_i))
    all_amplitudes = pd.concat(all_amplitudes, axis=1).T
    all_amplitudes.to_csv(os.path.join(path, 'ext_signal_ampl.csv'))
    print(all_amplitudes)
    
def connectivity_neighbour_corr(path, offset_samples=12000, debug=True):
    rec_fullfnames = _get_config_names(path)
    
    all_corrs = []
    for i, rec_fullfname in enumerate(rec_fullfnames):
        data = mea1k.read_raw_data(*os.path.split(rec_fullfname), convert2vol=True, 
                                   to_df=True, subtract_dc_offset=True, 
                                   col_slice=slice(offset_samples,None,None))
        
        fullfname = os.path.join('./assets/mea1k_configs/9x3x16_meshgrid', 
                                 os.path.basename(rec_fullfname).replace('.raw.h5', '.csv'))
        mesh_config = pd.read_csv(fullfname, index_col=0)
        
        for tile_i in mesh_config.tile.unique():
            tile_d = data.reindex(mesh_config[mesh_config.tile == tile_i].el)
            
            tile_corr = np.corrcoef(tile_d.values)
            
            # flatten 
            tile_corr = np.triu(tile_corr, k=1)
            tile_corr[tile_corr==0] = np.nan
            tile_corr_el2el = pd.DataFrame(tile_corr, index=tile_d.index, 
                                           columns=tile_d.index).unstack()

            if debug:
                for el_i, trace in tile_d.iterrows():
                    plt.plot(trace)
                    plt.ylabel(f"e{el_i}")
                plt.figure()
                plt.imshow(tile_corr, cmap='viridis')
                plt.colorbar()
                plt.show()
            all_corrs.append(tile_corr_el2el)
    
    all_corrs = pd.concat(all_corrs)
    all_corrs.groupby(level=(0,1)).mean()
    all_corrs.to_csv(os.path.join(path, 'neighbour_el_corr.csv'))
    
def main():
    implant_mapping = mea1k.get_implant_mapping(NAS_DIR, DEVICE_NAME)
    
    # # first brain recording checking frquencies in the brain
    subdir = 'Simon/impedance/device_headmount_new3EpoxyWalls/connectivity_bonded4_D5_brain_1024_rec1'
    # implanted_connectivity_freq_bands(os.path.join(NAS_DIR, subdir), implant_mapping, 
    #                                   debug=False)
    plot_implanted_connectivity_freq_bands(os.path.join(NAS_DIR, subdir))
    
    
    # # connectivity before implantation ext voltage clamped signal
    # subdir = 'Simon/impedance/device_headmount_new3EpoxyWalls/impedance_bonded4_D0_1KHz_1024_rec1'
    # connectivity_parallel_ext_signal(os.path.join(NAS_DIR, subdir), unit='V', 
    #                                  unit_scale=1e-3, debug=False,
    #                                  offset_samples=12000)
    # plot_connectivity_parallel_ext_signal(os.path.join(NAS_DIR, subdir))
    
    # # connectivity between neighbours
    # subdir = 'implant_devices/241101_headstage09_50pad1shank/meshConfigs_passive_brain_rec3'
    # connectivity_neighbour_corr(os.path.join(NAS_DIR, subdir), offset_samples=12000, debug=False)
    # plot_connectivity_neighbour_corr(os.path.join(NAS_DIR, subdir))
    
    

if __name__ == '__main__':
    main()