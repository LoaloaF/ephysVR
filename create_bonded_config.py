import os
import maxlab
import pandas as pd

import ephys_constants as C

def main():
    ELECTRODE_DEVICE_NAME = 'H1278pad4shank'
    HEADSTAGE_DEVICE_NAME = 'MEA1K06'
    date = '241211'
    batch = 5
    IMPLANT_DEVICE_NAME = f"{date}_{HEADSTAGE_DEVICE_NAME}_{ELECTRODE_DEVICE_NAME}B{batch}"
    
    fullfname = os.path.join(C.NAS_DIR, "implant_devices", IMPLANT_DEVICE_NAME, "bonding",
                             f'bonding_mapping_{IMPLANT_DEVICE_NAME}.csv')
    
    bonding_mapping = pd.read_csv(fullfname, index_col=False)
    print(bonding_mapping)
    mea1k_els = bonding_mapping[bonding_mapping.routed==True].mea1k_el.values
    print(mea1k_els)
    print(len(mea1k_els))
    
    array = maxlab.chip.Array()
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(mea1k_els)
    print(f"Routing {len(mea1k_els)} electrodes...")   
    array.route()
    array.download()
    
    config_fname = fullfname.replace(".csv", ".cfg")
    array.save_config(config_fname)
    print(f"Saved config to {config_fname}")
    
if __name__ == "__main__":
    main()