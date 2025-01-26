import os
# import maxlab
import pandas as pd

import ephys_constants as C
from mea1k_ephys import get_implant_mapping

def main():
    implant_name = C.DEVICE_NAME_RAT011
    
    print(implant_name)
    bonding_mapping = get_implant_mapping(C.NAS_DIR, implant_name)
    print(bonding_mapping)
    mea1k_els = bonding_mapping[(bonding_mapping.connectivity_order <= 2) & 
                                (bonding_mapping.shank_id <= 2) &
                                (bonding_mapping.mea1k_connectivity > 20)]
                                # (bonding_mapping.mea1k_connectivity >.8)]
    print(mea1k_els)
    print(len(mea1k_els))
    
    # array = maxlab.chip.Array()
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