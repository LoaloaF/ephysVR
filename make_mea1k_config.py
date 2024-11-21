import os
from glob import glob
import time
import pandas as pd
import maxlab

import ephys_constants as C
from mea1k_ephys import get_implant_mapping
from mea1k_utils import start_saving, stop_saving

from mea1k_utils import setup_array

def try_routing(els):
    print(f"\n\nMaking config for mapping {len(els)} shank-connected-pads to"
            f" electrodes on MEA1K  chip...")
    
    array = setup_array(els)
    array.download()
    succ_routed = [m.electrode for m in array.get_config().mappings]
    failed_routing = [el for el in els if el not in succ_routed]
    print(f"Failed routing {len(failed_routing)}: {failed_routing}")
    array.close()
    return succ_routed, failed_routing
    

def main():
    implant_mapping = get_implant_mapping(C.NAS_DIR, C.DEVICE_NAME)
    # slice to electrodes under pads that are routed to a shank PI electrode
    implant_mapping = implant_mapping[implant_mapping.shank_id.notna()]

    sel_which_rank = 1
    els = implant_mapping[(implant_mapping.connectivity_order==sel_which_rank) & (implant_mapping.mea1k_connectivity>20)].mea1k_el
    while True:
        succ_routed, failed_routing = try_routing(els)

        if len(failed_routing) == 0:
            print("Done.")
            break
        sel_which_rank += 1
        print(f"Trying alternative electrodes with connectivity rank {sel_which_rank}")
        
        # check which pad the failed electrodes are associated with
        missing_pads = implant_mapping[implant_mapping.mea1k_el.isin(failed_routing)].pad_id
        missing_pads = implant_mapping[implant_mapping.pad_id.isin(missing_pads)]
        
        alt_els = missing_pads[missing_pads.connectivity_order==sel_which_rank].mea1k_el
        good_enough_connec_mask = missing_pads[missing_pads.connectivity_order==sel_which_rank].mea1k_connectivity > 20
        print(f"{sum(good_enough_connec_mask)} / {len(good_enough_connec_mask)} alternative electrodes have good enough connectivity")
        alt_els = alt_els[good_enough_connec_mask]
        els = succ_routed + alt_els.tolist()

    array = setup_array(els)
    array.download()
    config_fullfname = os.path.join(C.NAS_DIR, "implant_devices", C.DEVICE_NAME, f"bonded_{C.DEVICE_NAME}_241120_{len(els)}chnls.cfg")
    
    #csv of config
    rows = [(m.channel, m.electrode, m.x, m.y) for m in array.get_config().mappings]
    config_df = pd.DataFrame(rows, columns=["channel", "electrode", "x", "y"])
    config_df.to_csv(config_fullfname.replace(".cfg", ".csv"), index=False)
    print(config_df)
    
    # save config in mea1k specific format
    array.save_config(config_fullfname)
    array.close()

if __name__ == "__main__":
    main()