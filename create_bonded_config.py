import os
import maxlab
import pandas as pd

import ephys_constants as C

def main():
    implant_device_sub_dir = "Simon/impedance/device_headmount_new3EpoxyWalls/impedance_bonded4_D0_1KHz_1024_rec1"
    # implant_device_sub_dir = "Simon/impedance/device_headmount_new3EpoxyWalls/impedance_bonded4_D0_1KHz_1024_rec1"
    path = os.path.join(C.NAS_DIR, implant_device_sub_dir)

    pads = pd.read_pickle(f"{path}/final_pads.pkl")
    electrodes = pads.index.values
    
    array = maxlab.chip.Array()
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(electrodes)
    print(f"Routing {len(electrodes)} electrodes...")   
    array.route()
    array.download()
    
    array.save_config(f"{path}/bonded_new.cfg")
    print(f"Saved config to {path}/bonded_new.cfg")
    
if __name__ == "__main__":
    main()