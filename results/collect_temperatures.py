import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

RUNID = "val_gt24"

result_folder = os.path.join(Path("p:/11210070-usgscoop-202324-arcticxb/runs/"), RUNID + "/")

fpaths = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if (os.path.isfile(os.path.join(result_folder, f)) and '.nc' in str(f))]

fpaths = np.array(fpaths)

sort_ids = np.argsort(np.int32(np.array([fpath[-13:-3] for fpath in fpaths])))

sorted_fpaths = fpaths[sort_ids]

all_results = np.zeros((len(sorted_fpaths), 150))

for i, fpath in enumerate(sorted_fpaths):
    
    print(i, fpath)
        
    ds = xr.load_dataset(fpath)
        
    T = ds['ground_temperature_distribution'].values[1]
        
    all_results[i, :] = T        
        
    ds.close()

    print(f"done with timestep {i} / {len(sorted_fpaths)}")
        
np.savetxt(Path(f"{RUNID}_temperature_AllLayers.txt"), all_results)
        