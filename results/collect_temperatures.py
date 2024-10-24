import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

RUNID = "val_gt17"

result_folder = os.path.join(Path("p:/11210070-usgscoop-202324-arcticxb/runs/"), RUNID + "/")

fpaths = [os.path.join(result_folder, f) for f in os.listdir(result_folder) if (os.path.isfile(os.path.join(result_folder, f)) and '.nc' in str(f))]

all_results = np.zeros((len(fpaths), 150))

for i, fpath in enumerate(fpaths):
    
    print(i, fpath)
    
    with open(fpath) as f:
        
        ds = xr.load_dataset(f)
        
        T = ds['ground_temperature_distribution'].values[1]
        
        all_results[i, :] = T        
        
        ds.close()
        
np.savetxt(os.path.join(Path("./test/"), f"{RUNID}_temperature_AllLayers.txt"), all_results)
        