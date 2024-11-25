# import packages
import os
from pathlib import Path
import sys

import numpy as np
import xarray as xr
import pandas as pd

from scipy.integrate import trapezoid

proj_dir = os.getcwd()

sys.path.append(proj_dir)

# read input
runids = ['sa_base', 'sa_base_alt'] + [f"sa_lvl1_{int(i)}" for i in np.arange(1, 68+1)] + [f"sa_lvl2_{int(i)}" for i in np.arange(1, 18+1)]  + [f"sa_lvl3_{int(i)}" for i in np.arange(1, 2+1)] 

V_start_list = []
V_end_list = []

i = 1

for runid in runids:
    try: 
        run_output_dir = Path(f"p:/11210070-usgscoop-202324-arcticxb/runs/{runid}/")

        output_ids = np.int32(np.loadtxt(os.path.join(run_output_dir, "timestep_output_ids.txt")))

        # reference variables
        max_distance_from_top = 5  # m

        # read data at start
        output_id_start = output_ids[0]
        fname_start = (10 - len(str(int(output_id_start)))) * '0' + str(int(output_id_start)) + ".nc"
        ds_start = xr.load_dataset(os.path.join(run_output_dir, fname_start))
        x_start = ds_start.xgr.values
        z_start = ds_start.zgr.values
        ds_start.close()
        
        output_id_end = output_ids[-1]
        fname_end = (10 - len(str(int(output_id_end)))) * '0' + str(int(output_id_end)) + ".nc"
        ds_end = xr.load_dataset(os.path.join(run_output_dir, fname_end))
        x_end = ds_end.xgr.values
        z_end = ds_end.zgr.values
        ds_end.close()
        
        z_mask_start = (z_start > np.max(z_start) - max_distance_from_top)
        z_mask_end = (z_end > np.max(z_end) - max_distance_from_top)
        
        V_start = trapezoid(z_mask_start * z_start, x_start)
        V_end = trapezoid(z_mask_end * z_end, x_end)
        
    except FileNotFoundError:
        
        V_start = 0
        V_end = 0
        
    V_start_list.append(V_start)
    V_end_list.append(V_end)

    print(f'Completed {i}/{len(runids)}')
    
    i += 1
    
data = {
    'runid': runids,
    'V_start[m2]': V_start_list,
    'V_end[m2]': V_end_list,
    'dV[m2]': np.array(V_end_list) - np.array(V_start_list),
}

df = pd.DataFrame(data)

save_path = f'./results/volume_changes.csv'

df.to_csv(save_path)