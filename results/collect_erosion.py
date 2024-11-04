import os
from pathlib import Path
import sys

import numpy as np
import xarray as xr
import pandas as pd

proj_dir = os.getcwd()

sys.path.append(proj_dir)

from utils.bathymetry import calculate_bluff_edge_toe_position, calculate_shoreline_position


runid = sys.argv[1]

run_output_dir = Path(f"p:/11210070-usgscoop-202324-arcticxb/runs/{runid}/")

output_ids = np.int32(np.loadtxt(os.path.join(run_output_dir, "timetsep_output_ids.txt")))

t_list = []
x_shore_line_list = []
x_bluff_edge_list = []

i = 0

for output_id in output_ids:
    
    fname = (10 - len(str(int(output_id)))) * '0' + str(int(output_id)) + ".nc"
            
    ds = xr.load_dataset(os.path.join(run_output_dir, fname))
    
    t = ds.timestamp
    
    xgr = ds.xgr.values
    zgr = ds.zgr.values

    ds.close()
    
    x_shore_line = calculate_shoreline_position(xgr, zgr)
    x_bluff_edge, __, __ = calculate_bluff_edge_toe_position(xgr, zgr)
    
    t_list.append(t)
    x_shore_line_list.append(x_shore_line)
    x_bluff_edge_list.append(x_bluff_edge)

    i += 1
    
    print(f'{i}/{len(output_ids)}')

df = pd.DataFrame(data={
    'time':t_list, 
    'x_shore_line':x_shore_line_list,
    'x_bluff_edge':x_bluff_edge_list,
    })

save_path = f'./results/erosion_rates/{runid}.csv'

df.to_csv(save_path)