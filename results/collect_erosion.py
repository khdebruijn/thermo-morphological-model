# import packages
import os
from pathlib import Path
import sys

import numpy as np
import xarray as xr
import pandas as pd

proj_dir = os.getcwd()

sys.path.append(proj_dir)

from utils.bathymetry import calculate_bluff_edge_toe_position, calculate_shoreline_position

# read input
runid = sys.argv[1]

run_output_dir = Path(f"p:/11210070-usgscoop-202324-arcticxb/runs/{runid}/")

output_ids = np.int32(np.loadtxt(os.path.join(run_output_dir, "timestep_output_ids.txt")))

# reference variables
per1 = {
    "start_time_be": pd.to_datetime("2009-08-02"), # these dates are relevant for the bluff edge only
    "end_time_be": pd.to_datetime("2011-07-15"),
    # "start_time_sl": pd.to_datetime(),
    # "end_time_sl": pd.to_datetime(),
    "x_shore_line_start": -1,   # i.e., distance to baseline at start
    "x_bluff_edge_start": 162.01508156899843,   # i.e., distance to baseline at start
    "x_shore_line_end": -1,   # i.e., distance to baseline at end
    "x_bluff_edge_end": 162.4068294126812,   # i.e., distance to baseline at end
}

per2 = {
    "start_time_be": pd.to_datetime("2012-07-11"), # these dates are relevant for the bluff edge only
    "end_time_be": pd.to_datetime("2015-07-05"),
    # "start_time_sl": pd.to_datetime(),
    # "end_time_sl": pd.to_datetime(),
    "x_shore_line_start": -1,   # i.e., distance to baseline at start
    "x_bluff_edge_start": 164.52084174135774,   # i.e., distance to baseline at start
    "x_shore_line_end": -1,   # i.e., distance to baseline at end
    "x_bluff_edge_end": 179.48248124206415,   # i.e., distance to baseline at end
}

per3 = {
    "start_time_be": pd.to_datetime("2016-08-27"), # these dates are relevant for the bluff edge only
    "end_time_be": pd.to_datetime("2018-07-30"),
    # "start_time_sl": pd.to_datetime(),
    # "end_time_sl": pd.to_datetime(),
    "x_shore_line_start": -1,   # i.e., distance to baseline at start
    "x_bluff_edge_start": 187.25161246050337,   # i.e., distance to baseline at start
    "x_shore_line_end": -1,   # i.e., distance to baseline at end
    "x_bluff_edge_end": 195.1759190877606,   # i.e., distance to baseline at end
}

# get relevant period information
if 'per1' in runid:
    per = per1
elif 'per2' in runid:
    per = per2
elif 'per3' in runid:
    per = per3

# initialize output lists
t_list = []
x_shore_line_list = []
x_bluff_edge_list = []

# used for printing progress to screen
i = 0

# loop through output
for output_id in output_ids:
    
    fname = (10 - len(str(int(output_id)))) * '0' + str(int(output_id)) + ".nc"
            
    ds = xr.load_dataset(os.path.join(run_output_dir, fname))
    
    t = ds.timestamp.values
    
    xgr = ds.xgr.values
    zgr = ds.zgr.values

    ds.close()
    
    x_shore_line = calculate_shoreline_position(xgr, zgr)[0]
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

reference_id_be = df['time'] == per['start_time_be']
reference_offset_be = df[reference_id_be].x_bluff_edge.values

df['relative_erosion_bluff_edge'] = df['x_bluff_edge'] - reference_offset_be 
df['relative_x_bluff_edge'] = df['relative_erosion_bluff_edge'] + per['x_bluff_edge_start']

df['relative_erosion_shore_line'] = df['x_shore_line'] - reference_offset_be
df['relative_x_shore_line'] = df['relative_erosion_shore_line'] + per['x_bluff_edge_start']

# reference_id_sl = df['time'] == per['start_time_sl']
# reference_offset_sl = df[reference_id_sl].x_shore_line.values

# df['relative_erosion_shre_line'] = df['x_shore_line'] - reference_offset_sl
# df['relative_x_shore_line'] = df['relative_erosion_shre_line'] + per['x_bluff_edge_start']

save_path = f'./results/erosion_rates/{runid}.csv'

df.to_csv(save_path)