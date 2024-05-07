import os
import sys

from IPython import get_ipython
import numpy as np
import pandas as pd

from utils.model import Simulation

def main(sim):
    """run this function to perform a simulation

    Args:
        sim (Simulation): instance of the Simulation class
    """
    config = sim.read_config(config_file="config.yaml")
    
    # read temporal parameters
    sim.set_temporal_params(
        config.model.time_start,
        config.model.time_end,
        config.model.timestep
        )
    
    # load in forcing data
    sim.load_forcing(fname_in_ts_datasets="era5.csv")
    
    # this variable is used to determine if xbeach should be ran for each timestep
    xb_times = sim.timesteps_with_xbeach_active(
        os.path.join(sim.proj_dir, "database/raw_datasets/erikson/Hindcast_1981_2/BTI_WavesAndStormSurges_1981-2100.csv"),
        from_projection=True
        )
    
    # generate initial grid files and save them
    xgr, zgr, ne_layer = sim.generate_initial_grid(
        config.model.nx, 
        config.model.ny, 
        config.model.len_x, 
        config.model.len_y,
        config.model.bathy_path,
        config.model.bathy_grid_path)
    
    # initialize thermal model
    sim.initialize_thermal_module()

    # loop through (xbeach) timesteps
    for timestep_id in range(len(sim.T)):
        
        # loop through thermal subgrid timestep
        for subgrid_timestep_id in np.arange(config.model.timestep * 3600 / config.thermal.dt):
            
            sim.thermal_update(timestep_id)
            
        # check if xbeach is enabled for current timestep
        if xb_times[timestep_id]:
            
            # calculate the current thaw depth
            sim.find_thaw_depth()
            
            # export current thaw depth to a file
            sim.write_ne_layer()
                        
             # generate params.txt file 
            sim.xbeach_setup(timestep_id)
            
            print(f"starting xbeach for timestep {sim.timestamps[timestep_id]}")
            
            # call xbeach (could include batch file?)
            run_succesful = sim.start_xbeach(
                os.path.join(sim.proj_dir, "xbeach/XBeach_1.24.6057_Halloween_win64_netcdf/xbeach.exe"),
                sim.cwd
            )
            
            if run_succesful:  
                print(f"xbeach ran succesfully for timestep {sim.timestamps[timestep_id]} to {sim.timestamp[timestep_id+1]}")
            else:
                print(f"xbeach failed to run for timestep {sim.timestamps[timestep_id]} to {sim.timestamp[timestep_id+1]}")
                
            # copy updated morphology to thermal module, and update the thermal grid with the new morphology
            sim.update_grid("sedero.txt")
            
        # write output variables to output file every output interval
        if timestep_id in sim.temp_output_ids:
            sim.write_output(timestep_id)
                
    return sim.xgr, sim.zgr


if __name__ == '__main__':
    
    # reduce ipython cache size to free up memory
    ipython = get_ipython()
    if ipython:
        ipython.Completer.cache_size = 5

    # set the 'runid' to the model run that you would like to perform
    runid = sys.argv[1]
    
    # initialize simulation
    sim = Simulation(runid)

    # generate_params(cfg)    
    main(sim)