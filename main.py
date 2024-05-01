import os
import sys

from IPython import get_ipython
import numpy as np
import pandas as pd

from utils.model import Simulation

from utils.load import read_config
# from utils.model import generate_params, generate_grid, start_xbeach, write_xbeach_output

def main(sim):
    """run this function to perform a simulation

    Args:
        sim (Simulation): instance of the Simulation class
    """
    
    config = sim.config
    
    # read temporal parameters
    sim.set_temporal_params(
        config.model.time_start,
        config.model.time_end,
        config.model.timestep
        )
    
    # this variable is used to determine if xbeach should be ran for each timestep
    xb_times = sim.timesteps_with_xbeach_active(
        os.join(sim.proj_dir, "database/raw_datasets/erikson/Hindcast_1981_2/BTI_WavesAndStormSurges_1981-2100.csv")
        )
    
    # generate initial grid files and save them
    xgr, zgr, ne_layer = sim.generate_initial_grid(
        config.model.nx, 
        config.model.ny, 
        config.model.len_x, 
        config.model.len_y,
        config.model.bathy_path,
        config.model.bathy_grid_path)
    
    
    # generate file with forcing for thermal module
    sim.generate_thermal_forcing_timeseries(
        config.model.time_start,
        config.model.time_end,
        config.model.timestep,
        os.path.join(sim.proj_dir, "datasets/ts_datasets/thermal_forcing.csv")
        )
    
    # generate thermal model 1D output model files
    

    # loop through timesteps
    for i in range(len(sim.T)):
        

        # call thermal update routine
        sea_ice = False # placeholder
        if not(sea_ice and not config.thermal.with_sea_ice):
            pass
        
        # generate updated 'ne_layer' file
        current_bath = sim.update_bed_sedero("sedero.txt")

        # write thaw depth to output file every output interval
        if i in sim.thermal_output_ids:
            sim.write_thermal_output()
            
        # check if xbeach is enabled for current timestep
        if xb_times[i]:
            print(f"starting xbeach for timestep {sim.timestamps[i]}")
            t_end   = config.xbeach.tstop
            
             # generate params.txt file 
             # (including: grid/bathymetry, waves input, flow, tide and surge,
             # water level, wind input, sediment input, avalanching, vegetation, 
             # drifters ipnut, output selection)
            sim.xbeach_setup(i)
                       
            # call xbeach (could include batch file?)
            run_succesful = sim.start_xbeach(
                os.path.join(sim.proj_dir, "xbeach/XBeach_1.24.6057_Halloween_win64_netcdf/xbeach.exe"),
                sim.cwd
            )
            
            if run_succesful:  
                print(f"xbeach ran succesfully for timestep {sim.timestamps[i]} to {sim.timestamp[i+1]}")
            else:
                print(f"xbeach failed to run for timestep {sim.timestamps[i]} to {sim.timestamp[i+1]}")

            # copy updated morphology to thermal module and to new output file
            updated_bed = sim.update_bed_sedero("sedero.txt")
            
            # update grid
            pass
    
    return 


if __name__ == '__main__':
    
    # reduce ipython cache size to free up memory
    ipython = get_ipython()
    ipython.Completer.cache_size = 5

    # set the 'runid' to the model run that you would like to perform
    runid = sys.argv[1]
    
    # initialize simulation
    sim = Simulation(runid)
        
    # read configuration file with parameters
    sim.read_config("config.yaml")

    # generate_params(cfg)    
    main(sim)