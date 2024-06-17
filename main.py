import os
from pathlib import Path
import sys
import time

from IPython import get_ipython
import numpy as np
import pandas as pd

from utils.model import Simulation
from utils.bathymetry import generate_schematized_bathymetry
from utils.miscellaneous import textbox


def main(sim, print_report=False):
    """run this function to perform a simulation

    Args:
        sim (Simulation): instance of the Simulation class
    """
    print(textbox("INITIALIZING SIMULATION"))
    print(f"{repr(sim)}")
    
    config = sim.config
    print("succesfully read configuration")

    # read temporal parameters
    sim.set_temporal_params(
        config.model.time_start,
        config.model.time_end,
        config.model.timestep
        )
    print("succesfully set temporal parameters")
        
    # load in forcing data
    sim.load_forcing(fname_in_ts_datasets="era5.csv")
    print("succesfully loaded forcing")
    
    # this variable is used to determine if xbeach should be ran for each timestep
    xb_times = sim.timesteps_with_xbeach_active(
        os.path.join(sim.proj_dir, "database/ts_datasets/storms_erikson.csv"),
        )
    print("succesfully generated xbeach times")
    
    # generate schematized bathymetry
    if sim.config.bathymetry.with_schematized_bathymetry:
        xgr, zgr = generate_schematized_bathymetry(
            bluff_flat_length=sim.config.bathymetry.bluff_flat_length,
        
            bluff_height=sim.config.bathymetry.bluff_height, 
            bluff_slope=sim.config.bathymetry.bluff_slope,
            
            beach_width=sim.config.bathymetry.beach_width, 
            beach_slope=sim.config.bathymetry.beach_slope,
            
            nearshore_max_depth=sim.config.bathymetry.nearshore_max_depth, 
            nearshore_slope=sim.config.bathymetry.nearshore_slope,
            
            offshore_max_depth=sim.config.bathymetry.offshore_max_depth, 
            offshore_slope=sim.config.bathymetry.offshore_slope,
            
            contintental_flat_width=sim.config.bathymetry.continental_flat_width,
            
            with_artificial=sim.config.bathymetry.with_artificial,
            artificial_max_depth=sim.config.bathymetry.artificial_max_depth,
            artificial_slope=sim.config.bathymetry.artificial_slope,
            
            N=sim.config.bathymetry.N
        )
        
        np.savetxt("x.grd", xgr)
        np.savetxt("bed.dep", zgr)
        
        print("succesfully generated schematized bathymetry")
    
    
    # generate initial grid files and save them
    xgr, zgr, ne_layer = sim.generate_initial_grid(
        bathy_path=sim.config.bathymetry.depfile,
        bathy_grid_path=sim.config.bathymetry.xfile
        )
    print("succesfully generated grid")
    
    # initialize xbeach module
    sim.initialize_xbeach_module()
    print("succesfully initialized xbeach module")
    
    # initialize thermal model
    sim.initialize_thermal_module()
    print("succesfully initialized thermal module")
    
    # initialize solar flux calculator
    if sim.config.thermal.with_solar_flux_calculator:
        sim.initialize_solar_flux_calculator(
            sim.config.model.time_zone_diff,
            angle_min=sim.config.thermal.angle_min,
            angle_max=sim.config.thermal.angle_max,
            delta_angle=sim.config.thermal.delta_angle,
            t_start=sim.config.thermal.t_start,
            t_end=sim.config.thermal.t_end,
            )
    print("succesfully initialized solar flux calculator\n")
    
    # show CFL values (they have already been checked to be below 0.5)
    print(textbox("CFL VALUES (for 1D thermal models)"))
    print(f"current maximum CFL: {np.max(sim.cfl_matrix):.4f}\n")

    # loop through (xbeach) timesteps
    print(textbox("STARTING SIMULATION"))
    
    for timestep_id in range(len(sim.T)):
        
        print(f"timestep {timestep_id+1}/{len(sim.T)}")
        
        # write output variables to output file every output interval
        if timestep_id in sim.temp_output_ids:
            sim.write_output(timestep_id)
            
        # check if xbeach is enabled for current timestep
        if xb_times[timestep_id] and sim.config.xbeach.with_xbeach:
            
            # calculate the current thaw depth
            sim.find_thaw_depth()
            
            # export current thaw depth to a file
            sim.write_ne_layer()
                        
             # generate params.txt file 
            sim.xbeach_setup(timestep_id)
            
            print(f"starting xbeach for timestep {sim.timestamps[timestep_id]}")
            
            # call xbeach (could include batch file?)
            run_succesful = sim.start_xbeach(
                os.path.join(sim.proj_dir, Path("xbeach/XBeach_1.24.6057_Halloween_win64_netcdf/xbeach.exe")),
                sim.cwd
            )
            try:
                if run_succesful:  
                    print(f"xbeach ran succesfully for timestep {sim.timestamps[timestep_id]} to {sim.timestamps[timestep_id+1]}")
                else:
                    print(f"xbeach failed to run for timestep {sim.timestamps[timestep_id]} to {sim.timestamps[timestep_id+1]}")
            except IndexError:
                # index error occurs when xbeach is called during the final time step, this catches it
                print(f"xbeach ran succesfully for final timestep timestep ({sim.timestamps[timestep_id]})")
            
            print()
                        
            # copy updated morphology to thermal module, and update the thermal grid with the new morphology
            sim.update_grid("xboutput.nc")  # this thing right here is pretty slow (TO BE CHANGED)
                                    
        # loop through thermal subgrid timestep
        for subgrid_timestep_id in np.arange(0, config.model.timestep * 3600, config.thermal.dt):
            
            sim.thermal_update(timestep_id, subgrid_timestep_id)
            

    print(textbox("SIMULATION FINISHED"))
    
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

    main(sim)
    