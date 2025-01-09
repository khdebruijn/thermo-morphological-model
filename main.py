import os
from pathlib import Path
import sys
import time

from IPython import get_ipython
import numpy as np
import pandas as pd

from utils.model import Simulation
from utils.bathymetry import generate_schematized_bathymetry
from utils.miscellaneous import textbox, datetime_from_timestamp


def main(sim):
    """run this function to perform a simulation

    Args:
        sim (Simulation): instance of the Simulation class
    """
    t_start = time.time()
    
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
    sim.load_forcing(
        os.path.join(sim.proj_dir, sim.config.data.forcing_data_path)
    )
    print("succesfully loaded forcing")
    
    # load hydrodynamic forcing
    sim.initialize_hydro_forcing(
        os.path.join(sim.proj_dir, sim.config.data.storm_data_path),
        )
    print("succesfully loaded hydrodynamic forcing")
    
    # this variable is used to determine if xbeach should be ran for each timestep (not looking at 2% runup yet)
    xb_times = sim.timesteps_with_xbeach_active()
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
            
            N=sim.config.bathymetry.N,
            artificial_flat=sim.config.bathymetry.artificial_flat
        )
        
        np.savetxt("x.grd", xgr)
        np.savetxt("bed.dep", zgr)
        
        print("succesfully generated schematized bathymetry")
    
    
    # generate initial grid files and save them
    xgr, zgr, ne_layer = sim.generate_initial_grid(
        nx=sim.config.bathymetry.nx if 'nx' in sim.config.bathymetry.keys() else None,
        bathy_path=sim.config.bathymetry.depfile,
        bathy_grid_path=sim.config.bathymetry.xfile
        )
    print("succesfully generated grid")
    
    # initialize xbeach module
    sim.initialize_xbeach_module()
    print("succesfully initialized xbeach module")
    
    # initialize first xbeach timestep
    if sim.config.xbeach.with_xbeach:
        sim.xbeach_times[0] = sim.check_xbeach(0)
    else:
        sim.xbeach_times[0] = 0
    
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
    
    ################################################
    ##                                            ##
    ##            # MAIN LOOP                     ##
    ##                                            ##
    ################################################
    
    for timestep_id in np.arange(len(sim.T)):
        
        print(f"timestep {timestep_id+1}/{len(sim.T)}")
        
        # write output variables to output file every output interval
        if timestep_id in sim.temp_output_ids:
            
            sim.write_output(timestep_id, t_start)
            
            print("sucessfully generated output")
            
        # used for validation of the temperature model
        if 'save_ground_temp_layers' in sim.config.output.keys():
            
            sim.save_ground_temp_layers_in_memory(
                timestep_id, 
                layers=sim.config.output.save_ground_temp_layers,
                heat_fluxes=sim.config.output.heat_fluxes,
                write=(timestep_id == np.arange(len(sim.T))[-1]),
                )
        
        # check whether to run XBeach or not for this timestep
        if sim.config.xbeach.with_xbeach and not all(np.abs(sim.thaw_depth) < 0.001):
            sim.xbeach_times[timestep_id] = sim.check_xbeach(timestep_id)
        else:
            sim.xbeach_times[timestep_id] = 0
            
        # check if xbeach is enabled for current timestep
        if sim.xbeach_times[timestep_id] and sim.config.xbeach.with_xbeach:
            
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
                    print(f"succesfully ran xbeach for timestep {sim.timestamps[timestep_id]} to {sim.timestamps[timestep_id+1]}")
                else:
                    print(f"failed to run xbeach for timestep {sim.timestamps[timestep_id]} to {sim.timestamps[timestep_id+1]}")
            except IndexError:
                # index error occurs when xbeach is called during the final time step, this catches it
                print(f"xbeach ran succesfully for final timestep timestep ({sim.timestamps[timestep_id]})")
                        
            # if this was one of the first storms, the output is of higher temporal resolution and it is saved in the results folder
            if sim.copy_this_xb_output:
                sim.copy_xb_output_to_result_dir(fp_xbeach_output="xboutput.nc")
                print("succesfully generated high resolution storm output")
                
            # check if xbeach should be ran for the next timestep (if so, the x-grid doesn't update since the same grid is necessary for the hotstart feature)
            if sim.config.xbeach.with_xbeach and timestep_id + 1 < len(sim.T):
                sim.xbeach_times[timestep_id + 1] = sim.check_xbeach(timestep_id + 1)
            else:
                sim.xbeach_times[timestep_id + 1] = 0
                        
            # copy updated morphology to thermal module, and update the thermal grid with the new morphology
            sim.update_grid(timestep_id, fp_xbeach_output="xboutput.nc")  # this thing right here is pretty slow (TO BE CHANGED)
            
            # copy all xb output to the results folder 
            # # (this is necessary as the xb output in the run folder will be overwritten after the next xb iteration)
            sim.dump_xb_output(timestep_id)
        
        # loop through thermal subgrid timestep
        for subgrid_timestep_id in np.arange(0, config.model.timestep * 3600, config.thermal.dt):
            
            sim.thermal_update(timestep_id, subgrid_timestep_id)
            
        # calculate the current thaw depth
        sim.find_thaw_depth()
            
        print()
    
    # write xbeach timesteps
    sim.write_xb_timesteps()

    print(textbox("SIMULATION FINISHED"))
    print(f"{repr(sim)}")
    print(f"Simulation started at: {datetime_from_timestamp(t_start)}")
    print(f"Simulation finished at: {datetime_from_timestamp(time.time())}")
    print(f"Total simulation time: {(time.time() - t_start) / 3600:.1f} hours")
    
    return sim.xgr, sim.zgr

if __name__ == '__main__':
    
    ##| To run script from Terminal:
    ##| cd C:\Users\bruij_kn\OneDrive - Stichting Deltares\Documents\GitHub\thermo-morphological-model
    ##| python main.py run_id
    
    # reduce ipython cache size to free up memory
    ipython = get_ipython()
    if ipython:
        ipython.Completer.cache_size = 5

    # set the 'runid' to the model run that you would like to perform
    runid = sys.argv[1]
    
    # initialize simulation
    sim = Simulation(runid)

    main(sim)
    