import os

import numpy as np

from utils.load import read_config
from utils.model import generate_params, generate_grid, start_xbeach, write_xbeach_output

def main(config):
    
    # read temporal parameters
    t_start = config.model.time_start
    t_end = config.model.time_end
    dt = config.model.timestep
    
    # this variable will be used to keep track of time
    T = np.arange(t_start, t_end, dt)  # units are in hours
    
    # determine when storms occur (using raw_datasets/erikson/Hindcast_1981_2/BTI_WavesAndStormSurges_1981-2100.csv)
    st = np.zeros(T.shape)  # array of the same shape as t (0 when no storm, 1 when storm)
    
    # determine when xbeach is ran regardless of storms
    ct = np.zeros(T.shape)
    ct[0:-1:int(config.model.call_xbeach_inter)] = 1  # inter-storm interval can be controlled with the config.model.call_xbeach configuration
    
    # determine when to run xbeach (and write this to a file) --> during storms and every ~1 week (during ice-free season? or also during winter?)
    tt = (st + ct) > 0
    
    # generate initial grid files
    nx = config.model.nx
    ny = config.model.ny
    len_x = config.model.len_x
    len_y = config.model.len_y
    
    xgrid, ygrid = generate_grid(nx, ny, len_x, len_y)
    
    # save grid files
    np.savetxt(config.xbeach.xfile, xgrid)
    np.savetxt(config.xbeach.yfile, ygrid)
    
    # generate initial bathymetric files
        #  load initial bed
        #  LIMITED BATHYMETRIC DATA, DO I JUST ASSUME EQUILLIBRIUM PROFILE?
    bed_fp = config.xbeach.bedfile
    # generate_initial_morph(config)
    
    # generate thermal model 1D output model files

    # loop through timesteps
    for i in range(len(T)):
        

        # call thermal update routine
        
        # generate updated 'ne_layer' file

        # check if xbeach is enabled for current timestep
        if tt[i]:
            t_end   = config.xbeach.tstop
            
            
            
            # generate params.txt file (including: grid/bathymetry, waves input, flow, tide and surge input, water level, wind input, sediment input, 
            #                                      avalanching, vegetation input, drifters ipnut, output selection)

            # call xbeach

            # copy updated morphology to thermal module and to new output file
    
    
    return 


if __name__ == '__main__':

    # set the 'runid' to the model run that you would like to perform
    runid = '00'
    
    # set working directory
    os.chdir(os.path.join('runs/', runid, '/'))

    # read configuration file with parameters
    cfg = read_config('config.yaml')

    generate_params(cfg)
    
    main(cfg)