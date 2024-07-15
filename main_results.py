import os
from pathlib import Path
import sys

from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt

from utils.results import SimulationResults

def main(runid, args, attempt_counter_max=100):
    
    # initialize simulation
    print(f"initializing {runid}")
    results = SimulationResults(runid=runid)

    # compute and write bluff edge and shore line
    if "bluff_and_toe" in args:
        print('Computing bluff toe & shoreline positions')
        results.get_bluff_toes_and_shorelines()

    print(f"creating animations for {runid}")
    
    # determine number of frames
    frame_num = len(results.timestep_output_ids)
    
    # failsave (matplotlib sometimes crashes)
    attempt_counter = 0
    
    ##########################
    ##  define kwargs here  ##
    ##########################
    make_animation = False
    xmin, xmax = 1300, 1400
    fps = frame_num / 120  # for a 120 second animation
    

    while attempt_counter < attempt_counter_max:
        
        try:
        
            if "bed" in args:
                results.bed_level_animation(fps=fps, make_animation=make_animation, xmin=xmin, xmax=xmax)
                args.remove('bed')
                plt.close('all')
            
            if "heat" in args:
                results.heat_forcing_animation(fps=fps, make_animation=make_animation, xmin=xmin, xmax=xmax)
                args.remove('heat')
                plt.close('all')
            
            if "temp_heat" in args:
                results.temperature_heatforcing_animation(fps=fps, make_animation=make_animation, xmin=xmin, xmax=xmax)
                args.remove('temp_heat')
                plt.close('all')
            
            if "temp" in args:
                results.temperature_animation(fps=fps, make_animation=make_animation, xmin=xmin, xmax=xmax)
                args.remove('temp')
                plt.close('all')
                
            if "all" in args:
                results.bed_temperature_thawdepth_heatforcing_animation(fps=fps, make_animation=make_animation, xmin=xmin, xmax=xmax)
                args.remove('all')
                plt.close('all')
            
            break
    
        except ValueError:
            
            attempt_counter += 1
        
    print(f"completed {runid}")
    
    return None
    
def set_mpl_defaults(
    SMALL_SIZE = 16,
    MEDIUM_SIZE = 20,
    BIGGER_SIZE = 25,
):
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    return None


if __name__=="__main__":
    
    # reduce ipython cache size to free up memory
    ipython = get_ipython()
    if ipython:
        ipython.Completer.cache_size = 5
        
    # set defaults
    set_mpl_defaults()
    
    # set the 'runid' to the model run that you would like to perform
    runid = np.array(sys.argv)[1]
    
    # possible options for args: bluff_and_toe bed heat temp_heat temp
    args = list(np.array(sys.argv)[2:])
    
    main(runid, args)
    