import os
from pathlib import Path
import sys

from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt

from utils.results import SimulationResults

def main(runid):
    
    print(f"initializing {runid}")
    results = SimulationResults(runids=[runid])
    
    print(f"creating animations for {runid}")
    frame_num = len(results.timestep_output_ids)
    fps = frame_num / 120  # for a 120 second animation

    # results.bed_level_animation(runid, fps=fps)
    results.heat_forcing_animation(runid, fps=fps)
    results.temperature_animation(runid, fps=fps)
    
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
    runids = np.array(sys.argv)[1:]
    
    for runid in runids:
        main(runid)