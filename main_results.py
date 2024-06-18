import os
from pathlib import Path
import sys
import time

from IPython import get_ipython
import numpy as np
import pandas as pd

from utils.model import Simulation
from utils.results import SimulationResults
from utils.bathymetry import generate_schematized_bathymetry
from utils.miscellaneous import textbox


def process_results(config_path):
    """Run this function to collect, process, and store results from all simulations.    
    
    Args:
            output_file_name (string): name of the output file to be generated
            result_dir (Path): path to folder containing results
            run_ids (List): list of runid's which should be processed
            
    """
    
    cwd = os.getcwd()
    
    
    simres = SimulationResults(config_path)
    print("Initialization succesful")
    
    simres.collect_results()
    print("Collecting results succesful")
    
    simres.create_ds()
    print("Dataset creation succesful")
    
    simres.write_netcdf()
    print("Writing of netcdf file succesful")
            
            

if __name__ == '__main__':
    
    ##| To run script from Terminal:
    ##| cd C:\Users\bruij_kn\OneDrive - Stichting Deltares\Documents\GitHub\thermo-morphological-model
    ##| python main_results.py results/result_config.yaml
    
    # reduce ipython cache size to free up memory
    ipython = get_ipython()
    if ipython:
        ipython.Completer.cache_size = 5
                
    # read name of config file
    config_path = Path(sys.argv[1])
        
    process_results(config_path)
    