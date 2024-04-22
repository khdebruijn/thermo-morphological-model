import os

from datetime import datetime

import subprocess
import numpy as np

def generate_grid(nx, ny, len_x, len_y):
    """Generates grid files (x.grd & y.grd) from given nx and ny.

    Args:
        nx (int): number of gridpoints in x-direction
        ny (int): number of gridpoints in y-direction
        len_x (float): total length of the grid in x-direction
        len_y (float): total length of the grid in y-direction
    """
    if not nx==0 and not ny==0:
        x = np.linspace(0, len_x, nx)
        y = np.linspace(0, len_y, ny)
        
        xgrid, ygrid = np.meshgrid(x, y)
        
    elif not nx==0:
        xgrid = np.linspace(0, len_x, nx)
        ygrid = np.array([])
        
    else:
        xgrid = np.array([])
        ygrid = np.linspace(0, len_y, ny)
    
    return xgrid, ygrid

def generate_params(config):
    """
    This function takes the config variable and generates a params.txt file
    """
    
    with open('params.txt', 'w') as f:
        f.write("---------- \n")
        f.write("\n")
        f.write("XBEACH")
        f.write(f"date [YYYY-MM-DD HH:MM:SS.XXXXXX]: {datetime.now()} \n")
        f.write("function: generate_params() \n")
        f.write("\n")
        f.write("---------- \n")
        f.write("\n")
        
        f.write("---------- \n")
        f.write("-GRID INPUT- \n")
        f.write("\n")
        f.write(f"nx = {config.model.nx} \n")
        f.write(f"ny = {config.model.ny} \n")

        f.write("---------- \n")

        f.write("-NUMERICS INPUT- \n")

        
    f = open(' params.txt', 'r')
    
    return f

def start_xbeach(xbeach_path, params_path):
    """
    Running this function starts the XBeach module as a subprocess.
    --------------------------
    xbeach_path: str
        string containing the file path to the xbeach executible from the project directory
    params_path: str
        string containing the file path to the params.txt file from the project directory
    --------------------------

    returns boolean (True if process was a sucess, False if not)
    """

    # Command to run XBeach
    command = [xbeach_path, params_path]

    # Call XBeach using subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to finish and get the return code
    stdout, stderr = process.communicate()
    return_code = process.returncode

    return return_code == 0


def write_xbeach_output(output_path, save_path):
    """
    Running this function writes the xbeach output file (i.e., morphological update) to the wrapper.
    --------------------------
    output_path: str
        string containing the file path to the xbeach output from the project directory
    save_path: str
        string containing the save path for the morphological update
    --------------------------
    """
    
    # Read output file
    morph = np.loadtxt(os.path.join(output_path, "morph.txt"))
    
    # Convert to correct format
    pass

    # Save output file
    np.savetxt(os.path.join(save_path, "morph.txt"))
    
    return 