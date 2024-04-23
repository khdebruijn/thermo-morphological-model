import os
import yaml

from datetime import datetime

import subprocess
import numpy as np

class Simulation():
    def __init__(self, runid):
        self.runid = runid
        self._set_directory()
    
    def __repr__(self) -> str:
        return str(self.directory)
    
    def _set_directory(self):
        os.chdir(os.path.join('runs/', self.runid, '/'))
        self.directory = os.path.join('runs/', self.runid, '/')
        
        return None
    
    def read_config(self, config_file):
        '''
        Creates configuration variables from file
        ------
        config_file: .yaml file (string path)
            file containing dictionary with dataset creation information
        ''' 
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        
        return None
        
    def set_temporal_params(self, t_start, t_end, dt):
        # this variable will be used to keep track of time
        self.T = np.arange(t_start, t_end, dt)  # units are in hours
        
        # determine when storms occur (using raw_datasets/erikson/Hindcast_1981_2/BTI_WavesAndStormSurges_1981-2100.csv)
        st = np.zeros(self.T.shape)  # array of the same shape as t (0 when no storm, 1 when storm)
        
        # determine when xbeach is ran regardless of storms
        ct = np.zeros(self.T.shape)
        ct[0:-1:int(self.config.model.call_xbeach_inter)] = 1  # inter-storm interval can be controlled with the config.model.call_xbeach configuration
        
        # determine when to run xbeach (and write this to a file) --> during storms and every ~1 week (during ice-free season? or also during winter?)
        tt = (st + ct) > 0

    def generate_grid(self, nx, ny, len_x, len_y):
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
            
        self.nx = nx
        self.ny = ny
        self.len_x = len_x
        self.len_y = len_y
        self.xgrid = xgrid
        self.ygrid = ygrid
        
        return xgrid, ygrid
    
    def export_grid(self):
        np.savetxt(self.config.xbeach.xfile, self.xgrid)
        np.savetxt(self.config.xbeach.yfile, self.ygrid)

    
    def generate_bed(self, fp_bed):
        
        self.bed = ...
        self.bed_timeseries = ...
        pass
    
    def update_bed_sedero(self, fp_xbeach_output):
        # Read output file
        cum_sedero = np.loadtxt(os.path.join(fp_xbeach_output, "..."))
        
        # update bed level
        self.bed += cum_sedero
    
    def export_bed(self):
        self.bed.to_csv("...")
    
    def initialize_erodible_layer(self, fp_active_layer):
        pass
    
    def load_tide_conditions(self, fp_wave):
        pass
    
    
    def load_wave_conditions(self, fp_wave):
        
        with open(fp_wave) as f:
            pass
        
    def load_wind_conditions(self, fp_wind):
        pass
        
    def run_simulation(self):
        pass
    
    
    
    @classmethod
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


    # def write_xbeach_output(self, output_path, save_path):
    #     """
    #     Running this function writes the xbeach output file (i.e., morphological update) to the wrapper.
    #     --------------------------
    #     output_path: str
    #         string containing the file path to the xbeach output from the project directory
    #     save_path: str
    #         string containing the save path for the morphological update
    #     --------------------------
    #     """
        
        

    #     # Save output file
    #     np.savetxt(os.path.join(save_path, "morph.txt"))
        
    #     return 




# def generate_params(config):
#     """
#     This function takes the config variable and generates a params.txt file
#     """
    
#     with open('params.txt', 'w') as f:
#         f.write("---------- \n")
#         f.write("\n")
#         f.write("XBEACH")
#         f.write(f"date [YYYY-MM-DD HH:MM:SS.XXXXXX]: {datetime.now()} \n")
#         f.write("function: generate_params() \n")
#         f.write("\n")
#         f.write("---------- \n")
#         f.write("\n")
        
#         f.write("---------- \n")
#         f.write("-GRID INPUT- \n")
#         f.write("\n")
#         f.write(f"nx = {config.model.nx} \n")
#         f.write(f"ny = {config.model.ny} \n")

#         f.write("---------- \n")

#         f.write("-NUMERICS INPUT- \n")

        
#     f = open(' params.txt', 'r')
    
#     return f