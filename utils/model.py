import os
import yaml

from datetime import datetime

import subprocess
import numpy as np
import pandas as pd

import xbTools
from xbTools.grid.creation import xgrid, ygrid
from xbTools.xbeachtools import XBeachModelSetup

class Simulation():
    def __init__(self, runid):
        self.runid = runid
        self._set_directory()
    
    def __repr__(self) -> str:
        return str(self.directory)
    
    def _set_directory(self):
        # set working directory
        self.proj_dir = os.getcwd()
        self.cwd = os.path.join(os.getcwd(), 'runs', self.runid)
        
        os.chdir(self.cwd)    
        
        return None
    
    def read_config(config_file):
        '''
        Creates configuration variables from file
        ------
        config_file: .yaml file
            file containing dictionary with dataset creation information
        ''' 
    
        class AttrDict(dict):
            """
            This class is used to make it easier to work with dictionaries and allows 
            values to be called similar to attributes
            """
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self
                    
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
            
        config = AttrDict(cfg)
                
        for key in cfg:
            config[key] = AttrDict(cfg[key])
                
        return config
        
    def set_temporal_params(self, t_start, t_end, dt):
        # this variable will be used to keep track of time
        self.timestamps = pd.date_range(start=t_start, end=t_end, freq=f'{dt}h')
        
        # time indexing is easier for numerical models    
        self.T = np.arange(0, len(self.timestamps), 1) 
        

    def generate_initial_grid(self, nx, ny, len_x, len_y, 
                              bathy_path=None,
                              bathy_grid_path=None):
        """Generates grid files (x.grd & y.grd) from given nx and ny.

        Args:
            nx (int): number of gridpoints in x-direction
            ny (int): number of gridpoints in y-direction
            len_x (float): total length of the grid in x-direction
            len_y (float): total length of the grid in y-direction
            bathy_path: path to initial bathymetry (if None, uses bed.dep in current working directory)
            bathy_grid_path: path to initial bathymetry grid (if None, uses x.grd in current working directory)
            
        note that the current version is 1D and does not implement a grid in y-direction
        """
        
        if not bathy_path:
            self._load_bathy(
                os.path.join(self.proj_dir, bathy_path)
            )
        else:
            self._load_bathy(os.path.join(self.cwd, "bed.dep"))
            
        if not bathy_grid_path:
            self._load_grid_bathy(
                os.path.join(self.proj_dir, bathy_grid_path)
            )
        else:
            self._load_grid_bathy(os.path.join(self.cwd, "x.grd"))
                
        self.xgr, self.zgr = xgrid(self.bathy_grid, self.bathy_initial, dxmin=2)
        self.zgr = np.interp(self.xgr, self.bath_grid, self.bathy_initial)
        
        
        self.bathy_current = np.copy(self.zgr)
        self.bathy_timeseries = [self.zgr]
        
        return self.xgr, self.zgr
    
    # def export_grid(self):
    #     np.savetxt(self.config.xbeach.xfile, self.xgr)
    #     np.savetxt(self.config.xbeach.yfile, self.zgr)

    
    def _load_bathy(self, fp_initial_bathy):
        with open(fp_initial_bathy) as f:
            self.bathy_initial = np.loadtxt(f)
            
    def _load_grid_bathy(self, fp_bathy_grid):            
        with open(fp_bathy_grid) as f:
            self.bathy_grid = np.loadtxt(f)   
                
    def update_bed_sedero(self, fp_xbeach_output):
        # Read output file
        cum_sedero = np.loadtxt(os.path.join(fp_xbeach_output, "..."))
        
        # update bed level
        self.bed_current += cum_sedero
        
        # save current bed
        self.bed_timeseries.append(self.bed_current)
    
    def export_bed(self):
        self.bed.to_csv("...")
    
    def initialize_erodible_layer(self, fp_active_layer):
        pass
    
    def load_tide_conditions(self, fp_wave):
        pass
    
    
    # def load_wave_conditions(self, fp_wave):
        
    #     with open(fp_wave) as f:
    #         pass
        
    def load_wind_conditions(self, fp_wind):
        pass
    
    def timesteps_with_xbeach_active(self, fp_storm):
        
        self.storm_timing = self._when_storms(fp_storm)
        self.xbeach_no_storms = self._when_xbeach_inter(self.config.model.call_xbeach_inter)
        self.xbeach_times = self.storm_timing + self.xbeach_no_storms

        return self.xbeach_times
        
    def _when_storms(self, fp_storm):
        
        # determine when storms occur (using raw_datasets/erikson/Hindcast_1981_2/BTI_WavesAndStormSurges_1981-2100.csv)
        st = np.zeros(self.T.shape)  # array of the same shape as t (0 when no storm, 1 when storm)
        
        self.conditions = np.zeros(self.T.shape)  # also directly read wave conditions here
        
        with open(fp_storm) as f:
            
            df = pd.read_csv(f)
            
            for index, data in df.iterrows():
                
                duration = int(data["Storm_duration(days)"] * 24)  # in hours
                
                day = 0
                
                for hour in range(duration+1):
                    
                    
                    
                    if hour >= 24:
                        
                        day += 1
                        hour -= 24
                    
                    timestamp = datetime(
                        data.start_date_of_storm_year, 
                        data.start_date_of_storm_month, 
                        data.start_date_of_storm_day + day, 
                        12 + hour,  # assume storms always start at 12:00:00 during the day
                        0, 
                        0
                        )

                    t = np.argmax((timestamp == self.timestamps))  # we get the current timestep here
                    
                    st[t] += 1  #  make sure xbeach will be active for this timestep
                    
                    self.conditions[t] = {
                       "Hso(m)": data[" Hso(m)"],
                       "Hs(m)": data["Hs(m)"],
                       "Dp(deg)": data["Dp(deg)"],
                       "Tp(s)": data["Tp(s)"],
                       "SS(m)": data["SS(m)"],
                       "Hindcast_or_projection": data["Hindcast_or_projection"]
                        }  # safe storm conditions for this timestep as well
        return st
                
    def _when_xbeach_inter(self, call_xbeach_inter):
        
        ct = np.zeros(self.T.shape)
        
        ct[:,call_xbeach_inter] = 1
        
        return ct
        
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