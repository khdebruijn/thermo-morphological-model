import os
from pathlib import Path
import shutil
import time
import yaml

from datetime import datetime
import math

from IPython.display import display
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from scipy.interpolate import interp1d
import xarray as xr

import xbTools
from xbTools.grid.creation import xgrid
from xbTools.xbeachtools import XBeachModelSetup
from xbTools.general.executing_runs import xb_run_script_win
from xbTools.general.wave_functions import dispersion

from utils.visualization import block_print, enable_print
import utils.miscellaneous as um

class Simulation():
    
    ################################################
    ##                                            ##
    ##            # GENERAL FUNCTIONS             ##
    ##                                            ##
    ################################################
    
    def __init__(self, runid, config_file="config.yaml"):
        """Initializer of the Simulation class. The runid is assigned, which is then used in the _set_directory method(). It also reads the configuration file.
        -----
        runid: string
            string that contains the name of the folder in the runs directory, which is used for reading inputs and writing outputs.
        config_file: string
            string that contains the name of the configuration file, default config.yaml, in the runid folder"""
        self.runid = runid
        self.read_config(config_file)
        self._set_directory()
        
    def __repr__(self) -> str:
        """Provides a string representation of the current simulation."""
        
        description = \
            f"RUNID: {self.runid}\n" + \
            f"PROJECT DIRECTORY: {self.proj_dir}\n" + \
            f"CURRENT WORKING DIRECTORY: {self.cwd}\n" + \
            f"TIMESERIES DIRECTORY: {self.ts_dir}\n" + \
            f"RESULTS DIRECTORY: {self.result_dir}\n"
            
        return description
    
    def _set_directory(self):
        """This method sets up the different directories used during the simulation, i.e. the project directory, the current working directory, the directory
        containing the timeseries related to the forcing, and it creates a result directory with associated files."""
        # set working directory
        self.proj_dir = os.getcwd()
        self.cwd = os.path.join(os.getcwd(), 'runs', self.runid)
        self.ts_dir = os.path.join(self.proj_dir, "database/ts_datasets/")
        
        # change working directory to folder containing the config.yaml file
        os.chdir(self.cwd)    

        # setup output location
        if self.config.output.use_default_output_path:  # check if results should be stored in the working directory
            self.result_dir = os.path.join(self.cwd, "results/")
        else:  # or somewhere else
            self.result_dir = os.path.join(Path(self.config.output.output_path), self.runid)
            
        # create output directory
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
    
    def read_config(self, config_file):
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
                    
        cwd = os.path.join(os.getcwd(), 'runs', self.runid)
                    
        with open(os.path.join(cwd, config_file)) as f:
            cfg = yaml.safe_load(f)
            
        self.config = AttrDict(cfg)
                
        for key in cfg:
            self.config[key] = AttrDict(cfg[key])
               
        return self.config
        
    def set_temporal_params(self, t_start, t_end, dt):
        """This method sets the temporal parameters used during the simulation."""
        
        # set start and end time, and time step
        self.dt = dt
        self.t_start = pd.to_datetime(t_start, dayfirst=True)
        self.t_end = pd.to_datetime(t_end, dayfirst=True)
        
        # check how many times this simulation should be repeated
        rpt = 1 if 'repeat_sim' not in self.config.model.keys() else self.config.model.repeat_sim

        # this variable will be used to keep track of time
        unrepeated_timestamps = pd.date_range(start=self.t_start, end=self.t_end, freq=f'{self.dt}h', inclusive='left')
        
        # repeat timestamps if necessary
        self.timestamps = unrepeated_timestamps
        for i in range(rpt - 1):
            self.timestamps = self.timestamps.append(unrepeated_timestamps)
        
        # time indexing is easier for numerical models    
        self.T = np.arange(0, len(self.timestamps), 1) 
        
        # this array defines when to generate output files
        self.temp_output_ids = np.arange(0, len(self.timestamps), self.config.output.output_res)
        
        # output timestep ids and timestamps to results directory
        self._check_and_write("timestamps", self.timestamps, self.result_dir)
        self._check_and_write("timestep_ids", self.T, self.result_dir)
        self._check_and_write("timestep_output_ids", self.temp_output_ids, self.result_dir)
        
    def generate_initial_grid(self, 
                              nx=None, 
                              len_x=None, 
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
        
        # Load initial bathymetry and the accompanying grid
        if bathy_path:
            self._load_bathy(
                os.path.join(self.cwd, bathy_path)
            )
        else:
            self._load_bathy(os.path.join(self.cwd, "bed.dep"))
            
        if bathy_grid_path:
            self._load_grid_bathy(
                os.path.join(self.cwd, bathy_grid_path)
            )
        else:
            self._load_grid_bathy(os.path.join(self.cwd, "x.grd"))
        
        # check whether to use an xbeach generated xgrid or to generate uniform grid using linspace
        if nx and self.config.bathymetry.with_nx:
            self.xgr = np.linspace(min(self.bathy_grid), max(self.bathy_grid), nx)
        else:
            # transform into a more suitable grid for xbeach
            self.xgr, self.zgr = xgrid(self.bathy_grid, self.bathy_initial, dxmin=2, ppwl=self.config.bathymetry.ppwl)
        
        # interpolate bathymetry to grid
        self.zgr = np.interp(self.xgr, self.bathy_grid, self.bathy_initial)
        
        # also initialize the current active layer depth here (denoted by "ne_layer", or non-erodible layer)
        self.thaw_depth = np.zeros(self.xgr.shape)
        
        # set origin of x-grid
        self.x_ori = np.min(self.xgr)
        
        return self.xgr, self.zgr, self.thaw_depth

    
    def _load_bathy(self, fp_initial_bathy):
        """Method to load the initial bathymetry.
        ------
        fp_initial_bathy: string
            filepath to the initial bathymetry (relative to the project directory)
        """
        with open(fp_initial_bathy) as f:
            self.bathy_initial = np.loadtxt(f)
            
    def _load_grid_bathy(self, fp_bathy_grid):
        """Method to load the initial bathymetry (x) grid.
        ------
        fp_initial_bathy: string
            filepath to the initial bathymetry x grid (relative to the project directory)
        """
        with open(fp_bathy_grid) as f:
            self.bathy_grid = np.loadtxt(f)
            
            
    def load_forcing(self, fpath):
        """This function loads in the forcing data and makes it an attribute of the simulation instance"""
        
        # check whether or not forcing conditions should be repeated
        rpt = 1 if 'repeat_sim' not in self.config.model.keys() else self.config.model.repeat_sim
        
        # read in forcing concditions
        self.forcing_data = self._get_timeseries(self.t_start, self.t_end, fpath, repeat=rpt)
        
        return None
    
    ################################################
    ##                                            ##
    ##            # XBEACH FUNCTIONS              ##
    ##                                            ##
    ################################################
    
    def initialize_xbeach_module(self):
        """This method initializes the xbeach module. Currently only used to set values for the first output at t=0.

        Returns:
            None
        """       
        # first get the correct forcing timestep
        row = self.forcing_data.iloc[0]
        
        # with associated forcing values
        self.current_air_temp = row["2m_temperature"]  # also used in output
        self.current_sea_temp = row['sea_surface_temperature']  # also used in output
        self.current_sea_ice = row["sea_ice_cover"]  # not used in this function, but loaded in preperation for output
        
        self.wind_direction, self.wind_velocity = self._get_wind_conditions(timestep_id=0)
        
        self.water_level = 0
        self.xb_check = 0
        self.R2 = 0
        
        self.storm_write_counter = 0
        
        return None
    
    def initialize_hydro_forcing(self, fp_storm):
        """This function is used to initialize hydrodynamic forcing conditions, and what the conditions are. It uses either 
        'database/ts_datasets/storms_erikson.csv' (for cmip hindcast) or 'database/ts_datasets/storms_engelstad.csv' 
        (for era5 hindcast).

        Args:
            fp_storm (Path): path to storm dataset

        Returns:
            array: array of length T that for each timestep contains hydrodynamic forcing
        """
        # Initialize conditions array
        self.conditions = np.zeros(self.T.shape, dtype=object)  # also directly read wave conditions here
        
        # Initialize zero conditions
        self.zero_conditions = {
                    # "Hso(m)": 0.001,
                    # "Hs(m)": 0.001,
                    "Hso(m)": 0.2,  # placeholder
                    "Hs(m)": 0.2,  # placeholder
                    # "Hso(m)": 0,
                    # "Hs(m)": 0,
                    "Dp(deg)": 270,                    
                    # "Dp(deg)": 0,
                    "Tp(s)": 10,
                    # "Tp(s)": 0,
                    # "WL(m)": 0,
                    "Hindcast_or_projection": 0,
                    }
        
        # read file and mask out correct timespan
        with open(fp_storm) as f:
            
            df = pd.read_csv(f, parse_dates=['time'])
                                    
            mask = (df['time'] >= self.t_start) * (df['time'] <= self.t_end)
            
            df = df[mask]

        # Loop through complete data to save conditions            
        # df_dropna = df.dropna(axis=0)
                        
        for i, row in df.iterrows():
            
            index = np.argwhere(self.timestamps==row.time)
                        
            if not row.isnull().values.any():
                            
                # safe storm conditions for this timestep as well            
                self.conditions[index] = {
                        "Hs(m)": row["Hs(m)"],
                        "Dp(deg)": row["Dp(deg)"],
                        "Tp(s)": row["Tp(s)"],
                        "WL(m)": row["WL(m)"],
                            }
                
            else:
                conds = self.zero_conditions
                conds['WL(m)'] = row['WL(m)']
                
                self.conditions[index] = conds
                
        self.water_levels = np.tile(df['WL(m)'].values, self.config.model.repeat_sim)
        
        return self.conditions
    
    def timesteps_with_xbeach_active(self):
        """This function gets the timestep ids for which xbeach should be active, without looking at 2% runup threshold yet.

        Returns:
            array: array of length T that for each timestep contains a 1 if xbeach should be ran and 0 if not.
        """
        
        # get inter-storm timestep ids
        self.xbeach_inter = self._when_xbeach_inter(self.config.model.call_xbeach_inter)
        
        # get sea-ice timestep ids
        self.xbeach_sea_ice = self._when_xbeach_no_sea_ice(self.config.wrapper.sea_ice_threshold)
        
        # initialize xbeach storms array
        self.xbeach_storms = np.zeros(self.xbeach_inter.shape)
        
        # initialize xbeach_times array
        self.xbeach_times = np.zeros(self.xbeach_inter.shape)

        return self.xbeach_times
    
    
    def _when_xbeach_inter(self, call_xbeach_inter):
        """This function determines the timestamps that xbeach is called regardless of sea-ice or storms.

        Args:
            call_xbeach_inter (int): Call xbeach every 'call_xbeach_inter' timestamps, regardsless of the presence of sea ice or a storm.

        Returns:
            array: array of length T that for each timestep contains a 1 if xbeach should be ran and 0 if not.
        """
        ct = np.zeros(self.T.shape)
        
        # set xbeach active at provided intervals
        ct[::call_xbeach_inter] = 1
        
        return ct
    
    def _when_xbeach_no_sea_ice(self, sea_ice_threshold):
        """This function determines when xbeach should not be ran due to sea ice, based on a threshold value)

        Args:
            sea_ice_threshold (float): _description_

        Returns:
            array: array of length T that for each timestep contains a 1 if xbeach can be ran and 0 if not (w.r.t. sea ice).
        """
        it =  (self.forcing_data.sea_ice_cover.values < sea_ice_threshold)
                
        return it
    
    def _when_xbeach_storms(self, timestep_id):
        """This function checks whether or not there is actually a storm during the upcoming each timestep, and is based on a 2% runup threshold

        Args:
            timestep_id (int): current timestep

        Returns:
            int: 1 for storm, 0 for no storm
        """
        # read hydrodynamic conditions for current timestep
        H = self.conditions[timestep_id]['Hs(m)']
        T = self.conditions[timestep_id]['Tp(s)']
        wl = self.conditions[timestep_id]['WL(m)']
        
        # assume these are the deep water conditions
        H0 = H
        L0 = 9.81 * T**2 / (2 * np.pi)
        
        # determine the +- 2*sigma envelope (for the stockdon, 2006 formulation)
        sigma = H / 4
        mask = np.nonzero((self.zgr > wl - 2*sigma) * (self.zgr < wl + 2*sigma))
        z_envelope = self.zgr[mask]
        x_envelope = self.zgr[mask]
            
        # if waves are too small, the envelope doesn't exist on the grid, so this method will fail
        try:
            # compute beta_f as the average slope in this envelope
            dz = z_envelope[np.argmax(x_envelope)] - z_envelope[np.argmin(x_envelope)]
            dx = x_envelope[np.argmax(x_envelope)] - x_envelope[np.argmin(x_envelope)]
        
        # and in that case, the local angle of the two grid points nearest to the water level is used
        except ValueError:
            
            dry_mask = self.zgr > wl
            dry_indices = np.nonzero(dry_mask)
            
            wet_mask = np.ones(dry_mask.shape) - dry_mask
            wet_indices = np.nonzero(wet_mask)
            
            first_dry_id = np.min(dry_indices)
            last_wet_id = np.max(wet_indices)
            
            x1, z1 = self.xgr[first_dry_id], self.zgr[first_dry_id]
            x2, z2 = self.xgr[last_wet_id], self.zgr[last_wet_id]
            
            # x1, x2 = self.xgr[wet_mask[-1]], self.xgr[dry_mask[0]]
            # z1, z2 = self.zgr[wet_mask[-1]], self.zgr[dry_mask[0]]
            
            dz = z2 - z1
            dx = x2 - x1
            
        # compute beta_f            
        beta_f = np.abs(dz / dx)

        # now the empirical formulation by Stockdon et al. (2006) can be used to determine R2%
        self.R2 = 1.1 * (0.35 * beta_f * (H0 * L0)**0.5 + (H0 * L0 * (0.563 * beta_f**2 + 0.004))**0.5 / 2)
        
        run_xb_storm = int(self.R2 + wl > self.config.wrapper.xb_threshold)
        
        return run_xb_storm
    
    def check_xbeach(self, timestep_id):
        """This function checks whether XBeach should be ran for the upcoming timestep.

        Args:
            timestep_id (int): id of the current timestep

        Returns:
            int: whether or not to run XBeach. 1 if yes, 0 if no.
        """
        
        self.xbeach_storms[timestep_id] = self._when_xbeach_storms(timestep_id)
        
        self.xbeach_times[timestep_id] = self.xbeach_inter[timestep_id] + self.xbeach_sea_ice[timestep_id] * self.xbeach_storms[timestep_id]
                
        return self.xbeach_times[timestep_id]
        
    def xbeach_setup(self, timestep_id):
        """This function initializes an xbeach run, i.e., it writes all inputs to files
        """
        # create instance of XBeachModelSetup (https://github.com/openearth/xbeach-toolbox/blob/main/xbTools/xbeachtools.py)
        self.xb_setup = XBeachModelSetup(f"Run {self.cwd}: timestep {timestep_id}")
        
        # set the grid
        self.xb_setup.set_grid(
            self.xgr, 
            None, 
            self.zgr, 
            posdwn=-1,
            xori=0,
            yori=0,
            # alfa=self.config.bathymetry.grid_orientation - 180,  # counter-clockwise from the east
            thetamin=self.config.xbeach.thetamin,
            thetamax=self.config.xbeach.thetamax,
            dtheta=self.config.xbeach.dtheta,
            thethanaut=self.config.xbeach.thetanaut,
            )
        
        # check zero conditions or normal conditions should be used
        if self.xbeach_inter[timestep_id] and not self.xbeach_storms[timestep_id] * self.xbeach_sea_ice[timestep_id]:
            conditions = self.zero_conditions
            conditions['WL(m)'] = self.water_levels[timestep_id]
        else:
            conditions = self.conditions[timestep_id]
        
        # set the waves
        self.xb_setup.set_waves('parametric', {
            # need to give each parameter as series (in this case, with length 1)
            "Hm0":conditions["Hs(m)"],  # file contains 'Hso(m)' (offshore wave height, in deep water) and 'Hs(m)' (nearhsore wave height, at 10m isobath)
            "Tp":conditions["Tp(s)"],
            # "mainang":conditions["Dp(deg)"],  # relative to true north
            'mainang': 270,  # default value for 1D XBeach
            "gammajsp": 3.3,  # value recommended by Kees
            "s": 1000,  # value recommended by Kees
            "duration": self.dt * 3600,
            "dtbc": 60, # placeholder
            "fnyq":1, # placeholder
        })
        
        # turn of wave model if there is no storm
        if not self.xbeach_storms[timestep_id]:
            self.xb_setup.wbctype = 'off'
        
        # load in wind and water level data
        wind_direction, wind_velocity = self._get_wind_conditions(timestep_id)
        wl = self.water_levels[timestep_id]  # used for output
        
        # check if this is a storm timestep that should be written in its entirety
        if self.config.xbeach.write_first_storms:
            tintg = self.config.xbeach.tintg_storms  # set the output interval
            self.config.xbeach.write_first_storms -= 1  # ensure 1 less storm is written
            self.copy_this_xb_output = True  # this variable is later checked to see if the xbeach output should be copied to the results folder
            self.storm_write_counter += 1  # variable used in the filename when storm is copied to results folder and renamed
        else:
            tintg = self.dt * 3600
            self.copy_this_xb_output = False
        
        # (including: grid/bathymetry, waves input, flow, tide and surge,
        # water level, wind input, sediment input, avalanching, vegetation, 
        # drifters ipnut, output selection)
        params = {
            # grid parameters
            # - already specified with xb_setup.set_grid(...)
            
            # sediment parameters
            "D50": self.config.xbeach.D50,
            "D90": self.config.xbeach.D50 * 1.5,  # placeholder
            "rhos": self.config.xbeach.rho_solid,
            # "reposeangle": self.config.xbeach.reposeangle,  # currently unused, since dryslp and wetslp are already defined
            "dryslp": self.config.xbeach.dryslp,
            "wetslp": self.config.xbeach.wetslp,
            
            # flow boundary condition parameters
            "front": "abs_1d",
            "back": "wall",
            "left": "neumann",
            "right": "neumann",
            
            # flow parameters
            "facSk": 0.15 if not "facSk" in self.config.xbeach.keys() else self.config.xbeach.facSk,
            "facAs": 0.20 if not "facAs" in self.config.xbeach.keys() else self.config.xbeach.facAs,
            "facua": 0.175 if not "facua" in self.config.xbeach.keys() else self.config.xbeach.facua,

            # general
            "befriccoef":self.config.xbeach.bedfriccoef,  # placeholder
            
            # model time
            "tstop":self.dt * 3600,  # convert from [h] to [s]
            "CFL": 0.95 if not "CFL_xbeach" in self.config.wrapper.keys() else self.config.wrapper.CFL_xbeach,
            
            # morphology parameters
            "morfac": 1,
            "morstart": 0,
            "ne_layer": "ne_layer.txt",
            "lsgrad": 0 if not "lsgrad" in self.config.xbeach.keys() else self.config.xbeach.lsgrad,
            
            # physical constant
            "rho": self.config.xbeach.rho_sea_water,
            
            # physical processes
            "avalanching": 1,  # Turn on avalanching
            "morphology": 1,  # Turn on morphology
            "sedtrans": 1,  # Turn on sediment transport
            "wind": 1 if self.config.xbeach.with_wind else 0,  # Include wind in flow solver
            "struct": 1 if self.config.xbeach.with_ne_layer else 0,  # required for working with ne_layer
            
            "flow": 1 if self.xbeach_storms[timestep_id] else 0,  # Turn on flow calculation (off if no storm)
            "lwave": 1 if self.xbeach_storms[timestep_id] else 0,  # Turn on short wave forcing on nlsw equations and boundary conditions (off if no storm)
            "swave": 1 if self.xbeach_storms[timestep_id] else 0,  # Turn on short waves (off if no storm)
            
            # tide boundary conditions
            "tideloc": 0,
            "zs0":wl,

            # wave boundary conditions
            "instat": self.config.xbeach.wbctype,
            "bcfile": self.config.xbeach.bcfile,
            "wavemodel":"surfbeat",
            
            # wind boundary condition
            "windth": wind_direction if self.config.xbeach.with_wind else 0,  # degrees clockwise from the north
            "windv": wind_velocity if self.config.xbeach.with_wind else 0,
            
            # hotstart (during a storm, use the previous xbeach timestep as hotstart for current timestep)
            # "writehotstart": 1 if self.xbeach_times[timestep_id + 1] else 0,  # Write hotstart during simulation
            # "hotstart": 1 if (self.xbeach_times[timestep_id - 1] and timestep_id != 0) else 0,  # Initialize simulation with hotstart
            # "hotstartfileno": 1 if (self.xbeach_times[timestep_id - 1] and timestep_id != 0) else 0,  # Initialize simulation with hotstart
            
            # output variables
            "outputformat":"netcdf",
            "tintg":tintg,
            "tstart":0,
            "nglobalvar":[
                "x",
                "y",
                "zb",  # bed level (1D array)
                "zs",  # water level (1D array)
                "H",  # wave height (1D array)
                "runup",  # run up (value)
                "sedero",  # cumulative erosion/sedimentation (1D array)
                "E",  # wave energy (1D array)
                "Sxx",  # radiation stress (1D array)
                "Sxy",  # radiation stress (1D array)
                "Syy",  # radiation stress (1D array)
                "thetamean",  # mean wave angle  (radians)
                "vmag",  # Velocity magnitude in cell centre (1D array)
                "urms",  # Orbital velocity (1D array)
                ]
        }
        
        self.xb_setup.set_params(params)
        
        # block printing while writing the output (the xbeach toolbox by default prints that it can't plot parametric conditions)
        block_print()
        
        # write model setup
        self.xb_setup.write_model(self.cwd, figure=False)
        
        # close figures generated during writing
        plt.close()
        
        # re-enable print
        enable_print()
        
        # the hotstart can not be added using the python toolbox, so it is manually added to the params.txt file here
        hotstart_in_toolbox = False
        if not hotstart_in_toolbox:
            with open('params.txt', 'r') as f:
                text = f.readlines()
            
            # add check too see if this is the last timestep
            writehotstart = 0
            if timestep_id + 1 < len(self.xbeach_times):
                writehotstart = 1
            
            hotstart_text = [
                "%% hotstart (during a storm, use the previous xbeach timestep as hotstart for current timestep)\n\n",
                f"writehotstart  = {writehotstart}\n",
                f"hotstart       = {1 if (self.xbeach_times[timestep_id - 1] and timestep_id != 0) else 0}\n",
                f"hotstartfileno = {1 if (self.xbeach_times[timestep_id - 1] and timestep_id != 0) else 0}\n",
                "\n"
                ]
                
            for i, line in enumerate(text):
                if r"%% Output variables" in line:
                    text = text[:i] + hotstart_text + text[i:]
                
            with open('params.txt', 'w') as f:
                f.writelines(text)
        
        return None
    
    def start_xbeach(self, xbeach_path, params_path, batch_fname="run.bat"):
        """
        Running this function starts the XBeach module as a subprocess.
        --------------------------
        xbeach_path: str
            string containing the file path to the xbeach executible from the project directory
        params_path: str
            string containing the file path to the params.txt file from the project directory
        batch_fname: str
            name used for the generated batch file
        --------------------------

        returns boolean (True if process was a sucess, False if not)
        """
        with open(batch_fname, "w") as f:
            f.write(f'cd "{self.cwd}"\n')
            f.write(f'call "{xbeach_path}"')
        
        # Command to run XBeach
        command = [str(os.path.join(self.cwd, batch_fname))]

        # # Call XBeach using subprocess
        return_code = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode

        return return_code == 0
    
    def copy_xb_output_to_result_dir(self, fp_xbeach_output="xboutput.nc"):
        
        destination_file = os.path.join(self.result_dir, f"storm{self.storm_write_counter}.nc")
        
        shutil.copy(fp_xbeach_output, destination_file)
        
        return None
        
    def _get_wind_conditions(self, timestep_id):
        """This function gets the wind conditions from the forcing dataset. Wind direction is defined in degrees
        clockwise from the north (i.e., east = 90 degrees)

        Args:
            timestep_id (int): timestep id for the current timestep for which wind dta is requested

        Returns:
            tuple: (wind_direction, wind_velocity)
        """
        row = self.forcing_data.iloc[timestep_id]
        
        u = row["10m_u_component_of_wind"]
        v = row["10m_v_component_of_wind"]
        
        direction = math.atan2(u, v) / (2*np.pi) * 360  # clockwise from the nord
        
        velocity = math.sqrt(u**2 + v**2)
        
        return direction, velocity
    
    ################################################
    ##                                            ##
    ##            # THERMAL FUNCTIONS             ##
    ##                                            ##
    ################################################
    def initialize_thermal_module(self):
        """This function initializes the thermal module of the model.

        Raises:
            ValueError: raised if CFL > config.wrapper.CFL_thermal
        """
        
        # read initial conditions
        ground_temp_distr_dry, ground_temp_distr_wet = self._generate_initial_ground_temperature_distribution(
            self.forcing_data, 
            self.t_start, 
            self.config.thermal.grid_resolution,
            self.config.thermal.max_depth
            )
        
        # save the grid resolution
        self.dz = self.config.thermal.max_depth / (self.config.thermal.grid_resolution - 1) 
               
        # save thermal grid distribution
        self.thermal_zgr = ground_temp_distr_dry[:,0]
                
        # initialize temperature matrix, which is used to keep track of temperatures through the grid
        self.temp_matrix = np.zeros((len(self.xgr), self.config.thermal.grid_resolution))
        
        # initialize the associated grid
        self.abs_xgr, self.abs_zgr = um.generate_perpendicular_grids(
            self.xgr, 
            self.zgr, 
            resolution=self.config.thermal.grid_resolution, 
            max_depth=self.config.thermal.max_depth
            )
        
        # set the above determined initial conditions for the xgr
        for i in range(len(self.temp_matrix)):
            if self.zgr[i] >= self.config.thermal.MSL:  # assume that the initial water level is at zero
                self.temp_matrix[i,:] = ground_temp_distr_dry[:,1]
            else:
                self.temp_matrix[i,:] = ground_temp_distr_wet[:,1]
        
        # set initial ghost node temperature as a copy of the surface node of the temperature matrix
        self.ghost_nodes_temperature = self.temp_matrix[:,0]
        
        # find and write the initial thaw depth
        self.find_thaw_depth()
        self.write_ne_layer()
        
        # with the temperature matrix, the initial state (frozen/unfrozen can be determined). This should be the only place where state is defined through temperature
        frozen_mask = (self.temp_matrix <= self.config.thermal.T_melt)
        unfrozen_mask = np.ones(frozen_mask.shape) - frozen_mask
        
        # define soil properties (k, density, nb, Cs & Cl)
        self.define_soil_property_matrices(len(self.xgr), define_nb=True)
            
        self.enthalpy_matrix = \
            frozen_mask * \
                self.Cs_matrix * self.temp_matrix + \
            unfrozen_mask * \
                (self.Cl_matrix * self.temp_matrix + \
                (self.Cs_matrix - self.Cl_matrix) * self.config.thermal.T_melt + \
                self.config.thermal.L_water_ice * self.nb_matrix)  # unit of L_water_ice is already corrected to use water density
        
        # calculate the courant-friedlichs-lewy number matrix
        self.k_matrix = frozen_mask * self.k_frozen_matrix + unfrozen_mask * self.k_unfrozen_matrix
        self.cfl_matrix = self.k_matrix / self.soil_density_matrix * self.config.thermal.dt / self.dz**2
        
        if np.max(self.cfl_matrix >= self.config.wrapper.CFL_thermal):
            # raise ValueError(f"CFL should be smaller than {self.config.wrapper.CFL_thermal}, currently {np.max(self.cfl_matrix):.4f}")
            # print(f"CFL should be smaller than 0.5, currently {np.max(self.cfl_matrix):.4f}")
            pass
        
        # get the 'A' matrix, which is used to make the numerical scheme faster. It is based on second order central differences for internal points
        # at the border points, the grid is extended with an identical point (i.e. mirrored), in order to calculate the second derivative
        self.A_matrix = um.get_A_matrix(self.config.thermal.grid_resolution)
        
        # initialize angles
        self._update_angles()
        
        # initialize output
        self.factors = np.zeros(self.xgr.shape)
        self.sw_flux = np.zeros(self.xgr.shape)
        
        self.latent_flux = np.zeros(self.xgr.shape)  # also used in output
        self.lw_flux = np.zeros(self.xgr.shape)  # also used in output  # also used in output
        
        self.convective_flux = np.zeros(self.xgr.shape)
        self.heat_flux = np.zeros(self.xgr.shape)

        return None
    
    def define_soil_property_matrices(self, N, define_nb=True):
        """This function is ran to easily define (and redefine) matrices with soil properties. 
        It is only a function of the x-grid, since the perpendicular z-grid is does not change in size.
        
        Args:
            N (int): number of surface grid points
            define_nb (bool): whether or not to (re)define nb. Defaults to True.
            """
        
        # initialize linear distribution of k, starting at min value and ending at max value (at a depth of 1m)
        id_kmax = np.argmin(np.abs(self.thermal_zgr - self.config.thermal.depth_constant_k))  # id of the grid point at which the maximum k should be reached
        self.k_unfrozen_distr = np.append(
            np.linspace(
                self.config.thermal.k_soil_unfrozen_min, 
                self.config.thermal.k_soil_unfrozen_max,
                len(self.thermal_zgr[:id_kmax])), 
            np.ones(len(self.thermal_zgr[id_kmax:])) * self.config.thermal.k_soil_unfrozen_max)
        self.k_frozen_distr = np.append(
            np.linspace(
                self.config.thermal.k_soil_frozen_min, 
                self.config.thermal.k_soil_frozen_max,
                len(self.thermal_zgr[:id_kmax])), 
            np.ones(len(self.thermal_zgr[id_kmax:])) * self.config.thermal.k_soil_frozen_max)
        
        # initialize k-matrix
        self.k_frozen_matrix =  np.tile(self.k_frozen_distr, (N, 1))
        self.k_unfrozen_matrix = np.tile(self.k_unfrozen_distr, (N, 1))
        
        # initialize distribution of ground ice content
        if define_nb:
            self.nb_distr = Simulation._compute_nb_distr(
                nb_max=self.config.thermal.nb_max,
                nb_min=self.config.thermal.nb_min,
                nb_max_depth=self.config.thermal.nb_max_depth,
                nb_min_depth=self.config.thermal.nb_min_depth,
                N=self.config.thermal.grid_resolution,
                max_depth=self.config.thermal.max_depth
            )            
            
            self.nb_matrix = np.tile(self.nb_distr, (N, 1))
            
        # calculate / read in density
        if self.config.thermal.rho_soil == "None":
            self.soil_density_matrix = self.nb_matrix * self.config.thermal.rho_water + (1 - self.nb_matrix) * self.config.thermal.rho_particle
        else:
            self.soil_density_matrix = np.ones(self.nb_matrix.shape) * self.config.thermal.rho_soil
        
        # using the states, the initial enthalpy can be determined. The enthalpy matrix is used as the 'preserved' quantity, and is used to numerically solve the
        # heat balance equation. Enthalpy formulation from Ravens et al. (2023).
        self.Cs_matrix = self.config.thermal.c_soil_frozen / self.soil_density_matrix
        self.Cl_matrix = self.config.thermal.c_soil_unfrozen / self.soil_density_matrix
        
        return None
    
    @classmethod
    def _compute_nb_distr(self, nb_max, nb_min, nb_max_depth, nb_min_depth, N, max_depth):
        """This function returns an nb distribution, with a sigmoid type curve connecting constant values above and below the min and max depth

        Args:
            nb_max (float): maximum value of nb (used close to surface),
            nb_min (float): minimum value of nb (used at depth),
            nb_max_depth (float): depth at which the nb value starts going down,
            nb_min_depth (float): depth at which minimum nb value is reached. Below this depth, nb is constant,
            N (int): number of (evenly spaced) grid points,
            max_depth (float): maixmum depth

        Returns:
            array: array containing nb values for the given grid.
        """
        nb = np.zeros(N)
        z = np.linspace(0, max_depth, N)

        mid = (nb_max_depth + nb_min_depth) / 2
            
        nb = (1 / (1 + np.exp(-(z - mid) * 10 / (nb_min_depth - nb_max_depth)))) * (nb_min - nb_max) + nb_max
        
        nb[np.argwhere(z<=nb_max_depth)] = nb_max
        nb[np.argwhere(z>=nb_min_depth)] = nb_min
            
        return nb
    
    def print_and_return_A_matrix(self):
        """This function prints and returns the A_matrix"""
        print(self.A_matrix)
        return self.A_matrix
    
    def _generate_initial_ground_temperature_distribution(self, df, t_start, n, max_depth):
        """This method generates an initial ground temperature distribution using soil temperature in different layers (read from 'df'),
        at the first time step 't_start'. The depth between 0 and 'max_depth' is divided in 'n' grid points.
        
        The ECMWF Integrated Forecasting System (IFS) has a four-layer representation of soil, where the surface is at 0cm: 
        Layer 1: 0 - 7cm, 
        Layer 2: 7 - 28cm, 
        Layer 3: 28 - 100cm, 
        Layer 4: 100 - 289cm. 
        Soil temperature is set at the middle of each layer, and heat transfer is calculated at the interfaces between them. 
        It is assumed that there is no heat transfer out of the bottom of the lowest layer. Soil temperature is defined over the whole globe, 
        even over ocean. Regions with a water surface can be masked out by only considering grid points where the land-sea mask has a value greater than 0.5. 
        This parameter has units of kelvin (K). Temperature measured in kelvin can be converted to degrees Celsius (°C) by subtracting 273.15.
        
        Temperature is linearly interpolated for the entire depth, and assumed constant below the center of Layer 4, as well as constant above the center 
        of layer 1. We differentiate between wet and dry initial conditions, assuming sea level at z=0. A maximum depth of 3m is assumed, with no heat 
        exchange from the lower layers.
        """                                    
        if "initial_ground_temp_path" not in self.config.data.keys(): 
            
            if "init_multi_linear_approx" in self.config.data.keys() and self.config.data.init_multi_linear_approx:
            
                # The temperature of the dry points can be reconstructed following the BLUE performed in database/erikson_ground_temp.ipynb
                def reconstruct_initial_conditions_era5(era5_points, X_hat_all, level):
                    """Takes ERA5 temperature data at the four defined levels and uses previously determined coefficients (from multi-linear 
                    regression, see notebook 'erikson_ground_tmep.ipynb') to compute better initial conditions from ERA5.

                    Args:
                        era5_points (array): array of length 4 with temperatures from ERA5 data (in Celcius!)
                        X_hat_all (array): array of length 5 with coefficients
                        level (int): current level to compute the reconstructed temperature for

                    Returns:
                        float: reconstructed temperature for the specified depth (in K)
                    """
                    
                    T1_era5, T2_era5, T3_era5, T4_era5 = era5_points
                    
                    level_to_index = {'50': 0, '100':1, '200': 2, '295': 3}
                    
                    i = level_to_index[str(int(level))]
                    
                    reconstructed_ic = \
                        X_hat_all[i][0] * T1_era5 + \
                        X_hat_all[i][1] * T2_era5 + \
                        X_hat_all[i][2] * T3_era5 + \
                        X_hat_all[i][3] * T4_era5 + \
                        X_hat_all[i][4] + \
                        273.15  # Celcius to Kelvin
                    
                    return reconstructed_ic
                
                # Start with loading in the X_hat
                X_hat_all = np.loadtxt(os.path.join(self.proj_dir, Path(r'database\ts_datasets\X_hat_groundtemp_reconstruct.txt')))
                
                era5_points = np.array([
                    df.soil_temperature_level_1.values[0] - 273.15,
                    df.soil_temperature_level_2.values[0] - 273.15,
                    df.soil_temperature_level_3.values[0] - 273.15,
                    df.soil_temperature_level_4.values[0] - 273.15
                ])
                
                dry_points = np.array([
                    [0.0, reconstruct_initial_conditions_era5(era5_points, X_hat_all, 50)],
                    [0.5, reconstruct_initial_conditions_era5(era5_points, X_hat_all, 50)],
                    [1.0, reconstruct_initial_conditions_era5(era5_points, X_hat_all, 100)],
                    [2.0, reconstruct_initial_conditions_era5(era5_points, X_hat_all, 200)],
                    [2.95, reconstruct_initial_conditions_era5(era5_points, X_hat_all, 295)],
                    [max_depth, reconstruct_initial_conditions_era5(era5_points, X_hat_all, 295)],
                ])
            
            else:
                
                dry_points = np.array([
                    [0, df.soil_temperature_level_1.values[0]],
                    [(0.07+0)/2, df.soil_temperature_level_1.values[0]],
                    [(0.28+0.07)/2, df.soil_temperature_level_2.values[0]],
                    [(1+0.28)/2, df.soil_temperature_level_3.values[0]],
                    [(2.89+1)/2, df.soil_temperature_level_4.values[0]],
                    [max_depth, df.soil_temperature_level_4.values[0]],
                ])
            
            wet_points = np.array([
                [0, df.soil_temperature_level_1_offs.values[0]],
                [(0.07+0)/2, df.soil_temperature_level_1_offs.values[0]],
                [(0.28+0.07)/2, df.soil_temperature_level_2_offs.values[0]],
                [(1+0.28)/2, df.soil_temperature_level_3_offs.values[0]],
                [(2.89+1)/2, df.soil_temperature_level_4_offs.values[0]],
                [max_depth, df.soil_temperature_level_4_offs.values[0]],
            ])
            
        else:
            
            # read data into dataframe
            df = pd.read_csv(os.path.join(self.proj_dir, self.config.data.initial_ground_temp_path), parse_dates=['time'])
                        
            # select correct row
            mask = (df['time'] == t_start)
            df = df[mask]
                        
            # read in points
            dry_points = np.array([
                [0.0, df['T50cm'].values[0] + 273.15],
                [0.5, df['T50cm'].values[0] + 273.15],
                [1.0, df['T100cm'].values[0] + 273.15],
                [2.0, df['T200cm'].values[0] + 273.15],
                [2.95, df['T295cm'].values[0] + 273.15],
                [max_depth, df['T295cm'].values[0] + 273.15],
            ])
            
            wet_points = dry_points
            
            
        ground_temp_distr_dry = um.interpolate_points(dry_points[:,0], dry_points[:,1], n)
        ground_temp_distr_wet = um.interpolate_points(wet_points[:,0], wet_points[:,1], n)
        
        return ground_temp_distr_dry, ground_temp_distr_wet
        
    def thermal_update(self, timestep_id, subgrid_timestep_id):
        """This function is called each subgrid timestep of each timestep, and performs the thermal update of the model.
        The C-matrices are not updated as they are a function of only density.

        Args:
            timestep_id (int): id of the current timestep
            subgrid_timestep_id (int): id of the current subgrid timestep
        """
        # get the phase masks, based on enthalpy
        frozen_mask, inbetween_mask, unfrozen_mask = self._get_phase_masks()
        
        # get temperature matrix
        self.temp_matrix = self._temperature_from_enthalpy(frozen_mask, inbetween_mask, unfrozen_mask)
        
        # determine the actual k-matrix using the masks
        self.k_matrix = \
            frozen_mask * self.k_frozen_matrix + \
            inbetween_mask * ((self.k_frozen_matrix + self.k_unfrozen_matrix) / 2) + \
            unfrozen_mask * self.k_unfrozen_matrix
        
        # get the new boundary condition
        self.ghost_nodes_temperature = self._get_ghost_node_boundary_condition(timestep_id, subgrid_timestep_id)
        self.bottom_boundary_temperature = self._get_bottom_boundary_temperature()
                
        # aggregate temperature matrix
        aggregated_temp_matrix = np.concatenate((
            self.ghost_nodes_temperature.reshape(len(self.xgr), 1),
            self.temp_matrix,
            self.bottom_boundary_temperature.reshape(len(self.xgr), 1)
            ), axis=1) 
        
        # determine the courant-friedlichs-lewy number matrix
        self.cfl_matrix = self.k_matrix / self.soil_density_matrix * self.config.thermal.dt / self.dz**2
        
        if np.max(self.cfl_matrix >= self.config.wrapper.CFL_thermal):
            # raise ValueError(f"CFL should be smaller than {self.config.wrapper.CFL_thermal}, currently {np.max(self.cfl_matrix):.4f}")
            # print(f"CFL should be smaller than 0.5, currently {np.max(self.cfl_matrix):.4f}")
            pass
            
        # get the new enthalpy matrix
        self.enthalpy_matrix = self.enthalpy_matrix + \
                               self.cfl_matrix * (aggregated_temp_matrix @ self.A_matrix)
        
        return None
    
    def _get_phase_masks(self):
        "Returns phase masks from current enthalpy distribution."
        # determine state masks (which part of the domain is frozen, in between, or unfrozen (needed to later calculate temperature from enthalpy))
        frozen_mask = (self.enthalpy_matrix)  < (self.config.thermal.T_melt * self.Cs_matrix)
        
        unfrozen_mask = (self.enthalpy_matrix) > (
            self.config.thermal.T_melt * self.Cl_matrix + \
            (self.Cs_matrix - self.Cl_matrix) * self.config.thermal.T_melt + \
            self.config.thermal.L_water_ice * self.nb_matrix
            )
        
        inbetween_mask = np.ones(frozen_mask.shape) - frozen_mask - unfrozen_mask
        
        return frozen_mask, inbetween_mask, unfrozen_mask
    
    def _temperature_from_enthalpy(self, frozen_mask, inbetween_mask, unfrozen_mask):
        """Returns the temperature matrix for given phase masks."""
        temp_matrix = \
            frozen_mask * \
                (self.enthalpy_matrix / self.Cs_matrix) + \
            inbetween_mask * \
                (self.config.thermal.T_melt) + \
            unfrozen_mask * \
                (self.enthalpy_matrix - \
                (self.Cs_matrix - self.Cl_matrix) * self.config.thermal.T_melt - \
                self.config.thermal.L_water_ice * self.nb_matrix) / \
                    (self.Cl_matrix)
                    
        return temp_matrix
     
    def _get_ghost_node_boundary_condition(self, timestep_id, subgrid_timestep_id):
        """This function uses the forcing at a specific timestep to return an array containing the ghost node temperature.

        Args:
            timestep_id (int): index of the current timestep.
            subgrid_timestep_id (int): id of the current subgrid timestep

        Returns:
            array: temperature values for the ghost nodes.
        """
        # first get the correct forcing timestep
        row = self.forcing_data.iloc[timestep_id]
        
        # with associated forcing values
        self.current_air_temp = row["2m_temperature"]  # also used in output
        self.current_sea_temp = row['sea_surface_temperature']  # also used in output
        self.current_sea_ice = row["sea_ice_cover"]  # not used in this function, but loaded in preperation for output
        
        # update the water level
        self.water_level = (self._update_water_level(timestep_id, subgrid_timestep_id=subgrid_timestep_id))
        
        dry_mask = (self.zgr >= self.water_level + self.R2)
        wet_mask = (self.zgr < self.water_level + self.R2)
        
        # determine convective transport from air (formulation from Man, 2023)
        self.wind_direction, self.wind_velocity = self._get_wind_conditions(timestep_id=timestep_id)  # also used in output
        convective_transport_air = Simulation._calculate_sensible_heat_flux_air(
            self.wind_velocity, 
            self.temp_matrix[:,0], 
            self.current_air_temp, 
            )
        
        # use either an educated guess from Kobayashi et al (1999), or their entire formulation
        # the entire formulation is a bit slower because of the dispersion relation that needs to be solved
        if self.config.thermal.with_convective_transport_water_guess:
            hc = self.config.thermal.hc_guess
        else:
            
            raise Exception("This piece of code is outdated")
            
            # determine hydraulic parameters for convective heat transfer computation
            
            if self.xb_times[timestep_id] and self.config.xbeach.with_xbeach:
                
                data_path = os.path.join(self.cwd, "xboutput.nc")
                
                ds = xr.load_dataset(data_path)
                
                H = ds.H.values.flatten()

                wl = self._update_water_level(timestep_id)
                
                d = np.maximum(wl - ds.zb.values.flatten(), np.zeros(H.shape))
                T = self.conditions[timestep_id]["Tp(s)"]
                
                ds.close()
                                
            else:
                H = self.conditions[timestep_id]['Hs(m)']
                
                wl = self._update_water_level(timestep_id)
                
                d = np.maximum(wl - self.zgr, 0)
                T = self.conditions[timestep_id]['Tp(s)']
                
            # determine convective transport from water (formulation from Kobayashi, 1999)
            hc = Simulation._calculate_sensible_heat_flux_water(
                H, T, d, self.config.xbeach.rho_sea_water,
                CW=3989, alpha=0.5, nu=1.848*10**-6, ks=2.5*1.90*10**-3, Pr=13.4
            )
        
        # scale hc with temperature difference
        convective_transport_water = hc * (self.current_sea_temp - self.temp_matrix[:,0])
            
        # compute total convective transport
        self.convective_flux = dry_mask * convective_transport_air + wet_mask * convective_transport_water  # also used in output
        
        if subgrid_timestep_id == 0:  # determine radiation fluxes only during first subgrid timestep, as they are constant for each subgrid timestep
        
            # determine radiation, assuming radiation only influences the dry domain
            self.latent_flux = dry_mask * (row["mean_surface_latent_heat_flux"] if self.config.thermal.with_latent else 0)  # also used in output
            self.lw_flux = dry_mask * (row["mean_surface_net_long_wave_radiation_flux"] if self.config.thermal.with_longwave else 0)  # also used in output
            
            if self.config.thermal.with_solar:  # also used in output
                I0 = row["mean_surface_net_short_wave_radiation_flux"]  # float value
                
                if self.config.thermal.with_solar_flux_calculator:
                    self.sw_flux = dry_mask * self._get_solar_flux(I0, timestep_id)  # sw_flux is now an array instead of a float
                else:
                    self.sw_flux = dry_mask * I0
            else:
                self.sw_flux = np.zeros(self.xgr.shape)
            
            # save this constant flux
            self.constant_flux = self.latent_flux + self.lw_flux + self.sw_flux
        
        # add all heat fluxes  together (also used in output)
        self.heat_flux = self.convective_flux + self.constant_flux
        
        # compute heat flux factors
        if 'heat_flux_factors' in self.config.thermal.keys():
            self.heat_flux_factors = (np.abs(self.angles / (2 * np.pi) * 360) < self.config.thermal.surface_flux_angle) * self.config.thermal.surface_flux_factor
        else:
            self.heat_flux_factors = np.ones(self.xgr.shape)
        
        # multiply with heat flux factor
        self.heat_flux = self.heat_flux * self.heat_flux_factors
        
        # determine temperature of the ghost nodes
        ghost_nodes_temperature = self.temp_matrix[:,0] + self.heat_flux * self.dz / self.k_matrix[:,0]
        
        return ghost_nodes_temperature
    
    def _update_water_level(self, timestep_id, subgrid_timestep_id=0):


        # get the water level
        if self.xbeach_times[timestep_id]:
            
            # load dataset
            ds = xr.load_dataset(os.path.join(self.cwd, "xboutput.nc")).squeeze()  # get xbeach data
            
            # select only the final timestep
            ds = ds.sel(globaltime=np.max(ds.globaltime.values))
            
            # load zs values
            zs_values = ds.zs.values.flatten()
            
            # remove nan values and get maximum
            self.water_level = np.max(zs_values[~np.isnan(zs_values)])
            
            # close dataset
            ds.close()

            
        else:
            self.water_level = self.water_levels[timestep_id]
            
        return self.water_level
    
    def _get_bottom_boundary_temperature(self):
        """This function returns a bottom boundary condition temperature, based on the geothermal gradient. It accounts for the
        angle that each 1D thermal grid makes with the horizontal.

        Returns:
            array: bottom temperature
        """
        bottom_temp = self.temp_matrix[:,-1]
        
        vertical_dist = np.cos(np.abs(self.angles)) * self.dz
        
        return bottom_temp + vertical_dist * self.config.thermal.geothermal_gradient
    
    @classmethod
    def _calculate_sensible_heat_flux_air(self, v_w, T_soil_surface, T_air, L_e=0.003, nu_air=1.33*10**-5, Pr=0.71, k_air=0.024):
        """This function computes the sensible heat flux Qs [W/m2] at the soil surface, using a formulation described by Man (2023).
            A positive flux means that the flux is directed into the soil.

        Args:
            v_w (float): wind speed at 10-meter height [m/s],
            T_soil_surface (array): an array containing the soil surface temperature [K],
            T_air (float): air temperature [K],
            L_e (float): convective length scale [m]. Defaults to 0.003.
            nu_air (float, optional): air kinematic viscosity. Defaults to 1.33*10**-5.
            Pr (float, optional): Prandtl number. Defaults to 0.71.
            k_air (float, optional): thermal conductivity of air. Defaults to 0.024.

        Returns:
            array: array containing the sensible heat flux for each point for which a soil surface temperature was provided.
        """
       
        Qs =  0.0296 * (v_w * L_e / nu_air)**(4/5) * Pr**(1/3) * k_air / L_e * (T_air - T_soil_surface)
       
        return Qs
    
    @classmethod
    def _calculate_sensible_heat_flux_water(
        H, T, d, rho_seawater, CW=3989, alpha=0.5, nu=1.848*10**-6, ks=2.5*1.90*10**-3, Pr=13.4
        ):
        """This function calculates the sensible heat flux between (sea) water and soil. The computation is based on Kobayashi 
        et al (1999) for the general formulation of the heat flux, Kobayashi & Aktan (1986) for specific formulations of 
        parameters, and Jonsson (1966) for specific parameter values. Note: the formulation by Kobayashi et al (1999) is meant
        specifically for breaking waves. A positive flux means that the flux is directed into the soil."""
        # check for nan values in H array
        mask = np.nonzero(1 - np.isnan(H))
        
        # calculate k based on linear wave theory
        kr = []
        for dr in d[mask]:
            
            kr.append(dispersion(2 * np.pi / T, dr))
        
        # convert to array
        kr = np.array(kr)
        
        # compute volumetric heat capacity
        cw = CW * rho_seawater
        
        # compute representative fluid velocity immediately outside the boundary layer
        u_b = np.pi * H[mask] / (T * np.sinh(kr * dr))
        
        # fw is set to 0.05
        fw = Simulation._get_fw()
        
        # calculate u-star, which is necessary for determining the final coefficient
        u_star = np.sqrt(0.5 * fw) * u_b
        
        # the roughness should be checked to be above 70, otherwise another formulation should be used for the coefficient E
        roughness = u_star * ks / nu
        
        # parameter depending on whether the turbulent boundary layer flow is hydraulically smooth or fully rough
        E = 0.52 * roughness**0.45 * Pr**0.8
        
        # sensible heat flux factor [W/m2/K]
        hc = alpha * fw * cw * u_b / (1 + np.sqrt(0.5 * fw) * E)
        
        return hc
        
    @classmethod
    def _get_fw():
        """this computation is required to get values from the graph providing fw in Jonsson (1966)"""
        # Reynolds number
        # Re = np.max(d[mask]) * u_b / nu
        
        # maximum surface elevation, and particle amplitude.
        # a = H[mask] / 2
        
        # z0 = - d[mask] + h_bed
        # amx = a * np.cosh(kr * (d[mask] + z0)) / np.sinh(kr * d[mask])
        # amz = a * np.sinh(kr * (d[mask] + z0)) / np.sinh(kr * d[mask])
        
        # with the Reynolds number and the maximum particle displacement divided by k, a value for fw can be read.
        # this value has a high uncertainty. 
        # a value of 0.05 is used here, which was found to be somewhat representible
        
        return 0.05

    
    def update_grid(self, timestep_id, fp_xbeach_output="sedero.txt"):
        """This function updates the current grid, calculates the angles of the new grid with the horizontal, generates a new thermal grid 
        (perpendicular to the existing grid), and fits the previous temperature and enthalpy distributions to the new grid."""
        # update the current bathymetry
        cum_sedero = self._get_cum_sedero(fp_xbeach_output=fp_xbeach_output)  # placeholder
        
        # update bed level
        bathy_current = self.zgr + cum_sedero
        
        # only update the grid of there actually was a change in bed level
        if not all(cum_sedero == 0):
        
            # generate a new xgrid and zgrid (but only if the next timestep does not require a hotstart, which requires the same xgrid)
            if timestep_id + 1 < len(self.xbeach_times) and not self.xbeach_times[timestep_id + 1]:
                                
                self.xgr_new, self.zgr_new = xgrid(self.xgr, bathy_current, dxmin=2, ppwl=self.config.bathymetry.ppwl)
                self.zgr_new = np.interp(self.xgr_new, self.xgr, bathy_current)
                
                # ensure that the grid doesn't extend further offshore than the original grid (this is a bug in the xbeach python toolbox)
                while self.xgr_new[0] < self.x_ori:
                    self.xgr_new = self.xgr_new[1:]
                    self.zgr_new = self.zgr_new[1:]
                
            else:
                self.xgr_new = self.xgr
                self.zgr_new = bathy_current

            # generate perpendicular grids for next timestep (to cast temperature and enthalpy)
            self.abs_xgr_new, self.abs_zgr_new = um.generate_perpendicular_grids(
                self.xgr_new, 
                self.zgr_new, 
                resolution=self.config.thermal.grid_resolution, 
                max_depth=self.config.thermal.max_depth
            )
            
            # cast temperature matrix
            if self.config.thermal.grid_interpolation == "linear_interp_with_nearest":
                self.temp_matrix = um.linear_interp_with_nearest(self.abs_xgr, self.abs_zgr, self.temp_matrix, self.abs_xgr_new, self.abs_zgr_new)
                self.enthalpy_matrix = um.linear_interp_with_nearest(self.abs_xgr, self.abs_zgr, self.enthalpy_matrix, self.abs_xgr_new, self.abs_zgr_new)
            
            elif self.config.thermal.grid_interpolation == "linear_interp_z":
                
                # Compute density
                rho_value = self.config.thermal.nb_max * self.config.thermal.rho_water + (1 - self.config.thermal.nb_max) * self.config.thermal.rho_particle
                
                # From denisty, compute specific heat
                Cs_value = self.config.thermal.c_soil_frozen / rho_value
                Cl_value = self.config.thermal.c_soil_unfrozen / rho_value
                
                # Compute top fill value for submerged sediment
                if (self.current_sea_temp <= self.config.thermal.T_melt):
                    enthalpy_submerged_sediment = self.current_sea_temp * Cs_value
                else:
                    enthalpy_submerged_sediment = self.current_sea_temp * Cl_value + (Cs_value - Cl_value) * self.config.thermal.T_melt + self.config.thermal.L_water_ice * self.config.thermal.nb_max
                
                # get current water level
                self.water_level = (self._update_water_level(timestep_id))
                
                # Interpolate enthalpy to new grid             
                self.enthalpy_matrix = um.linear_interp_z(
                    self.abs_xgr, 
                    self.abs_zgr, 
                    self.enthalpy_matrix, 
                    self.abs_xgr_new, 
                    self.abs_zgr_new,
                    water_level=self.water_level,
                    fill_value_top_water=enthalpy_submerged_sediment,
                    fill_value_top_air='nearest',
                    )
                
                # interpolate nb to new grid
                self.nb_matrix = um.linear_interp_z(
                    self.abs_xgr,
                    self.abs_zgr,
                    self.nb_matrix,
                    self.abs_xgr_new,
                    self.abs_zgr_new,
                    water_level=self.water_level,
                    fill_value_top_water=self.config.thermal.nb_max,
                    fill_value_top_air=self.config.thermal.nb_max,
                )
                
                # redefine matrices with soil properties
                self.define_soil_property_matrices(len(self.xgr_new), define_nb=False)
                
            else:
                raise ValueError("Invalid value for grid_interpolation")
            
            # set the grid to be equal to this new grid
            self.xgr = self.xgr_new
            self.zgr = self.zgr_new
            
            self.abs_xgr = self.abs_xgr_new
            self.abs_zgr = self.abs_zgr_new
            
            # update the angles
            self._update_angles()
        
        return None
        
    def _update_angles(self):
        """This function geneartes an array of local angles (in radians) for the grid, based on the central differences method.
        """
        self.angles = np.gradient(self.zgr, self.xgr)
        
        return self.angles
    
    def _get_cum_sedero(self, fp_xbeach_output):
        """This method updates the current bed given the xbeach output.
        ---------
        fp_xbeach_output: string
            filepath to the xbeach sedero (sedimentation-erosion) output relative to the current working directory."""
            
        # Read output file
        ds = xr.load_dataset(fp_xbeach_output)
        ds = ds.sel(globaltime = np.max(ds.globaltime.values)).squeeze()
        
        cum_sedero = ds.sedero.values
        xgr = ds.x.values
        
        ds.close()

        # Create an interpolation function
        interpolation_function = interp1d(xgr, cum_sedero, kind='linear', fill_value='extrapolate')
        
        # interpolate values to the used grid
        interpolated_cum_sedero = interpolation_function(self.xgr)
        
        return interpolated_cum_sedero
    
    def _get_solar_flux(self, I0, timestep_id):
        """This function is used to obtain an array of incoming solar radiation for some timestep_id, with values for each grid point in the computational domain.

        Args:
            I0 (float): incoming radiation (flat surface)
            timestep_id (int): index of the current timestep

        Returns:
            array: incoming solar radiation for each grid point in the computational domain
        """        
        # get current timestamp
        timestamp = self.timestamps[timestep_id]
        
        # get id of current timestamp w.r.t. solar_flux_map (-1 because: 'minimum id is 0' and 'minimum day of year is 1')
        id_t = timestamp.dayofyear - 1  # factors are associated with day of year only (as it is based on maximum angle per day)
        
        # get correct row in solar flux map (so row corresponding to current day of the year)
        row = self.solar_flux_map[id_t, :]
        
        # transform angles to ids (first convert to degrees)
        ids_angle = np.int32((self.angles / (2*np.pi) * 360 - self.config.thermal.angle_min) / self.config.thermal.delta_angle)
        
        # use ids to get correct factors in correct order from the row of solar fluxes
        self.factors = row[ids_angle]
        
        solar_flux = I0 * self.factors
        
        return solar_flux
        
    def initialize_solar_flux_calculator(self, timezone_diff, angle_min=-89, angle_max=89, delta_angle=1, t_start='2000-01-01', t_end='2001-01-01'):
        """This function initializes a mapping variable for the solar flux calculator. This is required because the enhancement factor is calculated using the
        maximum insolance per day, making it impossible to calculate each hour seperately. Specific values for enhancement factor are indexed using an angle,
        followed by the number of the current day of the year minus 1.

        Args:
            timezone_diff (int): describes the difference in timezone between UTC and the area of interest
            angle_min (int, optional): minimum angle in the mapping variable. Defaults to -89.
            angle_max (int, optional): maximum angle in the mapping. Defaults to 89.
            delta_angle (int, optional): diffence between mapped angles. Defaults to 1.
            t_start (str, optional): start of the daterange used to calculate the enhancement factor. It is recommended to use a full leap year. Defaults to '2000-01-01'.
            t_end (str, optional): end of the daterange used to calculate the enhancement factor. It is recommended to use a full leap year.. Defaults to '2001-01-01'.

        Returns:
            dictionary: the mapping variable used to quickly obtain solar flux enhancement factor values.
        """        
        self.solar_flux_angles = np.arange(angle_min, angle_max+1, delta_angle)
        
        t_start_datetime = pd.to_datetime(t_start)
        t_end_datetime = pd.to_datetime(t_end)
        self.solar_flux_times = pd.date_range(t_start_datetime, t_end_datetime, freq='1h', inclusive='left')
                
        self.solar_flux_map = np.zeros((np.int32(len(self.solar_flux_times)/24), len(self.solar_flux_angles)))
        
        for angle in self.solar_flux_angles:
            
            angle_id = np.nonzero(angle==self.solar_flux_angles)
            
            # for each integer angle in the angle range, an array of enhancement factors is saved, indexable by N (i.e., the N-th day of the year)
            self.solar_flux_map[:, angle_id] = self._calculate_solar_flux_factors(self.solar_flux_times, angle, timezone_diff).reshape((-1, 1, 1))
            
        # can't have negative factors (which may occur in winter when the angle between light rays and a flat surface is negative but between light rays and inclined surface (facing southward) is positive)
        self.solar_flux_map[np.nonzero(self.solar_flux_map < 0)] = 0
            
        np.savetxt(os.path.join(self.result_dir, 'solar_flux_map.txt'), self.solar_flux_map)
        
        return self.solar_flux_map
    
    def _calculate_solar_flux_factors(self, daterange, angle, timezone_diff):
        """
        This function calculates the effective solar radiation flux on a sloped surface. The method from Buffo (1972) is used, 
        assuming that the radiaton on the surface already includes the atmospheric transmission coefficient. Using the radiation data for a flat surface 
        and the angle of the incoming rays with the flat sruface, the intensity of the incoming rays can be estimated, which can then be projected on an inclined
        surface.

        Args:
            daterange (daterange): the dates for which to calculate the solar flux.
            angle (float): incline of the surface for which to calculate the enhancement factors.
            timezone_diff (float): difference in hours for the timezone which is modelled relative to UTC.

        Returns:
            array: enhancement factors for radiation for the given angle for each day in the daterange
        """       
        
        # 1) latitude and orientation
        phi = self.config.model.latitude / 360 * 2 * np.pi
        beta = (90 - self.config.bathymetry.grid_orientation) / 360 * 2 * np.pi  # clockwise from the north
        
        # 2) local angles
        alpha = angle / 360 * 2 * np.pi
        
        # 3) declination, Sarbu (2017)
        delta = 23.45 * np.sin(
            (360/365 * (284 + daterange.dayofyear.values)) / 360 * 2 * np.pi
            ) / 360 * 2 * np.pi
        
        # 4) hour angle (for Alaska timezone difference w.r.t. UTC is -8h)
        local_hour_of_day = daterange.hour.values + timezone_diff
            # convert to hour angle
        h = (((local_hour_of_day - 12) % 24)/24) * 2 * np.pi
            # convert angles to range [-pi, pi]
        mask = np.nonzero(h>=np.pi)
        h[mask] = -((2 * np.pi) - h[mask])
        
        # 5) calculate altitude angle off of the horizontal that the suns rays strike a horizontal surface
        A = np.arcsin(np.cos(phi) * np.cos(delta) * np.cos(h) + np.sin(phi) * np.sin(delta))
        
        # 6) calculate the (unmodified) azimuth
        AZ_no_mod = np.arcsin((np.cos(delta) * (np.sin(h)) / np.cos(A)))
        
        AZ_mod = np.copy(AZ_no_mod)  # create a copy of the unmodified azimuth, which will be modified
        
        # correct azimuth for when close to solstices (“Central Beaufort Sea Wave and Hydrodynamic Modeling Study Report 1: Field Measurements and Model Development,” n.d.)
        ew_AM_mask = np.nonzero(
            (A > 0) * (np.cos(h) <= np.tan(delta) / np.tan(phi)) * (local_hour_of_day <= 12)
            ) # east-west AM mask
        ew_PM_mask = np.nonzero(
            (A > 0) * (np.cos(h) <= np.tan(delta) / np.tan(phi)) * (local_hour_of_day > 12)
            ) # east-west PM mask
        
        # modify azimuth
        AZ_mod[ew_AM_mask] = -np.pi + np.abs(AZ_no_mod[ew_AM_mask])
        AZ_mod[ew_PM_mask] = np.pi - AZ_no_mod[ew_PM_mask]
        
        # convert from (clockwise from south) to (clockwise from east)
        Z = AZ_mod + 1/2 * np.pi

        # 7) calculate multiplication factor for computational domain
        sin_theta = np.sin(A) * np.cos(alpha) - np.cos(A) * np.sin(alpha) * np.sin(Z - beta)
        theta = np.arcsin(sin_theta)
        
        # 8) calculate multiplication factor for flat surface
        sin_0 = np.sin(A) * np.cos(0) - np.cos(A) * np.sin(0) * np.sin(Z - beta)
        
        # 9) in order to avoid very peaky scales, let us take the daily maximum and use that for scaling.
        sin_theta_2d = sin_theta.reshape((-1, 24))
        sin_theta_daily_max = np.max(sin_theta_2d, axis=1).flatten()

        sin_0_2d = sin_0.reshape((-1, 24))
        sin_0_daily_max = np.max(sin_0_2d, axis=1).flatten()

        # 10) calculate enhancement factor for each day
        factor = sin_theta_daily_max / sin_0_daily_max
        
        # 11) filter out values where it the angle theta is negative (as that means radiation hits the surface from below)
        shadow_mask = np.zeros(factor.shape)
        
        for i, row in enumerate(theta.reshape((-1, 24))):
            if all(row < 0):
                shadow_mask[i] = 1
            
        factor[np.nonzero(shadow_mask)] = 0
        
        return factor
        
    def write_ne_layer(self):
        """This function writes the thaw depth obtained from the thermal update to a file to be used by xbeach.
        """
        np.savetxt(os.path.join(self.cwd, "ne_layer.txt"), self.thaw_depth)
                
        return None
        
    def find_thaw_depth(self):
        """Finds thaw depth based on the z-values of the two nearest thaw points."""
        # initialize thaw depth array
        self.thaw_depth = np.zeros(self.xgr.shape)

        # get the points from the temperature models
        x_matrix, z_matrix = um.generate_perpendicular_grids(
            self.xgr, self.zgr, 
            resolution=self.config.thermal.grid_resolution, 
            max_depth=self.config.thermal.max_depth)
        
        # determine indices of thaw depth in perpendicular model
        indices = um.count_nonzero_until_n_zeros(
            self.temp_matrix > self.config.thermal.T_melt, 
            dN=(self.config.thermal.N_thaw_threshold if "N_thaw_threshold" in self.config.thermal.keys() else 1)
            )
        
        # find associated coordinates of these points
        x_thaw = x_matrix[np.arange(x_matrix.shape[0]), indices]
        z_thaw = z_matrix[np.arange(x_matrix.shape[0]), indices]
        
        # sort 
        sort_indices = np.argsort(x_thaw)
        x_thaw_sorted = x_thaw[sort_indices]
        z_thaw_sorted = z_thaw[sort_indices]
        
        # loop through the grid        
        for i, x, z in zip(np.arange(len(self.xgr)), self.xgr, self.zgr):
            # try to find two points between which to interpolate for the thaw depth, otherwise set thaw depth to 0
            try:
                mask1 = np.nonzero((x_thaw_sorted < x))
                x1 = x_thaw_sorted[mask1][-1]
                z1 = z_thaw_sorted[mask1][-1]
                
                mask2 = np.nonzero((x_thaw_sorted >= x))
                x2 = x_thaw_sorted[mask2][0]
                z2 = z_thaw_sorted[mask2][0]
                
                z_thaw_interpolated = z1 + (z2 - z1)/(x2 - x1) * (x - x1)
                
                self.thaw_depth[i] = z - (z_thaw_interpolated)
            except:
                self.thaw_depth[i] = 0
        
        # ensure thaw depth is larger than zero everywhere
        self.thaw_depth = np.max(np.column_stack((self.thaw_depth, np.zeros(self.thaw_depth.shape))), axis=1)
        
        return self.thaw_depth
        
    ################################################
    ##                                            ##
    ##            # OUTPUT FUNCTIONS              ##
    ##                                            ##
    ################################################
        
    def write_output(self, timestep_id, t_start):
        """This function writes output in the results folder, and creates subfolders for each timestep for which results are output.
        """        
        # create dataset
        result_ds = xr.Dataset(
            coords={
                "xgr":self.xgr,  # 1D series of x-values
                "depth_id":np.arange(self.config.thermal.grid_resolution),  # 1D series of id's representing the node number (zero meaning surface, one the first node below surface, etc.)
                }
            )
        
        # time variables
        result_ds['timestep_id'] = timestep_id
        result_ds['timestamp'] = self.timestamps[timestep_id]
        result_ds['cumulative_computational_time'] = time.time() - t_start
        
        # bathymetric variables
        result_ds["zgr"] = (["xgr"], self.zgr)  # 1D series of z-values
        result_ds["angles"] = (["xgr"], self.angles)  # 1D series of angles (in radians)
        
        # hydrodynamic variables (note: obtained from xbeach output from previous timestep, so not necessarily accurate with other output data)
        if timestep_id and os.path.exists(os.path.join(self.cwd, "xboutput.nc")) and self.xbeach_times[timestep_id-1]:  # check if an xbeach output file exists (it shouldn't at the first timestep)
            
            ds = xr.load_dataset(os.path.join(self.cwd, "xboutput.nc")).squeeze()  # get xbeach data
            ds = ds.sel(globaltime=np.max(ds.globaltime.values))  # select only the final timestep
            
            # determine the x coordinates from the computational grid
            xgr_xb = ds.x.values
            
            # use x coordinates from the computational grid instead of global values
            result_ds = result_ds.assign_coords(xgr_xb=xgr_xb)

            result_ds['wave_height'] = (["xgr_xb"], ds.H.values.flatten())  # 1D series of wave heights (associated with xgr.txt)
            result_ds['zb'] = (["xgr_xb"], ds.zb.values.flatten()) # 1D series of bed levels
            result_ds['zs'] = (["xgr_xb"], ds.zs.values.flatten()) # 1D series of water levels
            result_ds['wave_energy'] = (["xgr_xb"], ds.E.values.flatten())  # 1D series of wave energies (associated with xgr.txt)
            result_ds['radiation_stress_xx'] = (["xgr_xb"], ds.Sxx.values.flatten())  # 1D series of radiation stresses (associated with xgr.txt)
            result_ds['radiation_stress_xy'] = (["xgr_xb"], ds.Sxy.values.flatten())  # 1D series of radiation stresses (associated with xgr.txt)
            result_ds['radiation_stress_yy'] = (["xgr_xb"], ds.Syy.values.flatten())  # 1D series of radiation stresses (associated with xgr.txt)
            # result_ds['mean_wave_angle'] = (["xgr_xb"], ds.thetamean.values.flatten())  # 1D series of mean wave angles in radians (associated with xgr.txt)
            result_ds['velocity_magnitude'] = (["xgr_xb"], ds.vmag.values.flatten())  # 1D series of velocities (associated with xgr.txt)
            result_ds['orbital_velocity'] = (["xgr_xb"], ds.urms.values.flatten())  # 1D series of velocities (associated with xgr.txt                
        
            ds.close()
            
        else:
            
            xgr_xb = self.xgr[np.nonzero(self.zgr <= 0)]
            
            result_ds = result_ds.assign_coords(xgr_xb=xgr_xb)
            
            for varname in ['wave_height', 'wave_energy', 'zb', 'zs',
                            'radiation_stress_xx', 'radiation_stress_xy', 'radiation_stress_yy', 
                            'mean_wave_angle', 'velocity_magnitude', 'orbital_velocity']:
                result_ds[varname] = (["xgr_xb"], np.zeros(xgr_xb.shape))
        
        # water level
        result_ds['water_level'] = self._update_water_level(timestep_id)
        
        # computed 2% runup
        result_ds['run_up2%'] = self.R2
        
        # temperature variables
        result_ds['thaw_depth'] = (["xgr"], self.thaw_depth)  # 1D series of thaw depths
        result_ds['abs_xgr'] = (["xgr", "depth_id"], self.abs_xgr)  # 1D series of x-values (corresponding to ground_temperature_distribution.txt and grount_enthalpy_distribution.txt)
        result_ds['abs_zgr'] = (["xgr", "depth_id"], self.abs_zgr)  # 1D series of z-values (corresponding to ground_temperature_distribution.txt and grount_enthalpy_distribution.txt)
        result_ds['ground_temperature_distribution'] = (["xgr", "depth_id"], self.temp_matrix)  # 2D grid of temperature values (associated with abs_xgr.txt and abs_zgr.txt)
        result_ds['ground_enthalpy_distribution'] = (["xgr", "depth_id"], self.enthalpy_matrix)  # 2D grid of enthalpy values (associated with abs_xgr.txt and abs_zgr.txt)
        result_ds['nb'] = (["xgr", "depth_id"], self.nb_matrix)  # 2D grid of nb values
        result_ds['k'] = (["xgr", "depth_id"], self.k_matrix)  # 2D grid of k values
        result_ds['rho'] = (["xgr", "depth_id"], self.soil_density_matrix)  # 2D grid of density values
        result_ds['2m_temperature'] = self.current_air_temp  # single value
        result_ds['sea_surface_temperature'] = self.current_sea_temp  # single value
        
        # heat flux variables
        result_ds['solar_radiation_factor'] = (["xgr"], self.factors)  # 1D series of factors
        result_ds['solar_radiation_flux'] = (["xgr"], self.sw_flux)  # 1D series of heat fluxes
        result_ds['long_wave_radiation_flux'] = (["xgr"], self.lw_flux)  # 1D series of heat fluxes
        result_ds['latent_heat_flux'] = (["xgr"], self.latent_flux)  # 1D series of heat fluxes
        result_ds['convective_heat_flux'] = (["xgr"], self.convective_flux)  # 1D series of heat fluxes
        result_ds['total_heat_flux'] = (["xgr"], self.heat_flux)  # 1D series of heat fluxes
        
        # sea ice variables
        result_ds['sea_ice_cover'] = self.current_sea_ice  # single value
        
        # wind variables
        result_ds['wind_velocity'] = self.wind_velocity  # single value
        result_ds['wind_direction'] = self.wind_direction  # single value (degrees, clockwise from the north)
        
        result_ds.to_netcdf(os.path.join(self.result_dir, (10 - len(str(int(timestep_id)))) * '0' + str(int(timestep_id)) + ".nc"))
        
        result_ds.close()
        
        return None
    
    def save_ground_temp_layers_in_memory(self, timestep_id, layers=[], heat_fluxes=[], write=False):
        """This function saves the ground temperature directly into a single dataframe, 
        which is helpfule for validation purposes.

        Args:
            timestep_id (int): id of the current timestep
            layers (list, optional): list of the layers to save. Defaults to [].
            heat_fluxes (list, optional): list of heat fluxes to save. Defaults to [].
            write (bool, optional): whether or not to write. Defaults to False.
        """
        # define colnames
        col_names = ['time'] + ['air_temp[K]'] + [f'temp_{layer}m[K]' for layer in layers] + heat_fluxes
        
        values = [self.timestamps[timestep_id], self.current_air_temp]
        
        # loop through layers to find corresponding temperature
        for layer in layers:
            
            index_x = int(self.temp_matrix.shape[0] // 2)
            index_z = int(layer * self.config.thermal.grid_resolution // self.config.thermal.max_depth)
            
            values.append(self.temp_matrix[index_x, index_z])
            
        # find heat fluxes
        values.append(self.heat_flux[index_x])
        values.append(self.lw_flux[index_x])
        values.append(self.sw_flux[index_x])
        values.append(self.latent_flux[index_x])
        values.append(self.convective_flux[index_x])

        # create dataframe at first timestep
        if timestep_id == 0:
                        
            self.temperature_timeseries = pd.DataFrame(dict(zip(col_names, values)), index=[0])
            
        else:
            
            # add temperature and heat fluxes to dataframe
            self.temperature_timeseries = self.temperature_timeseries._append(
                dict(zip(col_names, values)), ignore_index=True
            )
                
        # write output at final timestep
        if write:
            self.temperature_timeseries.to_csv(
                os.path.join(self.result_dir, f"{self.runid}_ground_temperature_timeseries.csv")
            )
            
        return None
        
    
    def _check_and_write(self, varname, save_var, dirname):
        
        if varname in self.config.output.output_vars:
            np.savetxt(os.path.join(dirname, f"{varname}" + ".txt"), save_var)
        
        return None
    
    def dump_xb_output(self, timestep_id):
        """This method copies the XB output from the run folder (including log files, param files, etc.) to the results directory."""
        
        destination_folder = os.path.join(self.result_dir, "xb_files/", (10 - len(str(int(timestep_id)))) * '0' + str(int(timestep_id)) + '/')

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            
        shutil.copytree(self.cwd, destination_folder, ignore=shutil.ignore_patterns('config.yaml', 'results/'), dirs_exist_ok=True)
        
        # os.rename(os.path.join())

        return None
    
    def write_xb_timesteps(self):
        """Used at the end of simulation, once all xbeach timesteps have been determined using 
        the sea ice threshold, intermediate xbeach timesteps, and storm conditions"""
        
        # output the xbeach timestep ids
        self._check_and_write('xbeach_times', self.xbeach_times, self.result_dir)
        
        return None
        
    # functions below are used to quickly obtain values for forcing data
    def _get_sw_flux(self, timestep_id):
        return self.forcing_data["mean_surface_net_short_wave_radiation_flux.csv"].values[timestep_id]
    def _get_lw_flux(self, timestep_id):
        return self.forcing_data["mean_surface_net_long_wave_radiation_flux.csv"].values[timestep_id]
    def _get_latent_flux(self, timestep_id):
        return self.forcing_data["mean_surface_latent_heat_flux.csv"].values[timestep_id]
    def _get_snow_depth(self, timestep_id):
        return self.forcing_data["snow_depth.csv"].values[timestep_id]
    def _get_sea_ice(self, timestep_id):
        return self.forcing_data["sea_ice_cover.csv"].values[timestep_id]
    def _get_2m_temp(self, timestep_id):
        return self.forcing_data["2m_temperature.csv"].values[timestep_id]
    def _get_sea_temp(self, timestep_id):
        return self.forcing_data["sea_surface_temperature.csv"].values[timestep_id]
    def _get_u_wind(self, timestep_id):
        return self.forcing_data["10m_u_component_of_wind.csv"].values[timestep_id]
    def _get_v_wind(self, timestep_id):
        return self.forcing_data["10v_u_component_of_wind.csv"].values[timestep_id]
    def _get_soil_temp(self, timestep_id, level=1):
        if not level in [1, 2, 3, 4]:
            raise ValueError("'level' variable should have a value of 1, 2, 3, or 4")
        return self.forcing_data[f"soil_temperature_level_{level}.csv"].values[timestep_id]

    def _get_timeseries(self, tstart, tend, fpath, repeat=1):
        """returns timeseries start from tstart and ending at tend. The filepath has to be specified.
        
        returns: pd.DataFrame of length T"""
        
        # read forcing file
        with open(fpath) as f:
            df = pd.read_csv(f, parse_dates=['time'])
                    
            # mask out correct time frame
            mask = (df["time"] >= tstart) * (df["time"] < tend)
            
            # repeat time frame if required
            df = pd.concat([df[mask]] * repeat, ignore_index=True)
                        
        return df
    
    ################################################
    ##                                            ##
    ##            # PLOTTING FUNCTIONS            ##
    ##                                            ##
    ################################################
    def plot_bathymetry(self, timestep_ids=[]):
        
        fig, ax = plt.subplots(figsize=(15, 5))

        for timestep_id in timestep_ids:
            
            dir = os.path.join(self.result_dir, str(timestep_id))
            
            xgr = np.loadtxt("xgr.txt")
            zgr = np.loadtxt("zgr.txt")
        
            ax.plot(xgr, zgr, label=f"timestep_id: {timestep_id}")
        
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        
        ax.legend()
                
        return fig
    
    
    
    ################################################
    ##                                            ##
    ##            # LEGACY CODE                   ##
    ##                                            ##
    ################################################
    
    # Old nb code
        # self.nb_distr = np.ones(self.thermal_zgr.shape)
        # idz = self.config.thermal.grid_resolution * self.config.thermal.nb_switch_depth / self.config.thermal.max_depth
        # self.nb_distr[:int(idz)] = self.config.thermal.nb_max  # set nb close to surface (nb_max)
        # self.nb_distr[int(idz):] = self.config.thermal.nb_min  # set nb at greater depth (nb_min)
    
    # Old code for writing output:
            # create directory
        #     if not os.path.exists(result_dir_timestep):
        #         os.makedirs(result_dir_timestep)
                
        #     # bathymetric variables
        #     self._check_and_write('xgr', self.xgr, dirname=result_dir_timestep)  # 1D series of x-values
        #     self._check_and_write('zgr', self.zgr, dirname=result_dir_timestep)  # 1D series of z-values
        #     self._check_and_write('angles', self.angles, dirname=result_dir_timestep)  # 1D series of angles (in radians)
            
        #     # hydrodynamic variables (note: obtained from previous xbeach timestep, so not necessarily accurate with other output data)
        #     xb_output_path = os.path.join(self.cwd, "xboutput.nc")
            
        #     if os.path.isfile(xb_output_path):  # check if an xbeach output file exists (it shouldn't at the first timestep)
                
        #         ds = xr.load_dataset(os.path.join(self.cwd, "xboutput.nc"))  # get xbeach data
                
        #         self._check_and_write('wave_height', ds.H.values.flatten(), dirname=result_dir_timestep)  # 1D series of wave heights (associated with xgr.txt)
        #         self._check_and_write('run_up', np.ones(1) * (ds.runup.values.flatten()), dirname=result_dir_timestep)  # single value
        #         self._check_and_write('storm_surge', np.ones(1) * (self.current_storm_surge), dirname=result_dir_timestep)  # single value
        #         self._check_and_write('wave_energy', ds.E.values.flatten(), dirname=result_dir_timestep)  # 1D series of wave energies (associated with xgr.txt)
        #         self._check_and_write('radiation_stress_xx', ds.Sxx.values.flatten(), dirname=result_dir_timestep)  # 1D series of radiation stresses (associated with xgr.txt)
        #         self._check_and_write('radiation_stress_xy', ds.Sxy.values.flatten(), dirname=result_dir_timestep)  # 1D series of radiation stresses (associated with xgr.txt)
        #         self._check_and_write('radiation_stress_yy', ds.Syy.values.flatten(), dirname=result_dir_timestep)  # 1D series of radiation stresses (associated with xgr.txt)
        #         self._check_and_write('mean_wave_angle', ds.thetamean.values.flatten(), dirname=result_dir_timestep)  # 1D series of mean wave angles in radians (associated with xgr.txt)
        #         self._check_and_write('velocity_magnitude', ds.vmag.values.flatten(), dirname=result_dir_timestep)  # 1D series of velocities (associated with xgr.txt)
        #         self._check_and_write('orbital_velocity', ds.urms.values.flatten(), dirname=result_dir_timestep)  # 1D series of velocities (associated with xgr.txt)
                
        #         ds.close()
            
        #     else:        
        #         self._check_and_write('wave_height', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of wave heights (associated with xgr.txt)
        #         self._check_and_write('run_up', np.ones(1) * (0), dirname=result_dir_timestep)  # single value
        #         self._check_and_write('storm_surge', np.ones(1) * (0), dirname=result_dir_timestep)  # single value
        #         self._check_and_write('wave_energy', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of wave energies (associated with xgr.txt)
        #         self._check_and_write('radiation_stress_xx', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of radiation stresses (associated with xgr.txt)
        #         self._check_and_write('radiation_stress_xy', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of radiation stresses (associated with xgr.txt)
        #         self._check_and_write('radiation_stress_yy', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of radiation stresses (associated with xgr.txt)
        #         self._check_and_write('mean_wave_angle', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of mean wave angles in radians (associated with xgr.txt)
        #         self._check_and_write('velocity_magnitude', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of velocities (associated with xgr.txt)
        #         self._check_and_write('orbital_velocity', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of velocities (associated with xgr.txt)
            
        #     # temperature variables
        #     self._check_and_write('thaw_depth', self.thaw_depth, dirname=result_dir_timestep)  # 1D series of thaw depths
        #     self._check_and_write('abs_xgr', self.abs_xgr.flatten(), dirname=result_dir_timestep)  # 1D series of x-values (corresponding to ground_temperature_distribution.txt and grount_enthalpy_distribution.txt)
        #     self._check_and_write('abs_zgr', self.abs_zgr.flatten(), dirname=result_dir_timestep)  # 1D series of z-values (corresponding to ground_temperature_distribution.txt and grount_enthalpy_distribution.txt)
        #     self._check_and_write('ground_temperature_distribution', self.temp_matrix.flatten(), dirname=result_dir_timestep)  # 1D series of temperature values (associated with abs_xgr.txt and abs_zgr.txt)
        #     self._check_and_write('ground_enthalpy_distribution', self.enthalpy_matrix.flatten(), dirname=result_dir_timestep)  # 1D series of enthalpy values (associated with abs_xgr.txt and abs_zgr.txt)
        #     self._check_and_write("2m_temperature", np.ones(1) * (self.current_air_temp), dirname=result_dir_timestep)  # single value
        #     self._check_and_write("sea_surface_temperature",  np.ones(1) * (self.current_sea_temp), dirname=result_dir_timestep)  # single value
            
        #     # heat flux variables
        #     self._check_and_write('solar_radiation_factor', self.factors, dirname=result_dir_timestep)  # 1D series of factors
        #     self._check_and_write('solar_radiation_flux', self.sw_flux, dirname=result_dir_timestep)  # 1D series of heat fluxes
        #     self._check_and_write('long_wave_radiation_flux', np.ones(1) * (self.lw_flux), dirname=result_dir_timestep)  # single value of heat flux
        #     self._check_and_write('latent_heat_flux',np.ones(1) * (self.latent_flux), dirname=result_dir_timestep)  # single value of heat flux
        #     self._check_and_write('convective_heat_flux', self.convective_flux, dirname=result_dir_timestep)  # 1D series of heat fluxes
        #     self._check_and_write('total_heat_flux', self.heat_flux, dirname=result_dir_timestep)  # 1D series of heat fluxes
            
        #     # sea ice variables
        #     self._check_and_write('sea_ice_cover', np.ones(1) * (self.current_sea_ice), dirname=result_dir_timestep)  # single value
            
        #     # wind variables
        #     self._check_and_write('wind_velocity', np.ones(1) * (self.wind_velocity), dirname=result_dir_timestep)  # single value
        #     self._check_and_write('wind_direction', np.ones(1) * (self.wind_direction), dirname=result_dir_timestep) # single value (degrees, clockwise from the north)
            
        #     return None
        
    
        
    
    # Old code from determining aggregated matrices (but that turned out to be the wrong order)
            
        # # determine which part of the domain is frozen and unfrozen (for the aggregated temperature matrix)
        # frozen_mask_aggr = (aggregated_temp_matrix < self.config.thermal.T_melt)
        # unfrozen_mask_aggr = np.ones(frozen_mask_aggr.shape) - frozen_mask_aggr
        
        # # determine k-matrix (extend k-distribution with one copied value at the top and bottom boundary to include ghost nodes)
        # k_frozen_distr_aggr = np.concatenate((
        #     np.array([self.k_frozen_distr[0]]),
        #     self.k_frozen_distr,
        #     np.array([self.k_frozen_distr[-1]])))
        
        # k_unfrozen_distr_aggr = np.concatenate((
        #     np.array([self.k_unfrozen_distr[0]]),
        #     self.k_frozen_distr,
        #     np.array([self.k_unfrozen_distr[-1]])))
        
        # self.k_matrix = frozen_mask_aggr * np.tile(k_frozen_distr_aggr, (len(self.xgr), 1)) + \
        #                 unfrozen_mask_aggr * np.tile(k_unfrozen_distr_aggr, (len(self.xgr), 1))
        
        # # determine aggregated density matrix
        # self.soil_density_matrix_aggr = np.hstack((
        #     self.soil_density_matrix[:,0].reshape((-1, 1)),
        #     self.soil_density_matrix,
        #     self.soil_density_matrix[:,-1].reshape((-1, 1)),
        # ))
    
    # Old code from ghost nodes
    # temperature difference for convective heat transfer (define temperature flux as positive when it is directed into the ground)
    # temp_diff_at_interface = (air_temp - self.temp_matrix[:,0]) * dry_mask + (sea_temp - self.temp_matrix[:,0]) * wet_mask
    # # apply boundary conditions
    # frozen_mask = (self.temp_matrix[:,0] < self.config.thermal.T_melt)
    # unfrozen_mask = np.ones(frozen_mask.shape) - frozen_mask

    # cfl_matrix = frozen_mask * self.cfl_frozen + unfrozen_mask * self.cfl_unfrozen
    
    # calculate the new enthalpy
        # 1) calculate temperature diffusion
    # ghost_nodes_enth = self.enthalpy_matrix[:,0] + cfl_matrix * (-self.temp_matrix[:,0] + self.temp_matrix[:,1])

        # 2) add radiation, assuming radiation only influences the dry domain
    
    # ghost_nodes_enth += \
    #     (self.config.model.timestep*3600) * dry_mask * \
    #     (latent_flux + sw_flux + lw_flux) / \
    #     (frozen_mask * (0.5 * self.dz) * 1 * 1 * self.config.thermal.rho_soil_frozen + unfrozen_mask * (0.5 * self.dz) * 1 * 1 * self.config.thermal.rho_soil_unfrozen)
    
        # 3) add convective heat transfer from water and air
    # ghost_nodes_enth += \
    #     (self.config.model.timestep*3600) * \
    #     temp_diff_at_interface * \
    #         (frozen_mask * self.config.thermal.k_soil_frozen + unfrozen_mask * self.config.thermal.k_soil_unfrozen) * \
    #     self.dz / \
    #         (frozen_mask * (0.5 * self.dz) * 1 * 1 * self.config.thermal.rho_soil_frozen + unfrozen_mask * (0.5 * self.dz) * 1 * 1 * self.config.thermal.rho_soil_unfrozen)
    
    # determine the temperature distribution
    # ghost_nodes_temperature = \
    #     frozen_mask * \
    #         (ghost_nodes_enth / (self.config.thermal.c_soil_frozen / self.config.thermal.rho_soil_frozen)) + \
    #     unfrozen_mask * \
    #         (ghost_nodes_enth - \
    #         (self.config.thermal.c_soil_unfrozen - self.config.thermal.c_soil_frozen) / self.config.thermal.rho_soil_frozen * self.config.thermal.T_melt - \
    #         self.config.thermal.L_water_ice / self.config.thermal.rho_ice * self.config.thermal.nb) \
    #             / (self.config.thermal.c_soil_unfrozen / self.config.thermal.rho_soil_unfrozen)
    
    
    # Old code from solar flux calculator:
    # def solar_flux_calculator(self, timestep_id, I0, timezone_diff):
    #     """
    #     This function calculates the effective solar radiation flux on a sloped surface. The method from Buffo (1972) is used, 
    #     assuming that the radiaton on the surface already includes the atmospheric transmission coefficient. Using the radiation data for a flat surface 
    #     and the angle of the incoming rays with the flat sruface, the intensity of the incoming rays can be estimated, which can then be projected on an inclined
    #     surface.

    #     Args:
    #         timestep_id (int): index of the current timestep.
    #         I0 (float): incoming radiation for the current timestep on a flat surface
    #         timezone_diff (float): difference in hours for the timezone which is modelled relative to UTC.

    #     Returns:
    #         array: incoming radiation for sloped surfaces for the computational domain for the current timestep.
    #     """       
    #     # 1) current timestamp
    #     current_timestamp = self.timestamps[timestep_id]
        
    #     # 2) latitude and orientation
    #     phi = self.config.model.latitude / 360 * 2 * np.pi
    #     beta = (90 - self.model.grid_orientation) / 360 * 2 * np.pi
        
    #     # 3) local angles
    #     alpha = -self.angles 
        
    #     # 4) declination, Sarbu (2017)
    #     delta = 23.45 * np.sin(
    #         (360/365 * (284 + current_timestamp.dayofyear)) / 360 * 2 * np.pi
    #         )
        
    #     # 5) hour angle, for Alaska timezone difference w.r.t. UTC is -8h
    #     local_hour_of_day = current_timestamp.hour + timezone_diff
    #     # convert to hour angle
    #     h = (((local_hour_of_day - 12) % 24)/24) * 2 * np.pi
    #     # convert angles to range [-pi, pi]
    #     mask = np.nonzero(h>=np.pi)
    #     h[mask] = -((2 * np.pi) - h[mask])
        
    #     # 6) calculate altitude angle off of the horizontal that the suns rays strike a horizontal surface
    #     A = np.arcsin(np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(h))
        
    #     # 7) calculate the azimuth
    #     AZ = np.cos(delta) * (np.sin(h)) / np.cos(A)
        
    #     # correct azimuth for when close to solstices (“Central Beaufort Sea Wave and Hydrodynamic Modeling Study Report 1: Field Measurements and Model Development,” n.d.)
    #     ew_AM_mask = np.nonzero((np.cos(h) > np.tan(delta) / np.tan(phi)) + (local_hour_of_day <= 12))# east-west AM mask
    #     ew_PM_mask = np.nonzero((np.cos(h) > np.tan(delta) / np.tan(phi)) + (local_hour_of_day > 12))# east-west PM mask

    #     AZ[ew_AM_mask] = -np.pi + np.abs(AZ[ew_AM_mask])
    #     AZ[ew_PM_mask] = np.pi - np.abs(AZ[ew_PM_mask])
        
    #     # convert to azimuth measured clockwise from the east
    #     Z = np.arcsin(np.cos(delta) * np.sin(h) / np.cos(A)) + 1/2 * np.pi

    #     # 8) calculate multiplication factor for computational domain
    #     sin_theta = np.sin(A) * np.cos(alpha) - np.cos(A) * np.sin(alpha) * np.sin(Z - beta)
    #     # # filter out values larger than 1 (this is only relevant when theta is actually calculated with the arcsin())
    #     # sin_theta[np.nonzero(sin_theta>1)] = 1
    #     # # filter out values smaller than 0 (these do not reach the surface)
    #     # sin_theta[np.nonzero(sin_theta<0)] = 0
        
    #     # 9) calculate multiplication factor for flat surface
    #     sin_0 = np.sin(A) * np.cos(0) - np.cos(A) * np.sin(0) * np.sin(Z - beta)
    #     # # filter out values larger than 1 (this is only relevant when theta is actually calculated with the arcsin())
    #     # sin_0[np.nonzero(sin_0>1)] = 1
    #     # # filter out values smaller than 0 (these do not reach the surface)
    #     # sin_0[np.nonzero(sin_0<0)] = 0
        
    #     # 10) in order to avoid very peaky scales, let us take the daily maximum and use that for scaling.
    #     sin_theta_2d = sin_theta[:-1].reshape((-1, 24))
    #     sin_theta_daily_max_2d = np.max(sin_theta_2d, axis=1)
    #     sin_theta_daily_max = np.repeat(sin_theta_daily_max_2d.flatten(), 24)

    #     sin_0_2d = sin_0[:-1].reshape((-1, 24))
    #     sin_0_daily_max_2d = np.max(sin_0_2d, axis=1)
    #     sin_0_daily_max = np.repeat(sin_0_daily_max_2d.flatten(), 24)
        
    #     # 10) compute corrected values for incoming solar radiation
    #     I_theta = sin_theta / sin_0 * I0
        
    #     return I_theta
    
        # # slope aspect clockwise from the north
        # beta = (90 - self.config.model.grid_orientation) / 360 * 2 * np.pi
        
        # A = np.arcsin(np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.cos(h))
        
        # # calculate azimuth
        # AZ = np.arcsin(-np.cos(delta) * np.sin(h) / np.cos(A))
        
        # # need to correct for when close to solstices
        # if np.cos(h) <= np.tan(delta) / np.tan(phi):
        #     if local_hour_of_day <= 12:
        #         AZ = -np.pi + np.abs(AZ)
        #     else:
        #         AZ = np.pi - AZ
        
        # # calculate Z
        # Z = AZ + 1/2 * np.pi
        
        # # calculate angle between the surface and the radiation
        # theta = np.arcsin(np.sin(A) * np.cos(alpha) - np.cos(A) * np.sin(alpha) * np.sin(Z - beta))
        
        # # calculate angle-corrected radiation
        # I = I0_p * np.sin(theta)
                
        # return I
        
    
    # Old code to start xbeach:
            
        # os.system('start "" "' + str(os.path.join(params_path, batch_fname)) + '"')
        
        # First a batch file is generated to be executed
        # xb_run_script_win(
        #     xb=self.xb_setup,
        #     N=1,
        #     maindir=self.cwd,
        #     xbeach_exe=xbeach_path
        #     )
        
        # command = ['"' + str(os.path.join(self.cwd, batch_fname)) + '"']

        # # get the return code 
        # stdout, stderr = process.communicate()
        # return_code = process.returncode
        
        # error_message = stderr.decode()
        # print(error_message)
    
    
    # Old storm timing function
    # def _when_storms_projection(self, fp_storm):
        
    #     # determine when storms occur (using raw_datasets/erikson/Hindcast_1981_2/BTI_WavesAndStormSurges_1981-2100.csv)
    #     st = np.zeros(self.T.shape)  # array of the same shape as t (0 when no storm, 1 when storm)
        
    #     self.conditions = np.zeros(self.T.shape, dtype=object)  # also directly read wave conditions here
        
    #     with open(fp_storm) as f:
            
    #         df = pd.read_csv(f)
            
    #         mask = (df.time >= self.t_start) * (df.time <= self.t_end)
            
    #         df = df[mask]
            
    #     mask = 
        
    #     for storm_time in df.time.values:
            
    #         index = np.argwhere(self.timestamps==storm_time)
            
    #         st[index] = 1
            
    #         self.conditions[t] = {
    #                    "Hso(m)": data["Hso(m)"],
    #                    "Hs(m)": data["Hs(m)"],
    #                    "Dp(deg)": data["Dp(deg)"],
    #                    "Tp(s)": data["Tp(s)"],
    #                    "SS(m)": data["SS(m)"],
    #                    "Hindcast_or_projection": data["Hindcast_or_projection"]
    #                     }  # safe storm conditions for this timestep as well
            
    #         for index, data in df.iterrows():
                
    #             duration = int(data["Storm_duration(days)"] * 24)  # in hours
                
    #             day = 0
                
    #             for hour in range(duration+1):
                    
    #                 day = hour // 24
    #                 hour = hour % 24
                    
    #                 if data.start_date_of_storm_month in [1, 3, 5, 7, 8, 10, 12] and data.start_date_of_storm_day + day > 31:
    #                     month_length = 31
    #                     end_of_month_storm = 1
    #                 elif data.start_date_of_storm_month in [4, 6, 9, 11] and data.start_date_of_storm_day + day > 30:
    #                     month_length = 30
    #                     end_of_month_storm = 1
    #                 elif data.start_date_of_storm_month in [2]:
    #                     if data.start_date_of_storm_year in np.arange(1940, 2200, 4) and data.start_date_of_storm_day + day > 29:
    #                         month_length = 29
    #                         end_of_month_storm = 1
    #                     elif data.start_date_of_storm_day + day > 28:
    #                         month_length = 28
    #                         end_of_month_storm = 1
    #                 else:
    #                     month_length = 0
    #                     end_of_month_storm = 0
                    
    #                 print("day: ", day)
    #                 print("hour: ", hour)
                    
    #                 timestamp = datetime(
    #                     data.start_date_of_storm_year, 
    #                     data.start_date_of_storm_month + end_of_month_storm, 
    #                     data.start_date_of_storm_day + day - month_length, 
    #                     hour,  # assume storms always start at 00:00:00 during the day
    #                     0, 
    #                     0
    #                     )

    #                 t = np.argmax((timestamp == self.timestamps))  # we get the current timestep here
                    
    #                 st[t] += 1  #  make sure xbeach will be active for this timestep
                                        
                    
    #     return st
            
    # part of old thaw depth method:        
    #   # get the number of grid points for each 1D model where the temperature exceeds the melting point (counted from the top until the first un-thawed point), 
        # normalize with the total number of points, and multiply with the total grid length.
        # self.thaw_depth = count_nonzero_until_zero((self.temp_matrix > self.config.thermal.T_melt)) / self.config.thermal.grid_resolution * self.config.thermal.max_depth
    
    # Also part of old thaw depth method (but a second iteration):
    # # determine in which interval to look for thaw depth
        # dx_min = np.min(self.xgr[1:] - self.xgr[:-1])
        # # loop through x-coordinates, and find thaw depth for each coordinate
        # for i in range(len(self.xgr)):
        #     # selection of points to look at for current grid coordinate
        #     mask = (x_matrix.flatten() > self.xgr[i] - 0.5 * dx_min) * (x_matrix.flatten() < self.xgr[i] + 0.5 * dx_min)
        #     # look at masked temperature matrix, use that to determine which points are thawed, and use the point with the highest z coordinate to calculate
        #     # the thaw depth
        #     self.thaw_depth[i] = self.zgr - np.max(self.zgr.flatten()[mask][self.temp_matrix.flatten()[mask] < self.config.thermal.T_melt])
                
            
    # def run_simulation(self):
    #     pass       
    


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