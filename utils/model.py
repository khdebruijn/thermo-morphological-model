import os
from pathlib import Path
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
from xbTools.grid.creation import xgrid, ygrid
from xbTools.xbeachtools import XBeachModelSetup
from xbTools.general.executing_runs import xb_run_script_win
from xbTools.general.wave_functions import dispersion

from utils.visualization import block_print, enable_print
from utils.miscellaneous import interpolate_points, get_A_matrix, count_nonzero_until_zero, generate_perpendicular_grids, linear_interp_with_nearest

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
        self._set_directory()
        self.read_config(config_file)
    
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
        
        os.chdir(self.cwd)    

        # setup output location
        if not os.path.exists(os.path.join(self.cwd, "results/")):
            os.mkdir(os.path.join(self.cwd, "results/"))
            
        self.result_dir = os.path.join(self.cwd, "results/")
    
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
                    
        with open(os.path.join(self.cwd, config_file)) as f:
            cfg = yaml.safe_load(f)
            
        self.config = AttrDict(cfg)
                
        for key in cfg:
            self.config[key] = AttrDict(cfg[key])
               
        return self.config
        
    def set_temporal_params(self, t_start, t_end, dt):
        """This method sets the temporal parameters used during the simulation."""
        
        # set start and end time, and time step
        self.dt = dt
        self.t_start = pd.to_datetime(t_start)
        self.t_end = pd.to_datetime(t_end)
        
        # this variable will be used to keep track of time
        self.timestamps = pd.date_range(start=self.t_start, end=self.t_end, freq=f'{self.dt}h', inclusive='left')
        
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
        if nx:
            self.xgr = np.linspace(min(self.bathy_grid), max(self.bathy_grid), self.config.model.nx)
        else:
            # transform into a more suitable grid for xbeach
            self.xgr, self.zgr = xgrid(self.bathy_grid, self.bathy_initial, dxmin=2)
        
        # interpolate bathymetry to grid
        self.zgr = np.interp(self.xgr, self.bathy_grid, self.bathy_initial)
        
        # save a copy of the grid, which serves as the bathymetry
        self.bathy_current = np.copy(self.zgr)
        
        # also initialize the current active layer depth here (denoted by "ne_layer", or non-erodible layer)
        self.thaw_depth = np.zeros(self.xgr.shape)
        
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
            
            
    def load_forcing(self, fname_in_ts_datasets="era5.csv"):
        """This function loads in the forcing data and makes it an attribute of the simulation instance"""
        # read in forcing concditions
        fpath = os.path.join(self.proj_dir, "database/ts_datasets/")
        self.forcing_data = self._get_timeseries(self.t_start, self.t_end, os.path.join(fpath, fname_in_ts_datasets))
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
        
        self.current_storm_surge = 0
        
        return None
    
    def xbeach_setup(self, timestep_id):
        """This function initializes an xbeach run, i.e., it writes all inputs to files
        """
        self.xb_setup = XBeachModelSetup(f"Run {self.cwd}: timestep {timestep_id}")
        
        self.xb_setup.set_grid(self.xgr, None, self.zgr, posdwn=-1)
        self.xb_setup.set_waves('parametric', {
            # need to give each parameter as series (in this case, with length 1)
            "Hm0":self.conditions[timestep_id]["Hs(m)"],  # file contains 'Hso(m)' (offshore wave height, in deep water) and 'Hs(m)' (nearhsore wave height, at 10m isobath)
            "Tp":self.conditions[timestep_id]["Tp(s)"],
            "mainang":self.conditions[timestep_id]["Dp(deg)"],  # relative to true north
            "gammajsp": 1.3,  # placeholder
            "s": 10,     # placeholder
            "duration": self.dt,
            "dtbc": 60, # placeholder
            "fnyq":1, # placeholder
        })
        
        wind_direction, wind_velocity = self._get_wind_conditions(timestep_id)
        self.current_storm_surge = self.conditions[timestep_id]["SS(m)"]  # used for output
        
        # (including: grid/bathymetry, waves input, flow, tide and surge,
        # water level, wind input, sediment input, avalanching, vegetation, 
        # drifters ipnut, output selection)
        self.xb_setup.set_params({
            # sediment parameters
            "D50": self.config.xbeach.D50,
            "rhos": self.config.xbeach.rho_solid,
            "reposeangle": self.config.xbeach.reposeangle,
            "dryslp": self.config.xbeach.dryslp,
            "wetslp": self.config.xbeach.wetslp,
            
            # flow boundary condition parameters
            "left": "neumann",
            "right": "neumann",
            
            #flow parameters
            # -----

            # general
            "befriccoef":self.config.xbeach.bedfriccoef,  # placeholder
            
            # grid parameters
            # most already specified with xb_setup.set_grid(...)
            "alfa": self.config.bathymetry.grid_orientation,  # counter-clockwise from the east
            "thetamin": -90,
            "thetamax": 90,
            "dtheta": 15,
            "thetanaut": 0,
            
            # model time
            "tstop":self.dt,
            
            # morphology parameters
            "morfac": 1,
            "morstart": 0,
            "ne_layer": "ne_layer.txt",
            
            # physical constant
            "rho": self.config.xbeach.rho_sea_water,
            "vicmol": self.config.xbeach.visc_kin,
            
            # physical processes
            "avalanching": 1,  # Turn on avalanching
            "flow": 1,  # Turn on flow calculation
            "lwave": 1,  # Turn on short wave forcing on nlsw equations and boundary conditions
            "morphology": 1,  # Turn on morphology
            "sedtrans": 1,  # Turn on sediment transport
            "swave": 1,  # Turn on short waves
            "swrunup": 1,  # Turn on short wave runup
            "viscosity": 1,  # Include viscosity in flow solver
            "wind": 1,  # Include wind in flow solver

            # tide boundary conditions
            "tideloc": 0,
            # "zs0file":
            "zs0":self.conditions[timestep_id]["SS(m)"],

            # wave boundary conditions
            "bcfile": "jonswap.txt",
            "wavemodel":"surfbeat",
            "break":"roelvink1",
            
            # wind boundary condition
            "windth": wind_direction,  # degrees clockwise from the north
            "windv": wind_velocity,
            
            # output variables
            "outputformat":"netcdf",
            "tint":3600,
            "tstart":0,
            "nglobalvar":[
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
        })
        
        # block printing while writing the output (the xbeach toolbox by default prints that it can't plot parametric conditions)
        block_print()
        
        # write model setup
        self.xb_setup.write_model(self.cwd)
        
        # close figures generated during writing
        plt.close()
        
        # re-enable print
        enable_print()
        
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
    
    # def _generate_batch_file(self):
    #     xb_run_script_win()  # not used yet (could develop in future)
    # def load_tide_conditions(self, fp_wave):
    #     pass
    # def load_wave_conditions(self, fp_wave):
    #     with open(fp_wave) as f:
    #         pass
        
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
    
    def timesteps_with_xbeach_active(self, fp_storm):
        """This function gets the timestep ids for which xbeach should be active.

        Args:
            fp_storm (Path): path to the csv file containing storm data.
            from_projection (bool, optional): choose whether to use the storm projection dataset (
                                              or something else, but that is not yet implemented). Defaults to True.

        Returns:
            array: array of length T that for each timestep contains a 1 if xbeach should be ran and 0 if not.
        """
        # get storm conditions timestep ids
        self.storm_timing = self._when_storms_projection(fp_storm)
        
        # get inter-storm timestep ids
        self.xbeach_inter = self._when_xbeach_inter(self.config.model.call_xbeach_inter)
        
        # get sea-ice timestep ids
        self.xbeach_sea_ice = self._when_xbeach_no_sea_ice(self.config.wrapper.sea_ice_threshold)

        # ran xbeach for storm and inter-storm timesteps, but never when too much sea ice
        self.xbeach_times = self.storm_timing * self.xbeach_sea_ice + self.xbeach_inter
        
        # output these xbeach timestep ids
        self._check_and_write('timestep_xbeach_ids', self.xbeach_times, self.result_dir)

        return self.xbeach_times
    
    def _when_storms_projection(self, fp_storm):
        """This function is used to determine when storms occur (using raw_datasets/erikson/Hindcast_1981_2/BTI_WavesAndStormSurges_1981-2100.csv)

        Args:
            fp_storm (Path): path to storm dataset

        Returns:
            array: array of length T that for each timestep contains a 1 if xbeach should be ran and 0 if not.
        """
        st = np.zeros(self.T.shape)  # array of the same shape as t (0 when no storm, 1 when storm)
        
        self.conditions = np.zeros(self.T.shape, dtype=object)  # also directly read wave conditions here
        
        with open(fp_storm) as f:
            
            df = pd.read_csv(f, parse_dates=['time'])
                        
            mask = (df['time'] >= self.t_start) * (df['time'] <= self.t_end)
            
            df = df[mask]
                    
        for i, row in df.iterrows():
            
            index = np.argwhere(self.timestamps==row.time)
            
            st[index] = 1
            
            # safe storm conditions for this timestep as well
            self.conditions[index] = {
                       "Hso(m)": row["Hso(m)"],
                       "Hs(m)": row["Hs(m)"],
                       "Dp(deg)": row["Dp(deg)"],
                       "Tp(s)": row["Tp(s)"],
                       "SS(m)": row["SS(m)"],
                       "Hindcast_or_projection": row["Hindcast_or_projection"]
                        }
        
        return st
                
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
        
        # for these intervals, if not conditions are supplied from the storm projections, set conditions to '0'
        zero_conditions = {
                    "Hso(m)": 0.001,
                    "Hs(m)": 0.001,
                    # "Hso(m)": 0,
                    # "Hs(m)": 0,
                    "Dp(deg)": 270,                    
                    # "Dp(deg)": 0,
                    "Tp(s)": 10,
                    # "Tp(s)": 0,
                    "SS(m)": 0,
                    "Hindcast_or_projection": 0,
                    }
        
        mask = np.nonzero((ct==1) * (self.conditions==0))

        self.conditions[mask] = zero_conditions
        
        return ct
    
    def _when_xbeach_no_sea_ice(self, sea_ice_threshold):
        """This function determines when xbeach should not be ran due to sea ice, based on a threshold value)

        Args:
            sea_ice_threshold (_type_): _description_

        Returns:
            array: array of length T that for each timestep contains a 1 if xbeach can be ran and 0 if not (w.r.t. sea ice).
        """
        it =  (self.forcing_data.sea_ice_cover.values < sea_ice_threshold)
                
        return it
        
    ################################################
    ##                                            ##
    ##            # THERMAL FUNCTIONS             ##
    ##                                            ##
    ################################################
    def initialize_thermal_module(self):
        """This function initializes the thermal module of the model.

        Raises:
            ValueError: raised if CFL > 0.5
        """
        
        # read initial conditions
        ground_temp_distr_dry, ground_temp_distr_wet = self._generate_initial_ground_temperature_distribution(self.forcing_data, 
                                                                                                             self.t_start, 
                                                                                                             self.config.thermal.grid_resolution,
                                                                                                             self.config.thermal.max_depth)
        # save the grid resolution
        self.dz = self.config.thermal.max_depth / (self.config.thermal.grid_resolution - 1) 
        
        self.thermal_zgr = ground_temp_distr_dry[:,0]
        
        # initialize temperature matrix, which is used to keep track of temperatures through the grid
        self.temp_matrix = np.zeros((len(self.xgr), self.config.thermal.grid_resolution))
        
        # initialize the associated grid
        self.abs_xgr, self.abs_zgr = generate_perpendicular_grids(self.xgr, self.zgr)
        
        # set the above determined initial conditions for the xgr
        for i in range(len(self.temp_matrix)):
            if self.zgr[i] >= self.config.thermal.wl_switch:  # assume that the initial water level is at zero
                self.temp_matrix[i,:] = ground_temp_distr_dry[:,1]
            else:
                self.temp_matrix[i,:] = ground_temp_distr_wet[:,1]
        
        # set initial ghost node temperature as a copy of the surface node of the temperature matrix
        self.ghost_nodes_temperature = self.temp_matrix[:,0]
        
        # find and write the initial thaw depth
        self.find_thaw_depth()
        self.write_ne_layer()
        
        # with the temperature matrix, the initial state (frozen/unfrozen can be determined)
        frozen_mask = (self.temp_matrix < self.config.thermal.T_melt)
        unfrozen_mask = np.ones(frozen_mask.shape) - frozen_mask
                    
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
        
        # initialize distribution of ground ice content, uniform for now (placeholder)
        self.nb_distr = np.ones(self.thermal_zgr.shape) * self.config.thermal.nb
        
        # using the states, the initial enthalpy can be determined. The enthalpy matrix is used as the 'preserved' quantity, and is used to numerically solve the
        # heat balance equation. Enthalpy formulation from Ravens et al. (2023).
        self.enthalpy_matrix = \
            frozen_mask * \
                self.config.thermal.c_soil_frozen * self.temp_matrix + \
            unfrozen_mask * \
                (self.config.thermal.c_soil_unfrozen * self.temp_matrix + \
                (self.config.thermal.c_soil_unfrozen - self.config.thermal.c_soil_frozen) * self.config.thermal.T_melt + \
                self.config.thermal.L_water_ice * self.config.thermal.nb)

        # initialize c-matrix
        self.c_matrix = frozen_mask * self.config.thermal.c_soil_frozen + unfrozen_mask * self.config.thermal.c_soil_unfrozen
        
        # initialize k-matrix
        self.k_matrix = frozen_mask * np.tile(self.k_frozen_distr, (len(self.xgr), 1)) + unfrozen_mask * np.tile(self.k_unfrozen_distr, (len(self.xgr), 1))
        
        # calculate the courant-friedlichs-lewy number matrix
        self.cfl_matrix = self.k_matrix / self.c_matrix * self.config.thermal.dt / self.dz**2
        
        # check that all cfl's are <0.5, which is required for this discretization of the 1D heat equation
        # self.cfl_frozen = (self.config.thermal.k_soil_frozen / self.rho_frozen / self.config.thermal.c_soil_frozen) * (self.config.thermal.dt / (self.dz)**2)
        # self.cfl_unfrozen = (self.config.thermal.k_soil_unfrozen / self.rho_unfrozen / self.config.thermal.c_soil_unfrozen) * (self.config.thermal.dt / (self.dz)**2)
        
        if np.max(self.cfl_matrix >= 0.5):
            raise ValueError(f"CFL should be smaller than 0.5, currently {np.max(self.cfl_matrix):.4f}")
        
        # get the 'A' matrix, which is used to make the numerical scheme faster. It is based on second order central differences for internal points
        # at the border points, the grid is extended with an identical point (i.e. mirrored), in order to calculate the second derivative
        self.A_matrix = get_A_matrix(self.config.thermal.grid_resolution)
        
        # initialize angles
        self._update_angles()
        
        # initialize output
        self.factors = np.zeros(self.xgr.shape)
        self.sw_flux = np.zeros(self.xgr.shape)
        self.lw_flux = np.zeros(self.xgr.shape)
        self.latent_flux = np.zeros(self.xgr.shape)
        self.convective_flux = np.zeros(self.xgr.shape)
        self.heat_flux = np.zeros(self.xgr.shape)

        return None
    
    def print_and_return_A_matrix(self):
        """This function prints and returns the A_matrix"""
        print(self.A_matrix)
        return self.A_matrix
    
    @classmethod
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
        dry_points = np.array([
            [0, df.soil_temperature_level_1.values[0]],
            [(0.07+0)/2, df.soil_temperature_level_1.values[0]],
            [(0.28+0.07)/2, df.soil_temperature_level_2.values[0]],
            [(1+0.28)/2, df.soil_temperature_level_3.values[0]],
            [(2.89+1)/2, df.soil_temperature_level_4.values[0]],
            [(2.29+max_depth)/2, df.soil_temperature_level_4.values[0]],
        ])
        
        ground_temp_distr_dry = interpolate_points(dry_points[0,:], dry_points[1,:], n)
        
        wet_points = np.array([
            [0, df.soil_temperature_level_1_offs.values[0]],
            [(0.07+0)/2, df.soil_temperature_level_1_offs.values[0]],
            [(0.28+0.07)/2, df.soil_temperature_level_2_offs.values[0]],
            [(1+0.28)/2, df.soil_temperature_level_3_offs.values[0]],
            [(2.89+1)/2, df.soil_temperature_level_4_offs.values[0]],
            [(2.29+max_depth)/2, df.soil_temperature_level_4_offs.values[0]],
        ])
        
        ground_temp_distr_wet = interpolate_points(wet_points[0,:], wet_points[1,:], n)
        
        return ground_temp_distr_dry, ground_temp_distr_wet
        
    def thermal_update(self, timestep_id, subgrid_timestep_id):
        """This function is called each subgrid timestep of each timestep, and performs the thermal update of the model.

        Args:
            timestep_id (int): id of the current timestep
            subgrid_timestep_id (int): id of the current subgrid timestep
        """
        # get the new boundary condition
        self.ghost_nodes_temperature = self._get_ghost_node_boundary_condition(timestep_id)
        self.bottom_boundary_temperature = self._get_bottom_boundary_temperature()
        
        # aggregate temperature matrix
        aggregated_temp_matrix = np.concatenate((
            self.ghost_nodes_temperature.reshape(len(self.xgr), 1),
            self.temp_matrix,
            self.bottom_boundary_temperature.reshape(len(self.xgr), 1)
            ), axis=1)
        
        # determine which part of the domain is frozen and unfrozen (for the aggregated temperature matrix)
        frozen_mask_aggr = (aggregated_temp_matrix < self.config.thermal.T_melt)
        unfrozen_mask_aggr = np.ones(frozen_mask_aggr.shape) - frozen_mask_aggr
        
        # determine c-matrix
        self.c_matrix = frozen_mask_aggr * self.config.thermal.c_soil_frozen + \
                        unfrozen_mask_aggr * self.config.thermal.c_soil_unfrozen
        
        # determine k-matrix (extend k-distribution with one copied value at the top and bottom boundary to include ghost nodes)
        k_frozen_distr_aggr = np.concatenate((
            np.array([self.k_frozen_distr[0]]),
            self.k_frozen_distr,
            np.array([self.k_frozen_distr[-1]])))
        k_unfrozen_distr_aggr = np.concatenate((
            np.array([self.k_unfrozen_distr[0]]),
            self.k_frozen_distr,
            np.array([self.k_unfrozen_distr[-1]])))
        
        self.k_matrix = frozen_mask_aggr * np.tile(k_frozen_distr_aggr, (len(self.xgr), 1)) + \
                        unfrozen_mask_aggr * np.tile(k_unfrozen_distr_aggr, (len(self.xgr), 1))
        
        # determine the courant-friedlichs-lewy number matrix
        self.cfl_matrix = self.k_matrix / self.c_matrix * self.config.thermal.dt / self.dz**2
        
        if np.max(self.cfl_matrix >= 0.5):
            raise ValueError(f"CFL should be smaller than 0.5, currently {np.max(self.cfl_matrix):.4f}")
        
        # get the new enthalpy matrix
        self.enthalpy_matrix = self.enthalpy_matrix + \
                               (self.cfl_matrix * self.c_matrix * aggregated_temp_matrix) @ self.A_matrix
        
        # determine state masks (which part of the domain is frozen, in between, or unfrozen (needed to later calculate temperature from enthalpy))
        frozen_mask = (self.enthalpy_matrix / self.config.thermal.c_soil_frozen < self.config.thermal.T_melt)
        inbetween_mask = ((self.enthalpy_matrix / self.config.thermal.c_soil_frozen) >= self.config.thermal.T_melt) * \
                         ((self.enthalpy_matrix - \
                             (self.config.thermal.c_soil_frozen - self.config.thermal.c_soil_unfrozen) * self.config.thermal.T_melt - \
                              self.config.thermal.L_water_ice * self.config.thermal.nb) < self.config.thermal.T_melt)
                          
        unfrozen_mask = np.ones(frozen_mask.shape) - frozen_mask - inbetween_mask
        
        # from this new enthalpy, the temperature distribution can be determined, depending on the state from the PREVIOUS timestep
        # again, the state masks are used to make this calculation faster
        self.temp_matrix = \
            frozen_mask * \
                (self.enthalpy_matrix / (self.config.thermal.c_soil_frozen)) + \
            inbetween_mask * \
                (self.config.thermal.T_melt) * \
            unfrozen_mask * \
                (self.enthalpy_matrix - \
                (self.config.thermal.c_soil_unfrozen - self.config.thermal.c_soil_frozen) * self.config.thermal.T_melt - \
                self.config.thermal.L_water_ice * self.config.thermal.nb) / \
                    (self.config.thermal.c_soil_unfrozen)
        
        return None
     
    def _get_ghost_node_boundary_condition(self, timestep_id):
        """This function uses the forcing at a specific timestep to return an array containing the ghost node temperature.

        Args:
            timestep_id (int): index of the current timestep.

        Returns:
            array: temperature values for the ghost nodes.
        """
        # first get the correct forcing timestep
        row = self.forcing_data.iloc[timestep_id]
        
        # with associated forcing values
        self.current_air_temp = row["2m_temperature"]  # also used in output
        self.current_sea_temp = row['sea_surface_temperature']  # also used in output
        self.current_sea_ice = row["sea_ice_cover"]  # not used in this function, but loaded in preperation for output
        
        # check wether or not this will be a storm timestep.
        if self.xbeach_times[timestep_id]:  # If it is, use (storm surge water level + runup)
            
            # get storm surge
            surge = self.conditions[timestep_id]["SS(m)"]
            
            # get runup
            ds = xr.load_dataset(os.path.join(self.cwd, "xboutput.nc"))
            runup = ds.runup.values.flatten()
            ds.close()
            
            # get water level to check whether to use convective heat transfer from air or sea
            water_level = surge + runup
            
        else:  # otherwise, use a water level of z=0
            water_level = 0
        
        dry_mask = (self.zgr >= water_level)
        wet_mask = (self.zgr < water_level)
        
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
            # determine hydraulic parameters for convective heat transfer computation
            if self.xb_times[timestep_id]:
                
                data_path = os.path.join(self.cwd, "xboutput.nc")
                
                ds = xr.load_dataset(data_path)
                
                H = ds.H.values.flatten()

                wl = self.config.thermal.wl_switch + ds.runup.values.flatten()
                
                d = np.maximum(wl - ds.zb.values.flatten(), np.zeros(H.shape))
                T = self.conditions[timestep_id]["Tp(s)"]
                
                ds.close()
                                
            else:
                H = np.ones(self.xgr.shape) * 0.001
                
                wl = self.config.thermal.wl_switch
                
                d = np.maximum(wl - self.zgr, 0)
                T = 10
                
            # determine convective transport from water (formulation from Kobayashi, 1999)
            hc = Simulation._calculate_sensible_heat_flux_water(
                H, T, d, self.config.xbeach.rho_sea_water,
                CW=3989, alpha=0.5, nu=1.848*10**-6, ks=2.5*1.90*10**-3, Pr=13.4
            )
        
        # scale hc with temperature difference
        convective_transport_water = hc * (self.current_sea_temp - self.temp_matrix[:,0])
            
        # compute total convective transport
        self.convective_flux = dry_mask * convective_transport_air + wet_mask * convective_transport_water  # also used in output
        
        # determine radiation, assuming radiation only influences the dry domain
        self.latent_flux = row["mean_surface_latent_heat_flux"] if self.config.thermal.with_latent else 0  # also used in output
        self.lw_flux = row["mean_surface_net_long_wave_radiation_flux"] if self.config.thermal.with_longwave else 0  # also used in output
        
        if self.config.thermal.with_solar:  # also used in output
            I0 = row["mean_surface_net_short_wave_radiation_flux"]  # float value
            
            if self.config.thermal.with_solar_flux_calculator:
                self.sw_flux = self._get_solar_flux(I0, timestep_id)  # sw_flux is now an array instead of a float
            else:
                self.sw_flux = np.ones(self.xgr.shape) * I0
        else:
            self.sw_flux = np.zeros(self.xgr.shape)
        
        # add all heat fluxes  together (also used in output)
        self.heat_flux = self.convective_flux + \
                         self.latent_flux + \
                         self.lw_flux + \
                         self.sw_flux
        
        # determine temperature of the ghost nodes
        ghost_nodes_temperature = self.temp_matrix[:,0] + self.heat_flux * self.dz / self.k_matrix[:,0]
        
        return ghost_nodes_temperature
    
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

    
    def update_grid(self, fp_xbeach_output="sedero.txt"):
        """This function updates the current grid, calculates the angles of the new grid with the horizontal, generates a new thermal grid 
        (perpendicular to the existing grid), and fits the previous temperature and enthalpy distributions to the new grid."""
        
        # generate perpendicular grids for previous timestep (to cast temperature and enthalpy)
        self.abs_xgr, self.abs_zgr = generate_perpendicular_grids(self.xgr, self.zgr)
        
        # update the current bathymetry
        cum_sedero = self._update_bed_sedero(fp_xbeach_output=fp_xbeach_output)  # placeholder
        
        # only update the grid of there actually was a change in bed level
        if not all(cum_sedero) == 0:
        
            # generate a new xgrid and zgrid
            self.xgr_new, self.zgr_new = xgrid(self.xgr, self.bathy_current, dxmin=2)
            self.zgr_new = np.interp(self.xgr_new, self.xgr, self.bathy_current)
            
            # generate perpendicular grids for next timestep (to cast temperature and enthalpy)
            self.abs_xgr_new, self.abs_zgr_new = generate_perpendicular_grids(self.xgr_new, self.zgr_new)
            
            # cast temperature matrix
            self.temp_matrix = linear_interp_with_nearest(self.abs_xgr, self.abs_zgr, self.temp_matrix, self.abs_xgr_new, self.abs_zgr_new)
            self.enthalpy_matrix = linear_interp_with_nearest(self.abs_xgr, self.abs_zgr, self.enthalpy_matrix, self.abs_xgr_new, self.abs_zgr_new)

            # set the grid to be equal to this new grid
            self.xgr = self.xgr_new
            self.zgr = self.zgr_new
            
            # update the angles
            self._update_angles()
        
    def _update_angles(self):
        """This function geneartes an array of local angles (in radians) for the grid, based on the central differences method.
        """
        self.angles = np.gradient(self.zgr, self.xgr)
        
        return self.angles
    
    def _update_bed_sedero(self, fp_xbeach_output):
        """This method updates the current bed given the xbeach output.
        ---------
        fp_xbeach_output: string
            filepath to the xbeach sedero (sedimentation-erosion) output relative to the current working directory."""
            
        # Read output file
        ds = xr.open_dataset(fp_xbeach_output).squeeze()
        cum_sedero = ds.sedero.values
        xgr = ds.globalx.values
        
        # Create an interpolation function
        interpolation_function = interp1d(xgr, cum_sedero, kind='linear', fill_value='extrapolate')
        
        # interpolate values to the used grid
        interpolated_cum_sedero = interpolation_function(self.xgr)
        
        # update bed level
        self.bathy_current += interpolated_cum_sedero
        
        return cum_sedero
    
    def _get_solar_flux(self, I0, timestep_id):
        """This function is used to obtain an array of incoming solar radiation for some timestep_id, with values for each grid point in the computational domain.

        Args:
            I0 (float): incoming radiation (flat surface)
            timestep_id (int): index of the current timestep

        Returns:
            array: incoming solar radiation for each grid point in the computational domain
        """        
        # get current timestamp
        timestamp = self.timestamps.iloc[timestep_id]
        
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
                
        self.solar_flux_map = np.zeros(len(self.solar_flux_times) / 24, len(self.solar_flux_angles))
        
        for angle in self.solar_flux_angles:
            angle_id = np.nonzero(angle==self.solar_flux_angles)
            # for each integer angle in the angle range, an array of enhancement factors is saved, indexable by N (i.e., the N-th day of the year)
            self.solar_flux_map[:, angle_id] = self._calculate_solar_flux_factors(self.solar_flux_times, angle, timezone_diff)
        
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
        # sin_theta_daily_max_repeated = np.repeat(sin_theta_daily_max, 24)

        sin_0_2d = sin_0.reshape((-1, 24))
        sin_0_daily_max = np.max(sin_0_2d, axis=1).flatten()
        # sin_0_daily_max_repeated = np.repeat(sin_0_daily_max, 24)

        # 10) calculate enhancement factor
        factor = sin_theta_daily_max / sin_0_daily_max
        
        # 11) filter out values where it the angle theta is negative (as that means radiation hits the surface from below)
        shadow_mask = np.nonzero(theta < 0)
        factor[shadow_mask] = 0
        
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
        x_matrix, z_matrix = generate_perpendicular_grids(
            self.xgr, self.zgr, 
            resolution=self.config.thermal.grid_resolution, 
            max_depth=self.config.thermal.max_depth)
        
        # determine indices of thaw depth in perpendicular model
        indices = count_nonzero_until_zero((self.temp_matrix > self.config.thermal.T_melt))
        
        # find associated coordinates of these points
        x_thaw = x_matrix[indices]
        z_thaw = z_matrix[indices]
        
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
                
                mask2 = np.nonzero((x_thaw_sorted > x))
                x2 = x_thaw_sorted[mask2][0]
                z2 = z_thaw_sorted[mask2][0]
                
                z_thaw_interpolated = z1 + (z2 - z1)/(x2 - x1) * (x - x1)
                
                self.thaw_depth[i] = z - (z_thaw_interpolated)
            except:
                self.thaw_depth[i] = 0
        
        return self.thaw_depth
        
        
    def write_output(self, timestep_id):
        """This function writes output in the results folder, and creates subfolders for each timestep for which results are output.
        """
        result_dir_timestep = os.path.join(self.result_dir, str(timestep_id) + "/")
        
        # create directory
        if not os.path.exists(result_dir_timestep):
            os.makedirs(result_dir_timestep)
            
        # bathymetric variables
        self._check_and_write('xgr', self.xgr, dirname=result_dir_timestep)  # 1D series of x-values
        self._check_and_write('zgr', self.zgr, dirname=result_dir_timestep)  # 1D series of z-values
        self._check_and_write('angles', self.angles, dirname=result_dir_timestep)  # 1D series of angles (in radians)
        
        # hydrodynamic variables (note: obtained from previous xbeach timestep, so not necessarily accurate with other output data)
        xb_output_path = os.path.join(self.cwd, "xboutput.nc")
        if os.path.isfile(xb_output_path):  # check if an xbeach output file exists (it shouldn't at the first timestep)
            ds = xr.load_dataset(os.path.join(self.cwd, "xboutput.nc"))  # get xbeach data
            self._check_and_write('wave_height', ds.H.values.flatten(), dirname=result_dir_timestep)  # 1D series of wave heights (associated with xgr.txt)
            self._check_and_write('run_up', np.ones(1) * (ds.runup.values.flatten()), dirname=result_dir_timestep)  # single value
            self._check_and_write('storm_surge', np.ones(1) * (self.current_storm_surge), dirname=result_dir_timestep)  # single value
            self._check_and_write('wave_energy', ds.E.values.flatten(), dirname=result_dir_timestep)  # 1D series of wave energies (associated with xgr.txt)
            self._check_and_write('radiation_stress_xx', ds.Sxx.values.flatten(), dirname=result_dir_timestep)  # 1D series of radiation stresses (associated with xgr.txt)
            self._check_and_write('radiation_stress_xy', ds.Sxy.values.flatten(), dirname=result_dir_timestep)  # 1D series of radiation stresses (associated with xgr.txt)
            self._check_and_write('radiation_stress_yy', ds.Syy.values.flatten(), dirname=result_dir_timestep)  # 1D series of radiation stresses (associated with xgr.txt)
            self._check_and_write('mean_wave_angle', ds.thetamean.values.flatten(), dirname=result_dir_timestep)  # 1D series of mean wave angles in radians (associated with xgr.txt)
            self._check_and_write('velocity_magnitude', ds.vmag.values.flatten(), dirname=result_dir_timestep)  # 1D series of velocities (associated with xgr.txt)
            self._check_and_write('orbital_velocity', ds.urms.values.flatten(), dirname=result_dir_timestep)  # 1D series of velocities (associated with xgr.txt)
            ds.close()
        else:        
            self._check_and_write('wave_height', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of wave heights (associated with xgr.txt)
            self._check_and_write('run_up', np.ones(1) * (0), dirname=result_dir_timestep)  # single value
            self._check_and_write('storm_surge', np.ones(1) * (0), dirname=result_dir_timestep)  # single value
            self._check_and_write('wave_energy', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of wave energies (associated with xgr.txt)
            self._check_and_write('radiation_stress_xx', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of radiation stresses (associated with xgr.txt)
            self._check_and_write('radiation_stress_xy', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of radiation stresses (associated with xgr.txt)
            self._check_and_write('radiation_stress_yy', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of radiation stresses (associated with xgr.txt)
            self._check_and_write('mean_wave_angle', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of mean wave angles in radians (associated with xgr.txt)
            self._check_and_write('velocity_magnitude', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of velocities (associated with xgr.txt)
            self._check_and_write('orbital_velocity', np.zeros(self.xgr.shape), dirname=result_dir_timestep)  # 1D series of velocities (associated with xgr.txt)
        
        # temperature variables
        self._check_and_write('thaw_depth', self.thaw_depth, dirname=result_dir_timestep)  # 1D series of thaw depths
        self._check_and_write('abs_xgr', self.abs_xgr.flatten(), dirname=result_dir_timestep)  # 1D series of x-values (corresponding to ground_temperature_distribution.txt and grount_enthalpy_distribution.txt)
        self._check_and_write('abs_zgr', self.abs_zgr.flatten(), dirname=result_dir_timestep)  # 1D series of z-values (corresponding to ground_temperature_distribution.txt and grount_enthalpy_distribution.txt)
        self._check_and_write('grount_temperature_distribution', self.temp_matrix.flatten(), dirname=result_dir_timestep)  # 1D series of temperature values (associated with abs_xgr.txt and abs_zgr.txt)
        self._check_and_write('ground_enthalpy_distribution', self.enthalpy_matrix.flatten(), dirname=result_dir_timestep)  # 1D series of enthalpy values (associated with abs_xgr.txt and abs_zgr.txt)
        self._check_and_write("2m_temperature", np.ones(1) * (self.current_air_temp), dirname=result_dir_timestep)  # single value
        self._check_and_write("sea_surface_temperature",  np.ones(1) * (self.current_sea_temp), dirname=result_dir_timestep)  # single value
        
        # heat flux variables
        self._check_and_write('solar_radiation_factor', self.factors, dirname=result_dir_timestep)  # 1D series of factors
        self._check_and_write('solar_radiation_flux', self.sw_flux, dirname=result_dir_timestep)  # 1D series of heat fluxes
        self._check_and_write('long_wave_radiation_flux', self.lw_flux, dirname=result_dir_timestep)  # 1D series of heat fluxes
        self._check_and_write('latent_heat_flux', self.latent_flux, dirname=result_dir_timestep)  # 1D series of heat fluxes
        self._check_and_write('convective_heat_flux', self.convective_flux, dirname=result_dir_timestep)  # 1D series of heat fluxes
        self._check_and_write('total_heat_flux', self.heat_flux, dirname=result_dir_timestep)  # 1D series of heat fluxes
        
        # sea ice variables
        self._check_and_write('sea_ice_cover', np.ones(1) * (self.current_sea_ice), dirname=result_dir_timestep)  # single value
        
        # wind variables
        self._check_and_write('wind_velocity', np.ones(1) * (self.wind_velocity), dirname=result_dir_timestep)  # single value
        self._check_and_write('wind_direction', np.ones(1) * (self.wind_direction), dirname=result_dir_timestep) # single value (degrees, clockwise from the north)
        
        return None
    
    def _check_and_write(self, varname, save_var, dirname):
        
        if varname in self.config.output.output_vars:
            np.savetxt(os.path.join(dirname, f"{varname}" + ".txt"), save_var)
        
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

    def _get_timeseries(self, tstart, tend, fpath):
        """returns timeseries start from tstart and ending at tend. The filepath has to be specified.
        
        returns: pd.DataFrame of length T"""
        
        with open(fpath) as f:
            df = pd.read_csv(f, parse_dates=['time'])
                    
            mask = (df["time"] >= tstart) * (df["time"] < tend)
            
            df = df[mask]
                        
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