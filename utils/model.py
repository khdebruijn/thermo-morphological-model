import os
import yaml

from datetime import datetime
import math

import subprocess
import numpy as np
import pandas as pd

import xbTools
from xbTools.grid.creation import xgrid, ygrid
from xbTools.xbeachtools import XBeachModelSetup
from xbTools.general.executing_runs import xb_run_script_win

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
        if not os.path.exists(os.join(self.cwd, "results/")):
            os.mkdir(os.path.join(self.cwd, "results/"))
            
        self.result_dir = os.path.join(self.cwd, "results/")
        
        # create result files
        for output_var in self.config.output.output_vars:
            with open(os.path.join(self.result_dir, output_var, ".txt"), "w") as f:   
                f.write("PLACEHOLDER HEADER")
        return None
    
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
                    
        with open(os.join(self.cwd, config_file)) as f:
            cfg = yaml.safe_load(f)
            
        self.config = AttrDict(cfg)
                
        for key in cfg:
            self.config[key] = AttrDict(cfg[key])
                                
        return self.config
        
    def set_temporal_params(self, t_start, t_end, dt):
        """This method sets the temporal parameters used during the simulation."""
        # this variable will be used to keep track of time
        self.timestamps = pd.date_range(start=t_start, end=t_end, freq=f'{dt}h')
        
        # time indexing is easier for numerical models    
        self.T = np.arange(0, len(self.timestamps), 1) 
        
        # set start and end time, and time step
        self.dt = dt
        self.t_start = pd.to_datetime(t_start)
        self.t_end = pd.to_datetime(t_end)
        
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
        
        # Load initial bathymetry and the accompanying grid
        if bathy_path:
            self._load_bathy(
                os.path.join(self.proj_dir, bathy_path)
            )
        else:
            self._load_bathy(os.path.join(self.cwd, "bed.dep"))
            
        if bathy_grid_path:
            self._load_grid_bathy(
                os.path.join(self.proj_dir, bathy_grid_path)
            )
        else:
            self._load_grid_bathy(os.path.join(self.cwd, "x.grd"))
        
        # transform into a more suitable grid for xbeach
        self.xgr, self.zgr = xgrid(self.bathy_grid, self.bathy_initial, dxmin=2)
        self.zgr = np.interp(self.xgr, self.bathy_grid, self.bathy_initial)
        
        # save a copy of the grid, which serves as the bathymetry
        self.bathy_current = np.copy(self.zgr)
        # self.bathy_timeseries = [self.zgr]
        
        # also initialize the current active layer depth here (denoted by "ne_layer", or non-erodible layer)
        self.thaw_depth = np.zeros(self.xgr.shape)
        
        return self.xgr, self.zgr, self.ne_layer
    
    # def export_grid(self):
    #     np.savetxt(self.config.xbeach.xfile, self.xgr)
    #     np.savetxt(self.config.xbeach.yfile, self.zgr)

    
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
    def xbeach_setup(self, timestep_id):
        """This function initializes an xbeach run, i.e., it writes all inputs to files
        """
        xb_setup = XBeachModelSetup(f"Run {self.directory}: timestep {timestep_id}")
        
        xb_setup.set_grid(self.xgr, None, self.zgr, posdwn=-1)
        
        xb_setup.set_waves('jonstable', {
            "Hm0":self.conditions[timestep_id]["Hs(m)"],
            "Tp":self.conditions[timestep_id]["Tp(s)"],
            "mainang":self.conditions[timestep_id]["Dp(degree)"],  # relative to true north
            "gammajsp": 1.3,  # placeholder
            "s": 0.18,     # placeholder
            "duration": self.dt,
            "dtbc": 60, # placeholder
        })
        
        wind_direction, wind_velocity = self._get_wind_conditions(timestep_id)
        
        xb_setup.set_params({
            # bed composition parameters
            "D50": 0.000245,  # placeholder
            "D90": 0.000367,  # placeholder
            
            # flow boundary condition parameters
            "left": "neumann",
            "right": "neumann",
            
            #flow parameters
            # -----
            # general
            "befriccoef":0.01,  # placeholder
            
            # grid parameters
            # most already specified with xb_setup.set_grid(...)
            "alfa": self.config.model.grid_orientation,
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
            "reposeangle": self.config.xbeach.reposeangle,
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
            "wavemodel":"surfbeat",
            "break":"roelvink1",
            "alpha":"0",  # direction of x-axis relative to east (placeholder)
            
            # wind boundary condition
            "windh": wind_direction,
            "windv": wind_velocity,
            
            # output variables
            "outputformat":"netcdf",
            "tint":3600,
            "tstart":0,
            "nglobalvar":["zb","zs","H","runup","sedero"]
        })
        
        print(xb_setup)
        
        xb_setup.write_model(self.cwd)
        
        return None
    
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
    
    # def _generate_batch_file(self):
    #     xb_run_script_win()  # not used yet (could develop in future)
            
    
    # def export_bed(self):
    #     self.bed.to_csv("...")
    
    # def initialize_erodible_layer(self, fp_active_layer):
    #     pass
    
    def load_tide_conditions(self, fp_wave):
        pass
    
    
    # def load_wave_conditions(self, fp_wave):
        
    #     with open(fp_wave) as f:
    #         pass
        
    def _get_wind_conditions(self, timestep_id):
        
        row = self.forcing_data[self.forcing_data.index.values == timestep_id]
        
        u = row["10m_u_component_of_wind"]
        v = row["10m_v_component_of_wind"]
        
        direction = math.atan2(u, v) / (2*np.pi) * 360
        
        velocity = math.sqrt(u**2 + v**2)
        
        return direction, velocity
    
    def timesteps_with_xbeach_active(self, fp_storm, from_projection=True):
        
        if from_projection:
            self.storm_timing = self._when_storms_projection(fp_storm)
        else:
            pass # not implemented yet
        
        self.xbeach_inter = self._when_xbeach_inter(self.config.model.call_xbeach_inter)
        
        self.xbeach_sea_ice = self._when_xbeach_no_sea_ice(self.config.xbeach.sea_ice_threshold)
        
        self.xbeach_times = (self.storm_timing + self.xbeach_inter) * self.xbeach_sea_ice

        return self.xbeach_times
        
    def _when_storms_projection(self, fp_storm):
        
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
    
    def _when_xbeach_no_sea_ice(self, sea_ice_threshold):
        
        it =  (self.forcing_data.sea_ice_cover.values < sea_ice_threshold)
        
        return it
        
    ################################################
    ##                                            ##
    ##            # THERMAL FUNCTIONS             ##
    ##                                            ##
    ################################################
    def initialize_thermal_module(self):
        
        # read initial conditions
        ground_temp_distr_dry, ground_temp_distr_wet = self._generate_initial_ground_temperature_distribution(self.forcing_data, 
                                                                                                             self.t_start, 
                                                                                                             self.config.thermal.grid_resolution,
                                                                                                             self.config.thermal.max_depth)
        # save the grid resolution
        self.dz = self.config.thermal.max_depth / (self.config.thermal.grid_resolution - 1) 
        
        self.thermal_zgr = ground_temp_distr_dry[:,0]
        
        # initialize temperature matrix, which is used to keep track of temperatures through the grid
        self.temp_matrix = np.zeros((self.config.model.nx, self.config.thermal.grid_resolution))
        
        # set the above determined initial conditions for the xgr
        for i in range(len(self.temp_matrix)):
            if self.zgr[i] >= self.config.thermal.wl_switch:  # assume that the initial water level is at zero
                self.temp_matrix[i,:] = ground_temp_distr_dry[:,1]
            else:
                self.temp_matrix[i,:] = ground_temp_distr_wet[:,1]
        
        # with the temperature matrix, the initial state (frozen/unfrozen can be determined)
        frozen_mask = (self.temp_matrix < self.config.thermal.T_melt)
        unfrozen_mask = np.ones(frozen_mask.shape) - frozen_mask
        
        # using the states, the initial enthalpy can be determined. The enthalpy matrix is used as the 'preserved' quantity, and is used to numerically solve the
        # heat balance equation. Enthalpy formulation from Ravens et al. (2023).
        self.enthalpy_matrix = \
            frozen_mask * self.config.thermal.c_frozen_soil * self.temp_matrix + \
            unfrozen_mask * (self.config.thermal.c_unfrozen_soil * self.temp_matrix + \
                (self.config.thermal.c_unfrozen_soil - self.config.thermal.c_frozen_soil) * self.config.thermal.T_melt + \
                    self.config.thermal.L_water_ice * self.config.thermal.nb)

        # calculate the courant-friedlichs-lewy number and check that it is < 0.5, which is required for this discretization of the 1D heat equation
        self.cfl_frozen = self.config.thermal.c_frozen_soil * self.config.thermal.k_frozen_soil * self.config.thermal.dt / (self.config.thermal.max_depth / (self.config.thermal.grid_resolution - 1))**2
        self.cfl_unfrozen = self.config.thermal.c_unfrozen_soil * self.config.thermal.k_unfrozen_soil * self.config.thermal.dt / (self.config.thermal.max_depth / (self.config.thermal.grid_resolution - 1))**2
        
        if self.cfl_frozen >= 0.5:
            raise ValueError("CFL should be smaller than 0.5!")
        if self.cfl_unfrozen >= 0.5:
            raise ValueError("CFL should be smaller than 0.5!")
        
        # get the 'A' matrix, which is used to make the numerical scheme faster. It is based on second order central differences for internal points
        # at the border points, the grid is extended with an identical point (i.e. mirrored), in order to calculate the second derivative
        self.A_matrix = get_A_matrix(self.config.thermal.grid_resolution)
        
        # which timesteps should have thermal output (i.e., thaw depth, and temperature distribution)
        self.thermal_output_ids = self.T[::self.config.output.thermal_output_res]
    
    def print_and_return_A_matrix(self):
        """This function prints and returns the A_matrix"""
        print(self.A_matrix)
        return self.A_matrix
    
    @classmethod
    def _generate_initial_ground_temperature_distribution(df, t_start, n, max_depth):
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
        row = df[df.time == t_start]
        
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
    
    def thermal_update(self, timestep_id):
        
        # first get the new boundary condition
        ghost_nodes_temperature = self._get_ghost_node_boundary_condition(timestep_id)

        # determine which part of the domain is frozen and unfrozen (needed to later calculate temperature from enthalpy)
        frozen_mask = (self.temp_matrix < self.config.thermal.T_melt)
        unfrozen_mask = np.ones(frozen_mask.shape) - frozen_mask
        
        # CFL number is different for different phases
        cfl_matrix = frozen_mask * self.cfl_frozen + unfrozen_mask * self.cfl_unfrozen
        
        # get the new enthalpy matrix
        self.enthalpy_matrix = self.enthalpy_matrix + cfl_matrix * (np.column_stack((ghost_nodes_temperature, self.temp_matrix)) @ self.A_matrix)
        
        # from this new enthalpy, the temperature distribution can be determined, depending on the state from the PREVIOUS timestep
        # again, the state masks are used to make this calculation faster
        self.temp_matrix = \
            frozen_mask * (self.enthalpy_matrix / self.config.thermal.c_unfrozen_soil) + \
            unfrozen_mask * (self.enthalpy_matrix - \
                            (self.config.thermal.c_unfrozen_soil - self.config.thermal.c_frozen_soil) * self.config.thermal.T_melt - \
                             self.config.thermal.L_water_ice * self.config.thermal.nb) / self.config.thermal.c_unfrozen_soil
        
        return None
     
    def _get_ghost_node_boundary_condition(self, timestep_id):
        
        # first get the correct forcing timestep
        mask = (self.forcing_data.index.values == timestep_id)
        row = self.forcing_data[mask]
        
        # with associated values
        air_temp = row["2m_temperature"].values[0]
        sea_temp = row['sea_surface_temperature'].values[0]
        
        # get water level to check whether to use convective heat transfer from air or sea
        water_level = np.loadtxt("...")
        
        dry_mask = (self.zgr >= water_level)
        wet_mask = (self.zgr < water_level)
        
        # temperature difference for convective heat transfer (define temperature flux as positive when it is directed into the ground)
        temp_diff_at_interface = (air_temp - self.temp_matrix[:,0]) * dry_mask + (sea_temp - self.temp_matrix[:,0]) * wet_mask
        
        # apply boundary conditions
        frozen_mask = (self.temp_matrix[:,0] < self.config.thermal.T_melt)
        unfrozen_mask = np.ones(frozen_mask.shape) - frozen_mask

        cfl_matrix = frozen_mask * self.cfl_frozen + unfrozen_mask * self.cfl_unfrozen
        
        # calculate the new enthalpy
        # 1) calculate temperature diffusion
        ghost_nodes_enth = self.enthalpy_matrix[:,0] + cfl_matrix * (-self.temp_matrix[:,0] + self.enthalpy_matrix[:,1])

        # 2) add radiation, assuming radiation only influences the dry domain
        ghost_nodes_enth += self.config.thermal.dt * dry_mask * \
            (row["mean_surface_latent_heat_flux"] + row["mean_surface_net_short_wave_radiation_flux"] + row["mean_surface_net_long_wave_radiation_flux"])
        
        # 3) add convective heat transfer from water and air
        ghost_nodes_enth += \
            self.config.thermal.dt * \
                temp_diff_at_interface * \
                    (frozen_mask * self.thermal.k_frozen_soil + unfrozen_mask * self.thermal.k.unfrozen_soil) * \
                self.dz # placeholder
        
        # determine the temperature distribution
        ghost_nodes_temperature = \
            frozen_mask * (self.ghost_nodes_enth / self.config.thermal.c_unfrozen_soil) + \
            unfrozen_mask * (self.ghost_nodes_enth - \
                            (self.config.thermal.c_unfrozen_soil - self.config.thermal.c_frozen_soil) * self.config.thermal.T_melt - \
                             self.config.thermal.L_water_ice * self.config.thermal.nb) / self.config.thermal.c_unfrozen_soil
        
        return ghost_nodes_temperature
    
    def update_grid(self, fp_xbeach_output="sedero.txt"):
        """This function updates the current grid, calculates the angles of the new grid with the horizontal, generates a new thermal grid 
        (perpendicular to the existing grid), and fits the previous temperature and enthalpy distributions to the new grid."""
        
        # generate perpendicular grids for previous timestep (to cast temperature and enthalpy)
        self.abs_xgr, self.abs_zgr = generate_perpendicular_grids(self.xgr, self.zgr)
        
        # update the current bathymetry
        self._update_bed_sedero(fp_xbeach_output=fp_xbeach_output)  # placeholder
        
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
        cum_sedero = np.loadtxt(fp_xbeach_output)
        
        # update bed level
        self.bathy_current += cum_sedero
        
        # save current bed
        self.bed_timeseries.append(self.bathy_current)
        
        return self.bathy_current
        
    def write_ne_layer(self):
        """This function writes the thaw depth obtained from the thermal update to a file to be used by xbeach.
        """
        np.savetxt(os.path.join(self.cwd, "ne_layer.txt"), self.thaw_depth)
        
        return None
        
    def find_thaw_depth(self):
        """Finds thaw depth based on the maximum z-value close to the x-grid coordinate (i.e., +-0.5*dx) that is unthawed."""
        # initialize thaw depth array
        self.thaw_depth = np.zeros(self.xgr.shape)

        # get the points from the temperature models
        x_matrix, z_matrix = generate_perpendicular_grids(
            self.xgr, self.zgr, 
            resolution=self.config.thermal.grid_resolution, 
            max_depth=self.config.thermal.max_depth)
        
        # determine in which interval to look for thaw depth
        dx_min = np.min(self.xgr[1:] - self.xgr[:-1])
        
        # loop through x-coordinates, and find thaw depth for each coordinate
        for i in range(len(self.xgr)):
            
            # selection of points to look at for current grid coordinate
            mask = (x_matrix.flatten() > self.xgr[i] - 0.5 * dx_min) * (x_matrix.flatten() < self.xgr[i] + 0.5 * dx_min)
            
            # look at masked temperature matrix, use that to determine which points are thawed, and use the point with the highest z coordinate to calculate
            # the thaw depth
            self.thaw_depth[i] = self.zgr - np.max(self.zgr.flatten()[mask][self.temp_matrix.flatten()[mask] < self.config.thermal.T_melt])
                
        return self.thaw_depth
        
    def write_output(self, timestep_id):
        """This function writes output in the results folder, and creates subfolders for each timestep for which results are output.
        """
        dir = os.path.join(self.result_dir, str(timestep_id) + "/")
        
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        np.savetxt(os.path.join(dir, "xgr"), self.xgr)
        np.savetxt(os.path.join(dir, "zgr"), self.zgr)
        
        np.savetxt(os.path.join(dir, "xgr"), self.thermal_zgr)

        if "bathymetry" in self.config.output.output_vars:
            np.savetxt(os.path.join(dir, "bathymetry"), self.bathy_current)
        if "ground_temperature_distribution" in self.config.output.output_vars:
            np.savetxt(os.path.join(dir, "xgrid_ground_temperature_distribution"), self.abs_xgr_new.flatten())
            np.savetxt(os.path.join(dir, "zgrid_ground_temperature_distribution"), self.abs_zgr_new.flatten())
            np.savetxt(os.path.join(dir, "ground_temperature_distribution"), self.temp_matrix.flatten())
        if "thaw_depht" in self.config.output.output_vars:
            np.savetxt(os.path.join(dir, "thaw_depht"), self.thaw_depth)
        
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

    @classmethod 
    def _get_timeseries(tstart, tend, fpath):
        """returns timeseries start from tstart and ending at tend. The filepath has to be specified.
        
        returns: pd.DataFrame of length T"""
        
        with open(fpath) as f:
            df = pd.read_csv(f)
            mask = (df["time"] >= tstart) * (df["time"] < tend)
            
            df = df[mask]
            
        return df
    
    # part of old thaw depth method:        
    #   # get the number of grid points for each 1D model where the temperature exceeds the melting point (counted from the top until the first un-thawed point), 
        # normalize with the total number of points, and multiply with the total grid length.
        # self.thaw_depth = count_nonzero_until_zero((self.temp_matrix > self.config.thermal.T_melt)) / self.config.thermal.grid_resolution * self.config.thermal.max_depth
    
            
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