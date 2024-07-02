from datetime import datetime
import os
from pathlib import Path
import yaml

import numpy as np
import xarray as xr

from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt

from utils.bathymetry import calculate_bluff_edge_toe_position, calculate_shoreline_position


class SimulationResults():
    
    ################################################
    ##                                            ##
    ##            # GENERAL FUNCTIONS             ##
    ##                                            ##
    ################################################
    
    def __init__(self, runids=[], result_dir=Path("p:/11210070-usgscoop-202324-arcticxb/runs/")):
        
        self.runids = runids
        
        self.result_dirs_dict = {self.runids[i]:os.path.join(result_dir, self.runids[i]) for i in range(len(self.runids))}
        
        timestep_output_ids_path = os.path.join(result_dir, self.runids[0] + "/")
        dir_list = [item for item in os.listdir(timestep_output_ids_path) if os.path.isdir(os.path.join(timestep_output_ids_path, item))]
        self.timestep_output_ids = np.sort(np.int32(np.array(dir_list)))
        
        self.timestamps = np.loadtxt(os.path.join(result_dir, self.runids[0] + "/", "timestamps.txt"))
        self.timestep_ids = np.loadtxt(os.path.join(result_dir, self.runids[0] + "/", "timestep_ids.txt"))

        var_list_path = os.path.join(timestep_output_ids_path, str(self.timestep_output_ids[0]) + "/")
        self.var_list = np.array([item[:-4] for item in os.listdir(var_list_path)])
        
        self._get_bluff_toes_and_shorelines()
        
        return None
                
    def _get_bluff_toes_and_shorelines(self):
        
        self.bluff_toes = {}
        self.shore_lines = {}
        
        for runid in self.runids:
            bluff_toe = []
            shore_line = []
            
            for timestep_output_id in self.timestep_output_ids:
                
                path_xgr = os.path.join(self.result_dirs_dict[runid], str(timestep_output_id), "xgr.txt")
                path_zgr = os.path.join(self.result_dirs_dict[runid], str(timestep_output_id), "zgr.txt")
                
                with open(path_xgr) as fx:
                    xgr = np.loadtxt(fx)
                    
                with open(path_zgr) as fz:
                    zgr = np.loadtxt(fz)
                    
                bluff_toe.append(calculate_bluff_edge_toe_position(xgr, zgr)[0])
                shore_line.append(calculate_shoreline_position(xgr, zgr))
        
            self.bluff_toes[runid] = bluff_toe
            self.shore_lines[runid] = shore_line
            
        return self.bluff_toes, self.shore_lines
        
    def get_var_timeseries(self, varname):
        
        var_dict = {}
        
        for runid in self.runids:
            var_list = []
            
            for timestep_output_id in self.timestep_output_ids:
                
                path = os.path.join(self.result_dirs_dict[runid], str(timestep_output_id), varname + ".txt")
        
                with open(path) as f:
                    
                    var = np.loadtxt(f)
                    
                    var_list.append(var)
                    
            var_dict[runid] = var_list
            
        return var_dict
    
    def get_var_timestep(self, varname, timestep_id):
        
        var_dict = {}
        
        for runid in self.runids:
            
            path = os.path.join(self.result_dirs_dict[runid], str(timestep_id), varname + ".txt")
            
            with open(path) as f:
            
                var_array = np.loadtxt(f)
                
            var_dict[runid] = var_array
            
        return var_dict
    
    def get_timestamps(self, timestep_ids):
        
        return self.timestamps[[timestep_ids]]
    
    def get_timestep_ids(self, timestamps):
        
        timestep_ids = []
        
        for timestamp in timestamps:
            
            timestep_ids.append(np.argwhere(timestamp==self.timestamps))
        
        return np.array(timestep_ids)
    
    def bed_level_animation(
        self, 
        RUNID,
        animation_timesteps=None,
        xmin=0, 
        xmax=1500, 
        aspect_equal=False, 
        cmap='plasma', 
        save=True, 
        save_folder=Path("results/"),
        save_name='bedlevel',
        fps=5
    ):
        print('creating bed level animation')
        # create figure
        fig, ax = plt.subplots(figsize=(15,5))
        
        # set visual properties
        ax.grid()
        ax.set_xlim((xmin, xmax))
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        
        if aspect_equal:
            ax.set_aspect('equal')
        
        # initialize water level line
        wl_line, = ax.plot((xmin, xmax), (0, 0), color='C0')
        
        # initialize other lines to loop through and set alpha
        lines=[]
        
        # determinae output steps that should be part of the animation
        if not animation_timesteps:
            animation_timesteps = self.timestep_output_ids
        
        # colormap for plotting the bathymetry
        cmap = colormaps[cmap]
        colors = cmap(np.linspace(0, 1, len(animation_timesteps)))
        
        # define animation function
        def animation_function(i):
            
            # set properties of existing lines
            for l in lines:
                l.set(alpha=0.2, linewidth=1)
            
            # get current timestep
            output_id = animation_timesteps[i]
            
            # set title
            timestamp = datetime.fromtimestamp(self.timestamps[np.where(output_id==self.timestep_ids)][0] * 10**-9)
            ax.set_title(f'timestep = {output_id} ({timestamp})')
            
            # load grid
            xgr = self.get_var_timestep("xgr", output_id)[RUNID]
            zgr = self.get_var_timestep("zgr", output_id)[RUNID]
            
            # load water level
            wl = self.get_var_timestep("storm_surge", output_id)[RUNID]
            
            # draw new bathymetry
            line, = ax.plot(xgr, zgr, color=colors[i], linewidth=3)
            lines.append(line)

            # update water level line
            wl_line.set_ydata([wl, wl])
            
            # print progress
            print(f'{output_id} / {len(animation_timesteps)}')
            
            return None
            
        # define animation
        animation = FuncAnimation(fig, animation_function, frames=range(len(animation_timesteps)))
        
        # save animation
        if save:
            animation.save(os.path.join(save_folder, str(RUNID) + save_name + ".mp4"), writer='ffmpeg', fps=fps)
        
        # close 
        plt.close()
            
        return animation
        
    def heat_forcing_animation(
        self,
        RUNID,
        animation_timesteps=None,
        xmin=0, 
        xmax=1500, 
        aspect_equal=False, 
        cmap='plasma', 
        save=True, 
        save_folder=Path("results/"),
        save_name='bedlevel_thawdepth_heatflux',
        fps=5
        ):
        print('creating bed level, thaw depth, heat forcing animation')
        # create figure
        fig, axs = plt.subplots(3, 1, figsize=(15,15))

        # set some visual properties
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()

        axs[0].set_xlim((xmin, xmax))
        axs[1].set_xlim((xmin, xmax))
        axs[2].set_xlim((xmin, xmax))

        axs[0].set_ylim((-10, 15))
        axs[1].set_ylim((-3, 3))
        axs[2].set_ylim((-1000, 1000))

        axs[0].set_title(("Bed level"))
        axs[1].set_title(("Thaw depth"))
        axs[2].set_title(("Surface heat fluxes"))

        axs[0].set_ylabel('z [m]')
        axs[1].set_ylabel('Thaw depth [m]')
        axs[2].set_ylabel('heat flux at surface [W/m2]')

        axs[2].set_xlabel('x [m]')
        
        if aspect_equal:
            axs[0].set_aspect('equal')

        # list with lines to set properties later
        lines = []
        
        # create initial line for water level
        wl_line, = axs[0].plot((xmin, xmax), (0,0), color='C0', label='water level')

        # create initial plot for thaw depth
        thaw_depth_line, = axs[1].plot([],[],label='thaw_depth', color='k')

        # create initial_plot for heat fluxes
        total_heat_flux_line, = axs[2].plot([],[],label='total_heat_flux', color='k', linewidth=3)
        long_wave_radiation_flux_line, = axs[2].plot([],[],label='long_wave_radiation_flux')
        solar_radiation_flux_line, = axs[2].plot([],[],label='solar_radiation_flux')
        latent_heat_flux_line, = axs[2].plot([],[],label='latent_heat_flux')
        convective_heat_flux_line, = axs[2].plot([],[],label='convective_heat_flux')

        # add legends
        axs[0].legend(loc='upper left')
        axs[1].legend(loc='upper left')
        axs[2].legend(loc='upper left')

        # determinae output steps that should be part of the animation
        if not animation_timesteps:
            animation_timesteps = self.timestep_output_ids
        
        # colormap for plotting the bathymetry
        cmap = colormaps[cmap]
        colors = cmap(np.linspace(0, 1, len(animation_timesteps)))
        
        # define animation function
        def animation_function(i):
            
            # set alpha of old bathymetry lines
            for l in lines:
                l.set(alpha=0.3)
            
            # get current timestep id
            output_id = animation_timesteps[i]
            timestamp = datetime.fromtimestamp(self.timestamps[np.where(output_id==self.timestep_ids)][0] * 10**-9)
                            
            # set current timestep id as figure title
            fig.suptitle(f'timestep = {output_id} ({timestamp})')
            
            # get necessary variables
            xgr = self.get_var_timestep("xgr", output_id)[RUNID]
            zgr = self.get_var_timestep("zgr", output_id)[RUNID]
            
            wl = self.get_var_timestep("storm_surge", output_id)[RUNID]
            
            thaw_depth = self.get_var_timestep("thaw_depth", output_id)[RUNID]
            
            total_heat_flux = self.get_var_timestep("total_heat_flux", output_id)[RUNID]
            long_wave_radiation_flux = self.get_var_timestep("long_wave_radiation_flux", output_id)[RUNID]
            solar_radiation_flux = self.get_var_timestep("solar_radiation_flux", output_id)[RUNID]
            latent_heat_flux = self.get_var_timestep("latent_heat_flux", output_id)[RUNID]
            convective_heat_flux = self.get_var_timestep("convective_heat_flux", output_id)[RUNID]
                
            # plot bathymetry (in plot 0)
            line, = axs[0].plot(xgr, zgr, color=colors[i])
            lines.append(line)

            # plot water level (in plot 0)
            wl_line.set_ydata([wl, wl])
            
            # plot thaw depth (in plot 1)
            thaw_depth_line.set_data(xgr, thaw_depth)
            
            # plot surface heat fluxes (in plot 2)
            total_heat_flux_line.set_data(xgr, total_heat_flux)
            long_wave_radiation_flux_line.set_data(xgr, long_wave_radiation_flux)
            solar_radiation_flux_line.set_data(xgr, solar_radiation_flux)
            latent_heat_flux_line.set_data(xgr, latent_heat_flux)
            convective_heat_flux_line.set_data(xgr, convective_heat_flux)
            
            # print progress
            print(f'{output_id} / {len(animation_timesteps)}')
            
            return None

        # define animation
        animation = FuncAnimation(fig, animation_function, frames=range(len(animation_timesteps)))
        
        # save animation
        if save:
            animation.save(os.path.join(save_folder, str(RUNID) + save_name + ".mp4"), writer='ffmpeg', fps=fps)
        
        # close 
        plt.close()
            
        return animation
    
    def temperature_animation(
        self,
        RUNID,
        animation_timesteps=None,
        xmin=0, 
        xmax=1500,
        vmin=-10,  # degrees Celcius
        vmax=10,   # degrees Celcius
        aspect_equal=False, 
        save=True, 
        save_folder=Path("results/"),
        save_name='temperature_thawdepth_heatflux',
        fps=5
        ):
        print('creating temperature, thaw depth, heat flux animation')
        # create figure
        fig, axs = plt.subplots(3, 1, figsize=(15,15), sharex=True)

        # create normalization
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # create scalarmappable
        sm = ScalarMappable(norm, cmap='seismic')

        # initialize temperature scatter plot
        temp_scatter = axs[0].scatter([], [], c=[], cmap='seismic', norm=norm, s=50)
        
        # create colorbar
        plt.colorbar(
            temp_scatter, 
            ax=axs[0], 
            orientation='horizontal', 
            fraction=0.05, 
            pad=0.05, 
            anchor=(0.9, 1), 
            ticks=np.arange(vmin, vmax, 1), 
            label='Temperature [degrees Celcius]', 
            aspect=40
            )

        # initialize lines
        wl_line, = axs[0].plot((0, 1500), (0,0), color='C0', label='water level')
        bathy_line, = axs[0].plot([],[], color='k', label='bed level', linewidth=3)
        thaw_line, = axs[0].plot([],[], color='r', label='thaw interface', linewidth=2)
        
        # create initial plot for thaw depth
        thaw_depth_line, = axs[1].plot([],[],label='thaw_depth', color='k')

        # create initial_plot for heat fluxes
        total_heat_flux_line, = axs[2].plot([],[],label='total_heat_flux', color='k', linewidth=3)
        long_wave_radiation_flux_line, = axs[2].plot([],[],label='long_wave_radiation_flux')
        solar_radiation_flux_line, = axs[2].plot([],[],label='solar_radiation_flux')
        latent_heat_flux_line, = axs[2].plot([],[],label='latent_heat_flux')
        convective_heat_flux_line, = axs[2].plot([],[],label='convective_heat_flux')

        # some visual stuff
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()

        axs[0].set_xlim((xmin, xmax))
        axs[1].set_xlim((xmin, xmax))
        axs[2].set_xlim((xmin, xmax))

        axs[0].set_ylim((-10, 15))
        axs[1].set_ylim((-3, 3))
        axs[2].set_ylim((-1000, 1000))

        axs[0].set_title(("Bed level + ground temperature distribution"))
        axs[1].set_title(("Thaw depth"))
        axs[2].set_title(("Surface heat fluxes"))

        axs[0].set_ylabel('z [m]')
        axs[1].set_ylabel('Thaw depth [m]')
        axs[2].set_ylabel('heat flux at surface [W/m2]')

        axs[2].set_xlabel('x [m]')
        
        if aspect_equal:
            axs[0].set_aspect('equal')

        # add legend()
        axs[0].legend(loc='upper left')
        axs[1].legend(loc='upper left')
        axs[2].legend(loc='upper left')
        
        # determinae output steps that should be part of the animation
        if not animation_timesteps:
            animation_timesteps = self.timestep_output_ids

        def animation_function(i):
            # get current timestep id
            output_id = animation_timesteps[i]
            timestamp = datetime.fromtimestamp(self.timestamps[np.where(output_id==self.timestep_ids)][0] * 10**-9)
                
            print(output_id)
            
            # set current timestep id as figure title
            fig.suptitle(f'timestep = {output_id} ({timestamp})')
            
            # get necessary variables
            xgr = self.get_var_timestep("xgr", output_id)[RUNID]
            zgr = self.get_var_timestep("zgr", output_id)[RUNID]
            
            wl = self.get_var_timestep("storm_surge", output_id)[RUNID]
            
            thaw_depth = self.get_var_timestep("thaw_depth", output_id)[RUNID]
            
            total_heat_flux = self.get_var_timestep("total_heat_flux", output_id)[RUNID]
            long_wave_radiation_flux = self.get_var_timestep("long_wave_radiation_flux", output_id)[RUNID]
            solar_radiation_flux = self.get_var_timestep("solar_radiation_flux", output_id)[RUNID]
            latent_heat_flux = self.get_var_timestep("latent_heat_flux", output_id)[RUNID]
            convective_heat_flux = self.get_var_timestep("convective_heat_flux", output_id)[RUNID]
            
            abs_xgr = self.get_var_timestep("abs_xgr", output_id)[RUNID]
            abs_zgr = self.get_var_timestep("abs_zgr", output_id)[RUNID]
            temp = self.get_var_timestep("ground_temperature_distribution", output_id)[RUNID] - 273.15
            temp_norm = sm.to_rgba(temp)  # get normalized temperature values (needed to plot)

            # plot bathymetry (in plot 0)
            bathy_line.set_data(xgr, zgr)
            thaw_line.set_data(xgr, zgr - thaw_depth)

            # plot water level (in plot 0)
            wl_line.set_ydata([wl, wl])
            
            # plot thaw interface (in plot 0)
            thaw_line.set_data(xgr, zgr - thaw_depth)
            
            # plot temperature (in plot 0)
            temp_scatter.set_offsets(np.column_stack((abs_xgr, abs_zgr)))
            temp_scatter.set_color(temp_norm)
            
            # plot thaw depth (in plot 1)
            thaw_depth_line.set_data(xgr, thaw_depth)
            
            # plot surface heat fluxes (in plot 2)
            total_heat_flux_line.set_data(xgr, total_heat_flux)
            long_wave_radiation_flux_line.set_data(xgr, long_wave_radiation_flux)
            solar_radiation_flux_line.set_data(xgr, solar_radiation_flux)
            latent_heat_flux_line.set_data(xgr, latent_heat_flux)
            convective_heat_flux_line.set_data(xgr, convective_heat_flux)
            
            # print progress
            print(f'{output_id} / {len(animation_timesteps)}')
            
            return None

        # define animation
        animation = FuncAnimation(fig, animation_function, frames=range(len(animation_timesteps)))
        
        # save animation
        if save:
            animation.save(os.path.join(save_folder, str(RUNID) + save_name + ".mp4"), writer='ffmpeg', fps=fps)
        
        # close 
        plt.close()
            
        return animation
