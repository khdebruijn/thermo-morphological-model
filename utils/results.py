from datetime import datetime, timedelta
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
    
    def __init__(self, runid, runs_folder=Path("p:/11210070-usgscoop-202324-arcticxb/runs/")):
        
        self.runid = runid
        
        self.result_dir = os.path.join(runs_folder, str(runid) + "/")
        
        self.timestep_output_ids = np.int32(np.loadtxt(os.path.join(runs_folder, str(self.runid) + "/", "timestep_output_ids.txt")))
        self.xbeach_times = (np.loadtxt(os.path.join(runs_folder, str(self.runid) + "/", "xbeach_times.txt")) >= 1)
        self.timestamps = np.loadtxt(os.path.join(runs_folder, str(self.runid) + "/", "timestamps.txt"))
        self.timestep_ids = np.int32(np.loadtxt(os.path.join(runs_folder, str(self.runid) + "/", "timestep_ids.txt")))
        
        ds = xr.open_dataset(os.path.join(self.result_dir, "0000000000.nc"))
        self.var_list = list(ds.coords) + list(ds.keys())
        ds.close()
                
        return None
                
    def get_bluff_toes_and_shorelines(self, save=True):

        self.bluff_edges = []
        self.bluff_toes = []
        self.shore_lines = []
        
        for timestep_output_id in self.timestep_output_ids:
            
            ds = xr.open_dataset(os.path.join(self.result_dir, str(timestep_output_id) + ".nc"))
            
            xgr = ds['xgr'].values
            zgr = ds['zgr'].values
            
            ds.close()
            
            be, bt, distance = calculate_bluff_edge_toe_position(xgr, zgr)
            sl = calculate_shoreline_position(xgr, zgr)
            
            self.bluff_edges.append(be)
            self.bluff_toes.append(bt)
            self.shore_lines.append(sl)
            
        self.bluff_edges = np.array(self.bluff_edges)
        self.bluff_toes = np.array(self.bluff_toes)
        self.shore_lines = np.array(self.shore_lines)
        
        if save:
            np.savetxt(os.path.join(self.result_dir, "bluff_edges.txt"), self.bluff_edges)
            np.savetxt(os.path.join(self.result_dir, "bluff_toes.txt"), self.bluff_toes)
            np.savetxt(os.path.join(self.result_dir, "shore_lines.txt"), self.shore_lines)

        return self.bluff_edges, self.shore_lines
    
    def get_var_timeseries(self, varname):
        
        var_list = []
        
        for timestep_output_id in self.timestep_output_ids:
            
            path = os.path.join(self.result_dir, str(timestep_output_id) + ".nc")

            ds = xr.open_dataset(path)
                
            var_list.append(ds[varname].values)  
            
            ds.close()
            
        return np.array(var_list)
    
    def get_var_timestep(self, varname, timestep_id):
                  
        path = os.path.join(self.result_dir, (10 - len(str(int(timestep_id)))) * '0' + str(int(timestep_id)) + ".nc")
        
        ds = xr.open_dataset(path)
        
        var_array = ds[varname].values
        
        ds.close()        
            
        return var_array
    
    def get_timestamps(self, timestep_ids):
        
        return self.timestamps[[timestep_ids]]
    
    def get_timestep_ids(self, timestamps):
        
        timestep_ids = []
        
        for timestamp in timestamps:
            
            timestep_ids.append(np.argwhere(timestamp==self.timestamps))
        
        return np.array(timestep_ids)
    
    def bed_level_animation(
        self, 
        animation_timesteps=None,
        xmin=0, 
        xmax=1500, 
        aspect_equal=False, 
        cmap='plasma', 
        save=True,
        make_animation=True,
        save_folder=Path("P:/11210070-usgscoop-202324-arcticxb/results/"),
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
        wl_line, = ax.plot((xmin, xmax), (0, 0), color='C0', label='water level')
        
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
            output_id = int(animation_timesteps[i])
            
            # set title
            timestamp = datetime.fromtimestamp(self.timestamps[np.nonzero(output_id==self.timestep_ids)][0] * 10**-9)
            ax.set_title(f'timestep = {output_id} \n({timestamp} UTC / {timestamp - timedelta(hours=9)} AKST / {timestamp - timedelta(hours=8)} AKDT)')
            
            # load grid
            xgr = self.get_var_timestep("xgr", output_id)
            zgr = self.get_var_timestep("zgr", output_id)
            
            # load water level
            wl = self.get_var_timestep("water_level", output_id)
            
            # draw new bathymetry
            line, = ax.plot(xgr, zgr, color=colors[i], linewidth=3)
            lines.append(line)

            # update water level line
            wl_line.set_ydata([wl, wl])
            
            # print progress
            print(f'{i} / {len(animation_timesteps)}')
            
            return None
                
        if make_animation:

            # define animation
            animation = FuncAnimation(fig, animation_function, frames=range(len(animation_timesteps)))
            
            # save animation
            if save:
                
                fpath = os.path.join(save_folder, str(self.runid) + save_name + ".mp4")
                
                animation.save(fpath, writer='ffmpeg', fps=fps)
            
            # close 
            plt.close()
        
        else:
            
            # create folder for images
            fig_dir = os.path.join(save_folder, str(self.runid) + "_" + save_name + '/')
            
            if not os.path.exists(fig_dir):
                
                os.makedirs(fig_dir)
            
            # loop through all frames and save figure after each one
            for frame in range(len(animation_timesteps)):
                
                animation_function(frame)
                
                fig.savefig(os.path.join(fig_dir, str(self.timestep_output_ids[frame]) + ".png"))
            
            plt.close()
            
        return None
        
    def heat_forcing_animation(
        self,
        animation_timesteps=None,
        xmin=0, 
        xmax=1500, 
        aspect_equal=False, 
        cmap='plasma', 
        save=True, 
        make_animation=False,
        save_folder=Path("P:/11210070-usgscoop-202324-arcticxb/results/"),
        save_name='bedlevel_thawdepth_heatflux',
        fps=5
        ):
        print('creating bed level, thaw depth, heat forcing animation')
        # create figure
        fig, axs = plt.subplots(3, 1, figsize=(15,20))

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
        thaw_depth_line, = axs[1].plot([],[],label='thaw depth', color='k')

        # create initial_plot for heat fluxes
        total_heat_flux_line, = axs[2].plot([],[],label='total heat flux', color='k', linewidth=3)
        long_wave_radiation_flux_line, = axs[2].plot([],[],label='long-wave radiation flux')
        solar_radiation_flux_line, = axs[2].plot([],[],label='solar radiation flux')
        latent_heat_flux_line, = axs[2].plot([],[],label='latent heat flux')
        convective_heat_flux_line, = axs[2].plot([],[],label='convective heat flux')

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
            fig.suptitle(f'timestep = {output_id} \n({timestamp} UTC / {timestamp - timedelta(hours=9)} AKST / {timestamp - timedelta(hours=8)} AKDT)')
            
            # get necessary variables
            xgr = self.get_var_timestep("xgr", output_id)
            zgr = self.get_var_timestep("zgr", output_id)
            
            wl = self.get_var_timestep("water_level", output_id)
            
            thaw_depth = self.get_var_timestep("thaw_depth", output_id)
            
            total_heat_flux = self.get_var_timestep("total_heat_flux", output_id)
            long_wave_radiation_flux = self.get_var_timestep("long_wave_radiation_flux", output_id)
            solar_radiation_flux = self.get_var_timestep("solar_radiation_flux", output_id)
            latent_heat_flux = self.get_var_timestep("latent_heat_flux", output_id)
            convective_heat_flux = self.get_var_timestep("convective_heat_flux", output_id)
                
            # plot bathymetry (in plot 0)
            line, = axs[0].plot(xgr, zgr, color=colors[i])
            lines.append(line)

            # plot water level (in plot 0)
            wl_line.set_ydata([wl, wl])
            
            # plot thaw depth (in plot 1)
            thaw_depth_line.set_data(xgr, thaw_depth)
            
            # plot surface heat fluxes (in plot 2)
            total_heat_flux_line.set_data(xgr, total_heat_flux)
            long_wave_radiation_flux_line.set_data(xgr, long_wave_radiation_flux*np.ones(xgr.shape))
            solar_radiation_flux_line.set_data(xgr, solar_radiation_flux)
            latent_heat_flux_line.set_data(xgr, latent_heat_flux*np.ones(xgr.shape))
            convective_heat_flux_line.set_data(xgr, convective_heat_flux)
            
            # print progress
            print(f'{i} / {len(animation_timesteps)}')
            
            return None

        if make_animation:

            # define animation
            animation = FuncAnimation(fig, animation_function, frames=range(len(animation_timesteps)))
            
            # save animation
            if save:
                
                fpath = os.path.join(save_folder, str(self.runid) + save_name + ".mp4")
                
                animation.save(fpath, writer='ffmpeg', fps=fps)
            
            # close 
            plt.close()
        
        else:
            
            # create folder for images
            fig_dir = os.path.join(save_folder, str(self.runid) + "_" + save_name + '/')
            
            if not os.path.exists(fig_dir):
                
                os.makedirs(fig_dir)
            
            # loop through all frames and save figure after each one
            for frame in range(len(animation_timesteps)):
                
                animation_function(frame)
                
                fig.savefig(os.path.join(fig_dir, str(self.timestep_output_ids[frame]) + ".png"))
                
            plt.close()
            
        return None
    
    def temperature_heatforcing_animation(
        self,
        animation_timesteps=None,
        xmin=0, 
        xmax=1500,
        vmin=-10,  # degrees Celcius
        vmax=10,   # degrees Celcius
        aspect_equal=False, 
        save=True, 
        make_animation=True,
        save_folder=Path("P:/11210070-usgscoop-202324-arcticxb/results/"),
        save_name='temperature_thawdepth_heatflux',
        fps=5
        ):
        print('creating temperature, thaw depth, heat flux animation')
        # create figure
        fig, axs = plt.subplots(3, 1, figsize=(15,20), sharex=True)

        # create normalization
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # create scalarmappable
        sm = ScalarMappable(norm, cmap='coolwarm')

        # initialize temperature scatter plot
        s = (1500) / (xmax-xmin)
        temp_scatter = axs[0].scatter([], [], color=[], s=s)
        
        # create colorbar
        plt.colorbar(
            sm, 
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
        thaw_depth_line, = axs[1].plot([],[],label='thaw depth', color='k')

        # create initial_plot for heat fluxes
        total_heat_flux_line, = axs[2].plot([],[],label='total heat flux', color='k', linewidth=3)
        long_wave_radiation_flux_line, = axs[2].plot([],[],label='long-wave radiation flux')
        solar_radiation_flux_line, = axs[2].plot([],[],label='solar radiation flux')
        latent_heat_flux_line, = axs[2].plot([],[],label='latent heat flux')
        convective_heat_flux_line, = axs[2].plot([],[],label='convective heat flux')
        
        # create plots for sea temperature label and air temperature label
        air_temp_label = axs[0].scatter([],[], label='2m air temperature: ')
        sea_temp_label = axs[0].scatter([],[], label='sea surface temperature: ')


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
        L0 = axs[0].legend(loc='upper left')
        axs[1].legend(loc='upper left')
        axs[2].legend(loc='upper left')
        
        # determinae output steps that should be part of the animation
        if not animation_timesteps:
            animation_timesteps = self.timestep_output_ids

        def animation_function(i):
            # get current timestep id
            output_id = animation_timesteps[i]
            timestamp = datetime.fromtimestamp(self.timestamps[np.where(output_id==self.timestep_ids)][0] * 10**-9)
                            
            # set current timestep id as figure title
            fig.suptitle(f'timestep = {output_id} \n({timestamp} UTC / {timestamp - timedelta(hours=9)} AKST / {timestamp - timedelta(hours=8)} AKDT)')
            
            # get necessary variables
            xgr = self.get_var_timestep("xgr", output_id)
            zgr = self.get_var_timestep("zgr", output_id)
            
            wl = self.get_var_timestep("water_level", output_id)
            
            thaw_depth = self.get_var_timestep("thaw_depth", output_id)
            
            total_heat_flux = self.get_var_timestep("total_heat_flux", output_id)
            long_wave_radiation_flux = self.get_var_timestep("long_wave_radiation_flux", output_id)
            solar_radiation_flux = self.get_var_timestep("solar_radiation_flux", output_id)
            latent_heat_flux = self.get_var_timestep("latent_heat_flux", output_id)
            convective_heat_flux = self.get_var_timestep("convective_heat_flux", output_id)
            
            abs_xgr = self.get_var_timestep("abs_xgr", output_id).flatten()
            abs_zgr = self.get_var_timestep("abs_zgr", output_id).flatten()
            temp = self.get_var_timestep("ground_temperature_distribution", output_id).flatten() - 273.15
            temp_norm = sm.to_rgba(temp)  # get normalized temperature values (needed to plot)
            
            air_temp = self.get_var_timestep("2m_temperature", output_id)
            sea_temp = self.get_var_timestep("sea_surface_temperature", output_id)

            # plot bathymetry (in plot 0)
            bathy_line.set_data(xgr, zgr)
            thaw_line.set_data(xgr, zgr - thaw_depth)

            # plot water level (in plot 0)
            wl_line.set_ydata([wl, wl])
            
            # plot thaw interface (in plot 0)
            thaw_line.set_data(xgr, zgr - thaw_depth)
            
            # plot temperature (in plot 0)
            temp_scatter.set_offsets(np.column_stack((abs_xgr.flatten(), abs_zgr.flatten())))
            temp_scatter.set_color(temp_norm)
            
            # plot thaw depth (in plot 1)
            thaw_depth_line.set_data(xgr, thaw_depth)
            
            # plot surface heat fluxes (in plot 2)
            total_heat_flux_line.set_data(xgr, total_heat_flux)
            long_wave_radiation_flux_line.set_data(xgr, long_wave_radiation_flux)
            solar_radiation_flux_line.set_data(xgr, solar_radiation_flux)
            latent_heat_flux_line.set_data(xgr, latent_heat_flux)
            convective_heat_flux_line.set_data(xgr, convective_heat_flux)
            
            # update temperature labels
            L0.get_texts()[3].set_text(f"2m temperature: {air_temp - 273.15:.1f} degrees C")
            L0.get_texts()[4].set_text(f"Sea surface temperature: {sea_temp - 273.15:.1f} degrees C")
            
            # print progress
            print(f'{i} / {len(animation_timesteps)}')
            
            return None

        if make_animation:

            # define animation
            animation = FuncAnimation(fig, animation_function, frames=range(len(animation_timesteps)))
            
            # save animation
            if save:
                
                fpath = os.path.join(save_folder, str(self.runid) + save_name + ".mp4")
                
                animation.save(fpath, writer='ffmpeg', fps=fps)
            
            # close 
            plt.close()
        
        else:
            
            # create folder for images
            fig_dir = os.path.join(save_folder, str(self.runid) + "_" + save_name + '/')
            
            if not os.path.exists(fig_dir):
                
                os.makedirs(fig_dir)
            
            # loop through all frames and save figure after each one
            for frame in range(len(animation_timesteps)):
                
                animation_function(frame)
                
                fig.savefig(os.path.join(fig_dir, str(self.timestep_output_ids[frame]) + ".png"))
                
            plt.close()
            
        return None
    
    def bed_temperature_thawdepth_heatforcing_xbtimes_animation(
        self,
        animation_timesteps=None,
        xmin=0, 
        xmax=1500,
        vmin=-10,  # degrees Celcius
        vmax=10,   # degrees Celcius
        cmap='plasma',
        aspect_equal=False, 
        save=True, 
        make_animation=True,
        save_folder=Path("P:/11210070-usgscoop-202324-arcticxb/results/"),
        save_name='bed_temperature_thawdepth_heatflux',
        fps=5
        ):
        print('creating bed level temperature, thaw depth, heat flux animation')
        
        # determinae output steps that should be part of the animation
        if not animation_timesteps:
            animation_timesteps = self.timestep_output_ids
        
        # create colors for the bed level
        cmap_bed = colormaps[cmap]
        colors=cmap_bed(np.linspace(0, 1, len(animation_timesteps)))
        norm_bed = Normalize()
        sm_bed = ScalarMappable(norm_bed, cmap=cmap_bed)
        
        # initialize lines list
        bed_level_lines = []
        
        # create figure
        fig, axs = plt.subplots(5, 1, figsize=(25,25), gridspec_kw={'height_ratios':[2,2,1,1, 0.5]}, layout='constrained')
        
        # create normalization
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # create scalarmappable
        sm = ScalarMappable(norm, cmap='coolwarm')

        # initialize temperature scatter plot
        s = (1500) / (xmax-xmin)
        temp_scatter = axs[1].scatter([], [], color=[], s=s)
        
        # create colorbar
        cbar_bed = plt.colorbar(
            sm_bed, 
            ax=axs[0], 
            orientation='vertical', 
            fraction=0.05, 
            pad=0.05, 
            ticks=[0, 1], 
            label='Color of bed relative to simulation length', 
            aspect=40
        )
        cbar_bed.ax.set_yticklabels(['start', 'end'])
        
        cbar_temp = plt.colorbar(
            sm, 
            ax=axs[1], 
            orientation='vertical', 
            fraction=0.05, 
            pad=0.05, 
            ticks=np.arange(vmin, vmax, 1), 
            label='Temperature [degrees Celcius]', 
            aspect=40
            )

        # initialize lines
        wl_line_bed, = axs[0].plot((0, 1500), (0,0), color='C0', label='water level')
        wl_line_temp, = axs[1].plot((0, 1500), (0,0), color='C0', label='water level')
        bathy_line, = axs[1].plot([],[], color='k', label='bed level', linewidth=3)
        thaw_line, = axs[1].plot([],[], color='r', label='thaw interface', linewidth=2)
        
        # create initial plot for thaw depth
        thaw_depth_line, = axs[2].plot([],[],label='thaw depth', color='k')

        # create initial_plot for heat fluxes
        total_heat_flux_line, = axs[3].plot([],[],label='total heat flux', color='k', linewidth=3)
        long_wave_radiation_flux_line, = axs[3].plot([],[],label='long-wave radiation flux')
        solar_radiation_flux_line, = axs[3].plot([],[],label='solar radiation flux')
        latent_heat_flux_line, = axs[3].plot([],[],label='latent heat flux')
        convective_heat_flux_line, = axs[3].plot([],[],label='convective heat flux')
        
        # create plots for sea temperature label and air temperature label
        air_temp_label = axs[1].scatter([],[], label='2m air temperature: ', alpha=0)
        sea_temp_label = axs[1].scatter([],[], label='sea surface temperature: ', alpha=0)
        
        # create initial plot for xbeach times plot
        axs[4].plot(np.arange(len(self.xbeach_times)), self.xbeach_times, color='k')
        xticks = np.arange(0, 8760 + 730, 730)
        xtick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axs[4].set_xticks(xticks, xtick_labels)
        axs[4].set_xticklabels(rot=10)
        current_timestep_line, = axs[4].plot([0, 0], [-2, 2], color='r', label='current timestep')

        # some visual stuff
        axs[0].grid()
        axs[1].grid()
        axs[2].grid()
        axs[3].grid()
        axs[4].grid()

        axs[0].set_xlim((xmin, xmax))
        axs[1].set_xlim((xmin, xmax))
        axs[2].set_xlim((xmin, xmax))
        axs[3].set_xlim((xmin, xmax))

        axs[0].set_ylim((-10, 20))
        axs[1].set_ylim((-10, 20))
        axs[2].set_ylim((-1, 4))
        axs[3].set_ylim((-1000, 1000))
        axs[4].set_ylim((-0.5, 1.5))
        axs[4].set_yticks((0, 1), ("Don't run XB", "Run XB"))

        axs[0].set_title(("Bed level"))
        axs[1].set_title(("Bed level + ground temperature distribution"))
        axs[2].set_title(("Thaw depth"))
        axs[3].set_title(("Surface heat fluxes"))
        axs[4].set_title(("Whether or not to XBeach"))

        axs[0].set_ylabel('z [m]')
        axs[1].set_ylabel('z [m]')
        axs[2].set_ylabel('Thaw depth [m]')
        axs[3].set_ylabel('heat flux at surface [W/m2]')

        axs[3].set_xlabel('x [m]')
        axs[4].set_xlabel('timestep')
        
        if aspect_equal:
            axs[0].set_aspect('equal')

        # add legend()
        L0 = axs[1].legend(loc='upper left')
        axs[2].legend(loc='upper left')
        axs[3].legend(loc='lower left')
        axs[4].legend(loc='upper left')

        def animation_function(i):
            # get current timestep id
            output_id = animation_timesteps[i]
            timestamp = datetime.fromtimestamp(self.timestamps[np.where(output_id==self.timestep_ids)][0] * 10**-9)
                            
            # set current timestep id as figure title
            fig.suptitle(f'timestep = {output_id} \n({timestamp} UTC / {timestamp - timedelta(hours=9)} AKST / {timestamp - timedelta(hours=8)} AKDT)')
                        
            # get necessary variables
            xgr = self.get_var_timestep("xgr", output_id)
            zgr = self.get_var_timestep("zgr", output_id)
            
            wl = self.get_var_timestep("water_level", output_id)
            
            thaw_depth = self.get_var_timestep("thaw_depth", output_id)
            
            total_heat_flux = self.get_var_timestep("total_heat_flux", output_id)
            long_wave_radiation_flux = self.get_var_timestep("long_wave_radiation_flux", output_id)
            solar_radiation_flux = self.get_var_timestep("solar_radiation_flux", output_id)
            latent_heat_flux = self.get_var_timestep("latent_heat_flux", output_id)
            convective_heat_flux = self.get_var_timestep("convective_heat_flux", output_id)
            
            abs_xgr = self.get_var_timestep("abs_xgr", output_id).flatten()
            abs_zgr = self.get_var_timestep("abs_zgr", output_id).flatten()
            temp = self.get_var_timestep("ground_temperature_distribution", output_id).flatten() - 273.15
            temp_norm = sm.to_rgba(temp)  # get normalized temperature values (needed to plot)
            
            air_temp = self.get_var_timestep("2m_temperature", output_id)
            sea_temp = self.get_var_timestep("sea_surface_temperature", output_id)
            
            # set alpha and lw of historic bathymetries (in plot 0)
            for l in bed_level_lines:
                l.set(alpha=0.2, linewidth=1)
            
            # plot bathymetry (in plot 0)
            line, = axs[0].plot(xgr, zgr, color=colors[i], linewidth=3)
            bed_level_lines.append(line)

            # plot bathymetry (in plot 1)
            bathy_line.set_data(xgr, zgr)
            thaw_line.set_data(xgr, zgr - thaw_depth)

            # plot water level (in plot 0 & 1)
            zs = self.get_var_timestep("zs", output_id)
            
                # some older simulations don't have xgr_xb for all datasets
            try:
                xgr_xb = self.get_var_timestep("xgr_xb", output_id)
            except KeyError:
                xgr_xb = self.get_var_timestep("xgr", output_id)
                
            wl_line_bed.set_data([xgr_xb, zs])
            wl_line_temp.set_data([xgr_xb, zs])
            
            # plot thaw interface (in plot 1)
            thaw_line.set_data(xgr, zgr - thaw_depth)
            
            # plot temperature (in plot 1)
            temp_scatter.set_offsets(np.column_stack((abs_xgr.flatten(), abs_zgr.flatten())))
            temp_scatter.set_color(temp_norm)
            
            # plot thaw depth (in plot 2)
            thaw_depth_line.set_data(xgr, thaw_depth)
            
            # plot surface heat fluxes (in plot 3)
            total_heat_flux_line.set_data(xgr, total_heat_flux)
            long_wave_radiation_flux_line.set_data(xgr, long_wave_radiation_flux)
            solar_radiation_flux_line.set_data(xgr, solar_radiation_flux)
            latent_heat_flux_line.set_data(xgr, latent_heat_flux)
            convective_heat_flux_line.set_data(xgr, convective_heat_flux)
            
            # update temperature labels (in plot 1)
            # L0.get_texts()[0].set_text(f"water level: {zs:.2f} m")
            L0.get_texts()[3].set_text(f"2m temperature: {air_temp - 273.15:.1f} degrees C")
            L0.get_texts()[4].set_text(f"Sea surface temperature: {sea_temp - 273.15:.1f} degrees C")
            
            # plot current timestep (in plot 4)
            current_timestep_line.set_xdata([output_id, output_id])
            
            # print progress
            print(f'{i} / {len(animation_timesteps)}')
            
            return None
        

        if make_animation:

            # define animation
            animation = FuncAnimation(fig, animation_function, frames=range(len(animation_timesteps)))
            
            # save animation
            if save:
                
                fpath = os.path.join(save_folder, str(self.runid) + save_name + ".mp4")
                
                animation.save(fpath, writer='ffmpeg', fps=fps)
            
            # close 
            plt.close()
        
        else:
            
            # create folder for images
            fig_dir = os.path.join(save_folder, str(self.runid) + "_" + save_name + '/')
            
            if not os.path.exists(fig_dir):
                
                os.makedirs(fig_dir)
            
            # loop through all frames and save figure after each one
            for frame in range(len(animation_timesteps)):
                
                animation_function(frame)
                
                fig.savefig(os.path.join(fig_dir, str(self.timestep_output_ids[frame]) + ".png"))
                
            plt.close()
            
        return None
    
    def temperature_animation(
        self,
        animation_timesteps=None,
        vmin=-10,  # degrees Celcius
        vmax=10,   # degrees Celcius
        aspect_equal=False, 
        save=True, 
        make_animation=True,
        save_folder=Path("P:/11210070-usgscoop-202324-arcticxb/results/"),
        save_name='temperature',
        fps=5
    ):
        print('creating temperature animation')
        # create figure
        fig, ax = plt.subplots(figsize=(15,5))

        # create normalization
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # create scalarmappable
        sm = ScalarMappable(norm, cmap='coolwarm')
        
        if aspect_equal:
            ax.set_aspect('equal')
        
        # determinae output steps that should be part of the animation
        if not animation_timesteps:
            animation_timesteps = self.timestep_output_ids

        def animation_function(i):
            ax.clear()
            
            # create colorbar
            plt.colorbar(
                sm, 
                ax=ax[0], 
                orientation='horizontal', 
                fraction=0.05, 
                pad=0.05, 
                anchor=(0.9, 1), 
                ticks=np.arange(vmin, vmax, 1), 
                label='Temperature [degrees Celcius]', 
                aspect=40
                )

            # some visual stuff
            ax.set_title("Ground temperature grid")
            ax.set_ylabel('z-id')
            ax.set_xlabel('x-grid')
            
            # get current timestep id
            output_id = animation_timesteps[i]
            timestamp = datetime.fromtimestamp(self.timestamps[np.where(output_id==self.timestep_ids)][0] * 10**-9)
                            
            # set current timestep id as figure title
            fig.suptitle(f'timestep = {output_id} \n({timestamp} UTC / {timestamp - timedelta(hours=9)} AKST / {timestamp - timedelta(hours=8)} AKDT)')
            
            # get necessary variables
            temp = self.get_var_timestep("ground_temperature_distribution", output_id)[:,::-1].T - 273.15
            temp_colors = sm.to_rgba(temp)
            
            psm = ax.pcolormesh(temp, rasterized=True,)
            psm.set_color(temp_colors)
            
            # print progress
            print(f'{i} / {len(animation_timesteps)}')
            
            return None

        if make_animation:

            # define animation
            animation = FuncAnimation(fig, animation_function, frames=range(len(animation_timesteps)))
            
            # save animation
            if save:
                
                fpath = os.path.join(save_folder, str(self.runid) + save_name + ".mp4")
                
                animation.save(fpath, writer='ffmpeg', fps=fps)
            
            # close 
            plt.close()
        
        else:
            
            # create folder for images
            fig_dir = os.path.join(save_folder, str(self.runid) + "_" + save_name + '/')
            
            if not os.path.exists(fig_dir):
                
                os.makedirs(fig_dir)
            
            # loop through all frames and save figure after each one
            for frame in range(len(animation_timesteps)):
                
                animation_function(frame)
                
                fig.savefig(os.path.join(fig_dir, str(self.timestep_output_ids[frame]) + ".png"))
                
            plt.close()
            
        return None