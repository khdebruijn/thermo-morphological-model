import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.miscellaneous import calculate_bluff_edge_toe_position, calculate_shoreline_position


def block_print():
    """Disables printing"""
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    """Enables printing."""
    sys.stdout = sys.__stdout__


class SimulationPlot(object):
    """
    ...
    """
    def __init__(self, sim):
        """Initializer for the SimulationPlot class.

        Args:
            sim (Simulation): an instance of a simulation class. The simulation does not have to be ran. 
            However, the simulation object has an associated config.yaml file, directories, and string representation.
        """
        sim.set_temporal_params(
            sim.config.model.time_start,
            sim.config.model.time_end,
            sim.config.model.timestep
        )
        self.sim = sim

        
    def plot_transect(self, timestep_id, ax=None, figsize=(15,5), equal_axis=False):
        """Plots a transect at some time instance.

        Args:
            timestep_id (int OR list OR str): timestep_id(s) to be plotted on the axis. If 'all', all timesteps with written output from the simulation will be plotted.
            ax (Axis, optional): Axis to plot the transects on. Default None, new axis is created.
            figsize (tuple, optional): Figure size of the new axis. Defaults to (15,5).
            equal_axis (boolean, optional): if set to True, axis will be scaled to be equal.
        --------------
        Returns:
            ax (Axis): Axis object containing the plot.
        """
        if not ax:
            fig, ax = plt.subplots(figsize)
            
        if type(timestep_id) == int:
            
            dirpath = os.path.join(self.sim.cwd, 'results/', f'{timestep_id}/')
            
            with open(os.path.join(dirpath, 'xgr.txt')) as f:
                xgr = np.loadtxt(f)
            with open(os.path.join(dirpath, 'zgr.txt')) as f:
                zgr = np.loadtxt(f)
                
            ax.plot(xgr, zgr, label=f'timestep_id: {timestep_id}')
        
        elif type(timestep_id) == list:
            
            for id in timestep_id:
                
                dirpath = os.path.join(self.sim.cwd, 'results/', f'{id}/')
            
                with open(os.path.join(dirpath, 'xgr.txt')) as f:
                    xgr = np.loadtxt(f)
                with open(os.path.join(dirpath, 'zgr.txt')) as f:
                    zgr = np.loadtxt(f)
                
            ax.plot(xgr, zgr, label=f'timestep_id: {id}')
        
        elif timestep_id == 'all':
            
            dirpath = os.path.join(self.sim.cwd, 'results/')
            
            dirlist = [item for item in os.listdir(dirpath) if os.path.isdir(item)]
            
            for id in dirlist:
                
                path = os.path.join(dirpath, id)
                
                with open(os.path.join(dirpath, 'xgr.txt')) as f:
                    xgr = np.loadtxt(f)
                with open(os.path.join(dirpath, 'zgr.txt')) as f:
                    zgr = np.loadtxt(f)
                
                ax.plot(xgr, zgr, label=f'timestep_id: {id}')
            
        else:
            raise TypeError("'timestep_id' should be an integer, list of integers, or 'all'.")
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        
        ax.set_title('Bathymetry')
        
        if equal_axis:
            ax.set_aspect('equal')
        
        ax.legend()
        
        return ax
    
    def plot_transect_with_thermal_grid(self, timestep_id, ax=None, figsize=(15,5), equal_axis=False):
        """Plots a transect at some time instance, with the corresponding thermal grid.

        Args:
            timestep_id (int): timestep_id to be plotted on the axis.
            ax (Axis, optional): Axis to plot the transects on. Default None, new axis is created.
            figsize (tuple, optional): Figure size of the new axis. Defaults to (15,5).
            equal_axis (boolean, optional): if set to True, axis will be scaled to be equal.
        --------------
        Returns:
            ax (Axis): Axis object containing the plot.
        """
        if not ax:
            fig, ax = plt.subplots(figsize)
            
        dirpath = os.path.join(self.sim.cwd, 'results/', f'{timestep_id}/')
        
        with open(os.path.join(dirpath, 'xgr.txt')) as f:
            xgr = np.loadtxt(f)
        with open(os.path.join(dirpath, 'zgr.txt')) as f:
            zgr = np.loadtxt(f)
        with open(os.path.join(dirpath, 'xgrid_ground_temperature_distribution.txt')) as f:
            abs_xgr = np.reshape(np.loadtxt(f), (len(xgr), -1))
        with open(os.path.join(dirpath, 'zgrid_ground_temperature_distribution.txt')) as f:
            abs_zgr = np.reshape(np.loadtxt(f), (len(xgr), -1))
        
        ax.plot(xgr, zgr, color='C0', label=f'Bathymetry')
        ax.scatter(xgr, zgr, color='k', s=10, label='Bathymetry grid points')
        
        for i in range(len(xgr)):
            x = abs_xgr[i,:]
            z = abs_zgr[i,:]
            
            if i == 0:
                ax.plot(x, z, color='C1', label='Thermal grid')
                ax.scatter(x, z, color='r', label='Thermal grid points')
            else:
                ax.plot(x, z, color='C1')
                ax.scatter(x, z, color='r')
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        
        ax.set_title(f'Grids at timestep_id: {timestep_id}')
        
        if equal_axis:
            ax.set_aspect('equal')
        
        ax.legend()
        
        return ax
        
    def plot_temperature_transect(self, timestep_id, ax=None, figsize=(15,5), equal_axis=False):
        """Plots a transect with ground temperature distribution at some timestep id.

        Args:
            timestep_id (int): timestep_id to be plotted on the axis.
            ax (Axis, optional): Axis to plot the transects on. Default None, new axis is created.
            figsize (tuple, optional): Figure size of the new axis. Defaults to (15,5).
            equal_axis (boolean, optional): if set to True, axis will be scaled to be equal.

        Returns:
            fig, ax: figure and axis containing the plot.
        """
        if not ax:
            fig, ax = plt.subplots(figsize)
            
        dirpath = os.path.join(self.sim.cwd, 'results/', f'{timestep_id}/')

        with open(os.path.join(dirpath, 'xgr.txt')) as f:
            xgr = np.loadtxt(f)
        with open(os.path.join(dirpath, 'zgr.txt')) as f:
            zgr = np.loadtxt(f)
        with open(os.path.join(dirpath, 'xgrid_ground_temperature_distribution.txt')) as f:
            abs_xgr = np.loadtxt(f)
        with open(os.path.join(dirpath, 'zgrid_ground_temperature_distribution.txt')) as f:
            abs_zgr = np.loadtxt(f)
        with open(os.path.join(dirpath, 'ground_temperature_distribution.txt')) as f:
            temp = np.loadtxt(f)
        
        ax.plot(xgr, zgr, color='k', label='Bathymetry')
        scatter = ax.scatter(abs_xgr, abs_zgr, c=temp, s=10, label='Temperature')
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        
        ax.set_title(f'Temperature distribution at timestep_id: {timestep_id}')
        
        if equal_axis:
            ax.set_aspect('equal')
            
        fig.colorbar(scatter, ax=ax, cax=ax, label='Temperature [K]')
        
        ax.legend()
        
        return fig, ax 
    
    def plot_thaw_depth_transect(self, timestep_id, ax=None, figsize=(15,5), equal_axis=False):
        """Plots a thaw depth transect at some time instance(s).

        Args:
            timestep_id (int OR list OR str): timestep_id(s) to be plotted on the axis. If 'all', all timesteps with written output from the simulation will be plotted.
            ax (Axis, optional): Axis to plot the transects on. Default None, new axis is created.
            figsize (tuple, optional): Figure size of the new axis. Defaults to (15,5).
            equal_axis (boolean, optional): if set to True, axis will be scaled to be equal.
        --------------
        Returns:
            ax (Axis): Axis object containing the plot.
        """
        if not ax:
            fig, ax = plt.subplots(figsize)
            
        if type(timestep_id) == int:
            
            dirpath = os.path.join(self.sim.cwd, 'results/', f'{timestep_id}/')
            
            with open(os.path.join(dirpath, 'xgr.txt')) as f:
                xgr = np.loadtxt(f)
            with open(os.path.join(dirpath, 'thaw_depth.txt')) as f:
                thaw_depth = np.loadtxt(f)
                
            ax.plot(xgr, thaw_depth, label=f'timestep_id: {timestep_id}')
        
        elif type(timestep_id) == list:
            
            for id in timestep_id:
                
                dirpath = os.path.join(self.sim.cwd, 'results/', f'{id}/')
            
                with open(os.path.join(dirpath, 'xgr.txt')) as f:
                    xgr = np.loadtxt(f)
                with open(os.path.join(dirpath, 'thaw_depth.txt')) as f:
                    thaw_depth = np.loadtxt(f)
                
            ax.plot(xgr, thaw_depth, label=f'timestep_id: {id}')
        
        elif timestep_id == 'all':
            
            dirpath = os.path.join(self.sim.cwd, 'results/')
            
            dirlist = [item for item in os.listdir(dirpath) if os.path.isdir(item)]
            
            for id in dirlist:
                
                path = os.path.join(dirpath, id)
                
                with open(os.path.join(dirpath, 'xgr.txt')) as f:
                    xgr = np.loadtxt(f)
                with open(os.path.join(dirpath, 'thaw_depth.txt')) as f:
                    thaw_depth = np.loadtxt(f)
                
                ax.plot(xgr, thaw_depth, label=f'timestep_id: {id}')
            
        else:
            raise TypeError("'timestep_id' should be an integer, list of integers, or 'all'.")
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('thaw_depth [m]')
        
        ax.set_title('Thaw depth transect')
        
        if equal_axis:
            ax.set_aspect('equal')
        
        ax.legend()
        
        return ax
    
    def plot_thaw_depth_with_bathymetry(self, timestep_id, ax=None, figsize=(15, 5), equal_axis=False):
        """Plots a thaw depth interface transect at some time instance, with the corresponding bathymetry.

        Args:
            timestep_id (int): timestep_id to be plotted on the axis.
            ax (Axis, optional): Axis to plot the transects on. Default None, new axis is created.
            figsize (tuple, optional): Figure size of the new axis. Defaults to (15,5).
            equal_axis (boolean, optional): if set to True, axis will be scaled to be equal.
        --------------
        Returns:
            ax (Axis): Axis object containing the plot.
        """
        if not ax:
            fig, ax = plt.subplots(figsize)
            
        dirpath = os.path.join(self.sim.cwd, 'results/', f'{timestep_id}/')
        
        with open(os.path.join(dirpath, 'xgr.txt')) as f:
            xgr = np.loadtxt(f)
        with open(os.path.join(dirpath, 'zgr.txt')) as f:
            zgr = np.loadtxt(f)
        with open(os.path.join(dirpath, 'thaw_depth.txt')) as f:
            thaw_depth = np.loadtxt(f)
        
        ax.plot(xgr, zgr, color='C0', label=f'Bathymetry')
        ax.scatter(xgr, zgr, color='k', s=10, label='Bathymetry grid points')
        
        ax.plot(xgr, zgr - thaw_depth, color='r', label=f'Thaw depth interface')
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        
        ax.set_title(f'Thaw depth transect at timestep_id: {timestep_id}')
        
        if equal_axis:
            ax.set_aspect('equal')
        
        ax.legend()
        
        return ax
    
    def plot_shoreline_position(self, ax=None, figsize=(15,5)):
        """Plots the shoreline position for the entire temporal simulation domain.

        Args:
            ax (Axis, optional): Axis to plot the transects on. Default None, new axis is created.
            figsize (tuple, optional): Figure size of the new axis. Defaults to (15,5).
        --------------
        Returns:
            ax (Axis): Axis object containing the plot.
        """
        if not ax:
            fig, ax = plt.subplots(figsize)
            
        dirpath = os.path.join(self.sim.cwd, 'results/')
            
        dirlist = [item for item in os.listdir(dirpath) if os.path.isdir(item)]
        
        dates = []
        shoreline_positions = []
        
        for id in dirlist:
            
            path = os.path.join(dirpath, id)
        
            with open(os.path.join(path, 'xgr.txt')) as f:
                xgr = np.loadtxt(f)
            with open(os.path.join(path, 'zgr.txt')) as f:
                zgr = np.loadtxt(f)

            shoreline_position = calculate_shoreline_position(xgr, zgr, cross_value=self.sim.config.thermal.wl_switch)
            
            dates.append(self.sim.timestamps[int(id)])
            shoreline_positions.append(shoreline_position)
            
        ax.plot(dates, shoreline_positions, color='C0', label='Shoreline position')
        ax.scatter(dates, shoreline_positions, color='r', s=10, label='Shoreline points')
            
        ax.set_xlabel('Date')
        ax.set_ylabel('shoreline_position (relative to baseline) [m]')
        
        ax.set_title(f'Shoreline position')
        
        ax.legend()
        
        return ax
            
            
    def plot_bluff_edge_position(self, ax=None, figsize=(15,5)):
        """Plots the bluff edge position for the entire temporal simulation domain.

        Args:
            ax (Axis, optional): Axis to plot the transects on. Default None, new axis is created.
            figsize (tuple, optional): Figure size of the new axis. Defaults to (15,5).
        --------------
        Returns:
            ax (Axis): Axis object containing the plot.
        """
        if not ax:
            fig, ax = plt.subplots(figsize)
            
        dirpath = os.path.join(self.sim.cwd, 'results/')
            
        dirlist = [item for item in os.listdir(dirpath) if os.path.isdir(item)]
        
        dates = []
        bluff_edge_positions = []
        
        for id in dirlist:
            
            path = os.path.join(dirpath, id)
        
            with open(os.path.join(path, 'xgr.txt')) as f:
                xgr = np.loadtxt(f)
            with open(os.path.join(path, 'zgr.txt')) as f:
                zgr = np.loadtxt(f)

            bluff_edge_position, bluff_toe_position = calculate_bluff_edge_toe_position(xgr, zgr)
            
            dates.append(self.sim.timestamps[int(id)])
            bluff_edge_positions.append(bluff_edge_position)
            
        ax.plot(dates, bluff_edge_positions, color='C0', label='Bluff edge position')
        ax.scatter(dates, bluff_edge_positions, color='r', s=10, label='Bluff edge points')
            
        ax.set_xlabel('Date')
        ax.set_ylabel('Bluff edge position (relative to baseline) [m]')
        
        ax.set_title(f'Bluff edge position')
        
        ax.legend()
        
        return ax

    def plot_bluff_edge_toe_and_shoreline_position(self, ax=None, figsize=(15,5)):
        """Plots the bluff edge, bluff toe, and shoreline position for the entire temporal simulation domain.

        Args:
            ax (Axis, optional): Axis to plot the transects on. Default None, new axis is created.
            figsize (tuple, optional): Figure size of the new axis. Defaults to (15,5).
        --------------
        Returns:
            ax (Axis): Axis object containing the plot.
        """
        if not ax:
            fig, ax = plt.subplots(figsize)
            
        dirpath = os.path.join(self.sim.cwd, 'results/')
            
        dirlist = [item for item in os.listdir(dirpath) if os.path.isdir(item)]
        
        dates = []
        
        bluff_edge_positions = []
        bluff_toe_positions = []
        shoreline_positions = []
        
        for id in dirlist:
            
            path = os.path.join(dirpath, id)
        
            with open(os.path.join(path, 'xgr.txt')) as f:
                xgr = np.loadtxt(f)
            with open(os.path.join(path, 'zgr.txt')) as f:
                zgr = np.loadtxt(f)

            bluff_edge_position, bluff_toe_position = calculate_bluff_edge_toe_position(xgr, zgr)
            shoreline_position = calculate_shoreline_position(xgr, zgr)
            
            dates.append(self.sim.timestamps[int(id)])
            
            bluff_edge_positions.append(bluff_edge_position)
            bluff_toe_positions.append(bluff_toe_position)
            shoreline_positions.append(shoreline_position)
            
        ax.plot(dates, bluff_edge_positions, color='C0', label='Bluff edge position')
        ax.scatter(dates, bluff_edge_positions, color='r', s=10, label='Bluff edge points')
        
        ax.plot(dates, bluff_toe_positions, color='C0', label='Bluff toe position')
        ax.scatter(dates, bluff_toe_positions, color='r', s=10, label='Bluff toe points')
        
        ax.plot(dates, shoreline_positions, color='C0', label='Shoreline position')
        ax.scatter(dates, shoreline_positions, color='r', s=10, label='Shoreline points')
            
        ax.set_xlabel('Date')
        ax.set_ylabel('Position (relative to baseline) [m]')
        
        ax.set_title(f'Bluff edge, bluff toe, and shoreline position')
        
        ax.legend()
        
        return ax
        