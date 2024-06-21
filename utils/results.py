import os
from pathlib import Path
import yaml

import numpy as np
import xarray as xr

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
        
        self.get_bluff_toes_and_shorelines()
        
        return None
                
    def get_bluff_toes_and_shorelines(self):
        
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
   
    
    
    
    