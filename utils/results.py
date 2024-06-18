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
    
    def __init__(self, config_path):
        
        self.result_config = self.read_config(config_path)
        
        self.runids = self.result_config.result_params.run_ids
        self.result_dir = Path(self.result_config.result_params.result_dir)
        
        timestep_output_ids_path = os.path.join(self.result_dir, self.runids[0] + "/")
        dir_list = [item for item in os.listdir(timestep_output_ids_path) if os.path.isdir(os.path.join(timestep_output_ids_path, item))]
        self.timestep_output_ids = np.sort(np.int32(np.array(dir_list)))

        var_list_path = os.path.join(timestep_output_ids_path, str(self.timestep_output_ids[0]) + "/")
        self.var_list = np.array([item[:-4] for item in os.listdir(var_list_path)])        
        
    def read_config(self, config_fpath):
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
                    
        cwd = os.getcwd()
                    
        with open(os.path.join(cwd, config_fpath)) as f:
            cfg = yaml.safe_load(f)
            
        self.config = AttrDict(cfg)
                
        for key in cfg:
            self.config[key] = AttrDict(cfg[key])
               
        return self.config
    
    def collect_results(self):
        
        self.all_data = {}
        
        for var in self.var_list:
            var_data = {}
        
            for runid in self.runids:
                
                run_data = []
                
                for id in self.timestep_output_ids:
                    
                    fpath = os.path.join(self.result_dir, runid + "/", str(id) + "/", var + ".txt")
            
                    with open(fpath) as f:
                        
                        array = np.loadtxt(f)
                        run_data.append(array)

                var_data[runid]  = np.array(run_data)  # if multidimensional, first dimension is time, second dimension is space
                
            # save array for each variable in total dictionary (dims=(runid, time, ...spatial))
            self.all_data[var] = var_data
            
        return None
    
    def create_ds(self):
        
        print(self.all_data)
        
        # self.ds = xr.Dataset()
        # self.ds = self.ds.assign_coords({
        #     "runid": self.runids,
        #     "time": self.timestep_output_ids
        # })
                    
        # for varname in self.all_data:
                        
        #     if len(self.all_data[varname].shape) > 2:
                
        #         array = self.all_data[varname]
                
        #         data = np.empty((len(self.runids), len(self.timestep_output_ids)), dtype=object)
                
        #         for i in range(array.shape[0]):
        #             for j in range(array.shape[1]):
        #                 data[i,j] = list(array[i,j])
            
        #     else:
                
        #         data = self.all_data[varname]
                
        #     self.ds[varname] = (['runid', 'time'], data)
            
        # print(self.ds)
                
        return None
    
    def write_netcdf(self):
        
        # output_fname = self.result_config.result_params.output_file_name
        # output_fpath = self.result_config.result_params.output_file_path
        
        # if output_fpath == "None":
        #     output_fpath = self.result_dir
        
        # save_path = os.path.join(output_fpath, output_fname)
        
        # print(save_path)
        
        # self.ds.to_netcdf(save_path)
        
        return None
    
    