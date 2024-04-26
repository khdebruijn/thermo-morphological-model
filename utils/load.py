import yaml

def read_config(config_file):
    '''
    Creates configuration variables from file
    ------
    config_file: .yaml file
        file containing dictionary with dataset creation information
    ''' 
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        
    class AttrDict(dict):
        """
        This class is used to make it easier to work with dictionaries and allows 
        values to be called similar to attributes
        """
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
            
    config = AttrDict(cfg)
        
    for key in cfg:
        config[key] = AttrDict(cfg[key])
    
    return config
