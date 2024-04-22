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
        
    return cfg
