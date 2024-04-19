from utils.load import read_config
from utils.model import start_xbeach, write_xbeach_output

def main(config):

    # determine when storms occur
    # determine when to run xbeach (and write this to a file) --> during storms and every ~1 week (during ice-free season? or also during winter?)

    # generate initial condtiions files
    # generate thermal model 1D output model files

    # loop through timesteps

        # call thermal update routine
        
        # generate updated 'ne_layer' file

        # if (run xbeach at this timestep):
    
            # generate params.txt file (including: grid/bathymetry, waves input, flow, tide and surge input, water level, wind input, sediment input, 
            #                                      avalanching, vegetation input, drifters ipnut, output selection)

            # call xbeach

            # copy updated morphology to thermal module and to new output file
    
    
    return 


if __name__ == '__main__':

    # read configuration file with parameters
    cfg = read_config('config.yaml')

    main(cfg)