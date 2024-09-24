import os
from pathlib import Path
import sys

from IPython import get_ipython

from moviepy.editor import ImageSequenceClip
import numpy as np


def main(runid):
    
    p_drive = Path(f"p:/11210070-usgscoop-202324-arcticxb/")
    
    # get results folder
    results_folder = os.path.join(p_drive, "results/")
    
    # get folder names in results folder
    folder_names = [f for f in os.listdir(results_folder)]

    # find folder name with runid, and set working directory to that folder
    for folder in folder_names:
        
        folder_string = str(folder)
        
        if runid in folder_string:
            
            os.chdir(os.path.join(results_folder, folder))
            
            break
    
    # get list of files
    files = [f for f in os.listdir(os.path.join(results_folder, folder)) if os.path.isfile(f)]
    
    # set fps
    fps = 6

    # create video clip
    clip = ImageSequenceClip(files, fps=fps)
    
    # set animation folder
    animation_folder = os.path.join(p_drive, "animations/")
    
    # create animation folder
    if not os.path.exists(animation_folder):
        os.mkdir(animation_folder)
    
    # define output path
    output_video_path = os.path.join(animation_folder, str(folder) + ".mp4")
    
    # write video
    clip.write_videofile(output_video_path, codec="libx264")
    
    print(f"MP4 animation saved as {output_video_path}")
    

if __name__=="__main__":
    
    # reduce ipython cache size to free up memory
    ipython = get_ipython()
    if ipython:
        ipython.Completer.cache_size = 5
            
    # set the 'runid' to the model run that you would like to perform
    runid = str(np.array(sys.argv)[1])
    
    
    main(runid)