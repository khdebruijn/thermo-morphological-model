import os
from pathlib import Path
import shutil

import numpy as np

dirname = Path('./val10_xb_bed_temperature_thawdepth_heatflux/')

fnames = os.listdir(dirname)

sorted_fnames = np.sort(np.int64([fname[:-4] for fname in fnames]))

i = 1

for fname in sorted_fnames:
    
    fpath = os.path.join(dirname, f'{fname}' + '.png')
    savepath = os.path.join(Path('./copied_files/'), f'{i}.png')
    
    shutil.copy(fpath, savepath)
        
    i += 1
    
    
# after this, run ffmpeg "-framerate 4 -i %d.png video.mp4" in the prompt