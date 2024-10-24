import os
from pathlib import Path
import shutil

import numpy as np


def main():
    
    # copy all relevant files to a local folder
    runids_to_copy = [
        f'cal_gt{i}' for i in np.arange(1, 81+1, 1)
    ]

    cwd = os.getcwd()

    for runid in runids_to_copy:
        
        fpath = os.path.join(Path("P:/11210070-usgscoop-202324-arcticxb/runs/"), runid + '/', f"{runid}_ground_temperature_timeseries.csv")
        
        save_path = os.path.join(cwd, f"{runid}_ground_temperature_timeseries.csv")
        
        shutil.copy(fpath, save_path)
        
    return None


if __name__ == "__main__":
    
    main()