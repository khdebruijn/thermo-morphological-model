import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_time_vector(datasets):
    '''Returns array with correct temporal stamps'''

    time_stop = datasets.t.shape[-1]
    temporal_res = datasets[0].temporal_res

    time_vector = np.linspace(0, (time_stop*temporal_res), time_stop)

    return time_vector

def plot_line_with_deviation(time_vector, variable, ax=None, **plt_kwargs):
    """
    >>>>>FROM<<<<<

    @Article{hess-27-4227-2023,
    AUTHOR = {Bentivoglio, R. and Isufi, E. and Jonkman, S. N. and Taormina, R.},
    TITLE = {Rapid spatio-temporal flood modelling via hydraulics-based graph neural networks},
    JOURNAL = {Hydrology and Earth System Sciences},
    VOLUME = {27},
    YEAR = {2023},
    NUMBER = {23},
    PAGES = {4227--4246},
    URL = {https://hess.copernicus.org/articles/27/4227/2023/},
    DOI = {10.5194/hess-27-4227-2023}
    }
    
    """
    ax = ax or plt.gca()

    df = pd.DataFrame(np.vstack((time_vector, variable))).T
    df = df.rename(columns={0: "time"})
    df = df.set_index('time')

    mean = df.mean(1)
    std = df.std(1)
    under_line = (mean - std)
    over_line = (mean + std)

    p = ax.plot(mean, linewidth=2, marker='o', **plt_kwargs)
    color = p[0].get_color()
    ax.fill_between(std.index, under_line, over_line, color=color, alpha=.3)
    return p

def interpolate_points(x, y, num_points):
    """
    Interpolate points linearly between given points (x, y).
    
    Parameters:
        x (array-like): X coordinates of the given points.
        y (array-like): Y coordinates of the given points.
        num_points (int): Number of points to interpolate between the given points.
    
    Returns:
        np.ndarray: Array of interpolated points.
    """
    # Generate interpolated points using linspace
    interpolated_x = np.linspace(x[0], x[-1], num_points)
    interpolated_y = np.interp(interpolated_x, x, y)
    
    # Combine provided points and interpolated points
    all_points = np.column_stack((interpolated_x, interpolated_y))
    
    return all_points