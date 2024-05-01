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

def get_A_matrix(n):
    """ This function is used to get the 'A' matrix, which is used to make the numerical scheme faster. It is based on second order central differences for internal points.
        Att the border points, the grid is extended with an identical point (i.e. mirrored), in order to calculate the second derivative.

    Args:
        n (int): number of points in the grid
    """
    # Initialize an empty square matrix
    matrix = np.zeros((n, n))

    # Fill the main diagonal with -2
    np.fill_diagonal(matrix, -2)

    # Fill the sub-diagonal with 1
    np.fill_diagonal(matrix[1:], 1)

    # Fill the super-diagonal with 1
    np.fill_diagonal(matrix[:, 1:], 1)
    
    matrix[0,0] += 1
    matrix[-1,-1] += 1

    return matrix

def count_nonzero_until_zero(matrix):
    """Returns the number of grid points with a nonzero input, counted for each row from the lowest index until the first zero input.

    Args:
        matrix (np.array): a matrix

    Returns:
        result: number of grid points for each row before a zero input.
    """
    return np.argmax(matrix == 0, axis=1)