import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

def textbox(text):
    """Function to quickly generate text boxes.

    Args:
        text (str): string of the text that should be in the textbox.
        
    Returns:
        str: String representation of the input text with a textbox around it.
    """
    row_length = len(text) + 6
    
    row0 = row_length * "%" + "\n"
    row1 = "%" + (row_length-2) * " " + "%" + "\n"
    row2 = f"%  {text}  %" + "\n"
    row3 = row1
    row4 = row_length * "%"

    total_text = row0 + row1 + row2 + row3 + row4
    
    return total_text

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
    
    matrix[0, 0] = -2
    
    new_row = np.zeros(matrix.shape[0])
    new_row[0] = 1
    
    matrix = np.vstack((new_row, matrix))

    return matrix

def count_nonzero_until_zero(matrix):
    """Returns the number of grid points with a nonzero input, counted for each row from the lowest index until the first zero input.

    Args:
        matrix (np.array): a matrix

    Returns:
        result: number of grid points for each row before a zero input (-1 if no zeros in the entire row)
    """
    
    matrix = np.column_stack((matrix, np.zeros(matrix.shape[0])))
    
    indices = np.argmax(matrix == 0, axis=1)
    
    mask = (indices == matrix.shape[1]-1)
    
    indices[mask] = -1
    
    return indices

def generate_perpendicular_grids(xgr, zgr, resolution=30, max_depth=3):
    """This function takes an xgrid and a zgrid, as well as a resolution and maximum depth, and returns a (temperature) grid perpendicular to the existing x and z-grid.
    ----------
    xgr: array
        1D array of x-values
    zgr: array
        1D array of z-values
    resolution: integer (default: 30)
        number of desired grid points in the 1D models
    max_depth: float (default: 3)
        maximum depth to be modelled
    ---------Returns----------
    Returns: x_matrix, z_matrix
    
    x_matrix: array
        2D array of x-values. Each row contains the x-values for a 1D model
    z_matrix: array
        2D array of z-values. Each row contains the z-values for a 1D model
    """
    temp_depth_grid = np.linspace(0, max_depth, resolution)
    initial_temp = np.linspace(-5, 5, resolution)

    thermal_matrix = np.zeros((xgr.shape[0], temp_depth_grid.shape[0]))

    for i in range(len(thermal_matrix)):
        thermal_matrix[i,:] = initial_temp
        
    gradient = np.gradient(zgr, xgr)

    orientation = np.arctan(gradient) - 0.5 * np.pi
        
    x_matrix = np.tile(xgr, (len(temp_depth_grid), 1)).T + np.outer(np.cos(orientation), temp_depth_grid)
    z_matrix = np.tile(zgr, (len(temp_depth_grid), 1)).T + np.outer(np.sin(orientation), temp_depth_grid)
    
    return x_matrix, z_matrix

def linear_interp_with_nearest(xgr, zgr, values, new_xgr, new_zgr):
    """This function takes an old x-grid and z-grid, and associated values (e.g., temperature), and casts them to a new grid of x and z values. 
    Linear interpolation is used for new grid points within the convex hull of the old grid, and a nearest value is assigned for values outside of the convex hull.

    Args:
        xgr (np.Array): x-grid (m x n), with every 1D model being assigned to a row
        zgr (np.Array): z-grid (m x n), with every 1D model being assigned to a row
        values (np.Array): matrix of same shape as xgr and zgr with associated valeus
        new_xgr (np.Array): new x-grid (p x q)
        new_zgr (np.Array): new z-grid (q x q)

    Returns:
        np.Array: new matrix of associated values for the new grid (shape p x q)
    """
    # Flatten the input arrays
    x_flat = xgr.flatten()
    z_flat = zgr.flatten()
    
    new_x_flat = new_xgr.flatten()
    new_z_flat = new_zgr.flatten()
    
    temperature_flat = values.flatten()

    # Create the interpolator
    interp_func = LinearNDInterpolator(list(zip(x_flat, z_flat)), temperature_flat)

    # Interpolate temperature values for new points
    new_temperature_values = interp_func(new_x_flat, new_z_flat)
    
    for i in range(len(new_temperature_values)):
        if np.isnan(new_temperature_values[i]):
            new_temperature_values[i] = temperature_flat[np.argmin(np.sqrt((x_flat - new_x_flat[i])**2 + (z_flat - new_z_flat[i])**2))]
             
    # Reshape the interpolated temperature values to match the shape of new_x_points and new_y_points
    new_temperature_values = new_temperature_values.reshape(new_xgr.shape)

    return new_temperature_values

def calculate_shoreline_position(xgr, zgr, cross_value=0):
    """Returns the x value at which the zgrid passes some horizontal line.

    Args:
        xgr (array): array containing x coordinates
        zgr (array): array containing z coordinates
        cross_value (float, optional): the value that the z grid should cross.

    Returns:
        float: the x-coordinate of the crossing.
    """
    
    zgr_adjusted = zgr - cross_value
    
    idx = np.where(np.diff(np.sign(zgr_adjusted)))[0]

    x1, x2 = xgr[idx], xgr[idx+1]
    z1, z2 = zgr_adjusted[idx], zgr_adjusted[idx+1]

    x_intersection = x1 + (0 - z1)/(z2 - z1) * (x2 - x1)
    
    return x_intersection
    
def calculate_bluff_edge_toe_position(xgr, zgr):
    """Calculates the bluff edge and toe position for a given profile. Method by Palaseanu-Lovejoy et al (2016).

    Args:
        xgr (array): array containing x coordinates
        zgr (array): array containing z coordinates

    Returns:
        tuple: tuple of length 2, with the first being the x value of the bluff edge and 
               the second being the x value of the bluff toe.
    """
    p1 = np.array((xgr[0], zgr[0]))
    p2 = np.array((xgr[-1], zgr[-1]))

    distances = np.zeros(xgr.shape)

    for i in range(len(xgr)):
        
        p3 = np.array((xgr[i], zgr[i]))
            
        distances[i] = np.cross(p2-p1, p3-p1) / np.linalg.norm(p2-p1)
        
        if zgr[i] < xgr[i] * (zgr[-1] - zgr[0]) / (xgr[-1] - xgr[0]) + zgr[0]:
            distances[i] *= -1
        
    bluff_edge_id = np.argmax(distances)
    bluff_toe_id = np.argmin(distances)
    
    return xgr[bluff_edge_id], xgr[bluff_toe_id]
    
    
# def get_time_vector(datasets):
#     '''Returns array with correct temporal stamps'''

#     time_stop = datasets.t.shape[-1]
#     temporal_res = datasets[0].temporal_res

#     time_vector = np.linspace(0, (time_stop*temporal_res), time_stop)

#     return time_vector

# def plot_line_with_deviation(time_vector, variable, ax=None, **plt_kwargs):
#     """
#     >>>>>FROM<<<<<

#     @Article{hess-27-4227-2023,
#     AUTHOR = {Bentivoglio, R. and Isufi, E. and Jonkman, S. N. and Taormina, R.},
#     TITLE = {Rapid spatio-temporal flood modelling via hydraulics-based graph neural networks},
#     JOURNAL = {Hydrology and Earth System Sciences},
#     VOLUME = {27},
#     YEAR = {2023},
#     NUMBER = {23},
#     PAGES = {4227--4246},
#     URL = {https://hess.copernicus.org/articles/27/4227/2023/},
#     DOI = {10.5194/hess-27-4227-2023}
#     }
    
#     """
#     ax = ax or plt.gca()

#     df = pd.DataFrame(np.vstack((time_vector, variable))).T
#     df = df.rename(columns={0: "time"})
#     df = df.set_index('time')

#     mean = df.mean(1)
#     std = df.std(1)
#     under_line = (mean - std)
#     over_line = (mean + std)

#     p = ax.plot(mean, linewidth=2, marker='o', **plt_kwargs)
#     color = p[0].get_color()
#     ax.fill_between(std.index, under_line, over_line, color=color, alpha=.3)
#     return p