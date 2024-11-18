import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

def write_dat_file(file_path, boundary_points, title):
    with open(file_path, 'w') as file:
        # Write header information
        file.write(f'TITLE = "{title}"\n')
        # Updated VARIABLES line to include "u", "v", and "p"
        file.write('VARIABLES = "x"\n"y"\n"u_mean"\n"v_mean"\n"w_mean"\n"p_mean"\n'
                   '"dissipation_mean"\n"vorticity_mean"\n"uu"\n"vv"\n"ww"\n"uv"\n"uw"\n"vw"\n"pp"\n')
        file.write(f'ZONE T="{title}"\nI={len(boundary_points)} J=1\nDATAPACKING=POINT\n')
        file.write('DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE '
                   'SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE)\n')

        # Write x, y, u, v, p columns of the points
        # Copy the columns directly without using fmt
        np.savetxt(file, boundary_points, delimiter=' ', header='', comments='', fmt='%e')

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

def write_dat_file(file_path, boundary_points, title):
    with open(file_path, 'w') as file:
        # Write header information
        file.write(f'TITLE = "{title}"\n')
        # Updated VARIABLES line to include "u", "v", and "p"
        file.write('VARIABLES = "x"\n"y"\n"u_mean"\n"v_mean"\n"w_mean"\n"p_mean"\n'
                   '"dissipation_mean"\n"vorticity_mean"\n"uu"\n"vv"\n"ww"\n"uv"\n"uw"\n"vw"\n"pp"\n')
        file.write(f'ZONE T="{title}"\nI={len(boundary_points)} J=1\nDATAPACKING=POINT\n')
        file.write('DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE '
                   'SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE)\n')

        # Write x, y, u, v, p columns of the points
        # Copy the columns directly without using fmt
        np.savetxt(file, boundary_points, delimiter=' ', header='', comments='', fmt='%e')

def plot_boundary_points(file_path):
    # Read data from the file skipping the first 20 rows
    data = np.loadtxt(file_path, skiprows=20)

    # Extract x, y, u, v, p coordinates
    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]
    v = data[:, 3]
    p = data[:, 5]

    # Find boundary points based on the rules
    right_boundary = data[(data[:, 0] == np.min(data[:, 0])) & (data[:, 2] != 0) & (data[:, 3] != 0) & (data[:, 5] != 0)]
    print('right boundary', right_boundary.shape)
    left_boundary = data[(data[:, 0] == np.max(data[:, 0])) & (data[:, 2] != 0) & (data[:, 3] != 0) & (data[:, 5] != 0)]
    #print('left boundary', left_boundary.shape)
    
    # Identify bottom boundary points
    bottom_boundary = np.zeros((len(data), 7))  # Updated to include 7 columns
    unique_x_values = np.unique(data[:, 0])
    print('unique_x_values', unique_x_values)
    
    for i, x_value in enumerate(unique_x_values[::-1]):  # Loop in reverse order (from bottom to top)
        col_data = data[data[:, 0] == x_value]
        col_data = col_data[col_data[:, 2] == 0]  # Zero u_mean
        col_data = col_data[col_data[:, 3] == 0]  # Zero v_mean
        col_data = col_data[col_data[:, 5] == 0]  # Zero p_mean
        if len(col_data) > 0:
            bottom_boundary[i, :] = col_data[np.argmax(col_data[:, 1])][:7]  # Updated to include 7 columns

    # Identify top boundary points
    top_boundary = np.zeros((len(unique_x_values), 7))  # Updated to include 7 columns

    for i, x_value in enumerate(unique_x_values):
        col_data = data[data[:, 0] == x_value]
        if len(col_data) > 0:
            top_boundary[i, :] = col_data[np.argmax(col_data[:, 1])][:7]  # Updated to include 7 columns

    # Identify nodal points below the bottom boundary
    nodal_points_below_bottom_boundary = []
    for x_value in unique_x_values:
        # Find the node in the bottom boundary for this column
        bottom_node = bottom_boundary[bottom_boundary[:, 0] == x_value][0]
        bottom_node_y = bottom_node[1]
        
        # Find all points below the bottom node in this column
        col_data = data[data[:, 0] == x_value]
        nodal_points_below_bottom_boundary.extend(col_data[col_data[:, 1] < bottom_node_y])

    nodal_points_below_bottom_boundary = np.array(nodal_points_below_bottom_boundary)
    print('nodal points below bottom boundary', nodal_points_below_bottom_boundary.shape)

    # Exclude zero values from bottom and top boundary arrays
    bottom_boundary = bottom_boundary[~np.all(bottom_boundary == 0, axis=1)]
    top_boundary = top_boundary[~np.all(top_boundary == 0, axis=1)]

    # Plot the scatter plot with all boundary points
    plt.scatter(x, y, s=1, label='Data Points')
    plt.scatter(right_boundary[:, 0], right_boundary[:, 1], color='red', label='Right Boundary', marker='x')
    plt.scatter(left_boundary[:, 0], left_boundary[:, 1], color='blue', label='Left Boundary', marker='x')
    plt.scatter(bottom_boundary[:, 0], bottom_boundary[:, 1], color='green', label='Bottom Boundary', marker='x')
    plt.scatter(top_boundary[:, 0], top_boundary[:, 1], color='orange', label='Top Boundary', marker='x')
    plt.scatter(nodal_points_below_bottom_boundary[:, 0], nodal_points_below_bottom_boundary[:, 1], color='purple',
                label='Nodal Points Below Bottom Boundary', marker='x')
    plt.title('Scatter Plot with All Boundary Points')
    plt.xlabel('X Nodal Positions')
    plt.ylabel('Y Nodal Positions')
    plt.legend()
    plt.show()

    # Write inlet.dat file
    write_dat_file("inlet.dat", right_boundary, "Inlet")

    # Write wall.dat file
    wall_boundary = np.concatenate((bottom_boundary, top_boundary))
    write_dat_file("wall.dat", wall_boundary, "Wall")
    
    # Write outlet.dat file
    write_dat_file("outlet.dat", left_boundary, "Outlet")
    
    # write mesh.dat file of all x y nodes from DNS_29_Periodic_Hills/alph05-4071-3036.dat minus nodal_points_below_bottom_boundary
    all_points = np.loadtxt(file_path, skiprows=20)
    # Create an array of tuples for easier comparison
    all_points_tuples = [tuple(row[:2]) for row in all_points]
    nodal_points_tuples = set(tuple(row[:2]) for row in nodal_points_below_bottom_boundary)
    # Filter out points that are in nodal_points_below_bottom_boundary
    filtered_points = np.array([point for point in all_points if tuple(point[:2]) not in nodal_points_tuples])
    write_dat_file("mesh.dat", filtered_points, "Mesh")

# Example usage:
file_path = "DNS_29_Periodic_Hills/alph05-4071-3036.dat"
plot_boundary_points(file_path)
