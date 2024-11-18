import numpy as np
import matplotlib.pyplot as plt

# Define the function to process the data
def process_data(mesh_file):
    # Load data from mesh.dat
    data = np.loadtxt(mesh_file, skiprows=20)

    # Extract columns
    x = data[:, 0]
    y = data[:, 1]
    u = data[:, 2]
    v = data[:, 3]
    p = data[:, 5]
    uv = data[:, 11]
    uu = data[:, 8]
    vv = data[:, 9]

    # Create pos and scal arrays
    pos = np.column_stack((x, y))
    scal = np.column_stack((u, v, p, vv, uv, uu))

    return {'pos': pos.T, 'scal': scal.T}

# Define the input and output filenames
input_mesh_file = 'mesh.dat'
output_npz_file = input_mesh_file.replace('.dat', '.npz')

# Process data
processed_data = process_data(input_mesh_file)

# Save data to .npz file
np.savez(output_npz_file, **processed_data)

print(f'Data saved to {output_npz_file}')

# Load data from saved .npz file
data = np.load(output_npz_file)

# Extract scal and pos arrays
scal = data['scal']
pos = data['pos']

# Display shapes of scal and pos arrays
print('scal', scal.shape)
print('pos', pos.shape)

# Extract individual coordinates
x = pos[:, 0]
y = pos[1]
u = scal[0]
v = scal[1]
p = scal[2]
uv = scal[4]
uu = scal[5]
vv = scal[3]

# Display shapes of individual coordinates
print('x', x.shape)
print('y', y.shape)
print('u', u.shape)
print('v', v.shape)
print('p', p.shape)
print('uv', uv.shape)
print('uu', uu.shape)
print('vv', vv.shape)
