# Run this script to check that the simulation pipeline works
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cfd_utils.su2_simulation_func import su2_simulation_func


# Get the geometry
airfoil_name = 'naca2410'
airfoil_coordinate_file_dir = Path('./airfoil_utils/airfoils')
airfoil_filename = airfoil_name + '.dat'
airfoil_coordinate_file_path = airfoil_coordinate_file_dir / airfoil_filename
X = np.loadtxt(airfoil_coordinate_file_path)

# # Visualize the geometry
# plt.plot(X[:, 0], X[:, 1])
# plt.axis('equal')
# plt.show()


# Define simulation properties
Mach_num = 0.25
mesh_factor_at_airfoil = 5.0
mesh_size_at_farfield = 5.0
farfield_factor = 100
fan_count = 20
type_of_simulation = 'EULER'
simulation_data_save_dir = Path(f'./dataset/{type_of_simulation}')
max_iterations = 4000
conv_residual = 1e-7
angle_of_attack = 0


# Run the simulation and save the data
simulation_info = su2_simulation_func(
    airfoil_name = airfoil_name,
    X = X,
    Mach_num = Mach_num,
    mesh_factor_at_airfoil = mesh_factor_at_airfoil,
    mesh_size_at_farfield = mesh_size_at_farfield,
    farfield_factor = farfield_factor,
    fan_count = fan_count,
    type_of_simulation = type_of_simulation,
    simulation_data_save_dir = simulation_data_save_dir,
    max_iterations = max_iterations,
    conv_residual = conv_residual,
    angle_of_attack_deg = angle_of_attack
)

if not simulation_info['converged']:
    print(f'DID NOT CONVERGE. Airfoil: {airfoil_name:{15}} AoA: {angle_of_attack:2d}')

# Print simulation info
import json
print(json.dumps(simulation_info, indent = 4))