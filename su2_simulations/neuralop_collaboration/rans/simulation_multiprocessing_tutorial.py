from pathlib import Path
import multiprocessing

import numpy as np
import argparse

from cfd_utils.su2_simulation_func import su2_simulation_func


# Get the list of all airfoils
airfoil_list = []
with open('./airfoil_utils/airfoil_names.txt', 'r') as f:
    for airfoil_name in f:
        airfoil_list.append(airfoil_name.strip()[:-4])

# Angles of attack to run for each airfoil
angle_of_attack_list = [-5, 0, 5, 10, 15]


def gen_airfoil_data(airfoil_name, angle_of_attack):
    # Get the geometry
    airfoil_name = airfoil_name
    airfoil_coordinate_file_dir = Path('./airfoil_utils/airfoils')
    airfoil_filename = airfoil_name + '.dat'
    airfoil_coordinate_file_path = airfoil_coordinate_file_dir / airfoil_filename
    X = np.loadtxt(airfoil_coordinate_file_path)

    # Define simulation properties
    Mach_num = 0.25
    mesh_factor_at_airfoil = 5.0
    mesh_size_at_farfield = 5.0
    farfield_factor = 100
    fan_count = 20
    type_of_simulation = 'RANS'
    simulation_data_save_dir = Path(f'./dataset/{type_of_simulation}')
    max_iterations = 4000
    conv_residual = 1e-7
    angle_of_attack_deg = angle_of_attack

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--airfoil_idx', type = int)
    args = parser.parse_args()
    airfoil_idx = args.airfoil_idx

    # Create processes for multiprocessing
    processes = []

    airfoil_name = airfoil_list[airfoil_idx]

    for angle_of_attack in angle_of_attack_list:
        p = multiprocessing.Process(target = gen_airfoil_data,
                                    args = [airfoil_name, angle_of_attack])
        p.start()
        processes.append(p)
    
    for process in processes:
        process.join()