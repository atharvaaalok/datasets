import csv
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import time

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from .mesh_func import mesh_func
from .mesh_utils import get_fan_points


def su2_adjoint_func(
    airfoil_name,
    angle_of_attack_deg,
    simulation_data_save_dir,
    type_of_simulation,
    max_iterations,
    conv_residual,
    show_errors = False
):
    # Save simulation information
    adjoint_simulation_info = {'airfoil_name': airfoil_name, 'AoA': angle_of_attack_deg}

    airfoil_simulation_dir = simulation_data_save_dir / airfoil_name / f'AoA_{str(angle_of_attack_deg)}'

    # Copy restart_flow.dat file into solution_flow.dat to be available for adjoint computations
    restart_flow_file_path = airfoil_simulation_dir / 'restart_flow.dat'
    solution_flow_file_path = airfoil_simulation_dir / 'solution_flow.dat'
    shutil.copy(restart_flow_file_path, solution_flow_file_path)
    

    # Run adjoint simulation
    simulation_start_time = time.time()

    cfg_file_path = airfoil_simulation_dir / f'su2_{type_of_simulation}.cfg'
    adjoint_cfg_file_path = airfoil_simulation_dir / f'su2_{type_of_simulation}_adjoint.cfg'
    shutil.copy(cfg_file_path, adjoint_cfg_file_path)

    # Modify appropriate contents of the adjoint configuration file
    lines = []
    with open(adjoint_cfg_file_path, 'r') as cfg_f:
        for line in cfg_f:
            if line.strip().startswith('MATH_PROBLEM='):
                line = f'MATH_PROBLEM= CONTINUOUS_ADJOINT\n'
            elif line.strip().startswith('ITER='):
                line = f'ITER= {max_iterations}\n'
            elif line.strip().startswith('CONV_RESIDUAL_MINVAL='):
                line = f'CONV_RESIDUAL_MINVAL= {math.log10(conv_residual)}\n'
            elif line.strip().startswith('CONV_FILENAME='):
                line = f'CONV_FILENAME= adjoint_history'
            lines.append(line)
    
    # Update configuration file information
    with open(adjoint_cfg_file_path, 'w') as cfg_f:
        cfg_f.writelines(lines)


    with open(os.devnull, 'w') as devnull:
        stdout = devnull if not show_errors else None
        subprocess.run(['SU2_CFD', adjoint_cfg_file_path.name], check = True,
                       cwd = adjoint_cfg_file_path.parent, stdout = stdout, stderr = subprocess.STDOUT)

    simulation_end_time = time.time()


    # Read last line from adjoint_history.csv and get the total iterations the simulation ran for
    adjoint_history_data_file_path = airfoil_simulation_dir / 'adjoint_history.csv'
    with open(adjoint_history_data_file_path, 'r') as history_f:
        for line in history_f:
            history_headers = [header.strip().strip('"') for header in line.strip().split(',')]
            break
        
        history_data = np.genfromtxt(adjoint_history_data_file_path, delimiter = ',', skip_header = 1)
        history_cols = {header: history_data[:, i] for i, header in enumerate(history_headers)}

        total_iterations = history_cols['Inner_Iter'][-1]


    adjoint_simulation_info['simulation_time'] = simulation_end_time - simulation_start_time
    adjoint_simulation_info['converged'] = True if total_iterations < (max_iterations - 1) else False
    adjoint_simulation_info['total_iterations'] = int(total_iterations)

    # Save simulation info in a json file
    adjoint_simulation_info_filename = airfoil_simulation_dir / f'adjoint_simulation_info.json'
    with open(adjoint_simulation_info_filename, 'w') as adjoint_simulation_json_file:
        json.dump(adjoint_simulation_info, adjoint_simulation_json_file, indent = 4)
    

    # Get the generated data into numpy arrays for easy loading and processing later

    # Read history data and save as .npy
    with open(adjoint_history_data_file_path, 'r') as history_f:
        for line in history_f:
            history_headers = []
            history_headers_repeated = [header.strip().strip('"') for header in line.strip().split(',')]
            history_headers_set = set()
            for header in history_headers_repeated:
                if header not in history_headers_set:
                    history_headers.append(header)
                    history_headers_set.add(header)
            break

        history_data = np.genfromtxt(adjoint_history_data_file_path, delimiter = ',', skip_header = 1)

        # Save as .npy file - table format to keep all data together
        N_history = history_data.shape[0]
        history_arr = np.zeros(N_history,
                               dtype = [(header, history_data.dtype) for header in history_headers])
        for i, header in enumerate(history_headers):
            history_arr[header] = history_data[:, i]
        np.save(airfoil_simulation_dir / 'adjoint_history.npy', history_arr)
    

    # Read adjoint surface flow data and save as .npy
    adjoint_surface_data_file_path = airfoil_simulation_dir / f'surface_adjoint.csv'
    with open(adjoint_surface_data_file_path, 'r') as surface_f:
        for line in surface_f:
            surface_headers = [header.strip().strip('"') for header in line.strip().split(',')]
            break

        surface_data = np.genfromtxt(adjoint_surface_data_file_path, delimiter = ',', skip_header = 1)

        # Save as .npy file - table format to keep all data together
        N_surface = surface_data.shape[0]
        surface_arr = np.zeros(N_surface,
                               dtype = [(header, surface_data.dtype) for header in surface_headers])
        for i, header in enumerate(surface_headers):
            surface_arr[header] = surface_data[:, i]
        np.save(airfoil_simulation_dir / 'surface_adjoint.npy', surface_arr)

    return adjoint_simulation_info


def su2_simulation_func(
    airfoil_name,
    X,
    Mach_num,
    mesh_factor_at_airfoil,
    mesh_size_at_farfield,
    farfield_factor,
    fan_count,
    type_of_simulation,
    simulation_data_save_dir,
    max_iterations,
    conv_residual,
    angle_of_attack_deg,
    template_cfg_file_path,
    cfl_adapt = 'YES',
    show_errors = False,
    save_boundary_data_only = False
):
    
    # Create directory for storing airfoil simulation data
    airfoil_simulation_dir = simulation_data_save_dir / airfoil_name / f'AoA_{str(angle_of_attack_deg)}'
    airfoil_simulation_dir.mkdir(parents = True, exist_ok = False)

    # Rotate airfoil clockwise by angle of attack
    angle_of_attack_rad = angle_of_attack_deg * (np.pi / 180)
    X = rotate_shape_clockwise(X, angle_of_attack_rad)

    # Put centroid at origin
    X = X - np.mean(X, axis = 0)

    # Save simulation information
    simulation_info = {'airfoil_name': airfoil_name, 'AoA': angle_of_attack_deg}

    # Save airfoil coordinate file in the simulation data directory for this airfoil
    np.save(airfoil_simulation_dir / (airfoil_name + '.npy'), X)


    # Create a mesh
    Mach_num = Mach_num
    airfoil_mesh_filename = str(airfoil_simulation_dir / airfoil_name)
    mesh_file_path_list = [f'{airfoil_mesh_filename}.msh', f'{airfoil_mesh_filename}.su2']
    mesh_info = create_simulation_mesh(X, Mach_num, mesh_factor_at_airfoil, mesh_size_at_farfield,
                                       farfield_factor, fan_count, mesh_file_path_list)

    # Add mesh info to simulation info and extract flow details useful for simulation
    Re = mesh_info['Re']
    simulation_info['mesh_info'] = mesh_info


    # Run simulation
    available_simulations = ['euler', 'rans']
    assert type_of_simulation in available_simulations, \
        f'{type_of_simulation} not found in available simulations: {available_simulations}'

    simulation_start_time = time.time()

    template_cfg_file_path = template_cfg_file_path
    cfg_file_path = airfoil_simulation_dir / f'su2_{type_of_simulation}.cfg'
    shutil.copy(template_cfg_file_path, cfg_file_path)

    # Modify appropriate contents of the configuration file
    lines = []
    with open(cfg_file_path, 'r') as cfg_f:
        for line in cfg_f:
            if line.strip().startswith('MACH_NUMBER='):
                line = f'MACH_NUMBER= {Mach_num}\n'
            elif line.strip().startswith('ITER='):
                line = f'ITER= {max_iterations}\n'
            elif line.strip().startswith('REYNOLDS_NUMBER='):
                line = f'REYNOLDS_NUMBER= {Re}\n'
            elif line.strip().startswith('CFL_ADAPT='):
                line = f'CFL_ADAPT= {cfl_adapt}\n'
            elif line.strip().startswith('CONV_RESIDUAL_MINVAL='):
                line = f'CONV_RESIDUAL_MINVAL= {math.log10(conv_residual)}\n'
            elif line.strip().startswith('MESH_FILENAME='):
                line = f'MESH_FILENAME= {airfoil_name}.su2\n'
            lines.append(line)
    
    # Update configuration file information
    with open(cfg_file_path, 'w') as cfg_f:
        cfg_f.writelines(lines)


    with open(os.devnull, 'w') as devnull:
        stdout = devnull if not show_errors else None
        subprocess.run(['SU2_CFD', cfg_file_path.name], check = True,
                       cwd = cfg_file_path.parent, stdout = stdout, stderr = subprocess.STDOUT)

    simulation_end_time = time.time()


    # Read last line from history.csv and get the total iterations the simulation ran for
    history_data_file_path = airfoil_simulation_dir / 'history.csv'
    with open(history_data_file_path, 'r') as history_f:
        for line in history_f:
            history_headers = [header.strip().strip('"') for header in line.strip().split(',')]
            break
        
        history_data = np.genfromtxt(history_data_file_path, delimiter = ',', skip_header = 1)
        history_cols = {header: history_data[:, i] for i, header in enumerate(history_headers)}

        total_iterations = history_cols['Inner_Iter'][-1]
    

    simulation_info['simulation_time'] = simulation_end_time - simulation_start_time
    simulation_info['converged'] = True if total_iterations < (max_iterations - 1) else False
    simulation_info['total_iterations'] = int(total_iterations)


    # Save simulation info in a json file
    simulation_info_filename = airfoil_simulation_dir / f'simulation_info.json'
    with open(simulation_info_filename, 'w') as simulation_json_file:
        json.dump(simulation_info, simulation_json_file, indent = 4)
    

    # Get the generated data into numpy arrays for easy loading and processing later

    # Read history data and save as .npy
    with open(history_data_file_path, 'r') as history_f:
        for line in history_f:
            history_headers = [header.strip().strip('"') for header in line.strip().split(',')]
            break

        history_data = np.genfromtxt(history_data_file_path, delimiter = ',', skip_header = 1)

        # Save as .npy file - table format to keep all data together
        N_history = history_data.shape[0]
        history_arr = np.zeros(N_history,
                               dtype = [(header, history_data.dtype) for header in history_headers])
        for i, header in enumerate(history_headers):
            history_arr[header] = history_data[:, i]
        np.save(airfoil_simulation_dir / 'history.npy', history_arr)
    

    # Read surface flow data and save as .npy
    surface_data_file_path = airfoil_simulation_dir / f'surface_flow.csv'
    with open(surface_data_file_path, 'r') as surface_f:
        for line in surface_f:
            surface_headers = [header.strip().strip('"') for header in line.strip().split(',')]
            break

        surface_data = np.genfromtxt(surface_data_file_path, delimiter = ',', skip_header = 1)

        # Save as .npy file - table format to keep all data together
        N_surface = surface_data.shape[0]
        surface_arr = np.zeros(N_surface,
                               dtype = [(header, surface_data.dtype) for header in surface_headers])
        for i, header in enumerate(surface_headers):
            surface_arr[header] = surface_data[:, i]
        np.save(airfoil_simulation_dir / 'surface_flow.npy', surface_arr)
    

    # Read volume flow data and save as .npy
    flow_vtk_reader = vtk.vtkXMLUnstructuredGridReader()
    flow_vtk_reader.SetFileName(airfoil_simulation_dir / 'flow.vtu')
    flow_vtk_reader.Update()
    grid = flow_vtk_reader.GetOutput()

    point_coordinates = vtk_to_numpy(grid.GetPoints().GetData())
    flow_data = flow_vtk_reader.GetOutput().GetPointData()
    point_array_names = [flow_data.GetArrayName(i) for i in range(flow_data.GetNumberOfArrays())]

    N_flow = point_coordinates.shape[0]
    dtype = point_coordinates.dtype
    column_headers = ['x', 'y', 'Density', 'Momentum_x', 'Momentum_y', 'Energy', 'Pressure',
                      'Temperature', 'Mach', 'Pressure_Coefficient', 'Velocity_x', 'Velocity_y']
    
    if type_of_simulation == 'RANS':
        RANS_column_extension = ['Nu_Tilde', 'Laminar_Viscosity', 'Skin_Friction_Coefficient_x',
                                 'Skin_Friction_Coefficient_y', 'Heat_Flux', 'Y_Plus',
                                 'Eddy_Viscosity']
        column_headers.extend(RANS_column_extension)
    
    flow_arr = np.zeros(N_flow, dtype = [(header, dtype) for header in column_headers])
    flow_arr['x'], flow_arr['y'] = point_coordinates[:, 0], point_coordinates[:, 1]
    flow_arr['Density'] = vtk_to_numpy(flow_data.GetArray('Density'))
    flow_arr['Momentum_x'] = vtk_to_numpy(flow_data.GetArray('Momentum'))[:, 0]
    flow_arr['Momentum_y'] = vtk_to_numpy(flow_data.GetArray('Momentum'))[:, 1]
    flow_arr['Energy'] = vtk_to_numpy(flow_data.GetArray('Energy'))
    flow_arr['Pressure'] = vtk_to_numpy(flow_data.GetArray('Pressure'))
    flow_arr['Temperature'] = vtk_to_numpy(flow_data.GetArray('Temperature'))
    flow_arr['Mach'] = vtk_to_numpy(flow_data.GetArray('Mach'))
    flow_arr['Pressure_Coefficient'] = vtk_to_numpy(flow_data.GetArray('Pressure_Coefficient'))
    flow_arr['Velocity_x'] = vtk_to_numpy(flow_data.GetArray('Velocity'))[:, 0]
    flow_arr['Velocity_y'] = vtk_to_numpy(flow_data.GetArray('Velocity'))[:, 1]

    if type_of_simulation == 'RANS':
        flow_arr['Nu_Tilde'] = vtk_to_numpy(flow_data.GetArray('Nu_Tilde'))
        flow_arr['Laminar_Viscosity'] = vtk_to_numpy(flow_data.GetArray('Laminar_Viscosity'))
        flow_arr['Skin_Friction_Coefficient_x'] = vtk_to_numpy(flow_data.GetArray('Skin_Friction_Coefficient'))[:, 0]
        flow_arr['Skin_Friction_Coefficient_y'] = vtk_to_numpy(flow_data.GetArray('Skin_Friction_Coefficient'))[:, 1]
        flow_arr['Heat_Flux'] = vtk_to_numpy(flow_data.GetArray('Heat_Flux'))
        flow_arr['Y_Plus'] = vtk_to_numpy(flow_data.GetArray('Y_Plus'))
        flow_arr['Eddy_Viscosity'] = vtk_to_numpy(flow_data.GetArray('Eddy_Viscosity'))

    np.save(airfoil_simulation_dir / 'flow.npy', flow_arr)


    if save_boundary_data_only:
        # Remove the volume data files to save space
        Path(airfoil_simulation_dir / 'restart_flow.dat').unlink()
        Path(airfoil_simulation_dir / 'flow.vtu').unlink()
        Path(airfoil_simulation_dir / 'flow.npy').unlink()
        Path(f'{airfoil_mesh_filename}.msh').unlink()
        Path(f'{airfoil_mesh_filename}.su2').unlink()


    return simulation_info


def rotate_shape_clockwise(X, theta):
    R_theta = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    X = X @ R_theta
    return X


def create_simulation_mesh(
    X,
    Mach_num,
    mesh_factor_at_airfoil,
    mesh_size_at_farfield,
    farfield_factor,
    fan_count,
    mesh_file_path_list
):
    # Compute minimum distance between any two consecutive points
    diffs = X[1:] - X[:-1]
    consecutive_point_dists = np.linalg.norm(diffs, axis = 1)
    consecutive_point_min_dist = consecutive_point_dists.min().item()
    consecutive_point_max_dist = consecutive_point_dists.max().item()

    mesh_info = {}

    mesh_info['consecutive_point_min_dist'] = consecutive_point_min_dist
    mesh_info['consecutive_point_max_dist'] = consecutive_point_max_dist


    # Define mesh properties
    mu = 0.000017894
    M = Mach_num
    rho = 1.225
    T = 288.15
    gamma = 1.4
    R_gas = 287.0
    ref_length = 1.0

    # Calculate quantities affecting the mesh sizing
    U = M * math.sqrt(gamma * R_gas * T)
    Re = (rho * U * ref_length) / mu

    cf = math.pow(2 * math.log10(Re) - 0.65, -2.3)
    tau_wall = (1/2) * rho * U * U * cf
    u_tau = math.sqrt(tau_wall / rho)

    y_plus = 1.0
    y_H = y_plus * (2 * mu) / (rho * u_tau)

    # Find growth ratio, number of layers and total boundary layer thickness
    if Re >= 5e5:
        delta_99 = (0.38 * ref_length) / math.pow(Re, (1/5))
    elif (Re >= 0.0) and (Re < 5e5):
        delta_99 = (4.91 * ref_length) / math.sqrt(Re)
    else:
        raise ValueError("Negative Re is Invalid!!!")

    G = 1.2
    final_layer_height = consecutive_point_max_dist / mesh_factor_at_airfoil
    N = math.ceil(1 + math.log10(final_layer_height / (y_H)) / math.log10(G))
    BL_thickness = y_H * (math.pow(G, N) - 1) / (G - 1)

    mesh_info['Re'] = Re
    mesh_info['y_H'] = y_H
    mesh_info['delta_99'] = delta_99
    mesh_info['G'] = G
    mesh_info['N'] = N
    mesh_info['BL_thickness'] = BL_thickness

    if BL_thickness < delta_99:
        raise ValueError('BL_thickness of the mesh is less than delta_99, Exiting!')


    # Create mesh
    mesh_size_at_airfoil = (1 / mesh_factor_at_airfoil) * consecutive_point_max_dist * (1.01)
    y_first_layer = y_H
    growth_ratio = G
    total_BL_thickness = BL_thickness
    mesh_size_at_airfoil = mesh_size_at_airfoil
    mesh_size_at_farfield = mesh_size_at_farfield
    farfield_factor = farfield_factor
    fan_count = fan_count
    threshold_angle = 25
    fan_points_coordinates_list = get_fan_points(X = X, threshold_angle = threshold_angle)

    
    mesh_start_time = time.time()

    mesh_details = mesh_func(
        X = X,
        y_first_layer = y_first_layer,
        growth_ratio = growth_ratio,
        total_BL_thickness = total_BL_thickness,
        mesh_size_at_airfoil = mesh_size_at_airfoil,
        mesh_size_at_farfield = mesh_size_at_farfield,
        farfield_factor = farfield_factor,
        model_name = 'airfoil',
        fan_points_coordinates_list = fan_points_coordinates_list,
        fan_count = fan_count,
        mesh_file_path_list = mesh_file_path_list
    )
    
    mesh_end_time = time.time()
    
    mesh_info['mesh_generation_time'] = mesh_end_time - mesh_start_time
    mesh_info['mesh_details'] = mesh_details

    return mesh_info