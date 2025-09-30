from pathlib import Path

import numpy as np


def load_airfoil(airfoil_name, category = 'airfoils'):
    # Get the path of the airfoil coordinate directory
    airfoil_coordinate_file_dir = Path(__file__).resolve().parent / category

    # Get the geometry
    if category == 'airfoils':
        file_extension = '.dat'
        np_loader = np.loadtxt
    elif category == 'anomalous_airfoils':
        file_extension = '.npy'
        np_loader = np.load
    
    airfoil_filename = airfoil_name + file_extension
    airfoil_coordinate_file_path = airfoil_coordinate_file_dir / airfoil_filename
    X = np_loader(airfoil_coordinate_file_path)
    
    return X