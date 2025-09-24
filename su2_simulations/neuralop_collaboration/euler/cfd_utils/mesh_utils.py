import numpy as np


def angles_between_panels(X):
    # Find panel vectors
    v = X[1:, :] - X[:-1, :]
    # Restack first panel at the end to also compute the angle between the first and last panel
    v = np.vstack([v, v[0]])

    # Find angle between consecutive panel vectors using dot products
    v1 = v[:-1]
    v2 = v[1:]
    dot_prod = np.sum(v1 * v2, axis = 1)
    norm_prod = np.linalg.norm(v1, axis = 1) * np.linalg.norm(v2, axis = 1)
    angles = np.degrees(np.arccos(dot_prod / norm_prod))

    return angles


def get_fan_points(X, threshold_angle):
    angles = angles_between_panels(X)

    # Get the points corresponding to angles greater than specified threshold
    idx = np.where(angles > threshold_angle)[0]
    N = X.shape[0]
    # Add 1 to the indices as the angle index 0 corresponds to the angle at point index 1
    fan_points_idx = idx + 1

    fan_points_coordinates_list = [tuple(row) for row in X[fan_points_idx].tolist()]

    return fan_points_coordinates_list