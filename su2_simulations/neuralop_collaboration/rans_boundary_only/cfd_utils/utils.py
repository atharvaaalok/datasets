import numpy as np
import matplotlib.pyplot as plt


def compute_area(X):
    # Shoelace formula - works for any simple polygon (piecewise-linear simple closed curve)
    area = np.sum(X[:-1, 0] * X[1:, 1] - X[1:, 0] * X[:-1, 1])

    return area.item()


def compute_normals(X):
    # Panel vectors - these are tangents between consecutive points on the shape
    dX = X[1:] - X[:-1]
    dx = dX[:, 0]
    dy = dX[:, 1]

    # Panel lengths
    lengths = np.sqrt(dx**2 + dy**2)

    # Get unit tangent vectors for the panels
    tx = dx / lengths
    ty = dy / lengths

    # Store into a matrix and
    T = np.stack([tx, ty], axis = 1)
    T = np.vstack([T[-1], T, T[0]])
    
    # Use average of the two surrounding tangents to approximate tangent at coordinate point
    T = (T[1:] + T[:-1]) / 2
    
    # Outward normals - rotate tangents 90 degree clockwise (dy, -dx) / length
    # This assumes that the points go around the curve in an counterclockwise fashion
    N = np.stack([T[:, 1], -T[:, 0]], axis = 1)
    N = N / np.linalg.norm(N, axis = 1, keepdims = True)

    # Signed area test - flip normals if points go around the curve in a clockwise fashion
    area = compute_area(X)
    if area < 0:
        N = -N

    return N