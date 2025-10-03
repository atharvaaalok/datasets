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


def compute_lift_drag(X, Cp, Cfx = None, Cfy = None):
    if Cfx is None:
        Cfx = Cp * 0
    if Cfy is None:
        Cfy = Cp * 0
    
    # Segment vectors
    dX = X[1:] - X[:-1]
    dx = dX[:, 0]
    dy = dX[:, 1]

    # Segment lengths
    lengths = np.sqrt(dx**2 + dy**2 + 1e-12)

    # Outward normals: rotate 90 deg clockwise (dy, -dx) / length
    nx = dy / lengths
    ny = -dx / lengths

    nx_times_length = dy
    ny_times_length = -dx

    # Average pressure per segment: shape (N-1,)
    Cp_avg = 0.5 * (Cp[1:] + Cp[:-1])
    Cfx_avg = 0.5 * (Cfx[1:] + Cfx[:-1])
    Cfy_avg = 0.5 * (Cfy[1:] + Cfy[:-1])

    # Pressure force per segment: -CP_avg * normal * length
    fx = -Cp_avg * nx_times_length + Cfx_avg * lengths
    fy = -Cp_avg * ny_times_length + Cfy_avg * lengths

    # Total forces
    CD = fx.sum()
    CL = fy.sum()

    return CL, CD