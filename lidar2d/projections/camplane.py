# lidar2d/projections/camplane.py
import numpy as np
from .base import Projection
from ..registry import register

@register("proj", "camplane")
class CamPlane(Projection):
    def project(self, points_xyz_i, calib, out_hw):
        H, W = out_hw
        X = points_xyz_i[:, :3].T  # (3,N)
        I = points_xyz_i[:, 3]
        # augment to (4,N)
        ones = np.ones((1, X.shape[1]), dtype=X.dtype)
        X4 = np.vstack([X, ones])
        # LiDAR->Cam
        Tr = calib["Tr"]  # (3,4)
        R0 = calib["R0"]  # (3,3)
        P2 = calib["P2"]  # (3,4)
        X_cam = R0 @ (Tr @ X4)  # (3,N)
        Z = X_cam[2] + 1e-6
        valid = Z > 0.1
        X_cam = X_cam[:, valid]
        I = I[valid]
        Z = Z[valid]
        X4c = np.vstack([X_cam, np.ones((1, X_cam.shape[1]))])
        uvw = P2 @ X4c  # (3,N)
        u = (uvw[0] / uvw[2]).round().astype(np.int32)
        v = (uvw[1] / uvw[2]).round().astype(np.int32)
        m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return {"uv": np.stack([u[m], v[m]], 1), "z": Z[m], "i": I[m]}
