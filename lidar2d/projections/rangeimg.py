# lidar2d/projections/rangeimg.py
import numpy as np
from .base import Projection
from ..registry import register

@register("proj", "range")
class RangeImg(Projection):
    def __init__(self, v_fov=(-24.9, 2.0)):  # Velodyne HDL32-like default
        self.vmin, self.vmax = np.radians(v_fov)

    def project(self, points_xyz_i, calib, out_hw):
        H, W = out_hw
        x, y, z, i = points_xyz_i.T
        r = np.sqrt(x*x + y*y + z*z) + 1e-6
        az = np.arctan2(y, x)            # -pi..pi
        el = np.arcsin(z / r)            # vertical angle
        u = ((az + np.pi) / (2*np.pi) * (W - 1)).astype(np.int32)
        el = np.clip(el, self.vmin, self.vmax)
        v = ((el - self.vmin) / (self.vmax - self.vmin) * (H - 1)).astype(np.int32)
        return {"uv": np.stack([u, v], 1), "z": r, "i": i}
