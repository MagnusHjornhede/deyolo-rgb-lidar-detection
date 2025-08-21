# lidar2d/projections/bev.py
import numpy as np
from .base import Projection
from ..registry import register

@register("proj", "bev")
class BEV(Projection):
    def __init__(self, x_range=(-30, 70), y_range=(-40, 40)):
        self.xr, self.yr = x_range, y_range

    def project(self, points_xyz_i, calib, out_hw):
        H, W = out_hw
        x, y, z, i = points_xyz_i.T
        m = (x >= self.xr[0]) & (x <= self.xr[1]) & (y >= self.yr[0]) & (y <= self.yr[1])
        x, y, z, i = x[m], y[m], z[m], i[m]
        # map x->row (H), y->col (W)
        u = ((y - self.yr[0]) / (self.yr[1] - self.yr[0]) * (W - 1)).astype(np.int32)
        v = ((self.xr[1] - x) / (self.xr[1] - self.xr[0]) * (H - 1)).astype(np.int32)
        return {"uv": np.stack([u, v], 1), "z": z, "i": i}
