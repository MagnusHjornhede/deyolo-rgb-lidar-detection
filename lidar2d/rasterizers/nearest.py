# lidar2d/rasterizers/nearest.py
import numpy as np
from .base import Rasterizer
from ..registry import register

@register("rast", "nearest")
class Nearest(Rasterizer):
    def rasterize(self, uv, values, out_hw):
        H, W = out_hw
        z = values["z"]
        # keep nearest depth if multiple points hit same pixel
        img = np.zeros((H, W), dtype=np.float32)
        # flatten index
        idx = uv[:,1] * W + uv[:,0]
        # choose nearest by z (smallest)
        order = np.argsort(z)  # far->near overwrite so near stays
        for k in order[::-1]:
            img.flat[idx[k]] = z[k]
        return img
