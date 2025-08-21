# lidar2d/rasterizers/bilinear.py
import numpy as np
from .base import Rasterizer
from ..registry import register

@register("rast", "bilinear")
class Bilinear(Rasterizer):
    def rasterize(self, uv, values, out_hw):
        # Lightweight splat to 4 neighbors with 1/d weighting
        H, W = out_hw
        z = values["z"].astype(np.float32)
        img = np.zeros((H, W), dtype=np.float32)
        wsum = np.zeros((H, W), dtype=np.float32)
        u = uv[:,0].astype(np.float32)
        v = uv[:,1].astype(np.float32)
        u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
        du, dv = u - u0, v - v0
        for offu, wu in [(0,1-du),(1,du)]:
            for offv, wv in [(0,1-dv),(1,dv)]:
                uu = np.clip(u0+offu, 0, W-1)
                vv = np.clip(v0+offv, 0, H-1)
                w = (wu*wv)
                np.add.at(img, (vv,uu), z*w)
                np.add.at(wsum,(vv,uu), w)
        m = wsum>0
        img[m] /= wsum[m]
        return img
