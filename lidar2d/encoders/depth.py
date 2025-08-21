# lidar2d/encoders/depth.py
import numpy as np
from .base import Encoder
from ..registry import register

@register("enc", "depth")
class Depth(Encoder):
    def __init__(self, dmax=80.0):
        self.dmax = dmax
    def encode(self, depth_map):
        dm = np.clip(depth_map, 0, self.dmax) / self.dmax
        return (dm*255).astype(np.uint8)
