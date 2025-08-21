# lidar2d/encoders/invdepth.py
import numpy as np
from .base import Encoder
from ..registry import register

@register("enc", "invdepth")
class InvDepth(Encoder):
    def __init__(self, eps=1e-3, scale=80.0):
        self.eps, self.scale = eps, scale
    def encode(self, depth_map):
        inv = 1.0 / (depth_map + self.eps)
        inv = np.clip(inv / (1.0/self.scale), 0, 1.0)
        return (inv*255).astype(np.uint8)
