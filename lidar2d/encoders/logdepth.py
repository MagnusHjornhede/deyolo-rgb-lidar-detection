# lidar2d/encoders/logdepth.py
import numpy as np
from .base import Encoder
from ..registry import register
@register("enc", "logdepth")
class LogDepth(Encoder):
    def __init__(self, alpha=5.0):
        self.alpha = alpha
    def encode(self, depth_map):
        x = np.log1p(self.alpha*depth_map)
        x /= x.max() + 1e-6
        return (x*255).astype(np.uint8)
