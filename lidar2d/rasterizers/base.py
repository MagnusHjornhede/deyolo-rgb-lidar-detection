# lidar2d/rasterizers/base.py
class Rasterizer:
    def rasterize(self, uv, values, out_hw):
        """
        uv: (M,2) integer pixels
        values: dict of arrays with same length M (e.g. {'z':..., 'i':...})
        out_hw: (H,W)
        Returns: dense 2D map as float32 (H,W)
        """
        raise NotImplementedError
