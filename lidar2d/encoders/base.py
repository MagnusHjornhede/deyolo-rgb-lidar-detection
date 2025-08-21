# lidar2d/encoders/base.py
class Encoder:
    def encode(self, depth_map):
        """
        depth_map: float32 (H,W) with 0 for missing
        Returns: float32 or uint8 (H,W) single-channel, ready to save/stack
        """
        raise NotImplementedError
