# lidar2d/projections/base.py
class Projection:
    def project(self, points_xyz_i, calib, out_hw):
        """
        points_xyz_i: (N,4) x,y,z,intensity in LiDAR frame
        calib: dict with needed matrices (e.g., Tr_velo_to_cam, R0_rect, P2)
        out_hw: (H,W)
        Returns: dict with keys:
          'uv': (M,2) integer pixel coords in image plane (0..W-1, 0..H-1)
          'z':  (M,) depth (meters or chosen metric)
          'i':  (M,) intensity (optional)
        """
        raise NotImplementedError
