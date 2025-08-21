# lidar2d/pipeline.py
import cv2, numpy as np
from .registry import build
from .io_kitti import read_velodyne, read_calib

class Lidar2D:
    def __init__(self, proj, rast, enc, out_hw):
        self.proj = proj
        self.rast = rast
        self.enc = enc
        self.out_hw = out_hw

    def process_one(self, velo_path, calib_path):
        pts = read_velodyne(velo_path)
        calib = read_calib(calib_path)
        P = self.proj.project(pts, calib, self.out_hw)
        depth = self.rast.rasterize(P["uv"], {"z": P["z"]}, self.out_hw)
        enc = self.enc.encode(depth)
        return enc, depth
