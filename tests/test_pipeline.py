# tests/test_pipeline.py
import yaml
import numpy as np
from lidar2d.registry import build
from lidar2d.pipeline import Lidar2D

def test_shapes():
    H,W = 64,128
    proj = build("proj","range")
    rast = build("rast","nearest")
    enc  = build("enc","invdepth")
    pipe = Lidar2D(proj,rast,enc,(H,W))
    # fake points in a ring
    N=1000
    ang = np.random.rand(N)*2*np.pi
    pts = np.stack([np.cos(ang)*10, np.sin(ang)*10, np.zeros(N), np.ones(N)],1)
    calib={"Tr":np.hstack([np.eye(3), np.zeros((3,1))]), "R0":np.eye(3), "P2":np.hstack([np.eye(3), np.zeros((3,1))])}
    P = proj.project(pts, calib, (H,W))
    depth = rast.rasterize(P["uv"], {"z":P["z"]}, (H,W))
    out = enc.encode(depth)
    assert out.shape==(H,W)
