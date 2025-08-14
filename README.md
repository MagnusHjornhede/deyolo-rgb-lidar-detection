# DEYOLO RGB + LiDAR Object Detection

This repository contains code and utilities for training and evaluating **DEYOLO** with **4-channel early fusion** (**RGB + projected LiDAR depth/reflectance**) for autonomous driving datasets. Currently, the main target dataset is **KITTI**. Support for other LiDAR-camera datasets (e.g., PandaSet) is planned.

## Features
- 4-Channel Input: RGB + LiDAR-projected depth/reflectance map
- Single-Sweep Projection: Forward-only LiDAR points from one sweep, aligned to the camera frame
- RGB-Only Baseline: Replace LiDAR channel with zeros to measure fusion gains
- DEYOLO Backbone adapted for LiDAR data
- Dataset Tools: KITTI format checking, splits, and LiDAR projection
- Coverage Metrics: % of image pixels with valid LiDAR returns

## Installation (venv, no conda)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt