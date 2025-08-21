# Experiments
- Variant naming: kitti_{proj}-{rast}-{enc}-{denoise}_{pack}
  e.g., kitti_cam-nearest-inv-guided_invD-inv-mask
- Change ONE axis per experiment.
- Always do a 1-epoch SMOKE on a fixed val subset first.
- Log: mAP@50, mAP@50–95, AP(Car/Ped/Cyc), FPS, VRAM.
