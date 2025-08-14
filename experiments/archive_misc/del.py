import cv2, glob
files = sorted(glob.glob(r"D:/datasets/KITTI_DEYOLO_ZEROIR/images/ir_test/*.png"))[:10]
for f in files:
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    print(f.split("\\")[-1], "sum=", int(im.sum()), "max=", int(im.max()))