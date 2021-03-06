import rawpy
import cv2
import numpy as np
from tqdm import tqdm
import glob
import os

src_L = []
dst_L = []

for i in glob.glob(os.path.join('nikon_val', '*.JPG')):
    jpg_name = i
    raw_name = i.replace('.JPG', '.NEF')
    print(raw_name)
    template = cv2.imread(jpg_name)
    aaa = rawpy.imread(raw_name)
    frame = aaa.postprocess(use_camera_wb=True, no_auto_bright=False)
    cv2.imwrite('YA.JPG', frame[:,:,::-1])


    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(frame, None)

    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(des1, des2, k=2)

    # apply Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)


    # ========= find homography with RANSAC =========
    src_L += [kp1[m.queryIdx].pt for m in good]
    dst_L += [kp2[m.trainIdx].pt for m in good]


src_pts = np.float32(src_L).reshape(-1, 1, 2)
dst_pts = np.float32(dst_L).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print(M)



# ref_image= cv2.resize(ref_image, (template.shape[1], template.shape[0]), interpolation=cv2.INTER_CUBIC)