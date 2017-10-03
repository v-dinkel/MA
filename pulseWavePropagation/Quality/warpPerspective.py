# https://stackoverflow.com/questions/22279069/how-to-align-multiple-camera-images-using-opencv

import cv2
import numpy as np
from numpy.linalg import inv

from Configuration import Config as cfg
from Tools import utils

def warpPerspective(referenceKp, referenceDes, imgDir, targetImg):
    print targetImg
    imgTarget = cv2.imread(imgDir+targetImg,0)
    targetKp, targetDes = sift.detectAndCompute(imgTarget, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(referenceDes, targetDes, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            # Removed the brackets around m
            good.append(m)
    src_pts = np.float32([referenceKp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([targetKp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #matchesMask = mask.ravel().tolist()
    dst = cv2.warpPerspective(imgTarget, inv(M), (1600, 1200))
    # plt.subplot(121),plt.imshow(img2),plt.title('Input')
    # plt.subplot(122),plt.imshow(dst),plt.title('Output')
    # plt.show()
    cv2.imwrite(imgDir+'\\warpPerspective\\'+targetImg,dst)

onlyImages = utils.getListOfImages(cfg.imgDir, cfg.imgFormats)

imgRef = cv2.imread(cfg.imgDir+onlyImages[0],0)
sift = cv2.SIFT()
referenceKp, referenceDes = sift.detectAndCompute(imgRef,None)

cv2.imwrite(cfg.imgDir+'warpPerspective\\'+onlyImages[0],imgRef)
for i in range(1,len(onlyImages)):
    warpPerspective(referenceKp, referenceDes, cfg.imgDir, onlyImages[i])

#img=cv2.drawKeypoints(img1,kp1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
