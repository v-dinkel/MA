from Configuration import Config as cfg
from Tools import utils

import cv2
import pdb
import matplotlib.pyplot as plt


img_names = utils.getListOfImages(cfg.imgDir+'BW_resized\\', cfg.imgFormats)
imgs = utils.loadAllImages(cfg.imgDir+'BW_resized\\', cfg.imgFormats, False)

img = imgs[0]

#find first vessel
bw_layer = img[:][:][0]

k=0
for row in bw_layer:

    if sum(row)>0:
        print 'ZES'
        newRow = row*255
        bw_layer[k]=newRow
        cv2.imshow('image', bw_layer)
        cv2.waitKey(0)
        pdb.set_trace()
    k+=1

laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()