
import cv2

def gaussianFilter(Image, kernelsize = 5):
    ##Gaussian Filtering
    ##Input: Image and kernel size

    filteredImg = cv2.GaussianBlur(Image, (kernelsize,kernelsize), 1)
    return filteredImg