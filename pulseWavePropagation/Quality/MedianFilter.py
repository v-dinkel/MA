import cv2

def medianFilter(Image, kernesize = 5):
    ##Input: image (RGB or green both ok); kernel size, default 5, change between 3 to 15
    ##Output: preprocessed RGB or green channel image

    filteredImg = cv2.medianBlur(Image, ksize=kernesize)
    return filteredImg
