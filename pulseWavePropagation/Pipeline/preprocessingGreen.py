from Quality.IlluminationCorrection import illuminationCorrection, illuminationCorrection2
from skimage import morphology
import time
import cv2
import numpy as np

def preprocessingGreen(Img_Resized_old, Img_Resized, Mask_old, Mask):
    time_step2_start = time.time()

    ###Histogram Equalization
    Img_green = Img_Resized_old[:,:,1]
    # Img_green = cv2.equalizeHist(Img_green)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Img_green = clahe.apply(Img_green)
    Img_green_filled = morphology.opening(Img_green, morphology.disk(3))
    Img_green_filled = cv2.medianBlur(Img_green_filled, ksize=5)

    IllumGreen = illuminationCorrection(Img_green_filled, kernel_size=35, Mask = Mask_old)
    IllumGreen = morphology.opening(IllumGreen, morphology.disk(3))
    IllumGreen = cv2.medianBlur(IllumGreen, ksize=5)

    ###Histogram Equalization
    Img_green_large = Img_Resized[:,:,1]
    # Img_green = cv2.equalizeHist(Img_green)
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    Img_green_large = clahe2.apply(Img_green_large)
    Img_green_filled_large = morphology.opening(Img_green_large, morphology.disk(3))
    Img_green_filled_large = cv2.medianBlur(Img_green_filled_large, ksize=5)

    IllumGreen_large = illuminationCorrection(Img_green_filled_large, kernel_size=35, Mask = Mask)
    IllumGreen_large = morphology.opening(IllumGreen_large, morphology.disk(3))
    IllumGreen_large = cv2.medianBlur(IllumGreen_large, ksize=5)

    time_step2 = time.time() - time_step2_start
    print 'Step 2: Preprocessing finished, spending time:', time_step2
    print '##############################################################'

    return Img_green_filled, IllumGreen_large

def loadData(imgDir, SingleImageFolder):
    Mask = np.load(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Mask.npy')
    Mask_old = np.load(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Mask_old.npy')
    Img_Resized = cv2.imread(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Img_Resized.tif')
    Img_Resized_old = cv2.imread(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Img_Resized_old.tif')
    return Mask, Mask_old, Img_Resized, Img_Resized_old

def saveData(imgDir, SingleImageFolder, Img_green_filled, IllumGreen_large):
    np.save(imgDir + '\\pipeline_steps\\preprocessGreen\\' + SingleImageFolder + '\\Img_green_filled.npy',Img_green_filled)
    np.save(imgDir + '\\pipeline_steps\\preprocessGreen\\' + SingleImageFolder + '\\IllumGreen_large.npy',IllumGreen_large)
    return