import numpy as np
import cv2

def resizeImage(Img0, downsizeRatio):
    Img_Resized_old = cv2.resize(Img0, dsize=None, fx=downsizeRatio, fy=downsizeRatio)
    Mask_old = np.zeros((Img_Resized_old.shape[:2]), dtype=np.uint8)
    Mask_old[20:-20, 20:-20] = 1

    Img_Resized = Img0.copy()
    Mask = np.zeros((Img_Resized.shape[:2]), dtype=np.uint8)
    Mask[20:-20, 20:-20] = 1
    Img = Img_Resized.copy()

    return Img, Img_Resized, Img_Resized_old, Mask, Mask_old

def saveData(imgDir, SingleImageFolder, Img, Img_Resized, Img_Resized_old, Mask, Mask_old):
    np.save(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Mask.npy', Mask)
    np.save(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Mask_old.npy', Mask_old)
    cv2.imwrite(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Img.tif', Img)
    cv2.imwrite(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Img_Resized.tif', Img_Resized)
    cv2.imwrite(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Img_Resized_old.tif',Img_Resized_old)
    return
