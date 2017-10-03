import time
import cv2
import numpy as np
from Configuration import Config as cfg
from VesselSegmentation.LineDetector import lineDetector2
from Tools.BinaryPostProcessing import binaryPostProcessing3


def vesselSegmentation(Img_green_filled, downsizeRatio, ImgName, discCenter, discRadius, ImgShow, Mask_old):

    time_step4_start = time.time()

    ''' BLACK AND WHITE IMAGE '''
    Img_green_reverse = 255 - Img_green_filled
    Img_BW, ResultImg = lineDetector2(Img_green_reverse, Mask_old)
    Img_BW = binaryPostProcessing3(Img_BW, removeArea=300, fillArea=100)

    BW_resized = cv2.resize(Img_BW*255, dsize=None, fx=1.0/downsizeRatio, fy=1.0/downsizeRatio)
    cv2.imwrite(cfg.imgDir + '\\BW_resized\\'+ImgName, BW_resized)

    # CONTINUE WITH RESIZED BW_IMAGE:
    Img_BW = BW_resized.copy()

    cv2.circle(Img_BW, center=(discCenter[1], discCenter[0]), radius=discRadius+5, color=0, thickness=-1)
    #VesselSkeleton_Pruned = skeletonization(Img_BW)

    time_step4 = time.time() - time_step4_start
    print 'Step 4: Vessel Segmentation finished, spending time:', time_step4
    print '##############################################################'


    ##################################################################################################
    print "End of Image Processing >>>", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    '''
    Images = [ BGR2RGB(ImgShow),  Img_BW,  VesselSkeleton_Pruned]
    Titles = [ 'ImgShow',  'Img_BW',  'VesselSkeleton_Pruned',  ] #'Img_BW', 'VesselSkeleton_connected',
    # plt.imshow(Images[0])
    #for i in xrange(0, len(Images)):
    #   plt.subplot(2,2, i+1), plt.imshow(Images[i], 'gray'), plt.title(Titles[i])
    #plt.show()
    '''

    return Img_BW

def saveData(imgDir, SingleImageFolder, Img_BW):
    np.save(imgDir + '\\pipeline_steps\\vesselSegmentation\\' + SingleImageFolder + '\\Img_BW.npy', Img_BW)
    return

def loadData(imgDir, SingleImageFolder):
    Mask_old = np.load(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Mask_old.npy')
    Img_green_filled = np.load(imgDir + '\\pipeline_steps\\preprocessGreen\\' + SingleImageFolder + '\\Img_green_filled.npy')
    return Mask_old, Img_green_filled