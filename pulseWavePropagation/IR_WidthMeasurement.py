from __future__ import division
import cv2
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from Tools.BGR2RGB import BGR2RGB
from Tools.ZoneMarker import zoneMarker
from Tools.SortFolder import natsort

from Configuration import Config as cfg
from Tools import utils

"""
This program is trying to realize the vessel analysis for Infared Images!!
"""


ImgNumber = 0
SegMethod = 2  ##Gabor 1 / LineDetector 2


"""provide the image folder location"""
# Imgfolder = 'Image\\folder\\path\\'
Imgfolder = cfg.imgDir
ImgFileList =  utils.getListOfImages(cfg.imgDir, cfg.imgFormats)
natsort(ImgFileList)
if ImgFileList.__contains__('Thumbs.db'):
    ImgFileList.remove('Thumbs.db')



"Read a image first: "
ImgName = Imgfolder + ImgFileList[ImgNumber]
Img0 = cv2.imread(ImgName)
print 'Img Name:', ImgNumber, ImgFileList[ImgNumber]


"""Resize the image"""
downsizeRatio = 1
Img_Resized = cv2.resize(Img0, dsize=None, fx=downsizeRatio, fy=downsizeRatio)
Mask = np.zeros((Img_Resized.shape[:2]), dtype=np.uint8)
Mask[20:-20, 20:-20] = 1
Img = Img_Resized.copy()
##############################################################

"""Step2: Preprocessing"""
from Quality.IlluminationCorrection import illuminationCorrection, illuminationCorrection2
from skimage import morphology

time_step2_start = time.time()

###Histogram Equalization
Img_green = Img_Resized[:,:,1]
# Img_green = cv2.equalizeHist(Img_green)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
Img_green = clahe.apply(Img_green)
Img_green_filled = morphology.opening(Img_green, morphology.disk(3))
Img_green_filled = cv2.medianBlur(Img_green_filled, ksize=5)

IllumGreen = illuminationCorrection(Img_green_filled, kernel_size=35, Mask = Mask)
IllumGreen = morphology.opening(IllumGreen, morphology.disk(3))
IllumGreen = cv2.medianBlur(IllumGreen, ksize=5)



time_step2 = time.time() - time_step2_start
print 'Step 2: Preprocessing finished, spending time:', time_step2
print '##############################################################'



##############################################################

"""Step3: Optic Disc Detectioin"""
from OpticDiscDetection.DiscDetection_IR import discDetection_IR

time_step3_start = time.time()

discCenter, discRadius = discDetection_IR(Img, Mask)
print 'Disk Parameter', discCenter, discRadius
# discCenter, discRadius = (501, 582), 80   #Image AIBL4_03

discRegionParameter = {}
discRegionParameter['rootPointRatio'] = 2  ###this ratio determins the region of searching for root node (rootPoingRatio, 1)
discRegionParameter['factor_B'] = (3, 1.5)   ##determine the region B
discRegionParameter['factor_C'] = (8, 1.5)   ##determine the region C
discRegionParameter['discCenter'] = discCenter   ##determine the region C
discRegionParameter['discRadius'] = discRadius   ##determine the region C


##Mark the disc parameters on the original image
ImgShow = Img_Resized.copy()  ###Imgshow is for showing purposes only
cv2.circle(ImgShow, center=(discCenter[1], discCenter[0]), radius=discRadius, color=(255,255,255), thickness=5)
cv2.circle(ImgShow, center=(discCenter[1], discCenter[0]), radius=2*discRadius, color=(255,255,255), thickness=2) #2-3 RegionB
cv2.circle(ImgShow, center=(discCenter[1], discCenter[0]), radius=3*discRadius, color=(255,255,255), thickness=2)
cv2.putText(ImgShow,"Zone B", (discCenter[1]-50, discCenter[0]-int(2.2*discRadius)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
cv2.circle(ImgShow, center=(discCenter[1], discCenter[0]), radius=5*discRadius, color=(255,255,255), thickness=2) #3-5 REgionC
cv2.putText(ImgShow,"Zone C", (discCenter[1]-50, discCenter[0]-int(4*discRadius)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)


time_step3 = time.time() - time_step3_start
print 'Step 3: Optic Disc Detection finished, spending time:', time_step3
print '##############################################################'



# ##############################################################

"""Step4: Vessel Segmentation"""
from VesselSegmentation.VesselSegmentation import vesselSegmentation
from VesselSegmentation.Skeletonization import skeletonization
from VesselSegmentation.LineDetector import lineDetector2
from Tools.BinaryPostProcessing import binaryPostProcessing3


time_step4_start = time.time()
downsizeRatio = 0.4
IllumGreen_resized = cv2.resize(IllumGreen, dsize=None, fx=downsizeRatio, fy=downsizeRatio)
Mask_resized = cv2.resize(Mask, dsize=None, fx=downsizeRatio, fy=downsizeRatio)
if SegMethod ==1:
    Img_BW = vesselSegmentation(IllumGreen_resized, Mask_resized)
    # Img_BW = vesselSegmentation(Img_green_tophat1, Mask)
    # cv2.circle(Img_BW, center=(discCenter[1], discCenter[0]), radius=discRadius+5, color=0, thickness=-1)

else:
    Img_green_reverse = 255 - Img_green_filled
    Img_green_reverse = cv2.resize(Img_green_reverse, dsize=None, fx=downsizeRatio, fy=downsizeRatio)
    Img_BW, ResultImg = lineDetector2(Img_green_reverse, Mask_resized)
    # Img_BW, ResultImg = lineDetector2(Img_green_tophat, Mask)
    Img_BW = binaryPostProcessing3(Img_BW, removeArea=300, fillArea=100)
    # cv2.circle(Img_BW, center=(discCenter[1], discCenter[0]), radius=discRadius+5, color=0, thickness=-1)

Img_BW = cv2.resize(Img_BW, dsize=(Img.shape[1], Img.shape[0]), fx=0, fy=0)
cv2.circle(Img_BW, center=(discCenter[1], discCenter[0]), radius=discRadius+5, color=0, thickness=-1)

time_step4 = time.time() - time_step4_start
print 'Step 4: Vessel Segmentation finished, spending time:', time_step4
print '##############################################################'
# ##############################################################
"""Step 5: Vessel Width Measurement"""
from VesselSegmentation.GetSkeletonPixelLocs import getSkeletonPixelLocs
from VesselSegmentation.GetSkeletonRegionBC import getSkeletonRegionBC
from VesselWidth.PreciseVesselWidthProfile import vesselWidthProfile
from VesselWidth.UpdateVesselWidth import updateVesselWidthRelatedParameters
from VesselWidth.GetVesselMeanWidth import getVesselMeanWidth
from VesselWidth.SeperateZoneAB import seperateZoneAB

VesselSkeleton_Pruned = skeletonization(Img_BW)
VesselSkeleton_C, BranchRemovedImg_labelC, VesselSkeleton_B, vesselLabels_RegionB, BranchRemovedImg_label_RegionB, branchPoint_List = \
                    getSkeletonRegionBC(VesselSkeleton_Pruned, discRegionParameter)
dict_segmentPixelLocs, endPoints_List, dict_chainCode = getSkeletonPixelLocs(BranchRemovedImg_labelC)


dict_splinePoints, dict_profileIntensity,  dict_side1, dict_side2, \
        dict_smoothSide1, dict_smoothSide2, dict_vesselWidth0, dict_vesselAngles0, dict_profileRows, dict_profileCols =\
        vesselWidthProfile(Img, IllumGreen, Img_BW, Mask, dict_segmentPixelLocs)

"""update the vessel width related parameters after the width calculation"""
dict_side1_updated = dict_side1.copy()
dict_side2_updated = dict_side2.copy()

dict_splinePoints_updated, dict_profileIntensity_updated, dict_side1_updated, dict_side2_updated, \
    dict_smoothSide1_updated, dict_smoothSide2_updated, dict_vesselWidth_updated       \
        = updateVesselWidthRelatedParameters(Img, dict_side1_updated, dict_side2_updated, dict_vesselAngles0)


# ##############################################################
"""Step 6: Get Vessel Width in ZoneA and ZoneB"""

dict_vesselWidth_ZoneB, dict_vesselWidth_ZoneC,  dict_segmentPixelLocs_ZoneB, dict_segmentPixelLocs_ZoneC =\
    seperateZoneAB(discRegionParameter, dict_segmentPixelLocs, dict_splinePoints_updated, dict_vesselWidth_updated)

dict_meanVesselWidth = getVesselMeanWidth(dict_vesselWidth_updated)
dict_meanVesselWidth_ZoneB =  getVesselMeanWidth(dict_vesselWidth_ZoneB)
dict_meanVesselWidth_ZoneC =  getVesselMeanWidth(dict_vesselWidth_ZoneC)



##################################################################################################
print "End of Image Processing >>>", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
plt.figure()

# Images = [ BGR2RGB(ImgShow),  Img_BW,  VesselSkeleton_Pruned]
# Titles = [ 'ImgShow',  'Img_BW',  'VesselSkeleton_Pruned',  ] #'Img_BW', 'VesselSkeleton_connected',
# # plt.imshow(Images[0])
# for i in xrange(0, len(Images)):
#     plt.subplot(2,2, i+1), plt.imshow(Images[i], 'gray'), plt.title(Titles[i])
# plt.show()


"""Plot the spline center image with cross profile"""
SplineImg = Img.copy()
SplineImg = zoneMarker(SplineImg, discRegionParameter)
for vesselkey in dict_meanVesselWidth_ZoneB.keys():
    temppixlocs = dict_segmentPixelLocs_ZoneB[vesselkey]
    temptxtloc = [int(temppixlocs[len(temppixlocs) // 2][0]), int(temppixlocs[len(temppixlocs) // 2][1])]
    cv2.putText(SplineImg, str(round(dict_meanVesselWidth_ZoneB[vesselkey], 2)), (temptxtloc[1], temptxtloc[0]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=3)
for vesselkey in dict_meanVesselWidth_ZoneC.keys():
    temppixlocs = dict_segmentPixelLocs_ZoneC[vesselkey]
    temptxtloc = [int(temppixlocs[len(temppixlocs) // 2][0]), int(temppixlocs[len(temppixlocs) // 2][1])]
    cv2.putText(SplineImg, str(round(dict_meanVesselWidth_ZoneC[vesselkey], 2)), (temptxtloc[1], temptxtloc[0]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=3)

plt.subplot(1, 1, 1), plt.imshow(SplineImg, 'gray'), plt.title("spline")
for vesselkey in dict_splinePoints_updated.keys():
    plt.plot(dict_splinePoints_updated[vesselkey][1], dict_splinePoints_updated[vesselkey][0], color='blue') #plot the center spline skeleton
    plt.plot(dict_smoothSide1_updated[vesselkey][1], dict_smoothSide1_updated[vesselkey][0], color='orange') #plot the smooth vessel edge
    plt.plot(dict_smoothSide2_updated[vesselkey][1], dict_smoothSide2_updated[vesselkey][0], color='orange') #plot another vessel edge

    for temp in xrange(0, len(dict_side1_updated[vesselkey][1]), 5):   #plot the cross profile
        plt.plot([dict_side1_updated[vesselkey][1][temp], dict_side2_updated[vesselkey][1][temp]],
                 [dict_side1_updated[vesselkey][0][temp], dict_side2_updated[vesselkey][0][temp]], color='white')
plt.xlim(0,Img_BW.shape[1])
plt.ylim(Img_BW.shape[0], 0)
plt.show()