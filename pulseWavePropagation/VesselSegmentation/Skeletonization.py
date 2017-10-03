from __future__ import division

import numpy as np
from skimage import measure, morphology

from BWmorphThin import bwmorph_thin
from BranchDetection import branchedPointsDetection
from DetectEndPoints import detectEndPoints
from RemoveSpur import removeSpur

"""
Purpuse: 1. get the skeleton of Img_BW obtained in VesselSegmentation.py
         2. remove the spur of the skeleton
         3. get the pixel location dictionary for each vessel segment

Input: Img_BW
Output: Skeleton image, dictionary of vessel segment pixel locations, endpointlist, branchpoint list,

"""

def skeletonization(Img_BW):
    Img_BW[Img_BW>0] =1

    height, width = Img_BW.shape[:2]

    #####Get the skeleton and remove spurs######
    # VesselSkeleton_original = morphology.skeletonize(Img_BW)
    # VesselSkeleton_original = np.uint8(VesselSkeleton_original)
    VesselSkeleton_original = bwmorph_thin(Img_BW)
    VesselSkeleton_original = np.uint8(VesselSkeleton_original)

    endPoints_List0 = detectEndPoints(VesselSkeleton_original)
    BranchResults, CrossPointsResult = branchedPointsDetection(VesselSkeleton_original)
    branchList = np.where((BranchResults + CrossPointsResult) == True)
    VesselSkeleton_Pruned = removeSpur(VesselSkeleton_original, endPoints_List0, branchList, spurLength=10)  #this one is better
    # VesselSkeleton_Pruned = removeSpur(VesselSkeleton_original,  spurLength=15)

    ###Cut the sharp vessels####
    VesselSkeleton = VesselSkeleton_Pruned
    # VesselSkeleton = VesselSkeleton_original


    # VesselSkeleton = cutSharpVessels(VesselSkeleton_Pruned)

    ######Remove the small stand-alone segments after pruning
    VesselSkeletonLabel = measure.label(VesselSkeleton)
    for i, region in enumerate(measure.regionprops(VesselSkeletonLabel)):
        if region.area < 50:
            VesselSkeleton[VesselSkeletonLabel == i + 1] = 0
        else:
            pass

    VesselSkeleton[:20, :] = 0
    VesselSkeleton[-20:, :] = 0
    VesselSkeleton[:, :20] = 0
    VesselSkeleton[:, -20:] = 0


    return VesselSkeleton

