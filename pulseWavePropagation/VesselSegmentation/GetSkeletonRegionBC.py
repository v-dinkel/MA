import cv2
import numpy as np
from skimage import measure, morphology

from CircularRegionMask import diskMask, regionBMask, regionCMask
from RemoveBranch import removeBranchPoints


def getSkeletonRegionBC(VesselSkeleton_Pruned, discRegionParameter):

    rootPointRatio = discRegionParameter['rootPointRatio']
    factor_A = discRegionParameter['factor_A']  ##determine the region B
    factor_B = discRegionParameter['factor_B'] ##determine the region B
    factor_C = discRegionParameter['factor_C'] ##determine the region C
    discCenter = discRegionParameter['discCenter']
    discRadius = discRegionParameter['discRadius']

    VesselSkeleton_Pruned[VesselSkeleton_Pruned > 0] = 1
    VesselSkeleton_Pruned = morphology.skeletonize(VesselSkeleton_Pruned)
    VesselSkeleton_Pruned = np.uint8(VesselSkeleton_Pruned)

    height, width = VesselSkeleton_Pruned.shape[:2]

    DiskMask_A, VesselSkeleton_A = regionBMask(VesselSkeleton_Pruned, discCenter, discRadius, factor_B=factor_A)
    DiskMask_B, VesselSkeleton_B = regionBMask(VesselSkeleton_Pruned, discCenter, discRadius, factor_B=factor_B)
    DiskMask_C, VesselSkeleton_C = regionCMask(VesselSkeleton_Pruned, discCenter, discRadius, factor_C=factor_C)

    ###Rmeove the small stand-alone vessel segmetns
    VesselSkeletonA_Label = measure.label(VesselSkeleton_A)
    for i, region in enumerate(measure.regionprops(VesselSkeletonA_Label)):
        if region.area < 50:
            VesselSkeleton_A[VesselSkeletonA_Label == i + 1] = 0
        else:
            pass
    VesselSkeletonB_Label = measure.label(VesselSkeleton_B)
    for i, region in enumerate(measure.regionprops(VesselSkeletonB_Label)):
        if region.area < 50:
            VesselSkeleton_B[VesselSkeletonB_Label == i + 1] = 0
        else:
            pass
    VesselSkeletonC_Label = measure.label(VesselSkeleton_C)
    for i, region in enumerate(measure.regionprops(VesselSkeletonC_Label)):
        if region.area < 50:
            VesselSkeleton_C[VesselSkeletonC_Label == i + 1] = 0
        else:
            pass

    BranchRemovedImg_C, branchPoint_List = removeBranchPoints(VesselSkeleton_C)
    BranchRemovedImg_labelC = measure.label(BranchRemovedImg_C)

    BranchRemovedImg_label_RegionB = cv2.bitwise_and(BranchRemovedImg_labelC, BranchRemovedImg_labelC, mask=DiskMask_B)
    vesselLabels_RegionB = np.unique(BranchRemovedImg_label_RegionB)
    vesselLabels_RegionB = list(vesselLabels_RegionB[vesselLabels_RegionB != 0])

    return VesselSkeleton_A, VesselSkeletonA_Label, VesselSkeleton_C, BranchRemovedImg_labelC, VesselSkeleton_B, vesselLabels_RegionB, BranchRemovedImg_label_RegionB, branchPoint_List
