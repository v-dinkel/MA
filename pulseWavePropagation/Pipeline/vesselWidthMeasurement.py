from VesselSegmentation.GetSkeletonPixelLocs import getSkeletonPixelLocs
from VesselSegmentation.GetSkeletonRegionBC import getSkeletonRegionBC
from VesselWidth.PreciseVesselWidthProfile import vesselWidthProfile
from VesselWidth.UpdateVesselWidth import updateVesselWidthRelatedParameters
from VesselWidth.GetVesselMeanWidth import getVesselMeanWidth
from VesselWidth.SeperateZoneAB import seperateZoneAB
from VesselSegmentation.Skeletonization import skeletonization
import numpy as np
import cv2

def vesselWidthMeasurement(Img_BW, discRegionParameter, Img, IllumGreen_large, Mask):

    VesselSkeleton_Pruned = skeletonization(Img_BW)
    VesselSkeleton_A, VesselSkeletonA_Label, VesselSkeleton_C, BranchRemovedImg_labelC, VesselSkeleton_B, vesselLabels_RegionB, BranchRemovedImg_label_RegionB, branchPoint_List = \
        getSkeletonRegionBC(VesselSkeleton_Pruned, discRegionParameter)
    dict_segmentPixelLocs, endPoints_List, dict_chainCode = getSkeletonPixelLocs(BranchRemovedImg_labelC)

    dict_splinePoints, dict_profileIntensity, dict_side1, dict_side2, \
    dict_smoothSide1, dict_smoothSide2, dict_vesselWidth0, dict_vesselAngles0, dict_profileRows, dict_profileCols = \
        vesselWidthProfile(Img, IllumGreen_large, Img_BW, Mask, dict_segmentPixelLocs)

    """update the vessel width related parameters after the width calculation"""
    dict_side1_updated = dict_side1.copy()
    dict_side2_updated = dict_side2.copy()

    dict_splinePoints_updated, dict_profileIntensity_updated, dict_side1_updated, dict_side2_updated, \
    dict_smoothSide1_updated, dict_smoothSide2_updated, dict_vesselWidth_updated \
        = updateVesselWidthRelatedParameters(Img, dict_side1_updated, dict_side2_updated, dict_vesselAngles0)

    # ##############################################################
    """Step 6: Get Vessel Width in ZoneA and ZoneB"""

    dict_vesselWidth_ZoneB, dict_vesselWidth_ZoneC, dict_segmentPixelLocs_ZoneB, dict_segmentPixelLocs_ZoneC = \
        seperateZoneAB(discRegionParameter, dict_segmentPixelLocs, dict_splinePoints_updated, dict_vesselWidth_updated)

    dict_meanVesselWidth = getVesselMeanWidth(dict_vesselWidth_updated)
    dict_meanVesselWidth_ZoneB = getVesselMeanWidth(dict_vesselWidth_ZoneB)
    dict_meanVesselWidth_ZoneC = getVesselMeanWidth(dict_vesselWidth_ZoneC)

    return dict_splinePoints_updated, dict_meanVesselWidth_ZoneB, dict_meanVesselWidth_ZoneC, dict_segmentPixelLocs_ZoneB, dict_segmentPixelLocs_ZoneC, dict_side1_updated, dict_side2_updated, dict_smoothSide1_updated, dict_smoothSide2_updated

def saveData(imgDir, SingleImageFolder, dict_splinePoints_updated, dict_meanVesselWidth_ZoneB, dict_meanVesselWidth_ZoneC, dict_segmentPixelLocs_ZoneB, dict_segmentPixelLocs_ZoneC, dict_side1_updated, dict_side2_updated, dict_smoothSide1_updated, dict_smoothSide2_updated ):
    np.save(imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_splinePoints_updated.npy',dict_splinePoints_updated)
    np.save(imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_meanVesselWidth_ZoneB.npy',dict_meanVesselWidth_ZoneB)
    np.save(imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_meanVesselWidth_ZoneC.npy',dict_meanVesselWidth_ZoneC)
    np.save(imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_segmentPixelLocs_ZoneB.npy',dict_segmentPixelLocs_ZoneB)
    np.save(imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_segmentPixelLocs_ZoneC.npy',dict_segmentPixelLocs_ZoneC)
    np.save(imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_side1_updated.npy',dict_side1_updated)
    np.save(imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_side2_updated.npy',dict_side2_updated)
    np.save(imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_smoothSide1_updated.npy',dict_smoothSide1_updated)
    np.save(imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_smoothSide2_updated.npy',dict_smoothSide2_updated)
    return

def loadData(imgDir, SingleImageFolder):
    Img_BW = np.load(imgDir + '\\pipeline_steps\\vesselSegmentation\\' + SingleImageFolder + '\\Img_BW.npy')
    Img = cv2.imread(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Img.tif')
    IllumGreen_large = np.load(imgDir + '\\pipeline_steps\\preprocessGreen\\' + SingleImageFolder + '\\IllumGreen_large.npy')
    Mask = np.load(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Mask.npy')
    return Img, Img_BW, IllumGreen_large, Mask