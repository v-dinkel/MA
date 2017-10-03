from Tools.ZoneMarker import zoneMarker
import cv2
import numpy as np
from Configuration import Config as cfg
import matplotlib.pyplot as plt

def plotVesselWidth(Img, Img_BW, ImgName, discRegionParameter, discCenter, discRadius, dict_meanVesselWidth_ZoneB, dict_segmentPixelLocs_ZoneB, dict_meanVesselWidth_ZoneC, dict_segmentPixelLocs_ZoneC, dict_splinePoints_updated, dict_smoothSide1_updated, dict_smoothSide2_updated, dict_side1_updated, dict_side2_updated):
    vesselLocsB = {}
    vesselLocsC = {}

    """Plot the spline center image with cross profile"""
    SplineImg = Img.copy()
    SplineImg = zoneMarker(SplineImg, discRegionParameter)
    for vesselkey in dict_meanVesselWidth_ZoneB.keys():
        thisCol = (0, 255, 0)
        keyName = vesselkey
        '''
        if vesselkey in splineMap.keys():
            thisCol = (0, 255, 0)
            keyName = splineMap[vesselkey][0]
        '''
        temppixlocs = dict_segmentPixelLocs_ZoneB[vesselkey]
        temptxtloc = [int(temppixlocs[len(temppixlocs) // 2][0]), int(temppixlocs[len(temppixlocs) // 2][1])]
        vesselLocsB[vesselkey] = temptxtloc
        cv2.putText(SplineImg, '[' + str(keyName) + ']' + str(round(dict_meanVesselWidth_ZoneB[vesselkey], 2)),
                    (temptxtloc[1], temptxtloc[0]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=thisCol, thickness=2)
    for vesselkey in dict_meanVesselWidth_ZoneC.keys():
        thisCol = (255, 0, 0)
        keyName = vesselkey
        '''
        if vesselkey in splineMap.keys():
            thisCol = (255, 0, 0)
            keyName = splineMap[vesselkey][0]
        '''
        temppixlocs = dict_segmentPixelLocs_ZoneC[vesselkey]
        temptxtloc = [int(temppixlocs[len(temppixlocs) // 2][0]), int(temppixlocs[len(temppixlocs) // 2][1])]
        vesselLocsC[vesselkey] = temptxtloc
        cv2.putText(SplineImg, '[' + str(keyName) + ']' + str(round(dict_meanVesselWidth_ZoneC[vesselkey], 2)),
                    (temptxtloc[1], temptxtloc[0]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=thisCol, thickness=2)

    '''Put disc center & size on disc'''
    cv2.putText(SplineImg, '(' + str(discCenter) + ',' + str(discRadius) + ')', (discCenter[1] - 100, discCenter[0]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(200, 200, 200), thickness=2)

    # draw spline
    '''

    '''

    cv2.imwrite(cfg.imgDir + '\\width_hq\\' + ImgName, SplineImg)
    # import pdb;
    # pdb.set_trace()

    plt.subplot(1, 1, 1), plt.imshow(SplineImg, 'gray'), plt.title(ImgName)
    for vesselkey in dict_splinePoints_updated.keys():
        plt.plot(dict_splinePoints_updated[vesselkey][1], dict_splinePoints_updated[vesselkey][0],
                 color='blue')  # plot the center spline skeleton
        plt.plot(dict_smoothSide1_updated[vesselkey][1], dict_smoothSide1_updated[vesselkey][0],
                 color='orange')  # plot the smooth vessel edge
        plt.plot(dict_smoothSide2_updated[vesselkey][1], dict_smoothSide2_updated[vesselkey][0],
                 color='orange')  # plot another vessel edge

        for temp in xrange(0, len(dict_side1_updated[vesselkey][1]), 2):  # plot the cross profile
            plt.plot([dict_side1_updated[vesselkey][1][temp], dict_side2_updated[vesselkey][1][temp]],
                     [dict_side1_updated[vesselkey][0][temp], dict_side2_updated[vesselkey][0][temp]], color='white')
    plt.xlim(0, Img_BW.shape[1])
    plt.ylim(Img_BW.shape[0], 0)
    plt.savefig(cfg.imgDir + '\\width\\' + ImgName)
    # plt.show()
    plt.clf()

    return vesselLocsB, vesselLocsC

def loadData(imgDir, SingleImageFolder):
    Img = cv2.imread(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Img.tif')
    Img_BW = np.load(imgDir + '\\pipeline_steps\\vesselSegmentation\\' + SingleImageFolder + '\\Img_BW.npy')
    dict_meanVesselWidth_ZoneB = np.load(
        imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_meanVesselWidth_ZoneB.npy').item()
    dict_segmentPixelLocs_ZoneB = np.load(
        imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_segmentPixelLocs_ZoneB.npy').item()
    dict_meanVesselWidth_ZoneC = np.load(
        imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_meanVesselWidth_ZoneC.npy').item()
    dict_segmentPixelLocs_ZoneC = np.load(
        imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_segmentPixelLocs_ZoneC.npy').item()
    dict_splinePoints_updated = np.load(
        imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_splinePoints_updated.npy').item()
    dict_smoothSide1_updated = np.load(
        imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_smoothSide1_updated.npy').item()
    dict_smoothSide2_updated = np.load(
        imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_smoothSide2_updated.npy').item()
    dict_side1_updated = np.load(
        imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_side1_updated.npy').item()
    dict_side2_updated = np.load(
        imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + SingleImageFolder + '\\dict_side2_updated.npy').item()
    #splineMap = np.load(imgDir + '\\pipeline_steps\\splineMapping\\' + SingleImageFolder + '\\splineMap.npy').item()

    return Img, Img_BW, dict_meanVesselWidth_ZoneB, dict_meanVesselWidth_ZoneC, dict_segmentPixelLocs_ZoneB, dict_segmentPixelLocs_ZoneC, dict_splinePoints_updated, dict_smoothSide1_updated, dict_smoothSide2_updated, dict_side1_updated, dict_side2_updated