from __future__ import division
import numpy as np
import cv2

from scipy import interpolate
from SplineInterpolation import splineInterpolation
from Quality.IlluminationCorrection import illuminationCorrection2

# def updateVesselWidth(dict_smoothSide1, dict_smoothSide2):
#     dict_vesselWidth = {}
#     for vesselKey in dict_smoothSide1.keys():
#         dict_vesselWidth[vesselKey] = np.hypot((dict_smoothSide1[vesselKey][0] - dict_smoothSide2[vesselKey][0]),
#                                            (dict_smoothSide1[vesselKey][1] - dict_smoothSide2[vesselKey][1]))
#
#     return dict_vesselWidth



def updateVesselWidthRelatedParameters(Img, dict_side1_updated, dict_side2_updated, dict_vesselAngles):
    if len(Img.shape) == 3:
        Img_green = Img[:,:,1]
    else:
        Img_green = Img.copy()

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Img_green = clahe.apply(Img_green)
    Img_green_illum =  illuminationCorrection2(Img_green, kernel_size = 35, filterOption=0)
    X = np.arange(0, Img_green.shape[0])
    Y = np.arange(0, Img_green.shape[1])
    InterpFunc = interpolate.RectBivariateSpline(X, Y, Img_green_illum)  # Img_green_illum


    dict_vesselWidth_updated = {}
    dict_splinePoints_updated = {} ##no need to update splinePoints
    dict_smoothSide1_updated = {}
    dict_smoothSide2_updated = {}
    dict_profileIntensity_updated = {}

    for vesselKey in dict_side1_updated.keys():
        if len(dict_side1_updated[vesselKey][0]) > 20:  #if the vessel is shorter than 20 then remove it
            dict_vesselWidth_updated[vesselKey] = np.hypot((dict_side1_updated[vesselKey][0] - dict_side2_updated[vesselKey][0]),
                                                   (dict_side1_updated[vesselKey][1] - dict_side2_updated[vesselKey][1]))

            # try:
            tempMaxWidth = np.max(dict_vesselWidth_updated[vesselKey])
            tempMaxWidth = np.ceil(tempMaxWidth) + 2

            """Later test the cases that part of the side1 and 2 are removed"""

            ##Update smoothside1 and 2
            sidePointList1 = np.array([dict_side1_updated[vesselKey][0].reshape(-1), dict_side1_updated[vesselKey][1].reshape(-1)])
            sidePointList2 = np.array([dict_side2_updated[vesselKey][0].reshape(-1), dict_side2_updated[vesselKey][1].reshape(-1)])
            dict_smoothSide1_updated[vesselKey] = splineInterpolation(sidePointList1, order=2)
            dict_smoothSide2_updated[vesselKey] = splineInterpolation(sidePointList2, order=2)


            ##Update the dict_profileIntensity
            centerline0 = np.concatenate(
                ((dict_side1_updated[vesselKey][0].reshape(-1,1) + dict_side2_updated[vesselKey][0].reshape(-1,1)) / 2.0, \
                 (dict_side1_updated[vesselKey][1].reshape(-1,1)+ dict_side2_updated[vesselKey][1].reshape(-1,1)) / 2.0),
                axis=1)


            controlPoint = centerline0[::5, :]
            order = 1
            tck, u = interpolate.splprep(controlPoint.transpose(), k=order, s=0)
            unew = np.linspace(0, 1.00, centerline0.shape[0])
            out = interpolate.splev(unew, tck)
            dict_splinePoints_updated[vesselKey] = out

            derivative = interpolate.spalde(unew,tck)  ##two lists are returned, the first list is derivative for row, second is for col.

            der_row = np.array(derivative[0])
            der_col = np.array(derivative[1])

            der = np.array([der_row[:, 1], der_col[:, 1]]).transpose()
            normal1 = np.dot(der, np.array([[0, 1], [-1, 0]]))
            normal1 = np.float32(normal1) / np.tile(np.sqrt(normal1[:, 0] ** 2 + normal1[:, 1] ** 2), (2, 1)).transpose()

            dict_vesselAngles[vesselKey] = normal1
            inc = np.arange(0, tempMaxWidth) - tempMaxWidth // 2
            tempProfileRows = np.dot(normal1[:, 0].reshape((-1, 1)), inc.reshape(1, -1)) + np.tile(
                out[0].reshape(-1, 1), (1, len(inc)))
            tempProfileCols = np.dot(normal1[:, 1].reshape((-1, 1)), inc.reshape(1, -1)) + np.tile(
                out[1].reshape(-1, 1), (1, len(inc)))

            dict_profileIntensity_updated[vesselKey] = InterpFunc.ev(tempProfileRows, tempProfileCols)

        else:
            dict_side1_updated.pop(vesselKey)
            dict_side2_updated.pop(vesselKey)


    return dict_splinePoints_updated, dict_profileIntensity_updated, dict_side1_updated, dict_side2_updated, \
           dict_smoothSide1_updated, dict_smoothSide2_updated, dict_vesselWidth_updated


