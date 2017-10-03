from __future__ import division

import cv2
import numpy as np
from scipy import interpolate

from VesselWidth.CreateVesselProfileMasks import createVesselProfileMasks
from VesselWidth.EdgeToolFunctions import find_maximum_gradient_columns, gaussian_filter_1d_sigma, compute_discrete_2d_derivative,\
    find_closest_crossing, find_most_connected_crossings, get_side
from VesselWidth.SplineInterpolation import splineInterpolation

from scipy.ndimage import filters

"""This is the updated vessel profile and width measurement program, according to ARIA implementation,
Reference: Bankhead, P., Scholfield, C.N., McGeown, J.G. and Curtis, T.M., 2012.
Fast retinal vessel detection and measurement using wavelets and edge location refinement.
PloS one, 7(3), p.e32435."""

def vesselWidthProfile(Img, IllumImage, Img_BW, Mask, dict_segmentPixelLocs):
    if len(Img.shape) == 3:
        Img_green = Img[:, :, 1]
    else:
        Img_green = Img

    Img_BW[Img_BW>0]=1
    Mask[Mask>0]=1

    height, width = Img_BW.shape[:2]

    if len(IllumImage.shape) ==3:
        IllumGreenReverse = np.uint8(255 - IllumImage[:, :, 1])
    else:
        IllumGreenReverse = np.uint8(255 - IllumImage)

    #print IllumGreenReverse.shape, " =?= ", Mask.shape
    try:
        IllumGreenReverse = cv2.bitwise_and(IllumGreenReverse, IllumGreenReverse, mask=Mask)
    except:
        print 'error creating IllumGreenReverse'
        #Mask = np.array([np.append(Img_BW[k], 0) for k in range(0, len(Img_BW))])
        import pdb; pdb.set_trace()

    DistTransform_vessel = cv2.distanceTransform(Img_BW, distanceType=cv2.cv.CV_DIST_L2,
                                                 maskSize=cv2.cv.CV_DIST_MASK_PRECISE)
    maxVesselWidth = 6*np.max(DistTransform_vessel)


    """smooth the skeleton image into a spline image."""
    ##for RectBivariateSpline
    X = np.arange(0, Img_green.shape[0])
    Y = np.arange(0, Img_green.shape[1])
    InterpFunc_Illum = interpolate.RectBivariateSpline(X, Y, IllumGreenReverse) #ImgGreenReverse
    InterpFunc = interpolate.RectBivariateSpline(X, Y, Img_green) #ImgGreenReverse
    ##The interpolated image should be the reverse image


    SplineImg = np.zeros(Img_BW.shape)
    dict_splinePoints = {}
    dict_centerPoint = {}
    dict_vesselAngles = {}
    dict_profileRows = {}
    dict_profileCols = {}
    dict_profileIntensity = {}
    dict_profileIntensity_Illum = {}

    for vesselLabel in dict_segmentPixelLocs.keys(): #['1', '2', '15', '25']
        pixelLocs0 = dict_segmentPixelLocs[vesselLabel]
        if pixelLocs0.shape[0] >= 21: #11 previously. This means that vessel length <11 are not used for width calculation.
            pixelLocs = pixelLocs0[5:-5, :]   #the first and end 5 pixels near branch points are removed
            controlPoint = pixelLocs[np.arange(len(pixelLocs))[::10], :]  ##select every other 5 points.
            for loc in controlPoint:
                SplineImg[int(loc[0]), int(loc[1])] = 1

            order = 1
            tck, u = interpolate.splprep(controlPoint.transpose(), k = order, s=0)
            unew = np.linspace(0, 1.00, len(pixelLocs))
            out = interpolate.splev(unew, tck)
            dict_splinePoints[vesselLabel] = out
            derivative = interpolate.spalde(unew, tck) ##two lists are returned, the first list is derivative for row, second is for col.

            der_row = np.array(derivative[0])
            der_col = np.array(derivative[1])

            der = np.array([der_row[:, 1], der_col[:, 1]]).transpose()
            normal1 = np.dot(der, np.array([[0, 1], [-1, 0]]))
            normal1 = np.float32(normal1) / np.tile(np.sqrt(normal1[:, 0] ** 2 + normal1[:, 1] ** 2), (2, 1)).transpose()


            dict_centerPoint[vesselLabel] = (out[0], out[1])
            dict_vesselAngles[vesselLabel] = normal1
            inc = np.arange(0, maxVesselWidth) - maxVesselWidth//2
            dict_profileRows[vesselLabel] = np.dot(normal1[:,0].reshape((-1, 1)), inc.reshape(1,-1)) + np.tile(out[0].reshape(-1, 1), (1, len(inc)))
            dict_profileCols[vesselLabel] = np.dot(normal1[:,1].reshape((-1, 1)), inc.reshape(1,-1)) + np.tile(out[1].reshape(-1, 1), (1, len(inc)))

            """Remove the ones outside image region"""
            dict_profileRows[vesselLabel][dict_profileRows[vesselLabel] >= height-1] = np.nan
            dict_profileRows[vesselLabel][dict_profileRows[vesselLabel] < 0] = np.nan
            dict_profileRows[vesselLabel][dict_profileCols[vesselLabel] >= width-1] = np.nan
            dict_profileRows[vesselLabel][dict_profileCols[vesselLabel] < 0] = np.nan

            dict_profileCols[vesselLabel][np.isnan(dict_profileRows[vesselLabel])] = np.nan

            dict_profileRows[vesselLabel] = dict_profileRows[vesselLabel][~np.isnan(dict_profileRows[vesselLabel]).any(axis=1)]
            dict_profileCols[vesselLabel] = dict_profileCols[vesselLabel][~np.isnan(dict_profileCols[vesselLabel]).any(axis=1)]

            dict_profileIntensity[vesselLabel] = InterpFunc.ev(dict_profileRows[vesselLabel], dict_profileCols[vesselLabel])
            dict_profileIntensity_Illum[vesselLabel] = InterpFunc_Illum.ev(dict_profileRows[vesselLabel], dict_profileCols[vesselLabel])

    ##################################################################
    """To calculate the acurate vessel width and vessel edges"""

    smooth_scale_parallel = 3
    smooth_scale_perpendicular = 0.15
    enforce_connectedness = True

    dict_ProfileMap, dict_BWVesselProfiles, dict_BWRegionProfiles = createVesselProfileMasks(dict_profileRows, dict_profileCols, Img_BW, Mask)
    dict_side1 = {}
    dict_side2 = {}
    dict_smoothSide1 = {}
    dict_smoothSide2 = {}
    dict_vesselWidth = {}
    dict_profileSide = {}
    for vesselKey in dict_profileRows.keys():

        #% Create the default side coordinates
        dict_side1[vesselKey] = (np.zeros(len(dict_centerPoint[vesselKey][0])), np.zeros(len(dict_centerPoint[vesselKey][1])))
        dict_side2[vesselKey] = (np.zeros(len(dict_centerPoint[vesselKey][0])), np.zeros(len(dict_centerPoint[vesselKey][1])))

        #Extract the profiles and coordinates
        im_profiles = dict_profileIntensity_Illum[vesselKey]
        im_profiles_rows = dict_profileRows[vesselKey]
        im_profiles_cols = dict_profileCols[vesselKey]
        n_profiles = dict_profileRows[vesselKey].shape[0]

        ##get the width estimate of each vessel
        col_central = dict_profileIntensity_Illum[vesselKey].shape[1] // 2
        BW_vessel_profiles = dict_BWVesselProfiles[vesselKey]
        binary_sums = np.sum(BW_vessel_profiles, 1)
        width_estimate0 = np.median(binary_sums) #[BW_vessel_profiles[:, col_central]]
        if width_estimate0 > col_central:
            width_estimate0 = col_central

        ##Compute a mean profile for the entire vessel, omitting pixels closer to other vessels
        BW_regions = dict_BWRegionProfiles[vesselKey]
        im_profiles_closest = im_profiles.copy()
        im_profiles_closest[np.bitwise_not(BW_regions)] = np.NaN
        profile_mean = np.nanmean(im_profiles_closest, 0)

        try:
            left_mean_col, right_mean_col = find_maximum_gradient_columns(profile_mean.copy(), width_estimate0)  #profile_mean will be changed inside the func
        except:
            pass

        width_estimate = right_mean_col - left_mean_col

        # Create 1D Gaussian filters for smoothing parallel and perpendicular to the vessel
        # Sigma values are based on the square root of the scaled width
        # estimates
        gv = gaussian_filter_1d_sigma(np.sqrt(width_estimate * smooth_scale_parallel)).reshape(-1, 1)
        gh = gaussian_filter_1d_sigma(np.sqrt(width_estimate * smooth_scale_perpendicular)).reshape(-1,1)

        #Apply Gaussian smoothing
        im_profiles_filtered = filters.convolve(im_profiles, np.dot(gv, gh.transpose()))

        #Compute 2nd derivative perpendicular to vessel orientation
        im_profiles_2d = compute_discrete_2d_derivative(im_profiles_filtered)
        ##Remove from consideration pixels outside the search region
        im_profiles_2d[np.bitwise_not(BW_regions)] = np.NAN

        diffs = np.diff(im_profiles_2d, axis=1)
        cross_offsets = -im_profiles_2d[:, :-1] / diffs
        cross_offsets[np.bitwise_or(cross_offsets>=1, cross_offsets<0)] = np.NAN
        cross = cross_offsets + np.tile(np.arange(0, cross_offsets.shape[1]), (cross_offsets.shape[0], 1))


        #Separate crossings according to whether they are positive -> negative
        #or negative -> positive, i.e. whether they are potentially rising or
        #falling edges, and only allow those on the appropriate size of the centerline
        cross_rising0 = cross.copy()
        cross_rising0[np.bitwise_or(diffs > 0, cross_rising0 > col_central)] = np.NAN
        cross_falling0 = cross.copy()
        cross_falling0[np.bitwise_or(diffs < 0, cross_falling0 < col_central)] = np.NAN

        #Look for the vessel edges, with or without a connectivity test.
        if not enforce_connectedness:
            search_left = im_profiles_2d[:, left_mean_col] <= 0
            col_left = find_closest_crossing(cross_rising0.copy(), left_mean_col, search_left)
            search_left = im_profiles_2d[:, right_mean_col] >= 0
            col_right = find_closest_crossing(cross_falling0.copy(), right_mean_col, search_left)

        else:
            cross_rising = find_most_connected_crossings(cross_rising0.copy(), left_mean_col, np.maximum(width_estimate/3.0, 1))
            col_left = np.nanmax(cross_rising, 1)
            cross_falling = find_most_connected_crossings(cross_falling0.copy(), right_mean_col, np.maximum(width_estimate/3.0, 1))
            col_right = np.nanmin(cross_falling, 1)


        #% Compute the side points, and store them
        rows = np.arange(0, n_profiles).reshape(-1, 1)
        inds_found = np.bitwise_and(np.isfinite(col_left), np.isfinite(col_right))
        dict_side1[vesselKey] = get_side(im_profiles_rows, im_profiles_cols, rows[inds_found], col_left[inds_found].reshape(-1, 1))
        dict_side2[vesselKey] = get_side(im_profiles_rows, im_profiles_cols, rows[inds_found], col_right[inds_found].reshape(-1,1))

        sidePointList1 = np.array([dict_side1[vesselKey][0].reshape(-1), dict_side1[vesselKey][1].reshape(-1)])
        sidePointList2 = np.array([dict_side2[vesselKey][0].reshape(-1), dict_side2[vesselKey][1].reshape(-1)])
        dict_smoothSide1[vesselKey] = splineInterpolation(sidePointList1, order=2)
        dict_smoothSide2[vesselKey] = splineInterpolation(sidePointList2, order=2)
        dict_vesselWidth[vesselKey] = np.hypot((dict_side1[vesselKey][0] - dict_side2[vesselKey][0]),
                                               (dict_side1[vesselKey][1] - dict_side2[vesselKey][1]))
        # dict_vesselWidth[vesselKey] = np.hypot((dict_smoothSide1[vesselKey][0] - dict_smoothSide2[vesselKey][0]),
        #                                        (dict_smoothSide1[vesselKey][1] - dict_smoothSide2[vesselKey][1]))


    """Remove the vessels where the vessel side is not detected correctly"""
    for vesselKey in dict_side1.keys():
        if len(dict_side1[vesselKey][0]) >= 20:
            pass
        else:  ##delete the vessels where the side is not detected correctly
            dict_side1.pop(vesselKey)
            dict_side2.pop(vesselKey)
            dict_smoothSide1.pop(vesselKey)
            dict_smoothSide2.pop(vesselKey)

            dict_splinePoints.pop(vesselKey)
            dict_profileIntensity.pop(vesselKey)
            dict_vesselWidth.pop(vesselKey)
            dict_vesselAngles.pop(vesselKey)


    ########################################################################
    """End of vessel profile measurement"""
    return dict_splinePoints, dict_profileIntensity, dict_side1, dict_side2, \
           dict_smoothSide1, dict_smoothSide2, dict_vesselWidth, dict_vesselAngles, dict_profileRows, dict_profileCols

    # return dict_splinePoints,  dict_profileIntensity, dict_profileSide, dict_side1, dict_side2, \
    #        dict_vesselWidth, dict_vesselAngles