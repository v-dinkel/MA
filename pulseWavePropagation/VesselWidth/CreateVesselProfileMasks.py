
import cv2
import numpy as np

def createVesselProfileMasks(dict_profileRows, dict_profileCols, Img_BW, Mask):
    Img_BW[Img_BW>0]=1
    Mask[Mask>0]=1


    ##dict_profileRows: stores all the row indexes for the vessel cross profile lines
    ##dict_profileCols: stores all the Col indexes for the vessel cross profile lines
    height, width = Img_BW.shape[:2]


    dict_ProfileMap = {}
    dict_BWVesselProfiles = {}
    dict_BWRegionProfiles = {}
    for vesselKey in dict_profileRows.keys():
        rows = np.int16(np.round(dict_profileRows[vesselKey]))
        cols = np.int16(np.round(dict_profileCols[vesselKey]))
        ProfileMap = Img_BW[rows, cols]
        # try:
        #     ProfileMap = Img_BW[rows, cols]
        # except:
        #     print 'Error key:', vesselKey
        dict_ProfileMap[vesselKey] = ProfileMap
        dict_BWVesselProfiles[vesselKey] = get_centerline_object(ProfileMap)
        dict_BWRegionProfiles[vesselKey] = ProfileMap == dict_BWVesselProfiles[vesselKey]

    # BW_VesselProfile_all = np.bitwise_and(ProfileMap, Img_BW)  ###equal to bw_vessel_profiles_all
    #
    # BW_vessel_profiles = get_centerline_object(BW_VesselProfile_all)
    #
    # BW_region_profiles = BW_vessel_profiles == BW_VesselProfile_all

    return dict_ProfileMap, dict_BWVesselProfiles, dict_BWRegionProfiles




def get_centerline_object(BW):

    ResultImg = BW.copy()
    col_center = BW.shape[1] // 2  ##get the center of the column

    Img_col_center = BW[:, col_center]
    for ii in xrange(col_center, BW.shape[1]):   ###Right part of the image
        Img_col_center = np.bitwise_and(Img_col_center, BW[:, ii])
        if not np.any(Img_col_center):
            ResultImg[:, ii:] = 0
            break
        ResultImg[:, ii] = Img_col_center

    Img_col_center = BW[:, col_center]
    for ii in xrange(col_center-1, -1, -1):  ###Left part of the image
        Img_col_center = np.bitwise_and(Img_col_center, BW[:, ii])
        if not np.any(Img_col_center):
            ResultImg[:, :ii] = 0
            break
        ResultImg[:, ii] = Img_col_center

    return ResultImg







