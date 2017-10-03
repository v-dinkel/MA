
import cv2
import numpy as np
from scipy.sparse import csr_matrix

from DetectEndPoints import detectEndPoints_2value
from RemoveBranch import removeBranchPoints



###The faster method, equal to np.where
def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)),
           shape=(data.max() + 1, data.size))
    indexes = [np.unravel_index(row.data, data.shape) for row in M]
    return indexes


"""This is the profiled code, takes around 0.5s to process"""
def getSkeletonPixelLocs(BranchRemovedImg_labelC):

    ###Get the endpoint list and dictionary of vessel pixel locations#######
    BranchRemovedImg_C = np.zeros(BranchRemovedImg_labelC.shape[:2], dtype=np.uint8)
    BranchRemovedImg_C[BranchRemovedImg_labelC>0] = 1

    endPoints_List, EndPoints_branchRemoved = detectEndPoints_2value(BranchRemovedImg_C)
    EndPoints_branchRemoved = np.uint8(EndPoints_branchRemoved)

    EndPoints_branchRemoved_labeled = cv2.bitwise_and(BranchRemovedImg_labelC, BranchRemovedImg_labelC,
                                                      mask=EndPoints_branchRemoved)

    allLabelList = np.unique(BranchRemovedImg_labelC)
    allLabelList = allLabelList[allLabelList!=0]
    allIndexes = get_indices_sparse(BranchRemovedImg_labelC)
    allEndpointIndex = get_indices_sparse(EndPoints_branchRemoved_labeled)

    dict_segmentPixelLocs = {}
    dict_chainCode = {}
    dict_endPointIndex = {}

    for label in allLabelList:  # labelNum+1
        labelIndex = np.array(allIndexes[label]).T

        if labelIndex.shape[0] > 2:
            ###get the start and end points####
            dict_endPointIndex[str(label)] = allEndpointIndex[label]

            ##########Track the skeleton centerline#########################
            chainCode = np.zeros(labelIndex.shape[0])
            indexSequence = np.zeros((labelIndex.shape[0], 2))
            startPoint = [dict_endPointIndex[str(label)][0][0], dict_endPointIndex[str(label)][1][0]]
            endPoint = [dict_endPointIndex[str(label)][0][1], dict_endPointIndex[str(label)][1][1]]
            TempImage = BranchRemovedImg_labelC.copy()
            for i in xrange(0, labelIndex.shape[0]):  # labelIndex.shape[0]
                indexSequence[i, :] = [startPoint[0], startPoint[1]]
                TempImage[startPoint[0], startPoint[1]] = 0
                if not (startPoint[0] == endPoint[0] and startPoint[1] == endPoint[1]):
                    if TempImage[startPoint[0], startPoint[1] + 1] == label:
                        chainCode[i] = 1
                        startPoint = [startPoint[0], startPoint[1] + 1]
                    elif TempImage[startPoint[0] - 1, startPoint[1] + 1] == label:
                        chainCode[i] = 2
                        startPoint = [startPoint[0] - 1, startPoint[1] + 1]
                    elif TempImage[startPoint[0] - 1, startPoint[1]] == label:
                        chainCode[i] = 3
                        startPoint = [startPoint[0] - 1, startPoint[1]]
                    elif TempImage[startPoint[0] - 1, startPoint[1] - 1] == label:
                        chainCode[i] = 4
                        startPoint = [startPoint[0] - 1, startPoint[1] - 1]
                    elif TempImage[startPoint[0], startPoint[1] - 1] == label:
                        chainCode[i] = 5
                        startPoint = [startPoint[0], startPoint[1] - 1]
                    elif TempImage[startPoint[0] + 1, startPoint[1] - 1] == label:
                        chainCode[i] = 6
                        startPoint = [startPoint[0] + 1, startPoint[1] - 1]
                    elif TempImage[startPoint[0] + 1, startPoint[1]] == label:
                        chainCode[i] = 7
                        startPoint = [startPoint[0] + 1, startPoint[1]]
                    elif TempImage[startPoint[0] + 1, startPoint[1] + 1] == label:
                        chainCode[i] = 8
                        startPoint = [startPoint[0] + 1, startPoint[1] + 1]
                    else:
                        chainCode[i] = -1
                else:
                    pass

            #########Get the parameters##############################################
            dict_segmentPixelLocs[str(label)] = indexSequence
            dict_chainCode[str(label)] = chainCode


        else:
            dict_segmentPixelLocs[str(label)] = labelIndex
            dict_chainCode[str(label)] = 0
            dict_endPointIndex[str(label)] = (np.array([-1, -1]), np.array([-1, -1]))

    return dict_segmentPixelLocs, endPoints_List, dict_chainCode




"""This program takes around 1.4s to process"""
def getSkeletonPixelLocs_bak(VesselSkeleton_C, BranchRemovedImg_labelC):

    ###Get the endpoint list and dictionary of vessel pixel locations#######
    BranchRemovedImg_C, branchPoint_List = removeBranchPoints(VesselSkeleton_C)
    BranchRemovedImg_C[BranchRemovedImg_C>0] = 1

    endPoints_List, EndPoints_branchRemoved = detectEndPoints_2value(BranchRemovedImg_C)
    EndPoints_branchRemoved = np.uint8(EndPoints_branchRemoved)

    EndPoints_branchRemoved_labeled = cv2.bitwise_and(BranchRemovedImg_labelC, BranchRemovedImg_labelC,
                                                      mask=EndPoints_branchRemoved)

    allLabelList = np.unique(BranchRemovedImg_labelC)
    allLabelList = allLabelList[allLabelList!=0]

    dict_segmentPixelLocs = {}
    dict_chainCode = {}
    dict_endPointIndex = {}

    for label in allLabelList:  # labelNum+1
        labelIndex = np.argwhere(BranchRemovedImg_labelC == label)
        # labelIndex = labelIndex[:, 1:]

        if len(labelIndex) >= 2:
            ###get the start and end points####
            dict_endPointIndex[str(label)] = np.where(EndPoints_branchRemoved_labeled == label)
            # dict_endPointIndex[str(label)] = get_indices_sparse(EndPoints_branchRemoved_labeled == label)

            ##########Track the skeleton centerline#########################
            chainCode = np.zeros(labelIndex.shape[0])
            indexSequence = np.zeros((labelIndex.shape[0], 2))
            startPoint = [dict_endPointIndex[str(label)][0][0], dict_endPointIndex[str(label)][1][0]]
            endPoint = [dict_endPointIndex[str(label)][0][1], dict_endPointIndex[str(label)][1][1]]
            TempImage = BranchRemovedImg_labelC.copy()
            for i in xrange(0, labelIndex.shape[0]):  # labelIndex.shape[0]
                indexSequence[i, :] = [startPoint[0], startPoint[1]]
                TempImage[startPoint[0], startPoint[1]] = 0
                if not (startPoint[0] == endPoint[0] and startPoint[1] == endPoint[1]):
                    if TempImage[startPoint[0], startPoint[1] + 1] == label:
                        chainCode[i] = 1
                        startPoint = [startPoint[0], startPoint[1] + 1]
                    elif TempImage[startPoint[0] - 1, startPoint[1] + 1] == label:
                        chainCode[i] = 2
                        startPoint = [startPoint[0] - 1, startPoint[1] + 1]
                    elif TempImage[startPoint[0] - 1, startPoint[1]] == label:
                        chainCode[i] = 3
                        startPoint = [startPoint[0] - 1, startPoint[1]]
                    elif TempImage[startPoint[0] - 1, startPoint[1] - 1] == label:
                        chainCode[i] = 4
                        startPoint = [startPoint[0] - 1, startPoint[1] - 1]
                    elif TempImage[startPoint[0], startPoint[1] - 1] == label:
                        chainCode[i] = 5
                        startPoint = [startPoint[0], startPoint[1] - 1]
                    elif TempImage[startPoint[0] + 1, startPoint[1] - 1] == label:
                        chainCode[i] = 6
                        startPoint = [startPoint[0] + 1, startPoint[1] - 1]
                    elif TempImage[startPoint[0] + 1, startPoint[1]] == label:
                        chainCode[i] = 7
                        startPoint = [startPoint[0] + 1, startPoint[1]]
                    elif TempImage[startPoint[0] + 1, startPoint[1] + 1] == label:
                        chainCode[i] = 8
                        startPoint = [startPoint[0] + 1, startPoint[1] + 1]
                    else:
                        chainCode[i] = -1
                else:
                    pass

            #########Get the parameters##############################################
            dict_segmentPixelLocs[str(label)] = indexSequence
            dict_chainCode[str(label)] = chainCode


        else:
            dict_segmentPixelLocs[str(label)] = labelIndex
            dict_chainCode[str(label)] = 0
            dict_endPointIndex[str(label)] = (np.array([-1, -1]), np.array([-1, -1]))

    return dict_segmentPixelLocs, dict_chainCode, endPoints_List, branchPoint_List


#############################################
####"""This is the unsorted vessel segments location finding program"""

# import numpy as np
# from skimage import measure
#
# from VesselTree.DetectEndPoints import detectEndPoints
# from VesselTree.BranchDetection import branchedPointsDetection
# from VesselTree.RemoveSpur_3 import removeSpur
# from VesselTree.RemoveBranch import removeBranchPoints
# from VesselTree.CircularRegionMask import diskMask, regionBMask, regionCMask
#
#

# def getSkeletonPixelLocs(VesselSkeleton_Pruned):
#     ###Get the endpoint list and dictionary of vessel pixel locations#######
#     endPoints_List = detectEndPoints(VesselSkeleton_Pruned)
#
#     BranchRemovedImg, branchPoint_List = removeBranchPoints(VesselSkeleton_Pruned)
#
#     BranchRemovedImg_label = measure.label(BranchRemovedImg)
#
#     dict_segmentPixelLocs = {}
#     for templabel in xrange(1, np.max(BranchRemovedImg_label) + 1):
#         pixLoc = np.where(BranchRemovedImg_label == templabel)
#         dict_segmentPixelLocs[str(templabel)] = pixLoc
#
#
#     return dict_segmentPixelLocs, endPoints_List, branchPoint_List
#################################################################################
