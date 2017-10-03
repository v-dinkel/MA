
import cv2
import numpy as np
from BranchDetection import branchedPointsDetection

"""This function takes 0.3s."""

def removeBranchPoints(Skeleton):
    Skeleton = np.uint8(Skeleton)
    Skeleton[Skeleton>0]=1

    height, width = Skeleton.shape[:2]

    BranchResults, CrossPointResults = branchedPointsDetection(Skeleton)
    BranchResults = BranchResults + CrossPointResults
    branchPoint_List = np.where(BranchResults == True)

    ###this will perfectly remove all the branch point and split the vessels
    kernel = np.uint8([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    src_depth = -1
    FilterResult = cv2.filter2D(Skeleton, src_depth, kernel)
    BranchRemovedImg = Skeleton.copy()
    BranchRemovedImg[FilterResult >= 13] = 0  ##all the points with more than 3 neibours are removed

    return BranchRemovedImg, branchPoint_List


def removeBranchPoints_bak(Skeleton):

    height, width = Skeleton.shape[:2]

    BranchResults2, CrossPointResults2 = branchedPointsDetection(Skeleton)
    BranchResults2 = BranchResults2 + CrossPointResults2
    branchPoint_List2 = np.where(BranchResults2 == True)

    BranchMask = np.zeros((height, width), dtype=np.uint8)
    for i in xrange(0, len(branchPoint_List2[0])):
        cv2.circle(BranchMask, (branchPoint_List2[1][i], branchPoint_List2[0][i]), radius=2, color=1, thickness=-1)

    BranchImg = cv2.bitwise_and(Skeleton, Skeleton, mask=BranchMask)
    BranchImg = np.uint8(BranchImg)
    BranchRemovedImg = Skeleton - BranchImg

    return BranchRemovedImg, branchPoint_List2