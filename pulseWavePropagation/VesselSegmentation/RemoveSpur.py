###Remove spurs using graphical theory


import cv2
import numpy as np
import mahotas as mh
from GraphModel import Skeleton2Graph, dijkstra_search


def removeSpur(Skeleton, endPoionts_List, newBranchList, spurLength):
    Skeleton[Skeleton>0]=1

    height, width = Skeleton.shape

    endPointRegionMask = np.zeros((height, width))
    endPointRegionMask = np.uint8(endPointRegionMask)

    for i in xrange(0, len(endPoionts_List[0])):
        cv2.circle(endPointRegionMask, (endPoionts_List[1][i], endPoionts_List[0][i]), radius=spurLength,  color = 1, thickness=-1)

    for i in xrange(0, len(newBranchList[0])):
        endPointRegionMask[newBranchList[0][i], newBranchList[1][i]] = 0

    branchInsideEndRegionTemplate = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    BranchInsideEndRegion = mh.morph.hitmiss(endPointRegionMask, branchInsideEndRegionTemplate)
    branchInsideEndRegion_index = np.argwhere(BranchInsideEndRegion == True)
    BranchEndRegionMask = np.zeros((height, width))
    BranchEndRegionMask = np.uint8(BranchEndRegionMask)
    for i in xrange(0, branchInsideEndRegion_index.shape[0]):
        cv2.circle(BranchEndRegionMask, (branchInsideEndRegion_index[i, 1], branchInsideEndRegion_index[i, 0]),
                   radius=int(1.5*spurLength),  color = 1, thickness=-1)

    for i in xrange(0, len(endPoionts_List[0])):
        BranchEndRegionMask[endPoionts_List[0][i], endPoionts_List[1][i]] = 0
    endPointInsideBranchRegionTemplate = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    EndPointInsideBranchRegion = mh.morph.hitmiss(BranchEndRegionMask, endPointInsideBranchRegionTemplate)
    endPointInsideBranchRegion_Index = np.argwhere(EndPointInsideBranchRegion == True)
    # for i in xrange(0, endPointInsideBranchRegion_Index.shape[0]):
    #     cv2.circle(VesselSkeleton_show, (endPointInsideBranchRegion_Index[i, 1], endPointInsideBranchRegion_Index[i, 0]),
    #                radius=3,  color = 0.8, thickness=-1)

    #####locate the corresponding branch point and end point
    BranchEnd_Points = [(x, y) for (x , y) in branchInsideEndRegion_index]
    EndBranch_Points = [(x, y) for (x , y) in endPointInsideBranchRegion_Index]

    # print EndBranch_Points
    # print BranchEnd_Points

    Correspond_BranchEnd_points = []
    for i in xrange(0, len(EndBranch_Points)):
        EndBranch_dist = np.zeros(len(BranchEnd_Points))
        for j in xrange(0, len(BranchEnd_Points)):
            EndBranch_dist[j] = (BranchEnd_Points[j][0] - EndBranch_Points[i][0])**2 + (BranchEnd_Points[j][1] - EndBranch_Points[i][1])**2
        mindist_index = np.argmin(EndBranch_dist)
        Correspond_BranchEnd_points.append(BranchEnd_Points[mindist_index])

    # print EndBranch_Points
    # print Correspond_BranchEnd_points


    VesselSkeleton_Pruned0 = Skeleton.copy()
    for i in xrange(0, len(EndBranch_Points)):
        minRow, maxRow = min(EndBranch_Points[i][0], Correspond_BranchEnd_points[i][0]), max(EndBranch_Points[i][0], Correspond_BranchEnd_points[i][0])
        minCol, maxCol = min(EndBranch_Points[i][1], Correspond_BranchEnd_points[i][1]), max(EndBranch_Points[i][1], Correspond_BranchEnd_points[i][1])
        PatchSkeleton = Skeleton[minRow - 3:maxRow + 3, minCol-3:maxCol+3]
        point1 = (EndBranch_Points[i][0] - minRow +3, EndBranch_Points[i][1]-minCol+3)
        point2 = (Correspond_BranchEnd_points[i][0] - minRow +3, Correspond_BranchEnd_points[i][1]-minCol+3)

        SkeletonGraph = Skeleton2Graph(PatchSkeleton)
        flag, cost, path = dijkstra_search(SkeletonGraph, point1, point2)

        if flag == True:
            for point in path:
                point_img = [point[0] + minRow -3, point[1]+minCol-3]
                # print EndBranch_Points[i], Correspond_BranchEnd_points[i], point, point_img
                VesselSkeleton_Pruned0[point_img[0], point_img[1]] = 0
            VesselSkeleton_Pruned0[Correspond_BranchEnd_points[i][0], Correspond_BranchEnd_points[i][1]] = 1
        else:
            pass


    VesselSkeleton_Pruned= VesselSkeleton_Pruned0
    # VesselSkeleton_Pruned1 = removeSpur1(Skeleton)
    # VesselSkeleton_Pruned = pymorph.union(pymorph.binary(VesselSkeleton_Pruned0), pymorph.binary(VesselSkeleton_Pruned1))

    return VesselSkeleton_Pruned




