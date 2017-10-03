###The return file format is: (array1, array2), array1 is the row numbers and array2 is the column numbers


import cv2
import numpy as np
import mahotas as mh



"""These are endpoint template"""
endpoint1=np.array([[0, 0, 0],
                        [0, 1, 0],
                        [2, 1, 2]])

endpoint2=np.array([[0, 0, 0],
                    [0, 1, 2],
                    [0, 2, 1]])

endpoint3=np.array([[0, 0, 2],
                    [0, 1, 1],
                    [0, 0, 2]])

endpoint4=np.array([[0, 2, 1],
                    [0, 1, 2],
                    [0, 0, 0]])

endpoint5=np.array([[2, 1, 2],
                    [0, 1, 0],
                    [0, 0, 0]])

endpoint6=np.array([[1, 2, 0],
                    [2, 1, 0],
                    [0, 0, 0]])

endpoint7=np.array([[2, 0, 0],
                    [1, 1, 0],
                    [2, 0, 0]])

endpoint8=np.array([[0, 0, 0],
                    [2, 1, 0],
                    [1, 2, 0]])

"""This code takes around 0.15s"""
def  detectEndPoints(skel):
    skel[skel>0]=1
    ep1=mh.morph.hitmiss(skel,endpoint1)
    ep2=mh.morph.hitmiss(skel,endpoint2)
    ep3=mh.morph.hitmiss(skel,endpoint3)
    ep4=mh.morph.hitmiss(skel,endpoint4)
    ep5=mh.morph.hitmiss(skel,endpoint5)
    ep6=mh.morph.hitmiss(skel,endpoint6)
    ep7=mh.morph.hitmiss(skel,endpoint7)
    ep8=mh.morph.hitmiss(skel,endpoint8)
    EndpointImg = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    endPoionts_List = np.where(EndpointImg>0)

    return endPoionts_List


"""This code takes around 0.15s"""
def  detectEndPoints_2value(skel):
    skel[skel>0]=1
    ep1=mh.morph.hitmiss(skel,endpoint1)
    ep2=mh.morph.hitmiss(skel,endpoint2)
    ep3=mh.morph.hitmiss(skel,endpoint3)
    ep4=mh.morph.hitmiss(skel,endpoint4)
    ep5=mh.morph.hitmiss(skel,endpoint5)
    ep6=mh.morph.hitmiss(skel,endpoint6)
    ep7=mh.morph.hitmiss(skel,endpoint7)
    ep8=mh.morph.hitmiss(skel,endpoint8)
    EndpointImg = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    endPoionts_List = np.where(EndpointImg>0)

    return endPoionts_List, EndpointImg



############################################################################################

"""This detect end point funciton is very fast, takes less than 0.01s, but this is not accurate for special endpoint."""
def detectEndPoints_Bak(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    # # now look through to find the value of 11
    # # this returns a mask of the endpoints, but if you just want the coordinates, you could simply return np.where(filtered==11)
    # out = np.zeros_like(skel)
    # out[np.where(filtered==11)] = 1

    endPoionts_List = np.where(filtered == 11)
    return endPoionts_List

"""This detect end point funciton is very fast, takes less than 0.01s, but this is not accurate for special endpoint."""
def detectEndPoints_2value_Bak(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    # # now look through to find the value of 11
    # # this returns a mask of the endpoints, but if you just want the coordinates, you could simply return np.where(filtered==11)
    # out = np.zeros_like(skel)
    # out[np.where(filtered==11)] = 1

    endPoionts_List = np.where(filtered == 11)
    EndpointImg = np.zeros(skel.shape[:2], dtype= np.uint8)
    EndpointImg[np.where(filtered == 11)] = 1

    return endPoionts_List, EndpointImg



"""This detect end point funciton is very slow, takes around 3s."""

def detectEndPoints0_bak(Skeleton):
    height, width = Skeleton.shape
    endPoints = np.zeros((height, width), dtype=bool)
    for i in xrange(1, height-1):
        for j in xrange(1, width-1):
            if Skeleton[i,j] == 1 and np.count_nonzero(Skeleton[i-1:i+2, j-1:j+2]) == 2:
                endPoints[i,j] = True
                # cv2.circle(Img_Show, (j, i), radius=3,  color = 200, thickness=-1)
            elif Skeleton[i,j] == 1 and np.count_nonzero(Skeleton[i-1:i+2, j-1:j+2]) == 3:
                Skeleton[i, j] = 0
                nonzeroIndex = np.argwhere(Skeleton[i-1:i+2, j-1:j+2] == 1)
                if np.abs(nonzeroIndex[0][0] - nonzeroIndex[1][0]) + np.abs(nonzeroIndex[0][1] - nonzeroIndex[1][1]) == 1:
                    endPoints[i,j] = True
                    # cv2.circle(Img_Show, (j, i), radius=3,  color = 200, thickness=-1)
                Skeleton[i, j] = 1

    endPoionts_List = np.where(endPoints==True)

    return endPoionts_List

"""This detect end point funciton is very slow, takes around 3s."""

def detectEndPoints2_bak(Skeleton):
    height, width = Skeleton.shape
    endPoints = np.zeros((height, width), dtype=bool)
    for i in xrange(1, height-1):
        for j in xrange(1, width-1):
            if Skeleton[i,j] == 1 and np.count_nonzero(Skeleton[i-1:i+2, j-1:j+2]) == 2:
                endPoints[i,j] = True
                # cv2.circle(Img_Show, (j, i), radius=3,  color = 200, thickness=-1)
            elif Skeleton[i,j] == 1 and np.count_nonzero(Skeleton[i-1:i+2, j-1:j+2]) == 3:
                Skeleton[i, j] = 0
                nonzeroIndex = np.argwhere(Skeleton[i-1:i+2, j-1:j+2] == 1)
                if abs(nonzeroIndex[0][0] - nonzeroIndex[1][0]) + abs(nonzeroIndex[0][1] - nonzeroIndex[1][1]) == 1:
                    endPoints[i,j] = True
                    # cv2.circle(Img_Show, (j, i), radius=3,  color = 200, thickness=-1)
                Skeleton[i, j] = 1


    endPoionts_List = np.where(endPoints==True)


    return endPoionts_List, endPoints