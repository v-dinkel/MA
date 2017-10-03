from __future__ import division
import numpy as np


def getLinemask(theta,masksize):

    if theta > 90:
        mask = getbasemask(180 - theta,masksize)
        linemask = rotatex(mask)
    else:
        linemask = getbasemask(theta,masksize)
    return linemask

def rotatex(mask):

    h, w = mask.shape
    rotatedmask=np.zeros((h,w))
    for i in xrange(0, h):
        rotatedmask[i, :] = mask[i, -1: :-1]
        # for j in xrange(0, w):
        #     rotatedmask[i,j] = mask[i, w-1-j]
    return rotatedmask


def getbasemask(theta, masksize):

    mask=np.zeros((masksize, masksize))
    halfsize = masksize // 2

    if theta == 0:
        mask[halfsize,:] = 1
    elif theta == 90:
        mask[:,halfsize]=1
    else:
        theta = theta*np.pi / 180
        x0=- halfsize
        y0=int(round(x0 * (np.tan(theta))))
        if y0 < - halfsize:
            y0= - halfsize
            x0 = int(round(y0 / (np.tan(theta))))

        x1=halfsize
        y1=int(round(x1 * (np.tan(theta))))
        if y1 > halfsize:
            y1=halfsize
            x1=int(round(y1 / (np.tan(theta))))

        pt0=[halfsize + y0, halfsize + x0]
        pt1=[halfsize + y1, halfsize + x1]

        mask=drawline(pt0,pt1,mask)
            # mask = np.uint(mask)
    return mask

def drawline(pt0 ,pt1, orgimg ):

    img=orgimg

    # pt0 = [1, 0]
    # pt1 = [3, 4]
    linepts=getlinepts(pt0,pt1)


    # print "points:",  pt0, pt1
    # print linepts

    for i in xrange(0,linepts.shape[0]):
        img[linepts[i,0],linepts[i,1]] = 1
    return img


def getlinepts(P1, P2):

    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)
    # length = int(np.sqrt(dX**2+dY**2))

    #predefine numpy array for output based on distance between points
    pixels = np.empty(shape=(np.maximum(dYa,dXa),2),dtype=np.int)
    pixels.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
        pixels[:,0] = P1X
        if negY:
            pixels[:,1] = np.arange(P1Y - 1, P1Y-dYa-1, -1)
        else:
            pixels[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
    elif P1Y == P2Y: #horizontal line segment
        pixels[:,1] = P1Y
        if negX:
            pixels[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
        else:
            pixels[:,0] = np.arange(P1X+1,P1X+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            # slope = dX.astype(np.float32)/dY.astype(np.float32)
            slope = dY/dX
            if negY:
                pixels[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
            else:
                pixels[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
            pixels[:,0] = ((pixels[:,1]-P1Y)/slope).astype(np.int) + P1X
        else:
            # slope = dY.astype(np.float32)/dX.astype(np.float32)
            slope = dY/dX
            if negX:
                pixels[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
            else:
                pixels[:,0] = np.arange(P1X+1,P1X+dXa+1)
            pixels[:,1] = (slope*(pixels[:,0]-P1X)).astype(np.int) + P1Y


    firstPoint = np.array([[P1[0], P1[1]]])
    linePixels =  np.concatenate((firstPoint, pixels))

    return linePixels

# def getlinepts(pt0, pt1):
#
#     if pt0[1] < pt1[1]:
#         x0=pt0[1]
#         y0=pt0[0]
#         x1=pt1[1]
#         y1=pt1[0]
#     else:
#         x0=pt1[1]
#         y0=pt1[0]
#         x1=pt0[1]
#         y1=pt0[0]
#     dx=x1 - x0
#     dy=y1 - y0
#     ind=0
#     linepts=np.zeros((len(xrange(x0,x1)),2))
#     step=1
#     if dx == 0:
#         x= x0
#         if dy < 0:
#             step=- 1
#         for y in xrange(int(y0), int(y1),step):
#             linepts[ind,:]= [y,x]
#             ind=ind + 1
#     else:
#         if abs(dy) > abs(dx):
#             if dy < 0:
#                 step=- 1
#             for y in xrange(int(y0),int(y1),step):
#                 x=round((dx / dy) * (y - y0) + x0)
#                 linepts[ind,:]=[y,x]
#                 ind=ind + 1
#         else:
#             for x in xrange(x0,x1):
#                 y=round((dy / dx) * (x - x0) + y0)
#                 linepts[ind,:]=[y,x]
#                 ind=ind + 1
#     return linepts

# ###Test######
# a = getLinemask(70, 3)
#
# print a