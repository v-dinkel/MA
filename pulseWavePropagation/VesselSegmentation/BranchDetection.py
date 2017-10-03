from __future__ import division

import numpy as np
import mahotas as mh

"""This program runs aobut 0.27s, cannot use numba.autojit, which makes it even slower"""
def branchedPointsDetection(skel):
    skel[skel>0]=1

    xbranch0  = np.array([[1,0,1],[0,1,0],[1,0,1]])
    xbranch1 = np.array([[0,1,0],[1,1,1],[0,1,0]])
    xbranch2 = np.array([[1,1],[1,1]])

    tbranch0 = np.array([[0,0,0],[1,1,1],[0,1,0]])
    tbranch1 = np.flipud(tbranch0)  #np.array([[0, 1, 0],[1, 1, 1],[0, 0, 0]]) #
    tbranch2 = tbranch0.T   #np.array([[0, 1, 0],[0, 1, 1],[0, 1, 0]]) #
    tbranch3 = np.fliplr(tbranch2)  #np.array([[0, 1, 0],[1, 1, 0],[0, 1, 0]]) #
    tbranch4 = np.array([[1,0,1],[0,1,0],[1,0,0]])
    tbranch5 = np.flipud(tbranch4)  #np.array([[1, 0, 0],[0, 1, 0],[1, 0, 1]])#
    tbranch6 = np.fliplr(tbranch4)  #np.array([[1, 0, 1],[0, 1, 0],[0, 0, 1]])#
    tbranch7 = np.fliplr(tbranch5)  #np.array([[0, 0, 1],[0, 1, 0],[1, 0, 1]])#

    ybranch0 = np.array([[1,0,1],[0,1,0],[2,1,2]])
    ybranch1 = np.flipud(ybranch0)  #np.array([[2, 1, 2],[0, 1, 0],[1, 0, 1]]) #
    ybranch2 = ybranch0.T   #np.array([[1, 0, 2],[0, 1, 1],[1, 0, 2]]) #
    ybranch3 = np.fliplr(ybranch2)  #np.array([[2, 0, 1],[1, 1, 0],[2, 0, 1]]) #

    ybranch4 = np.array([[0,1,2],[1,1,2],[2,2,1]])
    ybranch5 = np.flipud(ybranch4) #np.array([[2, 2, 1],[1, 1, 2],[0, 1, 2]])#
    ybranch6 = np.fliplr(ybranch4) #np.array([[2, 1, 0],[2, 1, 1],[1, 2, 2]])#
    ybranch7 = np.fliplr(ybranch5) #np.array([[1, 2, 2],[2, 1, 1],[2, 1, 0]])#

    x0=mh.morph.hitmiss(skel,xbranch0)   #mh.morph.hitmiss
    x1=mh.morph.hitmiss(skel,xbranch1)
    x2=mh.morph.hitmiss(skel,xbranch2)

    t0=mh.morph.hitmiss(skel,tbranch0)
    t1=mh.morph.hitmiss(skel,tbranch1)
    t2=mh.morph.hitmiss(skel,tbranch2)
    t3=mh.morph.hitmiss(skel,tbranch3)
    t4=mh.morph.hitmiss(skel,tbranch4)
    t5=mh.morph.hitmiss(skel,tbranch5)
    t6=mh.morph.hitmiss(skel,tbranch6)
    t7=mh.morph.hitmiss(skel,tbranch7)

    y0=mh.morph.hitmiss(skel,ybranch0)
    y1=mh.morph.hitmiss(skel,ybranch1)
    y2=mh.morph.hitmiss(skel,ybranch2)
    y3=mh.morph.hitmiss(skel,ybranch3)
    y4=mh.morph.hitmiss(skel,ybranch4)
    y5=mh.morph.hitmiss(skel,ybranch5)
    y6=mh.morph.hitmiss(skel,ybranch6)
    y7=mh.morph.hitmiss(skel,ybranch7)

    branchPoints = t0+t1+t2+t3+t4+t5+t6+t7+y0+y1+y2+y3+y4+y5+y6+y7
    crossPoitns = x0+x1+x2

    return branchPoints, crossPoitns

