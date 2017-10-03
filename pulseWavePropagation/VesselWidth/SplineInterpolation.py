from __future__ import division
import numpy as np
from scipy import interpolate

def splineInterpolation(pixelLocs, order):
    ##The pixelLocs passed to this function should be in the format of 2*N array

    # if len(pixelLocs) == 2:
    #     pixelLocs = np.array(pixelLocs).reshape(2, -1)
    # else:
    #     pixelLocs = np.array(pixelLocs).transpose()

    pixelNum = pixelLocs.shape[1]
    if pixelNum >= 11:  #the length of pixels
        controlPoint = pixelLocs[:, np.arange(pixelNum)[::5]]  ##select every other 5 points.
        tck, u = interpolate.splprep(controlPoint, k = order, s=0)
        unew = np.linspace(0, 1.00, pixelNum)
        splinepoints = interpolate.splev(unew, tck)  #splinepoints is in the format of [all rows index, all cols index]

        return splinepoints

    else:
        return pixelLocs

