# https://stackoverflow.com/questions/22279069/how-to-align-multiple-camera-images-using-opencv

import cv2
import time
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from os.path import isfile, join
from os import listdir
import operator
import pdb

from sklearn import datasets, linear_model
from pylab import *

from Configuration import Config as cfg
from Tools import utils

def getInitialCurve(sift, imgRef, referenceKp, referenceDes, imgDir, targetImg, count, curve):

    imgTarget = cv2.imread(imgDir+targetImg)
    targetKp, targetDes = sift.detectAndCompute(imgTarget, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    try:
        matches = bf.knnMatch(referenceDes, targetDes, k=2)
    except:
        print 'exception in get initial curve of movement translation'
        #curve.append(curve[-1])
        return curve[-1]

    # Apply ratio test
    good = []
    kpSizes = {}
    kpMap = {}
    kpDistanceMap = {}
    kpThreshold = 130.0 + 20.0 * float(count)
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    for m, k in matches:
        if m.distance < 0.75 * k.distance:
            # Removed the brackets around m
            good.append(m)
            kpMap[referenceKp[m.queryIdx].size] = (m.queryIdx, m.trainIdx)
            (x1,y1) = referenceKp[m.queryIdx].pt
            (x2,y2) = targetKp[m.trainIdx].pt
            kpDistanceMap[abs(x1-x2)+abs(y1-y2)] = (m.queryIdx, m.trainIdx)

    #out = utils.drawFeatureMatches(imgRef,referenceKp, imgTarget, targetKp, good)
    #cv2.imwrite(imgDir + '\\translation\\keypoints.jpg', out)

    sortedKp = sorted(kpMap.items(), key=operator.itemgetter(0), reverse=True )
    sortedKpDistance = sorted(kpDistanceMap.items(), key=operator.itemgetter(0), reverse=False)

    if len(sortedKpDistance)>0:
        # calculate x/y-displacement
        k = 0
        translateX = 0.0
        translateY = 0.0
        for kp in [i for i in sortedKpDistance if i[0]<kpThreshold]:
            (x1, y1) = referenceKp[kp[1][0]].pt
            (x2, y2) = targetKp[kp[1][1]].pt
            #if abs(x1-x2) < kpThreshold and abs(y1-y2) < kpThreshold:
            translateX += x1-x2
            translateY += y1-y2

            k += 1
            break # this causes to only take one keypoint-matching with the smallest distance
            if k>5:
                break

        #img=cv2.drawKeypoints(imgRef,referenceKp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #img2=cv2.drawKeypoints(imgTarget,targetKp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imwrite(imgDir + '\\translation\\keypoints.jpg', img)
        #cv2.imwrite(imgDir + '\\translation\\keypoints2.jpg', img2)
        #print k, translateX, translateY
        #import pdb; pdb.set_trace()
        try:
            #curve.append((translateX/k,translateY/k))
            return (translateX/k,translateY/k)
        except:
            import pdb;
            pdb.set_trace()
    else:
        print "No Keypoints for image "+targetImg

def performCorrection(X, Y, imgDir, targetImg):
    imgTarget = cv2.imread(imgDir + targetImg)
    # need the correction curve
    M = np.float32([[1, 0, X], [0, 1,  Y]])
    dst = cv2.warpAffine(imgTarget, M, (imgTarget.shape[1], imgTarget.shape[0]))
    cv2.imwrite(imgDir + '\\' + targetImg, dst)

def performBWCorrection(X, Y, imgDir, targetImg, targetName):
    # imgTarget = cv2.imread(imgDir + targetImg)
    # need the correction curve
    M = np.float32([[1, 0, X], [0, 1,  Y]])
    dst = cv2.warpAffine(targetImg, M, (targetImg.shape[1], targetImg.shape[0]))
    cv2.imwrite(imgDir + '\\pipeline_steps\\' + targetName, dst)
    return dst

def getLinearRegression(curve):
    x = [k[0] for k in curve]
    y = [k[1] for k in curve]

    xaxis = [k for k in range(0, len(x))]
    fitx = np.polyfit(xaxis, x, 1)
    fity = np.polyfit(xaxis, y, 1)

    fix_fn = np.poly1d(fitx)
    fiy_fn = np.poly1d(fity)

    X = [fix_fn(xx) for xx in xaxis]
    Y = [fiy_fn(xx) for xx in xaxis]

    return X , Y

def getRawCurve(curve):
    X = [k[0] for k in curve]
    Y = [k[1] for k in curve]

    return X, Y

def getPolynomialCurve(curve):
    points = np.array(curve)
    x = points[:, 0]
    y = points[:, 1]

    xaxis = [k for k in range(0, len(x))]

    # calculate polynomial
    xfit = np.polyfit(xaxis, x, 3)
    xfunc = np.poly1d(xfit)

    yfit = np.polyfit(xaxis, y, 3)
    yfunc = np.poly1d(yfit)

    X = [xfunc(xx) for xx in xaxis]
    Y = [yfunc(xx) for xx in xaxis]

    return X, Y

def getCurveFromMethod(curve, method):
    if method == 'raw':
        correctX, correctY = getRawCurve(curve)
    elif method == 'polynomial':
        correctX, correctY = getPolynomialCurve(curve)
    else:
        correctX, correctY = getLinearRegression(curve)
    return correctX, correctY

def run(dir = cfg.imgDir, method = 'linear'):
    curve = []
    curve.append((0, 0))

    time_start = time.time()
    print 'Translating movement in ', dir
    onlyImages = utils.getListOfImages(dir, cfg.imgFormats)

    imgRef = cv2.imread(dir+onlyImages[0])
    sift = cv2.SIFT()
    referenceKp, referenceDes = sift.detectAndCompute(imgRef,None)

    for i in range(1,len(onlyImages)):
        curve.append(getInitialCurve(sift, imgRef, referenceKp, referenceDes, dir, onlyImages[i], i, curve))

    correctX, correctY = getCurveFromMethod(curve, method)

    for i in range(0,len(onlyImages)):
        performCorrection(correctX[i], correctY[i], dir, onlyImages[i])

    plt.figure()
    x = [k[0] for k in curve]
    y = [k[1] for k in curve]

    for i in range(0, len(onlyImages)):
        plt.plot(x, label='x', color='g')
        plt.plot(y, label='y', color='b')
        plt.plot(correctX, label='fitx', color='c')
        plt.plot(correctY, label='fity', color='m')
        plt.xticks([k for k in range(1,len(x))])
        plt.axvline(x=float(i), color='r')
        plt.grid()
        plt.legend()

        append0 = '0'
        if i>=10.0:
            append0 = ''

        plt.savefig(cfg.imgDir + '\\translation\\' + append0 + str(i) + '.jpg')
        plt.clf()
    total_time = time.time() - time_start
    np.save(dir + '\\translation\\translationX.npy', correctX)
    np.save(dir + '\\translation\\translationY.npy', correctY)
    print 'Translating displacement finished, spending time:', total_time
    print '##############################################################'

#img=cv2.drawKeypoints(img1,kp1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def translateStaticMovement(imgDir, imgName, sameImageMask, ImgNumber):
    imgFolder = imgName.split('.')[0]
    if imgFolder == sameImageMask:
        pass
    else:
        correctX = np.load(imgDir + 'translation\\translationX.npy')
        correctY = np.load(imgDir + 'translation\\translationY.npy')
        mask_bw = np.load(imgDir+'\\pipeline_steps\\vesselSegmentation\\'+sameImageMask+'\\Img_BW.npy')
        transBW = performBWCorrection(correctX[ImgNumber],correctY[ImgNumber],imgDir, mask_bw, imgName)
        np.save(imgDir + 'pipeline_steps\\vesselSegmentation\\'+imgFolder+'\\static_BW.npy', transBW)
        #pdb.set_trace()
        #cv2.imwrite(cfg.imgDir + '\\pipeline_steps\\vesselSegmentation\\' + imgFolder +'\\bw_image.tif', mask_bw)