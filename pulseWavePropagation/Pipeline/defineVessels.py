import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
from Tools import utils
from Configuration import Config as cfg

splines = {}
img = []
fig = []

veins = []
arteries = []
cfgimgDir = ""

def selectVesselsManually(imgDir, ImgMask):
    path = imgDir + "pipeline_steps\\vesselWidthMeasurement\\"+ImgMask+"\\dict_splinePoints_updated.npy"
    global splines
    splines = np.load(path).item()
    global img
    img = cv2.imread(imgDir+ImgMask+".tif")
    global cfgimgDir
    cfgimgDir = imgDir
    for spline in splines.keys():
        utils.drawSpline(splines[spline],(0,255,0),img)

    global fig
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return arteries, veins

def onclick(event):
    selectedSpline = findClosestSpline(event.xdata, event.ydata)

    global arteries
    global veins
    col = (0,255,0)
    if event.button == 1:
        col = (255,0,0)
        if selectedSpline not in arteries:
            arteries.append(selectedSpline)
        if selectedSpline in veins:
            veins.remove(selectedSpline)
        print selectedSpline, ' Artery'
    if event.button == 2:
        col = (0, 255, 0)
        if selectedSpline in arteries:
            arteries.remove(selectedSpline)
        if selectedSpline in veins:
            veins.remove(selectedSpline)
        print selectedSpline, ' None'
    if event.button == 3:
        col = (0, 0, 255)
        if selectedSpline not in veins:
            veins.append(selectedSpline)
        if selectedSpline in arteries:
            arteries.remove(selectedSpline)
        print selectedSpline, ' Vein'

    print arteries
    print veins

    global fig
    oldFig = fig

    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(1,1,1)
    utils.drawSpline(splines[selectedSpline], col, img )
    ax.imshow(img)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.close(oldFig)
    plt.savefig(cfgimgDir + 'results\\vesselClasses.png')
    plt.show()

    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))

def findClosestSpline(x,y):
    minKey = ''
    minDist = 99999
    for spline in splines.keys():
        splinePoints = splines[spline]
        diff = min(abs(np.array(splinePoints[0]) - y) + abs(np.array(splinePoints[1]) - x))
        #print spline, ' has the diff = ', diff
        if diff<minDist:
            minKey = spline
            minDist = diff
    print 'selected spline: ',minKey
    return minKey