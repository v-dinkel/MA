
from __future__ import division
import numpy as np
import cv2
from Tools.Float2Uint import float2Uint
import warnings
import matplotlib.pyplot as plt

"""The Img is Infared Images
Return:
"""


def discDetection_IR(Img, Mask, imgDir, singleImageFolder):

    if len(Img.shape) == 3:
        Img_green = Img[:,:,1]
    else:
        Img_green = Img

    GaussianProb = gaussian_filter(shape =(Img.shape[:2]) , sigma = Img.shape[0]/4)

    BackgroundIllumImage = cv2.medianBlur(Img_green, ksize = 55)
    BackgroundIllumImage2 = BackgroundIllumImage*GaussianProb
    BackgroundIllumImage2 = float2Uint(BackgroundIllumImage2)
    DiscBW =  cv2.adaptiveThreshold(255-BackgroundIllumImage2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                      cv2.THRESH_BINARY, 155, -20)
    #DiscBW[Mask==0] = 0

    Disk_Contour, Hierarchy = cv2.findContours(DiscBW.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    try:
        Areas_all = np.zeros(len(Disk_Contour))
        for i in xrange(len(Disk_Contour)):
            Areas_all[i] = cv2.contourArea(Disk_Contour[i], True)
        Areas_all = abs(Areas_all)
        maxAreaIndex = np.argmax(Areas_all)
        Circle = cv2.minEnclosingCircle(Disk_Contour[maxAreaIndex])
        Center = (int(Circle[0][1]), int(Circle[0][0]))
        Radius = int(Circle[1])

        ''' # THIS IS ALTERNATE CIRCLE DETECTION, WHICH DOESNT WORK GOOD
        output = Img.copy()
        gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        # detect circles in the image
        print 'detecting circles'
    
        # THIS IS HEATMAP OF THE PICTURE
        import itertools
        x = np.array([range(0,Img.shape[0]) for k in range(0,Img.shape[1])])
        xx = list(itertools.chain.from_iterable(x))
        y = gray
        yy = list(itertools.chain.from_iterable(y))
    
        #heatmap, xedges, yedges = np.histogram2d(xx, yy, bins=(1600,1200))
        #extent = [0, 1600, 0, 1200]
    
        plt.clf()
        #plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.imshow(y, cmap='hot', interpolation='nearest')
        plt.savefig(imgDir + "\\test\\"+singleImageFolder+".png")
        plt.show()
        
        #HEATMAP WITH THRESHOLD
        threshold = 60
        yy = []
        for arr in y:
            yy.append([k if k > threshold else 0 for k in arr])
        plt.clf()
        plt.imshow(yy, cmap='hot', interpolation='nearest')
        plt.savefig(imgDir + "\\results\\" + zone + "\\outliers_dendrogram.png")
        #plt.show()
        #import pdb;
        #pdb.set_trace()
    
        #plt.hexbin(xx, yy, cmap=plt.cm.YlOrRd_r, gridsize =(1200,1600))
        #plt.axis([0, 1200, 0, 1600])
        #plt.show()
    
        #cv2.circle(output, center=(Center[1], Center[0]), radius=Radius, color=(255, 255, 255), thickness=2)
        #cv2.imwrite(imgDir + 'test\\disc.png', gray)
        '''

    except:
        warnings.warn('Optic Disk Was not Detected')
        Center, Radius = (DiscBW.shape[0] // 2, DiscBW.shape[1] // 2), 70


    return Center, Radius

def gaussian_filter(shape =(5,5), sigma=1):
    x, y = [edge //2 for edge in shape]
    grid = np.array([[((i**2+j**2)/(2.0*sigma**2)) for i in xrange(-x, x+1)] for j in xrange(-y, y+1)])
    g_filter = np.exp(-grid)/(2*np.pi*sigma**2)
    g_filter /= np.sum(g_filter)
    g_filter = np.transpose(g_filter)
    if g_filter.shape[0] != shape[0] or g_filter.shape[1] != shape[1]:
        g_filter = g_filter[:shape[0], :shape[1]]

    return g_filter

