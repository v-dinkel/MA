import time
import cv2
from OpticDiscDetection.DiscDetection_IR import discDetection_IR
import numpy as np

def opticDiscDetection(imgDir, SingleImageFolder, Img, Img_Resized, Mask):

    time_step3_start = time.time()

    discCenter, discRadius = discDetection_IR(Img, Mask, imgDir, SingleImageFolder)
    print 'Disk Parameter: ', discCenter, discRadius
    # import pdb; pdb.set_trace()
    #discCenter, discRadius = (553, 683), 140  # ImgDir3 = (567, 678), 145

    discRegionParameter = {}
    discRegionParameter['rootPointRatio'] = 2  ###this ratio determins the region of searching for root node (rootPoingRatio, 1)
    discRegionParameter['factor_A'] = (2, 1.5)  ##determine the region B
    discRegionParameter['factor_B'] = (3, 1.5)  ##determine the region B
    discRegionParameter['factor_C'] = (8, 1.5)  ##determine the region C
    discRegionParameter['discCenter'] = discCenter  ##determine the region C
    discRegionParameter['discRadius'] = discRadius  ##determine the region C

    ##Mark the disc parameters on the original image
    ImgShow = getImgShow(Img_Resized, discCenter, discRadius)

    tmp = ImgShow.copy()
    cv2.putText(tmp, '('+str(discCenter[0])+','+str(discCenter[1])+'),'+str(discRadius), (discCenter[1], discCenter[0]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(200, 200, 200), thickness=2)
    cv2.imwrite(imgDir + '\\pipeline_steps\\opticDiscDetection\\ImgShow_'+SingleImageFolder+'.tif', tmp)

    time_step3 = time.time() - time_step3_start
    print 'Step 3: Optic Disc Detection finished, spending time:', time_step3
    print '##############################################################'

    return discCenter, discRadius, discRegionParameter, ImgShow

def getStaticValues(Img_Resized, staticDiscRadius, staticDiscCenter):

    discCenter, discRadius = staticDiscCenter, staticDiscRadius  # ImgDir3 = (567, 678), 145
    discRegionParameter = {}
    discRegionParameter['rootPointRatio'] = 2  ###this ratio determins the region of searching for root node (rootPoingRatio, 1)
    discRegionParameter['factor_A'] = (2, 1.5)  ##determine the region B
    discRegionParameter['factor_B'] = (3, 1.5)  ##determine the region B
    discRegionParameter['factor_C'] = (8, 1.5)  ##determine the region C
    discRegionParameter['discCenter'] = discCenter  ##determine the region C
    discRegionParameter['discRadius'] = discRadius  ##determine the region C

    ImgShow =getImgShow(Img_Resized, discCenter, discRadius)

    return discCenter, discRadius, discRegionParameter, ImgShow

def loadValues(imgDir, SingleImageFolder, Img_Resized):
    discRegionParameter = np.load(imgDir + '\\pipeline_steps\\opticDiscDetection\\'+SingleImageFolder+'\\discRegionParameter.npy').item()
    discRadius = discRegionParameter['discRadius']
    discCenter = discRegionParameter['discCenter']
    ImgShow = getImgShow(Img_Resized, discCenter, discRadius)
    return discCenter, discRadius, discRegionParameter, ImgShow

def getImgShow(Img_Resized, discCenter, discRadius):
    ImgShow = Img_Resized.copy()  ###Imgshow is for showing purposes only
    cv2.circle(ImgShow, center=(discCenter[1], discCenter[0]), radius=discRadius, color=(255, 255, 255), thickness=5)
    cv2.circle(ImgShow, center=(discCenter[1], discCenter[0]), radius=2 * discRadius, color=(255, 255, 255),
               thickness=2)  # 2-3 RegionB
    cv2.circle(ImgShow, center=(discCenter[1], discCenter[0]), radius=3 * discRadius, color=(255, 255, 255),
               thickness=2)
    cv2.putText(ImgShow, "Zone B", (discCenter[1] - 50, discCenter[0] - int(2.2 * discRadius)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(200, 200, 200), thickness=2)
    cv2.circle(ImgShow, center=(discCenter[1], discCenter[0]), radius=5 * discRadius, color=(255, 255, 255),
               thickness=2)  # 3-5 REgionC
    cv2.putText(ImgShow, "Zone C", (discCenter[1] - 50, discCenter[0] - int(4 * discRadius)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
    return ImgShow

def loadData(imgDir, SingleImageFolder):
    Mask = np.load(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Mask.npy')
    Img_Resized = cv2.imread(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Img_Resized.tif')
    Img = cv2.imread(imgDir + '\\pipeline_steps\\resizing\\' + SingleImageFolder + '\\Img.tif')
    return Mask, Img_Resized, Img

def saveData(imgDir, SingleImageFolder, discRegionParameter, ImgShow):
    discCenter = discRegionParameter['discCenter']
    discRadius = discRegionParameter['discRadius']

    np.save(imgDir + '\\pipeline_steps\\opticDiscDetection\\' + SingleImageFolder + '\\discRegionParameter.npy',discRegionParameter)
    tmp = ImgShow.copy()
    cv2.putText(tmp, '(' + str(discCenter[0]) + ',' + str(discCenter[1]) + '),' + str(discRadius),
                (discCenter[1], discCenter[0]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(200, 200, 200), thickness=2)
    cv2.imwrite(imgDir + '\\pipeline_steps\\opticDiscDetection\\ImgShow_' + SingleImageFolder + '.tif', tmp)
    return
