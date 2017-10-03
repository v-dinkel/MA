import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import os
from shutil import copyfile
from matplotlib import pyplot as plt
import seaborn as sns

def loadAllImages(dir, imgFormats, bw=False):
    imgs = []

    for file_name in os.listdir(dir):
        if os.path.splitext(file_name)[-1].lower() in imgFormats:

            #img = cv2.cvtColor(cv2.imread(os.path.join(dir, file_name)), cv2.COLOR_BGR2RGB)[::-1, ::-1, :]
            if bw:
                img = cv2.imread(os.path.join(dir, file_name),0)
            else:
                img = cv2.imread(os.path.join(dir, file_name))
            imgs.append(img)

    return imgs

def getListOfImages(dir, imgFormats):
    #import pdb; pdb.set_trace()
    return [f for f in listdir(dir) if isfile(join(dir, f)) and f.split('.')[1] in [k.replace(".","") for k in imgFormats]]

def createGif(dir, img_names, targetName, targetDir):
    dataDir = dir  # must contain only image files
    # change directory gif directory
    os.chdir(dataDir)

    fileNamesString = " ".join(img_names)
    #import pdb; pdb.set_trace()
    os.system('convert2 -quiet -delay 12 ' + fileNamesString + ' '+targetName+'.gif')
    print fileNamesString

    #if targetDir is not dataDir:
    copyfile(dataDir+'\\'+targetName+'.gif', targetDir+targetName+'.gif')

    return

def drawSpline(spline, color, img):
    #xy1 = zip([x for x in dict_splinePoints_updated['26'][0]], [y for y in dict_splinePoints_updated['26'][1]])
    xy1 = zip([x for x in spline[0]], [y for y in spline[1]])
    for i in range(1, len(xy1)):
        cv2.line(img, (int(round(xy1[i - 1][1])), int(round(xy1[i - 1][0]))),
                 (int(round(xy1[i][1])), int(round(xy1[i][0]))), color, 2)
    return img

def fixMetadataJSON(baseDir):
    if not os.path.isfile(baseDir + 'metadata.json'):
        with open(baseDir+'metadata.txt', 'a') as f:
            f.write('}')
        copyfile(baseDir + 'metadata.txt', baseDir + 'metadata.json')

def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def getDiscFromFile(imgDir):
    f = open(imgDir+'disc.txt')
    for line in f:
        coords = line.replace('makeLine(', '').replace(');', '').replace(' ', '').split(',')
        center = (int(coords[1]),int(coords[0]))
        radius = int(coords[2])-int(coords[0])
    return radius, center

def getSequenceBloodPressure():
    return {    "191":121,
                "191L": 121,
                "200":128,
                "220":128,
                "229":149,
                "234":142,
                "237":104,
                "244":125,
                "262":140,
                "1273":150,
                "1610":107,
                "1641":170,
                "1650":146,
                "182": 145,
                "182L":145,
                "183":106,
                "190":127,
                "197":130,
                "227":132,
                "242":122,
                "317":172,
                "390":154,
                "529":145,
                "722":150,
                "1645":138,
                "1357":137,
                "1361": 136,
                "1374": 128,
                "1374R": 128,
                "1382": 152,
                "1417": 145,
                "1462": 163,
                "1525": 122,
                "1586": 117,
                "1603": 'n/a',
                "1606": 120,
                "1613": 118,
                "1628": 135,
                "1632": 130,
                "1636": 156,
                "1637": 134,
                "1639": 141,
                "1640": 148,
                "1649": 137,
                "217": 121,
                "288": 138,
                "403": 114,
                "445": 115,
                "471": 136,
                "486": 135,
                "660": 118,
                "666": 125,
                "737": 119,
                "757": 107,
                "1262": 102,
                "1272": 167,
                "1272R": 167,
                "1296": 156,
                "1329": 134,
                "194": 136,
                "203": 111,
                "214": 132,
                "236": 103,
                "254": 150,
                "272": 122,
                "273": 134,
                "281": 116,
                "282": 144,
                "322": 121,
                "327": 128,
                "355": 126,
                "409": 120,
                "411": 135,
                "413": 147,
                "511": 149,
                "518": 114,
                "518R": 114,
                "586": 141,
                "588": 140,
                "605": 134,
                "808": 135,
                "834": 144,
                "843": 135,
                "871": 162,
                "1269": 134,
                "1340": 130,
                "1343": 139,
                "1355": 159,
                "1424": 155,
                "1425": "n/a",
                "1478": 106,
                "1634": 132,
                "1643": 149,
                "1647": 124,
                "1652": 114,
                "1653": 142,
                "1654": 149,
                "1209": 106,
                "1260":122,
                "1264":128,
                "1284":118,
                "1387":134,
                "1404":148,
                "1430":158,
                "1537":149,
                "1538":162,
                "1543":168,
                "1547":157,
                "1572":121,
                "1611":160
    }

def getSequenceDiastolicBloodPressure():
    return {
        "182": 75,
        "182L": 75,
        "183": 64,
        "190": 83,
        "191": 82,
        "191L": 82,
        "197": 62,
        "200": 71,
        "220": 77,
        "227": 63,
        "229": 83,
        "234": 84,
        "237": 65,
        "242": 75,
        "244": 84,
        "262": 78,
        "317": 99,
        "390": 81,
        "529": 75,
        "722": 80,
        "1273": 99,
        "1610": 68,
        "1641": 99,
        "1645": 80,
        "1650": 99,
        "1357": 81,
        "1361": 88,
        "1374": 62,
        "1374R": 62,
        "1382": 73,
        "1417": 98,
        "1462": 96,
        "1525": 64,
        "1586": 84,
        "1603": "n/a",
        "1606": 61,
        "1613": 72,
        "1628": 74,
        "1632": 81,
        "1636": 81,
        "1637": 85,
        "1639": 75,
        "1640": 83,
        "1649": 80,
        "217": 74,
        "288": 78,
        "403": 70,
        "445": 50,
        "471": 90,
        "486": 70,
        "660": 73,
        "666": 85,
        "737": 73,
        "757": 61,
        "1262": 64,
        "1272": 84,
        "1272R": 84,
        "1296": 91,
        "1329": 70,
        "194": 80,
        "203": 77,
        "214": 70,
        "236": 83,
        "254": 82,
        "272": 86,
        "273": 85,
        "281": 66,
        "282": 90,
        "322": 69,
        "327": 72,
        "355": 77,
        "409": 73,
        "411": 80,
        "413": 71,
        "511": 79,
        "518": 77,
        "518R": 77,
        "586": 83,
        "588": 68,
        "605": 80,
        "808": 79,
        "834": 79,
        "843": 75,
        "871": 98,
        "1269": 74,
        "1340": 72,
        "1343": 69,
        "1355": 58,
        "1424": 76,
        "1425": "n/a",
        "1478": 75,
        "1634": 73,
        "1643": 91,
        "1647": 74,
        "1652": 66,
        "1653": 57,
        "1654": 81,
        "1209": 70,
        "1260": 69,
        "1264": 75,
        "1284": 77,
        "1387": 77,
        "1404": 96,
        "1430": 100,
        "1537": 85,
        "1538": 85,
        "1543": 98,
        "1547": 83,
        "1572": 85,
        "1611": 101
    }

def doubleMADsfromMedian(y,thresh=3.5):
    # source: https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
    # warning: this function does not check for NAs
    # nor does it address issues when
    # more than 50% of your data have identical values
    m = np.median(y)
    abs_dev = np.abs(y - m)
    left_mad = np.median(abs_dev[y <= m])
    right_mad = np.median(abs_dev[y >= m])
    y_mad = left_mad * np.ones(len(y))
    y_mad[y > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[y == m] = 0
    return modified_z_score > thresh


def hex_code_colors():
    import random
    a = hex(random.randrange(0,256))
    b = hex(random.randrange(0,256))
    c = hex(random.randrange(0,256))
    a = a[2:]
    b = b[2:]
    c = c[2:]
    if len(a)<2:
        a = "0" + a
    if len(b)<2:
        b = "0" + b
    if len(c)<2:
        c = "0" + c
    z = a + b + c
    return "#" + z.upper()

def drawFeatureMatches(img1, kp1, img2, kp2, matches):
    """
    Source: https://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python/26227854#26227854
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

'''
onlyImages = [f for f in listdir(imgDir) if isfile(join(imgDir, f)) and f.split('.')[1] in imgFormats]

imgName = "img_000000000_Default_000.tif"
img = cv2.imread(imgDir+imgName,0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

def printStatisticResults(statistics, statisticsWithoutEl):
    print ''
    print '---------------WITH ELASTICITY: '
    print 'Elasticity importance: ' + str(np.mean([st['featureImportances']['q1'] for st in statistics])) + ' ' + str(
        np.mean([st['featureImportances']['q2'] for st in statistics])) + ' ' + str(
        np.mean([st['featureImportances']['q3'] for st in statistics]))
    print '--PET---'
    print 'Accuracy: ' + str(np.mean([st['PETGroup']['Accuracy'] for st in statistics]))
    print 'Sensitivity: ' + str(np.mean([st['PETGroup']['Sensitivity'] for st in statistics]))
    print 'Specificity: ' + str(np.mean([st['PETGroup']['Specificity'] for st in statistics]))
    print '--MCI---'
    print 'Accuracy: ' + str(np.mean([st['MCIGroup']['Accuracy'] for st in statistics]))
    print 'Sensitivity: ' + str(np.mean([st['MCIGroup']['Sensitivity'] for st in statistics]))
    print 'Specificity: ' + str(np.mean([st['MCIGroup']['Specificity'] for st in statistics]))

    print '--CI---'
    print 'Accuracy: ' + str(np.mean([st['CIGroup']['Accuracy'] for st in statistics]))
    print 'Sensitivity: ' + str(np.mean([st['CIGroup']['Sensitivity'] for st in statistics]))
    print 'Specificity: ' + str(np.mean([st['CIGroup']['Specificity'] for st in statistics]))

    print ''
    print 'IMPROVEMENTS:'
    print '--PET---'
    accImp = np.mean([st['PETGroup']['Accuracy'] for st in statistics]) - np.mean([st['PETGroup']['Accuracy'] for st in statisticsWithoutEl])
    print 'Accuracy: ' + str(accImp)
    sensImp = np.mean([st['PETGroup']['Sensitivity'] for st in statistics]) - np.mean([st['PETGroup']['Sensitivity'] for st in statisticsWithoutEl])
    print 'Sensitivity: ' + str(sensImp)
    specImp = np.mean([st['PETGroup']['Specificity'] for st in statistics]) - np.mean([st['PETGroup']['Specificity'] for st in statisticsWithoutEl])
    print 'Specificity: ' + str(specImp)

    print '--MCI---'
    print 'Accuracy: ' + str(np.mean([st['MCIGroup']['Accuracy'] for st in statistics]) - np.mean(
        [st['MCIGroup']['Accuracy'] for st in statisticsWithoutEl]))
    print 'Sensitivity: ' + str(np.mean([st['MCIGroup']['Sensitivity'] for st in statistics]) - np.mean(
        [st['MCIGroup']['Sensitivity'] for st in statisticsWithoutEl]))
    print 'Specificity: ' + str(np.mean([st['MCIGroup']['Specificity'] for st in statistics]) - np.mean(
        [st['MCIGroup']['Specificity'] for st in statisticsWithoutEl]))
    print '--CI---'
    print 'Accuracy: ' + str(np.mean([st['CIGroup']['Accuracy'] for st in statistics]) - np.mean(
        [st['CIGroup']['Accuracy'] for st in statisticsWithoutEl]))
    print 'Sensitivity: ' + str(np.mean([st['CIGroup']['Sensitivity'] for st in statistics]) - np.mean(
        [st['CIGroup']['Sensitivity'] for st in statisticsWithoutEl]))
    print 'Specificity: ' + str(np.mean([st['CIGroup']['Specificity'] for st in statistics]) - np.mean(
        [st['CIGroup']['Specificity'] for st in statisticsWithoutEl]))

    print ''
    print '---------------WITHOUT ELASTICITY: '
    print 'Elasticity importance: ' + str(np.mean([st['featureImportances']['q1'] for st in statisticsWithoutEl])) + ' ' + str(
        np.mean([st['featureImportances']['q2'] for st in statisticsWithoutEl])) + ' ' + str(
        np.mean([st['featureImportances']['q3'] for st in statisticsWithoutEl]))
    print '--PET---'
    print 'Accuracy: ' + str(np.mean([st['PETGroup']['Accuracy'] for st in statisticsWithoutEl]))
    print 'Sensitivity: ' + str(np.mean([st['PETGroup']['Sensitivity'] for st in statisticsWithoutEl]))
    print 'Specificity: ' + str(np.mean([st['PETGroup']['Specificity'] for st in statisticsWithoutEl]))
    print '--MCI---'
    print 'Accuracy: ' + str(np.mean([st['MCIGroup']['Accuracy'] for st in statisticsWithoutEl]))
    print 'Sensitivity: ' + str(np.mean([st['MCIGroup']['Sensitivity'] for st in statisticsWithoutEl]))
    print 'Specificity: ' + str(np.mean([st['MCIGroup']['Specificity'] for st in statisticsWithoutEl]))

    print '--CI---'
    print 'Accuracy: ' + str(np.mean([st['CIGroup']['Accuracy'] for st in statisticsWithoutEl]))
    print 'Sensitivity: ' + str(np.mean([st['CIGroup']['Sensitivity'] for st in statisticsWithoutEl]))
    print 'Specificity: ' + str(np.mean([st['CIGroup']['Specificity'] for st in statisticsWithoutEl]))

def plotCorrelationMatrix(corr, title):
    plt.title(title)
    #corr = dataframe.corr()
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                annot=True)
    plt.show()
    return