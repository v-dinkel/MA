import cv2
import pdb

from Configuration import Config as cfg
from Tools import utils
from Tools import tifToGif
from Quality import GaussianFilter as GF
from Quality import MedianFilter as MF
from Quality import IlluminationCorrection as IC
from Quality import translateDisplacement as TD

from OpticDiscDetection import DiscDetection_IR

#img_names = utils.getListOfImages(cfg.imgDir, cfg.imgFormats)
#imgs = utils.loadAllImages(cfg.imgDir, cfg.imgFormats)

print 'Starting Pipeline'

from glob import glob
baseDirs = glob("C:\\Users\\DIN035\\Documents\\Project\\ViktorDinkel\\ViktorDinkel\\04_Analysis_HC\\*\\")
import pdb; pdb.set_trace()

'''
print 'Gaussian Filter'
k = 0
for img_name in img_names:
    ret = GF.gaussianFilter(imgs[k])
    cv2.imwrite(cfg.imgDir+img_name, ret)
    k += 1


print 'Median Filter'
imgs = utils.loadAllImages(cfg.imgDir+'\\pipeline\\', cfg.imgFormats)
k = 0
for img_name in img_names:
    ret = MF.medianFilter(imgs[k])
    cv2.imwrite(cfg.imgDir+'\\pipeline\\'+img_name, ret)
    k += 1

print 'Illumination Correction'
imgs = utils.loadAllImages(cfg.imgDir+'\\pipeline\\', cfg.imgFormats)
k = 0
for img_name in img_names:
    ret = IC.illuminationCorrection2(imgs[k],99, 1)
    cv2.imwrite(cfg.imgDir+'\\pipeline\\'+img_name, ret)
    k += 1
'''


'''
print 'Disc Detection'
k = 0
for img_name in img_names:
    ret = DiscDetection_IR.discDetection_IR(imgs[k], False)
    img = imgs[k].copy()
    #cv2.line(img,(0,0),(511,511),(255,0,0),1)
    cv2.circle(img, (ret[0][1],ret[0][0]), ret[1], (255,0,0),4)
    cv2.imwrite(cfg.imgDir+'\\pipeline2\\'+img_name, img)
    k += 1
'''

# ImgShow
# Img_BW
# VesselSkeleton_Pruned

#print 'Translate Displacement'
#TD.run(cfg.imgDir, "raw")


# print 'Tif to Gif'
# tifToGif.run(cfg.imgDir+'\\pipeline2\\translation')

'''
print 'Creating Gif'
utils.createGif(cfg.imgDir+'width\\', img_names)
#utils.createGif(cfg.imgDir+'\\translation\\', utils.getListOfImages(cfg.imgDir+'\\translation\\', cfg.imgFormats))
'''


'''
# display result (press 'q' to quit):
cv2.namedWindow('Transparency')
cv2.imshow('Transparency', img)
while (cv2.waitKey() & 0xff) != ord('q'): pass
cv2.destroyAllWindows()
'''