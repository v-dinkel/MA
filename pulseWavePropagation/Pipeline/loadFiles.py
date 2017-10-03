from Configuration import Config as cfg
from Tools import utils
from Tools.SortFolder import natsort

def loadFiles():
    Imgfolder = cfg.imgDir
    ImgFileList =  utils.getListOfImages(cfg.imgDir, cfg.imgFormats)
    natsort(ImgFileList)
    if ImgFileList.__contains__('Thumbs.db'):
        ImgFileList.remove('Thumbs.db')
    return ImgFileList, Imgfolder