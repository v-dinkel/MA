from Configuration import Config as cfg
from Tools import utils
import os

def createGifs(imgDir, createGifsOf):
    #Single usage: utils.createGif(cfg.imgDir+'\\ImgShow\\', img_names )
    print 'Creating Gifs'
    gifPath = imgDir + 'gifs\\'

    if createGifsOf['original']:
        originalPath = imgDir
        if os.path.exists(originalPath):
            img_names = utils.getListOfImages(originalPath, cfg.imgFormats)
            utils.createGif(originalPath, img_names, 'original', gifPath)

    if createGifsOf['translation']:
        translationPath = imgDir + 'translation'
        if os.path.exists(translationPath):
            img_names = utils.getListOfImages(translationPath, cfg.imgFormats)
            utils.createGif(translationPath, img_names, 'translation', gifPath)

    if createGifsOf['width']:
        widthPath = imgDir + 'width'
        if os.path.exists(widthPath):
            img_names = utils.getListOfImages(widthPath, cfg.imgFormats)
            utils.createGif(widthPath, img_names, 'width', gifPath)

    if createGifsOf['width_hq']:
        width_hqPath = imgDir + 'width_hq'
        if os.path.exists(width_hqPath):
            img_names = utils.getListOfImages(width_hqPath, cfg.imgFormats)
            utils.createGif(width_hqPath , img_names, 'width_hq', gifPath)

    if createGifsOf['test']:
        originalPath = imgDir + 'test'
        if os.path.exists(originalPath):
            img_names = utils.getListOfImages(originalPath, cfg.imgFormats)
            utils.createGif(originalPath, img_names, 'test', gifPath)

    # utils.createGif(cfg.imgDir+'\\ImgShow\\', img_names )
    # utils.createGif(cfg.imgDir+'\\Img_BW\\', img_names )
    # utils.createGif(cfg.imgDir+'\\VesselSkeleton_Pruned\\', img_names )
    # utils.createGif(cfg.imgDir+'\\BW_resized\\', img_names )
def justCreateGif(imgDir):
    if os.path.exists(imgDir):
        img_names = utils.getListOfImages(imgDir, cfg.imgFormats)
        print 'got image names'
        utils.createGif(imgDir, img_names, 'thisGif', imgDir)