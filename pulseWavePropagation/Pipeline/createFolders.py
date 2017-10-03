import os
from Tools import utils
from shutil import copyfile, rmtree

from Configuration import Config as cfg

def createFolders(imgDir):
    print 'Creating directories in ', imgDir

    if not os.path.exists(imgDir):
        os.makedirs(imgDir)

    directories = ['results', 'test', 'width', 'width_hq', 'translation', 'pipeline_steps', 'gifs', 'results\\A', 'results\\B', 'results\\C']
    for directory in directories:
        if not os.path.exists(imgDir+directory):
            os.makedirs(imgDir+directory)

    directories = ['test\\B', 'test\\C']
    for directory in directories:
        if not os.path.exists(imgDir+directory):
            os.makedirs(imgDir+directory)

    ps = 'pipeline_steps\\'
    subDirectories = [ps+'opticDiscDetection', ps+'preprocessGreen', ps+'resizing', ps+'splineMapping', ps+'vesselSegmentation', ps+'vesselWidthMeasurement']
    for directory in subDirectories:
        if not os.path.exists(imgDir+directory):
            os.makedirs(imgDir+directory)
    print '##############################################################'

def copyImages(baseDir, formats, targetDir):
    print 'Copying images from ', baseDir, ' to ', targetDir
    img_names = utils.getListOfImages(baseDir, formats)
    for img in img_names:
        copyfile(baseDir+img, targetDir+img)
    print '##############################################################'

def createSingleImageFolders(imgDir, singleImageName):
    subDirectories = os.walk(imgDir+'pipeline_steps\\').next()[1]
    for directory in subDirectories:
        if not os.path.exists(imgDir+'pipeline_steps\\'+directory+'\\'+singleImageName):
            os.makedirs(imgDir+'pipeline_steps\\'+directory+'\\'+singleImageName)


def deleteAllFolders(imgDir):
    print 'Deleting directories in ', imgDir
    directories = ['results', 'test', 'width', 'width_hq', 'translation', 'pipeline_steps']

    for directory in directories:
        if os.path.exists(imgDir + directory):
            rmtree(imgDir + directory)

def deleteFilesFromFolders(imgDir):
    print 'Deleting files in ', imgDir

    from glob import glob
    contents = glob(imgDir + "*\\")

    #delete everything but pipeline 1
    for content in contents:
        print content
        allRootContent = glob(content + "*.*")
        for rootContent in allRootContent:
            os.remove(rootContent)

        # deleteing everything in pipeline1 except results and gifs
        allPipelineContent = glob(content+'pipeline1\\' + "*.*")
        for pipelineContent in allPipelineContent:
            os.remove(pipelineContent)

        if os.path.exists(content+'pipeline1\\' + 'pipeline_steps\\'):
            rmtree(content+'pipeline1\\' + 'pipeline_steps\\')
        if os.path.exists(content+'pipeline1\\' + 'test\\'):
            rmtree(content+'pipeline1\\' + 'test\\')
        if os.path.exists(content+'pipeline1\\' + 'translation\\'):
            rmtree(content+'pipeline1\\' + 'translation\\')
        if os.path.exists(content+'pipeline1\\' + 'width\\'):
            rmtree(content+'pipeline1\\' + 'width\\')
        if os.path.exists(content+'pipeline1\\' + 'width_hq\\'):
            rmtree(content+'pipeline1\\' + 'width_hq\\')

        try:
            os.remove(content+'pipeline1\\gifs\\original.gif')
            os.remove(content + 'pipeline1\\gifs\\width_hq.gif')
        except:
            pass

    print 'done deleting file contents'


