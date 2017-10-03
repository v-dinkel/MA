from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pandas as pd

from Configuration import Config as cfg
from Tools import utils
from Quality import translateDisplacement as TD
from Quality import GaussianFilter as GF

from Pipeline import loadFiles, resizeImage, preprocessingGreen, opticDiscDetection, vesselSegmentation, \
    vesselWidthMeasurement, splineMapping, createGifs, plotVesselWidth, createFolders, widthAnalysis, DBActions, defineVessels

"""
==========================================================================
This program is trying to realize the vessel analysis for a sequence of Infrared Images!!
"""

plt.figure()

''' STEP 1 CONFIGURATION '''
''' get the configuration for the pipeline, depending on the pipeline step
the steps are: 
    1 = (Not part of this program) Image Stabilization with ImageJ-Plug-In. This also creates the necessary 'pipeline1' folder inside the sequence folder
    2 = Vessel width measurement
    3 = Define Vessel Classes (Artery/Vein)
'''
pipeline_step = 3
pathWithSequences = "C:\\Users\\DIN035\\Documents\\Project\\ViktorDinkel\\ViktorDinkel\\06_test\\"

# ENTER PATH OF FOLDER WHICH CONTAIN SEQUENCE FOLDER
# the structure should look like this:
#   \\analysispath\\sequence1\\pipeline1\\image_1-n.tif
#   \\analysispath\\sequence2\\pipeline1\\image_1-n.tif and so on
# every name can be dynamic but the pipeline1, this name must remain

from glob import glob
baseDirs = glob(pathWithSequences+"*\\")
#baseDirs = ['C:\\Users\\DIN035\\Documents\\Project\\ViktorDinkel\\ViktorDinkel\\04_Analysis_HC\\194-Day0-Seq1\\']

#finished = np.load(pathWithSequences+'\\finished.npy')
#finished = np.append(finished, '1424-Day0-Seq9')

for baseDir in baseDirs:

    #import pdb; pdb.set_trace()

    with open(pathWithSequences+'\\processed.txt', "a") as myProcess:
        myProcess.write(baseDir + "\n")

    '''try:'''
    '''
    baseDir = cfg.imgDir70
    '''
    print '++++++++++STARTING NEXT BASEDIR: ',baseDir
    cfg.baseDir = baseDir
    cfg.imgDir = baseDir+"pipeline1\\"
    cfg.imgDirMeta = baseDir+'metadata.json'
    cfg.gifImgDir = cfg.imgDir

    '''this adds a bracket to the metadata.txt (to make it a valid json) and saves it as jsn-file'''
    utils.fixMetadataJSON(cfg.baseDir)

    '''Read configuration'''
    if (pipeline_step==2):
        PipelineConfig, PipelineSteps, AnalysisSteps, createGifsOf = cfg.getConfigForMeasurement(baseDir)
    elif (pipeline_step==3):
        PipelineConfig, PipelineSteps, AnalysisSteps, createGifsOf = cfg.getConfigForVesselClassification(baseDir)
    else:
        print 'Pipeline Step is not part of this program'
        break
    PipelineConfig['staticDiscRadius'], PipelineConfig['staticDiscCenter'] = utils.getDiscFromFile(cfg.imgDir)
    #####################################################################

    ''' FOLDER PREPROCESSING '''
    """0.1 Create/Remove necessary folders"""
    if (PipelineSteps['deleteAllFolders']):
        createFolders.deleteAllFolders(cfg.imgDir)
    if (PipelineSteps['createDirs']):
        createFolders.createFolders(cfg.imgDir)
    if (PipelineSteps['copyOriginalFiles']):
        createFolders.copyImages(cfg.baseDir, cfg.imgFormats, cfg.imgDir)
    #####################################################################

    """0.2 Translate movement of all images in the folder"""
    '''own implementation of the image stabilization. doesn't work as well as ImageJ and is practically not used anymore'''
    if (PipelineSteps['translateMovement']):
        TD.run(cfg.imgDir, PipelineConfig['translationMethod'])
    #####################################################################

    doPipeline = True in [PipelineSteps[k] for k in PipelineSteps] or PipelineConfig['useStaticDiscParameters']
    ImgFileList, Imgfolder = loadFiles.loadFiles()

    '''The first frame of the sequence is used as the mask for all other frames of a sequence'''
    PipelineConfig['sameImageMask'] = ImgFileList[0].split(".")[0]

    if doPipeline:
        np.save(cfg.imgDir + 'PipelineConfig.npy', PipelineConfig)
        for ImgNumber in range(0,len(ImgFileList)):

            "Read a image first: "
            ImgName = Imgfolder + ImgFileList[ImgNumber]
            JustImgName = ImgFileList[ImgNumber]
            Img0 = cv2.imread(ImgName)
            print 'Img Name:', ImgNumber, ImgFileList[ImgNumber]

            SingleImageFolder = JustImgName.split('.')[0]
            createFolders.createSingleImageFolders(cfg.imgDir, SingleImageFolder)

            """1. Resize the image"""
            if (PipelineSteps['resizeImage']):
                Img, Img_Resized, Img_Resized_old, Mask, Mask_old = resizeImage.resizeImage(Img0, cfg.downsizeRatio)
                resizeImage.saveData(cfg.imgDir, SingleImageFolder, Img, Img_Resized, Img_Resized_old, Mask, Mask_old)
                ##############################################################


            """2: Preprocessing"""
            if (PipelineSteps['preprocessGreen']):
                Mask, Mask_old, Img_Resized, Img_Resized_old = preprocessingGreen.loadData(cfg.imgDir, SingleImageFolder)
                Img_green_filled, IllumGreen_large =  preprocessingGreen.preprocessingGreen(Img_Resized_old, Img_Resized, Mask_old, Mask)
                preprocessingGreen.saveData(cfg.imgDir, SingleImageFolder, Img_green_filled, IllumGreen_large)
                ##############################################################


            """3: Optic Disc Detection"""
            if (PipelineSteps['opticDiscDetection'] or PipelineConfig['useStaticDiscParameters']):
                if (PipelineSteps['opticDiscDetection']):
                    Mask, Img_Resized, Img = opticDiscDetection.loadData(cfg.imgDir, SingleImageFolder)
                    discCenter, discRadius, discRegionParameter, ImgShow = opticDiscDetection.opticDiscDetection(cfg.imgDir, SingleImageFolder, Img, Img_Resized, Mask)
                    opticDiscDetection.saveData(cfg.imgDir, SingleImageFolder, discRegionParameter, ImgShow)
                if (PipelineConfig['useStaticDiscParameters']):
                    Img_Resized = cv2.imread(cfg.imgDir + '\\pipeline_steps\\resizing\\'+SingleImageFolder+'\\Img_Resized.tif')
                    discCenter, discRadius, discRegionParameter, ImgShow = opticDiscDetection.getStaticValues(Img_Resized, PipelineConfig['staticDiscRadius'], PipelineConfig['staticDiscCenter'])
                    opticDiscDetection.saveData(cfg.imgDir, SingleImageFolder, discRegionParameter, ImgShow)
                #else:
                    #Img_Resized = cv2.imread(cfg.imgDir + '\\pipeline_steps\\resizing\\'+SingleImageFolder+'\\Img_Resized.tif')
                    #discCenter, discRadius, discRegionParameter, ImgShow = opticDiscDetection.loadValues(cfg.imgDir, SingleImageFolder, Img_Resized)
                    ###############################################################


            """4: Vessel Segmentation"""
            if (PipelineSteps['vesselSegmentation']):
                Mask_old, Img_green_filled = vesselSegmentation.loadData(cfg.imgDir, SingleImageFolder)
                Img_BW = vesselSegmentation.vesselSegmentation(Img_green_filled, cfg.downsizeRatio, JustImgName, discCenter, discRadius, ImgShow, Mask_old)
                vesselSegmentation.saveData(cfg.imgDir, SingleImageFolder, Img_BW)
                ###############################################################


            """5: Vessel Width Measurement"""
            if (PipelineSteps['vesselWidthMeasurement']):
                if (PipelineConfig['useSameImageMask']):
                    #TD.translateStaticMovement(cfg.imgDir, JustImgName, PipelineConfig['sameImageMask'], ImgNumber)
                    Img, tmp, IllumGreen_large, tmp3 = vesselWidthMeasurement.loadData(cfg.imgDir, SingleImageFolder)
                    tmp4, Img_BW, tmp2, Mask = vesselWidthMeasurement.loadData(cfg.imgDir, PipelineConfig['sameImageMask'])
                else:
                    Img, Img_BW, IllumGreen_large, Mask = vesselWidthMeasurement.loadData(cfg.imgDir, SingleImageFolder)
                dict_splinePoints_updated, dict_meanVesselWidth_ZoneB, dict_meanVesselWidth_ZoneC, dict_segmentPixelLocs_ZoneB, dict_segmentPixelLocs_ZoneC, dict_side1_updated, dict_side2_updated, dict_smoothSide1_updated, dict_smoothSide2_updated = vesselWidthMeasurement.vesselWidthMeasurement(Img_BW, discRegionParameter, Img, IllumGreen_large, Mask)
                vesselWidthMeasurement.saveData(cfg.imgDir, SingleImageFolder, dict_splinePoints_updated, dict_meanVesselWidth_ZoneB, dict_meanVesselWidth_ZoneC, dict_segmentPixelLocs_ZoneB, dict_segmentPixelLocs_ZoneC, dict_side1_updated, dict_side2_updated, dict_smoothSide1_updated, dict_smoothSide2_updated )
                ###############################################################


            """6: Spline Mapping of previous and current image(s)"""
            if (PipelineSteps['splineMapping']):
                ''' this could theoretically be removed as it is deprecated, but its referenced somewhere that's why its still here '''
                dict_splinePoints_updated = np.load(cfg.imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\'+SingleImageFolder+'\\dict_splinePoints_updated.npy')
                dict_meanVesselWidth_ZoneB = np.load(cfg.imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\'+SingleImageFolder+'\\dict_meanVesselWidth_ZoneB.npy')
                dict_meanVesselWidth_ZoneC = np.load(cfg.imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\'+SingleImageFolder+'\\dict_meanVesselWidth_ZoneC.npy')
                splineMap = splineMapping.splineMapping(Img0, JustImgName, prev_dict_splinePoints_updated, dict_splinePoints_updated)
                splineMaps[JustImgName] = splineMap
                vesselWidthsB[JustImgName] = dict_meanVesselWidth_ZoneB
                vesselWidthsC[JustImgName] = dict_meanVesselWidth_ZoneC
                np.save(cfg.imgDir + '\\pipeline_steps\\splineMapping\\'+SingleImageFolder+'\\splineMap.npy', splineMap)
                ###############################################################


            """7: Plot the vessel widths"""
            if (PipelineSteps['plotVesselWidths']):
                Img, Img_BW, dict_meanVesselWidth_ZoneB, dict_meanVesselWidth_ZoneC, dict_segmentPixelLocs_ZoneB, dict_segmentPixelLocs_ZoneC, dict_splinePoints_updated, dict_smoothSide1_updated, dict_smoothSide2_updated, dict_side1_updated, dict_side2_updated = plotVesselWidth.loadData(cfg.imgDir, SingleImageFolder)
                plotVesselWidth.plotVesselWidth(Img, Img_BW, JustImgName, discRegionParameter, discCenter, discRadius, dict_meanVesselWidth_ZoneB, dict_segmentPixelLocs_ZoneB, dict_meanVesselWidth_ZoneC, dict_segmentPixelLocs_ZoneC, dict_splinePoints_updated, dict_smoothSide1_updated, dict_smoothSide2_updated, dict_side1_updated, dict_side2_updated)
                #allVesselLocsB[ImgName] = vesselLocsB
                #allVesselLocsC[ImgName] = vesselLocsC
                ###############################################################

    """ 3. Make gif of the folder(s) """
    if (AnalysisSteps['createGifs']):
        ''' this animates the sequence and saves it in the 'gifs' folder '''
        createGifs.createGifs(cfg.imgDir, createGifsOf)

    """ 4. Analyse Vessel Widths """
    if (AnalysisSteps['analyzeVesselWidth']):
        ''' the actual extraction of the parameters (like elasticity) '''
        widthAnalysis.widthAnalysis(cfg.imgDir, ImgFileList)

    if (AnalysisSteps['exportToDB']):
        ''' the results are exported to the DB here into the results and sequence tables. That is basically and export
        of the classifiedVesselData.npy so you theoretically don't need to do it, if you don't intend to work with a DB '''
        DBActions.exportResults(cfg.imgDir, cfg.baseDir)

    if (AnalysisSteps['customDBActions']):
        '''custom DB actions are just a bunch of different methods for very different purposes, not a part of the actual pipeline'''

        #createFolders.createFolders(cfg.imgDir)
        #DBActions.fixResults(cfg.imgDir, cfg.baseDir, saveToDb = False)
        #DBActions.evaluateSystolicCurves(cfg.imgDir, cfg.baseDir)
        #DBActions.evaluateAll(cfg.baseDir)
        statisticData = DBActions.exportHighcharts(cfg.analysisDir)
        DBActions.initializeIndividuals(cfg.analysisDir, statisticData)
        #DBActions.featureAnalysis(cfg.analysisDir)
        #DBActions.evaluateImprovement(cfg.analysisDir)
        #DBActions.performCorrelationAnalysis(cfg.analysisDir)
        pass

    if (AnalysisSteps['evaluateDBResults']):
        '''some evaluation of the results, this is custom analysis and not part of the pipeline'''
        DBActions.evaluateResults(cfg.analysisDir)

    if (AnalysisSteps['defineVessels']):
        '''sequences, which got their vessels defined with this step, are entered into the file "finished.npy" and are skipped
        if you want to reDefine a sequence, remove it from the loaded finished-dictionary
        if you want to completely start over, just delete the finished.npy from the directory'''
        try:
            finished = np.load(pathWithSequences + '\\finished.npy')
        except:
            print "no finished data loaded"
            finished = np.array([])

        if baseDir.split('\\')[-2] in finished:
            continue

        arteries, veins = defineVessels.selectVesselsManually(cfg.imgDir, PipelineConfig['sameImageMask'])
        DBActions.updateVesselData(cfg.imgDir, arteries, veins)

        print 'done defining vessel of with: '
        print baseDir

        finished = np.append(finished, baseDir.split('\\')[-2])
        np.save(pathWithSequences + 'finished.npy', finished)
        break

    break

    '''except Exception, e:
        print 'ERROR WITH SEQUENCE: ' + baseDir
        print 'e-Message: ' + str(e)
    
        with open(pathWithSequences+'errors.txt', "a") as myfile:
            myfile.write(baseDir+"\n")'''

#print "creating just a gif"
#createGifs.justCreateGif(cfg.imgDir)
#exportToDB.copyResults()