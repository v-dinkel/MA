import cv2
import pdb
import pandas as pd

from Configuration import Config as cfg
from Tools import utils
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import math
import pylab
import scipy.optimize
import collections
import operator
from scipy.cluster.hierarchy import dendrogram, linkage
import itertools
import peakutils
import json


def removeNanVals(curve):

    threshold = 4.5
    if np.nan in curve:
        i = 0
        for val in curve:
            if math.isnan(val):
                mean = sum([k if not math.isnan(k) else 0 for k in curve])/len(curve)
                if i > 0 and i < len(curve)-1:
                    if not math.isnan(curve[i - 1]) and not math.isnan(curve[i + 1]):
                        curve[i] = (curve[i - 1] + curve[i + 1]) / 2
                    elif not math.isnan(curve[i - 1]):
                        if abs(mean - curve[i - 1])>threshold:
                            curve[i] = mean
                        else:
                            curve[i] = curve[i - 1]
                    elif not math.isnan(curve[i + 1]):
                        if abs(mean - curve[i + 1])>threshold:
                            curve[i] = mean
                        else:
                            curve[i] = curve[i + 1]
                    else:
                        print 'remove nan vals exception'
                        curve[i] = mean
                elif i == 0:
                    if not math.isnan(curve[i + 1]):
                        if abs(mean - curve[i + 1])>threshold:
                            curve[i] = mean
                        else:
                            curve[i] = curve[i + 1]
                    else:
                        curve[i] = mean
                else:
                    if not math.isnan(curve[i - 1]):
                        if abs(mean - curve[i - 1])>threshold:
                            curve[i] = mean
                        else:
                            curve[i] = curve[i - 1]
                    else:
                        curve[i] = mean
            i += 1
    return curve

def plotVesselWidth(vessels, idx):
    if type(idx) == type([]):
        for single_idx in idx:
            xaxis = [x for x in range(0, len(vessels[single_idx]))]
            plt.plot(xaxis, vessels[single_idx], label=single_idx, c=np.random.rand(3,))
    else:
        xaxis = [x for x in range(0, len(vessels[idx]))]
        plt.plot(xaxis, vessels[idx], label=idx, c=np.random.rand(3,))

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def fit_sin(tt, yy, mean, avgMin, avgMax):
    '''Src: https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy'''
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    spacing = np.mean([tt[k] - tt[k-1] for k in range(1,len(tt))])
    ff = np.fft.fftfreq(len(tt), spacing)   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    #print 'Freq: '+str(guess_freq)
    guess_freq = clamp(guess_freq, 0.5, 3.0) # clamp the frequency to not be beyond human possible
    #print 'ClmpedFreq: ' + str(guess_freq)
    guess_amp = max(mean-avgMin, mean-avgMax)#np.std(yy) * 2.**0.5
    guess_offset = mean #np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    if guess[1]>7.0: # clamp the frequency AGAIN to not bee beyond human possible
        guess[1] = 7.0
    #import pdb; pdb.set_trace()

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def fitSin(imgDir,vals, xaxis, vessel, zone, avgMin, avgMax, mean, elasticity):
    print 'Fitting Sin to Zone '+zone+' vessel #'+vessel
    N, amp, omega, phase, offset, noise = 500, 1., 2., .5, 4., 3
    tt = np.linspace(0, 10, N)
    yy = amp * np.sin(omega * tt + phase) + offset

    vals = np.array(vals)
    xaxis = np.array(xaxis)
    res = fit_sin(xaxis, vals, mean, avgMin, avgMax)
    #print("Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res)
    plt.figure(figsize=(15, 8))
    plt.gca().set_position((.1, .3, .8, .6))
    plt.plot(xaxis, vals, "-k", label="measured line", linewidth=1)
    plt.plot(xaxis, vals, "ok", label="measured data", linewidth=1)
    highResXaxis = np.arange(xaxis[0],xaxis[-1], .05)
    plt.plot(highResXaxis, res["fitfunc"](highResXaxis), "r-", label="fit sin", linewidth=2)

    rawMin = float("{0:.2f}".format(np.min(vals)))
    rawMax = float("{0:.2f}".format(np.max(vals)))
    rawPercent = float("{0:.2f}".format((((np.max(vals)/np.min(vals))-1)*100)))
    fitMin = float("{0:.2f}".format(np.min([res['offset']-res['amp'], res['offset']+res['amp']])))
    fitMax = float("{0:.2f}".format(np.max([res['offset']-res['amp'], res['offset']+res['amp']])))
    fitPercent = float("{0:.2f}".format(((fitMax/fitMin)-1)*100))
    fitPhase = float("{0:.2f}".format(res['phase']))
    fitFreq = float("{0:.2f}".format(res["omega"]/(2*np.pi)))
    fitOffset = float("{0:.2f}".format(res['offset']))
    fitScore = float("{0:.2f}".format(sum(abs(res["fitfunc"](xaxis) - vals))))

    plt.axhline(y=mean, color='b', linestyle='-', label='offset')

    minAvg = avgMin
    maxAvg = avgMax

    plt.axhline(y=minAvg, color='b', linestyle='-', label='minAvg')
    plt.axhline(y=maxAvg, color='b', linestyle='-', label='maxAvg')

    plt.xlabel("t in seconds")
    plt.ylabel("width in ?")
    plt.legend(loc="best")
    plt.title("Zone "+zone+", Vessel #" + str(vessel))
    plt.grid()

    plt.figtext(.01, .02, "Measured values:\n- Range = "+str(rawMin)+" - "+str(rawMax)+"\n- avgMin/Max = "+str(float("{0:.2f}".format(minAvg)))+" - "+str(float("{0:.2f}".format(maxAvg)))+"\n- Elasticity = "+str(elasticity)+"\n\nFitted value with Score = "+str(fitScore)+"\n- Range = "+str(fitMin)+" - "+str(fitMax)+"\n- Offset = "+str(fitOffset)+"\n- Phase = "+str(fitPhase)+"\n- Freq = "+str(fitFreq)+" cycles/s = "+str(fitFreq*60)+" bpm")
    plt.savefig(imgDir + "\\results\\"+zone+ '\\' + vessel + ".png")
    #plt.show()
    retDict = {
        'zone': zone,
        'vessel': vessel,
        'avgMin': minAvg,
        'avgMax': maxAvg,
        'elasticity': elasticity,
        'bpm': fitFreq*60,
        'phase': res['phase'],
        'offset': res['offset'],
        'score': fitScore
    }
    return retDict

def plotAllVessels(M, t, keys, zone):
    plt.figure()
    k = 0
    for vals in M:
        plt.plot(t, vals, label=keys[k], c=np.random.rand(3,))
        k += 1
    plt.title('Width of all vessels in Zone '+zone)
    plt.legend()
    plt.xlabel('time in s')
    plt.ylabel('vessel width in ?')
    plt.savefig(cfg.imgDir + "\\results\\"+zone+"\\allVesselWidths.png")
    plt.clf()

def widthAnalysis(imgDir, ImageFileList):

    """ Read time of the Frames """
    print "reading metadata: ", cfg.imgDirMeta
    with open(cfg.imgDirMeta) as data_file:
        data = json.load(data_file)

    """ Read the time for each frame """
    time = []
    ImageNumbers = [k.split('_')[1] for k in ImageFileList]

    for i in ImageNumbers:
        keyStr = 'FrameKey-'+str(int(i))+'-0-0'
        time.append(float(data[keyStr]['ElapsedTime-ms'])/1000.0)

    for i in range(1,len(time)):
        if time[i] == time[i-1]:
            time[i] = time[i]+.001

    performZoneAnalysis(imgDir, ImageFileList, time)
    return

def performZoneAnalysis(imgDir, ImageFileList, time):
    zones = ['B', 'C']
    for zone in zones:
        print '====== Width Analysis for Zone ' + zone + ' ======'
        """ Read and collect the measured widths"""
        measurements = {}
        k = 0
        for image in ImageFileList:
            imageFolder = image.split('.')[0]
            dict_meanVesselWidth_ZoneX = np.load(imgDir + '\\pipeline_steps\\vesselWidthMeasurement\\' + imageFolder + '\\dict_meanVesselWidth_Zone'+zone+'.npy').item()
            measurements[time[k]] = dict_meanVesselWidth_ZoneX
            k+=1
        print 'saving measurements in zone ', zone
        np.save(imgDir + '\\results\\'+zone+'\\measurements.npy', measurements)

        # loading not necessary but to preserve continuity
        measurements = np.load(imgDir + '\\results\\' + zone + '\\measurements.npy').item()
        time = sorted(measurements.keys())

        uniq = list(itertools.chain.from_iterable([measurements[k].keys() for k in time]))
        vessels = [k for k in set(uniq)]

        """ Create a Matrix with row = single vessel widths, col = time """
        M = []
        for key in vessels:
            vals = []
            for t in time:
                try:
                    vals.append(measurements[t][key])
                except:
                    vals.append(np.nan)
            M.append(vals)
        plotAllVessels(M, time, vessels, zone)

        """ fit the widths to a curve and get statistics about elasticity, frequency etc. """
        errorVals = []
        matrixOrder = ['zone', 'vessel', 'elasticity', 'bpm', 'phase', 'offset', 'avgMin', 'avgMax', 'score'] # later added: outlierClass[0=ok,1=outlier]; vesselClassification[0=none,1=artery,2=vein]
        outlierClassificationParameters = ['elasticity', 'bpm', 'phase']
        MM = []
        MMa = []

        for vessel in vessels:
            vals = [measurements[k][vessel] if  vessel in measurements[k].keys() else np.nan for k in time]

            nanIndeces = [i for i, ltr in enumerate(vals) if np.isnan(ltr)]
            timeWithoutNan = [i for j, i in enumerate(time) if j not in nanIndeces]
            valsWithoutNan = [i for j, i in enumerate(vals) if j not in nanIndeces]

            mean = np.mean(valsWithoutNan)
            std = np.std(valsWithoutNan)
            elasticity = (((mean+std)/(mean-std))-1.0)*100.0

            indexes_max, mean_peak_max, std_max = detect_peaks(timeWithoutNan, np.array(valsWithoutNan), .04, .2)
            indexes_min, mean_peak_min, std_min = detect_peaks(timeWithoutNan, np.array(valsWithoutNan)*-1, .04, .2)

            plt.clf()
            plt.plot(timeWithoutNan, valsWithoutNan)
            for i in indexes_max:
                plt.plot(timeWithoutNan[i], valsWithoutNan[i], 'o', color='r')
            for i in indexes_min:
                plt.plot(timeWithoutNan[i], valsWithoutNan[i], 'o', color='y')
            plt.axhline(y=mean_peak_max + std_max, color='b', linestyle='-')
            plt.axhline(y=(mean_peak_min * -1) - std_min, color='g', linestyle='-')
            #plt.savefig(imgDir + "\\results\\" + zone + '\\' + vessel + ".png")
            plt.clf()
            # plt.plot(timeWithoutNan, valsWithoutNan)

            try:
                #(imgDir,vals, xaxis, vessel, zone, avgMin, avgMax, mean, elasticity)
                retDict = fitSin(imgDir, valsWithoutNan, timeWithoutNan, vessel, zone, (mean_peak_min*-1)-std_min, mean_peak_max+std_max, mean, elasticity)
                if True in [pd.isnull(retDict[k]) for k in retDict.keys()]:
                    print 'NaN in vessel '+vessel+', excluding from results'
                    errorVals.append(vessel)
                else:
                    MM.append([retDict[k] for k in matrixOrder])
                    MMa.append([retDict[k] for k in outlierClassificationParameters])
            except:
                #import pdb; pdb.set_trace()
                print 'Error fitting curve to vessel '+vessel
                errorVals.append(vessel)

        print 'Vessels with errors: ', errorVals
        plt.close("all")

        #MM, indicesOfOutliers = findOutliersByClustering(MM, MMa, zone)
        #print 'outlierByClustering: ',[MM[int(k)][1] for k in indicesOfOutliers]
        #plt.close("all")

        # initialize outlier value
        for k in range(0,len(MM)):
            MM[int(k)].append(0.0)  # 0 stands for outlierClass=0; so it ISN'T an outlier

        MM = findOutliersByThresholds(MM)

        vesselClassificationParameters = ['phase']
        MMa = []
        keys = []
        for k in range(0, len(MM)):
            if MM[k][len(MM[k])-1] == 0.0:
                MMa.append(MM[k][4])
                keys.append(MM[k][1])

        if len(MMa)>0:
            arteries, veins = classifyVessels(MMa, keys)
        else:
            arteries = []; veins = []

        for i in range(0,len(MM)):
            if MM[i][1] in arteries:
                MM[i].append(1)
            elif MM[i][1] in veins:
                MM[i].append(2)
            else:
                MM[i].append(0)
        np.save(imgDir + '\\results\\'+zone+'\\classifiedVesselData.npy', MM)
    return

def performZoneAnalysis2(imgDir):
    zones = ['B', 'C']
    for zone in zones:
        print '====== Width Analysis for Zone ' + zone + ' ======'
        # loading not necessary but to preserve continuity
        measurements = np.load(imgDir + '\\results\\' + zone + '\\measurements.npy').item()
        time = sorted(measurements.keys())

        uniq = list(itertools.chain.from_iterable([measurements[k].keys() for k in time]))
        vessels = [k for k in set(uniq)]

        """ Create a Matrix with row = single vessel widths, col = time """
        M = []
        for key in vessels:
            vals = []
            for t in time:
                try:
                    vals.append(measurements[t][key])
                except:
                    vals.append(np.nan)
            M.append(vals)
        plotAllVessels(M, time, vessels, zone)

        """ fit the widths to a curve and get statistics about elasticity, frequency etc. """
        errorVals = []
        matrixOrder = ['zone', 'vessel', 'elasticity', 'bpm', 'phase', 'offset', 'avgMin', 'avgMax', 'score'] # later added: outlierClass[0=ok,1=outlier]; vesselClassification[0=none,1=artery,2=vein]
        outlierClassificationParameters = ['elasticity', 'bpm', 'phase']
        MM = []
        MMa = []

        for vessel in vessels:
            vals = [measurements[k][vessel] if  vessel in measurements[k].keys() else np.nan for k in time]

            nanIndeces = [i for i, ltr in enumerate(vals) if np.isnan(ltr)]
            timeWithoutNan = [i for j, i in enumerate(time) if j not in nanIndeces]
            valsWithoutNan = [i for j, i in enumerate(vals) if j not in nanIndeces]

            mean = np.mean(valsWithoutNan)
            std = np.std(valsWithoutNan)
            elasticity = (((mean+std)/(mean-std))-1.0)*100.0

            indexes_max, mean_peak_max, std_max = detect_peaks(timeWithoutNan, np.array(valsWithoutNan), .04, .2)
            indexes_min, mean_peak_min, std_min = detect_peaks(timeWithoutNan, np.array(valsWithoutNan)*-1, .04, .2)

            plt.clf()
            plt.plot(timeWithoutNan, valsWithoutNan)
            for i in indexes_max:
                plt.plot(timeWithoutNan[i], valsWithoutNan[i], 'o', color='r')
            for i in indexes_min:
                plt.plot(timeWithoutNan[i], valsWithoutNan[i], 'o', color='y')
            plt.axhline(y=mean_peak_max + std_max, color='b', linestyle='-')
            plt.axhline(y=(mean_peak_min * -1) - std_min, color='b', linestyle='-')

            plt.axhline(y=mean+std, color='r', linestyle='-')
            plt.axhline(y=mean-std, color='r', linestyle='-')

            plt.axhline(y=mean, color='g', linestyle='-')

            #plt.savefig(imgDir + "\\results\\" + zone + '\\' + vessel + ".png")
            #plt.clf()
            plt.show()
            # plt.plot(timeWithoutNan, valsWithoutNan)


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def findOutliersByThresholds(MM):
    heartrateThresholds = (50.0,130.0)
    elasticityMax = 20.0
    elasticityArr = reject_outliers(np.array([k[2] for k in MM if k[-1]==0.0]),1)
    scoreArr = reject_outliers(np.array([k[-2] for k in MM if k[-1]==0.0]),1)
    scoreMax = 20.0

    # Check if the values, which were not classified as outlier yet, are within thresholds
    # if they are not in a certein threshold, a 0.25 penalty is added to a maximum of 0.75
    for i in range(0,len(MM)):
        if MM[i][-1] == 0.0: #so it is not an outlier
            # elasticity = MM[i][2]
            if MM[i][2] not in elasticityArr or MM[i][2]>elasticityMax:
                MM[i][-1] = MM[i][-1] + 0.4
            # heartrate = MM[i][3]
            if not heartrateThresholds[0] < MM[i][3] < heartrateThresholds[1]:
                MM[i][-1] = MM[i][-1] + 0.1
            # score = MM[i][-3]
            if MM[i][-2] not in scoreArr or MM[i][-2]>scoreMax:
                MM[i][-1] = MM[i][-1] + 0.2
    return MM

def classifyVessels(MMa, keys):
    '''
    map = {}
    for key in range(0,len(keys)):
        map[keys[key]] = MMa[key]
    sortedMap = sorted(map.items(), key=operator.itemgetter(1))
    diff = [abs(sortedMap[k-1][1]-sortedMap[k][1]) for k in range(1, len(sortedMap))]
    try:
        maxDiff = diff.index(max(diff))
    except:
        import pdb; pdb.set_trace()
    arteries = [k[0] for k in sortedMap[:maxDiff+1]]
    veins = [k[0] for k in sortedMap[maxDiff+1:]]
    '''
    return [], []

def findOutliersByClustering(MM, MMa, zone):
    goodVessels = []
    badVessels = []
    Z = linkage(MMa, 'ward')
    allClusters = dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
    onlyTwoClusters = dendrogram(Z, leaf_rotation=90., leaf_font_size=8., truncate_mode='lastp', p=2)
    plt.clf()
    plt.close("all")

    if len(onlyTwoClusters)>1:
        lenOfFirst = int(onlyTwoClusters['ivl'][0].replace(")",'').replace("(",''))
        lenOfSecond = int(onlyTwoClusters['ivl'][1].replace(")", '').replace("(", ''))
        if lenOfFirst<lenOfSecond:
            badVessels = allClusters['ivl'][0:lenOfFirst]
            goodVessels = allClusters['ivl'][lenOfFirst:]
        elif lenOfFirst>lenOfSecond:
            badVessels = allClusters['ivl'][0:lenOfSecond]
            goodVessels = allClusters['ivl'][lenOfSecond:]
        else:
            goodVessels = allClusters['ivl'][:]

    for k in goodVessels:
        MM[int(k)].append(0.0) # 0 stands for outlierClass=0; so it ISN'T an outlier
    for k in badVessels:
        MM[int(k)].append(1.0) # 1 stands for outlierClass=1; so it IS an outlier

    ''' Just for plotting '''
    dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
    outStr = ''
    for k in badVessels:
        outStr = outStr+' '+MM[int(k)][1]
    plt.title('Hierarchical Clustering by elasticity, bpm and phase')
    plt.figtext(.01, .02, 'Outlier Vessels:'+outStr)
    plt.savefig(cfg.imgDir + "\\results\\" + zone + "\\outliers_dendrogram.png")
    plt.clf()
    plt.close("all")

    return MM, badVessels

def detect_peaks(time, vals, thresh = 0.02, minDist = 100):
    indexes = peakutils.indexes(np.array(vals), thres=thresh / max(vals), min_dist=minDist)
    if (len([vals[k] for k in indexes])>2): #reject outliers if there are enough peaks to select from
        mean_peak = np.mean(reject_outliers(np.array([vals[k] for k in indexes]),1.2))
        std = np.std(reject_outliers(np.array([vals[k] for k in indexes]), 1.2))
    else:
        mean_peak = np.mean([vals[k] for k in indexes])
        std = np.std([vals[k] for k in indexes])

    return indexes, mean_peak, std

