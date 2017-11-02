import pdb
import numpy as np
import MySQLdb
from operator import itemgetter
from matplotlib import pyplot as plt
from matplotlib.finance import candlestick2_ohlc
import itertools
import scipy.optimize
import peakutils
from Tools import utils
import seaborn as sns

def exportResults(imgDir, baseDir):
    ''' 'zone','vessel','elasticity','bpm','phase','offset','avgMin','avgMax','score','outlinerScore', vesselClass '''
    '''    0      1          2         3      4       5        6         7       8         9               10      '''

    zones = ['B','C']
    results = []
    for zone in zones:
        try:
            tmp = (np.load(imgDir + '\\results\\'+zone+'\\classifiedVesselData.npy'))
            [results.append(k) for k in tmp]
        except:
            print "no results found for zone "+zone
            #import pdb; pdb.set_trace()
    exportToDB(results, baseDir, imgDir)

def exportToDB(results, baseDir, imgDir):
    print '-----Writing results to database'
    db = MySQLdb.connect(host="localhost",  # your host, usually localhost
                         user="root",  # your username
                         passwd="",  # your password
                         db="vesselanalysis")  # name of the data base

    cur = db.cursor()

    id = baseDir.split('\\')[-2]
    PipelineConfig = np.load(imgDir+'PipelineConfig.npy').item()

    try:
        sql = 'INSERT INTO sequence (id, discRadius, discCenter, translationMethod) VALUES ("'+id+'", '+str(PipelineConfig['staticDiscRadius'])+', "'+str(PipelineConfig['staticDiscCenter'])+'", "'+PipelineConfig['translationMethod']+'");'
        cur.execute(sql)
        db.commit()
    except:
        print 'Already exists in database: '+id
        import pdb; pdb.set_trace
        pass

    for result in results:
        try:
            zone, vessel, elasticity, bpm, phase, offset, avgMin, avgMax, fittingScore, outlierScore, vesselClass  = result
            sql = "INSERT INTO results (sequence_id, zone, vessel, elasticity, bpm, phase, offset, avgMin, avgMax, fittingScore, outlierScore, vesselClass, verified) VALUES"\
            "('"+id+"', '"+zone+"', '"+vessel+"', "+elasticity+", "+bpm+", "+phase+", "+offset+", "+avgMin+", "+avgMax+", "+fittingScore+", "+outlierScore+", "+vesselClass+", 'None')"
            cur.execute(sql)
        except:
            print 'error in exportDB'
            import pdb; pdb.set_trace()
    db.commit()
    db.close()

def getLinearRegression(curve):
    x = [k[0] for k in curve]
    y = [k[1] for k in curve]

    fit = np.polyfit(x, y, 1)
    fix_fn = np.poly1d(fit)

    X = [fix_fn(xx) for xx in x]
    return X, fix_fn

def evaluateResults(analysisDir):
    print '-----Evaluating results from the database'
    db = MySQLdb.connect(host="localhost",  # your host, usually localhost
                         user="root",  # your username
                         passwd="",  # your password
                         db="vesselanalysis")  # name of the data base

    cur = db.cursor()
    sql = "SELECT sequence_id, zone, elasticity, offset FROM vesselanalysis.results WHERE verified = 'True' and vesselClass='1' and offset>=10.0"
    try:
        cur.execute(sql)
        results = cur.fetchall()
        resultsDict = {"B":{},"C":{}}
        for row in results:

            try:
                resultsDict[row[1]][row[0]][row[3]] = row[2]
            except:
                resultsDict[row[1]][row[0]] = {row[3]: row[2]}

    except:
        print "Error: unable to fecth data"

    # disconnect from server
    db.close()

    #badSequences = ['229-Day0-Seq1', '237-Day0-Seq4-IR1', '262-Day0-Seq1-IR1', '220-Day0-Seq7-IR1']
    excludeThese = ['229-Day0-Seq1', '237-Day0-Seq4-IR1', '262-Day0-Seq1-IR1', '220-Day0-Seq7-IR1']
    #"1273-Day0-Seq1", "1273-Day0-Seq3", "1650-Day0-Seq1-IR1", "1641-Day0-Seq4-IR1", "1650-Day0-Seq4-IR1"  ,"244-Day0-Seq1","191-Day0-Seq1-IR1", "200-Day0-Seq1", "191L-Day0-Seq3-IR1", '1610-Day0-Seq9',

    zones = ['B','C']
    if (True):
        for zone in zones:
            for sequence in resultsDict[zone].keys():
                if sequence not in excludeThese:
                    values = sorted(resultsDict[zone][sequence].items(), key=itemgetter(0))

                    plt.title('Zone '+zone+', '+sequence)
                    plt.plot([k[0] for k in values], [k[1] for k in values], "-k", label=sequence, color='g')
                    plt.plot([k[0] for k in values], [k[1] for k in values], "ok", label=sequence, color='g')
                    linearRegressionValues, regressionFunction = getLinearRegression(values)
                    plt.plot([k[0] for k in values], linearRegressionValues, "-k", label=sequence, color='r')
                    plt.xlabel("Vessel width")
                    plt.ylabel("elasticity in %")
                    plt.grid()
                    plt.legend()

                    plt.savefig(analysisDir+'initialPlots\\' +zone+"_"+sequence+'.png')
                    np.save(analysisDir +zone+"_"+sequence+'.npy', regressionFunction )
                    np.save(analysisDir + '\\values\\' + zone + "_" + sequence + '.npy', values)
                    plt.clf()


    onlyThese = ["1273-Day0-Seq1", "1273-Day0-Seq3", "1650-Day0-Seq1-IR1", "1641-Day0-Seq4-IR1", "1650-Day0-Seq4-IR1"  ,"244-Day0-Seq1","191-Day0-Seq1-IR1", "200-Day0-Seq1", "191L-Day0-Seq3-IR1", '1610-Day0-Seq9']
    meanBC = {}
    plt.figure(figsize=(15, 8))
    plt.title('All elasticity plots')
    xaxis = range(0,22)
    for sequence in resultsDict['B'].keys():
        if sequence not in excludeThese: # and sequence in onlyThese
            thisCol = np.random.rand(3,)
            noB = False
            noC = False
            try:
                bFn = np.load(analysisDir + "B_" + sequence + '.npy')
                bVals = [x*bFn[0]+bFn[1] for x in xaxis]
                plt.plot(xaxis, bVals, linestyle ="-", label=sequence+' B', color=thisCol)
            except:
                print sequence+" no values in zone B"
                noB = True
            try:
                cFn = np.load(analysisDir + "C_" + sequence + '.npy')
                cVals = [x*cFn[0]+cFn[1] for x in xaxis]
                plt.plot(xaxis, cVals, linestyle="--", label=sequence + ' C', color=thisCol)
            except:
                print sequence+" no values in zone C"
                noC = True
            if not noB and not noC:
                meanBC[sequence] = (np.array(bVals) + np.array(cVals))/2
            elif not noB:
                meanBC[sequence] = np.array(bVals)
            elif not noC:
                meanBC[sequence] = np.array(cVals)
            else:
                print sequence+' no values in zones B and C, what to do?'
                import pdb; pdb.set_trace()
    plt.axvline(x=10, color='r')
    plt.xlabel("Vessel width")
    plt.ylabel("elasticity in %")
    plt.grid()
    plt.legend()
    plt.savefig(analysisDir + 'allWidths.png')
    plt.clf()

    colors = []
    interval = 51.0
    for x in range(0,len(meanBC.keys())):
        #pdb.set_trace()
        if x < 4:
            rr = (float(x+1)*interval)%255/255.0
        else:
            rr = .1
        if 4 <= x < 8:
            gg = (float(x+1)*interval)%255/255.0
        else:
            gg = .2
        if 8 <= x < 12:
            bb = (float(x+1)*interval)%255/255.0
        else:
            bb = .3
        colors.append(np.array([rr,gg,bb]))


    plt.title('Elasticity of zones B,C averaged')
    k = 0
    for sequence in meanBC.keys():
        thisCol = np.random.rand(3, )
        plt.plot(xaxis,meanBC[sequence], linestyle="-", label=sequence, color=thisCol)
        k += 1
    plt.xlim(11, 22)
    plt.ylim(1.5, 7.5)
    plt.axvline(x=10, color='r')
    plt.xlabel("Vessel width")
    plt.ylabel("elasticity in %")
    plt.grid()
    plt.legend()
    plt.savefig(analysisDir + 'allMeanWidths.png')
    plt.clf()

    # HEARTRATE mapping
    #fig, ax = plt.subplots()
    #fig.subplots_adjust(bottom=0.2)

    bloodPressure = range(80, 200, 10)
    sequenceBloodPressure = utils.getSequenceBloodPressure()
    plt.clf()
    import math

    #import pdb; pdb.set_trace()
    xaxis = np.array([math.log(float(k) / float(120)) for k in bloodPressure])
    fig, ax = plt.subplots()
    ax.plot([-1.0, 1.0], [fitnessFunc(x) for x in [-1.0,1.0]], c='r', label='expected elasticity',alpha=0.3, lw=2.0)
    ax.plot([-1.0, 1.0], [fitnessFunc(x) for x in [-1.0, 1.0]], c='r', alpha=0.1, lw=40.0)
    #ax.set_aspect('equal')
    ax.grid(True, which='both')

    # set the x-spine (see below for more info on `set_position`)
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    plt.xlim(-1*.25,.5)
    plt.ylim(0, 10)

    #plt.plot(bloodPressure, math.log(np.array(bloodPressure)) / 3.0, c=(1.0, 69 / 255, 0), label='3.0% elasticity',alpha=0.3, lw=5.0)
    #plt.plot(bloodPressure, math.log(np.array(bloodPressure)) / 4.0, c='y', label='4.0% elasticity',alpha=0.3, lw=5.0)
    #plt.plot(bloodPressure, math.log(np.array(bloodPressure)) / 5.0, c='g', label='5.0% elasticity',alpha=0.3, lw=5.0)

    sequenceColors = {}
    for sequence in meanBC.keys():
        noB = False
        noC = False
        try:
            valuesB = np.load(analysisDir+'\\values\\B_' + sequence + '.npy')
        except:
            print 'no values B to load'
            noB = True
        try:
            valuesC = np.load(analysisDir+'\\values\\C_' + sequence + '.npy')
        except:
            print 'no values C to load'
            noC = True
        if not noB and not noC:
            allVals = sorted([k[1] for k in np.concatenate((valuesB, valuesC), axis=0)])
        elif not noB:
            allVals = sorted([k[1] for k in valuesB])
        elif not noC:
            allVals = sorted([k[1] for k in valuesC])
        else:
            import pdb; pdb.set_trace()

        #import pdb; pdb.set_trace()
        if sequence.split('-')[0] in sequenceColors.keys():
            thisCol = sequenceColors[sequence.split('-')[0]]
        else:
            thisCol = np.random.rand(3, )
            sequenceColors[sequence.split('-')[0]] = thisCol
        thisBloodPressure = float(sequenceBloodPressure[sequence.split('-')[0]])
        ax.plot([math.log(thisBloodPressure/120.0),math.log(thisBloodPressure/120.0)], [allVals[0], allVals[-1]], c=thisCol, lw=2.0, label=sequence.split('-')[0])
        ax.plot([math.log(thisBloodPressure/120.0)-.001, math.log(thisBloodPressure/120.0)+.001],[np.mean(allVals), np.mean(allVals)], c=thisCol, lw=4.0)

    plt.xlabel('Blood pressure (BP) in log(BP/120)')
    plt.ylabel('Elasticity in %')
    plt.legend()
    plt.show()

def fitnessFunc(x):
    return 5*x+4

def copyResults():
    print '-----Evaluating results from the database'
    db = MySQLdb.connect(host="localhost",  # your host, usually localhost
                         user="root",  # your username
                         passwd="",  # your password
                         db="vesselanalysis")  # name of the data base

    cur = db.cursor()
    sql = "SELECT * FROM vesselanalysis.results WHERE sequence_id ='1650-Day0-Seq4-IR1' and verified = 'True'"
    try:
        cur.execute(sql)
        results = cur.fetchall()

        sequence_id = '1650-Day0-Seq4-IR1_fixed'
        for row in results:
            #pdb.set_trace()
            sql = "INSERT INTO results (sequence_id, zone, vessel, elasticity, bpm, phase, offset, avgMin, avgMax, fittingScore, outlierScore, vesselClass, verified) VALUES" \
                  "('" + sequence_id + "', '" + row[1] + "', '" + row[2] + "', " + str(row[3]) + ", " + str(row[4]) + ", " + str(row[5]) + ", " + str(row[6]) + ", " + str(row[7]) + ", " + str(row[8]) + ", " + str(row[9]) + ", " + str(row[10]) + ", " + str(row[11]) + ", '"+str(row[12])+"')"
            #pdb.set_trace()
            cur.execute(sql)
        db.commit()

    except:
        print "Error: unable to fetch data"

    # disconnect from server
    db.close()

def fixResults(imgDir, baseDir, saveToDb = False):

    print '-----Fixing the DB results'
    if saveToDb:
        db = MySQLdb.connect(host="localhost",  # your host, usually localhost
                             user="root",  # your username
                             passwd="",  # your password
                             db="vesselanalysis")  # name of the data base
        cur = db.cursor()

    allSystolicCurves = []

    sequence_id = baseDir.split('\\')[-2]
    zones = ['B','C']
    for zone in zones:
        measurements = np.load(imgDir+'\\results\\'+zone+'\\measurements.npy').item()
        time = sorted(measurements.keys())

        uniq = list(itertools.chain.from_iterable([measurements[k].keys() for k in time]))
        vessels = [k for k in set(uniq)]

        for vessel in vessels:
            vals = [measurements[k][vessel] if  vessel in measurements[k].keys() else np.nan for k in time]

            nanIndeces = [i for i, ltr in enumerate(vals) if np.isnan(ltr)]
            timeWithoutNan = [i for j, i in enumerate(time) if j not in nanIndeces]
            valsWithoutNan = [i for j, i in enumerate(vals) if j not in nanIndeces]

            mean = np.mean(reject_outliers(np.array(valsWithoutNan), 1))
            std = np.std(reject_outliers(np.array(valsWithoutNan), 1))
            elasticity = (((mean+std)/(mean-std))-1.0)*100.0
            #reject_outliers(np.array(valsWithoutNan), 1)

            indexes_max, mean_peak_max, std_max = detect_peaks(timeWithoutNan, np.array(valsWithoutNan), .04, .2)
            indexes_min, mean_peak_min, std_min = detect_peaks(timeWithoutNan, np.array(valsWithoutNan)*-1, .04, .2)

            plt.clf()
            plt.plot(timeWithoutNan, valsWithoutNan)
            for i in indexes_max:
                plt.plot(timeWithoutNan[i], valsWithoutNan[i], 'o', color='r')
            for i in indexes_min:
                plt.plot(timeWithoutNan[i], valsWithoutNan[i], 'o', color='y')
            plt.axhline(y=mean_peak_max+std_max, color='b', linestyle='-')
            plt.axhline(y=(mean_peak_min*-1)-std_min, color='g', linestyle='-')

            allSystolicCurves.append(findSystolicCurve(indexes_min, indexes_max, valsWithoutNan, timeWithoutNan, zone, vessel))

            plt.savefig(imgDir + "\\test\\" + zone + '\\' + vessel + ".png")
            # plt.show()
            plt.clf()

            #minVals = np.array(valsWithoutNan)*-1
            #reject_outliers(np.array([minVals[k] for k in indexes_min]),1)

            #reject_outliers(np.array([k[2] for k in MM if k[-1] == 0.0]), 1)
            #imgDir,vals, xaxis, vessel, zone, avgMin, avgMax

            try:
                fitSin(imgDir, valsWithoutNan, timeWithoutNan, vessel, zone, (mean_peak_min*-1)-std_min, mean_peak_max+std_max, mean)
                #plt.plot(timeWithoutNan, valsWithoutNan)
            except:
                #import pdb; pdb.set_trace()
                print 'Error fitting curve to vessel '+vessel

            if saveToDb:
                sql = "Select id from results where sequence_id = '"+sequence_id+"' and zone='"+zone+"' and vessel='"+vessel+"'"
                try:
                    print 'Updating SQL entries'
                    cur.execute(sql)
                    results = cur.fetchall()

                    for row in results:
                        sql = "UPDATE results \
                                SET offset = " + str(mean) + ", avgMin = " + str(mean - std) + ", avgMax = " + str(mean + std) + ", elasticity = "+str(elasticity)+" \
                                WHERE id = "+str(row[0])
                        cur.execute(sql)
                    db.commit()

                except:
                    print "Error: unable to fetch data"

    # disconnect from server
    if saveToDb:
        db.close()
    np.save(imgDir+"\\test\\allSystolicCurves.npy", allSystolicCurves)
    return

def evaluateAll(baseDir):

    baseDir ='\\'.join(baseDir.split('\\')[:-2])+'\\'

    onlyThese = ["1273-Day0-Seq1", "1273-Day0-Seq3", "1650-Day0-Seq1-IR1", "1641-Day0-Seq4-IR1", "1650-Day0-Seq4-IR1",
                 "244-Day0-Seq1", "191-Day0-Seq1-IR1", "200-Day0-Seq1", "191L-Day0-Seq3-IR1"]
    #onlyThese = ["1273-Day0-Seq1","1273-Day0-Seq3"]

    plt.clf()
    i = 0
    colors = utils.get_spaced_colors(len(onlyThese))
    colors = [(0, 0, 0),
              (1, 0.48, 0),
              (.5, 0, 0),
              (1.0, 0, 0),
              (0, .33, .33),
              (0, .5, 0),
              (0, 1.0, 0),
              (.33, .33, 0),
              (1.0, 1.0, 0),
              (0, 0, 1.0),
              (.5, .5, .5)]
    speedThreshold = .235
    elasticityThreshold = 5.0
    for folder in onlyThese:
        #colors[i] = (float(colors[i][0]) / 255.0, float(colors[i][1]) / 255.0, float(colors[i][2]) / 255.0)
        print colors[i]
        data = np.load(baseDir + folder+"\\pipeline1\\test\\elasticityProfile.npy")
        expansionSpeed = data[0]
        elasticity = data[1]
        keys = sorted(expansionSpeed.keys())
        if len(keys) > 0:
            leftOfThreshold = float(len([expansionSpeed[k] for k in keys if expansionSpeed[k] < speedThreshold]))
            countLowerLeft = 0.0
            for leftVal in [expansionSpeed[k] for k in keys if expansionSpeed[k] < speedThreshold]:
                index = [expansionSpeed[k] for k in keys].index(leftVal)
                if [elasticity[k] for k in keys][index]<elasticityThreshold:
                    countLowerLeft += 1.0

            totalPoints = float(len(expansionSpeed))

            plt.plot([expansionSpeed[k] for k in keys], [elasticity[k] for k in keys], 'o', color=colors[i],
                 label=folder+' '+str(int((leftOfThreshold/totalPoints)*100))+'%, '+str(int((countLowerLeft)/(totalPoints)*100))+'%')
            #import pdb; pdb.set_trace()
        i += 1
    plt.axhline(y=elasticityThreshold, linewidth=1, color='red')
    plt.axvline(x=speedThreshold, linewidth=1, color='red')
    plt.xlim(0.09, 0.4)
    plt.ylim(0.0, 10.0)
    plt.title('Systolic curve distribution')
    plt.legend()
    plt.xlabel('time in s')
    plt.ylabel('vessel elasticity in %')
    plt.show()

def fit_sin(tt, yy):
    '''Src: https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy'''
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    #print 'Freq: '+str(guess_freq)
    guess_freq = clamp(guess_freq, 0.5, 3.0) # clamp the frequency to not bee beyond human possible
    #print 'ClmpedFreq: ' + str(guess_freq)
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
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

def fitSin(imgDir,vals, xaxis, vessel, zone, avgMin, avgMax, mean):
    print 'Fitting Sin to Zone '+zone+' vessel #'+vessel
    N, amp, omega, phase, offset, noise = 500, 1., 2., .5, 4., 3
    # N, amp, omega, phase, offset, noise = 50, 1., .4, .5, 4., .2
    # N, amp, omega, phase, offset, noise = 200, 1., 20, .5, 4., 1
    tt = np.linspace(0, 10, N)
    yy = amp * np.sin(omega * tt + phase) + offset

    #xaxis = np.array([k for k in range(0, len(vals))])
    vals = np.array(vals)
    xaxis = np.array(xaxis)
    res = fit_sin(xaxis, vals)
    #print("Amplitude=%(amp)s, Angular freq.=%(omega)s, phase=%(phase)s, offset=%(offset)s, Max. Cov.=%(maxcov)s" % res)
    plt.figure(figsize=(15, 8))
    plt.gca().set_position((.1, .3, .8, .6))
    plt.plot(xaxis, vals, "-k", label="measured line", linewidth=1)
    plt.plot(xaxis, vals, "ok", label="measured data", linewidth=1)
    plt.plot(xaxis, res["fitfunc"](xaxis), "r-", label="fit sin", linewidth=2)

    #pdb.set_trace()

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
    elasticity = float("{0:.2f}".format(((maxAvg / minAvg) - 1.0) * 100.0))

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
        'elasticity': ((maxAvg/minAvg)-1.0)*100.0,
        'bpm': fitFreq*60,
        'phase': res['phase'],
        'offset': res['offset'],
        'score': fitScore
    }
    return retDict

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def detect_peaks(time, vals, thresh = 0.02, minDist = 100):
    indexes = peakutils.indexes(np.array(vals), thres=thresh / max(vals), min_dist=minDist)
    mean_peak = np.mean([vals[k] for k in indexes])
    mean_peak = np.mean(reject_outliers(np.array([vals[k] for k in indexes]),1.2))
    std = np.std([vals[k] for k in indexes])
    std = np.std(reject_outliers(np.array([vals[k] for k in indexes]),1.2))
    return indexes, mean_peak, std

def findSystolicCurve(minPeaks, maxPeaks, valsWithoutNan, timeWithoutNan, zone, vessel):
    threshold = 0.30
    systolicCurves = []
    if (len(maxPeaks)>2 and len(minPeaks)>2):

        if maxPeaks[0] > minPeaks[0]:
            firstIndex = minPeaks
            secondIndex = maxPeaks
        else:
            firstIndex = maxPeaks
            secondIndex = minPeaks

        k = 0;
        while len(firstIndex)>1 and len(secondIndex)>0:
            diff = valsWithoutNan[firstIndex[0]]-valsWithoutNan[secondIndex[0]]
            diff2 = valsWithoutNan[firstIndex[1]]-valsWithoutNan[secondIndex[0]]
            #diff3 = valsWithoutNan[startIndex[i]] - valsWithoutNan[secondIndex[i]]

            print str(abs((diff*.35)))+" "+str(abs(diff2))+" "+str(abs((diff*.65)))

            if abs((diff*(0.5-threshold/2.0))) <= abs(diff2)  <= abs((diff*(0.5+threshold/2.0))):
                #print "opener is "+str(timeWithoutNan[firstIndex[0]])+", "+str(timeWithoutNan[secondIndex[0]])
                try:
                    #import pdb; pdb.set_trace()
                    if valsWithoutNan[firstIndex[1]] < valsWithoutNan[secondIndex[1]] < valsWithoutNan[secondIndex[0]]:
                        # diastolic peek maybe detected
                        plt.axvspan(timeWithoutNan[firstIndex[0]], timeWithoutNan[firstIndex[1]], color='red',
                                    alpha=0.5)

                        # check if the small diastolic peak is in the lower 2/3 of the large slope
                        largeSlope = valsWithoutNan[secondIndex[0]] - valsWithoutNan[firstIndex[1]]
                        smallPeak = valsWithoutNan[secondIndex[1]] - valsWithoutNan[firstIndex[1]]
                        if smallPeak <= largeSlope * .67:
                            indeces = [firstIndex[0], secondIndex[0], firstIndex[1], secondIndex[1]]
                            plt.axvspan(timeWithoutNan[firstIndex[1]], timeWithoutNan[secondIndex[1]], color='green',alpha=0.5)
                            systolicCurves.append({'complete': True, 'times': [timeWithoutNan[k] for k in indeces], 'vals': [valsWithoutNan[k] for k in indeces], 'zone': zone, 'vessel': vessel})
                except:
                    indeces = [firstIndex[0], secondIndex[0], firstIndex[1]]
                    systolicCurves.append({'complete': False, 'times': [timeWithoutNan[k] for k in indeces],
                                           'vals': [valsWithoutNan[k] for k in indeces], 'zone': zone, 'vessel': vessel})

            else:
                print "no opener at "+str(timeWithoutNan[firstIndex[0]])+", "+str(timeWithoutNan[secondIndex[0]])

            firstIndex = np.delete(firstIndex, 0, 0)

            if len(firstIndex)>0 and len(secondIndex)>0:
                tmpIndex = []
                if firstIndex[0]>secondIndex[0]:
                    print 'second becomes first'
                    tmpIndex = firstIndex
                    firstIndex = secondIndex
                    secondIndex = tmpIndex
            else:
                break
            k += 1
    return systolicCurves

def evaluateSystolicCurves(imgDir, baseDir):

    db = MySQLdb.connect(host="localhost",  # your host, usually localhost
                         user="root",  # your username
                         passwd="",  # your password
                         db="vesselanalysis")  # name of the data base
    cur = db.cursor()
    sequence_id = baseDir.split("\\")[-2]

    allSystolicCurves = np.load(imgDir+"\\test\\allSystolicCurves.npy")
    goodResults = []
    colorB = 'green'
    colorC = 'blue'
    veins = []
    for curves in allSystolicCurves:
        if len(curves)>0 and len([curve for curve in curves if curve['complete']])>0:
            for curve in [curve for curve in curves if curve['complete']]:

                sql = "Select verified from results where sequence_id = '" + sequence_id + "' and zone='" + curve['zone'] + "' and vessel='" + curve['vessel'] + "' and vesselClass='1'"
                cur.execute(sql)
                results = cur.fetchall()

                #import pdb; pdb.set_trace()
                if len(results)>0 and results[0][0] == 'True':
                    if curve['zone'] == "B":
                        color = colorB
                    else:
                        color = colorC
                    plt.plot(curve['times'], curve['vals'], c=np.random.rand(3,), label=curve['zone']+'_'+curve['vessel'])
                    plt.plot(curve['times'][1], curve['vals'][1], 'o' ,c=color)
                    goodResults.append(curve)
    plt.title('Systolic curve distribution')
    plt.legend()
    plt.xlabel('time in s')
    plt.ylabel('vessel width in pixel')
    plt.show()

    expansionSpeed = {}
    relaxationSpeed = {}

    expansionRate = {}
    elasticity = {}
    relaxationRate = {}
    import operator
    for curve in goodResults:
        expansionSpeed[curve["vals"][0]] = curve['times'][1] - curve['times'][0]
        relaxationSpeed[curve["vals"][0]] = curve['times'][2] - curve['times'][1]

        expansionRate[curve["vals"][0]] = abs(curve['vals'][1] / curve['vals'][0]) / (curve['times'][1] - curve['times'][0])
        relaxationRate[curve["vals"][0]] = abs(curve['vals'][2] / curve['vals'][1]) / (curve['times'][2] - curve['times'][1])

        elasticity[curve["vals"][0]] = ((curve['vals'][1] / curve['vals'][0])-1.0)*100.0

    keys = sorted(expansionSpeed.keys())
    plt.clf()
    plt.plot([expansionSpeed[k] for k in keys], [elasticity[k] for k in keys], 'o',color = 'red', label='expansion speed')
    #plt.plot(keys, [relaxationRate[k] for k in keys], color='green', label='relaxation speed')
    #plt.show()

    np.save(imgDir+"\\test\\elasticityProfile.npy", (expansionSpeed,elasticity))

    #import pdb; pdb.set_trace()
    print "done"

def updateVesselData(imgDir, arteries, veins):
    print '------writing vessel classes to DB'
    db = MySQLdb.connect(host="localhost",  # your host, usually localhost
                         user="root",  # your username
                         passwd="",  # your password
                         db="vesselanalysis")  # name of the data base
    cur = db.cursor()

    for artery in arteries:
        sql = "UPDATE results \
                SET vesselClass = 1, verified = 'True' \
                WHERE sequence_id = '"+imgDir.split("\\")[-3]+"' and vessel = '"+artery+"' and outlierScore<=0.1"
        cur.execute(sql)
    for vein in veins:
        sql = "UPDATE results \
                SET vesselClass = 2, verified = 'Manual' \
                WHERE sequence_id = '"+imgDir.split("\\")[-3]+"' and vessel = '"+vein+"'"
        cur.execute(sql)
    db.commit()

def changeElasticityValues(analysisDir):
    print '------writing vessel classes to DB'
    db = MySQLdb.connect(host="localhost",  # your host, usually localhost
                         user="root",  # your username
                         passwd="",  # your password
                         db="vesselanalysis")  # name of the data base
    cur = db.cursor()
    sql = 'SELECT id, avgMin, avgMax, elasticity FROM vesselanalysis.results WHERE \
    verified = "True" and vesselClass = 1 and offset >= 10.0'

    cur.execute(sql)
    results = cur.fetchall()

    if len(results) > 0:
        import pdb;
        pdb.set_trace()

        np.save(analysisDir+'\\stdElasticityValues', results)


def exportHighcharts(analysisDir):
    from scipy.stats.mstats import mquantiles
    print '------export highcharts'
    db = MySQLdb.connect(host="localhost",  # your host, usually localhost
                         user="root",  # your username
                         passwd="",  # your password
                         db="vesselanalysis")  # name of the data base
    cur = db.cursor()
    sql = 'SELECT sequence_id, elasticity, avgMin, avgMax FROM vesselanalysis.results WHERE \
    verified = "True" and vesselClass = 1 and offset >= 10.0'

    cur.execute(sql)
    results = cur.fetchall()
    allSequences = {}
    for result in results:
        seqKey = result[0].split('-')[0]
        if (seqKey[-1] == 'L' or seqKey[-1] == 'R'):
            seqKey = seqKey[:-1]
        #import pdb; pdb.set_trace()
        if seqKey in allSequences.keys():
            allSequences[seqKey].append(result[1])
        else:
            allSequences[seqKey] = []
            allSequences[seqKey].append(result[1])

    sequenceBloodPressures = utils.getSequenceBloodPressure()
    sequenceDiastolicBloodPressures = utils.getSequenceDiastolicBloodPressure()

    statisticData = {}
    for key in allSequences.keys():
        allSequences[key] = sorted(allSequences[key])
        #find and exclude outliers

        outlier = utils.doubleMADsfromMedian(allSequences[key])
        valsWithoutOutlier = [float("{0:.2f}".format(allSequences[key][k])) for k in range(0,len(allSequences[key])) if not outlier[k]]
        valsOutlier = [float("{0:.2f}".format(allSequences[key][k])) for k in range(0, len(allSequences[key])) if outlier[k]]

        statisticData[key] =    {
                                'quantiles': [ float("{0:.2f}".format(k)) for k in mquantiles(valsWithoutOutlier)],
                                'low': np.min(valsWithoutOutlier),
                                'high': np.max(valsWithoutOutlier),
                                'outliers': valsOutlier,
                                'category': sequenceBloodPressures[key],
                                'diastolic': sequenceDiastolicBloodPressures[key]
                                }
        #import pdb; pdb.set_trace()

    import json

    categories = []
    data = []
    outliers = []
    for key in statisticData.keys():
        col = utils.hex_code_colors()
        data.append(
            {
                'x': statisticData[key]['category'],
                'low': statisticData[key]['low'],
                'q1': statisticData[key]['quantiles'][0],
                'median': statisticData[key]['quantiles'][1],
                'q3': statisticData[key]['quantiles'][2],
                'high': statisticData[key]['high'],
                'name': key,
                'color': col
            }
        )

        categories.append(statisticData[key]['category'])
        for outlier in statisticData[key]['outliers']:
            outliers.append([statisticData[key]['category'],outlier])

    from operator import itemgetter
    sortedData = sorted(data, key=itemgetter('x'))
    sortedCategories = sorted(categories)
    sortedOutliers = sorted(outliers)

    #import pdb; pdb.set_trace()

    with open(analysisDir+'categories.json', 'w') as fp:
        json.dump(sortedCategories, fp)

    with open(analysisDir + 'data.json', 'w') as fp:
        json.dump(sortedData, fp)

    with open(analysisDir + 'outliers.json', 'w') as fp:
        json.dump(sortedOutliers, fp)

    keys = sorted([int(k) for k in statisticData.keys()])
    import csv
    with open(analysisDir+'qData.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['id']+['q1'] + ['median'] + ['q3'])
        for k in keys:
            spamwriter.writerow([str(k), statisticData[str(k)]['quantiles'][0], statisticData[str(k)]['quantiles'][1], statisticData[str(k)]['quantiles'][2]])
            #spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

    return statisticData

def initializeIndividuals(analysisDir, statisticData):

    print '------initializing individuals'
    db = MySQLdb.connect(host="localhost",  # your host, usually localhost
                         user="root",  # your username
                         passwd="",  # your password
                         db="vesselanalysis")  # name of the data base
    cur = db.cursor()

    f = open(analysisDir+'mlClassificationData.csv')
    header = False
    enrichedData = {}
    for line in f:
        if not header:
            header = True
        else:
            split = line.replace('\n','').split(',')
            id = split[0]
            enrichedData[id] = {
                'e2': split[1],
                'e4': split[2],
                'hypertension': split[3],
                'bp': split[4],
                'gender': split[8],
                'age': split[9],
                'suvr': split[10],
                'diastolic': split[11],
                'cl': split[12]
            }

    for key in statisticData.keys():

        id = key
        bp = str(statisticData[key]['category'])
        q1 = str(statisticData[key]['quantiles'][0])
        q2 = str(statisticData[key]['quantiles'][1])
        q3 = str(statisticData[key]['quantiles'][2])

        try:
            gender = enrichedData[key]['gender']
            age = enrichedData[key]['age']
            suvr = enrichedData[key]['suvr']
            e2 = enrichedData[key]['e2']
            e4 = enrichedData[key]['e4']
            cl = enrichedData[key]['cl']
            diastolic = enrichedData[key]['diastolic']
            hypertension = enrichedData[key]['hypertension']
        except:
            gender = 'n/a'
            age = 'n/a'
            suvr = 'n/a'
            e2 = 'n/a'
            e4 = 'n/a'
            cl = 'n/a'
            diastolic = 'n/a'
            hypertension = 'n/a'

        try:
            sql = 'INSERT INTO individuals (id, bp, E2, E4, q1, q2, q3, gender, age, suvr, class, hypertension, diastolic) VALUES ("'+id+'","'+bp+'","'+e2+'","'+e4+'","'+q1+'","'+q2+'","'+q3+'","'+gender+'","'+age+'","'+suvr+'","'+cl+'","'+hypertension+'","'+diastolic+'");'
            #print sql
            cur.execute(sql)
            print key+" inserted to DB"
        except:
            print key+' already in DB'

    db.commit()

def featureAnalysis(analysisDir):
    # Feature Importance with Extra Trees Classifier

    from pandas import read_csv

    print '------performing feature analysis'

    db = MySQLdb.connect(host="localhost",  # your host, usually localhost
                         user="root",  # your username
                         passwd="",  # your password
                         db="vesselanalysis")  # name of the data base
    cur = db.cursor()
    sql = 'SELECT * FROM individuals where class != "n/a" ;'
    cur.execute(sql)
    results = cur.fetchall()

    classMap = { "HC": 0, "AD":3, "MCI.pos": 2, "MCI.neg": 1 }
    allClasses = {'HC':[],'AD':[],'MCI.pos':[],'MCI.neg':[]}
    #
    names = ['BP','E4','q1','q2','q3','gender','age','suvr','E2','class']
    for result in results:
        result = [k for k in result]
        if result[6] == 'F':
            gender = 0
        else:
            gender = 1
        if result[8]=='n/a':
            result[8] = 1.0
        if result[1]=='n/a':
            result[1]=120
        bp = result[1]
        try:
            tmpArr = [
                result[0], #id
                int(result[1]), int(result[2]),     #bp e4
                float(bp)/float(result[3]), float(bp)/float(result[4]), float(bp)/float(result[5]), #q1 q2 q3
                gender, float(result[7]), float(result[8]),int(result[10]),  #gender age suvr e2
                classMap[result[9]] #class
            ]
        except:
            print 'Error while creating classification array'
        allClasses[result[9]].append(tmpArr)

    import csv
    url = analysisDir+"classification.data"
    with open(url, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #spamwriter.writerow([names[0]]+[names[1]]+[names[2]]+[names[3]]+[names[4]]+[names[5]]+[names[6]]+[names[7]]+[names[8]]+[names[9]])
        for Class in allClasses.keys():
            for vals in allClasses[Class]:
                spamwriter.writerow(vals)

    dataframe = read_csv(url, names=names)

    import pdb; pdb.set_trace()

    statistics = []
    statisticsWithoutEl = []

    statisticsHC = []
    statisticsHCWithoutEl = []

    statisticsAD = []
    statisticsADWithoutEl = []

    statisticsMCIpos = []
    statisticsMCIposWithoutEl = []

    statisticsMCIneg = []
    statisticsMCInegWithoutEl = []

    import copy
    worse = 0

    allTrainData = []
    allTrainDataWithoutEl = []

    for k in range(0,1000):
        print k
        #Initialize random data
        trainData, testHC, testAD, testMCIp, testMCIn = getTrainTestData(dataframe.copy())

        #import pdb; pdb.set_trace()

        trainData1 = copy.deepcopy(trainData)
        testHC1 = copy.deepcopy(testHC)
        testAD1 = copy.deepcopy(testAD)
        testMCIp1 = copy.deepcopy(testMCIp)
        testMCIn1 = copy.deepcopy(testMCIn)

        trainData1, testHC1, testAD1, testMCIp1, testMCIn1 = removeValues(trainData1, testHC1, testAD1, testMCIp1, testMCIn1,
                                                                     [0,5,6,8])
        stats = performModelTest(trainData1, names, testHC1, testMCIn1, testMCIp1, testAD1)

        trainData, testHC, testAD, testMCIp, testMCIn = removeValues(trainData, testHC, testAD, testMCIp, testMCIn, [0,2,3,4,5,6,8])
        statsWithoutEl = performModelTest(trainData, names, testHC, testMCIn, testMCIp, testAD)

        #if stats['scores']['HC']>statsWithoutEl['scores']['HC'] or stats['scores']['AD']>statsWithoutEl['scores']['AD'] or stats['scores']['MCIpos']>statsWithoutEl['scores']['MCIpos'] or stats['scores']['MCIneg']>statsWithoutEl['scores']['MCIneg']:
        statistics.append(stats)
        statisticsWithoutEl.append(statsWithoutEl)
        if stats['scores']['HC']<statsWithoutEl['scores']['HC'] or stats['scores']['AD']<statsWithoutEl['scores']['AD'] or stats['scores']['MCIpos']<statsWithoutEl['scores']['MCIpos'] or stats['scores']['MCIneg']<statsWithoutEl['scores']['MCIneg']:
            worse +=1

        if stats['scores']['HC']>statsWithoutEl['scores']['HC']:
            statisticsHC.append(stats)
            statisticsHCWithoutEl.append(statsWithoutEl)
        if stats['scores']['AD']>statsWithoutEl['scores']['AD']:
            statisticsAD.append(stats)
            statisticsADWithoutEl.append(statsWithoutEl)
        if stats['scores']['MCIpos']>statsWithoutEl['scores']['MCIpos']:
            statisticsMCIpos.append(stats)
            statisticsMCIposWithoutEl.append(statsWithoutEl)
        if stats['scores']['MCIneg']>statsWithoutEl['scores']['MCIneg']:
            statisticsMCIneg.append(stats)
            statisticsMCInegWithoutEl.append(statsWithoutEl)

        #import pdb; pdb.set_trace()

        allTrainData.append(trainData1)
        allTrainDataWithoutEl.append(trainData)

    print '#improvements: '+str(len(statistics))+' / '+str(k+1)+' = '+str(float(len(statistics))/float(k+1))
    print '#worse: ' + str(worse) + ' / ' + str(k + 1) + ' = ' + str(
        float(worse) / float(k + 1))

    print '#HCimp: ' + str(len(statisticsHC)) + ' / ' + str(k+1) + ' = ' + str(float(len(statisticsHC)) / float(k + 1))
    print '#ADimp: ' + str(len(statisticsAD)) + ' / ' + str(k+1) + ' = ' + str(float(len(statisticsAD)) / float(k + 1))
    print '#MCIposimp: ' + str(len(statisticsMCIpos)) + ' / ' + str(k+1) + ' = ' + str(float(len(statisticsMCIpos)) / float(k + 1))
    print '#MCInegimp: ' + str(len(statisticsMCIneg)) + ' / ' + str(k+1) + ' = ' + str(float(len(statisticsMCIneg)) / float(k + 1))

    #import pdb; pdb.set_trace()
    utils.printStatisticResults(statistics, statisticsWithoutEl)

    '''
    q1 = np.mean([k['q1'] for k in features])
    q2 = np.mean([k['q2'] for k in features])
    q3 = np.mean([k['q3'] for k in features])
    elaSum = np.sum([q1,q2,q3])
    suvr = np.mean([k['suvr'] for k in features])
    e4 = np.mean([k['E4'] for k in features])
    '''
    np.save(analysisDir+'\\featureAnalysisStatistics.npy', statistics)
    np.save(analysisDir + '\\featureAnalysisStatisticsWithoutEl.npy', statisticsWithoutEl)
    np.save(analysisDir + '\\traindata.npy', allTrainData)
    np.save(analysisDir + '\\traindata1.npy', allTrainDataWithoutEl)

    print 'DONE'

def evaluateImprovement(analysisDir):
    print 'Evaluating improvement...'
    statistics = np.load(analysisDir+'\\featureAnalysisStatistics.npy')
    statisticsWithoutEl = np.load(analysisDir + '\\featureAnalysisStatisticsWithoutEl.npy')


    plt.title('MCI Accuracy')
    plt.plot([k['MCIGroup']['Accuracy'] for k in statistics], '-', color=(1,0,0), label='+El')
    plt.plot([k['MCIGroup']['Accuracy'] for k in statisticsWithoutEl], '-', color=(0,1,0), label='-El')
    plt.legend()
    plt.show()

    plt.title('MCI Sensitivity')
    plt.plot([k['MCIGroup']['Sensitivity'] for k in statistics], '-', color=(1, 0, 0), label='+El')
    plt.plot([k['MCIGroup']['Sensitivity'] for k in statisticsWithoutEl], '-', color=(0, 1, 0), label='-El')
    plt.legend()
    plt.show()

    plt.title('MCI Specificity')
    plt.plot([k['MCIGroup']['Specificity'] for k in statistics], '-', color=(1, 0, 0), label='+El')
    plt.plot([k['MCIGroup']['Specificity'] for k in statisticsWithoutEl], '-', color=(0, 1, 0), label='-El')
    plt.legend()
    plt.show()

    trainData = np.load(analysisDir+'\\traindata.npy')
    trainData1 = np.load(analysisDir + '\\traindata1.npy')

    import csv
    from pandas import read_csv
    names = ['BP', 'E4', 'q1', 'q2', 'q3', 'gender', 'age', 'suvr', 'E2', 'class']
    url = analysisDir + "classification.data"
    dataframe = read_csv(url, names=names)

    allTrainIds = []
    for trainInterval in trainData:
        trainIds = []
        for trainVals in trainInterval:
            id = dataframe.loc[(dataframe['q1'] == trainVals[2]) & (dataframe['q2'] == trainVals[3]) & (dataframe['q3'] == trainVals[4])].index[0]
            trainIds.append(id)

        allTrainIds.append(trainIds)


def removeValues(trainData, testHC, testAD, testMCIp, testMCIn, elIndices):
    # sets the elasticity values at given indices to 0.0
    for i in range (0,len(trainData)):
        for index in elIndices:
            trainData[i][index] = 0.0

    for i in range (0,len(testHC)):
        for index in elIndices:
            testHC[i][index] = 0.0

    for i in range (0,len(testAD)):
        for index in elIndices:
            testAD[i][index] = 0.0

    for i in range (0,len(testMCIp)):
        for index in elIndices:
            testMCIp[i][index] = 0.0

    for i in range (0,len(testMCIn)):
        for index in elIndices:
            testMCIn[i][index] = 0.0

    return trainData, testHC, testAD, testMCIp, testMCIn

def getTrainTestData(df):
    percent = 70
    HC_train, HC_test = getRandomQuantiles(df.loc[df['class'] == 0].values, percent)
    MCIneg_train, MCIneg_test = getRandomQuantiles(df.loc[df['class'] == 1].values, percent)
    MCIpos_train, MCIpos_test = getRandomQuantiles(df.loc[df['class'] == 2].values, percent)
    AD_train, AD_test = getRandomQuantiles(df.loc[df['class'] == 3].values, percent)

    '''
    print 'class, train, test'
    print 'HC,'+str(len(HC_train))+','+str(len(HC_test))
    print 'MCIneg,' + str(len(MCIneg_train)) + ',' + str(len(MCIneg_test))
    print 'MCIpos,' + str(len(MCIpos_train)) + ',' + str(len(MCIpos_test))
    print 'AD,' + str(len(AD_train)) + ',' + str(len(AD_test))
    '''

    trainData = np.concatenate((HC_train, MCIneg_train, MCIpos_train, AD_train), axis=0)
    #import pdb; pdb.set_trace()
    return trainData, HC_test, AD_test, MCIpos_test, MCIneg_test


def getRandomQuantiles(inputArr, percent):
    import random
    arr = []

    indices = range(0,len(inputArr))
    random.shuffle(indices)
    for k in indices:
        arr.append(inputArr[k])
    arr = np.array(arr)
    #import pdb; pdb.set_trace()

    #random.shuffle(arr)
    interval = len(arr) * percent / 100
    train_data = arr[:interval]
    test_data = arr[interval:]
    #import pdb; pdb.set_trace()
    return train_data, test_data

def performModelTest(train_data, names, testHC, testMCIn, testMCIp, testAD):
    from sklearn.ensemble import ExtraTreesClassifier

    X = train_data[:, 0:len(names)-1]
    Y = train_data[:, len(names)-1]

    # feature extraction
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    flist = model.feature_importances_

    fImportances = {}
    import operator
    for k in range(0,len(names)-1):
        fImportances[names[k]] = flist[k]

    sorted_importances = sorted(fImportances.items(), key=operator.itemgetter(1), reverse = True)

    try:
        X_HC_test = testHC[:, 0:len(names) - 1]
        Y_HC_test = testHC[:, len(names) - 1]
    except:
        import pdb; pdb.set_trace()

    X_MCIn_test = testMCIn[:, 0:len(names) - 1]
    Y_MCIn_test = testMCIn[:, len(names) - 1]

    X_MCIp_test = testMCIp[:, 0:len(names) - 1]
    Y_MCIp_test = testMCIp[:, len(names) - 1]

    X_AD_test = testAD[:, 0:len(names) - 1]
    Y_AD_test = testAD[:, len(names) - 1]

    '''
    print '---------SCORES:'
    print 'HC: '+str(model.score(X_HC_test, Y_HC_test))
    print 'MCIneg: ' + str(model.score(X_MCIn_test, Y_MCIn_test))
    print 'MCIpos: ' + str(model.score(X_MCIp_test, Y_MCIp_test))
    print 'AD: ' + str(model.score(X_AD_test, Y_AD_test))
    print '---------------'
    '''

    # scores = mean accuracy
    scores = {'HC': model.score(X_HC_test, Y_HC_test),
              'MCIneg': model.score(X_MCIn_test, Y_MCIn_test),
              'MCIpos': model.score(X_MCIp_test, Y_MCIp_test),
              'AD': model.score(X_AD_test, Y_AD_test)}

    # PETGroup = [AD, MCIpos] - [MCIneg, HC]
    PETpos_X = np.concatenate((X_MCIp_test, X_AD_test), axis=0)
    PETneg_X = np.concatenate((X_MCIn_test, X_HC_test), axis=0)
    PETpos_predict = model.predict(PETpos_X)
    PETneg_predict = model.predict(PETneg_X)
    PETGroup = getConfusionParameters(PETpos_predict, PETneg_predict ,len(PETpos_X),len(PETneg_X))

    # CIGroup = [AD, MCIpos, MCIneg] - [HC]
    CIpos_X = np.concatenate((X_MCIn_test, X_MCIp_test, X_AD_test), axis=0)
    CIneg_X = X_HC_test
    CIpos_predict = model.predict(CIpos_X)
    CIneg_predict = model.predict(CIneg_X)
    CIGroup = getConfusionParameters(CIpos_predict, CIneg_predict ,len(CIpos_X),len(CIneg_X))

    # MCIGroup = [MCIpos],[MCIneg]
    MCIpos_X = X_MCIp_test
    MCIneg_X = X_MCIn_test
    MCIpos_predict = model.predict(MCIpos_X)
    MCIneg_predict = model.predict(MCIneg_X)
    MCIGroup = getConfusionParameters(MCIpos_predict, MCIneg_predict ,len(MCIpos_X),len(MCIneg_X))

    '''
    print '---------Features:'
    for imp in sorted_importances:
        print imp[0]+': '+str(imp[1])
    '''

    model = None

    returnStatistics = {
        'featureImportances': fImportances,
        'scores': scores,
        'PETGroup': PETGroup,
        'CIGroup': CIGroup,
        'MCIGroup': MCIGroup
    }

    return returnStatistics

def getConfusionParameters(groupK, groupH, nK, nH):
    TP = np.sum([1 for k in groupK if k >= 2])
    TN = np.sum([1 for k in groupH if k < 2])

    FN = nK - TP
    FP = nH - TN

    ErrorRate = float(FP + FN) / float(TP + FP + FN + TN)
    Accuracy = float(TP + TN) / float(TP + FP + FN + TN)
    Sensitivity = float(TP) / float(TP + FN)  # recall (REC) or true positive rate (TPR)
    Specificity = float(TN) / float(TN + FP)  # true negative rate (TNR)

    return {'ErrorRate': ErrorRate, 'Accuracy': Accuracy, 'Sensitivity': Sensitivity, 'Specificity': Specificity}

def performCorrelationAnalysis(analysisDir):
    import csv
    from pandas import read_csv
    names = ['BP', 'E4', 'q1', 'q2', 'q3', 'gender', 'age', 'suvr', 'E2', 'class']
    url = analysisDir + "classification.data"
    dataframe = read_csv(url, names=names)

    '''
    df_AD = dataframe.loc[dataframe['class'] == 3]
    df_HC = dataframe.loc[dataframe['class'] == 0]

    utils.plotCorrelationMatrix(df_HC.corr(), 'Correlation Matrix of HC Sequences')
    utils.plotCorrelationMatrix(df_AD.corr(), 'Correlation Matrix of AD Sequences')
    utils.plotCorrelationMatrix(df_HC.corr() - df_AD.corr(), 'Correlation Differences of HC/AD')
    '''

    dataframe['q1'] = dataframe['BP'] / dataframe['q1']
    dataframe['q2'] = dataframe['BP'] / dataframe['q2']
    dataframe['q3'] = dataframe['BP'] / dataframe['q3']

    df_PETpos = dataframe.loc[(dataframe['class'] == 3) | (dataframe['class'] == 2)]
    df_PETneg = dataframe.loc[(dataframe['class'] == 0) | (dataframe['class'] == 1)]
    #import pdb;
    #pdb.set_trace()
    utils.plotCorrelationMatrix(df_PETpos.corr(), 'Correlation Matrix of PET+ Sequences')
    utils.plotCorrelationMatrix(df_PETneg.corr(), 'Correlation Matrix of PET- Sequences')
    utils.plotCorrelationMatrix(df_PETneg.corr() - df_PETpos.corr(), 'Correlation Differences of PET+/PET-')

    '''
    df_SUVRpos = dataframe.loc[dataframe['suvr'] > 1.5]
    df_SUVRneg = dataframe.loc[dataframe['suvr'] <= 1.5]

    import pdb;
    pdb.set_trace()
    utils.plotCorrelationMatrix(df_SUVRpos.corr(), 'Correlation Matrix of SUVR+ Sequences')
    utils.plotCorrelationMatrix(df_SUVRneg.corr(), 'Correlation Matrix of SUVR- Sequences')
    utils.plotCorrelationMatrix(df_SUVRneg.corr() - df_SUVRpos.corr(), 'Correlation Differences of SUVR+/SUVR-')
    
    

    df_E4pos = dataframe.loc[dataframe['E4'] > 0]
    df_E4neg = dataframe.loc[dataframe['E4'] == 0]

    import pdb;
    pdb.set_trace()
    utils.plotCorrelationMatrix(df_E4pos.corr(), 'Correlation Matrix of E4+ Sequences')
    utils.plotCorrelationMatrix(df_E4neg.corr(), 'Correlation Matrix of E4- Sequences')
    utils.plotCorrelationMatrix(df_E4neg.corr() - df_E4pos.corr(), 'Correlation Differences of E4+/E4-')
    '''



