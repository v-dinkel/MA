from operator import itemgetter
import numpy as np
from Configuration import Config as cfg
from Tools import utils

def splineMapping(Img0, ImgName, prev_dict_splinePoints_updated, dict_splinePoints_updated):

    splineMap = {}
    splineDistances = {}
    if (len(prev_dict_splinePoints_updated.keys()) > 0):
        print 'BEGIN CALCULATING SPLINE SIMILARITIES'

        # Calculate euclidean distances between the splines
        for key1 in prev_dict_splinePoints_updated.keys():
            for key2 in dict_splinePoints_updated.keys():

                # get coordinates
                xy1 = zip([x for x in prev_dict_splinePoints_updated[key1][0]],
                          [y for y in prev_dict_splinePoints_updated[key1][1]])
                xy2 = zip([x for x in dict_splinePoints_updated[key2][0]],
                          [y for y in dict_splinePoints_updated[key2][1]])

                # source: https://stackoverflow.com/questions/1871536/euclidean-distance-between-points-in-two-different-numpy-arrays-not-within
                # --> second answer
                d0 = np.subtract.outer([x[0] for x in xy1], [x[0] for x in xy2])
                d1 = np.subtract.outer([x[1] for x in xy1], [x[1] for x in xy2])
                ret = np.hypot(d0, d1)

                # make sure to match the smaller vessel on the larger vessel
                nRows = ret.shape[0]
                nCols = ret.shape[1]
                if nRows > nCols:
                    ret = ret.transpose()

                sumVals = 0.0
                for vals in ret:
                    sumVals += min(vals)  # take power of 2 to punish further distances ?

                if key2 in splineDistances.keys():
                    splineDistances[key2][key1] = sumVals / len(ret)
                else:
                    splineDistances[key2] = {}
                    splineDistances[key2][key1] = sumVals / len(ret)

        # generate splineMap dictionary with sorted distances
        assignedKeys = []
        for key2 in splineDistances.keys():
            splineDistances[key2] = sorted(splineDistances[key2].items(), key=itemgetter(1))
            if key2 not in splineMap.keys():
                if splineDistances[key2][0][0] in assignedKeys:
                    print 'Key has already bin assigned'
                    # [splineMap[k][0] for k in splineMap.keys()]
                    # Key to Map to old Key: key2 to splineDistances[key2][0][0] (shortest distance to old key)
                    # But already reserved by lookupAssignmend[splineDistances[key2][0][0]]
                    # --> check lower distance, maybe they belong both to spline?

                    reservedByKey = ''
                    for key in splineMap.keys():
                        if splineMap[key][0] == splineDistances[key2][0][0]:
                            reservedByKey = key
                            break

                    matchedDistance = splineDistances[reservedByKey][0][1]
                    contesterDistance = splineDistances[key2][0][1]
                    alternateDistance = splineDistances[key2][1][1]
                    print 'Matched distance: ' + str(matchedDistance)
                    print 'Contester Distance: ' + str(contesterDistance)
                    print 'Alternate Distance: ' + str(alternateDistance)

                    test = Img0.copy()
                    contestedSpline = prev_dict_splinePoints_updated[splineDistances[key2][0][0]]
                    alreadyReservedBy = dict_splinePoints_updated[reservedByKey]
                    newContestant = dict_splinePoints_updated[key2]

                    test = utils.drawSpline(contestedSpline, (255, 0, 0), test)  # blue: old spline
                    test = utils.drawSpline(alreadyReservedBy, (0, 255, 0), test)  # green already reserved by
                    test = utils.drawSpline(newContestant, (0, 0, 255), test)  # red new contestant

                    altSpline = prev_dict_splinePoints_updated[splineDistances[key2][1][0]]
                    test = utils.drawSpline(altSpline, (255, 255, 0), test)  # teal alternate matching

                    # solve conflict
                    distanceThreshold = 35.0
                    if contesterDistance < matchedDistance:
                        # add matching
                        # print 'contester is better'
                        splineMap[key2] = splineDistances[key2][0]
                        # check if matched is too far away
                        if matchedDistance > distanceThreshold:
                            # unmatch the previously matched key
                            # print 'previous match got unmatched'
                            splineMap[reservedByKey] = (int(reservedByKey) + 100, splineDistances[key2][0][1])
                    elif contesterDistance < distanceThreshold:
                        # print 'previous match better, contester matching too'
                        splineMap[key2] = splineDistances[key2][0]
                    else:
                        # print 'contester is new spline, no match'
                        splineMap[key2] = (int(key2) + 100, splineDistances[key2][0][1])

                    cv2.imwrite(cfg.imgDir + '\\test\\' + ImgName, test)

                    # import pdb;pdb.set_trace()
                splineMap[key2] = splineDistances[key2][0]
                assignedKeys.append(splineDistances[key2][0][0])

        # if '02' in ImgFileList[ImgNumber] or '14' in ImgFileList[ImgNumber] or '15' in ImgFileList[ImgNumber]:
        #    print 'imagename match'
        #    import pdb; pdb.set_trace()

        # reassign the spline-dictionary-keys according to mapping
        new_spline_dict = {}
        for key in splineMap.keys():
            new_spline_dict[splineMap[key][0]] = dict_splinePoints_updated[key]

        # assign this splines to be used in next iteration
        prev_dict_splinePoints_updated = new_spline_dict

    else:
        # initial assignment of the splines
        prev_dict_splinePoints_updated = dict_splinePoints_updated

    '''

    for key2 in splineMapCandidates.keys():
        if key2 not in splineMap.keys():
            import pdb; pdb.set_trace()
            splineMap[key2] = {"key1": key1, 'dist': sum / len(ret)}
        else:
            thisDist = sum / len(ret)
            if splineMap[key2]['dist'] > thisDist:
                splineMap[key2]['key1'] = key1
                splineMap[key2]['dist'] = thisDist
    '''
    # print 'distance between two splines ('+str(key1)+','+str(key2)+') = '+str(sum)

    # for i in range (1, len(xy1)):
    #    cv2.line(SplineImg, (int(round(xy1[i-1][0])),int(round(xy1[i-1][1]))), (int(round(xy1[i][0])),int(round(xy1[i][1]))), (0, 0, 255), 2)

    # if len(splineMap.keys()) > 0:
    #   import pdb;
    #   pdb.set_trace()
    return splineMap