import numpy as np

def seperateZoneAB(discRegionParameter, dict_segmentPixelLocs, dict_splinePoints_updated, dict_vesselWidth_updated ):

    lengthLimit = 15
    discCenter = discRegionParameter['discCenter']
    discRadius = discRegionParameter['discRadius']

    dict_vesselWidth_updated_ZoneC = {}
    dict_vesselWidth_updated_ZoneB = {}
    dict_segmentPixelLocs_ZoneC = {}
    dict_segmentPixelLocs_ZoneB = {}
    for vesselkey in dict_segmentPixelLocs.keys():
        tempPixLocs = dict_segmentPixelLocs[vesselkey]
        tempDist = np.hypot(tempPixLocs[:,0] - discCenter[0], tempPixLocs[:,1] - discCenter[1])
        tempZoneCTracker = np.bitwise_and(tempDist >= 3*discRadius, tempDist <= 5*discRadius)
        tempZoneBTracker  = np.bitwise_and(tempDist > 2*discRadius, tempDist <= 3*discRadius)

        if np.sum(tempZoneCTracker) >= lengthLimit:
            dict_segmentPixelLocs_ZoneC[vesselkey] = dict_segmentPixelLocs[vesselkey][tempZoneCTracker, :]
        if np.sum(tempZoneBTracker) >= lengthLimit:
            dict_segmentPixelLocs_ZoneB[vesselkey] = dict_segmentPixelLocs[vesselkey][tempZoneBTracker, :]

        if vesselkey in dict_vesselWidth_updated.keys():
            tempSplineLength = len(dict_splinePoints_updated[vesselkey][0])
            firstIndex = (len(tempDist) - tempSplineLength) // 2
            tempZoneCTrackerCut = tempZoneCTracker[firstIndex:firstIndex + tempSplineLength]
            tempZoneBTrackerCut = tempZoneBTracker[firstIndex:firstIndex + tempSplineLength]

            if np.sum(tempZoneCTrackerCut) >= lengthLimit:
                dict_vesselWidth_updated_ZoneC[vesselkey] = dict_vesselWidth_updated[vesselkey][tempZoneCTrackerCut]

            if np.sum(tempZoneBTrackerCut) >= lengthLimit:
                dict_vesselWidth_updated_ZoneB[vesselkey] = dict_vesselWidth_updated[vesselkey][tempZoneBTrackerCut]

    return dict_vesselWidth_updated_ZoneB, dict_vesselWidth_updated_ZoneC, \
           dict_segmentPixelLocs_ZoneB, dict_segmentPixelLocs_ZoneC
