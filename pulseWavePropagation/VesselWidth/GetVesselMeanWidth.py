import numpy as np


def getVesselMeanWidth(dict_vesselWidth):
    ##first, get the mean vessel width for each segments
    dict_meanVesselWidth = {}
    for vesselkey in dict_vesselWidth.keys():
        segVesselWidths =  dict_vesselWidth[vesselkey]
        meanwidth0 = np.mean(segVesselWidths)
        std0 = np.std(segVesselWidths)
        filteredWidths = segVesselWidths[np.abs(segVesselWidths - meanwidth0) <= np.minimum(2*std0, 2.5)]
        if len(filteredWidths) > 30:  #if the width is less than 30 pixels, then remove it.
            meanWidth = np.mean(filteredWidths)
        # else:
        #     meanWidth = meanwidth0

            dict_meanVesselWidth[vesselkey] = meanWidth #[meanWidth, meanwidth0, std0]


    return dict_meanVesselWidth