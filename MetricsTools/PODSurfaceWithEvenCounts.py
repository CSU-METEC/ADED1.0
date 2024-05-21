import pandas as pd

def calcPODSurfaceWEvenCounts(classifiedDF, xCategory, yCategory, nxBins, nyBins):
    """Build a POD surface, POD = f(xCategory, yCategory) with even data counts in bins across each dimension."""
    podDF = pd.DataFrame()

    # filter data first to find only TP & FP results.  Makes bin counts even.
    filt = classifiedDF['tc_Classification'].isin(['TP', 'FN'])
    # calculate x bins for even count, determine which bin each data point is in.
    classifiedDF[f'tc_mPOD_{xCategory}Bin'] = None
    classifiedDF.loc[filt, f'tc_mPOD_{xCategory}Bin'], xbin_edges = pd.qcut(classifiedDF[xCategory][filt], q=nxBins,
                                                                        precision=0, retbins=True)
    # calculate y bins for even count, determine which bin each data point is in.
    classifiedDF[f'tc_mPOD_{yCategory}Bin'] = None
    classifiedDF[f'tc_mPOD_{yCategory}Bin'][filt], ybin_edges = pd.qcut(classifiedDF[yCategory][filt], q=nyBins,
                                                                        precision=0, retbins=True)

    xBins = classifiedDF[f'tc_mPOD_{xCategory}Bin'].unique()
    yBins = classifiedDF[f'tc_mPOD_{yCategory}Bin'].unique()

    for xInterval in xBins:
        if xInterval is None:
            continue
        xfilt = classifiedDF[f'tc_mPOD_{xCategory}Bin'] == xInterval
        xbinAvg = classifiedDF[f'{xCategory}'].loc[xfilt].mean()

        for yInterval in yBins:
            if yInterval is None:
                continue
            yfilt = classifiedDF[f'tc_mPOD_{yCategory}Bin'] == yInterval
            ybinAvg = classifiedDF[f'{yCategory}'].loc[yfilt].mean()
            dets = classifiedDF.loc[xfilt & yfilt]
            nBin = len(dets)

            nTP = len(dets.loc[dets['tc_Classification'] == 'TP'])

            nFN = len(dets.loc[dets['tc_Classification'] == 'FN'])
            if nBin > 0:
                pod = nTP / nBin  # probability of detection
            else:
                pod = None
            podDF = podDF.append(
                {'xInterval': xInterval, 'yInterval': yInterval, 'xbinAvg': xbinAvg, 'ybinAvg': ybinAvg,
                 'xy_binN': nBin, 'xy_binTP': nTP, 'xy_binFN': nFN, 'xy_pod': pod},
                ignore_index=True)

    podDF.sort_values(by=['xbinAvg', 'ybinAvg'], axis=0, inplace=True)
    podDF.reset_index(drop=True, inplace=True)

    return classifiedDF, podDF