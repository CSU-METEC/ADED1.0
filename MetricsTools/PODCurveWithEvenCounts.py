import pandas as pd
from MetricsTools.Bootstrapping import bootStrapping


def calcPODCurveWEvenCounts(classifiedDF, xCategory, nBins, TPLevels, xunit, xScaleFactor):
    """Build a POD curve (POD vs xCategory) with even data counts in each bin."""
    podDF = pd.DataFrame(columns=['binInterval', 'binAvg', 'binN', 'binTP', 'binFN', 'pod'])

    # filter data first to find only TP & FP results.  Makes bin counts even.
    filt = classifiedDF['tc_Classification'].isin(['TP', 'FN'])

    # calculate bins for even count, determine which bin each data point is in.
    classifiedDF[f'tc_mPOD_{xCategory}Bin_{xunit}'] = None
    xDataField = classifiedDF[xCategory][filt]*xScaleFactor
    bins, bin_edges = pd.qcut(xDataField, q=nBins, precision=0, retbins=True, duplicates='drop')

    classifiedDF.loc[filt, f'tc_mPOD_{xCategory}Bin_{xunit}'] = bins

    BinCounts = classifiedDF[f'tc_mPOD_{xCategory}Bin_{xunit}'].value_counts()

    TPCounts = classifiedDF.loc[classifiedDF['tc_Classification'] == 'TP'][
        f'tc_mPOD_{xCategory}Bin_{xunit}'].value_counts()
    FNCounts = classifiedDF.loc[classifiedDF['tc_Classification'] == 'FN'][
        f'tc_mPOD_{xCategory}Bin_{xunit}'].value_counts()

    filt1 = classifiedDF['tc_Classification'] == 'TP'
    filt2 = filt1 & (classifiedDF['tc_mLocalizationPrecision'] == TPLevels[0])
    filt3 = (classifiedDF['tc_mLocalizationPrecision'] == TPLevels[1]) | (classifiedDF['tc_mLocalizationPrecision'] == TPLevels[0])
    filt4 = filt1 & filt3

    TPCountsUnit = classifiedDF.loc[filt2][f'tc_mPOD_{xCategory}Bin_{xunit}'].value_counts()
    TPCountsGroup = classifiedDF.loc[filt4][f'tc_mPOD_{xCategory}Bin_{xunit}'].value_counts()

    for interval in BinCounts.index:
        try:
            nBin = BinCounts[interval]
            if interval in TPCounts:
                nTP = TPCounts[interval]
            else:
                nTP = 0
            if interval in TPCountsUnit:
                nTPUnit = TPCountsUnit[interval]
            else:
                nTPUnit = 0
            if interval in TPCountsGroup:
                nTPGroup = TPCountsGroup[interval]
            else:
                nTPGroup = 0
            if interval in FNCounts:
                nFN = FNCounts[interval]
            else:
                nFN = 0

            pod = nTP / nBin  # probability of detection
            negError, posError, fitData = bootStrapping(pod, nBin, 1000)
            podUnit = nTPUnit / nBin
            podGroup = nTPGroup / nBin
            negErrorUnit, posErrorUnit, fitDataUnit = bootStrapping(podUnit, nBin, 1000)
            negErrorGroup, posErrorGroup, fitDataGroup = bootStrapping(podGroup, nBin, 1000)

            binAvg = (classifiedDF[f'{xCategory}'].loc[classifiedDF[f'tc_mPOD_{xCategory}Bin_{xunit}'] == interval].mean())*xScaleFactor
            podDF = podDF.append(
                {'binInterval': interval, 'binAvg': binAvg, 'binN': nBin, 'binTP': nTP, 'binFN': nFN, 'pod': pod,
                 'podUnit': podUnit, 'podGroup': podGroup, 'negError': negError, 'posError': posError,
                 'negErrorUnit': negErrorUnit,
                 'posErrorUnit': posErrorUnit, 'negErrorGroup': negErrorGroup, 'posErrorGroup': posErrorGroup,
                 'dataFacility': fitData, 'dataUnit': fitDataUnit, 'dataGroup': fitDataGroup},
                ignore_index=True)

        except Exception as e:
            print(f'Could not calculate the probability of detection due to exception: {e}')
    podDF.sort_values('binAvg', axis=0, inplace=True)
    podDF.reset_index(drop=True, inplace=True)
    return classifiedDF, podDF
