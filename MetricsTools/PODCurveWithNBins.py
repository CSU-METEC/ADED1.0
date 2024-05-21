import pandas as pd
import numpy as np
from MetricsTools.Binner import bin2
from MetricsTools.Bootstrapping import bootStrapping


def calcPODCurveWNBins(combineDF, xCategory, nBins, TPLevels):
    """
    :param combineDF: Post classification dataframe
    :param xCategory: Category for calculating probability of detection
    :param nBins: Number of bins for calculating probability of detection
    :return: Dataframe with probability of detection calculated for xCategory

    ColumnNames: DET Classification: Classification of report detection
    """
    returnDF = pd.DataFrame(
        columns=['binInterval', 'binAvg', 'binTP', 'binFN', 'pod', 'binN'])
    try:
        # Bin values
        Bins = bin2(combineDF, xCategory, nBins)

        # Calculate POD for each Bin
        for interval in Bins:
            Min = interval.left.item()
            Max = interval.right.item()

            # Find the number of TP in interval
            filt = (combineDF[xCategory] > Min) & (combineDF[xCategory] <= Max)
            filt1 = filt & (combineDF['tc_Classification'] == 'TP')
            filt2 = filt & (combineDF['tc_Classification'] == 'FN')
            filt3 = filt & filt1 & (combineDF['tc_mLocalizationPrecision'] == TPLevels[0])
            filt4 = (combineDF['tc_mLocalizationPrecision'] == TPLevels[1]) | (combineDF['tc_mLocalizationPrecision'] == TPLevels[2])
            filt5 = filt & filt1 & filt4

            rows = combineDF.loc[filt]
            nTP = len(combineDF.loc[filt1])
            nTPUnit = len(combineDF.loc[filt3])
            nTPGroup = len(combineDF.loc[filt5])
            nFN = len(combineDF.loc[filt2])

            count = (nTP + nFN)
            countUnit = (nTPUnit + nFN)
            countGroup = (nTPGroup + nTPUnit + nFN)
            avg = rows[f'{xCategory}'].mean()

            # Calculate POD
            if count > 0:
                POD = nTP / count
                PODUnit = nTPUnit / countUnit
                PODGroup = (nTPGroup + nTPUnit) / countGroup
                negError, posError, fitData = bootStrapping(POD, count, 1000)
                negErrorUnit, posErrorUnit, fitDataUnit = bootStrapping(POD, count, 1000)
                negErrorGroup, posErrorGroup, fitDataGroup = bootStrapping(POD, count, 1000)

                # Append to returnDf
                returnDF = returnDF.append(
                    {'binInterval': interval, 'binAvg': avg, 'binTP': nTP, 'binFN': nFN, 'pod': POD, 'binN': count,
                     'podUnit': PODUnit,
                     'podGroup': PODGroup, 'negError': negError, 'posError': posError, 'negErrorUnit': negErrorUnit,
                     'posErrorUnit': posErrorUnit, 'negErrorGroup': negErrorGroup, 'posErrorGroup': posErrorGroup,
                     'dataFacility': fitData, 'dataUnit': fitDataUnit, 'dataGroup': fitDataGroup},
                    ignore_index=True)
            else:
                POD = 0
                PODUnit = 0
                PODGroup = 0
                negError, posError, fitData = bootStrapping(POD, count, 1000)
                negErrorUnit, posErrorUnit, fitDataUnit = bootStrapping(POD, count, 1000)
                negErrorGroup, posErrorGroup, fitDataGroup = bootStrapping(POD, count, 1000)

                # Append to returnDf
                returnDF = returnDF.append(
                    {'binInterval': interval, 'binAvg': avg, 'binTP': nTP, 'binFN': nFN, 'pod': POD, 'binN': count,
                     'podUnit': PODUnit,
                     'podGroup': PODGroup, 'negError': negError, 'posError': posError, 'negErrorUnit': negErrorUnit,
                     'posErrorUnit': posErrorUnit, 'negErrorGroup': negErrorGroup, 'posErrorGroup': posErrorGroup,
                     'dataFacility': fitData, 'dataUnit': fitDataUnit, 'dataGroup': fitDataGroup},
                    ignore_index=True)

    except Exception as e:
        print(f'Could not calculate probability of detection due to exception: {e}')

    return combineDF, returnDF
