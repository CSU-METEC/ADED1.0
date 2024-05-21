from MetricsTools.MinMaxMeanFinder import findMinMaxMean
import datetime


def calcMetricsDict(classifiedDF, summaryDF):
    metricsDict = {}
    count = len(classifiedDF)
    try:
        # Get pOperationalFactor from summaryDF
        # Todo: Come back to this
        metricsDict['pOperationalFactor'] = float(summaryDF['tc_mOperationalFactor'][0])
    except Exception as e:
        print(f'Could not calculate pOperationalFactor from classifiedDF due to exception: {e}')

    try:
        # Get False Positive, and False Negative fractions from summaryDF
        metricsDict['tcmFalsePositiveFraction'] = float(summaryDF['tc_mFalsePositiveFraction'])
        metricsDict['tcmFalseNegativeFraction'] = float(summaryDF['tc_mFalseNegativeFraction'])
    except Exception as e:
        print(f'Could not calculate tcFalsePositiveFraction from classifiedDF due to exception: {e}')

    try:
        # Get the localization for unit, group, and facility levels
        metricsDict['tcmLocalizationAccuracyUnit'] = float(summaryDF['tc_mLocalizationAccuracyUnit'])
        metricsDict['tcmLocalizationAccuracyGroup'] = float(summaryDF['tc_mLocalizationAccuracyGroup'])
        metricsDict['tcmLocalizationAccuracyFacility'] = float(summaryDF['tc_mLocalizationAccuracyFacility'])
    except Exception as e:
        print(f'Could not calculate tcmLocalizationAccuracy from classifiedDF due to exception: {e}')

    try:
        metricsDict['tcmFalsePositiveUnit'] = 1.0 - float(summaryDF['tc_mLocalizationAccuracyUnit'])
        metricsDict['tcmFalsePositiveGroup'] = 1.0 - float(summaryDF['tc_mLocalizationAccuracyGroup'])
        metricsDict['tcmFalsePositiveFacility'] = 1.0 - float(summaryDF['tc_mLocalizationAccuracyFacility'])
    except Exception as e:
        print(f'Could not calculate tcmLocalizationAccuracy from classifiedDF due to exception: {e}')

    # Get localization accuracy single coordinate min, max, and mean from summaryDF
    metricsDict = findMinMaxMean(classifiedDF, 'tc_mLocalizationAccuracy_SingleCoord', metricsDict, count)

    # Get localization accuracy single coordinate min, max, and mean from summaryDF
    metricsDict = findMinMaxMean(classifiedDF, 'tc_mLocalizationAccuracy_BoundingBox', metricsDict, count)

    # Get localization precision bounding box min, max, and mean from summaryDF
    metricsDict = findMinMaxMean(classifiedDF, 'tc_mLocalizationPrecision_BoundingBox', metricsDict, count)

    # Get quantification accuracy abs. min, max, and mean from summaryDF
    metricsDict = findMinMaxMean(classifiedDF, 'tc_mQuantificationAccuracyAbs', metricsDict, count)

    # Get quantification accuracy rel. min, max, and mean from classifiedDF
    metricsDict = findMinMaxMean(classifiedDF, 'tc_mQuantificationAccuracyRel', metricsDict, count)

    # Get quantification precision abs. min, max, and mean from summaryDF
    metricsDict = findMinMaxMean(classifiedDF, 'tc_mQuantificationPrecisionAbs', metricsDict, count)

    # Get quantification precision rel. min, max, and mean from summaryDF
    metricsDict = findMinMaxMean(classifiedDF, 'tc_mQuantificationPrecisionRel', metricsDict, count)

    # Get tcEmissionRateMinGHRCH4 and tcEmissionRateMaxGHRCH4 from summaryDF
    metricsDict = findMinMaxMean(classifiedDF, 'tc_C1MassFlow', metricsDict, count)

    # Get met temperature min, max, and mean from summaryDF
    metricsDict = findMinMaxMean(classifiedDF, 'tc_ExpTAtmAvg', metricsDict, count)

    # Get met wind speed min, max, and mean from summaryDF
    metricsDict = findMinMaxMean(classifiedDF, 'tc_ExpWindSpeedAvg', metricsDict, count)

    # Get met win direction min, max, and mean from summaryDF
    metricsDict = findMinMaxMean(classifiedDF, 'tc_ExpWindDirAvg', metricsDict, count)

    # Get EPBFU min, max, mean from summaryDF
    metricsDict = findMinMaxMean(classifiedDF, 'tc_EPBFU', metricsDict, count)

    # Get the tc_mDetectionTime min, max, mean from classifiedDF
    # (datetime.datetime.min + startTime).time()
    # try:
    #     if len(classifiedDF.loc[classifiedDF['tc_mDetectionTime'].isnull()]) != count:
    #         metricsDict[f'tc_mDetectionTimeMin'] = (classifiedDF['tc_mDetectionTime'].min()).total_seconds()/3600
    #         metricsDict[f'tc_mDetectionTimeMax'] = (classifiedDF['tc_mDetectionTime'].max()).total_seconds()/3600
    #         metricsDict[f'tc_mDetectionTimeMean'] = (classifiedDF['tc_mDetectionTime'].mean()).total_seconds()/3600
    #     else:
    #         metricsDict[f'tc_mDetectionTimeMin'] = 0
    #         metricsDict[f'tc_mDetectionTimeMax'] = 0
    #         metricsDict[f'tc_mDetectionTimeMean'] = 0
    # except Exception as e:
    #     print(f'Could not find min, max, and or mean detection time due to exception: {e}')

    try:
        # Get the count of quantification accuracy abs that were above and below zero
        filt = classifiedDF['tc_mQuantificationAccuracyAbs'] > 0
        filt1 = classifiedDF['tc_mQuantificationAccuracyAbs'] < 0

        metricsDict['pQuantificationAccuracyAbsPlus'] = classifiedDF[filt].shape[0]
        metricsDict['pQuantificationAccuracyAbsNeg'] = classifiedDF[filt1].shape[0]
    except Exception as e:
        print(f'Could not calculate QuantificationAccuracyAbs from classifiedDF due to exception: {e}')

    try:
        # Get bounding box accuracy percentages for true and false
        if classifiedDF['tc_mBoundingBoxAccuracy'].isnull().sum() == len(classifiedDF.index):
            metricsDict['BoundingBoxAccuracyPercentTrue'] = 0
            metricsDict['BoundingBoxAccuracyPercentFalse'] = 0
            metricsDict['BoundingBoxAccuracyTrue'] = 0
            metricsDict['BoundingBoxAccuracyFalse'] = 0
        else:
            # Get Bounding Box Accuracy count for true and false from summaryDF
            trueCount = len(classifiedDF[classifiedDF['tc_mBoundingBoxAccuracy'] == True])
            falseCount = len(classifiedDF[classifiedDF['tc_mBoundingBoxAccuracy'] == False])
            metricsDict['BoundingBoxAccuracyTrue'] = trueCount
            metricsDict['BoundingBoxAccuracyFalse'] = falseCount
            metricsDict['BoundingBoxAccuracyPercentTrue'] = 100 * (trueCount / (trueCount + falseCount))
            metricsDict['BoundingBoxAccuracyPercentFalse'] = 100 * (falseCount / (trueCount + falseCount))
    except Exception as e:
        print(f'Could not calculate BoundingBoxAccuracy from classifiedDF due to exception: {e}')

    return metricsDict
