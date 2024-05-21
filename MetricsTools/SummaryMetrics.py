import pandas as pd
from MetricsTools.FalsePositiveFraction import calcFalsePositiveFrac
from MetricsTools.FalseNegativeFraction import calcFalseNegativeFrac
from MetricsTools.LocalizationAccuracy import calcLocalizationAccuracy
from MetricsTools.OperationalFactor import calcOperationalFactor


def calcSummaryMetrics(classifiedDF, tStart, tEnd, offlineDF, TPLevels):
    """
    Calculates the metrics for summary file
    :param classifiedDF - a file of classified - controlled release and detection reports
    :param tStart - start datetime
    :param tEnd - end datetime
    :param offlineDF - the offline dataframe
    :param TPLevels - The TP levels
    :return summaryMetrics, a dataframe of calculated metrics
    """
    # Primary Metrics
    # Calculate False Positive Fraction
    FPF = calcFalsePositiveFrac(classifiedDF)

    # Calculate False Negative Fraction
    FNF = calcFalseNegativeFrac(classifiedDF)

    # Calculate Localization Accuracy
    LAU, LAG, LAF = calcLocalizationAccuracy(classifiedDF, TPLevels)

    # Calculate Operational Factor
    OF = calcOperationalFactor(tStart, tEnd, offlineDF)

    summaryMetrics = pd.DataFrame(data={'tc_mFalsePositiveFraction': [FPF],
                                        'tc_mFalseNegativeFraction': [FNF],
                                        'tc_mLocalizationAccuracyUnit': [LAU],
                                        'tc_mLocalizationAccuracyGroup': [LAG],
                                        'tc_mLocalizationAccuracyFacility': [LAF],
                                        'tc_mOperationalFactor': [OF]})

    return summaryMetrics
