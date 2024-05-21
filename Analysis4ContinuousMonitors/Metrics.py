import logging
import pandas as pd
from MetricsTools.MetricsDictionaryHandler import calcMetricsDict
from MetricsTools.MetricsOnTPDetections import calcMetricsOnTPDetections
from MetricsTools.SummaryMetrics import calcSummaryMetrics
from MetricsTools.distanceToSensor import calcDistanceToSensor
from Analysis4ContinuousMonitors.Logger import Logging

# Instantiate the logger with a name: __name__
thisLogger = logging.getLogger(__name__)
thisLogger.setLevel(logging.DEBUG)
# Format the instance of the log created
thisLog = Logging(thisLogger, 'Metrics.log')


def calcMetrics(classifiedDF, tStart, tEnd, offlineDF, sensorDF, varsDict):

    """
    Calculate metrics for test program.  Note, probability of detection curves are handled elsewhere.
    :param classifiedDF extracted through the classification scheme
    :param tStart Start datetime of experiment
    :param tEnd End datetime of experiment
    :param offlineDF Performer offline reports as a dataframe
    :param sensorDF Performer sensor information from performerlookup.json
    :param varsDict Dictionary of variables referenced in report
    :return primaryMetricsDF: Dataframe of primary metrics,
             combineDF: Updated with primary metrics,
             ProbabilityOfDetectionCurveDF: Dataframe with probability of detection calculated for xCategory
             ProbabilityOfDetectionSurfaceDF: Dataframe with probability of detection calculated for xCategory and
             yCategory
    """
    print("Calculating Metrics...")
    summaryMetrics = pd.DataFrame()

    TPLevels = ['CorrectUnit', 'CorrectGroup', 'CorrectFacility']

    try:
        # calculate metrics on each TP
        thisLog.debug("Calculating metrics for TP detections...")
        classifiedDF = calcMetricsOnTPDetections(classifiedDF)
        # Calculate the distance to the nearest sensor for each controlled release
        thisLog.debug("Calculating the closest distance between a sensor and a controlled release...")
        classifiedDF = calcDistanceToSensor(classifiedDF, sensorDF)
        # calculate summary metrics
        thisLog.debug("Calculating summary metrics...")
        summaryMetrics = calcSummaryMetrics(classifiedDF, tStart, tEnd, offlineDF, TPLevels)
        # Calculate vars for report
        thisLog.debug("Calculating variables for report...")
        metricsDict = calcMetricsDict(classifiedDF, summaryMetrics)
        varsDict.update(metricsDict)

    except Exception as e:
        print(f'Could not calculate primary metrics due to exception: {e}')
        thisLog.error(f'Could not calculate primary metrics due to exception: {e}')

    return summaryMetrics, classifiedDF, varsDict
