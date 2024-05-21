import numpy as np
import logging
from MetricsTools.BoundingBoxAccuracy import calcBoundingBoxAccuracy
from MetricsTools.DetectionTime import calcDetTime
from MetricsTools.QuantificationMetrics import calcQuantMetrics
from MetricsTools.LocalizationAccuracySingleCoor import calcLocalizationAccuracySinCoor
from MetricsTools.LocalizationPrecisionBoundingBox import calcLocalizationPrecisionBoundingBox
from MetricsTools.LocalizationAccuracyBoundingBox import calcLocalizationAccuracyBox
from Analysis4ContinuousMonitors.Logger import Logging


# Instantiate the logger with a name: __name__
thisLogger = logging.getLogger(__name__)
thisLogger.setLevel(logging.DEBUG)
# Format the instance of the log created
thisLog = Logging(thisLogger, 'MetricsOnTPDetections.log')

def calcMetricsOnTPDetections(classifiedDF):
    """
    For each true positive detection, calculate quantification and localization accuracy and precision metrics
    """
    TPDF = classifiedDF.loc[classifiedDF['tc_Classification'] == 'TP']

    # for each TP detection calculate metrics
    for index, row in TPDF.iterrows():
        # Calculating time to detection
        thisLog.debug("Calculating time to detection for this TP detection")
        classifiedDF.loc[index, 'tc_mDetectionTime'] = calcDetTime(row)
        # Calculating relative/absolute quantification accuracy/precision for this TP detection
        thisLog.debug("Calculating relative/absolute quantification accuracy/precision for this TP detection")
        QAA, QAR, QPA, QPR = calcQuantMetrics(row)
        classifiedDF.loc[index, 'tc_mQuantificationAccuracyAbs'] = QAA
        classifiedDF.loc[index, 'tc_mQuantificationAccuracyRel'] = QAR
        classifiedDF.loc[index, 'tc_mQuantificationPrecisionAbs'] = QPA
        classifiedDF.loc[index, 'tc_mQuantificationPrecisionRel'] = QPR
        # Calculating the localization accuracy - single coordinates
        thisLog.debug("Calculating localization accuracy - single coordinates for this TP detection")
        classifiedDF.loc[index, 'tc_mLocalizationAccuracy_SingleCoord'] = calcLocalizationAccuracySinCoor(row)
        # Calculating the localization accuracy - bounding box
        thisLog.debug("Calculating localization accuracy - bounding box for this TP detection")
        classifiedDF.loc[index, 'tc_mLocalizationAccuracy_BoundingBox'] = calcLocalizationAccuracyBox(row)
        # Calculating the localization precision - bounding box
        thisLog.debug("Calculating localization precision - bounding box for this TP detection")
        classifiedDF.loc[index, 'tc_mLocalizationPrecision_BoundingBox'] = calcLocalizationPrecisionBoundingBox(row)
        # Calculating the localization bounding box
        thisLog.debug("Calculating localization bounding box for this TP detection")
        classifiedDF.loc[index, 'tc_mBoundingBoxAccuracy'] = calcBoundingBoxAccuracy(row)

        # If there are no detections, add secondary metric columns to classifiedDF
        if len(TPDF.index) == 0:
            classifiedDF.loc[index, 'tc_mQuantificationAccuracyAbs'] = np.nan
            classifiedDF.loc[index, 'tc_mQuantificationAccuracyRel'] = np.nan
            classifiedDF.loc[index, 'tc_mQuantificationPrecisionAbs'] = np.nan
            classifiedDF.loc[index, 'tc_mQuantificationPrecisionRel'] = np.nan
            classifiedDF.loc[index, 'tc_mLocalizationAccuracy_SingleCoord'] = np.nan
            classifiedDF.loc[index, 'tc_mLocalizationAccuracy_BoundingBox'] = np.nan
            classifiedDF.loc[index, 'tc_mLocalizationPrecision_BoundingBox'] = np.nan
            classifiedDF.loc[index, 'tc_mBoundingBoxAccuracy'] = np.nan  # True/False
        thisLog.info("Successfully calculated the metrics for this detection")

    return classifiedDF
