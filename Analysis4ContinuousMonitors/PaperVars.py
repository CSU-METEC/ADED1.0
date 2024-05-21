import numpy as np

def setSectionTriggers(df, varsDict):
    """Set section triggers for tex file. 1 = include section.  0 = exclude section."""

    # Quantification Accuracy (absolute)
    if 'tc_mQuantificationAccuracyAbs' not in list(df.columns):
        varsDict['QuantAccAbsTrigger'] = 0
    elif len(df.loc[df['tc_mQuantificationAccuracyAbs'].isna() == False,'tc_mQuantificationAccuracyAbs']) > 0:
        varsDict['QuantAccAbsTrigger'] = 1
    else:
        varsDict['QuantAccAbsTrigger'] = 0

    # Quantification Accuracy (relative)
    if 'tc_mQuantificationAccuracyRel' not in list(df.columns):
        varsDict['QuantAccAbsTrigger'] = 0
    elif len(df.loc[df['tc_mQuantificationAccuracyRel'].isna() == False,'tc_mQuantificationAccuracyRel']) > 0:
        varsDict['QuantAccRelTrigger'] = 1
    else:
        varsDict['QuantAccRelTrigger'] = 0

    # Quantification Precision (absolute)
    if 'tc_mQuantificationPrecisionAbs' not in list(df.columns):
        varsDict['QuantAccAbsTrigger'] = 0
    elif len(df.loc[df['tc_mQuantificationPrecisionAbs'].isna() == False,'tc_mQuantificationPrecisionAbs']) > 0:
        varsDict['QuantPrecAbsTrigger'] = 1
    else:
        varsDict['QuantPrecAbsTrigger'] = 0

    # Quantification Precision (relative)
    if 'tc_mQuantificationPrecisionRel' not in list(df.columns):
        varsDict['QuantAccAbsTrigger'] = 0
    elif len(df.loc[df['tc_mQuantificationPrecisionRel'].isna() == False, 'tc_mQuantificationPrecisionRel']) > 0:
        varsDict['QuantPrecRelTrigger'] = 1
    else:
        varsDict['QuantPrecRelTrigger'] = 0

    # location accuracy Single coordinate
    if 'tc_mLocalizationAccuracy_SingleCoord' not in list(df.columns):
        varsDict['QuantAccAbsTrigger'] = 0
    elif len(df.loc[df['tc_mLocalizationAccuracy_SingleCoord'].isna() == False, 'tc_mLocalizationAccuracy_SingleCoord']) > 0:
        varsDict['LocAccSCTrigger'] = 1
    else:
        varsDict['LocAccSCTrigger'] = 0

    # localization accuracy bounding box
    if 'tc_mLocalizationAccuracy_BoundingBox' not in list(df.columns):
        varsDict['QuantAccAbsTrigger'] = 0
    elif len(df.loc[df['tc_mLocalizationAccuracy_BoundingBox'].isna() == False, 'tc_mLocalizationAccuracy_BoundingBox']) > 0:
        varsDict['LocAccBBTrigger'] = 1
    else:
        varsDict['LocAccBBTrigger'] = 0

    # bounding box accuracy
    if 'tc_mBoundingBoxAccuracy' not in list(df.columns):
        varsDict['QuantAccAbsTrigger'] = 0
    elif len(df.loc[df['tc_mBoundingBoxAccuracy'].isna() == False, 'tc_mBoundingBoxAccuracy']) > 0:
        varsDict['LocBBAccTrigger'] = 1
    else:
        varsDict['LocBBAccTrigger'] = 0

    # localization precision (bounding box)
    if 'tc_mLocalizationPrecision_BoundingBox' not in list(df.columns):
        varsDict['QuantAccAbsTrigger'] = 0
    elif len(df.loc[df['tc_mLocalizationPrecision_BoundingBox'].isna() == False, 'tc_mLocalizationPrecision_BoundingBox']) > 0:
        varsDict['LocPrecBBTrigger'] = 1
    else:
        varsDict['LocPrecBBTrigger'] = 0

    # Detection time
    TPDF = df.loc[df['tc_Classification'] == "TP"]
    if 'tc_mDetectionTime' not in list(df.columns):
        varsDict['DetectionTimeTrigger'] = 0
    elif TPDF['tc_mDetectionTime'].isnull().all():
        varsDict['DetectionTimeTrigger'] = 0
    else:
        varsDict['DetectionTimeTrigger'] = 1

    return varsDict

def my_function(row, col1, col2):
    return row[col2] - row[col1]

def setPaperVars(varsDict, tc_expSummaryDF, p_detectionsDF, classifiedDF, tcControlledReleaseDF):
    """Set variables used in paper"""
    # Count of experiments
    varsDict['nExperiments'] = len(tc_expSummaryDF)
    varsDict['nExperimentsRemovedFromAnalysis'] = None
    varsDict['nExperimentsIncludedInAnalysis'] = None

    # Calculate operational factor for test center
    totalTime = float(tc_expSummaryDF['tc_ExpDurationHrs'].sum()) * 3600
    varsDict['tcOpSecs'] = totalTime

    # Calculate the number of controlled releases
    varsDict['nControlledReleases'] = len(tcControlledReleaseDF)
    varsDict['nControlledReleasesIncludedInAnalysis'] = None
    varsDict['nControlledReleasesExcludedFromAnalysis'] = None

    # Count TruePositives (total and per level), FalsePositives, FalseNegatives
    varsDict['nTruePositives'] = int(len(classifiedDF.loc[classifiedDF['tc_Classification'] == 'TP']))
    varsDict['nTPEquipUnit'] = int(len(classifiedDF.loc[classifiedDF['tc_mLocalizationPrecision'] == 'CorrectUnit']))
    varsDict['nTPEquipGroup'] = int(len(classifiedDF.loc[classifiedDF['tc_mLocalizationPrecision'] == 'CorrectGroup']))
    varsDict['nTPFacility'] = int(len(classifiedDF.loc[classifiedDF['tc_mLocalizationPrecision'] == 'CorrectFacility']))
    varsDict['nFalsePositives'] = int(len(classifiedDF.loc[classifiedDF['tc_Classification'] == 'FP']))
    varsDict['nFalseNegatives'] = int(len(classifiedDF.loc[classifiedDF['tc_Classification'] == 'FN']))

    # Get nDetectionReports and nOfflineReports
    varsDict['nDetectionReports'] = None  # number of reports sent
    varsDict['nDetections'] = len(p_detectionsDF)  # number of unique emissionSourceIDs in reports
    varsDict['nOfflineReports'] = None

    # count detections included and excluded from analysis
    DETfilt = (classifiedDF['tc_Classification'] == 'TP') | (classifiedDF['tc_Classification'] == 'FP')
    varsDict['nDetectionsIncludedInAnalysis'] = len(classifiedDF.loc[DETfilt])
    varsDict['nDetectionsExcludedFromAnalysis'] = None

    return varsDict

