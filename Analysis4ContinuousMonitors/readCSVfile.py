import pandas as pd
import numpy as np
import pathlib
import datetime

def selectfiletype(dataFilePath, opExt):  # Reading files
    if opExt == '.xlsx':
        opDF = pd.read_excel(dataFilePath, engine='openpyxl')
    elif opExt == '.csv':
        opDF = pd.read_csv(dataFilePath)
    elif opExt == '.xls':
        opDF = pd.read_excel(dataFilePath, engine='xlrd')
    else:
        raise Exception(f'Unable to parse file, unhandled extension: {opExt}')
    return opDF

def readFile(pathToControlledReleaseDF, pathToDetectionsReportDF, pathToOutPutHeaderDF, PathToSensorDF, PathToOfflineDF):
    """
    Read csv files
    :param pathToControlledReleaseDF - path to controlled release file
    :param pathToDetectionsReportDF - path to detections report file
    :param pathToOutPutHeaderDF - path to outputheader file
    :param PathToSensorDF - path to sensorDF
    :param PathToOfflineDF - path to offline report
    """

    # Read csv file of controlled releases
    PathToControlledReleaseDF = pathlib.PurePath(pathToControlledReleaseDF)
    ControlledReleaseDF = selectfiletype(PathToControlledReleaseDF, PathToControlledReleaseDF.suffix)
    try:
        print("Converting controlled release string start datetime to datetime")
        ControlledReleaseDF['tc_ExpStartDatetime'] = ControlledReleaseDF["tc_ExpStartDatetime"].apply(lambda time: datetime.datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S%z") if time not in ['nan', 'Nan', 'NULL', 'Null', np.nan, None, str(np.nan)] else time)
        ControlledReleaseDF = ControlledReleaseDF.drop('tc_CRClassification', axis=1)
        if 'tc_CRClassification' in ControlledReleaseDF.columns.tolist():
            ControlledReleaseDF = ControlledReleaseDF.drop('tc_CRClassification', axis=1)
        ControlledReleaseDF['tc_CRClassification'] = None
    except Exception as e:
        print(f"Could not convert controlled release string start datetime to datetime due to {e}")

    try:
        print("Converting controlled release string end datetime to datetime")
        ControlledReleaseDF['tc_ExpEndDatetime'] = ControlledReleaseDF['tc_ExpEndDatetime'].apply(lambda time: datetime.datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S%z") if time not in ['nan', 'Nan', 'NULL', 'Null', np.nan, None, str(np.nan)] else time)
    except Exception as e:
        print(f"Could not convert controlled release string end datetime to datetime due to {e}")

    # Read csv file of detections report
    PathToDetectionReportDF = pathlib.PurePath(pathToDetectionsReportDF)
    DetectionReportDF = selectfiletype(PathToDetectionReportDF, PathToDetectionReportDF.suffix)
    if 'tc_DetClassification' in DetectionReportDF.columns.tolist():
        DetectionReportDF = DetectionReportDF.drop('tc_DetClassification', axis=1)
    DetectionReportDF['tc_DetClassification'] = None

    try:
        print("Converting detection FirstDatetimeSent string datetime to datetime")
        DetectionReportDF['p_FirstDatetimeSent'] = DetectionReportDF["p_FirstDatetimeSent"].apply(lambda time: datetime.datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S%z") if time not in ['nan', 'Nan', 'NULL', 'Null', np.nan, None, str(np.nan)] else time)
    except Exception as e:
        print(f"Could not convert detection FirstDatetimeSent string datetime to datetime due to {e}")

    try:
        print("Converting detection EmissionStartDatetime string datetime to datetime")
        DetectionReportDF['p_EmissionStartDatetime'] = DetectionReportDF["p_EmissionStartDatetime"].apply(lambda time: datetime.datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S%z") if time not in ['nan', 'Nan', 'NULL', 'Null', np.nan, None, str(np.nan)] else time)
    except Exception as e:
        print(f"Could not convert detection EmissionStartDatetime string datetime to datetime due to {e}")

    try:
        print("Converting detection EmissionEndDatetime string datetime to datetime")
        DetectionReportDF['p_EmissionEndDatetime'] = DetectionReportDF["p_EmissionEndDatetime"].apply(lambda time: datetime.datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S%z") if time not in ['nan', 'Nan', 'NULL', 'Null', np.nan, None, str(np.nan)] else time)
    except Exception as e:
        print(f"Could not convert detection EmissionEndDatetime string datetime to datetime due to {e}")

    # Read csv file of offline dataframe
    PathToOfflineDF = pathlib.PurePath(PathToOfflineDF)
    OfflineDF = selectfiletype(PathToOfflineDF, PathToOfflineDF.suffix)
    try:
        print("Converting OFFLINEREPORTDATETIME string datetime to datetime")
        OfflineDF['OFFLINEREPORTDATETIME'] = OfflineDF["OFFLINEREPORTDATETIME"].apply(lambda time: datetime.datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S%z") if time not in ['nan', 'Nan', 'NULL', 'Null', np.nan, None, str(np.nan)] else time)
    except Exception as e:
        print(f"Could not convert OFFLINEREPORTDATETIME string datetime to datetime due to {e}")

    try:
        print("Converting OFFLINEDATETIME string datetime to datetime")
        OfflineDF['OFFLINEDATETIME'] = OfflineDF["OFFLINEDATETIME"].apply(lambda time: datetime.datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S%z") if time not in ['nan', 'Nan', 'NULL', 'Null', np.nan, None, str(np.nan)] else time)
    except Exception as e:
        print(f"Could not convert OFFLINEDATETIME string datetime to datetime due to {e}")

    try:
        print("Converting ONLINEDATETIME string datetime to datetime")
        OfflineDF['ONLINEDATETIME'] = OfflineDF["ONLINEDATETIME"].apply(lambda time: datetime.datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S%z") if time not in ['nan', 'Nan', 'NULL', 'Null', np.nan, None, str(np.nan)] else time)
    except Exception as e:
        print(f"Could not convert ONLINEDATETIME string datetime to datetime due to {e}")

    try:
        print("Converting 'DATETIMESENT' string datetime to datetime")
        OfflineDF['DATETIMESENT'] = OfflineDF["DATETIMESENT"].apply(lambda time: datetime.datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S%z") if time not in ['nan', 'Nan', 'NULL', 'Null', np.nan, None, str(np.nan)] else time)
    except Exception as e:
        print(f"Could not convert DATETIMESENT string datetime to datetime due to {e}")

    try:
        print("Converting 'DATETIMERECEIVE' string datetime to datetime")
        OfflineDF['DATETIMERECEIVE'] = OfflineDF["DATETIMERECEIVE"].apply(lambda time: datetime.datetime.strptime(str(time), "%Y-%m-%d %H:%M:%S%z") if time not in ['nan', 'Nan', 'NULL', 'Null', np.nan, None, str(np.nan)] else time)
    except Exception as e:
        print(f"Could not convert 'DATETIMERECEIVE' string datetime to datetime due to {e}")

    # Read csv file of Output header file
    PathToOutputHeaderDF = pathlib.PurePath(pathToOutPutHeaderDF)
    OutputHeaderDF = selectfiletype(pathlib.Path(PathToOutputHeaderDF), PathToOutputHeaderDF.suffix)

    # Read csv file of sensor dataframe
    PathToSensorDataDF = pathlib.PurePath(PathToSensorDF)
    SensorDataDF = selectfiletype(PathToSensorDataDF, PathToSensorDataDF.suffix)

    return ControlledReleaseDF, DetectionReportDF, OutputHeaderDF, SensorDataDF, OfflineDF