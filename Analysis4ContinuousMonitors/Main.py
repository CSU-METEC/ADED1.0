import pytz
import logging
import pathlib
import datetime
import argparse
from Metrics import calcMetrics
from Analysis4ContinuousMonitors.Logger import Logging
from Analysis4ContinuousMonitors.readCSVfile import readFile
from Analysis4ContinuousMonitors.classification import classify
from FigureBuilderTools.SummaryGraphs import makeSummaryGraphs
from Analysis4ContinuousMonitors.PaperVars import setPaperVars
from Analysis4ContinuousMonitors.PaperVars import setSectionTriggers
from Analysis4ContinuousMonitors.WriteOutputFiles import writeOutputData
from FigureBuilderTools.MetricsFigureBuilder import metricsFigureBuilder
from Analysis4ContinuousMonitors.overWriteExistingFolder import overwrite_folder

# Instantiate the logger with a name: __name__
thisLogger = logging.getLogger(__name__)
thisLogger.setLevel(logging.DEBUG)
# Format the instance of the log created
thisLog = Logging(thisLogger, 'main.log')

def main(args):
    """
    :param args: The code cleans and processes test center data and detection reports for each performer using
    the maintenance record, offline reports, other exclusions unique to a performer
    :Extract METEC's controlled release data
    :Extract maintenance records
    :Extract performer data along with offline records
    :Extract other conditions to clean controlled releases data and performer report data
    :return:
    """

    # Make empty dictionary to store variables referenced in report
    varsDict = {}

    # pull out vars from args for ease of use
    tz = pytz.timezone("US/Mountain")
    startDateTime = tz.localize(datetime.datetime.strptime(args.startDateTime, "%Y%m%d%H%M")).astimezone(pytz.utc)
    endDateTime = tz.localize(datetime.datetime.strptime(args.endDateTime, "%Y%m%d%H%M")).astimezone(pytz.utc)
    outputFilepath = args.outputFilePath
    AnalysisYear = str(startDateTime.year)

    # Assumptions
    if AnalysisYear == "2024":
        # The buffertime before controlled releases is set to zero because experiments with precal were not conducted during this round of testing
        buffertimes = (5, 20)
    else:
        buffertimes = (20, 20)
    bufferTimeBefore = datetime.timedelta(minutes=buffertimes[0])
    bufferTimeAfter = datetime.timedelta(minutes=buffertimes[1])
    buffertimes = (bufferTimeBefore, bufferTimeAfter)
    bounds = (30, 50)  # Limits of points per bin

    # Define the quantiles to be usd
    if args.yQuantile == 'None' or args.yQuantile is None:
        yQuantile = None
    else:
        yQuantile = float(args.yQuantile)

    # setup directories for saving results
    thisLog.debug("Setting up directories for saving results")
    print("Setting up directories for saving results")
    saveDir = pathlib.PurePath(outputFilepath, "ADED_CM_Result")
    overwrite_folder(saveDir)  # overwrite an existing folder
    pathlib.Path(saveDir).mkdir(parents=True, exist_ok=True)
    figuresPath = saveDir / 'figures'
    dataPath = saveDir / 'data'
    pathlib.Path(figuresPath).mkdir(parents=True, exist_ok=True)
    pathlib.Path(dataPath).mkdir(parents=True, exist_ok=True)
    thisLog.info("Successfully setup directories for saving results")

    thisLog.debug("Reading the csv files...")
    print("Reading the csv files...")
    ControlledReleaseDF, DetectionReportDF, OutputHeaderDF, SensorDataDF, OfflineDF = readFile(pathToControlledReleaseDF=args.pathToControlledReleaseDF,
                                                                                               pathToDetectionsReportDF=args.pathToDetectionsReportDF,
                                                                                               pathToOutPutHeaderDF=args.pathToDetectionsReportDF,
                                                                                               PathToSensorDF=args.PathToSensorDF,
                                                                                               PathToOfflineDF=args.pathToDetectionsReportDF)
    thisLog.info("Successfully read the csv files...")

    thisLog.debug("Classifying controlled releases with detection reports...")
    print("Classifying controlled releases with detection reports...")
    classifiedDF, pDetectionsDF, tcControlledReleaseDF, varsDict = classify(tcControlledReleaseDF=ControlledReleaseDF,
                                                                            pDetectionsDF=DetectionReportDF,
                                                                            varsDict=varsDict,
                                                                            buffertimes=buffertimes)
    thisLog.info("Successfully classified controlled releases with detection reports...")

    # compute metrics
    thisLog.debug("Computing metrics for the classified system...")
    print("Computing metrics for the classified system...")
    summaryMetrics, classifiedDF, varsDict = calcMetrics(classifiedDF=classifiedDF, tStart=startDateTime,
                                                         tEnd=endDateTime, offlineDF=OfflineDF, sensorDF=SensorDataDF,
                                                         varsDict=varsDict)
    thisLog.info("Successfully computed metrics for the classified system...")

    # Make summary plots
    thisLog.debug("Making summary graphs...")
    print("Making summary graphs...")
    varsDict, experimentDF = makeSummaryGraphs(classifiedDF, tcControlledReleaseDF, figuresPath, varsDict)
    thisLog.info("Successfully plotted summary graphs...")

    # Make other plots using the metrics figure builder
    thisLog.debug("Make plots using the metrics figure builder...")
    print("Make plots using the metrics figure builder...")
    varsDict = metricsFigureBuilder(classifiedDF=classifiedDF, outputFilePath=figuresPath, varsDict=varsDict,
                                    yQuantile=yQuantile, bounds=bounds)
    thisLog.info("Successfully plotted figures using the metrics figurebuilder...")

    # Add things to varsDict
    thisLog.debug("Setting other variables to the dictionary...")
    print("Adding paper variables to the varsDict...")
    varsDict = setPaperVars(varsDict=varsDict, tc_expSummaryDF=experimentDF, p_detectionsDF=pDetectionsDF,
                            classifiedDF=classifiedDF, tcControlledReleaseDF=tcControlledReleaseDF)
    thisLog.info("Successfully added paper variables to the varsDict")

    thisLog.debug("Setting other variables to the dictionary...")
    print("Setting other variables to the dictionary...")
    dateTimeAndNameDict = {'startDate': startDateTime.strftime("%Y/%m/%d %H:%M"), 'endDate': endDateTime.strftime("%Y/%m/%d %H:%M"),
                           'reportGenerationDate': datetime.datetime.strftime(datetime.datetime.utcnow(), "%Y/%m/%d %H:%M")}
    varsDict.update(dateTimeAndNameDict)
    thisLog.info("Successfully added other variables to the dictionary")

    thisLog.debug("Setting operational factor and sectiontriggers...")
    print("Setting operational factor and sectiontriggers...")
    varsDict['tcOperationalFactor'] = varsDict['tcOpSecs'] / (endDateTime - startDateTime).total_seconds()
    varsDict = setSectionTriggers(classifiedDF, varsDict)
    thisLog.info("Successfully setup operational factors and sectiontriggers")

    # write output files to output dir
    print('Writing output data to csv...')
    thisLog.debug('Writing output data to csv...')
    writeOutputData(saveDir=saveDir, datapath=dataPath, pDetectionReportsDF=pDetectionsDF, columnHeaders=OutputHeaderDF,
                    tcControlledReleaseDF=tcControlledReleaseDF, classifiedDF=classifiedDF, varsDict=varsDict)
    thisLog.info("Successfully wrote output data to csv")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Controlled Release Analysis")
    p.add_argument('-st', '--startDateTime', help="Start date and time for analysis. Format - YYYYMMDDhhmm")
    p.add_argument('-et', '--endDateTime', help="End date and time for analysis. Format - YYYYMMDDhhmm")
    p.add_argument('-cr', '--pathToControlledReleaseDF', help="File path to controlled release data")
    p.add_argument('-dr', '--pathToDetectionsReportDF', help="File path to processed performer data")
    p.add_argument('-dh', '--pathToOutPutHeaderDF', help="File path to output header file")
    p.add_argument('-s', '--PathToSensorDF', help="Filepath to sensor data")
    p.add_argument('-o', '--outputFilePath', help="Filepath to save all output data")
    p.add_argument('-yQ', '--yQuantile', help="Upper percentile for y-axis data")

    args = p.parse_args()
    main(args)