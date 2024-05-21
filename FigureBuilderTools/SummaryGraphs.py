import os
import math
import logging
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  ## Import MaxNLocator
from Analysis4ContinuousMonitors.Logger import Logging

# Instantiate the logger with a name: __name__
thisLogger = logging.getLogger(__name__)
thisLogger.setLevel(logging.DEBUG)
# Format the instance of the log created
thisLog = Logging(thisLogger, 'Summarygraphs.log')


def strTOtimedelta(timeStr):
    """
    Converts calculated time-to-detection which is in a string format to datetime as it appears in the classified data
    file
    :param timeStr: time-to-detection in string format
    :returns total_seconds equivalent of time-to-detection
    """
    # replaces " days " with "-"
    time = timeStr.replace(" days ", "-")
    DaytimeSplit = time.split('-')
    # Gets integer of days from the split above
    day = int(DaytimeSplit[0])
    # Splits datetimes into hours, minutes, and seconds
    timeSplit = DaytimeSplit[1].split(':')
    hours = int(timeSplit[0])
    minutes = int(timeSplit[1])
    seconds = int(timeSplit[2])

    # Create a timedelta object representing a duration of time
    duration = dt.timedelta(days=day, hours=hours, minutes=minutes, seconds=seconds)

    # Calculate the total number of seconds in the timedelta object
    total_seconds = duration.total_seconds()
    return total_seconds

def roundup(x, n):
    """
    Code to roundup x to n
    """
    return int(math.ceil(x / float(n))) * n

def makeSummaryGraphs(classifiedDF, tcControlledReleaseDF, outputFolder, varsDict):
    """
    Code generates simple plots which consists of:
        1. A histogram of number of emission points vs emission rate
        2. A histogram of number of emission points vs equipment group
        3. A histogram of number of experiments vs experiment duration
        4. A histogram of number of experiments vs wind-direction
        5. A histogram of number of experiments vs windspeed
        6. A histogram of number of experiments vs Temperature
        7. A histogram of number of experiments vs start time
        8. A histogram of FPs nearest time to a controlled release
        9. A histogram of Time to detection.
    """

    # setting plot parameters
    plt.rcParams.update({'font.size': 7})
    figsize = (3.54331, 3.54331 / 1.5)
    alpha = 0.3

    # Extract false positive detections
    FPDF = classifiedDF[classifiedDF['tc_Classification'] == "FP"]
    # Extract True positive detections
    TPDF = classifiedDF.loc[classifiedDF['tc_Classification'] == "TP"]
    # Extract all controlled releases
    CRDF = classifiedDF.loc[(classifiedDF['tc_Classification'] == "TP") | (classifiedDF['tc_Classification'] == "FN")]
    # Extract only TP detections with calculated detection time
    tpdf = TPDF.dropna(axis=0, how='any', subset=["tc_mDetectionTime"], inplace=False).copy(deep=True)
    # Convert time to detection into strings
    tpdf["tc_mDetectionTime"] = tpdf["tc_mDetectionTime"].apply(lambda ttd: str(ttd))
    # Extract only positive time to detection
    tpdf = tpdf.loc[tpdf["tc_mDetectionTime"].apply(lambda ttd: int(ttd[0:2]) >= 0)]
    # Convert calculated time to detection to hours"
    tpdf["tc_mDetectionTime"] = tpdf["tc_mDetectionTime"].apply(lambda ttd: strTOtimedelta(str(ttd))/3600)

    # Extract experiments from the classified data file
    experimentDF = tcControlledReleaseDF.assign(count=lambda x: x.groupby(['tc_ExperimentID'])['tc_ExperimentID'].transform("count"))
    experimentDF = experimentDF.drop_duplicates(subset='tc_ExperimentID', keep="first")
    experimentDF = experimentDF.reset_index(drop=True)
    counts = experimentDF["count"].replace(np.nan, 'None')
    labels, counts = np.unique(counts, return_counts=True)
    varsDict['tcmMinEPCounts'] = min(list(labels))
    varsDict['tcmMaxEPCounts'] = max(list(labels))
    varsDict['tcmMeanEPCounts'] = experimentDF["count"].mean()
    varsDict['tcmMeanEPCounts'] = experimentDF["count"].mean()
    varsDict['tcEmissionRateMinLPM'] = float(CRDF['tc_EPBFE'].min())
    varsDict['tcEmissionRateMaxLPM'] = float(CRDF['tc_EPBFE'].max())

    try:
        print("Plotting a histogram of number of emission points vs emission rate..")
        thisLog.debug('Plotting a histogram of number of emission points vs emission rate..')
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel("Emission rate (slpm Whole Gas)")
        ax1.set_ylabel("Number of sources")
        bins = list(range(0, roundup(CRDF['tc_EPBFE'].max(), 20) + 1, 20))
        ax1.hist(list(CRDF['tc_EPBFE']), bins=bins, edgecolor='black')
        plt.xticks(bins, rotation=45)
        plt.xlim(bins[0], bins[-1])
        fig.tight_layout()
        plt.grid(alpha=alpha)
        path = os.path.join(outputFolder, "Hist_EmissionRate_slpm.png")
        plt.savefig(path,  dpi=400)
        print("saving: " + path)
        plt.close()  # close current figure
        thisLog.info('Successfully plotted a histogram of number of emission points vs emission rate..')
    except Exception as e:
        thisLog.error(f'Could not plot a histogram of number of emission points vs emission rate due to exception: {e}')
        print(f'Could not plot a histogram of number of emission points vs emission rate due to exception: {e}')

    try:
        print("Plotting a histogram of number of emission points vs equipment group..")
        thisLog.debug('Plotting a histogram of number of emission points vs equipment group..')
        # equipment group histogram
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel("Equipment group")
        ax1.set_ylabel("Number of sources")
        groupIDData = CRDF['tc_EquipmentGroupID'].replace(np.nan, 'None')
        labels, counts = np.unique(groupIDData, return_counts=True)
        ticks = range(len(counts))
        ax1.bar(ticks, counts, align='center', edgecolor='black')
        plt.xticks(ticks, labels, rotation=-90)
        fig.tight_layout()
        plt.grid(axis='y', alpha=alpha)
        path = os.path.join(outputFolder, "Hist_EquipmentGroup.png")
        plt.savefig(path, dpi=400)
        print("saving: " + path)
        plt.close()  # close current figure
        thisLog.info('Successfully plotted a histogram of number of emission points vs equipment group..')
    except Exception as e:
        thisLog.error(f'Could not plot a histogram of number of emission points vs equipment group due to exception: {e}')
        print(f'Could not plot a histogram of number of emission points vs equipment group due to exception: {e}')

    try:
        print("Plotting a histogram of number of experiments vs experiment duration.")
        thisLog.debug('Plotting a histogram of number of experiments vs experiment duration.')
        # hist of durations
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel("Experiment duration (hours)")
        ax1.set_ylabel("Number of experiments")
        expDF = CRDF.drop_duplicates(['tc_ExperimentID'])
        durs = expDF['tc_ExpDurationHrs'].tolist()
        bins = list(range(int(min(durs)), int(max(durs)) + 2))
        ax1.hist(durs, bins=bins, edgecolor='black')
        plt.xticks(bins)
        plt.xlim(bins[0], bins[-1])
        fig.tight_layout()
        plt.grid(alpha=alpha)
        path = os.path.join(outputFolder, "Hist_ExperimentDuration.png")
        plt.savefig(path, dpi=400)
        print('saving: ' + path)
        plt.close()  # close current figure
        thisLog.info('Successfully plotted a histogram of number of experiments vs experiment duration')
    except Exception as e:
        thisLog.error(f'Could not plot a histogram of number of experiments vs experiment duration due to exception: {e}')
        print(f'Could not plot a histogram of number of experiments vs experiment duration due to exception: {e}')

    try:
        print("Plotting a histogram of number of experiments vs wind direction.")
        thisLog.debug('Plotting a histogram of number of experiments vs experiment duration.')
        # wind dir per experiment
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel("Wind direction (Degrees CW from North)")
        ax1.set_ylabel("Number of experiments")
        binRange = 30  # degrees
        ax1.hist(experimentDF['tc_ExpWindDirAvg'], bins=range(0, 360 + binRange, binRange), edgecolor='black')
        ticks = [0, 90, 180, 270, 360]
        plt.xticks(ticks)
        fig.tight_layout()
        plt.grid(alpha=alpha)
        path = os.path.join(outputFolder, "Hist_WindDirection.png")
        plt.savefig(path, dpi=400)
        print("saving: " + path)
        plt.close()  # close current figure
        thisLog.info('Successfully plotted a histogram of number of experiments vs wind direction.')
    except Exception as e:
        thisLog.error(f'Could not plot a histogram of number of experiments vs wind direction due to exception: {e}')
        print(f'Could not plot a histogram of number of experiments vs wind direction due to exception: {e}')

    try:
        print("Plotting a histogram of number of experiments vs average windspeed.")
        thisLog.debug('Plotting a histogram of number of experiments vs average windspeed.')
        # wind speed per experiment
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel("Average wind speed (m/s)")
        ax1.set_ylabel("Number of experiments")
        bins = list(range(0, int(experimentDF['tc_ExpWindSpeedAvg'].max() + 2)))
        ax1.hist(experimentDF['tc_ExpWindSpeedAvg'], bins=bins, edgecolor='black')
        fig.tight_layout()
        plt.grid(alpha=alpha)
        path = os.path.join(outputFolder, "Hist_WindSpeed.png")
        plt.savefig(path, dpi=400)
        print("saving: " + path)
        plt.close()  # close current figure
        thisLog.info('Successfully plotted histogram of number of experiments vs average windspeed..')
    except Exception as e:
        thisLog.error(f'Could not plot a histogram of number of experiments vs average windspeed due to exception: {e}')
        print(f'Could not plot histogram of number of experiments vs average windspeed due to exception: {e}')

    try:
        print("Plotting a histogram of number of experiments vs average temperature.")
        thisLog.debug('Plotting a histogram of number of experiments vs average temperature.')
        # temperature per experiment
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel("Average temperature (Celsius)")
        ax1.set_ylabel("Number of experiments")
        ax1.hist(experimentDF['tc_ExpTAtmAvg'], edgecolor='black')
        fig.tight_layout()
        plt.grid(alpha=alpha)
        path = os.path.join(outputFolder, "Hist_Temperature.png")
        plt.savefig(path, dpi=400)
        print("saving: " + path)
        plt.close()  # close current figure
        thisLog.info('Successfully plotted histogram of number of experiments vs average temperature..')
    except Exception as e:
        thisLog.error(f'Could not plot a histogram of number of experiments vs average temperature due to exception: {e}')
        print(f'Could not plot histogram of number of experiments vs average temperature due to exception: {e}')

    try:
        print("Plotting a histogram of number of experiments vs start time (Hour UTC)")
        thisLog.debug('Plotting a histogram of number of experiments vs start time (Hour UTC)')
        # time of day experiment started
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel("Start time (Hour UTC)")
        ax1.set_ylabel("Number of experiments")
        startHours = [ts.hour for ts in experimentDF["tc_ExpStartDatetime"]]
        ticks = [0, 6, 12, 18, 24]
        y_ax = ax1.axes.get_yaxis()  ## Get X axis
        y_ax.set_major_locator(MaxNLocator(integer=True))  ## Set major locators to integer values
        ax1.hist(startHours, bins=24, edgecolor='black')
        plt.xticks(ticks)
        fig.tight_layout()
        plt.grid(alpha=alpha)
        path = os.path.join(outputFolder, "Hist_StartTime.png")
        plt.savefig(path, dpi=400)
        print("saving: " + path)
        plt.close()  # close current figure
        thisLog.info('Successfully plotted a histogram of number of experiments vs start time (Hour UTC)')
    except Exception as e:
        thisLog.error(f'Could not plot a histogram of number of experiments vs start time (Hour UTC) due to exception: {e}')
        print(f'Could not plot a histogram of number of experiments vs start time (Hour UTC) due to exception: {e}')

    try:
        # time from closest controlled release for FP detections histogram
        print("Plotting a histogram of number of False Positives vs timeFromClosestControlledRelease (hours)")
        thisLog.debug("Plotting a histogram of number of False Positives vs timeFromClosestControlledRelease (hours)")
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel("Time from closest controlled release (seconds)")
        ax1.set_ylabel("Number of false positives")
        ax1.hist(FPDF['timeFromClosestControlledRelease'], edgecolor='black')
        fig.tight_layout()
        plt.grid(alpha=alpha)
        path = os.path.join(outputFolder, "Hist_FPs_TimeFromNearestCR.png")
        plt.savefig(path, dpi=400)
        print("saving: " + path)
        plt.close()  # close current figure
        thisLog.info('Successfully plotted a histogram of number of False Positives vs timeFromClosestControlledRelease (hours)')
    except Exception as e:
        thisLog.error(f'Could not plot a histogram of number of False Positives vs timeFromClosestControlledRelease (hours) due to exception: {e}')
        print(f'Could not plot a histogram of number of False Positives vs timeFromClosestControlledRelease (hours) due to exception: {e}')

    try:
        print("Plotting a histogram of number of TP detections vs Time to detection")
        thisLog.debug("Plotting a histogram of number of TP detections vs Time to detection")
        fig, ax1 = plt.subplots(figsize=figsize)
        ax1.set_xlabel("Time to report (hours)")
        ax1.set_ylabel("Number of TP detections")

        if len(TPDF) > 0:
            varsDict[f'tc_mDetectionTimeMin'] = round(tpdf['tc_mDetectionTime'].min(), 3)
            varsDict[f'tc_mDetectionTimeMax'] = round(tpdf['tc_mDetectionTime'].max(), 3)
            varsDict[f'tc_mDetectionTimeMean'] = round(tpdf['tc_mDetectionTime'].mean(), 3)
            varsDict['DetectionTimeLowerCL'] = round(np.quantile(tpdf['tc_mDetectionTime'].tolist(), 2.5 / 100), 3)
            varsDict['DetectionTimeUpperCL'] = round(np.quantile(tpdf['tc_mDetectionTime'].tolist(), 97.5 / 100), 3)
        else:
            varsDict[f'tc_mDetectionTimeMin'] = 0
            varsDict[f'tc_mDetectionTimeMax'] = 0
            varsDict[f'tc_mDetectionTimeMean'] = 0
            varsDict['DetectionTimeLowerCL'] = 0
            varsDict['DetectionTimeUpperCL'] = 0

        ax1.hist(tpdf['tc_mDetectionTime'], edgecolor='black')
        fig.tight_layout()
        plt.grid(alpha=alpha)
        path = os.path.join(outputFolder, "Hist_detectionTime.png")
        plt.savefig(path, dpi=400)
        print("saving: " + path)
        plt.close()  # close current figure
        thisLog.info('Successfully plotted a histogram of number of TP detections vs Time to detection')
    except Exception as e:
        thisLog.error(f'Could not plot a histogram of number of TP detections vs Time to detection due to exception: {e}')
        print(f'Could not plot a histogram of number of TP detections vs Time to detection due to exception: {e}')

    return varsDict, experimentDF
