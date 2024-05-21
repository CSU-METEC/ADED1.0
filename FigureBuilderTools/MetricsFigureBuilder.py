import logging
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from PIL import Image
from FigureBuilderTools.OptimizeCurve import OptimizeCurve
from FigureBuilderTools.Histogram import histogram
from FigureBuilderTools.Histogram import newHistogram
from FigureBuilderTools.PODCurve import buildPODCurve
from FigureBuilderTools.stackedHistChart import stackedHistCSB
from FigureBuilderTools.categoricalScatter import categoricalScatter
from FigureBuilderTools.LogisticRegression import logisticRegression
from FigureBuilderTools.Quantification import quantErrorSubplots
from FigureBuilderTools.categoricalScatter import modifiedCategoricalScatter
from FigureBuilderTools.barchart import barChartPolarAxis
from FigureBuilderTools.scatterWithHist import scatterWithHistogram
from FigureBuilderTools.windrosePlot import windroseplot
from FigureBuilderTools.boxWhiskerPlot import whiskPlot
from FigureBuilderTools.pieChart import PieChart
from FigureBuilderTools.barchart import simpleBarPlot
from FigureBuilderTools.Quantification import quantification
from FigureBuilderTools.barchart import barhPlot
from FigureBuilderTools.boxWhiskerPlot import BoxWhisker
from FigureBuilderTools.AlternativePOD import alternativePodCurve
from Analysis4ContinuousMonitors.Logger import Logging

# Instantiate the logger with a name: __name__
thisLogger = logging.getLogger(__name__)
thisLogger.setLevel(logging.DEBUG)
# Format the instance of the log created
thisLog = Logging(thisLogger, 'metricsFigureBuilder.log')


def roundup(x,n):
    return int(math.ceil(x / float(n))) * n

def metricsFigureBuilder(classifiedDF, outputFilePath, varsDict, yQuantile, bounds):
    # Setting up dynamic ylimit
    varsDict['Quantile'] = yQuantile*100

    # TPDF is True Positives only
    TPDF = classifiedDF.loc[classifiedDF['tc_Classification'] == 'TP']

    # CRDF is for True Positives and False Negatives only
    CRDF = classifiedDF.loc[classifiedDF['tc_Classification'].isin(['TP', 'FN'])]

    try:
        #bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        bins1 = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                 0.9, 1]

        # Save logo to figures folder
        saveLogo(outputFilePath)

        try:
            print("Plotting localization Acurracy Single Coordinate plot for TPs")
            newHistogram(DF=TPDF,
                         xLabel='Localization Accuracy Single Coordinate',
                         yLabel='Number of True Positives',
                         xDataField='tc_mLocalizationAccuracy_SingleCoord',
                         filePath=outputFilePath,
                         saveName='localizationAccuracySingleCoor.png',
                         unit='meters',
                         bins=bins,
                         zoomView=True,
                         zvUpperlimit=21)
        except Exception as e:
            print(f'Could not build localization Acurracy Single Coordinate plot for TPs due to exception: {e}')
        try:
            print("Plotting Localization accuracy Bounding Box plot for TPs")
            newHistogram(DF=TPDF,
                         xLabel='Localization accuracy Bounding Box',
                         yLabel='Number of True Positives',
                         xDataField='tc_mLocalizationAccuracy_BoundingBox',
                         filePath=outputFilePath,
                         saveName='localizationAccuracyBoundingBox.png',
                         unit='meters',
                         bins=bins,
                         zoomView=True,
                         zvUpperlimit=21)
        except Exception as e:
            print(f'Could not build Localization accuracy Bounding Box for TPs due to exception: {e}')

        # Make histogram for # of TP vs localization precision bounding box
        histogram(TPDF, 'Localization Precision Bounding Box', 'Number of True Positives',
                  'tc_mLocalizationPrecision_BoundingBox', outputFilePath,
                  'localizationPrecisionBoundingBox.png', '$\mathregular{m^{2}}$')
        # Make histogram for # of TP vs Quantification accuracy absolute
        histogram(TPDF, 'Quantification Accuracy Absolute', 'Number of True Positives',
                  'tc_mQuantificationAccuracyAbs', outputFilePath,
                  'QuantificationAccuracyAbs.png', 'g/hr')
        # Make histogram for # of TP vs Quantification accuracy relative
        histogram(TPDF, 'Quantification Accuracy Relative', 'Number of True Positives',
                  'tc_mQuantificationAccuracyRel', outputFilePath,
                  'QuantificationAccuracyRel.png', 'unitless')
        # Make histogram for # of TP vs Quantification precision absolute
        histogram(TPDF, 'Quantification Precision Absolute', 'Number of True Positives',
                  'tc_mQuantificationPrecisionAbs', outputFilePath,
                  'QuantificationPrecisionAbs.png', 'g/hr')
        # Make histogram for # of TP vs Quantification precision relative
        histogram(TPDF, 'Quantification Precision Relative', 'Number of True Positives',
                  'tc_mQuantificationPrecisionRel', outputFilePath,
                  'QuantificationPrecisionRel.png', 'unitless')

        # make clay's new figures
        categoricalScatter(classifiedDF, xDataField='tc_mDistanceToClosestSensor',
                           yDataField='tc_EPBFE', catDataField='tc_Classification',
                           xLabel='Distance from sensor (m)', yLabel='Release rate (slpm whole gas)',
                           filePath=outputFilePath, saveName='scatter_EmissionRateVsDistance.png',
                           cats=['FN', 'TP'], s=25)

        categoricalScatter(CRDF, xDataField='tc_ExpDurationHrs',
                           yDataField='tc_EPBFE', catDataField='tc_Classification',
                           xLabel='Controlled Release Duration', yLabel='Release rate (slpm whole gas)',
                           filePath=outputFilePath, saveName='scatter_EmissionRateVsDuration.png',
                           cats=['FN', 'TP'], s=25)

        categoricalScatter(classifiedDF, xDataField='tc_ExpWindSpeedAvg',
                           yDataField='tc_EPBFE', catDataField='tc_Classification',
                           xLabel='Test Center Avg. Wind Speed', yLabel='Release rate (slpm whole gas)',
                           filePath=outputFilePath, saveName='scatter_EmissionRateVsAvgWindSpeed.png',
                           cats=['FN', 'TP'], s=25)

        stackedHistCSB(CRDF, xCategory='tc_C1MassFlow', yCategory='tc_Classification', xScaleFactor=1/1000,
                       filePath= outputFilePath, fileName='Hist_ReleaseRateByClassification_kgPERh.png',
                       xunit="kg/h", xLabel='Methane Release Rate', xBins=[0, 1, 2, 3, 4, 5, 6, 7, 8])

        stackedHistCSB(CRDF.sort_values('tc_EquipmentUnitID'), xCategory='tc_EquipmentUnitID', yCategory='tc_Classification',
                       filePath=outputFilePath, fileName='Hist_EquipmentUnitByClassification.png',
                       xLabel='Equipment unit', xTickRotation=90)

        stackedHistCSB(CRDF.sort_values('tc_EquipmentGroupID'), xCategory='tc_EquipmentGroupID', yCategory='tc_Classification',
                       filePath=outputFilePath, fileName='Hist_EquipmentGroupByClassification.png',
                       xLabel='Equipment group')

        stackedHistCSB(CRDF, xCategory='tc_ExpDurationHrs', yCategory='tc_Classification',
                       filePath=outputFilePath, fileName='Hist_DurationByClassification_h.png',
                       xBins=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                       xLabel='Controlled Release Duration', xunit="h")

        logisticRegression(Df=CRDF,
                           filePath=outputFilePath,
                           fileName='LogisticRegression_slpm.png',
                           desiredLDLFraction=0.9,
                           Nbootstrap=1000,
                           xCategory='tc_EPBFE',
                           xlabel='Whole Gas Release Rate',
                           xunits='slpm',
                           xstep=0.5,
                           legendLOC='best',
                           xmax=200)

        logisticRegression(Df=CRDF,
                           filePath=outputFilePath,
                           fileName="LogisticRegression_kgPERhr.png",
                           desiredLDLFraction=0.9,
                           Nbootstrap=1000,
                           xCategory='tc_C1MassFlow',
                           xScaleFactor=1/1000,
                           xlabel='Methane Gas Release Rate',
                           xunits='kg/h',
                           xstep=0.01,
                           legendLOC='best',
                           xmax=8)
    except Exception as e:
        print(f'Could not build metrics figures due to exception: {e}')

    try:
        print("Plotting logistic regression in m/s")
        logisticRegression(Df=CRDF,
                           desiredLDLFraction=0.9,
                           Nbootstrap=500,
                           xCategory='tc_ExpWindSpeedAvg',
                           xScaleFactor=1,
                           xlabel='Average windspeed',
                           xunits=r"$m/s$",
                           varPrefix='mps',
                           paperDict=varsDict,
                           xstep=0.01,
                           filePath=outputFilePath,
                           fileName="LogisticRegression_mps.png")
    except Exception as e:
        print(f'Could not plot logistic regression in m/s due to exception: {e}')

    try:
        print("Plotting logistic regression in hours")
        logisticRegression(Df=CRDF,
                           desiredLDLFraction=0.9,
                           Nbootstrap=500,
                           xCategory='tc_ExpDurationHrs',
                           xScaleFactor=1,
                           xlabel='Hours',
                           xunits=r"$h$",
                           varPrefix='hr',
                           paperDict=varsDict,
                           xstep=0.01,
                           filePath=outputFilePath,
                           fileName="LogisticRegression_hours.png")
    except Exception as e:
        print(f'Could not plot logistic regression in hours due to exception: {e}')

    try:
        _, _, _ = buildPODCurve(classifiedDF=CRDF,
                                fileName="podCurveEvenCountsWhole_slpm.png",
                                TPLevels=['CorrectUnit', 'CorrectGroup', 'CorrectFacility'],
                                xLabel='Whole Gas Release Rate',
                                xCategory='tc_EPBFE',
                                outputFilePath=outputFilePath,
                                xMax=200,
                                xunit='slpm')
    except Exception as e:
        print(f'Could not build PODcurve figures in slpm due to exception: {e}')

    try:
        _, _, _ = buildPODCurve(classifiedDF=CRDF,
                                fileName="podCurveEvenCountsWhole_kgPERhr.png",
                                TPLevels=['CorrectUnit', 'CorrectGroup', 'CorrectFacility'],
                                xLabel='Methane Gas Release Rate',
                                xCategory='tc_C1MassFlow',
                                xScaleFactor=1/1000,
                                outputFilePath=outputFilePath,
                                xMax=8,
                                xunit="kg/h")
    except Exception as e:
        print(f'Could not build PODcurve figures in kg/hr due to exception: {e}')

    try:
        # --------------------- hist and regressions at different range of windspeed --------------------------------
        windstep = 3
        windmax = int(math.ceil(CRDF['tc_ExpWindSpeedAvg'].max()))
        windBinLower = np.arange(start=0, stop=windmax, step=windstep)
        xUpper = 180
        for lower in np.nditer(windBinLower):
            upper = lower + windstep
            temp_filt = (CRDF['tc_ExpWindSpeedAvg'] >= lower) & (CRDF['tc_ExpWindSpeedAvg'] < upper)
            subDF = CRDF.loc[temp_filt]

            print("Plotting hist and logistic subplots for windspeed from {} to {} m/s...".format(lower, upper))
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 5), gridspec_kw={'height_ratios': [1, 2]})
            _, _, _ = stackedHistCSB(DF=subDF,
                                     xCategory='tc_EPBFE',
                                     yCategory='tc_Classification',
                                     xLabel='Whole Gas Release Rate (slpm)',
                                     xlim=(0, xUpper),
                                     yLabel='Count',
                                     nbins=np.linspace(0, xUpper, 21),
                                     density=False,
                                     gridAlpha=0.5,
                                     xTickRotation=0,
                                     fig=fig,
                                     axes=ax1)

            _, _, _ = logisticRegression(Df=subDF,
                                         filePath=outputFilePath,
                                         fileName="HistogramAndLogisticRegression_{}to{}_mps_slpm.png".format(lower, upper),
                                         desiredLDLFraction=0.90,
                                         Nbootstrap=1000,
                                         xCategory='tc_EPBFE',
                                         xlabel='Whole Gas Release Rate',
                                         xunits="slpm",
                                         xmax=xUpper,
                                         BinBy="wind speeds",
                                         BinLimit=[lower, upper],
                                         binUnit=r"$ms^{-1}$",
                                         xstep=0.5,
                                         fig=fig,
                                         legendLOC='best',
                                         axes=ax2)
            plt.close(fig)
    except Exception as e:
        print(f'Could not build a subplot of Histogram and Logistic Regression binned by wind speed due to exception: {e}')

    try:
        # --------------------- hist and regressions at different range of Duration --------------------------------
        print('Printing a subplot of Histogram and Logistic Regression binned by duration')
        DurStep = 3
        DurMax = int(math.ceil(CRDF['tc_ExpDurationHrs'].max()))
        DurBinLower = np.arange(start=0, stop=DurMax, step=DurStep)
        xUpper = 180
        for lower in np.nditer(DurBinLower):
            upper = lower + DurStep
            temp_filt = (CRDF['tc_ExpDurationHrs'] >= lower) & (CRDF['tc_ExpDurationHrs'] < upper)
            subDF = CRDF.loc[temp_filt]
            print("Plotting hist and logistic subplots for Duration from {} to {} hrs...".format(lower, upper))
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 5), gridspec_kw={'height_ratios': [1, 2]})
            _, _, _ = stackedHistCSB(DF=subDF,
                                     xCategory='tc_EPBFE',
                                     yCategory='tc_Classification',
                                     xLabel='Whole Gas Release Rate (slpm)',
                                     xlim=(0, xUpper),
                                     yLabel='Count',
                                     nbins=np.linspace(0, xUpper, 21),
                                     density=False,
                                     gridAlpha=0.5,
                                     xTickRotation=0,
                                     fig=fig,
                                     axes=ax1)
            _, _, _ = logisticRegression(Df=subDF,
                                         filePath=outputFilePath,
                                         fileName="HistogramAndLogisticRegression_{}to{}_hrs_slpm.png".format(lower, upper),
                                         desiredLDLFraction=0.90,
                                         Nbootstrap=1000,
                                         xCategory='tc_EPBFE',
                                         xlabel='Whole Gas Release Rate',
                                         xunits="slpm",
                                         xmax=xUpper,
                                         BinBy="durations",
                                         BinLimit=[lower, upper],
                                         binUnit='hrs',
                                         xstep=0.5,
                                         legendLOC='best',
                                         fig=fig,
                                         axes=ax2)
            plt.close(fig)
    except Exception as e:
        print(f'Could not build a subplot of Histogram and Logistic Regression binned by duration due to exception: {e}')
    # --------------------- hist and regressions for the whole data  --------------------------------

    try:
        xUpper = 180
        print("Plotting hist and logistic subplots for the whole data")
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 5), gridspec_kw={'height_ratios': [1, 2]})
        _, _, _ = stackedHistCSB(DF=CRDF,
                                 xCategory='tc_EPBFE',
                                 yCategory='tc_Classification',
                                 xLabel='Whole Gas Release Rate (slpm)',
                                 xlim=(0, xUpper),
                                 yLabel='Count',
                                 nbins=np.linspace(0, xUpper, 21),
                                 density=False,
                                 gridAlpha=0.5,
                                 xTickRotation=0,
                                 fig=fig,
                                 axes=ax1)
        _, _, _ = logisticRegression(Df=CRDF,
                                     filePath=outputFilePath,
                                     fileName='HistogramAndLogisticRegressionForAllWindSpeed_slpm.png',
                                     desiredLDLFraction=0.90,
                                     Nbootstrap=1000,
                                     xCategory='tc_EPBFE',
                                     xlabel='Whole Gas Release Rate',
                                     xunits='slpm',
                                     xmax=xUpper,
                                     xstep=0.5,
                                     fig=fig,
                                     axes=ax2,
                                     legendLOC='best',
                                     paperDict=varsDict,
                                     varPrefix="AllWindSpeed_")
        plt.close(fig)
    except Exception as e:
        print(f'Could not build a subplot of Histogram and Logistic Regression due to exception: {e}')

    try:
        print("Plotting logistic regression of mass of methane released")
        xUpper = 22
        logisticRegression(Df=CRDF,
                           filePath=outputFilePath,
                           fileName="LogisticRegressionForTotalMass_kg.png",
                           desiredLDLFraction=0.90,
                           Nbootstrap=1000,
                           xCategory='tc_C1_(kg*hrs)/hr',
                           xlabel='Mass of Methane',
                           xunits="kg",
                           xmax=xUpper,
                           xstep=0.01,
                           paperDict=varsDict,
                           legendLOC='best',
                           varPrefix="MethaneMass_kg_")
    except Exception as e:
        print(f'Could not build a Logistic Regression of total mass released due to exception: {e}')

    try:
        print("Plotting logistic regression normed by distance square")
        logisticRegression(Df=CRDF,
                           filePath=outputFilePath,
                           fileName="LogisticRegressionNormedByDistance.png",
                           desiredLDLFraction=0.90,
                           Nbootstrap=1000,
                           xCategory='tc_C1_kg/(hr*m^2)',
                           xlabel='Methane Mass Normed by the Square of Closest Sensor Distance',
                           xunits=r"$(kg~CH_4/h)/(m^{2})$",
                           xstep=0.0001,
                           paperDict=varsDict,
                           legendLOC='best',
                           varPrefix="MethaneMassNormedByDistance_")
    except Exception as e:
        print(f'Could not build a Logistic Regression normed by distance square due to exception: {e}')

    try:
        print("Plotting logistic regression of mass of methane released multiplied by wind speed")
        xUpper = 60
        logisticRegression(Df=CRDF,
                           filePath=outputFilePath,
                           fileName="LogisticRegressionForMethaneMassWindSpeedPerHr_kg_mpsPERhr.png",
                           desiredLDLFraction=0.90,
                           Nbootstrap=1000,
                           xCategory='tc_C1_(kg*mps)/hr',
                           xlabel='ReleaseRate * WindSpeed',
                           xunits=r"$(kg~CH_4/h)*(m/s)$",
                           xmax=xUpper,
                           xstep=0.01,
                           paperDict=varsDict,
                           legendLOC='best',
                           varPrefix="MethaneMassWindSpeed_kg_mpsPERhr_")
    except Exception as e:
        print(f'Could not build a Logistic Regression of mass released multiplied by wind speed due to exception: {e}')

    # --------------------- Plotting the quantificaation subplots  --------------------------------
    try:
        print("Plotting logistic regression of mass of methane released normed by wind speed")
        xUpper = 4
        logisticRegression(Df=CRDF,
                           filePath=outputFilePath,
                           fileName="LogisticRegressionForMethaneReleaseRateByWindSpeed_kgPERhr_mps.png",
                           desiredLDLFraction=0.90,
                           Nbootstrap=1000,
                           xCategory='tc_C1_kg/(hr*mps)',
                           xlabel='Release Rate Normed by WindSpeed',
                           xunits=r"$(kg~CH_4/h)/(m/s)$",
                           xmax=xUpper,
                           xstep=0.01,
                           paperDict=varsDict,
                           legendLOC='best',
                           varPrefix="MethaneMassWindSpeed_kgPERhr_mps_")
    except Exception as e:
        print(f'Could not build a Logistic Regression of mass of methane released normed by wind speed due to exception: {e}')

    try:
        print("Plotting the scatter plot wind-Temperature subplots")
        modifiedCategoricalScatter(df=CRDF,
                                   xDataField='tc_ExpTAtmAvg',
                                   yDataField='tc_ExpWindSpeedAvg',
                                   catDataField='tc_Classification',
                                   cats=['FN', 'TP'],
                                   xlabel="Average Temperature (Celsius)",
                                   ylabel='Avg Windspeed (m/s)',
                                   plotBYc="No",
                                   filePath=outputFilePath,
                                   fileName='AvgWindSpeedvsAvgTemp.png')
    except Exception as e:
        print(f'Could not build Windspeed Avg Temperature due to exception: {e}')

    try:
        print("Plotting Bar Chart Polar Axis")
        _, _ = barChartPolarAxis(df=CRDF,
                                 thetaData="tc_ExpWindDirAvg",
                                 radialData="tc_ExpWindSpeedAvg",
                                 catDataField="tc_Classification",
                                 cats=['FN', 'TP'],
                                 figsize=(3.54331, 3.54331),
                                 fileName='DetectionForAllWindDirectionAndWindSpeed.png',
                                 filePath=outputFilePath)
        _, _ = barChartPolarAxis(df=CRDF,
                                 thetaData="tc_ExpWindDirAvg",
                                 radialData="tc_ExpWindSpeedAvg",
                                 catDataField="tc_Classification",
                                 cats=['FN', 'TP'],
                                 rlimits=[0, 2],
                                 figsize=(3.54331, 3.54331),
                                 thetaticks=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                                 angles=[0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0],
                                 fileName='DetectionForWindDirectionAndWindSpeed_0to_2mps.png',
                                 filePath=outputFilePath)
    except Exception as e:
        print(f'Could not build bar chart polar axis due to exception: {e}')

    try:
        print("Plotting Scatter With Histogram - kg/hr")
        xMax = roundup(CRDF['tc_ExpDurationHrs'].max(), 1)
        yMax = roundup(CRDF["tc_C1MassFlow"].max()/1000, 1)
        xTicks = list(range(xMax))
        yTicks = list(range(yMax))

        _, _ = scatterWithHistogram(DF=CRDF,
                                    xDataField='tc_ExpDurationHrs', xunit="h", xlabel='Controlled Release Duration',
                                    yDataField="tc_C1MassFlow", yunit="kg/h", yScaleFactor=1/1000,
                                    ylabel='Methane Mass Flow Rate', catDataField="tc_Classification", cats=['FN', 'TP'],
                                    xticks=xTicks, xhistTick=[0, 50, 100, 150], yhistTick=[0, 100, 200, 300],
                                    yticks=yTicks, filePath=outputFilePath,
                                    fileName='scatterWHist_EmissionRateVsDuration_kgPERh_h.png')
    except Exception as e:
        print(f'Could not build Scatter with Histogram due to exception: {e}')

    try:
        print("Plotting Scatter With Histogram - slpm")
        xMax = roundup(CRDF['tc_ExpDurationHrs'].max(), 1)
        yMax = roundup(CRDF["tc_EPBFE"].max(), 10)
        xTicks = list(range(xMax))
        yTicks = list(range(0, yMax, 20))
        _, _ = scatterWithHistogram(DF=CRDF,
                                    xDataField='tc_ExpDurationHrs', xunit="h", xlabel='Controlled Release Duration',
                                    yDataField="tc_EPBFE", yunit="slpm", yScaleFactor=1,
                                    ylabel='Methane Mass Flow Rate', catDataField="tc_Classification", cats=['FN', 'TP'],
                                    xticks=xTicks, xhistTick=[0, 50, 100, 150], yhistTick=[0, 100, 200, 300],
                                    yticks=yTicks, filePath=outputFilePath,
                                    fileName='scatterWHist_EmissionRateVsDuration_slpm_h.png')
    except Exception as e:
        print(f'Could not build Scatter with Histogram due to exception: {e}')

    try:
        xUpper = 180
        print("Plotting POD Curve and logistic subplots for the whole data")
        fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 1]})
        _, _, _ = logisticRegression(Df=CRDF,
                                     desiredLDLFraction=0.90,
                                     Nbootstrap=1000,
                                     xCategory='tc_EPBFE',
                                     xlabel='Whole Gas Release Rate',
                                     xunits='(slpm)',
                                     xmax=xUpper,
                                     xstep=0.5,
                                     fig=fig1,
                                     legendLOC='best',
                                     axes=ax1)
        _, _, _ = buildPODCurve(classifiedDF=CRDF,
                                fileName="LogisticRegressionAndpodCurveEvenCountsWhole_slpm.png",
                                TPLevels=['CorrectUnit', 'CorrectGroup', 'CorrectFacility'],
                                xLabel='Whole Gas Release Rate',
                                xCategory='tc_EPBFE',
                                outputFilePath=outputFilePath,
                                fig=fig1,
                                axes=ax2,
                                xMax=xUpper,
                                xunit='(slpm)',
                                legendLOC='best')
        plt.close(fig1)
    except Exception as e:
        print(f'Could not build a subplot of POD Curve and Logistic Regression due to exception: {e}')

    try:
        xUpper = 5
        print("Plotting a subplot logistic regression of mass of methane released per wind speed")
        fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 1]})
        _, _, _ = logisticRegression(Df=CRDF,
                                     desiredLDLFraction=0.90,
                                     Nbootstrap=1000,
                                     xCategory='tc_C1_kg/(hr*mps)',
                                     xlabel='Release Rate per WindSpeed',
                                     xunits=r"$(kg~CH_4/h)/(m/s)$",
                                     xmax=xUpper,
                                     fig=fig1,
                                     axes=ax1,
                                     xstep=0.01)
        _, _, _ = buildPODCurve(classifiedDF=CRDF,
                                fileName="subplotLogisticRegressionForMethaneReleaseRateByWindSpeedpodCurveEvenCountsWhole_kgPERhr_mps.png",
                                TPLevels=['CorrectUnit', 'CorrectGroup', 'CorrectFacility'],
                                xLabel='Release Rate per WindSpeed',
                                xCategory='tc_C1_kg/(hr*mps)',
                                outputFilePath=outputFilePath,
                                fig=fig1,
                                axes=ax2,
                                xMax=xUpper,
                                legendLOC='best',
                                xunit=r"$(kg~CH_4/h)/(m/s)$")
        plt.close(fig1)
    except Exception as e:
        print(f'Could not build a subplot of Logistic Regression and POD of mass of methane released normed by wind speed due to exception: {e}')

    try:
        xUpper = 8
        print("Plotting POD Curve and logistic subplots for the whole data in kg/hr")
        fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 1]})
        _, _, _ = logisticRegression(Df=CRDF,
                                     desiredLDLFraction=0.90,
                                     Nbootstrap=1000,
                                     xCategory='tc_C1MassFlow',
                                     xlabel='Release Rate',
                                     xunits=r"$(kg~CH_4/h)$",
                                     xScaleFactor=1/1000,
                                     xstep=0.01,
                                     xmax=xUpper,
                                     fig=fig1,
                                     axes=ax1)
        _, _, _ = buildPODCurve(classifiedDF=CRDF,
                                fileName="LogisticRegressionAndpodCurveEvenCountsWhole_kgPERhr.png",
                                TPLevels=['CorrectUnit', 'CorrectGroup', 'CorrectFacility'],
                                xLabel='Release Rate',
                                xCategory='tc_C1MassFlow',
                                outputFilePath=outputFilePath,
                                fig=fig1,
                                axes=ax2,
                                xScaleFactor=1/1000,
                                xMax=xUpper,
                                xunit=r"$(kg~CH_4/h)$",
                                legendLOC='best')
        plt.close(fig1)
    except Exception as e:
        print(f'Could not build a subplot of POD Curve and Logistic Regression in kg/hr due to exception: {e}')

    try:
        print("Plotting alternative POD (single - emission rate)")
        unit = r"$\mathrm{(kg~CH_4/h)}$"
        thisVar = {} #Initialize the dictionary for collecting variables
        listOFnBins = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
        thisLog.debug("Generating the initial alternative POD curve (single - emission rate)")
        print("Generating the initial alternative POD curve (single - emission rate)")
        _, ax, thisVar = alternativePodCurve(Df=CRDF, desiredLDLFraction=0.90, Nbootstrap=1000,
                                             xScaleFactor=0.001, xData="tc_C1MassFlow", tData='Study Year',
                                             cData='tc_Classification', xunits=unit, digits=1, fontsize=8,
                                             legendLOC='lower right', bootstrapp=True, varPrefix="CF",
                                             paperDict=thisVar, listOFnBins=listOFnBins, CFnBins=10,
                                             xlabel=f'Release rate {unit}', ylabel='Probability of Detection [-]',
                                             filePath=outputFilePath, fileName='alternatePOD.png')
        # Clears the existing axes
        thisLog.debug("Generating the optimized alternative POD curve (single - emission rate)")
        print("Generating the optimized alternative POD curve (single - emission rate)")
        ax.cla()
        # Optimize the POD curve by finding the curve with the highest r-square with mean point per bin between 30 and 50
        nBin = OptimizeCurve(VariablesDict=thisVar, varPrefix="CF", listOFnBins=listOFnBins, bounds=bounds)
        # Using the optimized mean points per bin, make the plots again
        _, ax, varsDict = alternativePodCurve(Df=CRDF, desiredLDLFraction=0.90, Nbootstrap=1000,
                                              xScaleFactor=0.001, xData="tc_C1MassFlow", tData='Study Year',
                                              cData='tc_Classification', xunits=unit, digits=1, fontsize=8,
                                              legendLOC='lower right', bootstrapp=True, varPrefix="CF",
                                              paperDict=varsDict, listOFnBins=listOFnBins, CFnBins=nBin,
                                              xlabel=f'Release rate {unit}', ylabel='Probability of Detection [-]',
                                              filePath=outputFilePath, fileName='alternatePOD.png')
        thisLog.info("Successfully generated the optimized alternative POD curve (single - emission rate)")
    except Exception as e:
        print(f'Could not plot the alternate POD curve (single - emission rate) due to exception: {e}')
        thisLog.error(f'Could not plot the alternate POD curve (single - emission rate) due to exception: {e}')

    try:
        xUpper = 5
        print("Plotting a subplot logistic regression and curvefitting POD of mass of methane released per wind speed")
        fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 1]})
        unit = r"$(kg~CH_4/h)/(m/s)$"
        thisVar = {} #Initialize the dictionary for collecting variables
        listOFnBins = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
        thisLog.debug("Generating the initial alternative POD curve (POD of mass of methane released normed by wind speed)")
        print("Generating the initial alternative POD curve (POD of mass of methane released normed by wind speed)")
        _, _, _ = logisticRegression(Df=CRDF,
                                     desiredLDLFraction=0.90,
                                     Nbootstrap=1000,
                                     xCategory='tc_C1_kg/(hr*mps)',
                                     xlabel='ReleaseRate per WindSpeed',
                                     xunits=unit,
                                     xmax=xUpper,
                                     fig=fig1,
                                     axes=ax1,
                                     xstep=0.01)
        _, _, thisVar = alternativePodCurve(Df=CRDF, desiredLDLFraction=0.90, Nbootstrap=1000,
                                            xScaleFactor=1, xData='tc_C1_kg/(hr*mps)', tData='Study Year',
                                            cData='tc_Classification', xunits=unit, digits=1, fontsize=9,
                                            legendLOC='lower right', bootstrapp=True, varPrefix="CF",
                                            paperDict=thisVar, listOFnBins=listOFnBins, CFnBins=10,
                                            xlabel=f'ReleaseRate per windspeed {unit}', xMax=xUpper,
                                            ylabel='Probability of Detection', fig=fig1, axes=ax2,
                                            filePath=None, fileName=None)
        thisLog.debug("Generating the optimized alternative POD curve (POD of mass of methane released normed by wind speed)")
        print("Generating the optimized alternative POD curve (POD of mass of methane released normed by wind speed)")
        # Clears the existing axes
        ax2.cla()
        # Optimize the POD curve by finding the curve with the highest r-square with mean point per bin between 30 and 50
        nBin = OptimizeCurve(VariablesDict=thisVar, varPrefix="CF", listOFnBins=listOFnBins, bounds=bounds)
        # Using the optimized mean points per bin, make the plots again
        _, _, varsDict = alternativePodCurve(Df=CRDF, desiredLDLFraction=0.90, Nbootstrap=1000,
                                             xScaleFactor=1, xData='tc_C1_kg/(hr*mps)', tData='Study Year',
                                             cData='tc_Classification', xunits=unit, digits=1, fontsize=9,
                                             legendLOC='lower right', bootstrapp=True, varPrefix="CF",
                                             paperDict=varsDict, listOFnBins=listOFnBins, CFnBins=nBin,
                                             xlabel=f'ReleaseRate per windspeed {unit}', xMax=xUpper,
                                             ylabel='Probability of Detection', fig=fig1, axes=ax2,
                                             filePath=outputFilePath,
                                             fileName='LogisticRegressionAndAlternatePOD_kgPERhr_mps.png')
        thisLog.info("Successfully generated the optimized alternative POD curve (POD of mass of methane released normed by wind speed)")
        plt.close(fig1)
    except Exception as e:
        print(f'Could not build a subplot of logistic regression and curvefitting POD of mass of methane released normed by wind speed due to exception: {e}')
        thisLog.error(f'Could not build a subplot of logistic regression and curvefitting POD of mass of methane released normed by wind speed due to exception: {e}')


    try:
        print("Plotting the subplots of localization precision POD of mass of methane released normed by wind speed")
        fig = plt.figure(constrained_layout=True, figsize=(6, 3))
        gs = fig.add_gridspec(1, 2)
        positions = [(0, 0), (0, 1)]
        xData = ['tc_C1MassFlow', 'tc_C1_kg/(hr*mps)']
        xUnits = [r"$(kg~CH_4/h)$", r"$(kg~CH_4/h)/(m/s)$"]
        xLabels = ['Release Rate', 'ReleaseRate per windspeed']
        xMaxs = [8, 5]
        scaling = [1/1000, 1]

        for i in list(range(len(positions))):
            f_ax = fig.add_subplot(gs[positions[i]])
            columnHeader = xData[i]
            xunit = xUnits[i]
            label = xLabels[i]
            mx = xMaxs[i]
            scale = scaling[i]
            buildPODCurve(classifiedDF=CRDF,
                          TPLevels=['CorrectUnit', 'CorrectGroup', 'CorrectFacility'],
                          xLabel=label,
                          xCategory=columnHeader,
                          xScaleFactor=scale,
                          xMax=mx,
                          fig=fig,
                          axes=f_ax,
                          xunit=xunit,
                          legendLOC='lower right')
        fig.tight_layout()
        filename = "podCurveEvenCountsWhole.png"
        path = os.path.join(outputFilePath, filename)
        plt.savefig(path, dpi=400)

    except Exception as e:
        print(f'Could not build a the subplots localization precision POD of mass of methane released normed by wind speed due to exception: {e}')

    try:
        xUpper = 8
        print("Plotting POD Curve Fitting and logistic subplots for the whole data in kg/hr")
        fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 1]})
        unit = r"$\mathrm{(kg~CH_4/h)}$"
        thisVar = {}
        # List of the quartiles to divide your data points
        listOFnBins = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
        thisLog.debug("Generating the initial alternative POD curve (POD Curve Fitting and Logistic Regression in kg/hr)")
        print("Generating the initial alternative POD curve (POD Curve Fitting and Logistic Regression in kg/hr)")
        _, _, _ = logisticRegression(Df=CRDF,
                                     desiredLDLFraction=0.90,
                                     Nbootstrap=1000,
                                     xCategory='tc_C1MassFlow',
                                     xlabel='Release Rate',
                                     xunits=unit,
                                     xScaleFactor=1/1000,
                                     xstep=0.01,
                                     xmax=xUpper,
                                     fig=fig1,
                                     axes=ax1)
        _, _, thisVar = alternativePodCurve(Df=CRDF, desiredLDLFraction=0.90, Nbootstrap=1000,
                                            xScaleFactor=0.001, xData="tc_C1MassFlow", tData='Study Year',
                                            cData='tc_Classification', xunits=unit, digits=1, fontsize=9,
                                            legendLOC='lower right', bootstrapp=True, varPrefix="CF",
                                            paperDict=thisVar, listOFnBins=listOFnBins, CFnBins=10,
                                            xlabel=f'Release rate {unit}', fig=fig1, axes=ax2, xMax=8,
                                            ylabel='Probability of Detection', filePath=None, fileName=None)
        thisLog.debug("Generating the optimized alternative POD curve (POD Curve Fitting and Logistic Regression in kg/hr)")
        print("Generating the optimized alternative POD curve (POD Curve Fitting and Logistic Regression in kg/hr)")
        # Clears the existing axes
        ax2.cla()
        # Optimize the POD curve by finding the curve with the highest r-square with mean point per bin between 30 and 50
        nBin = OptimizeCurve(VariablesDict=thisVar, varPrefix="CF", listOFnBins=listOFnBins, bounds=bounds)
        # Using the optimized mean points per bin, make the plots again
        _, _, varsDict = alternativePodCurve(Df=CRDF, desiredLDLFraction=0.90, Nbootstrap=1000,
                                             xScaleFactor=0.001, xData="tc_C1MassFlow", tData='Study Year',
                                             cData='tc_Classification', xunits=unit, digits=1, fontsize=9,
                                             legendLOC='lower right', bootstrapp=True, varPrefix="CF",
                                             paperDict=varsDict, listOFnBins=listOFnBins, CFnBins=nBin,
                                             xlabel=f'Release rate {unit}', fig=fig1, axes=ax2, xMax=8,
                                             ylabel='Probability of Detection', filePath=outputFilePath,
                                             fileName='LogisticRegressionAndAlternatePOD_kgPERhr.png')
        thisLog.info("Successfully generated the optimized alternative POD curve (POD Curve Fitting and Logistic Regression in kg/hr)...")
        plt.close(fig1)
    except Exception as e:
        print(f'Could not build a subplot of POD Curve Fitting and Logistic Regression in kg/hr due to exception: {e}')
        thisLog.error(f'Could not build a subplot of POD Curve Fitting and Logistic Regression in kg/hr due to exception: {e}')

    try:
        xUpper = 8
        print("Plotting POD Curve Fitting and logistic subplots in hr")
        fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 8), gridspec_kw={'height_ratios': [1, 1]})
        unit = r"$h$"
        thisVar = {}
        # List of the quartiles to divide your data points
        listOFnBins = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
        thisLog.debug("Generating the initial alternative POD curve (POD Curve Fitting and Logistic Regression in hr)")
        print("Generating the initial alternative POD curve (POD Curve Fitting and Logistic Regression in hr)")
        _, _, _ = logisticRegression(Df=CRDF,
                                     desiredLDLFraction=0.90,
                                     Nbootstrap=1000,
                                     xCategory='tc_ExpDurationHrs',
                                     xlabel=f'Release duration {unit}',
                                     xunits=unit,
                                     xScaleFactor=1/1000,
                                     xstep=0.01,
                                     xmax=xUpper,
                                     fig=fig1,
                                     axes=ax1)
        _, _, thisVar = alternativePodCurve(Df=CRDF, desiredLDLFraction=0.90, Nbootstrap=1000,
                                            xScaleFactor=0.001, xData='tc_ExpDurationHrs', tData='Study Year',
                                            cData='tc_Classification', xunits=unit, digits=1, fontsize=9,
                                            legendLOC='lower right', bootstrapp=True, varPrefix="CF",
                                            paperDict=thisVar, listOFnBins=listOFnBins, CFnBins=10,
                                            xlabel=f'Release duration {unit}', fig=fig1, axes=ax2, xMax=8,
                                            ylabel='Probability of Detection', filePath=None, fileName=None)
        thisLog.debug("Generating the optimized alternative POD curve (POD Curve Fitting and Logistic Regression in hr)")
        print("Generating the optimized alternative POD curve (POD Curve Fitting and Logistic Regression in hr)")
        # Clears the existing axes
        ax2.cla()
        # Optimize the POD curve by finding the curve with the highest r-square with mean point per bin between 30 and 50
        nBin = OptimizeCurve(VariablesDict=thisVar, varPrefix="CF", listOFnBins=listOFnBins, bounds=bounds)
        # Using the optimized mean points per bin, make the plots again
        _, _, varsDict = alternativePodCurve(Df=CRDF, desiredLDLFraction=0.90, Nbootstrap=1000,
                                             xScaleFactor=0.001, xData='tc_ExpDurationHrs', tData='Study Year',
                                             cData='tc_Classification', xunits=unit, digits=1, fontsize=9,
                                             legendLOC='lower right', bootstrapp=True, varPrefix="CF",
                                             paperDict=varsDict, listOFnBins=listOFnBins, CFnBins=nBin,
                                             xlabel=f'Release duration {unit}', fig=fig1, axes=ax2, xMax=8,
                                             ylabel='Probability of Detection', filePath=outputFilePath,
                                             fileName='LogisticRegressionAndAlternatePOD_hr.png')
        thisLog.info("Successfully generated the optimized alternative POD curve (POD Curve Fitting and Logistic Regression in hr)")
        plt.close(fig1)
    except Exception as e:
        print(f'Could not build a subplot of POD Curve Fitting and Logistic Regression in hr due to exception: {e}')
        thisLog.error(f'Could not build a subplot of POD Curve Fitting and Logistic Regression in hr due to exception: {e}')

    try:
        xUpper = 8
        print("Plotting localization precision POD curve for the whole data in kg/hr")
        _, _, _ = buildPODCurve(classifiedDF=CRDF,
                                fileName="PodCurveEvenCountsWhole_kgPERhr.png",
                                TPLevels=['CorrectUnit', 'CorrectGroup', 'CorrectFacility'],
                                xLabel='Release Rate',
                                xCategory='tc_C1MassFlow',
                                outputFilePath=outputFilePath,
                                xScaleFactor=1/1000,
                                xMax=xUpper,
                                xunit=r"$(kg~CH_4/h)$",
                                legendLOC='best')
    except Exception as e:
        print(f'Could not build a localization precision POD curve in kg/hr due to exception: {e}')

    try:
        xUpper = 5
        print("Plotting localization precision POD curve for the whole data in kg/hr")
        _, _, _ = buildPODCurve(classifiedDF=CRDF,
                                fileName="PodCurveEvenCountsWhole_kgPERhr_mps.png",
                                TPLevels=['CorrectUnit', 'CorrectGroup', 'CorrectFacility'],
                                xLabel='ReleaseRate per WindSpeed',
                                xCategory='tc_C1_kg/(hr*mps)',
                                outputFilePath=outputFilePath,
                                xMax=xUpper,
                                xunit=r"$(kg~CH_4/h)/(m/s)$",
                                legendLOC='best')
    except Exception as e:
        print(f'Could not build a localization precision POD curve in kg/hr due to exception: {e}')

    try:
        print("Plotting box whiskers and histogram plot")
        whiskPlot(df=CRDF,
                  xDataField="tc_EquipmentGroupID",
                  yDataField="tc_C1MassFlow",
                  subDataField="tc_Classification",
                  y2label="Methane Mass Flow Rate",
                  yunit="kg/h",
                  yScaleFactor=1/1000,
                  fileName='BoxPlot_EmissionRateVsEquipmentGroup_kgPERh.png',
                  filePath=outputFilePath)
    except Exception as e:
        print(f'Could not build a box whiskers and histogram plot due to exception: {e}')

    try:
        print("Plotting a pie chart for Performer Alerts")
        PieChart(DF=classifiedDF,
                 DataField="tc_Classification",
                 Cats=["FP", "TP"],
                 figsize=(3, 2),
                 fileName='Classification_PerformerAlerts.png',
                 filePath=outputFilePath)
    except Exception as e:
        print(f'Could not build the pie-bar chart for Performer Alerts due to exception: {e}')

    try:
        print("Plotting a pie chart for detection classification")
        PieChart(DF=classifiedDF,
                 DataField="tc_Classification",
                 Cats=["FN", "TP"],
                 figsize=(3, 2),
                 fileName='Classification_ControlledReleases.png',
                 filePath=outputFilePath)
    except Exception as e:
        print(f'Could not build the pie-bar chart for detection classification due to exception: {e}')

    try:
        print("Plotting a pie chart for localization precision (Bounding Box)")
        PieChart(DF=TPDF,
                 DataField="tc_mBoundingBoxAccuracy",
                 Cats=[True, False],
                 figsize=(3, 2),
                 labels=['Inside', 'Outside'],
                 fileName='Classification_LocalizationPrecision_bbx.png',
                 filePath=outputFilePath)
    except Exception as e:
        print(f'Could not build the pie-bar chart for localization precision (Bounding Box) due to exception: {e}')

    try:
        print("Plotting the pie chart categorizing FP rate")
        fpdf = classifiedDF.loc[classifiedDF["tc_Classification"] == 'FP']
        PieChart(DF=fpdf,
                 DataField="FP category",
                 Cats=['Extra Reports', 'No experiment running'],
                 figsize=(3, 2),
                 labels=['Excess', 'No. C.R'],
                 colors=['teal', 'saddlebrown', 'darkred'],
                 fileName='Classification_FPrate.png',
                 filePath=outputFilePath)
    except Exception as e:
        print(f'Could not build the pie chart categorizing FP rate due to exception: {e}')

    try:
        print("Plotting a pie chart for Localization Precision (Equipment)")
        PieChart(DF=TPDF,
                 DataField="tc_mLocalizationPrecision",
                 Cats=["CorrectUnit", "CorrectGroup", "CorrectFacility"],
                 labels=['Unit', 'Group', 'Facility'],
                 fileName='Classification_LocalizationPrecision.png',
                 filePath=outputFilePath)
    except Exception as e:
        print(f'Could not build the pie-bar chart for Localization Precision (Equipment) due to exception: {e}')

    try:
        print("Plotting a bar chart of the distribution of emission points in each experiment")
        simpleBarPlot(DF=CRDF,
                      xData='tc_ExperimentID',
                      alpha=0.3,
                      fileName="Hist_SourcesPerExperiment2.png",
                      filePath=outputFilePath,
                      figSize=(3.54331, 3.54331 / 1.5),
                      xLabel="Sources per experiment",
                      yLabel="Number of controlled releases",
                      dataWranglingMethod="allCategorizedRows")
    except Exception as e:
        print(f'Could not build the bar chart of the distribution of emission points in each experiment: {e}')

    try:
        print("Plotting a bar chart of the distribution of experiments by emission point")
        simpleBarPlot(DF=CRDF,
                      xData='tc_ExperimentID',
                      alpha=0.3,
                      fileName="Hist_SourcesPerExperiment.png",
                      filePath=outputFilePath,
                      figSize=(3.54331, 3.54331 / 1.5),
                      xLabel="Sources per experiment",
                      yLabel="Number of Experiments",
                      dataWranglingMethod="selectedCategorizedRows")
    except Exception as e:
        print(f'Could not build the bar chart of the distribution of emission points in each experiment: {e}')

    try:
        print("Plotting horizontal bar charts")
        # Wrangle Data into the form for plotting
        CRsTPfracList = []
        CRsFNfracList = []
        locUnitfracList = []
        locGrpfracList = []
        locFacfracList = []
        countTPFN = []
        countTPs = []

        # Categorizing xData based on the range in upperBin
        WDF = classifiedDF.loc[(classifiedDF['tc_Classification'] == 'TP') | (classifiedDF['tc_Classification'] == 'FP') | \
                       (classifiedDF['tc_Classification'] == 'FN')]
        WDF["rateCategorization"] = None
        for i, row in WDF.iterrows():
            if (row['tc_C1MassFlow'] >= 0) and (row['tc_C1MassFlow'] < 10):
                WDF.loc[i, "rateCategorization"] = "[0, 10)"
            elif (row['tc_C1MassFlow'] >= 10) and (row['tc_C1MassFlow'] < 100):
                WDF.loc[i, "rateCategorization"] = "[10, 100)"
            elif (row['tc_C1MassFlow'] >= 100) and (row['tc_C1MassFlow'] < 1000):
                WDF.loc[i, "rateCategorization"] = "[100, 1000)"
            elif row['tc_C1MassFlow'] >= 1000:
                WDF.loc[i, "rateCategorization"] = "[1000, )"
            else:
                pass

        yVar = ["[10, 100)", "[100, 1000)", "[1000, )"]
        filter = (WDF["rateCategorization"] == yVar[0]) | (WDF["rateCategorization"] == yVar[1]) | (WDF["rateCategorization"] == yVar[2])
        df = WDF.loc[filter & (WDF['tc_Classification'] == 'TP')]

        # Calculate the percentage TP, FN, FP, and localization precision for different emission rate bin
        for y in yVar:
            nTP = len(WDF.loc[(WDF['tc_Classification'] == 'TP') & (WDF["rateCategorization"] == str(y))])
            nFN = len(WDF.loc[(WDF['tc_Classification'] == 'FN') & (WDF["rateCategorization"] == str(y))])
            nUnit = len(df.loc[(df["rateCategorization"] == str(y)) & (df['tc_mLocalizationPrecision'] == 'CorrectUnit')])
            nGrp = len(df.loc[(df["rateCategorization"] == str(y)) & (df['tc_mLocalizationPrecision'] == 'CorrectGroup')])
            nFac = len(df.loc[(df["rateCategorization"] == str(y)) & (df['tc_mLocalizationPrecision'] == 'CorrectFacility')])
            CRsTPfrac = float(f'{(nTP / (nTP + nFN)) * 100:.1f}')
            CRsTPfracList.append(CRsTPfrac)
            CRsFNfracList.append(float(100) - CRsTPfrac)
            countTPFN.append(nTP + nFN)
            locUnitfrac = float(f'{(nUnit / nTP) * 100:.1f}')
            locUnitfracList.append(locUnitfrac)
            locGrpfrac = float(f'{(nGrp / nTP) * 100:.1f}')
            locGrpfracList.append(locGrpfrac)
            locFacfracList.append(float(100) - locUnitfrac - locGrpfrac)
            countTPs.append(nTP)
        barsPERcategoryDict = {'Bar1': {'Data': [CRsTPfracList, CRsFNfracList],
                                        'Labels': ['TP', 'FN'],
                                        'Colors': ['orange', 'blue'],
                                        'Count': countTPFN},
                               'Bar2': {'Data': [locUnitfracList, locGrpfracList, locFacfracList],
                                        'Labels': ['Unit', 'Group', 'Facility'],
                                        'Colors': ['tan', 'skyblue', 'magenta'],
                                        'Count': countTPs}}
        title = 'Per group of bars - Top: Localization Precision.\nBottom: Controlled releases'
        _, _ = barhPlot(yVar=yVar,
                        barTags=['Bar1', 'Bar2'],
                        dataDict=barsPERcategoryDict,
                        height=0.30,
                        xTick=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        yTickLabel=yVar,
                        yLabel='Controlled Release (g/hr)',
                        xLabel='Percentage (%)',
                        figSize=(4.2, 3.54331/1.1),
                        ncol=3,
                        fontsize=7,
                        title=title,
                        fileName='DetectionAndLocalizationHbar.png',
                        filePath=outputFilePath)
    except Exception as e:
        print(f'Could not plot the horizontal bar chats due to :{e}')

    try:
        print("Add metrics to paper dict")
        df = CRDF.assign(count=lambda x: x.groupby(['tc_ExperimentID'])['tc_ExperimentID'].transform("count"))
        df = df.drop_duplicates(subset='tc_ExperimentID', keep="first")
        counts = df["count"].replace(np.nan, 'None')
        labels, counts = np.unique(counts, return_counts=True)
        varsDict['tcmMinEPCounts'] = min(list(labels))
        varsDict['tcmMaxEPCounts'] = max(list(labels))
        varsDict['tcmMeanEPCounts'] = df["count"].mean()
        varsDict['tcmMeanEPCounts'] = df["count"].mean()
    except Exception as e:
        print(f'Could not add metrics to paper dict due to: {e}')

    try:
        print("Plotting windrose")
        _, _ = windroseplot(df=CRDF,
                            windData="tc_ExpWindSpeedAvg",
                            directionData="tc_ExpWindDirAvg",
                            filePath=outputFilePath)
    except Exception as e:
        print(f'Could not plot a windrose due to exception: {e}')

    axisDataName = ''
    axisLabelRelease = ''
    axisLabelReported = ''
    yLabel = ''
    if 'METHANE' in TPDF['p_Gas'].tolist():
        axisDataName = 'tc_C1MassFlow'
        axisLabelRelease = 'Release Rate (g CH4/h)'
        axisLabelReported = 'Reported Rate (g CH4/h)'
        yLabel = 'Quantification error (g CH4/h)'
    elif 'Methane' in TPDF['p_Gas'].tolist():
        axisDataName = 'tc_C1MassFlow'
        axisLabelRelease = 'Release Rate (g CH4/h)'
        axisLabelReported = 'Reported Rate (g CH4/h)'
        yLabel = 'Quantification error (g CH4/h)'
    if 'methane' in TPDF['p_Gas'].tolist():
        axisDataName = 'tc_C1MassFlow'
        axisLabelRelease = 'Release Rate (g CH4/h)'
        axisLabelReported = 'Reported Rate (g CH4/h)'
        yLabel = 'Quantification error (g CH4/h)'
    elif 'THC' in TPDF['p_Gas'].tolist():
        axisDataName = 'tc_THCMassFlow'
        axisLabelRelease = 'Release Rate (g Whole gas/h)'
        axisLabelReported = 'Reported Rate (g Whole gas/h)'
        yLabel = 'Quantification error (g whole gas/h)'

    hrLimit = (0, roundup(TPDF['tc_ExpDurationHrs'].max(), 1))
    hrLabels = list(range(roundup(TPDF['tc_ExpDurationHrs'].max(), 1)))
    try:
        print("Plotting the Quantification Data with Error Subplots with color bar as duration")
        quantErrorSubplots(df=TPDF,
                           x1DataField=axisDataName, y1DataField='p_EmissionRate',
                           x2DataField=axisDataName, y2DataField='tc_mQuantificationAccuracyRel',
                           y3DataField='tc_mQuantificationAccuracyRel',
                           y2scalingfactor=100,
                           y3scalingfactor=100,
                           x2binUpperEdges=[10, 100, 1000, 10000, 100000],
                           quantile=yQuantile,
                           x1label=axisLabelRelease,
                           x1limits=(0, roundup(TPDF[str(axisDataName)].max(), 100)),
                           y1label=axisLabelReported,
                           y1limits=(0, roundup(TPDF[str('p_EmissionRate')].max(), 100)),
                           x2label=axisLabelRelease,
                           x2limits=(10, 10000),
                           x2ticks=[10, 100, 1000, 10000],
                           x2ticklabels=[10, 100, 1000, 10000],
                           x2scale='log',
                           y2label='Quantification Error (%)',
                           x3label=axisLabelRelease,
                           x3limits=(10, 10000),
                           x3ticks=[10, 100, 1000, 10000],
                           x3ticklabels=[10, 100, 1000, 10000],
                           x3scale='log',
                           y3label='Quantification Error (%)',
                           cDataField='tc_ExpDurationHrs',
                           clabel='Duration (h)',
                           cticks=hrLabels,
                           climits=hrLimit,
                           figsize=(5, 7.5),
                           dpi=400,
                           whiskLegendPosition='best',
                           gridAlpha=0,
                           paperDict=varsDict,
                           varPrefix="QuantError",
                           filePath=outputFilePath,
                           fileName='QuantificationSummary_ReleaseRate_Duration.png')
                   #x1limits = (0, 6700)
    except Exception as e:
        print(f'Could not build QuantError_ReleaseRate_Duration subplots due to exception: {e}')

    try:
        print("Plotting the Quantification Data with Error Subplots with color bar as duration-combined relative & absolute")
        quantErrorSubplots(df=TPDF,
                           x1DataField=axisDataName, y1DataField='p_EmissionRate',
                           x2DataField=axisDataName, y2DataField='tc_mQuantificationAccuracyAbs',
                           y3DataField='tc_mQuantificationAccuracyRel',
                           y2scalingfactor=1,
                           y3scalingfactor=100,
                           x2binUpperEdges=[10, 100, 1000, 10000, 100000],
                           quantile=yQuantile,
                           x1label=axisLabelRelease,
                           x1limits=(0, roundup(TPDF[str(axisDataName)].max(), 100)),
                           y1label=axisLabelReported,
                           y1limits=(0, roundup(TPDF[str('p_EmissionRate')].max(), 100)),
                           x2label=axisLabelRelease,
                           x2limits=(10, 10000),
                           x2ticks=[10, 100, 1000, 10000],
                           x2ticklabels=[10, 100, 1000, 10000],
                           x2scale='log',
                           y2label=yLabel,
                           x3label=axisLabelRelease,
                           x3limits=(10, 10000),
                           x3ticks=[10, 100, 1000, 10000],
                           x3ticklabels=[10, 100, 1000, 10000],
                           x3scale='log',
                           y3label='Quantification Error (%)',
                           cDataField='tc_ExpDurationHrs',
                           clabel='Duration (h)',
                           cticks=hrLabels,
                           climits=hrLimit,
                           figsize=(3.33, 6),
                           dpi=400,
                           whiskLegendPosition='best',
                           gridAlpha=0,
                           paperDict=varsDict,
                           varPrefix="QuantError",
                           filePath=outputFilePath,
                           fileName='QuantificationSummary_ReleaseRate_Duration_AR.png')
    except Exception as e:
        print(f'Could not build QuantError_ReleaseRate_Duration subplots due to exception: {e}')

    try:
        print("Plotting the Quantification Data with Error Subplots with color bar as windspeed")
        quantErrorSubplots(df=TPDF,
                           x1DataField=axisDataName, y1DataField='p_EmissionRate',
                           x2DataField=axisDataName, y2DataField='tc_mQuantificationAccuracyRel',
                           y3DataField='tc_mQuantificationAccuracyRel',
                           quantile=yQuantile,
                           y2scalingfactor=100,
                           y3scalingfactor=100,
                           x2binUpperEdges=[10, 100, 1000, 10000, 100000],
                           x1label=axisLabelRelease,
                           x1limits=(0, roundup(TPDF[str(axisDataName)].max(), 100)),
                           y1limits=(0, roundup(TPDF[str('p_EmissionRate')].max(), 100)),
                           y1label=axisLabelReported,
                           x2label=axisLabelRelease,
                           x2limits=(10, 10000),
                           x2ticks=[10, 100, 1000, 10000],
                           x2ticklabels=[10, 100, 1000, 10000],
                           x2scale='log',
                           y2label='Quantification Error (%)',
                           x3label=axisLabelRelease,
                           x3limits=(10, 10000),
                           x3ticks=[10, 100, 1000, 10000],
                           x3ticklabels=[10, 100, 1000, 10000],
                           x3scale='log',
                           y3label='Quantification Error (%)',
                           cDataField='tc_ExpWindSpeedAvg',
                           clabel='Wind Speed (m/s)',
                           cticks=list(range(0, roundup(TPDF['tc_ExpWindSpeedAvg'].max(), 2), 2)),
                           climits=(0, roundup(TPDF['tc_ExpWindSpeedAvg'].max(), 2)),
                           figsize=(5, 7.5),
                           dpi=400,
                           whiskLegendPosition='best',
                           gridAlpha=0,
                           filePath=outputFilePath,
                           fileName='QuantificationSummary_ReleaseRate_WindSpeed.png')
    except Exception as e:
        print(f'Could not build QuantError_ReleaseRate_Windspeed subplots due to exception: {e}')

    try:
        print("Plotting the Quantification Data with Error Subplots with color bar as methane release rate")
        quantErrorSubplots(df=TPDF,
                           x1DataField='tc_ExpWindSpeedAvg', y1DataField='p_EmissionRate',
                           x2DataField='tc_ExpWindSpeedAvg', y2DataField='tc_mQuantificationAccuracyRel',
                           y3DataField='tc_mQuantificationAccuracyRel',
                           y2scalingfactor=100,
                           y3scalingfactor=100,
                           quantile=yQuantile,
                           regression=False,
                           x2binUpperEdges=[2, 4, 6, 8, 10, 12],
                           x1label='Wind Speed (m/s)',
                           x1limits=(0, roundup(TPDF['tc_ExpWindSpeedAvg'].max(), 2)),
                           y1limits=(0, 10000),
                           y1label=axisLabelReported,
                           x2label='Wind Speed (m/s)',
                           x2limits=(0, roundup(TPDF['tc_ExpWindSpeedAvg'].max(), 2)),
                           x2ticks=list(range(0, roundup(TPDF['tc_ExpWindSpeedAvg'].max(), 2), 2)),
                           x2ticklabels=list(range(0, roundup(TPDF['tc_ExpWindSpeedAvg'].max(), 2), 2)),
                           x2scale='linear',
                           oneToOne=False,
                           y2label='Quantification Error (%)',
                           x3label='Wind Speed (m/s)',
                           x3limits=(0, 14),
                           x3ticks=list(range(0, roundup(TPDF['tc_ExpWindSpeedAvg'].max(), 2), 2)),
                           x3ticklabels=list(range(0, roundup(TPDF['tc_ExpWindSpeedAvg'].max(), 2), 2)),
                           x3scale='linear',
                           y3label='Quantification Error (%)',
                           cDataField=axisDataName,
                           clabel=axisDataName,
                           cticks=[1, 10, 100, 1000, 10000],
                           climits=(1, 10000),
                           cscale='log',
                           figsize=(5, 7.5),
                           dpi=400,
                           showmean=False,
                           gridAlpha=0,
                           whiskLegendPosition='best',
                           filePath=outputFilePath,
                           fileName='QuantificationSummary_WindSpeed_ReleaseRate.png')
    except Exception as e:
        print(f'Could not build QuantError_WIndSpeed_ReleaseRate subplots with markers as release rate due to exception: {e}')

    try:
        print("Plotting the Quantification Accuracy Abs data with whiskerbox subplots")
        quantErrorSubplots(df=TPDF,
                           x1DataField=axisDataName, y1DataField='p_EmissionRate',
                           x2DataField=axisDataName, y2DataField='tc_mQuantificationAccuracyAbs',
                           y3DataField='tc_mQuantificationAccuracyAbs',
                           x2binUpperEdges=[10, 100, 1000, 10000, 100000],
                           x1label=axisLabelRelease,
                           x1limits=(0, roundup(TPDF[str(axisDataName)].max(), 100)),
                           y1limits=(0, roundup(TPDF[str('p_EmissionRate')].max(), 100)),
                           y1label=axisLabelReported,
                           x2label=axisLabelRelease,
                           x2limits=(10, 10000),
                           x2ticks=[10, 100, 1000, 10000],
                           x2ticklabels=[10, 100, 1000, 10000],
                           x2scale='log',
                           quantile=yQuantile,
                           y2label=yLabel,
                           x3label=axisLabelRelease,
                           x3limits=(10, 10000),
                           x3ticks=[10, 100, 1000, 10000],
                           x3ticklabels=[10, 100, 1000, 10000],
                           x3scale='log',
                           y3label=yLabel,
                           cDataField='tc_ExpDurationHrs',
                           clabel='Duration (h)',
                           cticks=list(range(0, roundup(TPDF['tc_ExpDurationHrs'].max(), 1), 1)),
                           climits=(0, roundup(TPDF['tc_ExpDurationHrs'].max(), 1)),
                           figsize=(5, 7.5),
                           dpi=400,
                           gridAlpha=0,
                           whiskLegendPosition='best',
                           filePath=outputFilePath,
                           fileName='QuantificationAccuracyAbsoluteWithWhiskers.png')
                           #cticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    except Exception as e:
        print(f'Could not build Quantification Accuracy Abs data with whiskerbox subplots due to exception: {e}')

    try:
        print("Plotting a subplot of quantification estimation - scatter plot - Linear")
        fig = plt.figure(constrained_layout=True, figsize=(4, 4))
        gs = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(gs[(0, 0)])
        fig, _, varsDict = quantification(DF=TPDF,
                                          xData='tc_C1MassFlow',
                                          yData='p_EmissionRate',
                                          cData='tc_mQuantificationAccuracyRel',
                                          cFactor=100,
                                          xLimits=(0, roundup(TPDF[str(axisDataName)].max(), 100)),
                                          yQuantile=yQuantile,
                                          fontsize=8,
                                          xLabel='Release Rate (g CH4/h)',
                                          yLabel='Reported Rate (g CH4/h)',
                                          rectangularPatch=False,
                                          fig=fig, axes=ax1,
                                          legendInside=False,
                                          paperDict=varsDict,
                                          filePath=outputFilePath,
                                          fileName='quantificationScatter_Linear.png')
        fig.tight_layout()
        plt.close(fig)
    except Exception as e:
        print(f'Plotting a subplot of quantification estimation - scatter plot - Linear: {e}')

    try:
        print("Plotting a subplot of quantification estimation - scatter plot - Log")
        #fig = plt.figure(constrained_layout=True, figsize=(4.2, 3.5))
        fig = plt.figure(constrained_layout=True, figsize=(4, 4))
        gs = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(gs[(0, 0)])
        fig, _, _ = quantification(DF=TPDF,
                                   xData='tc_C1MassFlow',
                                   yData='p_EmissionRate',
                                   cData='tc_mQuantificationAccuracyRel',
                                   cFactor=100,
                                   fontsize=8,
                                   xLabel='Release Rate (g CH4/h)',
                                   yLabel='Reported Rate (g CH4/h)',
                                   rectangularPatch=False,
                                   xScale='log',
                                   yScale='log',
                                   xLimits=(0, 100000),
                                   xTicks=[0, 10, 100, 1000, 10000, 100000],
                                   yLimits=(0, 100000),
                                   yTicks=[0, 10, 100, 1000, 10000, 100000],
                                   fig=fig, axes=ax1,
                                   legendInside=False,
                                   filePath=outputFilePath,
                                   fileName='quantificationScatter_Log.png')
        fig.tight_layout()
        plt.close(fig)
    except Exception as e:
        print(f'Plotting a subplot of quantification estimation - scatter plot - Log: {e}')

    try:
        print("Plotting a subplot of boxplots of quantification estimation - relative error")
        fig = plt.figure(constrained_layout=True, figsize=(4.5, 7))
        gs = fig.add_gridspec(3, 1)
        ax1 = fig.add_subplot(gs[(0, 0)])
        ax2 = fig.add_subplot(gs[(1, 0)])
        ax3 = fig.add_subplot(gs[(2, 0)])
        fig, _, _, varsDict = BoxWhisker(DF=TPDF, xDataField='tc_C1MassFlow', yDataField='tc_mQuantificationAccuracyRel',
                                         xbinUpperEdges=[10, 100, 1000, 10000, 100000], xlabel='Release Rate (g CH4/h)',
                                         xlimits=(10, 10000), xticks=[10, 100, 1000, 10000], xticklabels=[10, 100, 1000, 10000],
                                         xscale='log', ylabel='Quantification Error (%)', yScaleFactor=100, quantile=yQuantile,
                                         fig=fig, axes=ax1, x2label='Sample Count', paperDict=varsDict)
        fig, _, _, _ = BoxWhisker(DF=TPDF, xDataField='tc_ExpWindSpeedAvg', yDataField='tc_mQuantificationAccuracyRel',
                                  xbinUpperEdges=[3, 6, 9, 12, 15], xlabel='Mean wind speed(m/s)', xlimits=(0, 12),
                                  xticks=[0, 3, 6, 9, 12], xticklabels=[0, 3, 6, 9, 12], xscale='linear',
                                  ylabel='Quantification Error (%)', yScaleFactor=100, quantile=yQuantile, fig=fig, axes=ax2)
        fig, _, _, _ = BoxWhisker(DF=TPDF, xDataField='tc_ExpDurationHrs', yDataField='tc_mQuantificationAccuracyRel',
                                  xbinUpperEdges=[3, 6, 9, 12, 15], xlabel='Release duration (h)', xlimits=(0, 12),
                                  xticks=[0, 3, 6, 9, 12], xticklabels=[0, 3, 6, 9, 12], xscale='linear',
                                  ylabel='Quantification Error (%)', yScaleFactor=100, quantile=yQuantile, fig=fig, axes=ax3,
                                  fileName='quantificationBoxplot_Rel.png', filePath=outputFilePath)
    except Exception as e:
        print(f'Plotting a subplot of boxplots of quantification estimation - relative error: {e}')

    try:
        print("Plotting a subplot of boxplots of quantification estimation - absolute error")
        fig2 = plt.figure(constrained_layout=True, figsize=(4.5, 7))
        gs = fig2.add_gridspec(3, 1)
        ax4 = fig2.add_subplot(gs[(0, 0)])
        ax5 = fig2.add_subplot(gs[(1, 0)])
        ax6 = fig2.add_subplot(gs[(2, 0)])
        fig, _, _, _ = BoxWhisker(DF=TPDF, xDataField='tc_C1MassFlow', yDataField='tc_mQuantificationAccuracyAbs',
                                  xbinUpperEdges=[10, 100, 1000, 10000, 100000], xlabel='Release Rate (g CH4/h)',
                                  xlimits=(10, 10000), xticks=[10, 100, 1000, 10000], xticklabels=[10, 100, 1000, 10000],
                                  xscale='log', ylabel='Abs. Quantification Error', quantile=yQuantile,
                                  fig=fig2, axes=ax4, x2label='Sample Count')
        fig, _, _, _ = BoxWhisker(DF=TPDF, xDataField='tc_ExpWindSpeedAvg', yDataField='tc_mQuantificationAccuracyAbs',
                                  xbinUpperEdges=[3, 6, 9, 12, 15], xlabel='Mean wind speed(m/s)', xlimits=(0, 12),
                                  xticks=[0, 3, 6, 9, 12], xticklabels=[0, 3, 6, 9, 12], xscale='linear',
                                  ylabel='Abs. Quantification Error', quantile=yQuantile, fig=fig2, axes=ax5)
        fig, _, _, _ = BoxWhisker(DF=TPDF, xDataField='tc_ExpDurationHrs', yDataField='tc_mQuantificationAccuracyAbs',
                                  xbinUpperEdges=[3, 6, 9, 12, 15], xlabel='Release duration (h)', xlimits=(0, 12),
                                  xticks=[0, 3, 6, 9, 12], xticklabels=[0, 3, 6, 9, 12], xscale='linear',
                                  ylabel='Abs. Quantification Error', quantile=yQuantile, fig=fig2, axes=ax6,
                                  fileName='quantificationBoxplot_Abs.png', filePath=outputFilePath)
    except Exception as e:
        print(f'Plotting a subplot of boxplots of quantification estimation - absolute error: {e}')

    return varsDict

def saveLogo(outputFilePath):
    im1 = Image.open(r"METEC Logo.png")
    # save a image using extension
    path = os.path.join(outputFilePath, 'METEC Logo.png')
    im1.save(path)
    return

