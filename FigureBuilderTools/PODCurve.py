import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
from matplotlib.font_manager import FontProperties
from MetricsTools.PODCurveWithEvenCounts import calcPODCurveWEvenCounts
from MetricsTools.PODCurveWithNBins import calcPODCurveWNBins

def buildPODCurve(classifiedDF, TPLevels, xLabel, xCategory='EPBFE', nxBins=6, outputFilePath=None,
                  method=None, BinMethod='EvenCount', fig=None, figsize=(3.54331/1.1, 3.54331/1.1), axes=None,
                  xMax=None, fontsize=7, xScaleFactor=1, yMax=None, xunit='slpm', fileName=None,
                  legendLOC='upper right'):
    """
    :param name: name added to file name to differentiate between even counts and n bins
    :param xLabel: x axis label
    :param podCurveDF: Probability of Detection dataframe
    :param outputFilePath: Output file path for figure
    :return: None
    Possible xParameters: {xCategory}EdgeLower, {xCategory}EdgeUpper, {xCategory}Center, {xCategory}Avg,
                          {xCategory}Count: where xCategory is the category POD was calculated
    """

    # todo: Chiemezie - add scaling factor and units args. scale x data, concatenate strings for xlabel and units
    # todo: Use method triggger to determine if to use POD by even count or by NBins

    if BinMethod =='EvenCount':
        classifiedDF, podCurveDF = calcPODCurveWEvenCounts(classifiedDF, xCategory, nxBins, TPLevels, xunit, xScaleFactor)
    elif BinMethod == 'NBins':
        classifiedDF, podCurveDF = calcPODCurveWNBins(classifiedDF, xCategory, nxBins, TPLevels)
    else:
        print("arguement method not recognized for buildPODCurve.  Please use EvenCount or NBins")
        return None, None, None

    try:
        xLower = []
        xUpper = []
        yUpper = []
        yLower = []
        yUpperUnit = []
        yLowerUnit = []
        yUpperGroup = []
        yLowerGroup = []
        x = []
        y = []
        yUnit = []
        yGroup = []
        bootStrappingMatrix = []

        for index, line in podCurveDF.iterrows():
            xLower.append(float(line['binAvg']) - float(line['binInterval'].left))
            xUpper.append(float(line['binInterval'].right - float(line['binAvg'])))
            x.append(float(line['binAvg']))
            y.append(float(line['pod']))
            yUpper.append(line['posError'])
            yLower.append(line['negError'])
            yUpperUnit.append(line['posErrorUnit'])
            yLowerUnit.append(line['negErrorUnit'])
            yUpperGroup.append(line['posErrorGroup'])
            yLowerGroup.append(line['negErrorGroup'])
            yUnit.append(float(line['podUnit']))
            yGroup.append(float(line['podGroup']))
            bootStrappingMatrix.append(line['dataFacility'])

        bootStrappingMatrix = np.array(bootStrappingMatrix)
        bootStrappingMatrix = np.transpose(bootStrappingMatrix)
        # fig, ax1 = plt.subplots()
        # plt.grid(axis='both', alpha=0.5)
        plt.rcParams.update({'font.size': fontsize})
        if fig is None:
            if figsize:
                fig, ax1 = plt.subplots(figsize=figsize)
            else:
                fig, ax1 = plt.subplots()
        elif axes:
            ax1 = axes
            fig = fig
        else:
            ax1 = plt.gca()
        plt.grid(axis='both', alpha=0.3)

        if xMax:
            ax1.set_xlim([0, xMax])

        ax1.errorbar(x=x, y=y, xerr=[xLower, xUpper], yerr=[yLower, yUpper], marker='s', ls='--',
                     label='Equip. Unit+Group+Facility')
        ax1.errorbar(x=x, y=yGroup, xerr=[xLower, xUpper], yerr=[yLowerGroup, yUpperGroup], marker='d', ls=':',
                     label='Equipment Unit+Group')
        ax1.errorbar(x=x, y=yUnit, xerr=[xLower, xUpper], yerr=[yLowerUnit, yUpperUnit], marker='o', ls='-',
                     label='Equipment Unit')

        ax1.set_ylim([0, 1])
        xMin, xMax = ax1.get_xlim()
        plt.xlabel(xLabel + " " + xunit, fontsize=fontsize)
        plt.ylabel('Probability of Detection', fontsize=fontsize)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Bin Count')
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        if yMax:
            maxValue = yMax
        else:
            maxValue = podCurveDF['binN'].max()

        ax2.plot(podCurveDF['binAvg'], podCurveDF['binN'], 'x', label='Count')
        value = 5 * round(float(maxValue) / 5)

        # Graph bootstrapping regression

        for row in bootStrappingMatrix:
            if method == 'sigmoid':
                xdata = [0] + x
                ydata = [0] + list(row)

                p0 = [np.median(xdata), 0.1]  # this is an mandatory initial guess
                popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, bounds=(0, [xMax, 1]), method='trf')
                xData = np.linspace(0, xMax, 1000)
                yData = sigmoid(xData, *popt)
                ax1.plot(xData, yData, alpha=0.05, color="Grey")

            elif method == 'linear':
                xdata = x
                ydata = row

                m = findSlope(xdata, ydata)
                b = findIntercept(xdata, ydata, m)

                xdata, ydata = linear(0, xMax, m, b)
                ax1.plot(xdata, ydata, alpha=0.05, color="Grey")

            elif method == 'log':
                xdata = x
                ydata = row
                coefficients = np.polyfit(np.log(xdata), ydata, 1)
                xdata, ydata = log(0, xMax, coefficients)
                ax1.plot(xdata, ydata, alpha=0.05, color="Grey")

        if value < maxValue:
            value += 5

        ax2.set_ylim([0, value])
        ax2.set_xlim(0)

        fontP = FontProperties()
        fontP.set_size(fontsize)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        ## Combine the legend handles and labels to make a new legend object
        handles = handles1 + handles2
        labels = labels1 + labels2
        leg = ax1.legend(handles=handles, labels=labels, loc=legendLOC, prop=fontP)

        leg.get_frame().set_edgecolor('black')

        # save
        if fileName:
            if outputFilePath:
                path = os.path.join(outputFilePath, fileName)
            else:
                path = os.path.join(os.getcwd(), fileName)
            print("saving: " + path)
            plt.savefig(path, dpi=400)

        return {'xData': x, 'yData': [y, yGroup, yUnit], 'xErrorData': [xLower, xUpper],
                'yErrorData': [yLower, yUpper, yLowerGroup, yUpperGroup, yLowerUnit, yUpperUnit], 'binCount': list(podCurveDF['binN'])}, ax1, ax2
    except Exception as e:
        print(f'Could not generate POD curve due to exception: {e}')
        return None, None, None

def modifiedbuildPODCurve(podCurveDF, xLabel, outputFilePath=None, name=None, method=None,
                        fig=None, axes=None, figsize=None, xMax=None, xMin=None, yMax=None, yMin=None, addFilename=None):
    """
    :param name: name added to file name to differentiate between even counts and n bins
    :param xLabel: x axis label
    :param podCurveDF: Probability of Detection dataframe
    :param outputFilePath: Output file path for figure
    :return: None
    Possible xParameters: {xCategory}EdgeLower, {xCategory}EdgeUpper, {xCategory}Center, {xCategory}Avg,
                          {xCategory}Count: where xCategory is the category POD was calculated
    """
    # todo: Check why they are one less bins for nBins
    #
    try:
        xLower = []
        xUpper = []
        yUpper = []
        yLower = []
        yUpperUnit = []
        yLowerUnit = []
        yUpperGroup = []
        yLowerGroup = []
        x = []
        y = []
        yUnit = []
        yGroup = []
        bootStrappingMatrix = []

        for index, line in podCurveDF.iterrows():
            xLower.append(float(line['binAvg']) - float(line['binInterval'].left))
            xUpper.append(float(line['binInterval'].right - float(line['binAvg'])))
            x.append(float(line['binAvg']))
            y.append(float(line['pod']))
            yUpper.append(line['posError'])
            yLower.append(line['negError'])
            yUpperUnit.append(line['posErrorUnit'])
            yLowerUnit.append(line['negErrorUnit'])
            yUpperGroup.append(line['posErrorGroup'])
            yLowerGroup.append(line['negErrorGroup'])
            yUnit.append(float(line['podUnit']))
            yGroup.append(float(line['podGroup']))
            bootStrappingMatrix.append(line['dataFacility'])

        bootStrappingMatrix = np.array(bootStrappingMatrix)
        bootStrappingMatrix = np.transpose(bootStrappingMatrix)

        #fig, ax1 = plt.subplots()
        plt.rcParams.update({'font.size': 7})
        if fig is None:
            if figsize:
                fig, ax1 = plt.subplots(figsize=figsize)
            else:
                fig, ax1 = plt.subplots()
        elif axes:
            ax1 = axes
            fig = fig
        else:
            ax1 = plt.gca()
        plt.grid(axis='both', alpha=0.5)
        #ax1.grid(axis='both', alpha=0.5)
        if xMax:
            ax1.set_xlim([0, xMax])

        ax1.errorbar(x=x, y=y, xerr=[xLower, xUpper], yerr=[yLower, yUpper], marker='s', ls='--', label='Facility')
        ax1.errorbar(x=x, y=yGroup, xerr=[xLower, xUpper], yerr=[yLowerGroup, yUpperGroup], marker='d', ls=':',
                     label='Equipment Group')
        ax1.errorbar(x=x, y=yUnit, xerr=[xLower, xUpper], yerr=[yLowerUnit, yUpperUnit], marker='o', ls='-',
                     label='Equipment Unit')

        ax1.set_ylim([0, 1])
        xMin, xMax = ax1.get_xlim()
        ax1.set_xlabel(xLabel)
        ax1.set_ylabel('Probability of Detection')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Bin Count')
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        if yMax:
            maxValue = yMax
        else:
            maxValue = podCurveDF['binN'].max()
        ax2.plot(podCurveDF['binAvg'], podCurveDF['binN'], 'x', label='Count')
        value = 5 * round(float(maxValue) / 5)

        # Graph bootstrapping regression
        # Todo: Add logistic Regression method. Plot all detections as either 0 or 1 (FN or TP) vs emission rate or
        #  something and fit a curve (sigmoid like) to the plot
        # Todo: Move into it's own script
        for row in bootStrappingMatrix:
            if method == 'sigmoid':
                xdata = [0] + x
                ydata = [0] + list(row)

                p0 = [np.median(xdata), 0.1]  # this is an mandatory initial guess
                popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, bounds=(0, [xMax, 1]), method='trf')
                xData = np.linspace(0, xMax, 1000)
                yData = sigmoid(xData, *popt)
                ax1.plot(xData, yData, alpha=0.05, color="Grey")

            elif method == 'linear':
                xdata = x
                ydata = row

                m = findSlope(xdata, ydata)
                b = findIntercept(xdata, ydata, m)

                xdata, ydata = linear(0, xMax, m, b)
                ax1.plot(xdata, ydata, alpha=0.05, color="Grey")

            elif method == 'log':
                xdata = x
                ydata = row
                coefficients = np.polyfit(np.log(xdata), ydata, 1)
                xdata, ydata = log(0, xMax, coefficients)
                ax1.plot(xdata, ydata, alpha=0.05, color="Grey")

        if value < maxValue:
            value += 5

        ax2.set_ylim([0, value])
        ax2.set_xlim(0)

        ax1.legend(loc='upper center', ncol=4)
        plt.tight_layout()

        if name and outputFilePath:
            if addFilename:
                filename = addFilename + "podCurve" + name + ".png"
            else:
                filename = "podCurve" + name + ".png"
            path = os.path.join(outputFilePath, filename)
            plt.savefig(path)
        return {'xData': x, 'yData': [y, yGroup, yUnit], 'xErrorData': [xLower, xUpper],
                'yErrorData': [yLower, yUpper, yLowerGroup, yUpperGroup, yLowerUnit, yUpperUnit], 'binCount': list(podCurveDF['binN'])}, ax1, ax2, fig
    except Exception as e:
        print(f'Could not generate POD curve due to exception: {e}')
        return None, None, None



def sigmoid(X, x0, k):
    Y = 1 / (1 + np.exp(-k * (X - x0)))
    return Y


def findSlope(x, y):
    n = len(x)
    m = ((n * sum(x * y)) - (sum(y) * sum(x))) / ((n * sum([xi ** 2 for xi in x])) - (sum(x) ** 2))
    return m


def findIntercept(x, y, m):
    n = len(x)
    b = (sum(y) - m * sum(x)) / n
    return b


def linear(xMin, xMax, m, b):
    y = []
    x = np.linspace(xMin, xMax, 1000)
    for xi in x:
        y.append(m * xi + b)
    return x, y


def log(xMin, xMax, coefficients):
    A = coefficients[0]
    B = coefficients[1]
    x = np.linspace(xMin, xMax, 1000)
    y = B + A * np.log(x)
    return x, y
