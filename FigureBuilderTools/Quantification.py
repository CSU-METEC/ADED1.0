import os
import math
import itertools
import statistics
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def roundup(x,n):
    return int(math.ceil(x / float(n))) * n

def rounddown(x,n):
    return int(math.floor(x / float(n))) * n

def ols_slope(xdata, ydata):
    X = np.vstack([xdata]).T
    slope = np.linalg.lstsq(X, ydata)[0]
    fit = slope * xdata
    rsquared = 1 - np.sum((ydata - fit) ** 2) / np.sum((ydata - np.mean(ydata)) ** 2)
    return slope[0], rsquared

def relativeErrorBounds(x=None, f=None, bound=None):
    if bound == 'upper':
        RE = ((f*x)-x)*100/x
    elif bound == 'lower':
        RE = ((x/f)-x)*100/x
    return RE

def factorsFromrelativeErrors(x=None, RE=None):
    if RE > 0:
        f = ((x*RE/100)+x)/x
    elif RE < 0:
        f = -1*(x/((x*RE/100)+x))
    else:
        f = 1
    return f

def unique(list1):
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    return unique_list

def roundToNearestInteger(x):
    diff = x - int(x)
    if diff >= 0.5:
        y = int(math.ceil(x / float(1))) * 1
    else:
        y = int(math.floor(x / float(1))) * 1
    return y

def createLabel(item):
    """Convert factor ticks into label depending on value of bin.
                 :param item = The bin value to convert to label."""
    if item in [-1, 1, 0]:
        label = str(abs(item))
    elif item < 0:
        label = str(1) + "/" + str(abs(item))
    else:
        label = str(item)
    return label

# def relativeErrorBounds(f=None, bound=None):
#     if bound == 'upper':
#         RE = (f-1)*100
#     elif bound == 'lower':
#         RE = ((1/f)-1)*100
#     return RE


def inverseRelativeError(x, RE):
    y = (x*RE/100) + x
    return y


def quantification(DF=None, xData=None, yData=None, cData=None, xFactor=1, yFactor=1, cFactor=1, xLimits=None,
                   yLimits=None, xScale='linear', yScale='linear', xLabel=None, yLabel=None, xTicks=None, yTicks=None,
                   xunit=None, yunit=None, qFactor=(2, 3, 5), regression='linear', oneToOne=True, yQuantile=None, s=25,
                   axesEdgeColor='black', figsize=(4, 4), dpi=400, fig=None, axes=None, fontsize=7,
                   filePath=None, fileName=None, varPrefix="", paperDict=None, rectangularPatch=True, xrect=None,
                   yrect=None, axesEdgeWidth=1, percentiles=None, legendInside=True):
    """Generate a scatterplot showing specified bins for data in df.
     Required Parameters:
     :param df = dataframe to pull data from.
     :param xData = data to plot on the x-axis
     :param yData = data to plot on the y-axis
     :param cData = data to use during binning
     Optional parameters:
     :param xLabel = label for x axis
     :param yLabel = limits of y-axis
     :param xScale = scale for x axis ("linear" or "log")
     :param yScale = scale for y axis ("linear" or "log")
     :param xLimits = limits of x-axis
     :param yLimits = limits of y-axis
     :param xTicks = location of major tick marks for x-axis
     :param yTicks = location of major tick marks for y-axis
     :param oneToOne = if True, draws a 1:1 line
     :param regression = plots a regression line
     :param yFactor = A multiplication factor for yData
     :param qFactor = Specifies the quantification factor to be shown
     :param s = specifies the size of data points
     :param gridAlpha = specifies the thickness of the grid lines
     :param figsize = (width, height) of figure in inches
     :param dpi = specifies image quality
     :param fig = the figure on which the plot is to be made
     :param axes = the axes within the figure on which the plot is made.
     :param filePath = directory to save figure to. Defaults to cwd.
     :param fileName = if specified saves figure to filePath/fileName
     :param varPrefix = specifies the prefix of a dictionary key name
     :param paperDict = dictionary of variables to be used in the paper
     :param yQuantile = Specifies a quantile
     :param rectangularPatch = if true, adds a box to the portion of the plot to blow-up
     :param xrect = Specifies the x-coordinates of the rectangularPatch
     :param yrect = Specifies the y-coordinates of the rectangularPatch
     :param axesEdgeWidth = Specifies the edge width of an axes
     """

    # Check if the column highlighted below contains any real value
    valueCheck = DF[xData]
    if all(valueCheck == 0) or valueCheck.isnull().all():
        return None

    valueCheck = DF[yData]
    if all(valueCheck == 0) or valueCheck.isnull().all():
        return None

    # Figure setup
    if fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    elif axes:
        ax = axes
        fig = fig
    else:
        ax = plt.gca()

    # Altering axes edges
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color(axesEdgeColor)
        ax.spines[axis].set_linewidth(axesEdgeWidth)

    # If percentiles are given, filter the DF using the percentiles provided
    if percentiles is None:
        # Drop NA values
        df = DF.dropna(axis=0, how='any', subset=[xData, yData], inplace=False).copy(deep=True)
    else:
        # Drop NA values
        WDF = DF.dropna(axis=0, how='any', subset=[xData, yData], inplace=False).copy(deep=True)
        lwr = rounddown(np.percentile(a=np.array(WDF[cData].tolist()), q=percentiles[0]), 1)
        hgr = roundup(np.percentile(a=np.array(WDF[cData].tolist()), q=percentiles[1]), 1)
        df = WDF.loc[(WDF[cData] >= lwr) & (WDF[cData] <= hgr)]

    # Scale data with factors
    df[xData] = df[xData].apply(lambda x: x * xFactor)
    df[yData] = df[yData].apply(lambda y: y * yFactor)
    df[cData] = df[cData].apply(lambda c: c * cFactor)

    # filter out the xData and yData to be plotted
    filt = (~df[xData].isna()) & ~(df[yData].isna())
    x = df.loc[filt, xData]
    y = df.loc[filt, yData]
    yMin = y.min()
    yMax = y.max()
    xMin = x.min()
    xMax = x.max()

    # plot 1:1 line
    if oneToOne:
        #pt1 = (xMin, xMax)
        pt1 = (0, xMax)
        ax.plot(pt1, pt1, label="1:1", color='peru', linewidth=1.5, ls='-')

    # fit data series and plot
    if regression == 'linear':
        # calculate regression
        p, R2 = ols_slope(x.astype("float"), y.astype("float"))
        x_fit = np.linspace(min(x), max(x))
        y_fit = x_fit * p
        ax.plot(x_fit, y_fit, label=r"${p:.3f}*x, R^2$ = {r:.2f}".format(p=p, r=R2), color='cyan', linewidth=1.5,
                ls='-')

    # plot x1, y1 scatter
    ax.scatter(x, y, s=s, edgecolors='black', label='Data')
    # Generate a dictionary with quantification factor as key and relative error equivalent as values
    colors = ['yellow', 'blue', 'red', 'grey', 'cyan', 'violet', 'lime', 'green', 'aqua', 'pink', 'navy']
    lineStyle = ['--', '-.', ':', '-']

    counter = 0
    # iterate through the factors supplied in qFactor
    for i in qFactor:
        # Find the lower and upper relative errors corresponding to the quantification factor; x.
        lB = relativeErrorBounds(x=xMax, f=i, bound='lower')
        uB = relativeErrorBounds(x=xMax, f=i, bound='upper')
        ER_bounds = [round(lB, 2), round(uB, 2)]

        # Filter out all data points with relative errors within the bounds; ER_bounds
        filter = (df[cData] >= ER_bounds[0]) & (df[cData] <= ER_bounds[1])
        thisDF = df.loc[filter]
        percentPoints = int(len(thisDF) * 100 / len(df))
        try:
            print("Plotting all the data within a region of quantification factor")
            # Find the xData range for thisDF
            xxMin = thisDF[xData].min()
            xxMax = thisDF[xData].max()
            # Plots the lower and upper
            y1Upper = int(inverseRelativeError(xxMin, ER_bounds[1]))
            y2Upper = int(inverseRelativeError(xxMax, ER_bounds[1]))
            y1Lower = int(inverseRelativeError(xxMin, ER_bounds[0]))
            y2Lower = int(inverseRelativeError(xxMax, ER_bounds[0]))

            ax.plot((0, xMax), (0, y2Upper), color='k', linestyle=lineStyle[counter], linewidth=2,
                    label=f'{percentPoints}% within a factor of {i}')
            # plot the upper bound of the quantification estimate for the factor
            ax.plot((0, xMax), (0, y2Lower), color='k', linestyle=lineStyle[counter], linewidth=2)
        except Exception as e:
            print(f"could not all the data within a region of quantification factor due to {e}")
        # Append to paper dictionary
        if paperDict:
            paperDict[f'{varPrefix}PercentageWithinFactor{i}'] = percentPoints
            paperDict[f'{varPrefix}quantFactor{i}'] = i
        counter = counter + 1

    # Add a rectangular patch to highlight a region.
    if rectangularPatch:
        # Create a Rectangle
        width = xrect[1] - xrect[0]
        height = yrect[1] - yrect[0]
        rect = patches.Rectangle((0, 0), width=width, height=height, linewidth=3, edgecolor='violet', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    # format x-axis based on method args
    ax.set_xscale(xScale)
    if xTicks:
        ax.set_xticks(xTicks)
    if xLimits:
        ax.set_xlim(xLimits)
    if xLabel and xunit:
        ax.set_xlabel(xLabel + " (" + xunit + ")", fontsize=fontsize)
    elif xLabel:
        ax.set_xlabel(xLabel, fontsize=fontsize)

    # format y-axis based on method args
    ax.set_yscale(yScale)
    if yTicks:
        ax.set_yticks(yTicks)
    if yLimits:
        ax.set_ylim(yLimits)
    elif yQuantile:
        ax.set_ylim(0, int(y.quantile(yQuantile)))
    else:
        ax.set_ylim(int(yMin), int(yMax))
    if yLabel and yunit:
        ax.set_ylabel(yLabel + " (" + yunit + ")", fontsize=fontsize)
    elif yLabel:
        ax.set_ylabel(yLabel, fontsize=fontsize)

    fontP = FontProperties()
    fontP.set_size(fontsize)
    if percentiles is not None:
        if legendInside:
            leg = ax.legend(ncol=2, loc='upper right', prop=fontP,
                            title=f'Data within {percentiles[0]} and {percentiles[1]} percentile')
        else:
            leg = ax.legend(ncol=2, loc="lower left", prop=fontP, bbox_to_anchor=(0, 1.02, 1, 0.08), borderaxespad=0,
                            mode="expand", title=f'Data within {percentiles[0]} and {percentiles[1]} percentile')
    elif yQuantile is not None:
        if legendInside:
            leg = ax.legend(ncol=2, loc='upper right', prop=fontP,
                            title=f'All data considered with upper y-axis trimmed at {yQuantile * 100} percentile')
        else:
            leg = ax.legend(ncol=2, loc="lower left", prop=fontP, bbox_to_anchor=(0, 1.02, 1, 0.08), borderaxespad=0,
                            mode="expand",
                            title=f'All data considered with upper y-axis trimmed at {yQuantile * 100} percentile')
    else:
        if legendInside:
            leg = ax.legend(ncol=2, loc='upper right', prop=fontP, title=f'All data considered in the analysis')
        else:
            leg = ax.legend(ncol=2, loc="lower left", prop=fontP, title=f'All data considered in the analysis',
                            bbox_to_anchor=(0, 1.02, 1, 0.08), borderaxespad=0, mode="expand")
    leg.get_frame().set_edgecolor('black')

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=dpi)

    return fig, ax, paperDict


def quantificationHist(DF=None, xData=None, xFactor=100, xLimits=None, xunit="%", Bin1=None, xQuantile=None,
                       xLabel="Quantification relative error", yLabel="Number of measurements", xTicks=None, yTicks=None,
                       rotation=45, gridAlpha=0.01, figsize=(3.54331, 3.54331/1.5), percentiles=None, insertAxes=False,
                       dpi=400, fig=None, axes=None, fontsize=7, filePath=None, fileName=None,
                       vertLines=((-50.00, 100.00, 2), (-66.67, 200.00, 3), (-80.00, 400.00, 5)), showLegend=True):

    """Generate a scatterplot showing specified bins for data in df.
         Required Parameters:
         :param df = dataframe to pull data from.
         :param xData = data to plot on the x-axis
         Optional parameters:
         :param xLabel = label for x axis
         :param yLabel = limits of y-axis
         :param xLimits = limits of x-axis
         :param yLimits = limits of y-axis
         :param xTicks = location of major tick marks for x-axis
         :param yTicks = location of major tick marks for y-axis
         :param xFactor = A multiplication factor for xData
         :param gridAlpha = specifies the thickness of the grid lines
         :param figsize = (width, height) of figure in inches
         :param dpi = specifies image quality
         :param fig = the figure on which the plot is to be made
         :param axes = the axes within the figure on which the plot is made.
         :param filePath = directory to save figure to. Defaults to cwd.
         :param fileName = if specified saves figure to filePath/fileName
         :param varPrefix = specifies the prefix of a dictionary key name
         :param paperDict = dictionary of variables to be used in the paper
         :param rotation = specifies the angular inclination of xTicks label
         :param percentiles = specifies the upper and lower percentile of the total data to be used in the analysis
         :param insertAxes = Specifies if the plot should include an axes in an existing plot.
         :param vertLines = Specifies the lower(LE) and upper(UE) error bounds with a factor f to show: (LE, UE, f)
         :param showLegend = If true, this shows the legend inside the plot"""

    # Check if the column highlighted below contains any real value
    valueCheck = DF[xData]
    if all(valueCheck == 0) or valueCheck.isnull().all():
        return None

    # Figure setup
    if fig is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    elif axes:
        ax1 = axes
        fig = fig
    else:
        ax1 = plt.gca()

    # Drop NA values
    WDF = DF.dropna(axis=0, how='any', subset=[xData], inplace=False).copy(deep=True)

    # Scale data with factors
    WDF[xData] = WDF[xData].apply(lambda c: c * xFactor)
    if xQuantile:
        yUpperATquantile = WDF[xData].quantile(xQuantile)

    # If percentiles are given, filter the DF using the percentiles provided
    lwr = rounddown(np.percentile(a=np.array(WDF[xData].tolist()), q=percentiles[0]), 1)
    hgr = roundup(np.percentile(a=np.array(WDF[xData].tolist()), q=percentiles[1]), 1)
    df = WDF.loc[(WDF[xData] >= lwr) & (WDF[xData] <= hgr)]

    # Make a histogram plot.
    if Bin1 is None:
        minRE = rounddown(df[xData].min(), 1)
        maxRE = roundup(df[xData].max(), 1)
        # Calculate the factors associated with minRE and maxRE using an arbitrary x-value: 100
        upperREFactors = list(range(2, int(roundup(abs(factorsFromrelativeErrors(x=100, RE=maxRE)), 1)) + 1, 1))
        lowerREFactors = list(range(1, int(roundup(abs(factorsFromrelativeErrors(x=100, RE=minRE)), 1)) + 1, 1))
        upperREbound = [round(relativeErrorBounds(x=100, f=f, bound='upper'), 2) for f in upperREFactors]
        lowerREbound = [round(relativeErrorBounds(x=100, f=f, bound='lower'), 2) for f in lowerREFactors]
        Bin = list(itertools.chain.from_iterable([lowerREbound, upperREbound]))
        # Make sure the list only contains unique values
        Bin1 = unique(Bin)
        Bin1.sort()

    # Plot histogram
    n, _, patches = ax1.hist(x=df[xData].tolist(), bins=Bin1, edgecolor='k', color='khaki')

    # Get the factors corresponding to Bin1
    Bin2 = []
    for re in Bin1:
        Bin2.append(roundToNearestInteger(factorsFromrelativeErrors(x=100, RE=re)))

    # To get the axis tick labels for ax1 and ax2
    counter = 0
    x1TickLabel = []
    x2TickLabel = []
    maxBin = Bin1[-1]
    for i in Bin1:
        if i < 0:
            if Bin2[counter] in [2, 5]:
                x1TickLabel.append(i)
                x2TickLabel.append(Bin2[counter])
            else:
                x1TickLabel.append("")
                x2TickLabel.append("")
        else:
            if maxBin > 2000 and ((i/100) % 2 == 0):
                x1TickLabel.append("")
                x2TickLabel.append("")
            else:
                x1TickLabel.append(i)
                x2TickLabel.append(Bin2[counter])
        counter = counter + 1

    # Plot vertical lines and calculate the percentage of measurements within an errorRange
    lineStyles = ['--', '-.', ':', '-']
    i = 0
    for errorRange in vertLines:
        filter1 = (df[xData] >= errorRange[0]) & (df[xData] <= errorRange[1])
        thisDF = df.loc[filter1]
        percentPoints = int(len(thisDF) * 100 / len(df))
        if errorRange[0] >= rounddown(df[xData].min(), 10):
            ax1.axvline(errorRange[0], color='k', ls=lineStyles[i], linewidth=1)
        if errorRange[1] <= roundup(df[xData].max(), 100):
            ax1.axvline(errorRange[1], color='k', ls=lineStyles[i], linewidth=1,
                        label=f'{percentPoints}% by a factor of {errorRange[2]}')
        i = i + 1

    # Show the mean and median quantification error line
    mean = round(statistics.mean(df[xData].tolist()), 2)
    median = round(statistics.median(df[xData].tolist()), 2)

    # Calculate the factor associated with the calculated mean and median
    meanLineLabel = f'{factorsFromrelativeErrors(x=100, RE=mean):.2f}'
    medianLineLabel = f'{factorsFromrelativeErrors(x=100, RE=median):.2f}'

    # Plot the mean and median vertical lines
    ax1.axvline(mean, color='blue', ls='-', linewidth=1, label=f'Mean: {mean}% ({meanLineLabel})')
    ax1.axvline(median, color='crimson', ls='-', linewidth=1, label=f'Median: {median}% ({medianLineLabel})')

    # Get the index of bars with factors greater or equal to 1
    colorBarIndex = [Bin1.index(r) for r in Bin1 if r >= 0]
    # Coloring some parts of the bars
    for i in colorBarIndex:
        if i == len(Bin1)-1:
            # This is because there is no bar after the last bin
            break
        else:
            patches[i].set_facecolor('olive')

    # Insert Axis
    if insertAxes:
        # Check if there is any relative error less than 0
        if df.loc[df[xData] < 0].empty:
            pass
        else:
            # x-axis bin of the axes insert
            neg_bin1 = list(range(rounddown(df[xData].min(), 10), 1, 10))
            # x-axis bin of the secondary axes insert
            neg_bin2 = []
            # Define location of axes insert
            axins1 = ax1.inset_axes([0.5, 0.5, 0.49, 0.48])
            # Plot histogram
            n1, _, _ = axins1.hist(x=df[xData].tolist(), bins=neg_bin1, edgecolor='k', color='khaki')
            # Define x-axis limit of axes insert
            axins1.set_xlim(-102, 2)
            # Add vertical lines at specified x-axis
            i = 0
            for errorRange in vertLines:
                if errorRange[0] >= rounddown(df[xData].min(), 10):
                    axins1.axvline(errorRange[0], color='k', ls=lineStyles[i], linewidth=1.5)
                i = i + 1
            for re in neg_bin1:
                if re == -100:
                    neg_bin2.append("")
                else:
                    neg_bin2.append(round(factorsFromrelativeErrors(x=100, RE=re), 2))
            if median < 0:
                axins1.axvline(median, color='crimson', ls='-', linewidth=1)
            if mean < 0:
                axins1.axvline(mean, color='blue', ls='-', linewidth=1)
            # Add the factors corresponding to each bin.
            j = 0
            for bin in neg_bin1:
                axins1.text(bin - 4.5, 0.1 * max(n1), f'{neg_bin2[j]}', rotation=90, verticalalignment='center',
                            fontsize=10, alpha=0.7, fontweight='medium')
                j = j + 1

    # Set ax1 axis parameters
    if xTicks is None:
        xTicks = Bin1
    ax1.set_xticks(xTicks)
    if yTicks is None:
        yTicks = list(range(0, roundup(max(list(n)), 10) + 1, 10))
    ax1.set_yticks(yTicks)
    if xLabel:
        ax1.set_xlabel(f'{xLabel} ({xunit})', fontsize=fontsize)
    if yLabel:
        ax1.set_ylabel(yLabel, fontsize=fontsize)
    if x1TickLabel:
        ax1.set_xticklabels(x1TickLabel, rotation=rotation, fontsize=fontsize)
    if xLimits is None:
        if xQuantile is None:
            xLimits = [Bin1[0] - 5, Bin1[-1] + 5]
        else:
            xLimits = [Bin1[0] - 5,  yUpperATquantile + 5]
    ax1.set_xlim(xLimits)

    # Characterize axis 2
    ax2 = ax1.twiny()
    ax2.set_xticks(Bin1)
    ax2.set_xlabel("Factor difference", fontsize=fontsize)
    if x2TickLabel:
        ax2.set_xticklabels(x2TickLabel, rotation=rotation, fontsize=fontsize)
    ax2.set_xlim(xLimits)

    # Remove ticks
    ax1.tick_params(axis='both', bottom=False, top=False, left=True, right=False)
    ax2.tick_params(axis='both', bottom=False, top=False, left=False, right=True)

    if showLegend:
        fontP = FontProperties()
        fontP.set_size(fontsize)
        if xQuantile is None:
            if insertAxes:
                lg = ax1.legend(loc='best', prop=fontP, ncol=2, bbox_to_anchor=(0.5, 0.45), borderaxespad=0.,
                                title=f'Data within {percentiles[0]}% and {percentiles[1]}% percentiles')
            else:
                lg = ax1.legend(loc='best', prop=fontP, ncol=2,
                                title=f'Data within {percentiles[0]}% and {percentiles[1]}% percentiles')
        else:
            if insertAxes:
                lg = ax1.legend(loc='best', prop=fontP, ncol=2, bbox_to_anchor=(0.5, 0.45), borderaxespad=0.,
                                title=f'Data within {percentiles[0]}% and {percentiles[1]}% percentiles. \nUpper x-axis trimmed at {xQuantile*100}% percentile')
            else:
                lg = ax1.legend(loc='best', prop=fontP, ncol=2,
                                title=f'Data within {percentiles[0]}% and {percentiles[1]}% percentiles. \nUpper x-axis trimmed at {xQuantile*100}% percentile')
        lg.get_frame().set_edgecolor('black')

    ax1.grid(axis='x', alpha=gridAlpha)
    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=dpi)

    return fig, ax1

def quantificationHistInv(DF=None, xData=None, cData=None, cFactor=100, xLimits=None, xunit="%", Bin1=None, xQuantile=None,
                          xLabel="Factor Difference", yLabel="Number of measurements", xTicks=None, yTicks=None,
                          rotation=45, gridAlpha=0.01, figsize=(3.54331, 3.54331/1.5), showLegend=True,
                          percentiles=None, dpi=400, fig=None, axes=None, fontsize=7, filePath=None,
                          fileName=None, varPrefix=None, paperDict=None, vertLines=(2,3,5)):

    """Generate a scatterplot showing specified bins for data in df.
         Required Parameters:
         :param df = dataframe to pull data from.
         :param xData = data to be used to generate x-axis data
         :param cData = data to be used to generate x-axis data
         Optional parameters:
         :param xLabel = label for x axis
         :param yLabel = limits of y-axis
         :param xLimits = limits of x-axis
         :param yLimits = limits of y-axis
         :param xTicks = location of major tick marks for x-axis
         :param yTicks = location of major tick marks for y-axis
         :param xFactor = A multiplication factor for xData
         :param gridAlpha = specifies the thickness of the grid lines
         :param figsize = (width, height) of figure in inches
         :param dpi = specifies image quality
         :param fig = the figure on which the plot is to be made
         :param axes = the axes within the figure on which the plot is made.
         :param filePath = directory to save figure to. Defaults to cwd.
         :param fileName = if specified saves figure to filePath/fileName
         :param varPrefix = specifies the prefix of a dictionary key name
         :param paperDict = dictionary of variables to be used in the paper
         :param rotation = specifies the angular inclination of xTicks label
         :param percentiles = specifies the upper and lower percentile of the total data to be used in the analysis
         :param insertAxes = Specifies if the plot should include an axes in an existing plot.
         :param vertLines = Specifies the lower(LE) and upper(UE) error bounds with a factor f to show: (LE, UE, f)
         :param showLegend = If true, this shows the legend inside the plot"""

    # Check if the column highlighted below contains any real value
    valueCheck = DF[xData]
    if all(valueCheck == 0) or valueCheck.isnull().all():
        return None

    # Figure setup
    if fig is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    elif axes:
        ax1 = axes
        fig = fig
    else:
        ax1 = plt.gca()

    # Drop NA values
    WDF = DF.dropna(axis=0, how='any', subset=[cData], inplace=False).copy(deep=True)

    # Scale data with factors
    WDF[cData] = WDF[cData].apply(lambda c: c * cFactor)
    # If percentiles are given, filter the DF using the percentiles provided
    if percentiles:
        lwr = rounddown(np.percentile(a=np.array(WDF[cData].tolist()), q=percentiles[0]), 1)
        hgr = roundup(np.percentile(a=np.array(WDF[cData].tolist()), q=percentiles[1]), 1)
        df = WDF.loc[(WDF[cData] >= lwr) & (WDF[cData] <= hgr)]
    else:
        df = WDF
    # Add a column of factors to the dataframe.
    df['Factors'] = None
    for i, row in df.iterrows():
        df.loc[i, 'Factors'] = factorsFromrelativeErrors(x=row[xData], RE=row[cData])

    if xQuantile:
        xUpperATquantile = df['Factors'].quantile(xQuantile)

    # Make a histogram plot.
    if Bin1 is None:
        minRE = rounddown(df['Factors'].min(), 1)
        maxRE = roundup(df['Factors'].max(), 1)
        # Find the range of factors
        Bin1 = list(range(minRE, maxRE, 1))
        # Sort Bin1
        Bin1.sort()

    # Get the relative error corresponding to the factors
    Bin2 = []
    arbNos = 100
    for f in Bin1:
        if f < 0:
            re = str(round(relativeErrorBounds(x=arbNos, f=-f, bound='lower'), 1))
        elif f == 0:
            re = ""
        else:
            re = str(round(relativeErrorBounds(x=arbNos, f=f, bound='upper'), 1))
        Bin2.append(re)

    # pair Bin1 and Bin2
    pairedBins = list(zip(Bin1, Bin2))

    # Checking situations to show tick labels
    if len(pairedBins) <= 21:
        x1TickLabel = [createLabel(item) for item in Bin1]
        x2TickLabel = Bin2
    else:
        counter = 0
        x1TickLabel = []
        x2TickLabel = []
        for i in Bin1:
            if abs(i) % 2 != 0:
                x1TickLabel.append("")
                x2TickLabel.append("")
            else:
                x1TickLabel.append(createLabel(i))
                x2TickLabel.append(Bin2[counter])
            counter = counter + 1

    # Plot histogram
    n, _, patches = ax1.hist(x=df['Factors'].tolist(), bins=Bin1, edgecolor='k', color='khaki')

    # Plot vertical lines and calculate the percentage of measurements within an errorRange
    lineStyles = ['--', '-.', ':', '-']
    i = 0
    for factor in vertLines:
        # Get the percentage of measurements within the factor: factor
        filter1 = (df['Factors'] >= -factor) & (df['Factors'] <= factor)
        thisDF = df.loc[filter1]
        percentPoints = int(len(thisDF) * 100 / len(df))
        # This is to ensure that the the vertical lines fall within the histograms and not outside
        if -factor >= Bin1[0]:
            ax1.axvline(-factor, color='k', ls=lineStyles[i], linewidth=1)
            # Label the relative error corresponding to the factor: -factor
            lab = Bin2[Bin1.index(-factor)]
            ax1.text(-factor - 0.3, 0.96*max(n), f'{lab}%', rotation=90, verticalalignment='center',
                        fontsize=6, alpha=0.6, fontweight='medium')
        if factor <= Bin1[-1]:
            ax1.axvline(factor, color='k', ls=lineStyles[i], linewidth=1,
                        label=f'{percentPoints}% by a factor of {factor}')
            # Label the relative error corresponding to the factor: factor
            lab = Bin2[Bin1.index(factor)]
            ax1.text(factor + 0.06, 0.96*max(n), f'{lab}%', rotation=90, verticalalignment='center',
                     fontsize=6, alpha=0.6, fontweight='medium')
        i = i + 1

    # Show the mean and median quantification error line
    meanErr = round(statistics.mean(df[cData].tolist()), 2)
    medianErr = round(statistics.median(df[cData].tolist()), 2)
    meanFac = factorsFromrelativeErrors(x=100, RE=meanErr)
    medianFac = factorsFromrelativeErrors(x=100, RE=medianErr)

    # Calculate the factor associated with the calculated mean and median
    meanLineLabel = f'{meanFac:.2f}'
    medianLineLabel = f'{medianFac:.2f}'

    # Plot the mean and median vertical lines
    ax1.axvline(meanFac, color='blue', ls='-', linewidth=1,
                label=f'Mean: {meanErr}% ({createLabel(round(meanFac, 2))})')
    if meanFac < 0:
        ax1.text(meanFac - 0.3, 0.96*max(n), f'{meanErr}%', rotation=90, verticalalignment='center',
                 fontsize=6, alpha=0.6, fontweight='medium')
    else:
        ax1.text(meanFac + 0.06, 0.96*max(n), f'{meanErr}%', rotation=90, verticalalignment='center',
                 fontsize=6, alpha=0.6, fontweight='medium')
    ax1.axvline(medianFac, color='crimson', ls='-', linewidth=1, label=f'Median: {medianErr}% ({createLabel(round(medianFac, 2))})')
    if medianFac < 0:
        ax1.text(medianFac - 0.3, 0.96*max(n), f'{medianErr}%', rotation=90, verticalalignment='center',
                 fontsize=6, alpha=0.6, fontweight='medium')
    else:
        ax1.text(medianFac + 0.06, 0.96*max(n), f'{medianErr}%', rotation=90, verticalalignment='center',
                 fontsize=6, alpha=0.6, fontweight='medium')

    # Get the index of bars with factors greater or equal to 1
    colorBarIndex = [Bin1.index(r) for r in Bin1 if r >= 1]
    # Coloring some parts of the bars
    for i in colorBarIndex:
        if i == len(Bin1)-1:
            # This is because there is no bar after the last bin
            break
        else:
            patches[i].set_facecolor('olive')

    # Set ax1 axis parameters
    if xTicks is None:
        xTicks = Bin1
    ax1.set_xticks(xTicks)
    if yTicks is None:
        yTicks = list(range(0, roundup(max(list(n)), 10) + 1, 10))
    ax1.set_yticks(yTicks)
    if xLabel:
        ax1.set_xlabel(f'{xLabel}', fontsize=fontsize)
    if yLabel:
        ax1.set_ylabel(yLabel, fontsize=fontsize)
    if x1TickLabel:
        ax1.set_xticklabels(x1TickLabel, fontsize=fontsize, rotation=rotation)
    if xLimits is None:
        if xQuantile is None:
            xLimits = [Bin1[0] - 0.5, Bin1[-1] + 0.5]
        else:
            xLimits = [Bin1[0] - 0.5,  xUpperATquantile + 0.5]
    ax1.set_xlim(xLimits)

    # Characterize axis 2
    ax2 = ax1.twiny()
    ax2.set_xticks(Bin1)
    ax2.set_xlabel("Relative Error (%)", fontsize=fontsize)
    if x2TickLabel:
        ax2.set_xticklabels(x2TickLabel, rotation=rotation, fontsize=fontsize)
    ax2.set_xlim(xLimits)

    # Remove ticks
    ax1.tick_params(axis='both', bottom=False, top=False, left=True, right=False)
    ax2.tick_params(axis='both', bottom=False, top=False, left=False, right=True)

    if showLegend:
        fontP = FontProperties()
        fontP.set_size(fontsize)
        if xQuantile is None:
            lg = ax1.legend(loc='center right', prop=fontP, ncol=1, title=f'Data within ({percentiles[0]}, {percentiles[1]}) percentiles')
        else:
            lg = ax1.legend(loc='center right', prop=fontP, ncol=1, title=f'Data within ({percentiles[0]}, {percentiles[1]}) percentiles. \nUpper x-axis trimmed at {xQuantile*100}% percentile')
        lg.get_frame().set_edgecolor('black')

    ax1.grid(alpha=gridAlpha)
    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=dpi)

    return fig, ax1

def quantErrorSubplots(df, x1DataField, y1DataField, x2DataField, y2DataField, x2binUpperEdges, y3DataField,
                       x1label=None, x1limits=None, x1ticks=None, x1ticklabels=None, x1scale='linear',
                       y1label=None, y1limits=None, y1ticks=None, y1ticklabels=None, y1scale='linear',
                       x2label=None, x2limits=None, x2ticks=None, x2ticklabels=None, x2scale='linear',
                       y2label=None, y2limits=None, y2ticks=None, y2ticklabels=None, y2scale='linear',
                       y3scalingfactor=1, y2scalingfactor=1,
                       x3label=None, x3limits=None, x3ticks=None, x3ticklabels=None, x3scale='linear',
                       y3label=None, y3limits=None, y3ticks=None, y3ticklabels=None, y3scale='linear',
                       zoomView=False, zoomViewXlim=None, zoomViewYlim=None, zoomAxesWidth=None, quantile=None,
                       zoomAxesHeight=None, cDataField=None, clabel=None, climits=None, cticks=None,
                       cticklabels=None, cscale='linear', regression='linear', oneToOne=True, s=25, Hline=True,
                       gridAlpha=0.5, figsize=(5, 7.5), filePath=None, fileName=None, paperDict=None,
                       whiskLegendPosition='upper right', varPrefix=None, showmean=True, dpi=None, fontsize=7):

    """Generate a box and whisker pot for data in df.  df[y2DataField] is grouped into several subsets by df[x2DataField].
     xbins are (lower, upper] where the upper edge of each bin is specified and the lower is taken as the previous bin's
     upper edge. The first bin is taken with a lower edge = 0.
     Required Parameters:
     :param df = dataframe to pull data from.
     :param x1DataField = column in df to use for x axis on subplot 1 (measured vs metered)
     :param y1DataField = column in df to use for y data on subplot 1 (measured vs metered)
     :param x2DataField = column in df to use for x axis on subplots 2 & 3 (%error vs flow & %error box and whisker)
     :param y2DataField = column in df to use for y data on subplots 2 & 3 (%error vs flow & %error box and whisker)
     :param x2binUpperEdges = Array of upper edges for bins in box and whisker. e.g. [0.1,1,10,100]
     Optional parameters:
     :param x1label = label for x axis of subplot 1
     :param x1limits = limits of x-axis of subplot 1
     :param x1ticks = location of major tick marks for x-axis of subplot 1
     :param x1ticklabels = labels for major tick marks for x-axis of subplot 1
     :param x1scale = scale for x axis ("linear" or "log") of subplot 1
     :param y1label = label for y axis of subplot 1
     :param y1limits = limits of y-axis of subplot 1
     :param y1ticks = location of major tick marks for y-axis of subplot 1
     :param y1ticklabels = labels for major tick marks for y-axis of subplot 1
     :param y1scale = scale for y axis ("linear" or "log") of subplot 1
     :param x2label = label for x axis of subplot 2 & 3
     :param x2limits = limits of x-axis of subplot 2 & 3
     :param x2ticks = location of major tick marks for x-axis of subplot 2 & 3
     :param x2ticklabels = labels for major tick marks for x-axis of subplot 2 & 3
     :param x2scale = scale for x axis ("linear" or "log") of subplot 2 & 3
     :param y2label = label for y axis of subplot 2 & 3
     :param y2limits = limits of y-axis of subplot 2 & 3
     :param y2ticks = location of major tick marks for y-axis of subplot 2 & 3
     :param y2ticklabels = labels for major tick marks for y-axis of subplot 2 & 3
     :param y2scale = scale for y axis ("linear" or "log") of subplot 2 & 3
     :param figsize = (width, height) of figure in inches
     :param filePath = directory to save figure to. Defaults to cwd.
     :param fileName = if specified saves figure to filePath/fileName"""

    # Check if the column highlighted below contains any real value
    valueCheck = df[y1DataField]
    if all(valueCheck == 0) or valueCheck.isnull().all():
        return None

    valueCheck = df[y2DataField]
    if all(valueCheck == 0) or valueCheck.isnull().all():
        return None

    #---------------Figure (setup subplots) ----------------------
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    plt.rcParams['patch.linewidth'] = 0.4

    # --------------- ax 1 (bridger estimate vs metered flowrate - scatter w linear regression) ----------------------
    filt = (~df[x1DataField].isna()) & ~(df[y1DataField].isna())
    x = df.loc[filt, x1DataField]
    y = df.loc[filt, y1DataField]
    if cDataField:
        c = df.loc[filt, cDataField]

    yMin = y.min()
    # plot 1:1 line
    if oneToOne:
        one = (min(x), max(x))
        ax1.plot(one, one, label="1:1", c='black')

    # plot x1, y1 scatter
    if cDataField:
        cmap = mpl.cm.get_cmap('jet')
        if cscale == 'linear':
            norm = mpl.colors.Normalize(vmin=climits[0], vmax=climits[1])
        elif cscale == 'log':
            norm = mpl.colors.LogNorm(vmin=climits[0], vmax=climits[1])
        mappable = ax1.scatter(x, y, c=c, cmap=cmap, norm=norm, s=s, edgecolors='black', label='Data')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("top", size="7%", pad=0.05)
        cb = fig.colorbar(mappable=mappable, cax=cax, orientation='horizontal', label=clabel, ticks=cticks)
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        #cb = fig.colorbar(mappable=mappable, cax=cax, label=clabel, ticks=cticks)
        #cb = fig.colorbar(mappable=mappable, ax=ax1, location='top', label=clabel, ticks=cticks)
    else:
        ax1.scatter(x, y, s=s, edgecolors='black', label='Data')

    # fit data series and plot
    if regression == 'linear':
        # calculate regression
        p, R2 = ols_slope(x.astype("float"), y.astype("float"))
        x_fit = np.linspace(min(x), max(x))
        y_fit = x_fit * p
        ax1.plot(x_fit, y_fit, label=r"${p:.3f}*x, R^2$ = {r:.2f}".format(p=p, r=R2), linestyle='dashed')

        if paperDict:
            paperDict[varPrefix+'LinearReg_Slope'] = round(p, 3)
            paperDict[varPrefix+'LinearReg_Bias'] = round(100*(p-1), 1)
            paperDict[varPrefix+'LinearReg_RSquare'] = round(R2, 2)

        # linear regression with intercept
        # p = np.polyfit(x.astype("float"), y.astype("float"), 1, full=True)
        # f = np.poly1d(p[0])
        #x_fit = np.linspace(min(x), max(x))
        #y_fit = f(x_fit)
        #SSE = p[1][0]
        #diff = y-y.mean()
        #square_diff = diff ** 2
        #SST = square_diff.sum()
        #R2 = 1-SSE/SST
        #ax1.plot(x_fit, y_fit, label=r"${f:}, R^2$ = {r:.2f}".format(f=str(f).strip(), r=R2),  linestyle='dashed')

    # format x-axis based on method args
    ax1.set_xscale(x1scale)
    if x1ticks:
        ax1.set_xticks(x1ticks, fontsize=fontsize)
    if x1ticklabels:
        ax1.set_xticklabels(x1ticklabels, fontsize=fontsize)
    if x1label:
        ax1.set_xlabel(x1label, fontsize=fontsize)
    if x1limits:
        ax1.set_xlim(x1limits)

    # format y-axis based on method args
    ax1.set_yscale(y1scale)
    if y1ticks:
        ax1.set_yticks(y1ticks, fontsize=fontsize)
    if y1ticklabels:
        ax1.set_yticklabels(y1ticklabels, fontsize=fontsize)
    if y1label:
        ax1.set_ylabel(y1label, fontsize=fontsize)
    if y1limits:
        ax1.set_ylim(y1limits)
    elif quantile:
        ax1.set_ylim(yMin, y.quantile(quantile))

    fontP = FontProperties()
    fontP.set_size(fontsize)
    leg = ax1.legend(ncol=3, loc='lower center', prop=fontP, bbox_to_anchor=(0, -0.8, 1.0, 0.4), borderaxespad=0,
                     mode="expand")
    #leg = ax1.legend(ncol=3, loc='lower center', prop=fontP, bbox_to_anchor=(0, -0.8, 1, 0.5), borderaxespad=0,
    #                 mode="expand")

    leg.get_frame().set_edgecolor('black')
    #ax1.legend(loc='lower right', prop=fontP)
    ax1.grid(alpha=gridAlpha)

    # --------------- ax 2 (% error vs metered flowrate - scatter) ----------------------
    filt = (~df[x2DataField].isna()) & ~(df[y2DataField].isna())
    x2 = df.loc[filt, x2DataField]
    y2 = df.loc[filt, y2DataField] * y2scalingfactor
    c2 = df.loc[filt, cDataField]

    y2Min = y2.min()
    if cDataField:
        # plot markers on ax2 using same cmap and norm as ax1
        ax2.scatter(x2, y2, c=c2, cmap=cmap, norm=norm, s=s, edgecolors='black', label='Data')
    else:
        ax2.scatter(x2, y2, s=s, edgecolors='black')

    ax2.set_xscale(x2scale)
    if x2ticks:
        ax2.set_xticks(x2ticks)
    if x2ticklabels:
        ax2.set_xticklabels(x2ticklabels, fontsize=fontsize)
    if x2label:
        ax2.set_xlabel(x2label, fontsize=fontsize)
    if x2limits:
        ax2.set_xlim(x2limits)

    # format y-axis based on method args
    ax2.set_yscale(y2scale)
    if y2ticks:
        ax2.set_yticks(y2ticks)
    if y2ticklabels:
        ax2.set_yticklabels(y2ticklabels, fontsize=fontsize)
    if y2label:
        ax2.set_ylabel(y2label, fontsize=fontsize)
    if y2limits:
        ax2.set_ylim(y2limits)
    elif quantile:
        ax2.set_ylim(y2Min, y2.quantile(quantile))

    ax2.grid(alpha=gridAlpha)

    #--------------- ax 3 (% error vs metered flowrate - box & whisker) ----------------------
    # select data from df for box whisker
    data = list()
    centers = list()
    widths = list()
    counts = list()
    labels = list()

    previousUE = 0
    for UE in x2binUpperEdges:
        filt = (df[x2DataField] > previousUE) & (df[x2DataField] <= UE) & ~(df[y3DataField].isna())
        c = list(df.loc[filt, y3DataField]*y3scalingfactor)
        mean_x = df.loc[filt, x2DataField].mean()
        max_x = df.loc[filt, x2DataField].max()
        min_x = df.loc[filt, x2DataField].min()
        width_x = max_x-min_x
        centers_x = min_x + width_x/2
        count = len(c)
        label = "({lower}, {upper}]".format(lower=previousUE, upper=UE)

        data.append(c)
        centers.append(centers_x)
        widths.append(width_x)
        counts.append(count)
        labels.append(label)
        previousUE = UE

    filt = ~(df[y3DataField].isna())
    yData = df.loc[filt, y3DataField]*y3scalingfactor
    yDataMin = yData.min()
    boxdict = ax3.boxplot(data, positions=centers, widths=widths, whis=(2.5, 97.5), showmeans=showmean, meanline=True)

    if Hline:
        ax3.axhline(y=0, color="C7", linestyle='--')

    #ax3.grid(alpha=0.3)
    if zoomView == True:
        # METHOD 3
        axins = inset_axes(ax3, width=zoomAxesWidth, height=zoomAxesHeight)
        axins.set_xlim(zoomViewXlim)
        axins.set_ylim(zoomViewYlim)
        axins.boxplot(data, positions=centers, widths=widths, whis=(2.5, 97.5), showmeans=showmean, meanline=True)
        axins.set_xscale(x3scale)


    if paperDict:
        paperDict[varPrefix+'Noutliers'] = sum([item.get_ydata().size for item in boxdict['fliers']])
        paperDict[varPrefix+'maxWhisker'] = round(max([item.get_ydata()[1] for item in boxdict['whiskers']]), 0)
        paperDict[varPrefix+'minWhisker'] = round(min([item.get_ydata()[1] for item in boxdict['whiskers']]), 0)

    # format x-axis based on method args
    ax3.set_xscale(x3scale)
    if x3ticks:
        ax3.set_xticks(x3ticks)
    if x3ticklabels:
        ax3.set_xticklabels(x3ticklabels, fontsize=fontsize)
    if x3label:
        ax3.set_xlabel(x3label, fontsize=fontsize)
    if x3limits:
        ax3.set_xlim(x3limits)

    # format y-axis based on method args
    ax3.set_yscale(y3scale)
    if y3ticks:
        ax3.set_yticks(y3ticks)
    if y3ticklabels:
        ax3.set_yticklabels(y3ticklabels, fontsize=fontsize)
    if y3label:
        ax3.set_ylabel(y3label, fontsize=fontsize)
    if y3limits:
        ax3.set_ylim(y3limits)
    elif quantile:
        ax3.set_ylim(yDataMin, yData.quantile(quantile))
    if showmean:
        leg = ax3.legend(handles=[boxdict['medians'][0], boxdict['means'][0]], labels=['median', 'mean'],
                   loc=whiskLegendPosition, prop=fontP)
    else:
        leg = ax3.legend(handles=[boxdict['medians'][0]], labels=['median'], loc='upper right', prop=fontP)



    leg.get_frame().set_edgecolor('black')
    # plot data counts on secondary x axis above figure
    ax4 = ax3.twiny()
    ax4.set_xscale(x3scale)
    ax4.set_xlim(x3limits)
    # ax4.set_xticks(centers)
    # ax4.set_xticklabels(counts)
    # ax4.minorticks_off()
    # ax4.set_xlabel('Sample Count')

    tickshold = []
    countshold = []
    j = 0
    for i in x2binUpperEdges:
        if i <= x3limits[0]:
            tickshold.append(x3limits[0])
            countshold.append("")
            centers.pop(j)
            counts.pop(j)
        else:
            pass
        j = j + 1
    if tickshold:
        xCenters = [list(set(tickshold))[0]]
        xCounts = [list(set(countshold))[0]]
        for x in centers:
            xCenters.append(x - x3limits[0])
        for y in counts:
            xCounts.append(y)
        xCenters.pop()
        xCounts.pop()
        ax4.set_xticks(xCenters)
        ax4.set_xticklabels(xCounts, fontsize=fontsize)
    else:
        ax4.set_xticks(centers)
        ax4.set_xticklabels(counts, fontsize=fontsize)
    ax4.minorticks_off()
    ax4.set_xlabel('Sample Count', fontsize=fontsize)

    # turn on grid and tight layout
    ax3.grid(alpha=0.3, axis='y')

    fig.tight_layout()

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=dpi)

    plt.close()  # close current figure

    return fig, ax1, ax2, ax3, ax4, paperDict


def QuantificationAccuracyPlot(classifiedDF, outputFilePath):
    Gases = {'THC': ['tc_THCMassFlow', 'red'], 'NMHC': ['tc_NMHCMassFlow', 'blue'],
             'METHANE': ['tc_C1MassFlow', 'green'],
             'ETHANE': ['tc_C2MassFlow', 'yellow'], 'PROPANE': ['tc_C3MassFlow', 'orange'],
             'BUTANE': ['tc_C4MassFlow', 'purple']}
    Min = 10000
    Max = 0
    try:
        # Plot TP data as x = tc_emissionrate, y = p_emissionrate with error bars of x (+- tc_uncertain) and y
        # directions (+- p_emissionrate upper and lower)

        fig, ax1 = plt.subplots()

        for gas, massFlowName in Gases.items():
            xLower = []
            xUpper = []
            yLower = []
            yUpper = []
            x = []
            y = []

            # Filt out TP rows by gas name
            filt1 = classifiedDF['p_Gas'] == gas
            data = classifiedDF.loc[filt1]

            for index, line in data.iterrows():
                xLower.append(float(line[massFlowName[0] + 'Uncertainty']))
                xUpper.append(float(line[massFlowName[0] + 'Uncertainty']))
                yLower.append(float(line['p_EmissionRateLower']))
                yUpper.append(float(line['p_EmissionRateUpper']))
                x.append(float(line[massFlowName[0]]))
                y.append(float(line['p_EmissionRate']))
            if x and y:
                ax1.errorbar(x=x, y=y, yerr=[yLower, yUpper], xerr=[xLower, xUpper], linestyle="None", marker='s',
                             mfc=massFlowName[1], label=gas)
                ax1.loglog(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),
                           label=gas + ' line of best fit')

                # # Find the largest value of x and y and set the scale to be min and max
                x_min, x_max = ax1.get_xlim()
                y_min, y_max = ax1.get_ylim()
                if x_min < Min:
                    Min = x_min
                if y_min < Min:
                    Min = y_min
                if x_max > Max:
                    Max = x_max
                if y_max > Max:
                    Max = y_max
        plt.loglog([Min, Max], [Min, Max], label='Line of Equality')
        plt.ylim(Min, Max)
        plt.xlim(Min, Max)

        plt.grid()
        ax1.legend()
        plt.xlabel('Test Center Emission Rate (g\h)')
        plt.ylabel('Performer Emission Rate (g\h)')
        path = os.path.join(outputFilePath, 'TPCurve.png')
        plt.savefig(path)
        return ax1

    except Exception as e:
        print(f'Could not build TP Plot due to exception: {e}')
        return None