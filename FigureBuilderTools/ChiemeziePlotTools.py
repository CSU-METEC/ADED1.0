import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib as mpl
import math
from statistics import mean
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from AnalysisTools.MetricsTools.PODSurfaceWithEvenCounts import calcPODSurfaceWEvenCounts
from matplotlib.projections import register_projection
import os
import math
import statistics
import itertools
#from FigureBuilderTools import zoomplot
#import zoomplot
import numpy as np
import pandas as pd
from windrosePlot import WindroseAxes
from scipy.interpolate import griddata
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
from scipy.optimize import curve_fit
import statsmodels.api as sm
import random
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

def logistic(coeff, x):  # Logistic Function
    try:
        y = 1 / (1 + math.exp(coeff[0] + coeff[1] * x))
    except:
        y = 0
    return y

def invlogistic(p, B):  # invlogistic Function
    x = (math.log(1 / (p) - 1) - B[0]) / B[1]
    return x

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

def maxList(a, b):
    count = []
    for tup in zip(a, b):
        count.append(max(tup))
    return count

def sumList(a, b):
    count = []
    for tup in zip(a, b):
        count.append(sum(tup))
    return count

def relativeErrorBounds(x=None, f=None, bound=None):
    if bound == 'upper':
        RE = ((f*x)-x)*100/x
    elif bound == 'lower':
        RE = ((x/f)-x)*100/x
    return RE

def inverseRelativeError(x, RE):
    y = (x*RE/100) + x
    return y

def roundToNearestInteger(x):
    diff = x - int(x)
    if diff >= 0.5:
        y = int(math.ceil(x / float(1))) * 1
    else:
        y = int(math.floor(x / float(1))) * 1
    return y

def factorsFromrelativeErrors(x=None, RE=None):
    if RE > 0:
        f = ((x*RE/100)+x)/x
    elif RE < 0:
        f = -1*(x/((x*RE/100)+x))
    else:
        f = 1
    return f

def istherezero(thisList):
    i = 0
    for n in thisList:
        if int(n) == int(0):
            i = 1
    if i == 1:
        thisList.append(int(0))
    return thisList


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

def returnSmallestList(*lists):
    listOFlists = []
    countPERlist = []
    for thisList in lists:
        listOFlists.append(thisList)
        countPERlist.append(len(thisList))
    # Find the list with the least number of elements
    index = countPERlist.index(min(countPERlist))
    return listOFlists[index]


def findTheMaxMultipleCommonToLists(*lists, f=1):
    listOFlists = []
    for thisList in lists:
        listOFMultiples = []
        for e in thisList:
            # Check if e is a multiple of f
            if (e % f) == 0:
                listOFMultiples.append(e)
        # Check for the maximum element
        listOFlists.append(listOFMultiples)
    if len(listOFlists) > 1:
        # If there is more than one list, find the list of elements common to the first 2 list
        a_set = set(listOFlists[0])
        b_set = set(listOFlists[1])
        commons = list(a_set & b_set)
        # iterate from the second list in listOFlists and compare each list with the common list: commons
        for index in range(1, len(listOFlists)):
            commonINboth = list(set(listOFlists[index]) & set(commons))
            # Reassign the list:commonINboth gotten earlier to facilitate comparison with the list: listOFlists[index]
            commons = commonINboth
        # After all iterations, append the final list: commons to listOFCommonMultiples
    else:
        commons = list(itertools.chain.from_iterable(listOFlists))

    return max(commons)

def defineTick(positions, nBars, i, h=0.5):
    """ This code generates the yTicks of bars
        :Param positions: The position of each y-data variable
        :Param h: The height of each bar
        :Param nBars: The number of bars for each category of y-data
        :Param i: The index of the specific bar under consideration
    """
    y = []
    num = nBars
    for j in positions:
        start = j - (nBars-1)*0.5*h
        stop = j + (nBars-1)*0.5*h
        y.append(list(np.linspace(start=start, stop=stop, num=num))[i])
    return y

def annotateBar(barContainer, ax, l, labelPosition='center', rotation=0):
    """ Script for annotating a bar.
        Required Parameters:
        :Param barContainer: The container of each bar
        :Param ax: The axes to be plotted
        :Param l: The label
        :Param labelPosition: The position of the text annotations.
        :Param rotation: the orientation of the annotations
        """
    if labelPosition == 'center':
        bl = barContainer.get_xy()
        x = 0.5 * barContainer.get_width() + bl[0]
        y = 0.5 * barContainer.get_height() + bl[1]
        text = ax.text(x, y, f'{l}%', ha='center', va='center', rotation=rotation)
    elif labelPosition == 'top':
        bl = barContainer.get_xy()
        x = barContainer.get_width() + bl[0] + 8*barContainer.get_height()
        y = 0.5 * barContainer.get_height() + bl[1]
        text = ax.text(x, y, l, ha='left', va='center', rotation=rotation)
    return text

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

def createTickLabel(item, divisor, trigger):
    """Convert factor ticks into label depending on value of bin.
        :param item = The bin value to convert to label."""
    if item in [-1, 1]:
        label = str(abs(item))
        trigger = True
    elif abs(item) % divisor == 0:
        label = str(abs(item))
        trigger = True
    else:
        label = ""
    return label, trigger

def modifiedBoxWhisker(df, xDataField, yDataField, xbinUpperEdges, xlabel=None, xlimits=None, xticks=None,
               xticklabels=None, xscale='linear', ylabel=None, ylimits=None, yticks=None, yticklabels=None,
               yscale='linear', zoomView=False, zoomViewXlim=None, zoomViewYlim=None, yScaleFactor=1, quantile=None,
               fig=None, axes=None, figsize=None, filePath=None, fileName=None):
    """
    Generate a box and whisker pot for data in df.  df[yDataField] is grouped into several subsets by df[xDataField].
     xbins are (lower, upper] where the upper edge of each bin is specified and the lower is taken as the previous bin's
     upper edge. The first bin is taken with a lower edge = 0.
     Required Parameters:
     :param df = dataframe to pull data from.
     :param xDataField = column in df to use for x axis
     :param yDataField = column in df to use for y data in each box & whisker set
     :param xbinUpperEdges = Array of upper edges for bins. e.g. [0.1,1,10,100]
     Optional parameters:
     :param xlabel = label for x axis of figure
     :param xlimits = limits of x-axis
     :param xticks = location of major tick marks for x-axis
     :param xticklabels = labels for major tick marks for x-axis
     :param xscale = scale for x axis ("linear" or "log")
     :param ylabel = label for y axis of figure
     :param ylimits = limits of y-axis
     :param yticks = location of major tick marks for y-axis
     :param yticklabels = labels for major tick marks for y-axis
     :param yscale = scale for y axis ("linear" or "log")
     :param figsize = (width, height) of figure in inches
     :param filePath = directory to save figure to. Defaults to cwd.
     :param fileName = if specified saves figure to filePath/fileName
     Return values:
     :returns fig =
     """

    # select data from df and format for box whisker
    yFilt = ~(df[yDataField].isna())
    yData = df.loc[yFilt, yDataField]*yScaleFactor
    yDataMin = yData.min()

    data = list()
    centers = list()
    widths = list()
    counts = list()
    labels = list()

    previousUE = 0
    for UE in xbinUpperEdges:
        filt = (df[xDataField] > previousUE) & (df[xDataField] <= UE) & ~(df[yDataField].isna())
        c = list(df.loc[filt, yDataField] * yScaleFactor)
        mean_x = df.loc[filt, xDataField].mean()
        max_x = df.loc[filt, xDataField].max()
        min_x = df.loc[filt, xDataField].min()
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

    boxdict = ax1.boxplot(data, positions=centers, widths=widths, whis=(2.5, 97.5), showmeans=True, meanline=True)

    if zoomView == True:
        axins = inset_axes(ax1, width=2.3, height=1.7)
        axins.set_xlim(zoomViewXlim)
        axins.set_ylim(zoomViewYlim)
        axins.boxplot(data, positions=centers, widths=widths, whis=(2.5, 97.5), showmeans=True, meanline=True)
        axins.set_xscale(xscale)

    # format x-axis based on method args
    ax1.set_xscale(xscale)
    if xticks:
        ax1.set_xticks(xticks)
    if xticklabels:
        ax1.set_xticklabels(xticklabels)
    if xlabel:
        ax1.set_xlabel(xlabel)
    if xlimits:
        ax1.set_xlim(xlimits)

    # format y-axis based on method args
    ax1.set_yscale(yscale)
    if yticks:
        ax1.set_yticks(yticks)
    if yticklabels:
        ax1.set_yticklabels(yticklabels)
    if ylabel:
        ax1.set_ylabel(ylabel)
    if ylimits:
        ax1.set_ylim(ylimits)
    elif quantile:
        ax1.set_ylim(yDataMin, yData.quantile(quantile))

    # plot data counts on secondary x axis above figure
    ax2 = ax1.twiny()
    ax2.set_xscale(xscale)
    ax2.set_xlim(xlimits)
    tickshold = []
    countshold = []
    j = 0
    for i in xbinUpperEdges:
        if i <= xlimits[0]:
            tickshold.append(xlimits[0])
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
            xCenters.append(x-xlimits[0])
        for y in counts:
            xCounts.append(y)
        xCenters.pop()
        xCounts.pop()
        ax2.set_xticks(xCenters)
        ax2.set_xticklabels(xCounts)
    else:
        ax2.set_xticks(centers)
        ax2.set_xticklabels(counts)
    ax2.minorticks_off()
    ax2.set_xlabel('Sample Count')

    # turn on grid and tight layout
    fontP = FontProperties()
    fontP.set_size(8)
    ax1.legend(handles=[boxdict['medians'][0], boxdict['means'][0]], labels=['median', 'mean'],
               loc='upper left', prop=fontP)
    ax1.grid(alpha=0.5)
    fig.tight_layout()

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)

    return fig, ax1, ax2


def plotByC(df, xDataField, yDataField, catDataField, yScaleFactor=1, filt=None,
            cbarPosition='top', cmap=None, s=25, norm=None, c=None, axes=None, fig=None, figsize=None,
            cscale='linear', climits=None, clabel=None, cticks=None):

    # Setting up the plot environment
    plt.rcParams.update({'font.size': 7})
    if fig is None:
        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()
    elif axes:
        ax = axes
        fig = fig
    else:
        ax = plt.gca()

    # Plotting with c
    x = df.loc[filt, xDataField]
    y = df.loc[filt, yDataField] * yScaleFactor
    if c is not None:
        mappable = ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=s, edgecolors='black', label='Data')
    else:
        c = df.loc[filt, catDataField]
        cmap = mpl.cm.get_cmap('jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(cbarPosition, size="7%", pad=0.2)
        if cscale == 'linear':
            norm = mpl.colors.Normalize(vmin=climits[0], vmax=climits[1])
            mappable = ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=s, edgecolors='black', label='Data')
            if cbarPosition == 'top' or cbarPosition == 'bottom':
                cb = fig.colorbar(mappable, orientation="horizontal", cax=cax, label=clabel, ticks=cticks)
                cax.xaxis.set_ticks_position(cbarPosition)
                cax.xaxis.set_label_position(cbarPosition)
            elif cbarPosition == 'right' or cbarPosition == 'left':
                cb = fig.colorbar(mappable, orientation="vertical", cax=cax, label=clabel, ticks=cticks)
                cax.yaxis.set_ticks_position(cbarPosition)
                cax.yaxis.set_label_position(cbarPosition)
        elif cscale == 'log':
            norm = mpl.colors.LogNorm(vmin=climits[0], vmax=climits[1])
            mappable = ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=s, edgecolors='black', label='Data')
            if cbarPosition == 'top' or cbarPosition == 'bottom':
                cb = fig.colorbar(mappable, orientation="horizontal", cax=cax, label=clabel, ticks=cticks)
                cax.xaxis.set_ticks_position(cbarPosition)
                cax.xaxis.set_label_position(cbarPosition)
            elif cbarPosition == 'right' or cbarPosition == 'left':
                cb = fig.colorbar(mappable, orientation="vertical", cax=cax, label=clabel, ticks=cticks)
                cax.yaxis.set_ticks_position(cbarPosition)
                cax.yaxis.set_label_position(cbarPosition)

    return mappable, cb, fig, ax, cmap, norm


def modifiedCategoricalScatter(df, xDataField, yDataField, catDataField, xlabel=None, xlimits=None, xticks=None,
                       xticklabels=None, xscale="linear", cscale='linear', climits=None, cticks=None, clabel=None,
                       cticklabels=None, ylabel=None, ylimits=None, yticks=None,yticklabels=None, yscale="linear",
                       yScaleFactor=1, cats=None, linearFit=False, regression=None, oneToOne=False, s=25, quantile=None,
                       gridAlpha=0.5,fig=None, axes=None, figsize=None, filePath=None, plotBYc="No", c=None, fileName=None,
                       cmap=None, norm=None, cbarPosition = 'top', paperDict=None, varPrefix=None):
    """scatter plot using data in df. Plots each category in catDataField as a different series.
    Required arguments:
    :param df - dataframe from which to collect data
    :param xDataField - column header in df to use for x data
    :param yDataField - column header in df to use for y data
    :param catDataField - column header in df to use for categories.
    Optional arguments:
    :param xlabel - string to use as x axis label
    :param xlimits - x axis limits in form [xmin, xmax]. If None axis is scaled to all data by default.
    :param xticks = location of major tick marks for x-axis
    :param xticklabels = labels for major tick marks for x-axis
    :param xscale = scale for x axis ("linear" or "log")
    :param ylabel - string to use as y axis label
    :param ylimits - y axis limits in form [ymin, ymax]. If None axis is scaled to all data by default.
    :param yticks = location of major tick marks for y-axis
    :param yticklabels = labels for major tick marks for y-axis
    :param yscale = scale for y axis ("linear" or "log")
    :param cats - optional list of categories in catDataField to include.  If provided, only data series corresponding to cats will be plotted.
    :param linearFit - Add linear fit of each series if True. Default = False
    :param s - marker size for scatter plot.  Default to s=25.
    :param gridAlpha - transparency of axes grid (default gridAlpha=0.5)
    :param filePath = directory to save figure to. Defaults to cwd.
    :param fileName = if specified saves figure to filePath/fileName
    :param cscale = can either be linear or log
    :param plotBYc = if Yes, use c or create c to use in the scatter plot, if No, plot without it
    :param c = A scalar or sequence of n numbers to be mapped to colors using cmap and norm.
    :param cmap = A Colormap instance or registered colormap name. cmap is only used if c is an array of floats.
    :param norm = If c is an array of floats, norm is used to scale the color data, c, in the range 0 to 1, in order to map into the colormap cmap. If None, use the default colors.Normalize.
    """

    # if specific list of categories to plot isn't provided, use unique values in df[catDataField]
    if figsize == None:
        figsize = (3.54331, 3.54331)

    # make figure
    plt.rcParams.update({'font.size': 7})
    if fig is None:
        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()
    elif axes:
        ax = axes
        fig = fig
    else:
        ax = plt.gca()

    filt = (~df[xDataField].isna()) & ~(df[yDataField].isna())
    xvar = df.loc[filt, xDataField]
    yvar = df.loc[filt, yDataField] * yScaleFactor
    # plot 1:1 line
    if oneToOne:
        one = (min(xvar), max(xvar))
        ax.plot(one, one, label="1:1", c='black')

    yMax = yvar.max()
    yMin = yvar.min()
    # Categorizing the data
    if plotBYc == "Yes": #Plotting method when c is given or c will be determined and used in the plot
        mappable, cb, fig, ax, cmap, norm = plotByC(df=df, xDataField=xDataField, yDataField=yDataField,
                                                    catDataField=catDataField, yScaleFactor=yScaleFactor,
                                                    filt=filt, cbarPosition=cbarPosition, cmap=cmap, s=s,
                                                    norm=norm, c=c, axes=ax, fig=fig, cscale=cscale,
                                                    climits=climits, clabel=clabel, cticks=cticks)

    elif plotBYc == "No":
        if cats == None:
            cats = df[catDataField].unique()
        cmap = None
        norm = None
        c = None

        markertypes = itertools.cycle(['o', 's', 'd', 'h', 'p', '^'])
        # plot each category as it's own series
        for cat in cats:
            tempDF = df.loc[(df[catDataField] == cat) & filt]
            x = tempDF[xDataField]
            xmax = tempDF[xDataField].max()
            y = tempDF[yDataField]
            ax.scatter(x, y, s=s, edgecolors='black', label=cat, marker=next(markertypes))
            if linearFit==True:
                # make linear fit to data series
                p = np.polyfit(x.astype("float"), y.astype("float"), 1)
                f = np.poly1d(p)
                x_new = np.linspace(0.1, xmax)
                y_new = f(x_new)
                ax.plot(x_new, y_new, label=r"{0:} fit = ${1:}$".format(cat, str(f).strip()), linestyle='dashed')

    # fit data series and plot
    if regression == 'linear':
        # calculate regression
        p, R2 = ols_slope(xvar.astype("float"), yvar.astype("float"))
        x_fit = np.linspace(min(xvar), max(xvar))
        y_fit = x_fit * p
        ax.plot(x_fit, y_fit, label=r"${p:.3f}*x, R^2$ = {r:.2f}".format(p=p, r=R2), linestyle='dashed')
        # Updating the paperDict with required variables
        if paperDict:
            paperDict[varPrefix + 'LinearReg_Slope'] = round(p, 3)
            paperDict[varPrefix + 'LinearReg_Bias'] = round(100 * (p - 1), 1)
            paperDict[varPrefix + 'LinearReg_RSquare'] = round(R2, 2)

    # format x-axis based on method args
    ax.set_xscale(xscale)
    if xticks:
        ax.set_xticks(xticks)
    if xticklabels:
        ax.set_xticklabels(xticklabels)
    if xlabel:
        ax.set_xlabel(xlabel)
    if xlimits:
        ax.set_xlim(xlimits)

    # format y-axis based on method args
    ax.set_yscale(yscale)
    if yticks:
        ax.set_yticks(yticks)
    if yticklabels:
        ax.set_yticklabels(yticklabels)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylimits:
        ax.set_ylim(ylimits)
    elif quantile:
        ax.set_ylim(yMin, yvar.quantile(quantile))

    fontP = FontProperties()
    fontP.set_size(8)
    ax.legend(loc='upper right', prop=fontP)
    fig.tight_layout()
    plt.grid(alpha=gridAlpha)

    # Save figure
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)
    #plt.close()  # close current figure

    return fig, ax, cmap, norm, c

def barChartPolarAxis(df,thetaData,radialData,catDataField,s=25,categorical="Yes",
                      cats=None, clabel=None, rlimits=None, tlimits=None,
                      figsize=None, n=6, cbarPosition = 'top',
                      thetaticks=None, angles=None,
                      fileName='Polarplot.png', filePath=None):
    """
        :param df - dataframe from which to collect data
        :param thetaData - column header in df to use for polar data
        :param radialData - column header in df to use for radial data
        :param catDataField - column header in df to use for categories.
        Optional arguments:
        :param thetalabel - string to use as polar data label
        :param tlimits - polar axis limits in form [min, max]. If None axis is scaled to all data by default.
        :param rlimits - radial axis limits in form [min, max]. If None axis is scaled to all data by default.
        :param polarticks = location of tick marks for the polar data
        :param n - number of concentric circles.  Default to n=6.
        :param s - marker size for scatter plot.  Default to s=25.
        :param gridAlpha - transparency of axes grid (default gridAlpha=0.5)
        :param filePath = directory to save figure to. Defaults to cwd.
        :param fileName = if specified saves figure to filePath/fileName
        """
    # make figure
    plt.rcParams.update({'font.size': 7})
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    ax = fig.add_subplot(111, projection='polar')
    if rlimits is not None:
        filt = (~df[thetaData].isna()) & ~(df[radialData].isna()) & (df[radialData] >= rlimits[0]) & (df[radialData] <= rlimits[1])
    else:
        filt = (~df[thetaData].isna()) & ~(df[radialData].isna())

    subdf = df.loc[filt]
    if categorical == "Yes":
        markertypes = itertools.cycle(['o', 's', 'd', 'h', 'p', '^'])
        if cats == None:
            cats = subdf[catDataField].unique()
        for cat in cats:
            tempDF = subdf.loc[(df[catDataField] == cat)]
            theta = tempDF[thetaData]
            rad = tempDF[radialData]
            scatterPlot = ax.scatter(theta, rad, edgecolors='black', s=s, label=cat, marker=next(markertypes), alpha=1)
        r = subdf.loc[filt, radialData]
        rmax = r.max()
        rmin = r.min()
        rmaxINT = roundup(rmax, 1)
        rminINT = rounddown(rmin, 1)
        radii = [round(x, 1) for x in list(np.linspace(rminINT, rmaxINT, n, endpoint=True))]
        ax.set_rgrids(radii, labels=radii, angle=0, fmt=None)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        if thetaticks:
            ax.set_thetagrids(angles=angles, labels=thetaticks)
        if tlimits:
            ax.set_thetamin(rounddown(tlimits[0], 1))
            ax.set_thetamax(roundup(tlimits[1], 1))
        fig.legend(loc='upper right')
    elif categorical == "No":
        tempDF = subdf.loc[~df[catDataField].isnull()]
        theta = tempDF[thetaData]
        r = tempDF[radialData]
        c = tempDF[catDataField]
        cmax = c.max()
        cmin = c.min()
        cmaxINT = roundup(cmax, 5)
        cminINT = rounddown(cmin, 5)
        cticks = list(range(cminINT, cmaxINT + 1, 5))
        cmap = plt.get_cmap('inferno')
        scatterPlot = ax.scatter(theta, r, c=c, s=s, cmap=cmap, edgecolors='black', alpha=0.7)
        cb = plt.colorbar(scatterPlot, ax=ax, pad=0.15, label=clabel, ticks=cticks)
        rmax = r.max()
        rmin = r.min()
        rmaxINT = roundup(rmax, 1)
        rminINT = rounddown(rmin, 1)
        radii = [round(x, 1) for x in list(np.linspace(rminINT, rmaxINT, n, endpoint=True))]
        ax.set_rgrids(radii, labels=radii, angle=0, fmt=None)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        #ax.set_theta_zero_location('W', offset=-180)
        if thetaticks:
            ax.set_thetagrids(angles=angles, labels=thetaticks)
        if tlimits:
            ax.set_thetamin(rounddown(tlimits[0], 1))
            ax.set_thetamax(roundup(tlimits[1], 1))
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)

    return fig, ax

def modifiedbuildPODSurface(podSurfaceDF, xLabel=None, yLabel=None, cbarPosition="top", fileName=None,
                            fig=None, axes=None, figsize=None, filePath=None):

    podSurfaceDF.dropna(axis=0, how='any', subset=['xbinAvg', 'ybinAvg', 'xy_pod'], inplace=True)
    x = np.linspace(podSurfaceDF['xbinAvg'].min(), podSurfaceDF['xbinAvg'].max(), 100)
    y = np.linspace(podSurfaceDF['ybinAvg'].min(), podSurfaceDF['ybinAvg'].max(), 100)
    X, Y = np.meshgrid(x, y)

    # todo: interpolate podSurfaceDF['xy_pod'], modify podSurfaceDF = podSurfaceDF where "xy_pod" is not none (remove them)

    podSurfaceDF.dropna(subset=['xy_pod'])
    Z = griddata((podSurfaceDF['xbinAvg'], podSurfaceDF['ybinAvg']), podSurfaceDF['xy_pod'], (X, Y), method='cubic')
    # For values greater than 1, set values equal to 1
    Z[Z >= 1] = 1

    plt.rcParams.update({'font.size': 7})
    if fig is None:
        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()
    elif axes:
        ax = axes
        fig = fig
    else:
        ax = plt.gca()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(cbarPosition, size="7%", pad=0.2)
    cmap = mpl.cm.get_cmap('jet')
    surf = ax.contourf(X, Y, Z, np.arange(0, 1.1, 0.1), cmap=cmap, vmax=1)
    if cbarPosition == 'top' or cbarPosition == 'bottom':
        fig.colorbar(surf, orientation="horizontal", cax=cax, label="POD", shrink=0.5, aspect=5)
        cax.xaxis.set_ticks_position(cbarPosition)
        cax.xaxis.set_label_position(cbarPosition)
    elif cbarPosition == 'right' or cbarPosition == 'left':
        fig.colorbar(surf, orientation="vertical", cax=cax, label="POD", shrink=0.5, aspect=5)
        cax.xaxis.set_ticks_position(cbarPosition)
        cax.xaxis.set_label_position(cbarPosition)

    xLabel = xLabel.replace(' ', '_')
    yLabel = yLabel.replace(' ', '_')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)

    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)

    return ax


def surfaceplots(df, xDataField, yDataField, nxBins=3, nyBins=3,xLabel=None, yLabel=None,
                 plotByPOD="Yes", fileName='surfacePlot.png', filePath=None):
    _, podDF = calcPODSurfaceWEvenCounts(df, xCategory=xDataField, yCategory=yDataField,
                                         nxBins=nxBins, nyBins=nyBins)
    ax = modifiedbuildPODSurface(podSurfaceDF=podDF, xLabel=xLabel, yLabel=yLabel,
                                 fileName=fileName, filePath=filePath)
    return ax


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


def modifiedlogisticRegression(Df, filePath=None, fileName=None, desiredLDLFraction=0.90, Nbootstrap=500,
                       xCategory='tc_EPBFE', xlabel='Whole gas emission rate', xunits='slpm', xmax=None,
                       xstep=0.01, figsize=None, fig=None, axes=None, digits=1, paperDict=None, varPrefix=None,
                       ScaleFactor=1):

    # Filtering out the desired columns from the classified dataframe and classifying detections as 1 or 0
    CRDF = Df.loc[Df['tc_Classification'].isin(['TP', 'FN'])]
    logisticDf = CRDF.filter([xCategory, 'tc_Classification'], axis=1)
    logisticDf['Detection'] = logisticDf['tc_Classification'].apply(lambda x: 1 if x == 'TP' else 0)
    if xunits is not None:
        logisticDf[xCategory] = logisticDf[xCategory].apply(lambda y: y*ScaleFactor)
    xDatafield = logisticDf[xCategory]
    if xmax:
        topFlowRate = xmax
    else:
        topFlowRate = xDatafield.max()

    # Fitting the data with the logistic regression model
    X = np.array(xDatafield.tolist()).reshape(-1, 1)  # X must be 2 dimensional
    X = sm.add_constant(X)  # Allows for the calculation of the model fit intercept
    Y = np.array(logisticDf['Detection'].tolist())
    logisticmodel = sm.Logit(Y, X)  # Creating an instance of the model
    fittedmodel = logisticmodel.fit(disp=0)  # Fitting the model
    modelCoef = fittedmodel.params[1]  # model fit coefficient
    modelIntercept = fittedmodel.params[0]  # model fit intercept
    p_values = fittedmodel.pvalues  # fitting P(significance) values
    B = [-modelIntercept, -modelCoef]
    # NOTE: I added the negative signs to the coefficients because I observed that I was getting the inverse of what Dan got without it.

    # Compute the desired lower detection limit
    ldl = invlogistic(desiredLDLFraction, B)

    # Create the background cloud of Monte Carlo fits
    idx = np.random.randint(1, len(logisticDf), size=(len(logisticDf), Nbootstrap))
    Bmc = np.zeros((2, Nbootstrap))  # Container for the bootstrapped model coefficients
    LDLmc = np.zeros((1, Nbootstrap))  # Container for the bootstrapped lower detection limit

    # Bootstrapping
    for i in list(range(Nbootstrap)):
            try:
                xbot = np.array(xDatafield.iloc[idx[:, i]].tolist()).reshape(-1, 1)  # Sampling from the given column via indexes corresponding to the elements of the column idx[:, i] and changing the dimension to match the expected form of X
                ybot = np.array(logisticDf['Detection'].iloc[idx[:, i]].tolist())  # Sampling from the given column via indexes corresponding to the elements of the column idx[:, i]
                xbot = sm.add_constant(xbot)  # allows for the calculation of the model fit intercept
                botmodel = sm.Logit(ybot, xbot)  # creating an instance of bootstrapping model
                botstrappedModel = botmodel.fit(disp=0)  # Fitting the bootstrapping model
                botCoef = botstrappedModel.params[1]  # Coefficient
                botIntercept = botstrappedModel.params[0]  # intercept
                 # Filling up Bmc where each column consist of intercept and coefficent for each model for each bootstrapp iteration
                Bmc[0, i] = -botIntercept
                Bmc[1, i] = -botCoef
                botB = [-botIntercept, -botCoef]  # Bootstrapping: list of Intercept and Coeffiecient of the model at run i
                LDLmc[0, i] = invlogistic(desiredLDLFraction, botB)
            except:
                # in some cases the randoms drawn will fail logistic regression.  In this case, catch and fill with None.
                Bmc[0, i] = None
                Bmc[1, i] = None
                LDLmc[0, i] = None
                continue

    # Generate the cloud plot
    #Xmc = list(np.arange(xstep, topFlowRate + xstep, xstep))
    Xmc = list(np.arange(0, topFlowRate + xstep, xstep))
    Ymc = np.zeros((len(Xmc), Nbootstrap))
    for j in list(range(Nbootstrap)):  # Calculating POD using the logistic function for each element in Xmc
        i = 0
        if Bmc[:, j] is not None:  # catch incase logistic regression failed above on iteration j.
            for x in Xmc:
                try:
                    coeff = list(Bmc[:, j])
                    Ymc[i, j] = logistic(coeff, x)
                    i = i + 1
                except:
                    Ymc[i, j] = None
                    i = i + 1

    # Generate the Regression plot
    #xReg = list(np.arange(xstep, topFlowRate + xstep, xstep))
    xReg = list(np.arange(0, topFlowRate + xstep, xstep))
    yReg = []  # Calculating the POD using logistics function for the actual logistics model
    for x in xReg:
            yReg.append(logistic(B, x))


    # Plot Labelling
    numerator = '1'
    denominator = '1 + e^{%5.3f%5.3f*x}' % (B[0], B[1])
    label = r'logistics:$\frac{1}{%s}$ (p=%.3e,%.3e)' % (denominator, p_values[0], p_values[1])
    percentage = "{:.0%}".format(desiredLDLFraction)

    # Preparing to plot
    Xmc = np.tile(np.array(Xmc).reshape(-1, 1), Nbootstrap)  # An Array that contains same columns to be used for plotting
    Xmc = pd.DataFrame(Xmc)
    Ymc = pd.DataFrame(Ymc)

    # Plotting
    plt.rcParams.update({'font.size': 7})

    # fig, ax1 = plt.subplots(figsize=(3.54331, 3.54331 / 1.5))

    if fig is None:
        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()
    elif axes:
        ax = axes
        fig = fig
    else:
        ax = plt.gca()

    # Plotting the bootstraps
    bsColor = '#B0C4DE'  # lightsteelblue https://www.webucator.com/article/python-color-constants-module/
    for p in list(range(Nbootstrap)):  # To plot each individual bootstrapped curve
        ax.plot(Xmc[p], Ymc[p], ls='solid', lw=0.5, c=bsColor, marker='None', zorder=1)
    # Plotting Regression
    ax.plot(xReg, yReg, ls='solid', lw=3, c='b', marker='None', label=label, zorder=2)
    # Plotting Detection threshold
    ax.plot([ldl, ldl], [0, desiredLDLFraction], ls=':', lw=0.5, c='k', marker='None', zorder=2,
             label=('{percent} Detection Limit = {ldl:4.' + str(digits) +'f} {units}').format(percent=percentage, ldl=ldl, units=xunits))  # vertical line
    ax.plot([0, ldl], [desiredLDLFraction, desiredLDLFraction], ls=':', lw=0.5, c='k', marker='None', zorder=2)  # horizontal line
    # Plotting the detections
    logisticDf.plot.scatter(x=xCategory, y='Detection', c='k', ax=ax, label='TP & FN', zorder=3)
    # Finishing Touches
    ax.tick_params(axis='both', labelsize=16)
    ax.set_ylabel('Probability of Detection', fontsize=16)
    ax.set_xlabel(xlabel+" ("+xunits+")", fontsize=16)
    if xmax:
        ax.set_xlim([0, xmax])
    else:
        ax.set_xlim([0, topFlowRate])
    ax.set_ylim([0, 1])
    fontP = FontProperties()
    fontP.set_size(14)
    ax.legend(loc='lower right', prop=fontP)
    plt.tight_layout()
    plt.grid(axis='both', alpha=0.5)

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)


    if paperDict:
        paperDict[varPrefix + 'ldlPercent'] = round(desiredLDLFraction*100, 0)
        paperDict[varPrefix + 'ldl'] = round(ldl, digits)
        paperDict[varPrefix + 'Intercept'] = round(B[0], 3)
        paperDict[varPrefix + 'Coeff'] = round(B[1], 3)
        paperDict[varPrefix + 'Intercept_PVal'] = p_values[0]
        paperDict[varPrefix + 'Coeff_PVal'] = p_values[1]

    return fig, ax, paperDict

def scatterWithHistogram(DF, xDataField, yDataField, catDataField, cats=None, s=25,
                        xlabel=None, xlimits=None, xticks=None, xticklabels=None, xscale="linear", xScaleFactor=1,
                        ylabel=None, ylimits=None, yticks=None, yticklabels=None, yscale="linear", yScaleFactor=1,
                        xunit=None, yunit=None, density=False, figsize=(5, 5), xhistTick=None, yhistTick=None,
                        gridAlpha=0.3, filePath=None, fileName=None):

    # setting up the plot zone
    plt.rcParams.update({'font.size': 7})
    fig = plt.figure(figsize=figsize)

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between the size of the marginal axes
    # and the main axes in both directions. Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7), left=0.1,
                          right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)

    # definitions for the axes
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # make a copy of DF and scale values in the copy
    df = DF.dropna(axis=0, how='any', subset=[xDataField, yDataField], inplace=False).copy(deep=True)
    df[xDataField] = df[xDataField] * xScaleFactor
    df[yDataField] = df[yDataField] * yScaleFactor

    if cats == None:
        # if no categories are specified, then use all unique values of catDataField
        df = df.sort_values(by=catDataField)
        cats = df[catDataField].unique()
    markertypes = itertools.cycle(['o', 's', 'd', 'h', 'p', '^'])

    # plot each category as it's own series
    for cat in cats:
        tempDF = df.loc[(df[catDataField] == cat)]
        x = tempDF[xDataField]
        y = tempDF[yDataField]
        ax.scatter(x, y, s=s, edgecolors='black', label=cat, marker=next(markertypes))

    # format x-axis based on method args
    ax.set_xscale(xscale)
    if xticks:
        ax.set_xticks(xticks)
    if xticklabels:
        ax.set_xticklabels(xticklabels)
    if xlabel:
        ax.set_xlabel(xlabel + " (" + xunit + ")")
    if xlimits:
        ax.set_xlim(xlimits)

    # format y-axis based on method args
    ax.set_yscale(yscale)
    if yticks:
        ax.set_yticks(yticks)
    if yticklabels:
        ax.set_yticklabels(yticklabels)
    if ylabel:
        ax.set_ylabel(ylabel + " (" + yunit + ")")
    if ylimits:
        ax.set_ylim(ylimits)
    ax.legend(loc='upper right')
    fig.tight_layout()

    # Plotting the histograms
    xData = list()
    yData = list()
    for cat in cats:
        filt = df[catDataField] == cat
        xData.append(df.loc[filt, xDataField].values)
        yData.append(df.loc[filt, yDataField].values)

    # Plotting vertical histogram
    xBins = ax.get_xticks()
    n1, _, patches = ax_histx.hist(xData, bins=xBins, histtype='bar', stacked=True, label=cats, edgecolor='black', density=density)
    a = n1[0]
    b = n1[1]
    count1 = maxList(a, b)
    dummy1 = [p.get_x() for p in ax_histx.patches]
    x = unique(dummy1)
    W1 = [p.get_width() for p in ax_histx.patches][0:len(x)]
    position1 = [(i+j/2) for i, j in zip(x, W1)]
    ax1 = ax_histx.twiny()
    ax1.set_xlim(ax_histx.get_xlim())
    ax1.set_xticks(position1)
    ax1TickLabel = [f'{int(x)}' for x in count1]
    ax1.set_xticklabels(ax1TickLabel)
    ax1.set_xlabel('count of controlled releases per bin')

    # Plotting horizontal histogram
    yBins = ax.get_yticks()
    n2, _, patches = ax_histy.hist(yData, bins=yBins, orientation='horizontal', histtype='bar', stacked=True, label=cats, edgecolor='black', density=density)
    a = n2[0]
    b = n2[1]
    count2 = maxList(a, b)
    dummy2 = [p.get_y() for p in ax_histy.patches]
    y = unique(dummy2)
    W2 = [p.get_height() for p in ax_histy.patches][0:len(y)]
    position2 = [(i + j / 2) for i, j in zip(y, W2)]
    ax2 = ax_histy.twinx()
    ax2.set_ylim(ax_histy.get_ylim())
    ax2.set_yticks(position2)
    ax2TickLabel = [f'{int(y)}' for y in count2]
    ax2.set_yticklabels(ax2TickLabel)
    ax2.set_ylabel('count of controlled releases per bin')

    # Formatting the axis of the histograms
    ax_histx.grid(alpha=gridAlpha)
    ax_histy.grid(alpha=gridAlpha)
    ax_histx.set_ylabel("Count")
    ax_histy.set_xlabel("Count")
    ax_histx.set_yticks(xhistTick)
    ax_histx.set_yticklabels(xhistTick, rotation=0)
    ax_histy.set_xticks(yhistTick)
    ax_histy.set_xticklabels(yhistTick, rotation=90)

    # Save figure
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)
    plt.close()  # close current figure

    return fig, ax

def windroseplot(df, windData, directionData, figsize=None, fileName='WindrosePlot.png', filePath=None):
    """
    What we know:
    - O is the True North of the wind direction data
    - Wind direction data are represented as degrees from North, clockwise
    - The windrose plot moves in the clockwise direction with 0 as the true North
    - Since the wind direction data are given as degrees from North in a counterclockwise direction and the windrose
    plots in the clockwise direction, this would be corrected by subtracting the wind direction angles from 360
    """
    plt.rcParams.update({'font.size': 7})
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(8, 8))
    plt.grid(axis='both', alpha=0.5)

    df.sort_values(by=directionData, ascending=True, inplace=True)
    register_projection(WindroseAxes)

    ws = df[windData].tolist()
    wd = df[directionData].tolist()
    #wd = [360 - x for x in wd]

    ax = WindroseAxes.from_ax()
    ax.bar(wd, ws, normed=True, opening=1.0, edgecolor='white')

    color_s = ['red', 'blue', 'lime', 'yellow', 'violet', 'aqua', 'pink', 'grey', 'darkred', 'navy', 'green']
    ax.set_legend(title='Wind Speed in m/s', bbox_to_anchor=(0.95, 1), loc='upper left',
                  handles=color_s, borderaxespad=0.)

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    labels = ["E", "N-E", "N", "N-W", "W", "S-W", "S", "S-E"]
    ax.set_thetagrids(angles=angles, labels=labels)

    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)

    return fig, ax

def whiskHistPlot(df,xDataField, yDataField, subDataField, subCats=None,
                  y1label=None, x2label=None, y2label=None, yunit=None, yScaleFactor=1,
                  gridAlpha=0.3, fileName=None, filePath=None):

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(4, 7))
    tempDF = df.dropna(axis=0, how='any', subset=[xDataField, yDataField], inplace=False).copy(deep=True)
    tempDF[yDataField] = tempDF[yDataField].apply(lambda y: y * yScaleFactor)

    tempDF = tempDF.sort_values(by=xDataField)
    xCats = list(tempDF[xDataField].unique())
    if subCats is None:
        subCats = list(tempDF[subDataField].unique())

    boxplot_data = []
    percentEP = []
    barchart_data = []
    #For the box and Whiskers Plot
    for cat in xCats:
        filt = (tempDF[xDataField] == cat)
        subdf = tempDF.loc[filt]
        EpFrac = (len(subdf)/len(df))*100
        percentEP.append(float(EpFrac))
        boxplot_data.append(np.array(subdf.loc[filt, yDataField].values))
    #For the bar plot
    for yCat in subCats:
        filt1 = (tempDF[subDataField] == yCat)
        barchart_data.append(tempDF.loc[filt1, xDataField].values)

    nc = len(set(tempDF[xDataField]))
    ax1.hist(barchart_data, bins=nc, rwidth=0.7, histtype='bar', stacked=True, label=subCats, edgecolor='black')
    ax1.set_xticks(np.linspace(0, nc - 1, 2 * nc + 1)[1::2])
    ax1.legend(loc='upper left')

    ax1.set_ylabel(y1label)
    ax1.grid(alpha=gridAlpha)
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks(np.linspace(0, nc - 1, 2 * nc + 1)[1::2])
    ax3.set_xticklabels(percentEP)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.set_xlabel('Percentage of Emission Points (%)')

    centers = list(np.linspace(0, nc - 1, 2 * nc + 1)[1::2])
    bplot = ax2.boxplot(boxplot_data,
                         vert=True,
                         sym='+',
                         patch_artist=True,
                         showmeans=True,
                         whis=(0, 100),
                         positions=centers,
                         meanline=True,
                         labels=xCats)
    colors = ['pink', 'lightblue', 'lightgreen', 'lime', 'yellow', 'aqua', 'navy']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylabel(y2label + " (" + yunit + ")")
    ax2.set_xlabel(x2label)
    ax2.grid(alpha=gridAlpha)
    ax2.legend(handles=[bplot['medians'][0], bplot['means'][0]], labels=['median', 'mean'])
    fig.tight_layout()

    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)
    return

def whiskPlot(df,xDataField, yDataField, subDataField, subCats=None, x1label="Equipment Group",
             y2label=None, yunit=None, yScaleFactor=1, gridAlpha=0.3, fileName=None, filePath=None):

    fig, ax1 = plt.subplots(figsize=(3.54331, 3.54331 / 1.1))
    tempDF = df.dropna(axis=0, how='any', subset=[xDataField, yDataField], inplace=False).copy(deep=True)
    tempDF[yDataField] = tempDF[yDataField].apply(lambda y: y * yScaleFactor)

    tempDF = tempDF.sort_values(by=xDataField)
    xCats = list(tempDF[xDataField].unique())

    boxplot_data = []
    percentEP = []
    #For the box and Whiskers Plot
    for cat in xCats:
        filt = (tempDF[xDataField] == cat)
        subdf = tempDF.loc[filt]
        EpFrac = (len(subdf)/len(df))*100
        percentEP.append(float(EpFrac))
        boxplot_data.append(np.array(tempDF.loc[filt, yDataField].values))

    nc = len(set(tempDF[xDataField]))
    centers = list(np.linspace(0, nc - 1, 2 * nc + 1)[1::2])
    bplot = ax1.boxplot(boxplot_data,
                        vert=True,
                        sym='+',
                        showmeans=True,
                        patch_artist=False,
                        whis=(0, 100),
                        positions=centers,
                        meanline=True,
                        labels=xCats)

    ax1.set_xticks(np.linspace(0, nc - 1, 2 * nc + 1)[1::2])
    ax1.legend(handles=[bplot['medians'][0], bplot['means'][0]], labels=['median', 'mean'], loc="upper right")
    if x1label:
        ax1.set_xlabel(x1label)
    ax1.set_ylabel(y2label + " (" + yunit + ")")
    ax1.grid(alpha=gridAlpha)
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticks(np.linspace(0, nc - 1, 2 * nc + 1)[1::2])
    ax3TickLabel = [f'{x:.2f}' for x in percentEP]
    ax3.set_xticklabels(ax3TickLabel)
    ax3.set_xlabel('Percentage of Emission Points (%)')
    fig.tight_layout()

    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)
    return

def PieHist(df, DataField, Cats=None, barData=None, barLabels=None,labels=None, plotByInsert=None,
            zoomViewYlim=None, fileName='FractionOfDetectionClassification.png', filePath=None,
            legendTitle="Detection Classification", paperDict=None, varPrefix=None):

    if plotByInsert==True:
        fig, ax = plt.subplots(figsize=(6, 8), subplot_kw=dict(aspect="equal"))
        data = []  # In this order %FP, %FN, %TP
        if Cats == None:
            Cats = ['FP', 'FN', 'TP']

        filt = df[DataField].isin(Cats)
        tempDF = df.loc[filt]
        nTD = len(tempDF)  # Total Number of Classified Detections
        nFP = len(tempDF.loc[df[DataField] == "FP"])  # Total Number of False Positive Detections
        nFN = len(tempDF.loc[df[DataField] == "FN"])  # Total Number of False Negative Detections
        nTP = len(tempDF.loc[df[DataField] == "TP"])  # Total Number of True Positive Detections

        if labels == None:
            labels = Cats
        data.append(nFP)
        data.append(nFN)
        data.append(nTP)
        if barData == None:
            barData = []  # In this order (1) fraction of FP to TP, (2) fraction of FP to FN
            barData.append(float(np.round((nFP / nTP), 2)))
            barData.append(float(np.round((nFP / nFN), 2)))
            if barLabels == None:
                barLabels = ['FP to TP', 'FP to FN']

        def func(pct, allvals):
            absolute = int(np.round(pct / 100. * np.sum(allvals)))
            return "{:.1f}%\n({:d})".format(pct, absolute)

        wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                          textprops=dict(color="w"))

        ax.legend(wedges, Cats,
                  title=legendTitle,
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))

        axins=ax.inset_axes([1, 0.8, 1, 1], transform=ax.transData)
        axins.set_ylim(zoomViewYlim)
        axinslabels = ['FP to TP', 'FP to FN']
        x = np.arange(len(axinslabels))  # the label locations
        width = 0.35  # the width of the bars
        rects = axins.bar(x, barData, width, color='#ffcc99')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axins.set_ylabel("Ratio")
        axins.set_xticks(x, axinslabels)
        axins.bar_label(rects, padding=3)
        plt.setp(autotexts, size=8, weight="bold")

        float(np.round((nFP / nTP), 2))
        if paperDict:
            paperDict[varPrefix + 'FalseNegativeFraction'] = float(np.round((nFN/(nTP+nFN+nFP)), 3))
            paperDict[varPrefix + 'FalsePositiveFraction'] = float(np.round((nFP/(nTP+nFN+nFP)), 3))
            paperDict[varPrefix + 'TruePositiveFraction'] = float(np.round((nTP/(nTP+nFN+nFP)), 3))
            paperDict[varPrefix + 'FPtoTP_Ratio'] = float(np.round((nFP/nTP), 2))
            paperDict[varPrefix + 'FPtoFN_Ratio'] = float(np.round((nFP/nFN), 2))

    elif plotByInsert == True:
        if Cats == None:
            Cats = ['FP', 'FN', 'TP']

        filt = df[DataField].isin(Cats)
        tempDF = df.loc[filt]
        nTD = len(tempDF)  # Total Number of Classified Detections
        nFP = len(tempDF.loc[df[DataField] == "FP"])  # Total Number of False Positive Detections
        nFN = len(tempDF.loc[df[DataField] == "FN"])  # Total Number of False Negative Detections
        nTP = len(tempDF.loc[df[DataField] == "TP"])  # Total Number of True Positive Detections
        if labels == None:
            labels = Cats
        if barData == None:
            barData = []  # In this order (1) fraction of FP to TP, (2) fraction of FP to FN
            barData.append(float(np.round((nFP / nTP), 2)))
            barData.append(float(np.round((nFP / nFN), 2)))
            if barLabels == None:
                barLabels = ['FP to TP', 'FP to FN']
        overall_ratios = [] #In this order %FP, %FN, %TP
        overall_ratios.append(float(np.round((nFP/nTD), 2)))
        overall_ratios.append(float(np.round((nFN/nTD), 2)))
        overall_ratios.append(float(np.round((nTP/nTD), 2)))
        # style choice
        plt.style.use('fivethirtyeight')

        # make figure and assign axis objects
        fig = plt.figure(figsize=(15, 7.5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        explode = [0.1, 0, 0]

        # rotate so that first wedge is split by the x-axis
        angle = -180 * overall_ratios[0]
        ax1.pie(overall_ratios, autopct='%1.1f%%', startangle=angle,
                labels=labels, explode=explode)

        xpos = 0
        bottom = 0
        width = .2
        colors = ['y', 'm', '#99ff99', '#ffcc99']

        for j in range(len(barData)):
            height = barData[j]
            ax2.bar(xpos, height, width, bottom=bottom, color=colors[j])
            ypos = bottom + ax2.patches[j].get_height() / 2
            bottom += height
            ax2.text(xpos, ypos, "%d%%" %
                     (ax2.patches[j].get_height() * 100), ha='center')

        plt.title('Percentage of False Positives')
        plt.legend(('FP to TP', 'FP to FN'))
        plt.axis('off')
        plt.xlim(-2.5 * width, 2.5 * width)

        # use ConnectionPatch to draw lines between the two plots
        # get the wedge data for the first group
        theta1, theta2 = ax1.patches[0].theta1, ax1.patches[0].theta2
        center, r = ax1.patches[0].center, ax1.patches[0].r
        bar_height = sum([item.get_height() for item in ax2.patches])
        x = r * np.cos(math.pi / 180 * theta2) + center[0]
        y = np.sin(math.pi / 180 * theta2) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, bar_height), xyB=(x, y),
                              coordsA="data", coordsB="data", axesA=ax2, axesB=ax1)
        con.set_color([0, 0, 0])
        con.set_linewidth(4)
        ax2.add_artist(con)

        x = r * np.cos(math.pi / 180 * theta1) + center[0]
        y = np.sin(math.pi / 180 * theta1) + center[1]
        con = ConnectionPatch(xyA=(-width / 2, 0), xyB=(x, y),
                              coordsA="data", coordsB="data", axesA=ax2, axesB=ax1)
        con.set_color([0, 0, 0])
        ax2.add_artist(con)
        con.set_linewidth(4)
        fig.tight_layout()
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)

    return

def PieChart(DF, DataField, Cats=None, labels=None, fileName=None, filePath=None, legendTitle="Classification",
             figsize=(3, 2), colors=None):

    # Checking if there is data in the column
    if DF[DataField].isnull().all():
        return
    df = DF.dropna(axis=0, how='any', subset=[DataField], inplace=False).copy(deep=True)
    fig, ax = plt.subplots(figsize=figsize)

    # Non string data
    if type(df[DataField].tolist()[0]) == float:
        data = []
        pUE = df[DataField].min()
        for UE in Cats:
            filt = (df[DataField] > pUE) & (df[DataField] <= UE)
            data.append(len(df.loc[filt]))
            pUE = UE
        if len(df)-sum(data) > 0:
            data.append(len(df)-sum(data))
    else:
        if Cats is None:
            Cats = df[DataField].unique()
        filt = df[DataField].isin(Cats)
        tempDF = df.loc[filt]
        data = []

        for cat in Cats:
            data.append(len(tempDF.loc[df[DataField] == cat]))

    def func(pct, allvals):
        absolute = int(np.round(pct / 100. * np.sum(allvals)))
        if pct == float(0):
            tag = ''
        else:
            tag = "{:.1f}%\n({:d})".format(pct, absolute)
        return tag

    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"), colors=colors)
    kw = dict(arrowprops=dict(arrowstyle="->"), va="center")

    #for p, label in zip(wedges, texts):
    #    ang = np.deg2rad((p.theta1 + p.theta2) / 2)
    #    y = np.sin(ang)
    #    x = np.cos(ang)
    #    horizontalalignment = "center" if abs(x) < abs(y) else "right" if x < 0 else "left"
    #    ax.annotate(label, xy=(0.75 * x, 0.75 * y), xytext=(1.3 * x, 1.3 * y),
    #                horizontalalignment=horizontalalignment, **kw)

    if labels == None:
        labels = Cats

    ax.legend(wedges, labels, title=legendTitle, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    fig.tight_layout()
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)

    return

def generalPlot(df, xDataField, yDataField, Cats=None, catplot=True, xLabel=None, yLabel=None,
                fileName=None, filePath=None):
    if catplot:
        sns.set_theme(style="ticks", color_codes=True)
        g = sns.catplot(x=xDataField, y=yDataField, order=Cats, data=df)
        g.set_axis_labels(xLabel, yLabel)
    else:
        ax = sns.boxplot(x=xDataField, y=yDataField, data=df)
        ax = sns.swarmplot(x=xDataField, y=yDataField, data=df, color=".25")
        ax.set_xlabel(xLabel, fontsize=20)
        ax.set_ylabel(yLabel, fontsize=20)
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path)
    return

def HistPOD(DF, xDataField, catDataField, upperXlimit=None, xScaleFactor=1, upperCatLimit=None, xLabel=None,
            xunit=None, catUnit=None, catScaleFactor=1, fileName=None, filePath=None):

    DF.dropna(axis=0, how='any', subset=[xDataField], inplace=True)
    df = DF.loc[DF['tc_Classification'].isin(['TP', 'FN'])]

    df[xDataField] = df[xDataField].apply(lambda y: y * xScaleFactor)
    df[catDataField] = df[catDataField].apply(lambda y: y * catScaleFactor)

    keyWordList = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15",
                   "X16", "X17", "X18", "X19", "X20"]
    countKey = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15",
                    "C16", "C17", "C18", "C19", "C20"]

    Det = ["TP", "FN"]
    xTicksLabels = [] #x-axix tick labels
    legendLabels = [] #labels for the legends
    xDict={}
    countDict={}
    nCats = len(upperCatLimit)
    n = 0
    while n < nCats:
        xDict[keyWordList[n]] = []
        countDict[countKey[n]] = []
        n = n + 1
    nCount = []
    xPreviousUE = 0
    for xUE in upperXlimit:
        xFilt = (df[xDataField] > xPreviousUE) & (df[xDataField] <= xUE) & ~(df[catDataField].isna())
        subDF = df.loc[xFilt]
        nCount.append(len(subDF))
        catPreviousUE = 0
        j = 0
        for cUE in upperCatLimit:
            catFilt = (subDF[catDataField] > catPreviousUE) & (subDF[catDataField] <= cUE)
            catDF = subDF.loc[catFilt]
            if len(catDF) > 0:
                nTP = len(catDF.loc[catDF["tc_Classification"] == Det[0]])
                nFN = len(catDF.loc[catDF["tc_Classification"] == Det[-1]])
                pod = float(np.round(nTP/(nTP+nFN), 2))
                if nTP == 0:
                    cKey = countKey[j]
                    countDict[cKey].append("")
                    countDict.update(dict([(cKey, countDict[cKey])]))
                else:
                    cKey = countKey[j]
                    countDict[cKey].append(len(catDF))
                    countDict.update(dict([(cKey, countDict[cKey])]))
            else:
                pod = 0
                cKey = countKey[j]
                countDict[cKey].append("")
                countDict.update(dict([(cKey, countDict[cKey])]))
            key = keyWordList[j]
            xDict[key].append(pod)
            xDict.update(dict([(key, xDict[key])]))
            if catUnit == None:
                legendLabels.append(f"{catPreviousUE}-{cUE}")
            else:
                legendLabels.append(f"{catPreviousUE}-{cUE} {catUnit}")
            catPreviousUE = cUE
            j = j + 1
        xTicksLabels.append(f"{xPreviousUE}-{xUE}")
        xPreviousUE = xUE

    legendLabels = legendLabels[:nCats]

    # Modifying the dataframe column header
    for y in range(0, nCats):
        xDict[legendLabels[y]] = xDict.pop(keyWordList[y])
    xDict["xTicksLabels"] = xTicksLabels
    Data = pd.DataFrame.from_dict(xDict)
    ax = Data.plot(x='xTicksLabels',
            kind='bar',
            stacked=False,
            xlabel=f"{xLabel} ({xunit})",
            ylabel='Probability of Detection',
            rot=0)

    # Combining the lists of counts in order
    dummy = []
    for i in range(0, nCats):
        dummy.append(countDict[countKey[i]])
    nc = list(itertools.chain.from_iterable(dummy))

    # Adding the count label on each bar
    c = 0
    for p in ax.patches:
        h = p.get_height()
        lb = nc[c]
        x = p.get_x() + p.get_width() / 2.
        ax.annotate(f'{lb}', xy=(x, h), xytext=(0, 4), rotation=90,
                    textcoords="offset points", ha="center", va="bottom")
        c = c + 1
    ax.yaxis.grid(visible=True, alpha=0.5)

    fontP = FontProperties()
    fontP.set_size(8)
    ax.legend(ncol=len(legendLabels), loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.08),
              borderaxespad=0, mode="expand", prop=fontP)

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)
    plt.close()  # close current figure
    return


def simpleBarPlot(DF, xData=None, yData=None, alpha=0.3, fileName=None, filePath=None, figSize=None, axes=None,
                  fig=None, xLabel=None, yLabel=None, dataWranglingMethod=None, paperDict=None):

    # Plotting parameter definition
    plt.rcParams.update({'font.size': 7})
    if fig is None:
        fig, ax = plt.subplots(figsize=figSize)
    elif axes:
        ax = axes
        fig = fig
    else:
        ax = plt.gca()

    # Selecting how the data to plotted should be wrangled
    if dataWranglingMethod == "allCategorizedRows":
        # This method is useful when categorizing rows according the count of unique elements in a column (xData),
        # and you are not intentionally dropping any row based on some conditions

        # Drop any row on column 'xData' with 'na'
        df = DF.dropna(axis=0, how='any', subset=[xData], inplace=False).copy(deep=True)
        # Create a new column 'count' that groups the rows according to the count of each unique element of 'xData'
        df = df.assign(count=lambda x: x.groupby([xData])[xData].transform("count"))
        # Extracts the "count" column
        counts = df["count"].replace(np.nan, 'None')
        # Extracts the unique elements of the "count" column and their frequency.
        labels, counts = np.unique(counts, return_counts=True)

    elif dataWranglingMethod == "selectedCategorizedRows":
        # This method is useful when categorizing rows according the count of unique elements in a column (xData),
        # and you are intentionally dropping any row based on some conditions. In this case we are keeping the first row.

        # Drop any row on column 'xData' with 'na'
        df = DF.dropna(axis=0, how='any', subset=[xData], inplace=False).copy(deep=True)
        # Create a new column 'count' that groups the rows according to the count of each unique element of 'xData'
        df = df.assign(count=lambda x: x.groupby([xData])[xData].transform("count"))
        # Select rows based on some condition
        df = df.drop_duplicates(subset=xData, keep="first")
        # Extracts the "count" column
        counts = df["count"].replace(np.nan, 'None')
        # Extracts the unique elements of the "count" column and their frequency.
        labels, counts = np.unique(counts, return_counts=True)

    elif dataWranglingMethod == "notCategorizedRows":
        # This method is used when you need not categorize the rows, no need to wrangle the data

        # Drop any row on column 'xData' with 'na'
        df = DF.dropna(axis=0, how='any', subset=[xData], inplace=False).copy(deep=True)
        # Extracts the "count" column
        counts = df[xData].replace(np.nan, 'None')
        # Extracts the unique elements of the "count" column and their frequency.
        labels, counts = np.unique(counts, return_counts=True)

    elif dataWranglingMethod == None:
        # This method is used when the x and y plot data are given as lists or arrays
        labels = xData
        counts = yData

    ticks = range(len(counts))
    ax.bar(ticks, counts, align='center', edgecolor='black')
    plt.xticks(ticks, labels)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    fig.tight_layout()
    plt.grid(axis='y', alpha=alpha)

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)
    plt.close()  # close current figure

    return

def simpleHistogram(DF, xData=None, alpha=0.3, fileName=None, filePath=None, figSize=None, axes=None, fig=None,
                    xLabel=None, yLabel=None):

    # Plotting parameter definition
    df = DF.dropna(axis=0, how='any', subset=[xData], inplace=False).copy(deep=True)
    plt.rcParams.update({'font.size': 7})

    if fig is None:
        fig, ax = plt.subplots(figsize=figSize)
    elif axes:
        ax = axes
        fig = fig
    else:
        ax = plt.gca()

    ax.hist(df[xData].tolist(), edgecolor='black')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    fig.tight_layout()
    plt.grid(axis='y', alpha=alpha)

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)
    plt.close()  # close current figure

    return

def quantification(DF=None, xData=None, yData=None, cData=None, xFactor=1, yFactor=1, cFactor=1, xLimits=None,
                   yLimits=None, xScale='linear', yScale='linear', xLabel=None, yLabel=None, xTicks=None, yTicks=None,
                   xunit=None, yunit=None, qFactor=(2, 3, 5), regression='linear', oneToOne=True, yQuantile=None, s=25,
                   axesEdgeColor='black', gridAlpha=0.5, figsize=(4, 4), dpi=400, fig=None, axes=None, fontsize=7,
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
        pt1 = (xMin, xMax)
        #pt1 = (0, xMax)
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

            # plot the upper bound of the quantification estimate for the factor
            # if (xScale == 'log') and (yScale == 'log'):
            #     ax.plot((0, xMax), (y1Upper, y2Upper), color='k', linestyle=lineStyle[counter], linewidth=2,
            #             label=f'{percentPoints}% within a factor of {i}')
            #     # plot the upper bound of the quantification estimate for the factor
            #     ax.plot((0, xMax), (y1Lower, y2Lower), color='k', linestyle=lineStyle[counter], linewidth=2)
            # else:
            #     ax.plot((0, xMax), (0, y2Upper), color='k', linestyle=lineStyle[counter], linewidth=2,
            #             label=f'{percentPoints}% within a factor of {i}')
            #     # plot the upper bound of the quantification estimate for the factor
            #     ax.plot((0, xMax), (0, y2Lower), color='k', linestyle=lineStyle[counter], linewidth=2)
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
                       dpi=400, fig=None, axes=None, fontsize=7, filePath=None, fileName=None, varPrefix=None,
                       vertLines=((-50.00, 100.00, 2), (-66.67, 200.00, 3), (-80.00, 400.00, 5)), paperDict=None,
                       showLegend=True):

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
    # Find the maximum multiple of 5 common to both lowerREFactors and upperREFactors
    #commonFactor = findTheMaxMultipleCommonToLists(lowerREFactors, upperREFactors, f=5)
    # Find the tick locations where you want tick labels to appear
    #markersOFinterest = list(itertools.chain.from_iterable([[1, 2, 3], list(range(5, commonFactor+1, 5))]))
    counter = 0
    x1TickLabel = []
    x2TickLabel = []
    maxBin = Bin1[-1]
    for i in Bin1:
        if i < 0:
            # if Bin2[counter] in markersOFinterest:
            #     x1TickLabel.append(i)
            #     x2TickLabel.append(Bin2[counter])
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

def parabolicModel(x, a, b):
    y = -a*(x**b)
    return y

def InverseParabolicModel(y, a, b):
    p = (y/a)**2
    x = (p)**(1/(2*b))
    return x

def alternativePodCurve(Df, filePath=None, fileName=None, desiredLDLFraction=0.90, Nbootstrap=500, xScaleFactor=1,
                        xData='tc_EPBFE', tData='Study Year', cData='tc_Classification', xlabel=None, ylabel=None,
                        xunits='slpm', figsize=(3.54331, 3.54331/1.1), fig=None, axes=None, digits=1, paperDict=None,
                        fontsize=9, varPrefix="", legendLOC='lower right', lowerBound=None, bootstrapp=False,
                        CFnBins=10, xMax=None, listOFnBins=(10), dpi=400):
    # Axes declaration
    plt.rcParams.update({'font.size': fontsize})

    if fig is None:
        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = plt.subplots()
    elif axes:
        ax = axes
        fig = fig
    else:
        ax = plt.gca()

    # Filtering out the desired columns from the classified dataframe
    CRDF = Df.loc[Df[cData].isin(['TP', 'FN'])]
    CRDF.dropna(axis=0, how='any', subset=[xData], inplace=True)
    selectedDF = CRDF.filter([tData, xData, cData], axis=1)
    selectedDF[xData] = selectedDF[xData].apply(lambda x: x * xScaleFactor)

    # Plot the lowerbound
    if lowerBound:
        ax.axvline(1, color='k', ls='--', linewidth=1.5, label='Arbitrary detection threshold')

    NumPointsPerBin = []
    xmin = selectedDF[xData].min()
    topFR = selectedDF[xData].max()
    if xMax is None:
        xmax = topFR
    else:
        xmax = xMax

    # pod calculation
    DataPoints = {'pods': [], 'xBinAvg': [], 'xBinLower': [], 'xBinUpper': []}
    AllbootstrappedPodDF = pd.DataFrame()
    for k, NB in enumerate(listOFnBins):
        try:
            xDataField = selectedDF[xData] # get a series of all xData
            # Divide all xData into bins of almost equal number of points
            bins, bin_edges = pd.qcut(xDataField, q=NB, precision=0, retbins=True, duplicates='drop')
            # Create a new column "Bins"
            selectedDF['Bins'] = bins
            # Summary of the count of each bin for all data
            BinCounts = selectedDF['Bins'].value_counts()
            # Summary of the count of each bin for true positives (TPs)
            TPCounts = selectedDF.loc[selectedDF[cData] == 'TP']['Bins'].value_counts()
            # Summary of the count of each bin for true positives (FNs)
            FNCounts = selectedDF.loc[selectedDF[cData] == 'FN']['Bins'].value_counts()
            # Iterate through all unique bins created
            for interval in BinCounts.index:
                # Get the count of points with the bin (interval)
                nBin = BinCounts[interval]
                NumPointsPerBin.append(nBin)
                # Get the number of points within the bin that was classified as TP
                if interval in TPCounts:
                    nTP = TPCounts[interval]
                else:
                    nTP = 0
                pod = nTP/nBin
                # Find the average of points that falls within the bin (interval)
                DF4Bin = selectedDF.loc[selectedDF['Bins'] == interval]
                binAvg = DF4Bin[xData].mean()
                xLower = DF4Bin[xData].min()
                xUpper = DF4Bin[xData].max()

                DataPoints['pods'].append(pod)
                DataPoints['xBinAvg'].append(binAvg)
                DataPoints['xBinLower'].append(xLower)
                DataPoints['xBinUpper'].append(xUpper)

                # Bootstrapping pods
                if bootstrapp:
                    i = 1
                    resample = 1
                    detectCounts = 0
                    nonDetectCounts = 0
                    bootstrappedPod = np.empty(shape=0)
                    while resample <= Nbootstrap:
                        # For each reSample calculate a detect or non detect
                        while i <= nBin:
                            value = random.uniform(0, 1)
                            if value <= pod:
                                detectCounts = detectCounts + 1
                            else:
                                nonDetectCounts = nonDetectCounts + 1
                            i = i + 1

                        # Calculate a new pod
                        if detectCounts + nonDetectCounts != 0:
                            POD = detectCounts / (detectCounts + nonDetectCounts)
                        else:
                            POD = 0

                        # Append POD to PODlist
                        bootstrappedPod = np.append(bootstrappedPod, np.array([POD]), axis=0)

                        # Increase resample and reset detect and non detect counts
                        resample = resample + 1
                        detectCounts = 0
                        nonDetectCounts = 0
                        i = 0

                    # Append all the bootstrap for this pod to a dataframe
                    bootstrappedPodDF = pd.DataFrame(bootstrappedPod)
                    bootstrappedPodDF = bootstrappedPodDF.T
                    AllbootstrappedPodDF = pd.concat([AllbootstrappedPodDF, bootstrappedPodDF])
                    AllbootstrappedPodDF = AllbootstrappedPodDF.reset_index(drop=True)
        except Exception as e:
            print(f"could not calculate pods for {NB} quartiles due to {e}")
            NumPointsPerBin = []  # Re-initialize count of points
            DataPoints = {'pods': [], 'xBinAvg': [], 'xBinLower': [], 'xBinUpper': []}  # Re-initialize
            AllbootstrappedPodDF = pd.DataFrame()  # Re-initialize
            continue

        # Get the number of points per Bin
        mean_nPoint = round(statistics.mean(NumPointsPerBin), 2)
        min_nPoint = min(NumPointsPerBin)
        max_nPoint = max(NumPointsPerBin)

        # Evaluate the correlation between emissionrate and pod using spearman rank correlation test
        x = np.array(DataPoints['xBinAvg'])
        y = np.array(DataPoints['pods'])
        correlation, p_value = spearmanr(x, y)

        # Fit the curve to the data
        try:
            params, covariance = curve_fit(parabolicModel, DataPoints['xBinAvg'], DataPoints['pods'],
                                           bounds=(-np.inf, [+np.inf, 1]))
        except Exception as e:
            print(f"could not fit curve due to {e}")
            NumPointsPerBin = []  # Re-initialize count of points
            DataPoints = {'pods': [], 'xBinAvg': [], 'xBinLower': [], 'xBinUpper': []}  # Re-initialize
            AllbootstrappedPodDF = pd.DataFrame()  # Re-initialize
            continue


        # Extract the optimized parameters
        a_opt, b_opt = params

        # Create a new x array for smooth curve plotting
        x_curve = np.linspace(0, xmax, 100)

        # Calculate the predicted y values using the fitted parameters
        y_curve = parabolicModel(x_curve, a_opt, b_opt)

        # Calculate the R^s of the model
        y_r = parabolicModel(np.array(DataPoints['xBinAvg']), a_opt, b_opt)
        R_square = r2_score(np.array(DataPoints['pods']), y_r)
        R_square = f'{round(R_square, 2):.2f}'

        # lower detection limit
        ldl = InverseParabolicModel(desiredLDLFraction, a_opt, b_opt)
        if ldl > (xmax * 20):
            ldl = None

        # Plot the fitted curve

        lb = ""
        marker = 'o'
        mColor = 'darkblue'
        curveColor = 'b'
        percentage = "{:.0%}".format(desiredLDLFraction)
        coeff = -1 * a_opt
        pow = b_opt
        label = r'%s curve fit: ${%5.3f*x}^{%5.3f}; R^2: {%s}$' % (lb, coeff, pow, R_square)
        subLabel = r'%s curve fits for various nPoint/bin' % (lb)
        bsColor = '#B0C4DE'
        # Plot the original data points
        if NB == CFnBins:
            ax.scatter(DataPoints['xBinAvg'], DataPoints['pods'], label=f'{lb} Data; nPoints/bin: {mean_nPoint} [{min_nPoint}, {max_nPoint}]',
                       color=mColor, s=20, zorder=2, marker=marker, edgecolors='k', linewidths=0.4)
            ax.plot(x_curve, y_curve, label=label, color=curveColor, lw=2, marker='None', zorder=2)
        else:
            if k == 0:
                ax.plot(x_curve, y_curve, color=curveColor, lw=1, ls=':', label=subLabel, marker='None', zorder=2)
            else:
                ax.plot(x_curve, y_curve, color=curveColor, lw=1, ls=':', marker='None', zorder=2)
        # Plot the lower bound

        # Plotting the bootstrap
        LDLs = []
        for p in list(range(Nbootstrap)):  # To plot each individual bootstrapped curve
            try:
                params_b, covariance_b = curve_fit(parabolicModel, DataPoints['xBinAvg'], AllbootstrappedPodDF[p],
                                                   bounds=(-np.inf, [+np.inf, 1]))
            except Exception as e:
                print(f"could not fit curve due to {e}")
                continue

            # Extract the optimized parameters
            a_b, b_b = params_b

            # Calculate the predicted y values using the fitted parameters
            y_curve_b = parabolicModel(x_curve, a_b, b_b)

            # Calculate all LDLs
            try:
                print(f"calculating LDL for iteration: {p}")
                b_ldl = InverseParabolicModel(desiredLDLFraction, a_b, b_b)
                LDLs.append(b_ldl)
            except Exception as e:
                print(f"could not calculate the LDL for iteration: {p} due to {e}")

            if NB == CFnBins:
                ax.plot(x_curve, y_curve_b, ls='solid', lw=0.5, c=bsColor, marker='None', zorder=1)

        # Check the LDLs calculated and create a label
        if LDLs: #if the list is empty
            if ldl is not None:
                lowerDL = np.percentile(LDLs, 2.5)
                upperDL = np.percentile(LDLs, 97.5)
                if upperDL >= (xmax * 20):
                    labUpperDL = "NA"
                    labLowerDL = round(lowerDL, digits)
                    lab = f'{ldl:.1f} [{labLowerDL:.1f}, {labUpperDL}]'
                else:
                    labUpperDL = round(upperDL, digits)
                    labLowerDL = round(lowerDL, digits)
                    lab = f'{ldl:.1f} [{labLowerDL:.1f}, {labUpperDL:.1f}]'
            else:
                labUpperDL = "NA"
                labLowerDL = 0
                lab = f'"NA" [{labLowerDL}, {labUpperDL}]'
        else:
            labUpperDL = "NA"
            labLowerDL = 0
            lab = f'"NA" [{labLowerDL}, {labUpperDL}]'

        # Plotting the detection limits
        if NB == CFnBins:
            if ldl is not None:
                if ldl > topFR or ldl < xmin:
                    ax.plot([ldl, ldl], [0, desiredLDLFraction], ls=':', lw=0.5, c='k', marker='None', zorder=2,
                            label=('{percent} Detection Limit is out of tested range').format(percent=percentage))
                else:
                    ax.plot([ldl, ldl], [0, desiredLDLFraction], ls=':', lw=0.5, c='k', marker='None', zorder=2,
                            label=('{percent} DL: {lab} {units}').format(percent=percentage, lab=lab, units=xunits))  # vertical line
                ax.plot([0, ldl], [desiredLDLFraction, desiredLDLFraction], ls=':', lw=0.5, c='k', marker='None',
                        zorder=2)  # horizontal line
            else:
                ax.axhline(y=desiredLDLFraction, ls=':', lw=0.5, c='k', marker='None', zorder=2,
                            label=('{percent} DL = NA').format(percent=percentage))

        #NumPointsPerBin = []
        # Add variables to paperDict
        binlab = 'n'+str(NB)
        if paperDict is not None:
            paperDict[varPrefix + binlab + 'rSquare'] = R_square
            paperDict[varPrefix + binlab + 'SpearmanCor'] = f'{round(correlation, 4):.4f}'
            paperDict[varPrefix + binlab + 'SpearmanPval'] = f'{round(p_value, 4):.4f}'
            paperDict[varPrefix + binlab + 'lowerLDL'] = labLowerDL
            paperDict[varPrefix + binlab + 'upperLDL'] = labUpperDL
            paperDict[varPrefix + binlab + 'minPPB'] = min_nPoint
            paperDict[varPrefix + binlab + 'maxPPB'] = max_nPoint
            paperDict[varPrefix + binlab + 'meanPPB'] = mean_nPoint
            if ldl is None:
                paperDict[varPrefix + binlab + 'LDL'] = "NA"
            else:
                paperDict[varPrefix + binlab + 'LDL'] = round(ldl, digits)

        # Re-initializing
        NumPointsPerBin = []  # Re-initialize count of points
        DataPoints = {'pods': [], 'xBinAvg': [], 'xBinLower': [], 'xBinUpper': []}  # Re-initialize
        AllbootstrappedPodDF = pd.DataFrame()

    # Finishing Touches
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if xMax:
        ax.set_xlim([0, xMax])
    else:
        ax.set_xlim([0, xmax])
    ax.set_ylim([0, 1])
    # Adding Title To Legends
    fontP = FontProperties()
    fontP.set_size(fontsize)
    lg = ax.legend(loc=legendLOC, prop=fontP)
    lg.get_frame().set_edgecolor('black')
    lg.get_frame().set_linewidth(0.5)

    fig.tight_layout()
    ax.grid(axis='both', alpha=0.1)
    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=dpi)

    return fig, ax, paperDict


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


def barhPlot(yVar, barTags=None, dataDict=None, fig=None, axes=None, height=0.5, xTick=None, yTick=None, yLabel=None,
             xLabel=None, yTickLabel=None, figSize=(7, 5), ncol=3, fontsize=7, fileName=None, filePath=None, alpha=0,
             ylimits=None, dpi=400, rotation=90, title=None):
    """Generate a horizontal bar plot showing for data in df.
             Required Parameters:
             :param yVar = Values for the y-axis.
             :param barTags = list of key names of each bar in dataDict.
             :param dataDict = A dictionary of the data, labels, and colors for the bar plot
             :param fig = the figure on which the plot is to be made
             :param axes = the axes within the figure on which the plot is made.
             :param height = Height of each bar
             :param xTick = The ticks position for x-axis data
             :param yLabel = Y-axis label
             :param xLabel = X-axis label
             :param yTickLabel = The label of the tick positions for x-axis data
             :param figSize = The figure size of the plot
             :param ncol = The number of columns for the legend
             :param fontsize = The fontsize for the labels
             :param fileName = The name assigned to the plot generated
             :param filePath = The path where the figure is to be saved
             :param alpha = The the visibility level of grid lines
             :param ylimits = The limits of the y-axis variable
             :param dpi = The measure of the quality of the image when saved.
             :param title = The title of the legend
             """
    # Plotting parameter definition
    plt.rcParams.update({'font.size': 7})
    if fig is None:
        fig, ax = plt.subplots(figsize=figSize)
    elif axes:
        ax = axes
        fig = fig
    else:
        ax = plt.gca()

    ax.tick_params(axis='y', which='both', pad=10)

    # Finding the markers for the bars for each yVar
    if len(barTags) <= 3:
        ind = list(range(len(yVar)))
    else:
        ind = list(range(0, (2*len(yVar)), 2))

    # The total number of bars per yVar.
    nBars = len(barTags)
    # Track labels to avoid repitition
    trackLabels = []
    # Start plotting individual bars
    for index, bar in enumerate(barTags):
        # Get the position of all the bars of the same data
        y = defineTick(ind, nBars, index, height)
        # A list of containers for all the bars in the stacked bar
        barcontainers = []
        # Iterate through the data to be plotted for each bar
        leftCat = [0]*len(yVar)
        for k, cat in enumerate(dataDict.get(bar).get('Data')):
            # Get color and label corresponding to each data
            color = dataDict.get(bar).get('Colors')[k]
            Label = dataDict.get(bar).get('Labels')[k]
            if Label in trackLabels:
                p = ax.barh(y=y, width=cat, align='center', height=height, left=leftCat, color=color,
                            edgecolor='black', linewidth=0.5)
            else:
                p = ax.barh(y=y, width=cat, align='center', height=height, left=leftCat, color=color, label=Label,
                            edgecolor='black', linewidth=0.5)
            trackLabels.append(Label)
            leftCat = [a+b for a, b in zip(leftCat, cat)]
            barcontainers.append(p)
            # Annotate each bar showing the percentage
            for b in list(range(len(yVar))):
                thisBar = p[b]
                l = dataDict.get(bar).get('Data')[k][b]
                if int(l) == 0:
                    continue
                else:
                    lab = round(l, 1)
                    _ = annotateBar(thisBar, ax, lab, labelPosition='center', rotation=0)
            #ax.bar_label(p, label_type='center', color='white', fontsize=fontsize, fmt=lambda x: '{:.1f}%'.format(x))
        barcontainers.append(barcontainers)
        count = dataDict.get(bar).get('Count')
        # Annotate the top of the bars with count
        #ax.bar_label(barcontainers[-1], padding=4, labels=[str(x) for x in count], fontsize=fontsize)
        for b in list(range(len(yVar))):
            thisBar = p[b]
            l = count[b]
            _ = annotateBar(thisBar, ax, round(l, 1), labelPosition='top', rotation=0)

    # Assign ticklabels to y-axis values
    ax.set_yticks(ind)
    if yTickLabel is None:
        yTickLabel = yVar
    if yTick is None:
        yTick = ind

    # Add x, y gridlines
    ax.grid(alpha=alpha)

    # Add Plot Title and Labels
    if yLabel:
        ax.set_ylabel(yLabel, fontsize=fontsize)
    if xLabel:
        ax.set_xlabel(xLabel, fontsize=fontsize)
    if xTick:
        ax.set_xticks(xTick)
    if yTick:
        ax.set_yticks(yTick)
    if yTickLabel:
        ax.set_yticklabels(yTickLabel, fontsize=fontsize, rotation=rotation)
    if ylimits:
        ax.set_ylim(ylimits)

    ax.margins(x=0)

    fontP = FontProperties()
    fontP.set_size(fontsize)
    leg = ax.legend(ncol=ncol, loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.08), borderaxespad=0,
                    mode="expand", prop=fontP, title=title)

    leg.get_frame().set_edgecolor('black')
    fig.tight_layout()

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=dpi)
    return ax, fig

def BoxWhisker(DF, xDataField, yDataField, xbinUpperEdges, xlabel=None, xlimits=None, xticks=None, xticklabels=None,
               xscale='linear', ylabel=None, ylimits=None, yticks=None, yticklabels=None, yscale='linear',
               yScaleFactor=1, quantile=None, percentiles=None, fig=None, axes=None, figsize=None, filePath=None,
               fileName=None, Hline=True, x2label='', whis=(5, 95), varPrefix="", paperDict=None):
    """
    Generate a box and whisker pot for data in df.  df[yDataField] is grouped into several subsets by df[xDataField].
     xbins are (lower, upper] where the upper edge of each bin is specified and the lower is taken as the previous bin's
     upper edge. The first bin is taken with a lower edge = 0.
     Required Parameters:
     :param df = dataframe to pull data from.
     :param xDataField = column in df to use for x axis
     :param yDataField = column in df to use for y data in each box & whisker set
     :param xbinUpperEdges = Array of upper edges for bins. e.g. [0.1,1,10,100]
     Optional parameters:
     :param xlabel = label for x axis of figure
     :param xlimits = limits of x-axis
     :param xticks = location of major tick marks for x-axis
     :param xticklabels = labels for major tick marks for x-axis
     :param xscale = scale for x axis ("linear" or "log")
     :param ylabel = label for y axis of figure
     :param ylimits = limits of y-axis
     :param yticks = location of major tick marks for y-axis
     :param yticklabels = labels for major tick marks for y-axis
     :param yscale = scale for y axis ("linear" or "log")
     :param figsize = (width, height) of figure in inches
     :param filePath = directory to save figure to. Defaults to cwd.
     :param fileName = if specified saves figure to filePath/fileName
     Return values:
     :returns fig =
     """
    # Remove values with Nan from the dataframe.
    df = DF.dropna(axis=0, how='any', subset=[xDataField, yDataField], inplace=False).copy(deep=True)

    # Using percentile to get the y-axis limits
    if percentiles:
        lwr = rounddown(np.percentile(a=np.array(df[yDataField].tolist()), q=percentiles[0])*yScaleFactor, 1)
        hgr = roundup(np.percentile(a=np.array(df[yDataField].tolist()), q=percentiles[1])*yScaleFactor, 1)
        if ylimits is None:
            ylimits = [lwr, hgr]

    # select data from df and format for box whisker
    yFilt = ~(df[yDataField].isna())
    yData = df.loc[yFilt, yDataField]*yScaleFactor
    yDataMin = yData.min()

    if quantile:
        yUpper = yData.quantile(quantile)

    data = list()
    centers = list()
    widths = list()
    counts = list()
    labels = list()

    previousUE = 0
    for UE in xbinUpperEdges:
        filt = (df[xDataField] > previousUE) & (df[xDataField] <= UE) & ~(df[yDataField].isna())
        c = list(df.loc[filt, yDataField] * yScaleFactor)
        mean_x = df.loc[filt, xDataField].mean()
        max_x = df.loc[filt, xDataField].max()
        min_x = df.loc[filt, xDataField].min()
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

    boxdict = ax1.boxplot(data, positions=centers, widths=widths, whis=whis, showmeans=True, meanline=True)

    if Hline:
        ax1.axhline(y=0, color="C7", linestyle='--')

    # format x-axis based on method args
    ax1.set_xscale(xscale)
    if xticks:
        ax1.set_xticks(xticks)
    if xticklabels:
        ax1.set_xticklabels(xticklabels)
    if xlabel:
        ax1.set_xlabel(xlabel)
    if xlimits:
        ax1.set_xlim(xlimits)

    # format y-axis based on method args
    ax1.set_yscale(yscale)
    if yticks:
        ax1.set_yticks(yticks)
    if yticklabels:
        ax1.set_yticklabels(yticklabels)
    if ylabel:
        ax1.set_ylabel(ylabel)
    if ylimits:
        ax1.set_ylim(ylimits)
    elif quantile:
        ax1.set_ylim(yDataMin, yUpper)

    # plot data counts on secondary x axis above figure
    ax2 = ax1.twiny()
    ax2.set_xscale(xscale)
    ax2.set_xlim(xlimits)
    tickshold = []
    countshold = []
    j = 0
    for i in xbinUpperEdges:
        if i <= xlimits[0]:
            tickshold.append(xlimits[0])
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
            xCenters.append(x-xlimits[0])
        for y in counts:
            xCounts.append(y)
        xCenters.pop()
        xCounts.pop()
        ax2.set_xticks(xCenters)
        ax2.set_xticklabels(xCounts)
    else:
        ax2.set_xticks(centers)
        ax2.set_xticklabels(counts)
    ax2.minorticks_off()
    ax2.set_xlabel(x2label)

    # turn on grid and tight layout
    fontP = FontProperties()
    fontP.set_size(8)
    ax1.legend(handles=[boxdict['medians'][0], boxdict['means'][0]], labels=['median', 'mean'], loc='upper right',
               prop=fontP, title=r'Whiskers:$({%d}^{th},{%d}^{th})$' % (whis[0], whis[1]))
    ax1.grid(alpha=0.3)

    if paperDict:
        paperDict[varPrefix + 'lowerWhis'] = whis[0]
        paperDict[varPrefix + 'upperWhis'] = whis[1]

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)

    return fig, ax1, ax2, paperDict

