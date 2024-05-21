import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def roundup(x,n):
    return int(math.ceil(x / float(n))) * n

def rounddown(x,n):
    return int(math.floor(x / float(n))) * n


def boxWhisker(df, xDataField, yDataField, xbinUpperEdges, xlabel=None, xlimits=None, xticks=None, xticklabels=None,
               xscale='linear', ylabel=None, ylimits=None, yticks=None, yticklabels=None, yscale='linear',
               figsize=(3.54331, 3.54331), filePath=None, fileName=None):
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
    """

    # Select data from df and format for box whisker
    data = list()
    centers = list()
    widths = list()
    counts = list()
    labels = list()

    # Plotting the boxes for the boxplots.
    previousUE = 0
    for UE in xbinUpperEdges:
        filt = (df[xDataField] > previousUE) & (df[xDataField] <= UE) & ~(df[yDataField].isna())
        c = list(df.loc[filt, yDataField])
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

    # Setting the parameters for figure building
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.boxplot(data, positions=centers, widths=widths)

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

    # plot data counts on secondary x axis above figure
    ax2 = ax1.twiny()
    ax2.set_xscale(xscale)
    ax2.set_xlim(xlimits)
    ax2.set_xticks(centers)
    ax2.set_xticklabels(counts)
    ax2.minorticks_off()
    ax2.set_xlabel('Sample Count')

    # turn on grid and tight layout
    ax1.grid(alpha=0.5)
    fig.tight_layout()

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path)

    plt.close()  #close current figure

    return fig, ax1, ax2

def modifiedBoxWhisker(df, xDataField, yDataField, xbinUpperEdges, xlabel=None, xlimits=None, xticks=None,
                       xticklabels=None, xscale='linear', ylabel=None, ylimits=None, yticks=None, yticklabels=None,
                       yscale='linear', zoomView=False, zoomViewXlim=None, zoomViewYlim=None, yScaleFactor=1,
                       quantile=None, fig=None, axes=None, figsize=None, filePath=None, fileName=None):
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
    Return fig, ax1, ax2
    """

    # select data from df and format for box whisker
    yFilt = ~(df[yDataField].isna())
    yData = df.loc[yFilt, yDataField]*yScaleFactor
    yDataMin = yData.min()

    # Initialize variables
    data = list()
    centers = list()
    widths = list()
    counts = list()
    labels = list()

    previousUE = 0
    for UE in xbinUpperEdges:
        filt = (df[xDataField] > previousUE) & (df[xDataField] <= UE) & ~(df[yDataField].isna())
        c = list(df.loc[filt, yDataField] * yScaleFactor)
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

    # Setting the parameters for figure building
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

    # Plotting boxplots
    boxdict = ax1.boxplot(data, positions=centers, widths=widths, whis=(2.5, 97.5), showmeans=True, meanline=True)

    # Inserting zoomview
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
    # Iterate through the upper xbinUpperEdges
    for i in xbinUpperEdges:
        # Check if the selected xbinUpperEdges is less than the lower bound of the xlimits
        if i <= xlimits[0]:
            # if so, characterize tick parameters for values greater than the minimum
            tickshold.append(xlimits[0])
            countshold.append("")
            centers.pop(j)
            counts.pop(j)
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

def whiskHistPlot(df,xDataField, yDataField, subDataField, subCats=None, y1label=None, x2label=None, y2label=None,
                  yunit=None, yScaleFactor=1, gridAlpha=0.3, fileName=None, filePath=None):

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
