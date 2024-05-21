import os
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def roundup(x,n):
    return int(math.ceil(x / float(n))) * n

def rounddown(x,n):
    return int(math.floor(x / float(n))) * n

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


def barChartPolarAxis(df,thetaData,radialData,catDataField,s=25,categorical="Yes", cats=None, clabel=None, rlimits=None,
                      tlimits=None, figsize=None, n=6, thetaticks=None, angles=None, fileName='Polarplot.png',
                      filePath=None):
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
