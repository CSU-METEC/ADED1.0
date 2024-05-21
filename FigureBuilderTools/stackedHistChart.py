import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd
from MetricsTools.Binner import bin2


def stackedHistCSB(DF, xCategory, yCategory, filePath=None, fileName='hist.png', xunit=None, xBins=None, xticks=None,
                   xLabel=None, xScaleFactor=1, yCats=None, yLabel='Count', nbins=10, density=False, gridAlpha=0.3,
                   xTickRotation=0, xlim=None, fig=None, axes=None, figSize=(3.54331, 3.54331 / 1.5)):
    """Makes a stacked histogram of df[xCategory] where each category in df[yCategory].unique() is a layer in the stack.
    :param df = dataframe from which to pull data
    :param xCategory = column name in df to use for xData (should be numeric)
    :param yCategory = column name in df to use for stack categories (should be categorical)
    :param filePath = path to folder for saving figure
    :param fileName = file name to save figure (without extension)
    :param xLabel = text string for x axis label
    :param yLabel = text string for y axis label (default yLabel='Count')
    :param nbins = number of bins for x data (default nbins=10)
    :param density = plot hist using density (default density=False)
    :param gridAlpha = transparency for axes grid (default gridAlpha = 0.5)
    """

    plt.rcParams.update({'font.size': 7})
    if fig is None:
        fig, ax = plt.subplots(figsize=figSize)
    elif axes:
        ax = axes
    else:
        ax = plt.gca()

    df = DF.dropna(axis=0, how='any', subset=[xCategory], inplace=False).copy(deep=True)
    df[xCategory] = df[xCategory].apply(lambda x: x * xScaleFactor)

    # collect data
    df = df.sort_values(by=[xCategory, yCategory])
    if yCats is None:
        yCats = df[yCategory].unique()
    xData = list()
    for yCat in yCats:
        filt = df[yCategory] == yCat
        xData.append(df.loc[filt, xCategory].values)

    # plot data
    if type(xData[0][0]) == str:
        # special formatting required if x is categorical (ignore nbins arguement and shift labels)
        nc = len(set(df[xCategory]))
        ax.hist(xData, bins=nc, rwidth=0.7, histtype='bar', stacked=True, label=yCats, edgecolor='black',
                density=density)
        plt.xticks(np.linspace(0, nc - 1, 2 * nc + 1)[1::2])
        bins = nc
    else:
        # just use hist method
        if xBins:
            ax.hist(xData, bins=xBins, histtype='bar', stacked=True, label=yCats, edgecolor='black', density=density)
            if xticks is None:
                xticks = xBins
            bins = xBins
        elif xBins == None:
            ax.hist(xData, bins=nbins, histtype='bar', stacked=True, label=yCats, edgecolor='black', density=density)
            bins = nbins

    # format plot
    plt.xticks(rotation=xTickRotation)
    if xticks:
        ax.set_xticks(xticks)
    ax.set_ylabel(yLabel)
    if xunit:
        ax.set_xlabel(xLabel+" ("+xunit+")")
    else:
        ax.set_xlabel(xLabel)
    yLimit = ax.get_ylim()
    if xlim:
        ax.set_xlim(xlim)
    ax.legend(yCats)
    fig.tight_layout()
    ax.grid(alpha=gridAlpha)

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)
    #plt.close()  # close current figure

    return {'xData': xData, 'bins': bins, 'Categories': yCats, 'yLimit': yLimit}, ax, fig


def stackedHist(classifiedDF, xCategory, yCategory, filePath, fileName='hist.png',
                xLabel=None, units='', yLabel='Count', xWhiteList=None, xBlackList=None,
                yWhiteList=None, yBlackList=None, normalized=False, xSorting=None, ySorting=None,
                xByYCategory=None, xOrder=None, yOrder=None, xBinCount=None, yBinCount=None):
    # todo: Add a way to sort x bins by data in y (like count of y in decending order)
    fig, ax = plt.subplots()

    dataMatrix, bins, labels, xType, yType = countBy(classifiedDF, xCategory, yCategory, xWhiteList, xBlackList,
                                                     yWhiteList,
                                                     yBlackList, normalized, xSorting, ySorting, xByYCategory, xOrder,
                                                     yOrder, xBinCount, yBinCount)
    if xType == 'numeric':
        for interval in bins:
            i = bins.index(interval)
            Min = interval.left.item()
            Max = interval.right.item()
            if not math.isnan(Min) and Min != 0.0:
                Min = round(Min, 3 - int(math.floor(math.log10(abs(Min)))) - 1)
            if not math.isnan(Max) and Max != 0.0:
                Max = round(Max, 3 - int(math.floor(math.log10(abs(Max)))) - 1)
            bins[i] = str(Min) + ', ' + str(Max) + ' ' + units
    elif xType == 'categorical':
        for Bin in bins:
            i = bins.index(Bin)
            bins[i] = str(Bin)
    if yType == 'numeric':
        for interval in labels:
            i = labels.index(interval)
            Min = interval.left.item()
            Max = interval.right.item()
            if not math.isnan(Min) and Min != 0.0:
                Min = round(Min, 3 - int(math.floor(math.log10(abs(Min)))) - 1)
            if not math.isnan(Max) and Max != 0.0:
                Max = round(Max, 3 - int(math.floor(math.log10(abs(Max)))) - 1)
            labels[i] = str(Min) + ', ' + str(Max) + ' ' + units
    # dataMatrix = np.transpose(dataMatrix)
    dataMatrix = pd.DataFrame(dataMatrix)
    # previousColumn = None
    # for column in dataMatrix:
    #     ax.bar(bins, column, edgecolor='black', bottom=previousColumn)
    #     previousColumn = list(column)
    dataMatrix.plot(kind='bar', stacked=True, edgecolor='black', rot=0)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend(labels)
    plt.grid(axis='y', alpha=0.5)
    plt.xticks(ticks=list(range(0, len(bins))), labels=bins)

    path = os.path.join(filePath, fileName)
    plt.savefig(path)
    print("saving: " + path)

    return ax


def countBy(df, xCategory, yCategory, xWhiteList, xBlackList, yWhiteList, yBlackList, normalized, xSorting,
            ySorting, xByYCategory, xOrder, yOrder, xBinCount, yBinCount):
    # Look at x and y Categories in df. Bin values depending on if values in each category are numeric or categorical
    xBins, xType = binner(df, xCategory, xWhiteList, xBlackList, sorting=xSorting, count=xBinCount,
                          xByYCategory=xByYCategory, category2=yCategory, order=xOrder)
    yBins, yType = binner(df, yCategory, yWhiteList, yBlackList, sorting=ySorting, count=yBinCount, order=yOrder)
    dataMatrix = np.zeros((len(xBins), len(yBins)))
    if xType == 'categorical':
        for xBin in xBins:
            filt = df[xCategory] == xBin
            xIdx = xBins.index(xBin)
            for yBin in yBins:
                if yType == 'categorical':
                    filt1 = df[yCategory] == yBin
                else:
                    filt1 = (yBin.left.item() <= df[yCategory]) & (yBin.right.item() >= df[yCategory])
                yIdx = yBins.index(yBin)
                count = len(df.loc[filt & filt1])
                dataMatrix[xIdx, yIdx] = count
    elif xType == 'numeric':
        for interval in xBins:
            Min = interval.left.item()
            Max = interval.right.item()
            filt = (df[xCategory] > Min) & (df[xCategory] <= Max)
            xIdx = xBins.index(interval)
            for yBin in yBins:
                filt1 = df[yCategory] == yBin
                yIdx = yBins.index(yBin)
                count = len(df.loc[filt & filt1])
                dataMatrix[xIdx, yIdx] = count
    i = 0
    removeRows = []
    removeItems = []
    for row in dataMatrix:
        if all(row == 0.0):
            removeRows.append(i)
            removeItems.append(xBins[i])
        i += 1
    xBins = [item for item in xBins if item not in removeItems]
    dataMatrix = np.delete(dataMatrix, removeRows, 0)

    if normalized:
        row_sums = dataMatrix.sum(axis=1)
        dataMatrix = dataMatrix / row_sums[:, np.newaxis]
        # row_sums = dataMatrix.sum()
        # dataMatrix = dataMatrix / row_sums

    return dataMatrix, xBins, yBins, xType, yType


def binner(df, category, whiteList, blackList, sorting, count=5, xByYCategory=None, category2=None, order=None):
    bins = []
    Type = None
    df.dropna(subset=[category], inplace=True)
    catCount = 0
    numericCount = 0
    for index, row in df.iterrows():
        if type(row[category]) == int or type(row[category]) == float:
            numericCount += 1
        elif type(row[category]) == str:
            catCount += 1

    if catCount > 0 and numericCount == 0:
        # Find all unique values in column
        values = df[category].unique()
        values = [i for i in values if i]
        bins = list(values)
        bins = binSorter(bins, df, category, sorting, xByYCategory=xByYCategory, category2=category2, order=order)
        Type = 'categorical'
        # bins = [x for x in bins if x != np.nan]
    elif numericCount > 0 and catCount == 0:
        bins = list(bin2(df, category, count))
        Type = 'numeric'
    else:
        # Raise an error
        pass
    if whiteList:
        bins = whiteListFilt(whiteList)
    if blackList:
        bins = blackListFilt(bins, blackList)

    return bins, Type


def whiteListFilt(whiteList):
    return whiteList


def blackListFilt(bins, blackList):
    return bins.remove(blackList)


def binSorter(bins, df, category, sorting, category2=None, xByYCategory=None, order=None):
    binCounts = {}
    newBins = []
    if order:
        # Check to see if contains all bins
        for Bin in order:
            if Bin not in bins:
                order.remove(Bin)
        newBins = order
    else:
        for Bin in bins:
            if xByYCategory == 'Total':
                filt = (df[category] == Bin)
            elif category2 and xByYCategory:
                filt = (df[category] == Bin) & (df[category2] == xByYCategory)
            else:
                filt = (df[category] == Bin)

            count = len(df[category][filt])
            binCounts[Bin] = count
        if sorting == 'descending':
            binCounts = dict(sorted(binCounts.items(), key=lambda item: item[1], reverse=True))
        if sorting == 'ascending':
            binCounts = dict(sorted(binCounts.items(), key=lambda item: item[1], reverse=False))
        if sorting == 'byYDescending':
            binCounts = dict(sorted(binCounts.items(), key=lambda item: item[1], reverse=True))
        if xByYCategory == 'Total':
            binCounts = dict(sorted(binCounts.items(), key=lambda item: item[1], reverse=True))
        if not sorting and not xByYCategory:
            binCounts = dict(sorted(binCounts.items(), key=lambda item: item[1]))

        for Bin, count in binCounts.items():
            newBins.append(Bin)

    return newBins
