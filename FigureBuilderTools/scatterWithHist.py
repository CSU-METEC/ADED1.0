import matplotlib.pyplot as plt
import itertools
import os

def maxList(a, b):
    count = []
    for tup in zip(a, b):
        count.append(max(tup))
    return count

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
