import os
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable

def ols_slope(xdata, ydata):
    """
    A code that calculates the ordinary least squares (OLS) slope and the coefficient of determination (R-squared) for
    a given set of x and y data points.
    :param xdata: list/array of x-axis data
    :param ydata: list/array of y-axis data
    :return: slope and r-square
    """
    # Creates a matrix X where each row represents an x value. This is done by stacking the x values vertically and
    # then transposing the result to obtain a column vector.
    X = np.vstack([xdata]).T
    # Uses numpy's least squares function np.linalg.lstsq() to calculate the slope of the linear regression line.
    slope = np.linalg.lstsq(X, ydata)[0]
    # Calculates the predicted values (fit) using the obtained slope and the x data points given that intercept is 0.
    fit = slope * xdata
    # Calculates the coefficient of determination (R-squared) by comparing the sum of squares of the residuals from the
    # regression line to the total sum of squares of the y data points.
    rsquared = 1 - np.sum((ydata - fit) ** 2) / np.sum((ydata - np.mean(ydata)) ** 2)
    # return slope and r-square
    return slope[0], rsquared

def categoricalScatter(df, xDataField, yDataField, catDataField, xLabel, yLabel, filePath, saveName, cats=None, s=25,
                       gridAlpha=0.5, xlim=None, ylim=None, linearFit=False):
    """scatter plot using data in df. Plots each category in catDataField as a different series.
    :param df - dataframe from which to collect data
    :param xDataField - column header in df to use for x data
    :param yDataField - column header in df to use for y data
    :param catDataField - column header in df to use for categories.
    :param xLabel - string to use as x axis label
    :param yLabel - string to use as y axis label
    :param filePath - directory to save figure to
    :param saveName - filename (without extension) to save figure
    :param cats - optional list of categories in catDataField to include.  If provided, only data series corresponding to cats will be plotted.
    :param s - marker size for scatter plot.  Default to s=25.
    :param gridAlpha - transparency of axes grid (default gridAlpha=0.5)
    :param xlim - x axis limits in form [xmin, xmax]. If None axis is scaled to all data by default.
    :param ylim - y axis limits in form [ymin, ymax]. If None axis is scaled to all data by default.
    :param linearFit - Add linear fit of each series if True
    """

    plt.rcParams.update({'font.size': 7})
    # make sure x and y data have non-zero values
    valueCheck = df[xDataField]
    if all(valueCheck == 0):
        return
    valueCheck = df[yDataField]
    if all(valueCheck == 0):
        return

    # if specific list of categories to plot isn't provided, use unique values in df[catDataField]
    if cats == None:
        cats = df[catDataField].unique()

    try:
        # make figure
        fig, ax = plt.subplots(figsize=(3.54331, 3.54331))
        markertypes = itertools.cycle(['o', 's', 'd', 'h', 'p', '^'])

        # plot each category as it's own series
        for cat in cats:
            tempDF = df.loc[(df[catDataField] == cat) & (~df[xDataField].isnull()) & (~df[yDataField].isnull())]
            x = tempDF[xDataField]
            xmax = tempDF[xDataField].max()
            y = tempDF[yDataField]
            ax.scatter(x, y, s=s, edgecolors='black', label=cat, marker=next(markertypes))
            if linearFit == True:
                # make linear fit to data series
                p = np.polyfit(x.astype("float"), y.astype("float"), 1)
                f = np.poly1d(p)
                x_new = np.linspace(0, xmax)
                y_new = f(x_new)
                ax.plot(x_new, y_new, label=r"{0:} fit = ${1:}$".format(cat, str(f).strip()), linestyle='dashed')

        # format plot
        ax.set_ylabel(yLabel)
        ax.set_xlabel(xLabel)
        plt.legend()
        fig.tight_layout()
        plt.grid(alpha=gridAlpha)

        # set axes limits
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        # save and close
        path = os.path.join(filePath, saveName)
        plt.savefig(path)
        print("saving: " + path)
        plt.close()  # close current figure
        return ax
    except Exception as e:
        print(f'Could not build scatter due to exception: {e}')
        return None


def plotByC(df, xDataField, yDataField, catDataField, yScaleFactor=1, filt=None, cbarPosition='top', cmap=None, s=25,
            norm=None, c=None, axes=None, fig=None, figsize=None, cscale='linear', climits=None, clabel=None,
            cticks=None, xScaleFactor=1):
    """
    This code create a scatter plot with color-coded data points based on a categorical variable in a Pandas DataFrame.
    It also allows for customization of various plot parameters such as color scale, color bar position, and label.
    :param df: Dataframe of data
    :param xDataField: The column header for x-axis data in the dataframe; df
    :param yDataField: The column header for y-axis data in the dataframe; df
    :param catDataField: The column header for the categorical data in the dataframe; df
    :param yScaleFactor: The factor used to scale the y-axis data
    :param filt: The filter indicating when xDataField and yDataField are not empty
    :param cbarPosition: The position to locate the color bar
    :param cmap: The colormap
    :param s: Size of the markers
    :param norm: Normalization object
    :param c: Used to specify the color of each point
    :param axes: The axes to make the plot
    :param fig: The figure to make the plot
    :param figsize: The size of the figure
    :param cscale: The scale to use for the categorical data
    :param climits: The limits of the categorical data
    :param clabel: The label of the categorical data
    :param cticks: The position of ticks on the color bar
    :return:
    """

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
    x = df.loc[filt, xDataField] * xScaleFactor
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
                               xticklabels=None, xscale="linear", cscale='linear', climits=None, cticks=None,
                               clabel=None, ylabel=None, ylimits=None, yticks=None,yticklabels=None, yscale="linear",
                               yScaleFactor=1, cats=None, linearFit=False, regression=None, oneToOne=False, s=25,
                               quantile=None, gridAlpha=0.5, fig=None, axes=None, figsize=None, filePath=None,
                               plotBYc="No", c=None, fileName=None, cmap=None, norm=None, cbarPosition = 'top',
                               paperDict=None, varPrefix=None):
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

