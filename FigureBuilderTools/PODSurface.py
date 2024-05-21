import os
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from MetricsTools.PODSurfaceWithEvenCounts import calcPODSurfaceWEvenCounts


def buildPODSurface(podSurfaceDF, xLabel, yLabel,outputFilePath):
    try:
        x = np.linspace(podSurfaceDF['xbinAvg'].min(), podSurfaceDF['xbinAvg'].max(), 100)
        y = np.linspace(podSurfaceDF['ybinAvg'].min(), podSurfaceDF['ybinAvg'].max(), 100)

        X, Y = np.meshgrid(x, y)
        # todo: interpolate podSurfaceDF['xy_pod'], modify podSurfaceDF = podSurfaceDF where "xy_pod" is not none
        #  (remove them)
        podSurfaceDF.dropna(subset=['xy_pod'])
        Z = griddata((podSurfaceDF['xbinAvg'], podSurfaceDF['ybinAvg']), podSurfaceDF['xy_pod'], (X, Y),
                     method='cubic')
        # For values greater than 1, set values equal to 1
        Z[Z >= 1] = 1
        fig, ax = plt.subplots()
        # ax = fig.gca(projection='3d')
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
        #                        linewidth=0, antialiased=False)
        # left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        # ax = fig.add_axes([left, bottom, width, height])
        surf = ax.contourf(X, Y, Z, np.arange(0, 1.1, 0.1), cmap=cm.jet, vmax=1)

        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        # ax.view_init(azim=0, elev=90)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        xLabel = xLabel.replace(' ', '_')
        yLabel = yLabel.replace(' ', '_')

        name = "podSurface.png"

        path = os.path.join(outputFilePath, name)
        plt.savefig(path)
        return ax
    except Exception as e:
        print(f'Could not generate POD Surface due to exception: {e}')
        return None


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

def surfaceplots(df, xDataField, yDataField, nxBins=3, nyBins=3,xLabel=None, yLabel=None, fileName='surfacePlot.png',
                 filePath=None):

    _, podDF = calcPODSurfaceWEvenCounts(df, xCategory=xDataField, yCategory=yDataField, nxBins=nxBins, nyBins=nyBins)
    ax = modifiedbuildPODSurface(podSurfaceDF=podDF, xLabel=xLabel, yLabel=yLabel, fileName=fileName, filePath=filePath)

    return ax
