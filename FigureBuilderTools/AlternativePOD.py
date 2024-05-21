import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
import statistics
from matplotlib.font_manager import FontProperties
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


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
