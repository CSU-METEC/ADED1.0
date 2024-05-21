import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

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
