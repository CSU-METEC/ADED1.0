import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

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

    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data), textprops=dict(color="w"), colors=colors)

    #kw = dict(arrowprops=dict(arrowstyle="->"), va="center")
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
