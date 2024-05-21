import math as M
import numpy as np
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

def logistic(coeff, x):  # Logistic Function
    try:
        y = 1 / (1 + M.exp(coeff[0] + coeff[1] * x))
    except:
        y = 0
    return y


def invlogistic(p, B):  # invlogistic Function
    x = (M.log(1 / (p) - 1) - B[0]) / B[1]
    return x


def logisticRegression(Df, filePath=None, fileName=None, desiredLDLFraction=0.90, Nbootstrap=500, BinLimit=None, xScaleFactor=1,
                       BinBy=None, binUnit=None, xCategory='tc_EPBFE', xlabel='Whole gas emission rate', xunits='slpm',
                       xmax=None, xstep=0.01, figsize=None, fig=None, axes=None, digits=1, paperDict=None, fontsize=9,
                       varPrefix=None, legendLOC='lower right', xPoints=None):

    # Filtering out the desired columns from the classified dataframe and classifying detections as 1 or 0
    CRDF = Df.loc[Df['tc_Classification'].isin(['TP', 'FN'])]
    CRDF.dropna(axis=0, how='any', subset=[xCategory], inplace=True)
    logisticDf = CRDF.filter([xCategory, 'tc_Classification'], axis=1)
    logisticDf['Detection'] = logisticDf['tc_Classification'].apply(lambda x: 1 if x == 'TP' else 0)
    logisticDf[xCategory] = logisticDf[xCategory].apply(lambda y: y * xScaleFactor)
    if xmax:
        topFlowRate = xmax
    else:
        topFlowRate = logisticDf[xCategory].max()
    minFlowRate = logisticDf[xCategory].min()
    # Fitting the data with the logistic regression model
    X = np.array(logisticDf[xCategory].tolist()).reshape(-1, 1)  # X must be 2 dimensional
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
                xbot = np.array(logisticDf[xCategory].iloc[idx[:, i]].tolist()).reshape(-1, 1)  # Sampling from the given column via indexes corresponding to the elements of the column idx[:, i] and changing the dimension to match the expected form of X
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
    Xmc = list(np.arange(xstep, topFlowRate + xstep, xstep))
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
    xReg = list(np.arange(xstep, topFlowRate + xstep, xstep))
    yReg = []  # Calculating the POD using logistics function for the actual logistics model
    for x in xReg:
            yReg.append(logistic(B, x))

    # Plot Labelling
    denominator = '1 + e^{%5.3f%5.3f*x}' % (B[0], B[1])
    label = r'logistics:$\frac{1}{%s}$ (p=%.3e,%.3e)' % (denominator, p_values[0], p_values[1])
    percentage = "{:.0%}".format(desiredLDLFraction)

    # Preparing to plot
    Xmc = np.tile(np.array(Xmc).reshape(-1, 1), Nbootstrap)  # An Array that contains same columns to be used for plotting
    Xmc = pd.DataFrame(Xmc)
    Ymc = pd.DataFrame(Ymc)

    # Plotting
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

    # Plotting the bootstraps
    bsColor = '#B0C4DE'  # lightsteelblue https://www.webucator.com/article/python-color-constants-module/
    for p in list(range(Nbootstrap)):  # To plot each individual bootstrapped curve
        ax.plot(Xmc[p], Ymc[p], ls='solid', lw=0.5, c=bsColor, marker='None', zorder=1)
    # Plotting Regression
    ax.plot(xReg, yReg, ls='solid', lw=3, c='b', marker='None', label=label, zorder=2)
    # Plotting Detection threshold
    if ldl > topFlowRate or ldl < minFlowRate:
        ax.plot([ldl, ldl], [0, desiredLDLFraction], ls=':', lw=0.5, c='k', marker='None', zorder=2,
                label=('{percent} Detection Limit is out of tested range').format(percent=percentage))
    else:
        ax.plot([ldl, ldl], [0, desiredLDLFraction], ls=':', lw=0.5, c='k', marker='None', zorder=2,
                 label=('{percent} Detection Limit = {ldl:4.' + str(digits) +'f} {units}').format(percent=percentage, ldl=ldl, units=xunits))  # vertical line
    ax.plot([0, ldl], [desiredLDLFraction, desiredLDLFraction], ls=':', lw=0.5, c='k', marker='None', zorder=2)  # horizontal line
    # Plotting the detections
    logisticDf.plot.scatter(x=xCategory, y='Detection', c='k', ax=ax, label='TP & FN', zorder=3)
    # Finishing Touches
    ax.set_ylabel('Probability of Detection')
    ax.set_xlabel(xlabel+" "+xunits)
    if xmax:
        ax.set_xlim([0, xmax])
    else:
        ax.set_xlim([0, topFlowRate])
    ax.set_ylim([0, 1])
    # Adding Title To Legends
    fontP = FontProperties()
    fontP.set_size(fontsize)

    # This is when the logistic plotted for all data (i.e not binned by a secondary data)
    if BinBy==None:
        ax.legend(loc='lower right', prop=fontP)
    else: # You have to supply Binlimit to specify the bin limit of the BinBy data which you are plotting
        if BinLimit:
            # This is called when the logistic regression is plotted for binned data within the limits; BinLimit
            legendTitle = f"Plot for {BinBy}; {BinLimit[0]}{binUnit} to {BinLimit[-1]}{binUnit}"
            ax.legend(loc=legendLOC, title=legendTitle, prop=fontP)
    plt.tight_layout()
    plt.grid(axis='both', alpha=0)

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


    # Plot labeling
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


def bivariateLogistic(df, independentParams, dependentParam, indParamBins, x1, x2):

    # following example from https://towardsdatascience.com/the-binomial-regression-model-everything-you-need-to-know-5216f1a483d3

    #drop unneccessary columns
    temp = df.loc[:, np.array(independentParams)]
    temp[dependentParam] = df.loc[:, dependentParam]
    df = temp

    #cut independent params into bins
    if indParamBins:
        for key in indParamBins.keys():
            binLabels = indParamBins[key][1:]  # binLabels are upper edge of bins
            df[key+'_bin'] = pd.cut(df[key],indParamBins[key], labels=binLabels)  # cut data into bins and store in new column
            df.pop(key)    # drop unbinned data column
            independentParams.remove(key)   # remove key for unbinned independent param
            independentParams.append(key+'_bin')  # add key for binned independent param

    # Group by independent params
    groups = df.groupby(independentParams)
    groupSize = groups.count()      # number of passes in group
    groupDetections = groups.sum()  # number of detections in the group

    # transform groups back to flat table (this allows access to groupby columns again)
    groupSize.to_csv('groupSize.csv')
    groupSize = pd.read_csv('groupSize.csv', header=0)
    groupDetections.to_csv('groupDetections.csv')
    groupDetections = pd.read_csv('groupDetections.csv', header=0)

    #collect all fields into one nice flat grouped DF
    df_grouped = pd.DataFrame()
    for item in independentParams:
        df_grouped[item] = groupSize[item]
    df_grouped['Total'] = groupSize[dependentParam]
    df_grouped['TP'] = groupDetections[dependentParam]
    df_grouped['FN'] = df_grouped['Total'] - df_grouped['TP']
    df_grouped['POD'] = df_grouped['TP']/df_grouped['Total']
    df_grouped_all = df_grouped
    df_grouped = df_grouped.dropna()

    # Construct the Binomial model's regression formula in Patsy syntax.
    formula = "POD ~"
    for item in independentParams:
        formula = formula + " Q('" + item +"') +"
    formula = formula[:-2]

    # setup training data frames
    from patsy import dmatrices
    y_train, X_train = dmatrices(formula, df_grouped, return_type='dataframe')

    # feed X_train and y_train into an instance of the Binomial Regression model class and train the model
    import statsmodels.api as sm
    binom_model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
    mv_res = binom_model.fit()

    print(mv_res.summary())

    B0 = mv_res.params.values[0]
    B1 = mv_res.params.values[1]
    B2 = mv_res.params.values[2]
    p = np.zeros(x1.shape)
    for (x, y), value in np.ndenumerate(x1):
        x1i = value
        x2i = x2[x, y]
        p[x, y] = bvlogistic(B0, B1, B2, x1i, x2i)

    fig, ax = plt.subplots()
    h = plt.contourf(x2, x1, p)
    plt.axis('scaled')
    plt.colorbar()
    #plt.show()

    return fig, ax

def bvlogistic(B0, B1, B2, x1, x2):
    p = 1/(1+M.exp(-(B0+B1*x1+B2*x2)))
    return p