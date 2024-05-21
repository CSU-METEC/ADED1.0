import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def plotMeasurementDist(df, figsize=(3.5,3.5), filePath=None, fileName=None, cumfileName=None, gridAlpha=0.5):

    # select data into 4 different vectors, and sort each in ascending order (smallest to largest)
    cr = df.loc[df['WindType'] == 'HRRR', 'cr_kgh_CH4_mean'].sort_values().dropna()
    hrrr = df.loc[df['WindType'] == 'HRRR', 'b_kgh'].sort_values().dropna()
    nam12 = df.loc[df['WindType'] == 'NAM12', 'b_kgh'].sort_values().dropna()
    sonic = df.loc[df['WindType'] == 'Sonic', 'b_kgh'].sort_values().dropna()

    # make empirical cumulative dist (ecdf)
    cr_fracs = np.linspace(0,1,cr.size+1)
    cr_fracs = cr_fracs[1:cr_fracs.size]
    b_fracs = np.linspace(0,1,hrrr.size+1)
    b_fracs = b_fracs[1:b_fracs.size]

    # plot (x = emission rate (kg/h), y = edf (fraction of samples))
    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cr, cr_fracs, label='Controlled Releases')
    ax.plot(hrrr, b_fracs, label='HRRR')
    ax.plot(nam12, b_fracs, label='NAM12')
    ax.plot(sonic, b_fracs, label='Sonic')
    ax.legend()
    ax.set_xlabel('Emission Rate (kg/h)')
    ax.set_ylabel('Fraction of sources')
    ax.set_ylim((0, 1))
    ax.set_xscale('linear')
    ax.grid(alpha=gridAlpha)
    fig.tight_layout()

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path)

    plt.close()  # close current figure

    # plot cumulative sum
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cr.cumsum(), cr_fracs, label='Controlled Releases')
    ax.plot(hrrr.cumsum(), b_fracs, label='HRRR')
    ax.plot(nam12.cumsum(), b_fracs, label='NAM12')
    ax.plot(sonic.cumsum(), b_fracs, label='Sonic')
    ax.legend()
    ax.set_xlabel('Cumulative Emission Rate (kg/h)')
    ax.set_ylabel('Fraction of sources')
    ax.set_ylim((0, 1))
    ax.set_xscale('linear')
    fig.tight_layout()

    # save
    if cumfileName:
        if filePath:
            path = os.path.join(filePath, cumfileName)
        else:
            path = os.path.join(os.getcwd(), cumfileName)
        print("saving: " + path)
        plt.savefig(path)

    plt.close()  # close current figure

    return

def plotErrorDist(df, figsize=(5,3.5), filePath=None, fileName=None, gridAlpha=0.5, paperDict=None):

    # select data into 4 different vectors, and sort each in ascending order (smallest to largest)
    hrrr = df.loc[df['WindType'] == 'HRRR', 'FlowError_percent'].sort_values().dropna()
    nam12 = df.loc[df['WindType'] == 'NAM12', 'FlowError_percent'].sort_values().dropna()
    sonic = df.loc[df['WindType'] == 'Sonic', 'FlowError_percent'].sort_values().dropna()

    # make empirical cumulative dist (ecdf)
    b_fracs = np.linspace(0,1,hrrr.size+1)
    b_fracs = b_fracs[1:b_fracs.size]

    # plot (x = emission rate (kg/h), y = edf (fraction of samples))
    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(hrrr, b_fracs, label='HRRR')
    ax.plot(nam12, b_fracs, label='NAM12')
    ax.plot(sonic, b_fracs, label='Sonic')
    ax.legend()
    ax.set_xlabel('% Error')
    ax.set_ylabel('Fraction of sources')
    ax.set_ylim((0, 1))
    ax.set_xlim((-100, 250))
    ax.set_xscale('linear')
    ax.grid(alpha=gridAlpha)
    fig.tight_layout()

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path)

    plt.close()  # close current figure

    # calculate values for paperDict
    if paperDict:
        #fraction of data points within factor of two error [-50%, +100%] error
        paperDict['HRRR_facTwoError_percent'] = round(hrrr[(hrrr <= 100) & (hrrr >= -50)].size/hrrr.size * 100, 1)
        paperDict['NAM_facTwoError_percent'] = round(nam12[(nam12 <= 100) & (nam12 >= -50)].size / nam12.size * 100, 1)
        paperDict['Sonic_facTwoError_percent'] = round(sonic[(sonic <= 100) & (sonic >= -50)].size / sonic.size * 100, 1)

        # fraction of data points within [-20%, +20%] error
        paperDict['HRRR_twentyPerError_percent'] = round(hrrr[(hrrr <= 20) & (hrrr >= -20)].size / hrrr.size * 100, 1)
        paperDict['NAM_twentyPerError_percent'] = round(nam12[(nam12 <= 20) & (nam12 >= -20)].size / nam12.size * 100, 1)
        paperDict['Sonic_twentyPerError_percent'] = round(sonic[(sonic <= 20) & (sonic >= -20)].size / sonic.size * 100, 1)

        lcl_idx = round(hrrr.size*0.025, 0)-1
        ucl_idx = round(hrrr.size * 0.975, 0)-1

        hrrr.reset_index(drop=True, inplace=True)
        nam12.reset_index(drop=True, inplace=True)
        sonic.reset_index(drop=True, inplace=True)

        paperDict['HRRR_ErrorCI_LCL_percent'] = round(hrrr[lcl_idx], 1)
        paperDict['HRRR_ErrorCI_UCL_percent'] = round(hrrr[ucl_idx], 1)
        paperDict['NAM_ErrorCI_LCL_percent'] = round(nam12[lcl_idx], 1)
        paperDict['NAM_ErrorCI_UCL_percent'] = round(nam12[ucl_idx], 1)
        paperDict['Sonic_ErrorCI_LCL_percent'] = round(sonic[lcl_idx], 1)
        paperDict['Sonic_ErrorCI_UCL_percent'] = round(sonic[ucl_idx], 1)

    return paperDict


def plotGenie(df, figsize=(3.5,3.5), filePath=None, fileName=None, gridAlpha=0.5, paperDict=None):

    # select data into 4 different vectors, and sort by controlled release in ascending order (smallest to largest)
    pData = pd.DataFrame()
    pData['cr'] = df.loc[df['WindType'] == 'HRRR', 'cr_kgh_CH4_mean']
    pData['meas'] = df.loc[df['WindType'] == 'HRRR', 'b_kgh']
    pData.sort_values('cr', inplace=True)
    pData.dropna(inplace=True)
    hrrrSums = pData.cumsum()
    hrrrSums['percentError'] = (hrrrSums['meas']-hrrrSums['cr'])/hrrrSums['cr'] * 100
    counts = np.linspace(1, hrrrSums['meas'].size, hrrrSums['meas'].size)

    pData = pd.DataFrame()
    pData['cr'] = df.loc[df['WindType'] == 'NAM12', 'cr_kgh_CH4_mean']
    pData['meas'] = df.loc[df['WindType'] == 'NAM12', 'b_kgh']
    pData.sort_values('cr', inplace=True)
    pData.dropna(inplace=True)
    namSums = pData.cumsum()
    namSums['percentError'] = (namSums['meas'] - namSums['cr']) / namSums['cr'] * 100

    pData = pd.DataFrame()
    pData['cr'] = df.loc[df['WindType'] == 'Sonic', 'cr_kgh_CH4_mean']
    pData['meas'] = df.loc[df['WindType'] == 'Sonic', 'b_kgh']
    pData.sort_values('cr', inplace=True)
    pData.dropna(inplace=True)
    sonicSums = pData.cumsum()
    sonicSums['percentError'] = (sonicSums['meas'] - sonicSums['cr']) / sonicSums['cr'] * 100

    # plot (x = emission rate (kg/h), y = ecdf (fraction of samples))
    plt.rcParams.update({'font.size': 7})
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=figsize)

    # release rate vs cumulative release rate
    ax1.scatter(sonicSums['cr'], pData['cr'], s=5, c='black', label='rate')
    ax1.plot([], [], '-b', label='count') # make fake line to label count in axis legend
    ax1b = ax1.twinx()
    ax1b.plot(sonicSums['cr'], counts,'-b', label='count')
    ax1.set_xlabel('Cumulative release rate (kg/h)')
    ax1.set_ylabel('Controlled release rate (kg/h)')
    ax1b.set_ylabel('Cumulative sample count')
    ax1.grid(alpha=gridAlpha)
    ax1.legend()


    # cumulative estimate vs cumulative release rate
    ax2.plot(hrrrSums['cr'], hrrrSums['cr'], label='1:1', c='black')
    ax2.plot(hrrrSums['cr'], hrrrSums['meas'], label='HRRR')
    ax2.plot(namSums['cr'], namSums['meas'], label='NAM12')
    ax2.plot(sonicSums['cr'], sonicSums['meas'], label='Sonic')
    ax2.set_xlabel('Cumulative release rate (kg/h)')
    ax2.set_ylabel('Cumulative estimate (kg/h)')
    ax2.grid(alpha=gridAlpha)
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.legend()

    # percent error in cumulative estimate vs cumulative release rate
    ax3.plot(hrrrSums['cr'], hrrrSums['percentError'], label='HRRR')
    ax3.plot(namSums['cr'], namSums['percentError'], label='NAM12')
    ax3.plot(sonicSums['cr'], sonicSums['percentError'], label='Sonic')
    ax3.set_xlabel('Cumulative release rate (kg/h)')
    ax3.set_ylabel('Error in cumulative estimate (%)')
    ax3.grid(alpha=gridAlpha)
    ax3.set_xscale('linear')
    ax3.set_yscale('linear')
    #ax3.legend()
    ax3.set_ylim((-40, 40))

    fig.tight_layout()

    # save
    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path)

    plt.close()  # close current figure

    return paperDict
