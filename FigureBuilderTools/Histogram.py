import matplotlib.pyplot as plt
import os


# todo: Add gridlines

def histogram(df, xLabel, yLabel, xDataField, filePath, saveName, unit):

    valueCheck = df[xDataField]
    if all(valueCheck == 0) or valueCheck.isnull().all():
        return None

    try:
        plt.rcParams.update({'font.size': 7})
        fig, ax = plt.subplots(figsize=(3.54331, 3.54331 / 1.5))

        ax.set_ylabel(yLabel)

        if unit != 'unitless':
            ax.set_xlabel(f'{xLabel} ({unit})')
        else:
            ax.set_xlabel(f'{xLabel}')

        filt = ~df[xDataField].isna()
        ax.hist(df.loc[filt, xDataField], edgecolor='black')
        ax.axvline(df.loc[filt, xDataField].mean(), color='k', linestyle='dashed', linewidth=1)
        fig.tight_layout()
        plt.grid(alpha=0.3)
        path = os.path.join(filePath, saveName)
        plt.savefig(path, dpi=400)
        print("saving: " + path)
        return ax
    except Exception as e:
        print(f'Could not build histogram due to exception: {e}')
        return None

def newHistogram(DF, xLabel, yLabel, xDataField, filePath, saveName, unit, bins, zoomView=False, zvUpperlimit=None):
    df = DF.dropna(axis=0, how='any', subset=[xDataField], inplace=False).copy(deep=True)
    plt.rcParams.update({'font.size': 7})
    valueCheck = df[xDataField]

    if all(valueCheck == 0) or valueCheck.isnull().all():
        return
    fig, ax = plt.subplots(figsize=[5, 4])

    ax.set_ylabel(yLabel)

    if unit != 'unitless':
        ax.set_xlabel(f'{xLabel} ({unit})')
    else:
        ax.set_xlabel(f'{xLabel}')

    ax.hist(df[xDataField], bins=bins, edgecolor='black')
    ax.axvline(df[xDataField].mean(), color='k', linestyle='dashed', linewidth=1)
    ax.grid(alpha=0.3)
    if zoomView==True:
        if len(df.loc[df[xDataField] < zvUpperlimit]) > 0:
            axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
            sBins = list(x for x in range(0, zvUpperlimit))
            axins.hist(df[xDataField], bins=sBins, edgecolor='black')
            axins.set_xlim(0, 21)
            axins.grid(alpha=0.3)
            ax.indicate_inset_zoom(axins, edgecolor='r')
    fig.tight_layout()
    path = os.path.join(filePath, saveName)
    plt.savefig(path, dpi=400)
    print("saving: " + path)
    return ax

def simpleHistogram(DF, xData=None, alpha=0.3, fileName=None, filePath=None, figSize=None, axes=None, fig=None,
                    xLabel=None, yLabel=None):
    # Plotting parameter definition
    df = DF.dropna(axis=0, how='any', subset=[xData], inplace=False).copy(deep=True)
    plt.rcParams.update({'font.size': 7})

    if fig is None:
        fig, ax = plt.subplots(figsize=figSize)
    elif axes:
        ax = axes
        fig = fig
    else:
        ax = plt.gca()

    ax.hist(df[xData].tolist(), edgecolor='black')
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

