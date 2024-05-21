import matplotlib.pyplot as plt
import os


def doubleHistogram(df, xLabel, yLabel, value, filePath, saveName, unit, bins):
    plt.rcParams.update({'font.size': 7})
    valueCheck = df[value]

    if all(valueCheck == 0) or valueCheck.isnull().all():
        return None

    try:
        fig, ax = plt.subplots(2, figsize=(3.54331, 3.54331 / 1.5))

        fig.text(0.004, 0.5, yLabel, va='center', rotation='vertical')

        if unit != 'unitless':
            ax[1].set_xlabel(f'{xLabel} ({unit})')
        else:
            ax[1].set_xlabel(f'{xLabel}')

        ax[0].hist(df[value], bins=bins, edgecolor='black')
        ax[1].hist(df[value], bins=bins[:11], edgecolor='black')

        ax[0].axvline(df[value].mean(), color='k', linestyle='dashed', linewidth=1)
        ax[1].axvline(df[value].mean(), color='k', linestyle='dashed', linewidth=1)
        fig.tight_layout()
        path = os.path.join(filePath, saveName)
        plt.savefig(path)
        print("saving: " + path)
        return ax
    except Exception as e:
        print(f'Could not build histogram due to exception: {e}')
        return None
