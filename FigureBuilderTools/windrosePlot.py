import os
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from matplotlib.projections import register_projection

def windroseplot(df, windData, directionData, figsize=None, fileName='WindrosePlot.png', filePath=None):
    """
    What we know:
    - O is the True North of the wind direction data
    - Wind direction data are represented as degrees from North, clockwise
    - The windrose plot moves in the clockwise direction with 0 as the true North
    - Since the wind direction data are given as degrees from North in a counterclockwise direction and the windrose
    plots in the clockwise direction, this would be corrected by subtracting the wind direction angles from 360
    """
    plt.rcParams.update({'font.size': 7})
    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure(figsize=(8, 8))
    plt.grid(axis='both', alpha=0.5)

    df.sort_values(by=directionData, ascending=True, inplace=True)
    register_projection(WindroseAxes)

    ws = df[windData].tolist()
    wd = df[directionData].tolist()
    #wd = [360 - x for x in wd]

    ax = WindroseAxes.from_ax()
    ax.bar(wd, ws, normed=True, opening=1.0, edgecolor='white')

    color_s = ['red', 'blue', 'lime', 'yellow', 'violet', 'aqua', 'pink', 'grey', 'darkred', 'navy', 'green']
    ax.set_legend(title='Wind Speed in m/s', bbox_to_anchor=(0.95, 1), loc='upper left',
                  handles=color_s, borderaxespad=0.)

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    labels = ["E", "N-E", "N", "N-W", "W", "S-W", "S", "S-E"]
    ax.set_thetagrids(angles=angles, labels=labels)

    if fileName:
        if filePath:
            path = os.path.join(filePath, fileName)
        else:
            path = os.path.join(os.getcwd(), fileName)
        print("saving: " + path)
        plt.savefig(path, dpi=400)

    return fig, ax
