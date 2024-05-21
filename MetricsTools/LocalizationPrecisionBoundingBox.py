import geopy.distance as distance
import numpy as np

def calcLocalizationPrecisionBoundingBox(row):

    LPBB = np.nan

    try:
        if row['p_Latitude1'] and row['p_Latitude2'] and row['p_Longitude1'] and row['p_Longitude2']:
            Length = distance.distance((row['p_Latitude1'], row['p_Longitude1']),
                                       (row['p_Latitude2'], row['p_Longitude1'])).m
            Height = distance.distance((row['p_Latitude1'], row['p_Longitude1']),
                                       (row['p_Latitude1'], row['p_Longitude2'])).m
            LPBB = Length * Height
    except Exception as e:
        print(f'Could not calculate localization precision bounding box due to exception: {e}')
    return LPBB
