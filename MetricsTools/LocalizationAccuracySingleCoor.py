import geopy.distance as distance
import numpy as np

def calcLocalizationAccuracySinCoor(row):
    """
    :param row: Post classification dataframe
    :return: combineDF with Localization Accuracy with a single coordinate calculated

    ColumnNames: DET Classification: Classification of report detection
                 LATITUDE1: Latitude 1 given by performer report
                 LONGITUDE1: Longitude 1 given by performer report
                 LATITUDE: Latitude of emission point given by EP summary
                 LONGITUDE: Longitude of emission point given by EP summary
    """
    LASC = np.nan

    try:
        if row['p_Latitude1'] and row['p_Latitude1']:
            lat = row['p_Latitude1']
            long = row['p_Longitude1']

            LASC = distance.distance((lat, long), (row['tc_Latitude'], row['tc_Longitude'])).m

    except Exception as e:
        print(f'Could not calculate Localization Accuracy (Single Coordinate) due to exception: {e}')
    return LASC
