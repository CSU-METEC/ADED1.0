import geopy.distance as distance
import numpy as np


def calcLocalizationAccuracyBox(row):
    """
    :param row: Post classification dataframe row
    :return: combineDF with Localization Accuracy Box calculated

    ColumnNames: DET Classification: Classification of report detection
                 LATITUDE1: Latitude 1 given by performer report
                 LONGITUDE1: Longitude 1 given by performer report
                 LATITUDE: Latitude of emission point given by EP summary
                 LONGITUDE: Longitude of emission point given by EP summary
    """
    LABB = np.nan

    def mean(val1, val2):
        try:
            mean = (val1 + val2) / 2
            return mean
        except Exception as e:
            print(f'Could not calculate mean for values {val1}, {val2} due to exception: {e}')

    try:
        if row['p_Latitude1'] and row['p_Latitude2'] and row['p_Longitude1'] and row['p_Longitude2']:
            lat = mean(row['p_Latitude1'], row['p_Latitude2'])
            long = mean(row['p_Longitude1'], row['p_Longitude2'])
            LABB = distance.distance((lat, long), (row['tc_Latitude'], row['tc_Longitude'])).m

    except Exception as e:
        print(f'Could not calculate Localization Accuracy (Bounding Box) due to exception: {e}')
    return LABB
