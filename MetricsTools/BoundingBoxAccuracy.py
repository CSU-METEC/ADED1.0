import numpy as np

def calcBoundingBoxAccuracy(row):
    """
    :param combineDF: Post classification dataframe
    :return: combineDF with Bounding Box Accuracy calculated

    ColumnNames: DET Classification: Classification of report detection
                 LATITUDE1: Latitude 1 given by performer report
                 LONGITUDE1: Longitude 1 given by performer report
                 LATITUDE: Latitude of emission point given by EP summary
                 LONGITUDE: Longitude of emission point given by EP summary
    """
    BBA = np.nan
    try:
        lat = []
        long = []
        if row['p_Latitude1'] and row['p_Latitude2'] and row['p_Longitude1'] and row['p_Longitude2']:
            tc_lat = row['tc_Latitude']
            tc_long = row['tc_Longitude']
            lat.extend([row['p_Latitude1'], row['p_Latitude2']])
            long.extend([row['p_Longitude1'], row['p_Longitude2']])
            lat.sort()
            long.sort()
            if (lat[0] <= tc_lat <= lat[1]) and (long[0] <= tc_long <= long[1]):
                BBA = True
            else:
                BBA = False
    except Exception as e:
        print(f'Could not calculate Bounding Box Accuracy due to exception: {e}')
    return BBA