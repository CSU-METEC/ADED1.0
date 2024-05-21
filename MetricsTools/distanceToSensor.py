import geopy.distance as distance

def calcDistanceToSensor(classifiedDF, sensorDF):
    """
    Calculate 2D distance (m) from controlled release to nearest sensor and save to classifiedDF
    :param classifiedDF - the dataframe resulting from pairing controlled release with detection reports
    :param sensorDF - A dataframe of sensor positioning and equipment groups monitored by a solution
    :return
    """
    for cIndex, row in classifiedDF.iterrows():  # for each row of classifiedDF
        minDistance = 1000  # set minDistance arbitrarily high
        sensorName = None   # set sensor name to none
        if row['tc_Classification'] == 'TP' or row['tc_Classification'] == 'FN':  # if row is a controlled release
            epSystem = row['tc_EPID'].partition('-')[0]  # get equipment group from emission point ID
            for sIndex, s in sensorDF.iterrows():   # for each sensor
                if epSystem in s["MonitoredSystems"]:   # if the sensor monitors the equipment group
                    # calculate distance from emission point to sensor location:
                    distanceToSensor = distance.distance((row['tc_Latitude'], row['tc_Longitude']), (s['lat'], s['long'])).m
                    if distanceToSensor < minDistance:
                        minDistance = distanceToSensor
                        sensorName = s['name']

        if sensorName:
            classifiedDF.loc[cIndex, 'tc_mDistanceToClosestSensor'] = minDistance
            classifiedDF.loc[cIndex, 'tc_mClosestSensorName'] = sensorName
        else:
            classifiedDF.loc[cIndex, 'tc_mDistanceToClosestSensor'] = None
            classifiedDF.loc[cIndex, 'tc_mClosestSensorName'] = None

    return classifiedDF
