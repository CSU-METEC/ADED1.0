def calcDetTime(row, pStartTimeName = 'p_FirstDatetimeSent', tcStartTimeName='tc_ExpStartDatetime'):
    """
    Calculate detection time metric
    :param row True Positive detection row
    :param tcStartTimeName Name of start date time column for performer
    :param pStartTimeName Name of start date time column for test center
    :return DT Detection time in hours
    """
    try:
        DT = row[pStartTimeName] - row[tcStartTimeName]
        # hrs = DT / 3600.0 # DT is a timedelta since it was calculated as datetime-datetime
        return DT
    except Exception as e:
        print(f'Could not calculate detection time due to exception: {e}')
        return 0
