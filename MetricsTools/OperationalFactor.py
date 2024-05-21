
def calcOperationalFactor(tStart, tEnd, offlineDF):
    """
    :param tStart: Start datetime of experiment
    :param tEnd: End datetime of experiment
    :param offlineDF: Performer offline reports as a dataframe
    :return: Operational factor
     ColumnNames: OFFLINEDATETIME: Offline datetime of report
                  ONLINEDATETIME: Online datetime of report
                  *I believe both of these fields are camel cased
    """
    tOffline = 0
    try:
        if offlineDF.empty:
            OF = 1
        else:
            # Find the total time of the experiment
            tTotal = (tEnd - tStart).total_seconds()
            # Find all of the offline reports that
            filt = (offlineDF['OFFLINEDATETIME'] >= tStart) & (offlineDF['ONLINEDATETIME'] <= tEnd)
            reports = offlineDF.loc[filt]
            # Find the total offline time
            for index, row in reports.iterrows():
                tOffline = tOffline + (row['ONLINEDATETIME'] - row['OFFLINEDATETIME']).total_seconds()
            # Calculate OF
            OF = 1 - (tOffline / tTotal)
        return OF
    except Exception as e:
        print(f'Could not calculate operational factor due to exception: {e}')
        return 1