import datetime
import pandas as pd


def calcAvgTimeOnUnit(classifiedDF):
    df = pd.DataFrame(columns=['Facility', 'nEquip', 'Avg. Survey Time per Equip Facility (seconds)'])
    facilities = {'1/2': 6, '3': 7, '4/5': 18}
    # Go through each facilityID and do the following
    for facility, count in facilities.items():
        # Filter the facility from the classifiedDF
        filt = classifiedDF['p_FacilityID'] == facility
        filt1 = classifiedDF['p_EquipmentType'] != 'Combustor'
        filt2 = filt & filt1
        values = classifiedDF.loc[filt2]['p_SurveyTime']
        values = values.dropna()
        newCount = float(values.size)
        sTS = datetime.time()
        sTs = (sTS.hour * 60 + sTS.minute) * 60 + sTS.second
        for index, t in values.items():
            time = (t.hour * 60 + t.minute) * 60 + t.second
            sTs = (sTs + time / 60)

        # Find the average survey time from survey time series
        avgTime = sTs / newCount
        metric = avgTime / count

        # Append the df
        df = df.append(
            {'Facility': facility, 'nEquip': count, 'Avg. Survey Time per Equip Facility (seconds)': metric},
            ignore_index=True)

    return df
