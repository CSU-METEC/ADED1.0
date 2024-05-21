
def findMinMaxMean(classifiedDF, fieldName, metricsDict, count):
    try:
        if len(classifiedDF.loc[classifiedDF[fieldName].isnull()]) != count:
            CRDF = classifiedDF.loc[classifiedDF['tc_Classification'].isin(['TP', 'FN'])]
            metricsDict[f'{fieldName}Min'] = float(CRDF[fieldName].min())
            if float(CRDF[fieldName].max()) != 0.0:
                metricsDict[f'{fieldName}Max'] = float(CRDF[fieldName].max())
            else:  # If the value is zero, must be an integer for LaTeX formatting reasons
                metricsDict[f'{fieldName}Max'] = 0
            metricsDict[f'{fieldName}Mean'] = float(CRDF[fieldName].mean())
        else:
            metricsDict[f'{fieldName}Min'] = 0
            metricsDict[f'{fieldName}Max'] = 0
            metricsDict[f'{fieldName}Mean'] = 0
    except Exception as e:
        print(f'Could not calculate fieldName from classifiedDF due to exception: {e}')
    return metricsDict
