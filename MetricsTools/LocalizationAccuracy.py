def calcLocalizationAccuracy(classifiedDF, TPLevelNames):
    """
    :param TPLevelNames: List of classification level names
    :param classifiedDF: Post classification dataframe
    :return: Localization accuracy at the equipment, group, and facility levels
    ColumnNames: tc_Classification: Classification of report detection
                 tc_mLocalizationPrecision: The localization level at which the emission was detected
    """
    NTP = len(classifiedDF.loc[classifiedDF['tc_Classification'] == 'TP'])
    NFP = len(classifiedDF.loc[classifiedDF['tc_Classification'] == 'FP'])
    NRD = NTP + NFP
    try:
        # Find number of TP at each detection level
        filt = (classifiedDF['tc_mLocalizationPrecision'] == TPLevelNames[0])
        filt1 = (classifiedDF['tc_mLocalizationPrecision'] == TPLevelNames[1])
        filt2 = (classifiedDF['tc_mLocalizationPrecision'] == TPLevelNames[2])
        NTPUnit = len(classifiedDF.loc[filt])
        NTPGroup = len(classifiedDF.loc[filt1])
        NTPFacility = len(classifiedDF.loc[filt2])

        # Calculate LA for each detection level
        LAUnit = NTPUnit / NRD
        LAGroup = (NTPUnit + NTPGroup) / NRD
        LAFacility = (NTPUnit + NTPGroup + NTPFacility) / NRD
        return LAUnit, LAGroup, LAFacility
    except Exception as e:
        print(f'Could not calculate localization accuracy due to exception: {e}')
        return 0, 0, 0
