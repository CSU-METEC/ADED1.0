
def calcFalsePositiveFrac(classifiedDF):
    """
    :param pClassificationName: Name of column with detection classification
    :param classifiedDF: Post classification dataframe
    :return: False Positive fraction

    ColumnNames: DET Classification: Classification of report detection
    """
    try:
        filtTP = (classifiedDF['tc_Classification'] == 'TP')
        filtFP = (classifiedDF['tc_Classification'] == 'FP')
        NFP = len(classifiedDF.loc[filtFP])
        NTP = len(classifiedDF.loc[filtTP])

        FPF = NFP / (NFP + NTP)
        return FPF
    except Exception as e:
        print(f'Could not calculate false positive fraction due to exception: {e}')
        return 0