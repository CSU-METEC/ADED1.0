def calcFalseNegativeFrac(classifiedDF):
    """
    :param classifiedDF: Post classification dataframe
    :return: False Negative Fraction
    ColumnNames: tc_Classification: Classification of detection/controlled release
    """
    try:
        filtFN = (classifiedDF['tc_Classification'] == 'FN')
        NFN = len(classifiedDF.loc[filtFN])
        NCR = len(classifiedDF[(classifiedDF['tc_Classification'] == 'FN') | (classifiedDF['tc_Classification'] == 'TP')])

        FNF = NFN / NCR
        return FNF
    except Exception as e:
        print(f'Could not calculate false negative fraction due to exception: {e}')
