import pandas as pd

def calcQuantMetrics(row):
    """
    :param row - 1 TP detection from classifiedDF
    :return: QAA - quantification accuracy absolute (g/hr)
    :return: QAR - quantification accuracy relative (%)
    :returns: QPA - quantification precision absolute (g/hr)
    :returns: QPR - quantification precision relative (%)
    """
    # todo: Update in future to account for detections with multiple species reported separately.
    # QAA = 0
    # QAR = 0
    # QPA = 0
    # QPR = 0
    QAA = pd.NA
    QAR = pd.NA
    QPA = pd.NA
    QPR = pd.NA

    def calcQAAandQAR(row, massFlowName):
        """
        :param row: True positive row to be calculated
        :param massFlowName: Mass flow rate column name determined by performer's gas
        :return: combineDF with absolute and relative quantitative accuracy calculated for row

        ColumnNames: EmissionRate: Emission rate determined by performer
        """
        # QAA = 0
        # QAR = 0
        QAA = pd.NA
        QAR = pd.NA

        if row['p_EmissionRate'] and row[massFlowName]:  # if both data fields are not None
            QAA = row['p_EmissionRate'] - row[massFlowName]
            QAR = QAA / row[massFlowName]

        return QAA, QAR

    def calcQPAandQPR(row, massFlowName):
        """
        :param row: 1 TP row from ClassifiedDF
        :return: combineDF with Quantitative Precision absolute and relative calculated

        ColumnNames: p_EmissionRateLower: Lower bound of emission rate determined by performer
                     p_EmissionRateUpper: Upper bound of emission rate determined by performer
        """
        #QPA = 0
        #QPR = 0
        QPA = pd.NA
        QPR = pd.NA

        if row['p_EmissionRateLower'] and row['p_EmissionRateUpper'] and row[
            massFlowName]:  # if all required data fields are not None
            QPA = row['p_EmissionRateUpper'] - row['p_EmissionRateLower']
            QPR = QPA / row[massFlowName]

        return QPA, QPR

    try:
        gas = row['p_Gas']
        if gas == 'THC':
            massFlowName = 'tc_THCMassFlow'
        elif gas == 'NMHC':
            massFlowName = 'tc_NMHCMassFlow'
        elif gas == 'METHANE':
            massFlowName = 'tc_C1MassFlow'
        elif gas == 'Methane':
            massFlowName = 'tc_C1MassFlow'
        elif gas == 'ETHANE':
            massFlowName = 'tc_C2MassFlow'
        elif gas == 'PROPANE':
            massFlowName = 'tc_C3MassFlow'
        elif gas == 'BUTANE':
            massFlowName = 'tc_C4MassFlow'
        else:
            massFlowName = None

        if massFlowName:
            QAA, QAR = calcQAAandQAR(row, massFlowName)
            QPA, QPR = calcQPAandQPR(row, massFlowName)

    except Exception as e:
        print(f'Could not calculate quantification due to exception: {e}')

    return QAA, QAR, QPA, QPR

