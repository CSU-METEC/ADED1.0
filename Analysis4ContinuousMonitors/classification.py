import pandas as pd
import logging
from Analysis4ContinuousMonitors.Logger import Logging

# Instantiate the logger with a name: __name__
thisLogger = logging.getLogger(__name__)
thisLogger.setLevel(logging.DEBUG)
# Format the instance of the log created
thisLog = Logging(thisLogger, 'classification.log')

def classify(tcControlledReleaseDF, pDetectionsDF, varsDict, buffertimes=(20, 20)):
    """
    A classification scheme as described in the protocol v1
    :param tcControlledReleaseDF: A clean dataframe of controlled releases
    :param pDetectionsDF: A clean dataframe of performer reports
    :param varsDict: A dictionary for collating important variables
    :param buffertimes: The buffertime to apply before and after a controlled release
    :return classified dataframe, pDetectionsDF, tcControlledReleaseDF, varsDict
    """

    # Classify the controlled release - detection report
    print("Pair controlled release data with detection report")
    thisLog.debug('Pairing of controlled release with detection reports..')
    classifiedDF, pDetectionsDF, tcControlledReleaseDF, varsDict = pairData(tcControlledReleaseDF, pDetectionsDF, buffertimes, varsDict)
    thisLog.info('Successfully classified controlled releases and detection reports')
    # Categorize the false positive classifications
    thisLog.debug('Categorize the false positive detections...')
    classifiedDF = categorizeFPs(classifiedDF, buffertimes)
    thisLog.info('Successfully categorized false positive detection')
    # Calculate the closest time between false positive detection and controlled release
    thisLog.debug('Calculate the closest time between controlled releases and false positive detections...')
    classifiedDF = timeBetweenCRsAndFPs(classifiedDF)
    thisLog.info('Successfully calculated the closest time between controlled releases and false positive detections')

    return classifiedDF, pDetectionsDF, tcControlledReleaseDF, varsDict


def pairData(tcControlledReleaseDF, pDetectionsDF, buffertimes, varsDict):
    """
     The code matches performer detections to test center controlled releases.
    :param tcControlledReleaseDF - test center controlled release data
    :param  pDetectionsDF - performer unique detection data
    :param buffertimes - The buffertime to apply before and after a controlled release
    :returns classifiedDF - combined data frame with all detections and controlled releases classified as TP, FP, or FN
    """
    print("Performing Classification of Detections...")
    # Create a return data frame that is a combination of both data frames
    classifiedDF = pd.DataFrame()

    print("Performing Classification of Detections...Equipment Unit Detections")
    # Sort CR data by equipment unit then by emission rate in descending order)
    tcControlledReleaseDF.sort_values(by=['tc_ExperimentID', 'tc_EquipmentUnitID', 'tc_EPBFE'], ascending=False,
                                      inplace=True)
    if len(pDetectionsDF.index) > 0:
        thisLog.debug('Performing Classification of Detections...Equipment Unit Detections')
        for i, thisCR in tcControlledReleaseDF.iterrows():
            try:
                # a.) if controlled release is already classified continue to next CR
                if thisCR['tc_CRClassification']:
                    thisLog.debug('This controlled release (CR) has been classified already, check the next CR')
                    continue

                # b.) Find all possible detection reports: a)not yet classified, b)during the experiment, c) on same EqUnit
                # Then sort by emission rate in descending order
                thisLog.debug('This CR has not been classified, find all detection reports associated with the CR')
                filt = (pDetectionsDF['tc_DetClassification'].isna()) & \
                       (pDetectionsDF['p_EquipmentUnit'] == thisCR['tc_EquipmentUnitID']) & \
                       (thisCR['tc_ExpStartDatetime'] - buffertimes[0] <= pDetectionsDF['p_EmissionStartDatetime']) & \
                       (thisCR['tc_ExpEndDatetime'] + buffertimes[1] >= pDetectionsDF['p_EmissionStartDatetime'])
                possibleDetections = pDetectionsDF.loc[filt]
                if possibleDetections.empty:  # if no possible detections at equipment level continue to next CR
                    thisLog.debug('The CR does not have any detection report associated with it')
                    continue

                # else match detection with the largest emission rate
                possibleDetections = possibleDetections.sort_values(by=['p_EmissionRate'], ascending=False, inplace=False)
                matchedDetection = possibleDetections.head(1)
                detIdx = matchedDetection.index[0]
                pDetectionsDF.loc[detIdx, "tc_DetClassification"] = 'TP'
                pDetectionsDF.loc[detIdx, "tc_mLocalizationPrecision"] = 'CorrectUnit'

                # mark the controlled release as TP
                tcControlledReleaseDF.loc[i, "tc_CRClassification"] = 'TP'
                # Get rows to pair together and append to ClassifiedDF for return
                det = pDetectionsDF.loc[detIdx]
                cr = tcControlledReleaseDF.loc[i]
                Row = pd.concat([cr, det])
                Row = pd.DataFrame(Row)
                # Transform the dataframe (rows to columns)
                Row = Row.transpose()
                # Concat the row to the dataframe
                classifiedDF = pd.concat([classifiedDF, Row])
                classifiedDF = classifiedDF.reset_index(drop=True)
                thisLog.info('Successfully paired this controlled release at the equipment unit level')
                # move to next CR in Equipment unit level for loop
            except Exception as e:
                thisLog.error('Could not pair data at the equipment unit level')
                print(f'Could not pair data at the equipment unit level for tcControlledRelease {i} due to exception: {e}')

    # ---------------------------------------------------------------------------------------------------
    # pair equipment group matches second (Protocol Classification steps 4 & 5)
    # ---------------------------------------------------------------------------------------------------
    print("Performing Classification of Detections...Equipment Group Detections")
    # Sort CR data by equipment group then by emission rate in descending order)
    tcControlledReleaseDF.sort_values(by=['tc_ExperimentID', 'tc_EquipmentGroupID', 'tc_EPBFE'], ascending=False,
                                      inplace=True)
    if len(pDetectionsDF.index) > 0:
        thisLog.debug('Performing Classification of Detections...Equipment Group Detections')
        for i, thisCR in tcControlledReleaseDF.iterrows():
            try:
                # a.) if controlled release is already classified continue to next CR
                if thisCR['tc_CRClassification']:
                    thisLog.debug('This controlled release (CR) has been classified already, check the next CR')
                    continue

                # b.) Find all possible detection reports: a)not yet classified, b)during the experiment, c) on same EqGroup
                # Then sort by emission rate in descending order
                thisLog.debug('This CR has not been classified, find all detection reports associated with the CR')
                filt = (pDetectionsDF['tc_DetClassification'].isna()) & \
                       (pDetectionsDF['p_EquipmentGroup'] == thisCR['tc_EquipmentGroupID']) & \
                       (thisCR['tc_ExpStartDatetime'] - buffertimes[0] <= pDetectionsDF['p_EmissionStartDatetime']) & \
                       (thisCR['tc_ExpEndDatetime'] + buffertimes[1] >= pDetectionsDF['p_EmissionStartDatetime'])
                possibleDetections = pDetectionsDF.loc[filt]
                if possibleDetections.empty:  # if no possible detections at equipment level continue to next CR
                    thisLog.debug('The CR does not have any detection report associated with it')
                    continue

                # else match detection with the largest emission rate
                possibleDetections = possibleDetections.sort_values(by=['p_EmissionRate'], ascending=False, inplace=False)
                matchedDetection = possibleDetections.head(1)
                detIdx = matchedDetection.index[0]
                pDetectionsDF.loc[detIdx, "tc_DetClassification"] = 'TP'
                pDetectionsDF.loc[detIdx, "tc_mLocalizationPrecision"] = 'CorrectGroup'
                # mark this CR as TP
                tcControlledReleaseDF.loc[i, "tc_CRClassification"] = 'TP'
                # Get rows to pair together and append to ClassifiedDF for return
                det = pDetectionsDF.loc[detIdx]
                cr = tcControlledReleaseDF.loc[i]
                Row = pd.concat([cr, det])
                Row = pd.DataFrame(Row)
                # Transform the dataframe (rows to columns)
                Row = Row.transpose()
                # Concat the row to the dataframe
                classifiedDF = pd.concat([classifiedDF, Row])
                classifiedDF = classifiedDF.reset_index(drop=True)
                thisLog.info('Successfully paired this controlled release at the equipment unit level')
                # move to next CR in Equipment Group level for loop
            except Exception as e:
                thisLog.error('Could not pair data at the equipment group level')
                print(f'Could not pair data at the equipment group level for tcControlledRelease {i} due to exception: {e}')

    # ---------------------------------------------------------------------------------------------------
    # pair facility matches third (Protocol Classification steps 6 & 7)
    # ---------------------------------------------------------------------------------------------------
    print("Performing Classification of Detections...Facility Detections")
    # Sort CR data by emission rate in descending order
    tcControlledReleaseDF.sort_values(by=['tc_ExperimentID', 'tc_EPBFE'], ascending=False, inplace=True)
    if len(pDetectionsDF.index) > 0:
        thisLog.debug('Performing Classification of Detections...Facility Detections')
        for i, thisCR in tcControlledReleaseDF.iterrows():
            try:
                # a.) if controlled release is already classified continue to next CR
                if thisCR['tc_CRClassification']:
                    thisLog.debug('This controlled release (CR) has been classified already, check the next CR')
                    continue

                # b.) Find all possible detection reports: a)not yet classified, b)during the experiment
                # Then sort by emission rate in descending order
                filt = (pDetectionsDF['tc_DetClassification'].isna()) & \
                       (thisCR['tc_ExpStartDatetime'] - buffertimes[0] <= pDetectionsDF['p_EmissionStartDatetime']) & \
                       (thisCR['tc_ExpEndDatetime'] + buffertimes[1] >= pDetectionsDF['p_EmissionStartDatetime'])
                possibleDetections = pDetectionsDF.loc[filt]
                if possibleDetections.empty:  # if no possible detections at facility level continue to next CR
                    thisLog.debug('The CR does not have any detection report associated with it')
                    continue

                # else match detection with the largest emission rate
                possibleDetections = possibleDetections.sort_values(by=['p_EmissionRate'], ascending=False, inplace=False)
                matchedDetection = possibleDetections.head(1)
                detIdx = matchedDetection.index[0]
                # detMask = pDetectionsDF["p_EmissionSourceID"] == matchedDetection["p_EmissionSourceID"].values[0]
                pDetectionsDF.loc[detIdx, "tc_DetClassification"] = 'TP'
                pDetectionsDF.loc[detIdx, "tc_mLocalizationPrecision"] = 'CorrectFacility'
                # mark this CR as TP
                tcControlledReleaseDF.loc[i, "tc_CRClassification"] = 'TP'
                # Get rows to pair together and append to ClassifiedDF for return
                det = pDetectionsDF.loc[detIdx]
                cr = tcControlledReleaseDF.loc[i]
                Row = pd.concat([cr, det])
                Row = pd.DataFrame(Row)
                # Transform the dataframe (rows to columns)
                Row = Row.transpose()
                # Concat the row to the dataframe
                classifiedDF = pd.concat([classifiedDF, Row])
                classifiedDF = classifiedDF.reset_index(drop=True)
                thisLog.info('Successfully paired this controlled release at the facility level')
                # move to next CR in Facility level for loop
            except Exception as e:
                thisLog.error('Could not pair data at the facility level')
                print(f'Could not pair data at the equipment facility level for tcControlledRelease {i} due to exception: {e}')

    # ---------------------------------------------------------------------------------------------------
    # Mark remaining CR as False Negative (Protocol Classification step 8)
    # ---------------------------------------------------------------------------------------------------
    print("Performing Classification of Detections...False Negative Controlled Releases")
    try:
        thisLog.debug('Performing Classification of Controlled Releases as False Negatives (FNs)')
        FNmask = tcControlledReleaseDF["tc_CRClassification"].isna()
        tcControlledReleaseDF.loc[FNmask, "tc_CRClassification"] = 'FN'
        FNs = tcControlledReleaseDF.loc[FNmask]
        classifiedDF = pd.concat([classifiedDF, FNs], ignore_index=True)
        classifiedDF = classifiedDF.reset_index(drop=True)
        thisLog.info('Successfully classified controlled releases as FNs')
    except Exception as e:
        thisLog.error('Could not classify controlled releases as FNs')
        print(f'Could not perform classification of FN controlled releases due to exception: {e}')

    # ---------------------------------------------------------------------------------------------------
    # Mark remaining detections as False Positive (Protocol Classification step 9)
    # ---------------------------------------------------------------------------------------------------
    print("Performing Classification of Detections...False Positive Detections")
    if len(pDetectionsDF.index) > 0:
        try:
            thisLog.debug('Performing Classification of Detection Reports as False Positives')
            FPmask = pDetectionsDF['tc_DetClassification'].isna()
            pDetectionsDF.loc[FPmask, 'tc_DetClassification'] = 'FP'
            FPs = pDetectionsDF[FPmask]
            classifiedDF = pd.concat([classifiedDF, FPs])
            classifiedDF = classifiedDF.reset_index(drop=True)
            thisLog.info('Successfully classified detection reports as FPs')
        except Exception as e:
            thisLog.error('Could not classify detection reports as FPs')
            print(f'Could not perform classification on FP detections due to exception: {e}')

    # ---------------------------------------------------------------------------------------------------
    # Add CR that were classified as either maintenance or offline (maintenance or OFFLINE)
    # ---------------------------------------------------------------------------------------------------
    print("Performing Classification of Detections...Other Controlled Releases")
    try:
        thisLog.debug('Add other controlled releases not already classified as TP or FN (if any)')
        otherCRMask = ~tcControlledReleaseDF["tc_CRClassification"].isin(['TP', 'FN'])
        otherCRs = tcControlledReleaseDF[otherCRMask]
        classifiedDF = pd.concat([classifiedDF, otherCRs])
        classifiedDF = classifiedDF.reset_index(drop=True)
        thisLog.info('Successfully added other controlled releases not already classified as TP or FN (if any)')
    except Exception as e:
        thisLog.error('Could not add other controlled releases not already classified as TP or FN (if any)')
        print(f'Could not perform classification on other controlled releases due to exception: {e}')

    # ---------------------------------------------------------------------------------------------------
    # Add detections that were classified as OFF_FACILITY, Maintenance, or not dur exp
    # ---------------------------------------------------------------------------------------------------
    try:
        print("Performing Classification of Detections...Other Detections")
        thisLog.debug('Add other detection reports not already classified as TP or FP (if any)')
        otherDetMask = ~pDetectionsDF['tc_DetClassification'].isin(['TP', 'FP'])
        otherDets = pDetectionsDF.loc[otherDetMask, 'tc_DetClassification']
        classifiedDF = pd.concat([classifiedDF, otherDets])
        classifiedDF = classifiedDF.reset_index(drop=True)
        thisLog.info('Successfully added other detection reports not already classified as TP or FP (if any)')
    except Exception as e:
        thisLog.error('Could not add other detection reports not already classified as TP or FP')
        print(f'Could not perform classification on other controlled releases due to exception: {e}')

    # Merge tc_DetClassification and tc_CRClassification into single tc_classification column
    thisLog.debug('Merging tc_DetClassification and tc_CRClassification into single tc_classification column')
    classifiedDF['tc_Classification'] = classifiedDF['tc_CRClassification']
    filt = (classifiedDF['tc_Classification'].isna())
    classifiedDF.loc[filt, 'tc_Classification'] = classifiedDF['tc_DetClassification'][filt]
    classifiedDF.drop(columns=['tc_CRClassification', 'tc_DetClassification'], inplace=True)
    classifiedDF = classifiedDF.reset_index(drop=True)
    thisLog.info('Successfully Merged tc_DetClassification and tc_CRClassification into single tc_classification column')

    return classifiedDF, pDetectionsDF, tcControlledReleaseDF, varsDict


def categorizeFPs(classifiedDF, buffertimes):
    """
    The code categorize false positives according to the protocol into (1) False Positives due to redundant reports,
    and (2) False Positives due to no ongoing controlled release.
    :param classifiedDF - The data file from pairing controlled release with detection reports
    :param buffertimes - The buffer times to be applied to the controlled release
    :return classifiedDF
    """

    # Create a dataframe column: "FP category"
    classifiedDF["FP category"] = None
    # Extract the true and false positive detections dataframes respectively
    filtFP = classifiedDF["tc_Classification"] == 'FP'
    filtTP = classifiedDF["tc_Classification"] == 'TP'
    TPDF = classifiedDF.loc[filtTP]
    FPDF = classifiedDF.loc[filtFP]

    # Check if the false positive detection dataframe is not empty
    if not FPDF.empty:
        thisLog.debug('Categorizing false positive detections...')
        # Make a list of all unique experiment IDs
        UniqueExperimentList = TPDF["tc_ExperimentID"].unique()
        # Iterate through the unique experiment IDs
        for id in UniqueExperimentList:
            # Get all controlled releases corresponding to experimentID
            FilteredTPDF = TPDF.loc[TPDF["tc_ExperimentID"] == id]
            # Iterate through the filtered true positive detections
            for i, row in FilteredTPDF.iterrows():
                tc_st = row['tc_ExpStartDatetime']
                tc_et = row['tc_ExpEndDatetime']
                filt1 = (FPDF["FP category"].isna()) & \
                        (tc_st - buffertimes[0] <= FPDF['p_EmissionStartDatetime']) & \
                        (tc_et + buffertimes[1] >= FPDF['p_EmissionStartDatetime'])

                # Check if there were other detections sent during a controlled release already classified as TP
                ExtraDetectionsDF = FPDF.loc[filt1]
                if not ExtraDetectionsDF.empty:
                    # Mark the false positive detections that are redundant TP detections
                    thisFilt1 = (filtFP & classifiedDF["FP category"].isna()) & \
                                (tc_st - buffertimes[0] <= classifiedDF['p_EmissionStartDatetime']) & \
                                (tc_et + buffertimes[1] >= classifiedDF['p_EmissionStartDatetime'])
                    classifiedDF.loc[thisFilt1, "FP category"] = 'Extra Reports'
                    FPDF.loc[filt1, "FP category"] = 'Extra Reports'

        # Categorize the remaining FPs as no ongoing controlled release
        remainingFPDF = FPDF.loc[FPDF["FP category"].isna()]
        if not remainingFPDF.empty:
            thisFilt2 = (filtFP & classifiedDF["FP category"].isna())
            classifiedDF.loc[thisFilt2, "FP category"] = 'No experiment running'
            FPDF.loc[FPDF["FP category"].isna(), "FP category"] = 'No experiment running'
        thisLog.info('Successfully categorized false positive detections')

    return classifiedDF

def calcTimeDelta(beforeTime, afterTime):
    """
    The code calculates the time between 2 datetimes
    :param beforeTime - first datetime
    :param afterTime - second datetime
    :return timedelta
    """
    dur = afterTime - beforeTime
    timedelta = dur.total_seconds() / 3600.0
    return timedelta

def timeBetweenCRsAndFPs(classifiedDF):
    """
    The code calculates the time between a detection report sent when there is no ongoing controlled release and the
    closest controlled release.
    :param classifiedDF - The classified data file
    :return classified dataframe
    """
    # Extract all detection reports sent when there is no ongoing controlled release
    FProws = classifiedDF.loc[(classifiedDF["tc_Classification"] == 'FP') & (classifiedDF["FP category"] == 'No experiment running')]
    # Get all controlled releases classified as either TP or FN
    CRDF = classifiedDF.loc[classifiedDF["tc_Classification"].isin(['TP', 'FN'])]
    # For each FP, find the 2 controlled releases that sandwiched the detection report
    for index, row in FProws.iterrows():
        try:
            print('Calculating the time difference between a false positive detection and the nearest CR...')
            thisLog.debug('Calculating the time difference between a false positive detection and the nearest CR...')
            # Get all controlled releases with emission end times less than the reported emission start datetime
            beforeDF = CRDF.loc[CRDF['tc_ExpStartDatetime'] < row['p_EmissionStartDatetime']]
            beforeDF = beforeDF.sort_values(by=['tc_ExpStartDatetime'], ascending=True, inplace=False)
            # Get the last controlled release with emission startdatetime < reported emission start datetime
            if not beforeDF.empty:
                SelectedBeforeDF = beforeDF.tail(1)
                idx1 = SelectedBeforeDF.index[0]
                timedelta1 = calcTimeDelta(CRDF.loc[idx1, "tc_ExpStartDatetime"], row["p_EmissionStartDatetime"])
            else:
                timedelta1 = None

            # Get all controlled releases with emission start times greater than the reported emission start datetime
            afterDF = CRDF.loc[CRDF['tc_ExpStartDatetime'] > row['p_EmissionStartDatetime']]
            afterDF = afterDF.sort_values(by=['tc_ExpStartDatetime'], ascending=True, inplace=False)
            # Get the first controlled release with emission startdatetime > reported emission start datetime
            if not afterDF.empty:
                SelectedAfterDF = afterDF.head(1)
                idx2 = SelectedAfterDF.index[0]
                timedelta2 = calcTimeDelta(row["p_EmissionStartDatetime"], CRDF.loc[idx2, "tc_ExpStartDatetime"])
            else:
                timedelta2 = None

            # Assign calculated timedelta (timedelta1 and timedelta2)
            if timedelta1 and timedelta2:
                classifiedDF.loc[index, 'timeFromClosestControlledRelease'] = min(timedelta1, timedelta2)
            elif timedelta1:
                classifiedDF.loc[index, 'timeFromClosestControlledRelease'] = timedelta1
            else:
                classifiedDF.loc[index, 'timeFromClosestControlledRelease'] = timedelta2
            thisLog.info('Successfully categorized false positive detections')

        except Exception as e:
            print(f'Could not find closes controlled release to FP detection {index} due to exception: {e}')
            thisLog.error(f'Could not find closes controlled release to FP detection {index} due to exception: {e}')

    return classifiedDF


