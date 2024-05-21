import math
import random


def bootStrapping(pod, count, nResamples, ci=0.95):
    """
    :param pod: Calculated pod between 0 and 1
    :param count: Number of points in the bin
    :param nResamples: Number of resamples or bootstraps
    :param ci: confidence interval
    :return:
    """
    i = 1
    resample = 1
    detectCounts = 0
    nonDetectCounts = 0
    PODList = []
    try:
        while resample <= nResamples:
            # For each reSample calculate a detect or non detect
            while i <= count:
                value = random.uniform(0, 1)
                if value <= pod:
                    detectCounts += 1
                else:
                    nonDetectCounts += 1
                i += 1

            # Calculate a new pod
            if detectCounts + nonDetectCounts != 0:
                POD = detectCounts / (detectCounts + nonDetectCounts)
            else:
                POD = 0

            # Append POD to PODlist
            PODList.append(POD)

            # Increase resample and reset detect and non detect counts
            resample += 1
            detectCounts = 0
            nonDetectCounts = 0
            i = 0

        # Get the max and min POD from PODList
        unOrderedPODList = PODList
        PODList.sort()
        # Get the 95% confidence interval range
        lidx = math.floor(nResamples * (1 - ci) / 2)
        uidx = math.ceil(nResamples * (1 - (1 - ci) / 2))
        podMin = PODList[lidx]
        podMax = PODList[uidx]

        # Calculate negError and posError
        negError = pod - podMin
        posError = podMax - pod

        return negError, posError, unOrderedPODList

    except Exception as e:
        print(f'Could not calculate neg and or pos error due to exception: {e}')
        return 0, 0, []
