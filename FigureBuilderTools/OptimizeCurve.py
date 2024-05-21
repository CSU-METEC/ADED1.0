def OptimizeCurve(VariablesDict, varPrefix, listOFnBins, bounds):
    # initialize
    r_square = 0
    nBin = None
    # iterate through the listOFnBins
    for nB in listOFnBins:
        # Check if for this NBin, the mean points per Bin fall into specified range
        meanPPB = int(VariablesDict[varPrefix + 'n'+str(nB) + 'meanPPB'])
        minPPB = int(VariablesDict[varPrefix + 'n'+str(nB) + 'minPPB'])
        maxPPB = int(VariablesDict[varPrefix + 'n'+str(nB) + 'maxPPB'])

        r_sq = float(VariablesDict[varPrefix + 'n'+str(nB) + 'rSquare'])
        # check if the meanPPB is between bounds[0] and bounds[1]
        #if (meanPPB >= bounds[0]) and (meanPPB <= bounds[1]):
        if (max([meanPPB, minPPB, maxPPB]) >= bounds[0]) and (min([meanPPB, minPPB, maxPPB]) <= bounds[1]):
            # check if the corresponding r_square is bigger than the initalized value
            if r_sq >= r_square:
                # update the initialization
                r_square = r_sq
                nBin = nB

    return nBin