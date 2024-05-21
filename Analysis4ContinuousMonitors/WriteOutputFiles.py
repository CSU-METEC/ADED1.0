import shutil
import pathlib
import os
import pandas as pd
import math


def writeOutputData(saveDir, datapath, pDetectionReportsDF, columnHeaders, tcControlledReleaseDF, classifiedDF,
                    varsDict, reportTemplate=None):

    """Write all output data to directory"""
    try:
        filepath = pathlib.PurePath(datapath, 'detectionReports.csv')
        pDetectionReportsDF.fillna(pd.NA, inplace=True)  # Remove nan and fill with -
        pDetectionReportsDF.to_csv(filepath, index=False)
    except Exception as e:
        print(f'Could not save detection report due to exception: {e}')

    try:
        filepath = pathlib.PurePath(datapath, 'testCenterControlledReleases.csv')
        tcControlledReleaseDF.fillna('-', inplace=True)  # Remove nan and fill with -
        tcControlledReleaseDF.to_csv(filepath, index=False)
    except Exception as e:
        print(f'Could not save test center due to exception: {e}')

    try:
        filepath = pathlib.PurePath(datapath, 'OutputHeaderDescriptions.csv')
        columnHeaders.to_csv(filepath, index=False)
    except Exception as e:
        print(f'Could not output header file due to exception: {e}')

    try:
        filepath = pathlib.PurePath(datapath, 'classifiedData.csv')
        classifiedDF.replace({pd.NaT: ""}, inplace=True)
        classifiedDF.fillna(pd.NA, inplace=True)  # Remove nan and fill with -
        classifiedDF.to_csv(filepath, index=False)
    except Exception as e:
        print(f'Could not save the classified data due to exception: {e}')

    # Write vars dict to report
    try:
        print("Writing the vars dictionary to papervars")
        writeFile(saveDir, 'paperVars.tex', varsDict)
    except Exception as e:
        print(f'Could not write the paperVars file due to exception: {e}')
    try:
        print("Writing the CMReports")
        copyTemplate(pathlib.PurePath(saveDir, 'CMReport.tex'), reportTemplate)
    except Exception as e:
        print(f'Could not write the CM reports file due to exception: {e}')

def getVars(vars):
    varList = ['tcmMeanEPCounts', 'tcWindSpeedAvgMin', 'tcWindSpeedAvgMax', 'tcWindSpeedAvgMean', 'tcTAtmAvgMin',
               'tcTAtmAvgMax', 'tcTAtmAvgMean']
    script = ""
    for key, value in vars.items():
        key = str(key)
        key = key.replace('_', '')
        key = key.replace('1', 'One')
        key = key.replace('2', 'Two')
        key = key.replace('3', 'Three')
        key = key.replace('4', 'Four')
        key = key.replace('5', 'Five')
        key = key.replace('6', 'Six')
        key = key.replace('7', 'Seven')
        key = key.replace('8', 'Eight')
        key = key.replace('9', 'Nine')
        key = key.replace('0', 'Zero')
        # Todo: add more characters to replace
        if type(value) == str:
            value = value.replace('%', ' percent')
            value = value.replace('>', ' greater than ')
            value = value.replace('<', ' less than ')

        if type(value) == float:
            # Round the value to 3 sig figs
            if not math.isnan(value) and value != 0.0:
                value = round(value, 3 - int(math.floor(math.log10(abs(value)))) - 1)
        if key in varList:
            value = float(value)
            value = round(value, 3 - int(math.floor(math.log10(abs(value)))) - 1)
        script += ''.join(['\n\\newcommand{\\var', str(key), '}{', str(value), '}'])
    return script

def writeFile(outputFile, fileName, varsDictionary):
    # Create variable file
    varsScript = getVars(varsDictionary)
    path = os.path.join(outputFile, fileName)
    with open(path, 'w') as f:
        f.write(varsScript)

def copyTemplate(dst, reportTemplate):
    cwd = os.getcwd()
    src = pathlib.PurePath(cwd, reportTemplate)
    shutil.copyfile(src, dst)


