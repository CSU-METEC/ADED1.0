This document describes how to run this code. Note that even though the code can be used, it is currently being updated and optimized for better performance.

# CODE SETUP
For Pycharm users: Edit configuration by setting Script path to ...\ADEDPostProcessingGitHub\Analysis4ContinuousMonitors\Main.py and set the working directory to ...\ADEDPostProcessingGitHub\Analysis4ContinuousMonitors
The code uses Anaconda3 - Python 3.8 as Python Interpreter
Parameters include:
    -st (The start time of the analysis)
    "YYYYMMDDHHmm"
    -et (The end time of the analysis)
    "YYYYMMDDHHmm"
    -cr (The path to the cleaned controlled release data. Ensure that the file follows the format of this document: "...\InputFolder\controlledReleaseDF_format.csv")
    "...\InputFolder\controlledReleaseDF.csv"
    -dr (The path to the cleaned detection report data. Ensure that the file follows the format of this document: "...\InputFolder\detectionReports_format.csv")
    "...\InputFolder\detectionReports.csv"
    -dh (The path to the data dictionary of the classifiedData.csv file. Ensure that the code points to the path: "...\InputFolder\OutputHeaderDescriptions.csv")
    "...\InputFolder\OutputHeaderDescriptions.csv"
    -s (The path to the sensorDataDF. Ensure that the file follows the format of this document: "...\InputFolder\SensorDataDF_format.csv")
    "...\InputFolder\SensorDataDF.csv"
    -o (The path to the output folder)
    "....\OutputFolder"
    -yQ
    0.95

INPUT: The code needs 4 files to run: A controlled release file, a Detection report file, an Offline report file, and a Sensor file.
1. A controlled release file ("testCenterControlledReleases.csv") supplied by METEC at the end of an ADED CM testing. The data should only include controlled releases classified as true positives (TP) and false negatives (FN).
2. All valid detection report data supplied by the performer at the end of testing. The data should only include valid detection reports to be classified as either true positives (TP) or false positives (FP).
3. Offline report data supplied by the performer indicating offline periods.
4. Sensor data supplied by either the test center or the performer shows each sensor installed and the equipment units and groups monitored by the sensor

OUTPUT: The codes output 5 files: 'detectionReports.csv', 'testCenterControlledReleases.csv', 'OutputHeaderDescriptions.csv' 'classifiedData.csv', and 'paperVars.tex'
1. 'detectionReports.csv' - A CSV file of the detection file supplied to the code
2. 'testCenterControlledReleases.csv' - A CSV file of the controlled release file supplied
3. 'OutputHeaderDescriptions.csv' - A CSV file of the data dictionary from the input folder
4. 'classifiedData.csv' - A CSV file of the classified data file which is a result of pairing 'testCenterControlledReleases.csv' with 'detectionReports.csv'
5. 'paperVars.tex' - A latex file summarizing variables calculated in the code
