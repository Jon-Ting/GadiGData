# Goal: Generate plots of properties contained in MD simulation log files
# Author: Jonathan Yik Chang Ting
# Date: 23/11/2020
"""
- To do:
    - Enable subplots
    - Lindemann index
"""

import os
import sys
import lammps_logfile
import matplotlib.pyplot as plt


def plotProp(logFileName, N, propList):
    logFile = lammps_logfile.File(logFileName)
    x = logFile.get("Time", run_num=N)
    for prop in propList:
        y = logFile.get(prop, run_num=N)
        plt.plot(x, y)
    plt.xlabel("Time (ps)")
    if len(propList) > 1: plt.ylabel("Energies (eV)")
    else: plt.ylabel(prop)
    plt.title("{0} over Time".format(prop))


if __name__ == '__main__':
    if len(sys.argv) == 4:
        plotProp(sys.argv[1], int(sys.argv[2]), [sys.argv[3]])
        plt.show()
    dirPath, dirNameStr, N, stage, propList = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5:]
    # print(dirPath, dirFileName, N, prop_list)  # DEBUG
    for dirName in os.listdir(dirPath):
        if dirNameStr in dirName:
            logFileName = "{0}/{0}S{1}.log".format(dirName, stage)
            plotProp(logFileName, N, propList)
    plt.legend()
    plt.show()
