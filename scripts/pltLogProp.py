# Goal: Generate plots of properties contained in MD simulation log files
# Author: Jonathan Yik Chang Ting
# Date: 23/11/2020
"""
- To do:
    - Enable subplots
"""

import sys
import lammps_logfile
import matplotlib.pyplot as plt


if __name__ == '__main__':
    logFileName, N, propList = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    # print(logFileName, N, prop_list)  # DEBUG
    logFile = lammps_logfile.File(logFileName)
    x = logFile.get("Time", run_num=N)
    for prop in propList:
        y = logFile.get(prop, run_num=N)
        plt.plot(x, y)
    plt.xlabel("Time (ps)")
    if len(propList) > 1: plt.ylabel("Energies (eV)")
    else: plt.ylabel(prop)
    plt.title("{0} over Time".format(prop))
    plt.legend(propList)
    plt.show()
