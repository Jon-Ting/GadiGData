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
import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sns
import lammps_logfile as llog
import matplotlib.pyplot as plt


sns.set_style('ticks')
sns.set_palette(palette='cubehelix')
sns.set_context("paper", rc={'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12, 'lines.linewidth': 1.5, 'lines.markersize': 10})


def movingAvg(origArr, winSize=10):
    return np.convolve(origArr, np.ones(winSize), 'valid') / winSize


def plotEqPE(logFileName, warmStep=20, avgWinSize=[10, 500], examPeriod=500):
    plt.clf()
    logFile = llog.File(logFileName)
    minStep = len(logFile.get("Step", run_num=0))  # Minimisation steps
    time, potEng = logFile.get("Time", run_num=-1)[minStep:], logFile.get("PotEng", run_num=-1)[minStep:]
    overlapIdxList = list(range(1000, (logFile.get_num_partial_logs()-1) * 1000, 1000))  # 0.1 ps
    time, potEng = np.delete(time, overlapIdxList), np.delete(potEng, overlapIdxList)
    lineWidth, legendList = 0.5, ['0.1']
    ax = sns.lineplot(x=time[warmStep:], y=potEng[warmStep:], linewidth=lineWidth)
    for winSize in avgWinSize:
        timeAvg, potEngAvg = movingAvg(origArr=time, winSize=winSize), movingAvg(origArr=potEng, winSize=winSize)
        legendList.append(str(winSize/10))
        lineWidth += 1
        ax = sns.lineplot(x=timeAvg[warmStep:], y=potEngAvg[warmStep:], linewidth=lineWidth, ax=ax)
    window1, window2 = potEng[-examPeriod*2:-examPeriod], potEng[-examPeriod:]  # Examine the last {examPeriod*2} ps
    tStat, pVal = scipy.stats.ttest_ind(window1, window2, equal_var=True)  # Equal variance (pooled) t-test 
    legendTitle = 'T({0}): {1:.2f},  p: {2:.5f}\nAverage Window Size (ps)'.format(int(examPeriod*2-2), tStat, pVal)
    plt.xlabel("Time (ps)")
    plt.ylabel("Potential Energy (eV)")
    plt.title("{0} Stage 0 Potential Energy over Time".format(logFileName.split('/')[-1][:-6]), y=1.04)
    plt.legend(legendList, title=legendTitle)
    plt.tight_layout()
    return tStat, pVal


def plotProp(logFileName, propList, N=-1):
    logFile = llog.File(logFileName)
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
        plotProp(sys.argv[1], [sys.argv[2]], int(sys.argv[3]))
        plt.show()
    else:
        dirNameStr = "L10"
        pThresh = 0.00001  # Assuming H0, the observed difference could be expected 1 time out of 100000 repetitions of the same study
        avgWinSize, warmStep, examPeriod = [10, 500], 20, 500  # 0.1 ps
        simDirPath = "/scratch/q27/jt5911/SimAnneal"
        # simDirPath = "/mnt/c/Users/User/Documents/PhD/Workstation/Data/SimAnneal"  # DEBUG
        figDirPath = "{0}/figures/eqPE".format(simDirPath)        
        for typeDirName in os.listdir(simDirPath):
            typeDirPath = "{0}/{1}".format(simDirPath, typeDirName)
            for dirName in os.listdir(typeDirPath):
                if dirNameStr in dirName:
                    logFileName = "{0}/{1}/{1}S0.log".format(typeDirPath, dirName)
                    try:
                        tStat, pVal = plotEqPE(logFileName, avgWinSize=avgWinSize, warmStep=warmStep, examPeriod=examPeriod)
                        plt.savefig(fname="{0}/{1}.png".format(figDirPath, logFileName.split('/')[-1][:-6]))
                        if pVal > pThresh: print("{0} equilibrated (p = {1:.5f})".format(dirName, pVal))
                        else: print("{0} unequilibrated (p = {1:.6f})".format(dirName, pVal))
                        # plt.show()
                    except NotADirectoryError:
                        print("{0} not a directory".format(dirName))
                        continue

