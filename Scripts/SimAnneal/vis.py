# Goal: Generate plots of properties contained in MD simulation log files
# Author: Jonathan Yik Chang Ting
# Date: 23/11/2020
"""
Help on package vis:

NAME
    vis

DESCRIPTION
    Visualisation module for nanoparticle simulations
    ================================================

TO DO
    - Automatically identify melting range of NP from RDF
"""

import os
import sys
import glob
import subprocess
import scipy.stats
import numpy as np
import pandas as pd
import seaborn as sns
import lammps_logfile as llog
import matplotlib.pyplot as plt
import banpei
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

sns.set()
sns.set_style('ticks')
sns.set_palette(palette='cubehelix')
sns.set_context("paper", rc={'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 12, 'lines.linewidth': 1.5, 'lines.markersize': 8})
simDirPath = "/scratch/q27/jt5911/SimAnneal"
figDirPath = "{0}/figures/eqPE".format(simDirPath)
expStructNames = ["AuBP1_25Pd", "AuBP1_50Pd", "AuBP1_75Pd", "H1_25Pd", "H1_50Pd", "H1_75Pd", "Pd4_25Pd", "Pd4_25Pd"]


def zScorePeakAlg(y, lag, threshold, influence):
    """
    Implementation of peak detction algorithm from http://stackoverflow.com/a/22640362/6029703

    Example:
    y = np.array([1,1,1.1,1,0.9,1,1,1.1,1,0.9,1,1.1,1,1,0.9,1,1,1.1,1,1,1,
              1,1.1,0.9,1,1.1,1,1,0.9,1,1.1,1,1,1.1,1,0.8,0.9,1,1.2,
              0.9,1,1,1.1,1.2,1,1.5,1,3,2,5,3,2,1,1,1,0.9,1,1,3,2.6,4,
              3,3.2,2,1,1,0.8,4,4,2,2.5,1,1,1])
    lag, threshold, influence = 10, 2, 1
    result = zScorePeakAlg(y, lag, threshold, influence)
    plt.plot(np.arange(1, len(y)+1), y)
    plt.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
    """
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0] * len(y)
    stdFilter = [0] * len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y) - 1):
        if abs(y[i] - avgFilter[i - 1]) > threshold * stdFilter [i - 1]:
            if y[i] > avgFilter[i - 1]:
                signals[i] = 1
            else:
                signals[i] = -1
            filteredY[i] = influence*y[i] + (1-influence)*filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i - lag):i])
            stdFilter[i] = np.std(filteredY[(i - lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i - lag):i])
            stdFilter[i] = np.std(filteredY[(i - lag):i])
    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))


def pltBL_NCP(inpDir, numFrame=10, intFrame=10):
    sns.set_context(rc={'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 12, 'lines.linewidth': 1.5, 'lines.markersize': 8})
    dfAu, dfPd = pd.read_csv("{0}/Au_nanoparticle_dataset.csv".format(inpDir), header=0), pd.read_csv("{0}/Pd_nanoparticle_dataset.csv".format(inpDir), header=0)
    headerList, titleList = ["Avg_bonds", "Std_bonds", "Max_bonds", "Min_bonds"], ["Average", "Standard Deviation", "Maximum", "Minimum"]
    dfAuBond, dfPdBond = dfAu[headerList], dfPd[headerList]
    figAu, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
    figPd, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(8, 6))
    axListAu, axListPd = [ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]
    dfLists, axLists = [dfAuBond, dfPdBond], [axListAu, axListPd]
    for (i, df) in enumerate(dfLists):
        for (j, header) in enumerate(headerList):
            df[headerList[j]].plot.kde(ax=axLists[i][j], title=titleList[j])
    BLfilePath = "{0}/*R5*od_BOND_length.csv".format(inpDir)
    for BLfile in glob.glob(BLfilePath): 
        dfAll = pd.read_csv(BLfile, header=1)
        dfAuStats, dfPdStats = dfAll.iloc[:, 24:28], dfAll.iloc[:, 10:14]
        dfStatLists = [dfAuStats, dfPdStats]
        for (i, df) in enumerate(dfStatLists):
            for (j, header) in enumerate(headerList):
                axLists[i][j].axvline(x=df.iloc[0, j])
    plt.xlabel("Bond Length (Angstrom)")
    # plt.title("Bond Length Statistics Distribution")
    plt.tight_layout()


def pltADF_NCP(inpDir, numFrame=10, intFrame=10):
    ADFfilePath = "{0}/od_G3.csv".format(inpDir)
    sns.set_context(rc={'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 12, 'lines.linewidth': 1.5, 'lines.markersize': 8})
    for ADFfile in glob.glob(ADFfilePath): 
        dfAll = pd.read_csv(ADFfile, header=1)
        dfADF = dfAll[dfAll["      Type1"] == "      Total"]
        df = dfADF.copy()
        df.drop(["      Type1", "      Type2", "      Type3", "     Angles", "        Avg", "    Std Dev"], axis=1, inplace=True)
        df = df.convert_dtypes()
        df.set_index("      Frame", inplace=True)
        df.columns = df.columns.astype(float)
        df = df.transpose()
        for j in range(numFrame):
            df.iloc[:, j*intFrame] = df.iloc[:, j*intFrame:(j+1)*intFrame].mean(axis=1)
        df = df.iloc[:, 0:numFrame*intFrame:intFrame]
        df.plot(kind="line")
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel("Bond Angle (Degree)")
    plt.ylabel("Angular Distribution Function")
    plt.title("Heating Phase Total ADF")
    plt.tight_layout()


def pltRDF_NCP(inpDir, numFrame=10, intFrame=10):
    RDFfilePath = "{0}/*R5*od_GR.csv".format(inpDir)
    sns.set_context(rc={'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 12, 'lines.linewidth': 1.5, 'lines.markersize': 8})
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
    axList, titleList, legendList = [ax1, ax2, ax3, ax4], ["Total RDF", "Au-Au RDF", "Au-Co RDF", "Co-Co RDF"], []
    for RDFfile in glob.glob(RDFfilePath): 
        legendList.append(RDFfile)
        dfAll = pd.read_csv(RDFfile, header=1)
        dfTot = dfAll[dfAll["      Type1"] == "      Total"]
        df11 = dfAll[(dfAll["      Type1"] == "          1") & (dfAll["      Type2"] == "          1")]
        df12 = dfAll[(dfAll["      Type1"] == "          1") & (dfAll["      Type2"] == "          2")]
        df22 = dfAll[dfAll["      Type1"] == "          2"]
        for (i, dfRDF) in enumerate((dfTot, df11, df12, df22)):
            df = dfRDF.copy()
            df.drop(["      Type1", "      Type2"], axis=1, inplace=True)
            df = df.convert_dtypes()
            df.set_index("      Frame", inplace=True)
            df.columns = df.columns.astype(float)
            df = df.transpose()
            for j in range(numFrame):
                df.iloc[:, j*intFrame] = df.iloc[:, j*intFrame:(j+1)*intFrame].mean(axis=1)
            df = df.iloc[:, 0:numFrame*intFrame:intFrame]
            df.plot(kind="line", ax=axList[i], title=titleList[i]) #, legend=False)
    #plt.legend(legendList, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fancybox=True, shadow=True)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel("Interatomic Distance (Angstrom)")
    plt.ylabel("Radial Distribution Function")
    # plt.title("Heating Phase RDF")
    #cmap = sns.set_palette(palette='cubehelix')
    #customLines = [Line2D([0], [0], color=cmap(0.5), lw=2)] * 5
    #plt.legend(customLines, legendList, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fancybox=True, shadow=True)
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.25, hspace=0.1)
    plt.tight_layout()


def pltRDF(numBin=200, numCurve=10, interval=1, plotCoord=True, useZScore=False):
    with open("/scratch/q27/jt5911/SimAnneal/L10/AuCo20COL10/AuCo20COL10S1.rdf") as f:
        lines = f.readlines()[3:]  # Remove irrelevant title lines
        rdfLists, coordLists, legendList = [], [], []
        for (i, line) in enumerate(lines):
            if i == 0 or i % (numBin+1) == 0:
                rList, rdfList, coordList = [], [], []
                legendList.append(int(int(line.split()[0]) / 1000))
            else:
                rList.append(float(line.split()[1]))
                rdfList.append(float(line.split()[2]))
                coordList.append(float(line.split()[3]))
            if (i+1) % (numBin+1) == 0:
                rdfLists.append(rdfList)
                coordLists.append(coordList)
    rdfLists = rdfLists[:numCurve:interval]
    if plotCoord:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))
        for rdfList in rdfLists:
            sns.lineplot(x=rList, y=rdfList, ax=ax1)
            peaks, _ = find_peaks(rdfList)
            plt.plot(peaks, rdfList[peaks], "x")
            if useZScore:  # Not recommended for this case
                lag, threshold, influence = 10, 2, 1
                rdfArray = np.array(rdfList)
                peaks = zScorePeakAlg(rdfArray, lag=lag, threshold=threshold, influence=influence)
                ax1.plot(np.arange(1, len(rdfArray)+1), rdfArray)
                ax2 = ax1.twinx()
                ax2.step(np.arange(1, len(rdfArray)+1), peaks["signals"], color="red", lw=2)
        ax1.set_xlabel("Interatomic Distance (Angstrom)")
        ax1.set_ylabel("Occurence")
        ax1.set_title("Heating Phase RDF")
        ax1.legend(legendList, title="Heat time (ps)")
        for coordList in coordLists: sns.lineplot(x=rList, y=coordList, ax=ax2)
        ax2.set_xlabel("Interatomic Distance (Angstrom)")
        ax2.set_ylabel("Coordination Number")
        ax2.set_title("Heating Phase CN")
        ax2.legend(legendList, title="Heat time (ps)")
        plt.subplots_adjust(left=0.03, right=0.95, bottom=0.2, top=0.9, wspace=0.25, hspace=0.1)
        plt.tight_layout()
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
        varThresh, first3peaks, peaksList = 3, np.zeros((1, 3)), []
        for (i, rdfList) in enumerate(rdfLists):
            sns.lineplot(x=rList, y=rdfList, ax=ax1)
            peaks, _ = find_peaks(rdfList, distance=20, width=20)
            peaksList.append(peaks)
            melted = False
            if i != 0:
                for (j, peak) in enumerate(peaks):
                    if (not peak+varThresh > first3peaks[j]) or (not peak-varThresh < first3peaks[j]):
                        meltFrame = i
                        print("2nd RDF peak disappeared!")
                        break
            if melted: break
            else: first3peaks = np.diff(peaks)[:3]
        plt.xlabel("Interatomic Distance (Angstrom)")
        plt.ylabel("Occurence")
        plt.title("Heating Phase RDF")
        ax1.legend(legendList, title="Heat time (ps)")
        for (i, peaks) in enumerate(peaksList):
            sns.scatterplot(x=np.array(rList)[peaks], y=np.array(rdfLists[i])[peaks], marker="+", ax=ax1)
        plt.tight_layout()
    return meltFrame
 

def pltLind(winSize=10, numFrameLag=2):
    with open("/g/data/q27/jt5911/NCPac/od_LINDEX.dat") as f:
        lines = f.readlines()[3:]  # Remove irrelevant title lines
        frame, lind, i = [], [], 1
        for line in lines:
            if line.split()[1] == "0.000000": continue  # Exclude zeroes from time frame averaging
            frame.append(int(line.split()[0]))
            lind.append(float(line.split()[1]))
    sstModel = banpei.SST(w=winSize, L=numFrameLag)  # Change point detection via singular spectrum transformation
    changePointScores = sstModel.detect(lind, is_lanczos=True)
    sns.lineplot(x=frame, y=lind, color='r')#, marker='o', markersize=5)
    plt.xlabel("Frame")
    plt.ylabel("Lindemann Index")
    plt.legend(["Lindemann Index"])
    ax2 = plt.twinx()
    sns.lineplot(x=frame, y=changePointScores, color='b')#, marker='s', color='b', ax=ax2)
    plt.ylabel("Change-Point Score")
    plt.legend(["Change-Point Score"])
    plt.title("Change in Lindemann Index over Time")
    plt.tight_layout()
    return np.argmax(changePointScores)


def movingAvg(origArr, winSize=10):
    return np.convolve(origArr, np.ones(winSize), 'valid') / winSize


def pltEqPE(logFileName, warmStep=20, avgWinSize=[10, 500], examPeriod=500):
    plt.clf()
    logFile = llog.File(logFileName)
    minStep = len(logFile.get("Step", run_num=0)) if "minimization" in logFile.output_before_first_run else 0
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
    tStat, pVal = scipy.stats.ttest_ind(window1, window2, equal_var=False)  # Welch's unequal variance t-test 
    legendTitle = 'T({0}): {1:.2f},  p: {2:.5f}\nAverage Window Size (ps)'.format(int(examPeriod*2-2), tStat, pVal)
    plt.xlabel("Time (ps)")
    plt.ylabel("Potential Energy (eV)")
    plt.title("{0} Stage 0 Potential Energy over Time".format(logFileName.split('/')[-1][:-6]), y=1.04)
    plt.legend(legendList, title=legendTitle)
    plt.tight_layout()
    return tStat, pVal


def pltLogProp(logFileName, xVar, propList, N=-1):
    logFile = llog.File(logFileName)
    x = logFile.get(xVar, run_num=N)
    for prop in propList:
        y = logFile.get(prop, run_num=N)
        plt.plot(x, y)
    if xVar == "Time": plt.xlabel("Time (ps)")
    elif xVar == "Temp": plt.xlabel("Temp (K)")
    if len(propList) > 1: plt.ylabel("Energies (eV)")
    else: plt.ylabel(prop)
    plt.title("{0} over {1}".format(prop, xVar))


def checkEq(typeDirPath, dirName, pThresh, avgWinSize, warmStep, examPeriod, skip=False, overwrite=False):
    logFileName = "{0}/{1}/{1}S0.log".format(typeDirPath, dirName)
    # logFileName = "{0}/{1}/log.lammps".format(typeDirPath, dirName)
    try:
        if skip:
            if os.stat("{0}/{1}/config.yml".format(typeDirPath, dirName)).st_size != 0: return
        tStat, pVal = pltEqPE(logFileName, avgWinSize=avgWinSize, warmStep=warmStep, examPeriod=examPeriod)
        if "DONE!" not in str(subprocess.check_output(["tail", "-5", logFileName])): return
        if not overwrite: 
            if os.path.isfile("{0}/{1}.png".format(figDirPath, logFileName.split('/')[-1][:-6])): return
        plt.savefig(fname="{0}/{1}.png".format(figDirPath, logFileName.split('/')[-1][:-6]))
        if pVal > pThresh:
            print("{0} equilibrated (p = {1:.5f})".format(dirName, pVal))
            with open("{0}/{1}/config.yml".format(typeDirPath, dirName), "w") as f: f.write("S0eq: true\n")
        else:
            print("{0} unequilibrated (p = {1:.6f})".format(dirName, pVal))
            with open("{0}/{1}/config.yml".format(typeDirPath, dirName), "w") as f: f.write("S0eq: false\n")
    except (NotADirectoryError, FileNotFoundError) as err:
        # print(err)
        return


if __name__ == '__main__':
    pThresh = 0.00001  # Assuming H0, the observed difference could be expected 1 time out of 100000 repetitions of the same study
    avgWinSize, warmStep, examPeriod = [10, 500], 20, 500  # 0.1 ps
    if len(sys.argv) == 2:
        if sys.argv[1] == "lind":  # Lindemann index plot
            changePointIdx = pltLind(winSize=10, numFrameLag=2)
            print("NP started melting from frame {0}".format(changePointIdx - 1))
        elif sys.argv[1] == "rdf":  # RDF plots
            pltRDF_NCP(numFrame=1, intFrame=1)  # 7, 2 for heat phase
            #meltFrame = pltRDF(numBin=200, numCurve=7, interval=1, plotCoord=False, useZScore=False)
            #print("NP melted by frame {0}".format(meltFrame))
        elif sys.argv[1] == "adf":  # ADF plots
            pltADF_NCP(inpDir="/g/data/q27/jt5911/NCPac", numFrame=1, intFrame=1)
        elif sys.argv[1] == "exp":  # Experimental structure validation
            #pltRDF_NCP(inpDir="/g/data/q27/jt5911/ExpStruct", numFrame=1, intFrame=1)
            #pltADF_NCP(inpDir="/g/data/q27/jt5911/ExpStruct", numFrame=1, intFrame=1)
            pltBL_NCP(inpDir="/g/data/q27/jt5911/ExpStruct", numFrame=1, intFrame=1)
        plt.show()
    elif len(sys.argv) == 3:  # Check equilibration of specified BNP directory
        typeDirPath = "{0}/{1}".format(simDirPath, sys.argv[1])
        checkEq(typeDirPath, sys.argv[2], pThresh, avgWinSize, warmStep, examPeriod, skip=False, overwrite=True)
    elif len(sys.argv) == 5:  # Plot of specified x & y properties
        pltLogProp(logFileName=sys.argv[1], xVar=sys.argv[2], propList=[sys.argv[3]], N=int(sys.argv[4]))
        plt.show()
    else:  # Check equilibration of directories covered by dirNameStr
        dirNameStr = 'L10'
        for typeDirName in os.listdir(simDirPath):
            typeDirPath = "{0}/{1}".format(simDirPath, typeDirName)
            try:
                for dirName in os.listdir(typeDirPath):
                    if dirNameStr in dirName:
                        checkEq(typeDirPath, dirName, pThresh, avgWinSize, warmStep, examPeriod, skip=False, overwrite=True)
            except NotADirectoryError as err:
                # print(err)
                continue
