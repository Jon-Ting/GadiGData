# Goal: Figure out the relationship between simulation system size and walltime to be requested
# Author: Jonathan Yik Chang Ting
# Date: 17/12/2020

import os
import re
import pickle
import pandas as pd


RUN_LOCK = "run.lock"
simDirPath = "/scratch/q27/jt5911/SimAnneal"
dfCols = ['Name', 'Elements', 'Number of MPI tasks', 'Number of atoms', 'Total number of neighbours', 'Average neighbours per atom', 'Walltime (s)', 'Min memory allocated (MB)', 'Average memory allocated (MB)', 'Max memory allocated (MB)']


def extractProp(logFileName):
    """Extract particular quantities relevant to size-time relationship out of given LAMMPS log file"""
    with open(logFileName, 'r') as f:
        lineList = f.readlines()
        minMem, avgMem, maxMem = 0, 0, 0
        for i, line in enumerate(lineList):
            if 'processor' in line: numTask = int(line.strip().split()[0]) * int(line.strip().split()[2]) * int(line.strip().split()[4])
            elif 'reading atoms' in line: numAtom = int(lineList[i+1].strip().split()[0])
            elif 'Total #' in line: totNN = int(line.split()[-1])
            elif 'Ave' in line: avgNN = float(line.split()[-1])
            elif 'time:' in line: time = int(line.split(':')[-1]) + int(line.split(':')[-2])*60 + int(line.split(':')[-3])*3600
            elif 'Mbytes' in line:
                minMem += float(line.split()[-6])
                avgMem += float(line.split()[-4])
                maxMem += float(line.split()[-2])
            else: continue
    return (numTask, numAtom, totNN, avgNN, time, minMem, avgMem, maxMem)


def sortHuman(targetList):
    """Sort a given list in a more 'human' way"""
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
    targetList.sort(key=alphanum)
    return targetList


def collateLists(varLists, colNames):
    """Turn lists containing quantities of interest into a dataframe"""
    dataDict = {}
    for i, colName in enumerate(colNames): dataDict[colName] = varLists[i]
    df = pd.DataFrame(dataDict)
    # nameList = sortHuman(varLists[0])
    # dfSorted = df.set_index('Name').reindex(nameList).reset_index()
    # print("Sorted data frame:\n", dfSorted)
    # return dfSorted
    return df

def writeExcel(df, filePath='{0}/sizeTime.xlsx'.format(os.getcwd())):
    """Write a given dataframe to an Excel workbook"""
    print("Writing to Excel sheet...")
    # df.to_excel(filePath, sheet_name='Sheet1')
    writer = pd.ExcelWriter(filePath, engine='xlsxwriter')
    df.to_excel(writer, startrow=1, sheet_name='Sheet1', index=False)
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    for i, col in enumerate(df.columns):
        column_len = df[col].astype(str).str.len().max()
        column_len = max(column_len, len(col)) + 2
        worksheet.set_column(i, i, column_len)
    writer.save()
    return workbook


def tabLog(inpPath=os.getcwd()):
    """Tabulate quantities of interest from LAMMPS log files in an Excel workbook"""
    if os.path.isfile("{0}/varList.pickle".format(inpPath)):
        with open('{0}/varList.pickle'.format(inpPath), 'rb') as f: varLists = pickle.load(f)
    else:
        nameList, eleList, numTaskList, numAtomList, totNNList, avgNNList, timeList, minMemList, avgMemList, maxMemList = [], [], [], [], [], [], [], [], [], []
        varLists = [nameList, eleList, numTaskList, numAtomList, totNNList, avgNNList, timeList, minMemList, avgMemList, maxMemList]
        bnpTypes = [i for i in os.listdir(inpPath) if os.path.isdir('{0}/{1}'.format(inpPath, i))]
        for bnpType in bnpTypes:
            bnpTypePath = "{0}/{1}".format(inpPath, bnpType)
            for bnpDirName in os.listdir(bnpTypePath):
                try:
                    if os.path.isfile("{0}/{1}/{2}".format(bnpTypePath, bnpDirName, RUN_LOCK)): continue
                    print("Name:", bnpDirName)
                    elements = "".join(re.findall(r'[A-Z][a-z]', bnpDirName))
                    logFileName = "{0}/{1}/{1}S0.log".format(bnpTypePath, bnpDirName)
                    variables = (bnpDirName, elements) + extractProp(logFileName)
                    for i, varList in enumerate(varLists): varList.append(variables[i])
                except (NotADirectoryError, FileNotFoundError) as error:
                    print(error)
                    continue
        with open('{0}/varList.pickle'.format(inpPath), 'wb') as f: pickle.dump(varLists, f)
    df = collateLists(varLists=varLists, colNames=dfCols)
    workbook = writeExcel(df=df, filePath='{0}/sizeTime.xlsx'.format(inpPath))


if __name__ == '__main__':
    print("Tabulating values of interest from LAMMPS log files to an Excel sheet...")
    tabLog(inpPath=simDirPath)
