"""
Goal: Convert LAMMPS dump files into xyz format
Author: Jonathan Yik Chang Ting
Date: 6/1/2021
"""
import sys

NCPAC_DIR = "/home/564/jt5911/NCPac"

if __name__ == '__main__':
    # dumpDirPath = sys.argv[2] if len(sys.argv) == 2 else "/scratch/q27/jt5911/SimAnneal/L10/AuCo20COL10/AuCo20COL10S1"
    #dumpDirPath = "/scratch/q27/jt5911/SimAnneal/L10/PtCo150COL10/PtCo150COL10S1"
    dumpDirPath = "/scratch/q27/jt5911/SimAnneal/L10/AuCo20COL10/AuCo20COL10S1"
    # dirName = dumpDirPath.split("/")[-1]
    dumpFiles = dump("{0}/*".format(dumpDirPath))
    dumpFiles.sort()
    xyzFile = xyz(dumpFiles)
    xyzFile.one("{0}/bnp.xyz".format(NCPAC_DIR))
