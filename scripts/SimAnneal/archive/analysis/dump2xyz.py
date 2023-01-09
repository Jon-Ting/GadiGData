"""
Goal: Convert LAMMPS dump files into xyz format
Author: Jonathan Yik Chang Ting
Date: 6/1/2021
"""
import sys

NCPAC_DIR = "/g/data/q27/jt5911/NCPac"

if __name__ == '__main__':
    # dumpDirPath = sys.argv[2] if len(sys.argv) == 2 else "/scratch/q27/jt5911/SimAnneal/L10/AuCo20COL10/AuCo20COL10S1"
    dumpDirPath = "/scratch/q27/jt5911/SimAnneal/RAL/AuPd40TO50RAL0/AuPd40TO50RAL0S1"
    dumpDirPath = "/home/564/jt5911/test"
    # dirName = dumpDirPath.split("/")[-1]
    dumpFiles = dump("{0}/*".format(dumpDirPath))
    #dumpFiles.sort()
    xyzFile = xyz(dumpFiles)
    xyzFile.many("/home/564/jt5911/testNew/new")
    # xyzFile.one("{0}/bnp.xyz".format(NCPAC_DIR))
