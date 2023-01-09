"""
Goal: Convert LAMMPS dump files into xyz format
Author: Jonathan Yik Chang Ting
Date: 6/1/2021
"""
import sys
from os import listdir
from os.path import isfile, join
from ase.io import write
from ase.io.lammpsdata import read_lammps_data


if __name__ == '__main__':
    fileList =  [f for f in listdir("/home/564/jt5911/test/") if isfile(join("test", f))]
    for f in fileList:
        mnp = read_lammps_data("test/{0}".format(f), style='atomic', units='metal')
        write(filename="testNew/{0}.xyz".format(f[:-4]), images=mnp, format="xyz")
    print("Done!")
