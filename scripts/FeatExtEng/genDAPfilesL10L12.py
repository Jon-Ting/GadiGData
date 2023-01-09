import os
from os.path import isdir, exists
import pandas as pd
import multiprocessing
from multiprocessing import Pool
import shutil
from zipfile import ZipFile


# Important variables to check!
confCnt = 188466
eleComb = 'AuCo'
sourceDir = f"/g/data/q27/jt5911/PostSim/{eleComb}L12"

# Variables that are more constant
targetDir = f"/scratch/q27/jt5911/SimAnneal/{eleComb}"
outMDfile = 'MDout.csv'  # Need S2.log files!
headerLine = f"CSIRO Nanostructure Databank - {eleComb} Nanoparticle Data Set\n"

print("Copying xyz files to individual directories and relabelling numerically...")
for NPconf in os.listdir(sourceDir):
    if 'min' not in NPconf: continue  # Skip the unminimised configurations
    confID = str(confCnt).zfill(7)
    print(f"  Conformation ID: {confID}")
    confDir = f"{targetDir}/{confID}"
    if not isdir(f"{targetDir}/{confID}"): os.mkdir(confDir)
    with open(f"{sourceDir}/{NPconf}", 'r') as f1:
        with open(f"{confDir}/{confID}.xyz", 'w') as f2:
            f2.write(f1.readline())
            f1.readline()  # Replace second line with CSIRO header
            f2.write(headerLine)
            f2.write(''.join([line for line in f1.readlines()]))
    # shutil.copy(f"{sourceDir}/{NPconf}", f"{confDir}/{confID}.xyz")
    confCnt += 1
print("All DONE!")
