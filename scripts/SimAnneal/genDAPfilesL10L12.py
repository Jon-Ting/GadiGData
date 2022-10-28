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

print("Copying xyz files to individual directories and relabelling numerically...")
for NPconf in os.listdir(sourceDir):
    if 'min' not in NPconf: continue  # Skip the unminimised configurations
    confID = str(confCnt).zfill(7)
    print(f"  Conformation ID: {confID}")
    confDir = f"{targetDir}/{confID}"
    if not isdir(f"{targetDir}/{confID}"): os.mkdir(confDir)
    shutil.copy(f"{sourceDir}/{NPconf}", f"{confDir}/{confID}.xyz")
    confCnt += 1
print("All DONE!")
