#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 23:25:19 2023

@author: tingwenyang
"""

import os
import numpy as np
import pandas as pd
import pickle

npName = 'AuPt20COL12'
varThreshs, corrThreshs = [0.01, 0.03, 0.05], [0.90, 0.95, 0.99]
dfScaledNoLVHCs = np.array([None] * 2501)
print(dfScaledNoLVHCs)
dfNoLVHCs = np.array([None] * 2501)

print(f"Combining {npName} pickle files...")
for (i, varThresh) in enumerate(varThreshs[0:1]):
    for (j, corrThresh) in enumerate(corrThreshs[1:2]):
        for k in range(2501):
            print(k)
            with open(f"pickleFiles/{npName}/biFeat_dfScaledNoLVHCs_laplacianScore150_{npName}_{k}.pickle", 'rb') as f: XSelected = pickle.load(f)
            dfScaledNoLVHCs[k] = XSelected
            with open(f"pickleFiles/{npName}/biFeat_dfNoLVHCs_laplacianScore150_{npName}_{k}.pickle", 'rb') as f: dfNoLVHC = pickle.load(f)
            dfNoLVHCs[k] = dfNoLVHC
            
with open(f"pickleFiles/biFeat_dfScaledNoLVHCs_laplacianScore150_{npName}.pickle", 'wb') as f: pickle.dump(dfScaledNoLVHCs, f)
with open(f"pickleFiles/biFeat_dfNoLVHCs_laplacianScore150_{npName}.pickle", 'wb') as f: pickle.dump(dfNoLVHCs, f)
