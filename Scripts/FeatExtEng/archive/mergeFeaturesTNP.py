import csv
import math
from multiprocessing import Pool
import os
from os.path import isdir, exists
import pandas as pd


runTask = 'concatNPfeats'  # 'mergeReformatData' or 'concatNPfeats' or 'debug'
runParallel = True
    
PROJECT, USER_NAME, ELE_COMB = 'q27', 'jt5911', 'AuPdPt'
MDoutFName = f"/scratch/{PROJECT}/{USER_NAME}/{ELE_COMB}/MDout.csv"
featSourcePath = f"/scratch/{PROJECT}/{USER_NAME}/{ELE_COMB}/finalData/Features"
featEngPath = f"/scratch/{PROJECT}/{USER_NAME}/{ELE_COMB}/finalData/FeatEng"
finalDataFName = f"/scratch/{PROJECT}/{USER_NAME}/{ELE_COMB}/finalData/{ELE_COMB}_nanoparticle_data.csv"

N_AVOGADRO = 6.02214076 * 10**23  # (atoms/mol)
A3_PER_M3 = 10 ** 30  # (Angstrom^3/m^3)
elePropDict = {'Au': {'rho': 19320, 'm': 0.196967, 'bulkE': -3.93}, 
               'Pt': {'rho': 21450, 'm': 0.195084, 'bulkE': -5.77}, 
               'Pd': {'rho': 12020, 'm': 0.10642, 'bulkE': -3.91}}  # density (kg/m^3), molar mass (kg/mol), cohesive energy per atom for bulk system (eV/atom) obtained from https://www.ctcms.nist.gov/potentials/system/ based on Zhou's work (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.69.144113), assuming lowest energy crystal structure in bulk

# All features (Elements ordered alphabetically)
ALL_HEADERS_LIST = ['T', 'P', 'Potential_E', 'Kinetic_E', 'Total_E', 
                    'N_atom_total', 'N_Au', 'N_Pd', 'N_Pt', 'N_atom_bulk', 'N_atom_surface', 'Vol_bulk_pack', 'Vol_sphere', 
                    'R_min', 'R_max', 'R_diff', 'R_avg', 'R_std', 'R_skew', 'R_kurt',
                    'S_100', 'S_111', 'S_110', 'S_311', 
                    'Curve_1-10', 'Curve_11-20', 'Curve_21-30', 'Curve_31-40', 'Curve_41-50', 'Curve_51-60', 'Curve_61-70', 'Curve_71-80', 'Curve_81-90', 'Curve_91-100', 'Curve_101-110', 'Curve_111-120', 'Curve_121-130', 'Curve_131-140', 'Curve_141-150', 'Curve_151-160', 'Curve_161-170', 'Curve_171-180',
                    
                    'MM_TCN_avg', 'MM_BCN_avg', 'MM_SCN_avg', 'MM_SOCN_avg', 'MM_TGCN_avg', 'MM_BGCN_avg', 'MM_SGCN_avg', 'MM_SOGCN_avg', 
                    'MM_TCN_0', 'MM_TCN_1', 'MM_TCN_2', 'MM_TCN_3', 'MM_TCN_4', 'MM_TCN_5', 'MM_TCN_6', 'MM_TCN_7', 'MM_TCN_8', 'MM_TCN_9', 'MM_TCN_10', 'MM_TCN_11', 'MM_TCN_12', 'MM_TCN_13', 'MM_TCN_14', 'MM_TCN_15', 'MM_TCN_16', 'MM_TCN_17', 'MM_TCN_18', 'MM_TCN_19', 'MM_TCN_20', 
                    'MM_BCN_0', 'MM_BCN_1', 'MM_BCN_2', 'MM_BCN_3', 'MM_BCN_4', 'MM_BCN_5', 'MM_BCN_6', 'MM_BCN_7', 'MM_BCN_8', 'MM_BCN_9', 'MM_BCN_10', 'MM_BCN_11', 'MM_BCN_12', 'MM_BCN_13', 'MM_BCN_14', 'MM_BCN_15', 'MM_BCN_16', 'MM_BCN_17', 'MM_BCN_18', 'MM_BCN_19', 'MM_BCN_20', 
                    'MM_SCN_0', 'MM_SCN_1', 'MM_SCN_2', 'MM_SCN_3', 'MM_SCN_4', 'MM_SCN_5', 'MM_SCN_6', 'MM_SCN_7', 'MM_SCN_8', 'MM_SCN_9', 'MM_SCN_10', 'MM_SCN_11', 'MM_SCN_12', 'MM_SCN_13', 'MM_SCN_14', 'MM_SCN_15', 'MM_SCN_16', 'MM_SCN_17', 'MM_SCN_18', 'MM_SCN_19', 'MM_SCN_20', 
                    'MM_SOCN_0', 'MM_SOCN_1', 'MM_SOCN_2', 'MM_SOCN_3', 'MM_SOCN_4', 'MM_SOCN_5', 'MM_SOCN_6', 'MM_SOCN_7', 'MM_SOCN_8', 'MM_SOCN_9', 'MM_SOCN_10', 'MM_SOCN_11', 'MM_SOCN_12', 'MM_SOCN_13', 'MM_SOCN_14', 'MM_SOCN_15', 'MM_SOCN_16', 'MM_SOCN_17', 'MM_SOCN_18', 'MM_SOCN_19', 'MM_SOCN_20', 
                    'MM_TGCN_0', 'MM_TGCN_1', 'MM_TGCN_2', 'MM_TGCN_3', 'MM_TGCN_4', 'MM_TGCN_5', 'MM_TGCN_6', 'MM_TGCN_7', 'MM_TGCN_8', 'MM_TGCN_9', 'MM_TGCN_10', 'MM_TGCN_11', 'MM_TGCN_12', 'MM_TGCN_13', 'MM_TGCN_14', 'MM_TGCN_15', 'MM_TGCN_16', 'MM_TGCN_17', 'MM_TGCN_18', 'MM_TGCN_19', 'MM_TGCN_20', 
                    'MM_BGCN_0', 'MM_BGCN_1', 'MM_BGCN_2', 'MM_BGCN_3', 'MM_BGCN_4', 'MM_BGCN_5', 'MM_BGCN_6', 'MM_BGCN_7', 'MM_BGCN_8', 'MM_BGCN_9', 'MM_BGCN_10', 'MM_BGCN_11', 'MM_BGCN_12', 'MM_BGCN_13', 'MM_BGCN_14', 'MM_BGCN_15', 'MM_BGCN_16', 'MM_BGCN_17', 'MM_BGCN_18', 'MM_BGCN_19', 'MM_BGCN_20', 
                    'MM_SGCN_0', 'MM_SGCN_1', 'MM_SGCN_2', 'MM_SGCN_3', 'MM_SGCN_4', 'MM_SGCN_5', 'MM_SGCN_6', 'MM_SGCN_7', 'MM_SGCN_8', 'MM_SGCN_9', 'MM_SGCN_10', 'MM_SGCN_11', 'MM_SGCN_12', 'MM_SGCN_13', 'MM_SGCN_14', 'MM_SGCN_15', 'MM_SGCN_16', 'MM_SGCN_17', 'MM_SGCN_18', 'MM_SGCN_19', 'MM_SGCN_20', 
                    'MM_SOGCN_0', 'MM_SOGCN_1', 'MM_SOGCN_2', 'MM_SOGCN_3', 'MM_SOGCN_4', 'MM_SOGCN_5', 'MM_SOGCN_6', 'MM_SOGCN_7', 'MM_SOGCN_8', 'MM_SOGCN_9', 'MM_SOGCN_10', 'MM_SOGCN_11', 'MM_SOGCN_12', 'MM_SOGCN_13', 'MM_SOGCN_14', 'MM_SOGCN_15', 'MM_SOGCN_16', 'MM_SOGCN_17', 'MM_SOGCN_18', 'MM_SOGCN_19', 'MM_SOGCN_20', 
                    
                    'AuM_TCN_avg', 'AuM_BCN_avg', 'AuM_SCN_avg', 'AuM_SOCN_avg', 'AuM_TGCN_avg', 'AuM_BGCN_avg', 'AuM_SGCN_avg', 'AuM_SOGCN_avg', 
                    'AuM_TCN_0', 'AuM_TCN_1', 'AuM_TCN_2', 'AuM_TCN_3', 'AuM_TCN_4', 'AuM_TCN_5', 'AuM_TCN_6', 'AuM_TCN_7', 'AuM_TCN_8', 'AuM_TCN_9', 'AuM_TCN_10', 'AuM_TCN_11', 'AuM_TCN_12', 'AuM_TCN_13', 'AuM_TCN_14', 'AuM_TCN_15', 'AuM_TCN_16', 'AuM_TCN_17', 'AuM_TCN_18', 'AuM_TCN_19', 'AuM_TCN_20', 
                    'AuM_BCN_0', 'AuM_BCN_1', 'AuM_BCN_2', 'AuM_BCN_3', 'AuM_BCN_4', 'AuM_BCN_5', 'AuM_BCN_6', 'AuM_BCN_7', 'AuM_BCN_8', 'AuM_BCN_9', 'AuM_BCN_10', 'AuM_BCN_11', 'AuM_BCN_12', 'AuM_BCN_13', 'AuM_BCN_14', 'AuM_BCN_15', 'AuM_BCN_16', 'AuM_BCN_17', 'AuM_BCN_18', 'AuM_BCN_19', 'AuM_BCN_20', 
                    'AuM_SCN_0', 'AuM_SCN_1', 'AuM_SCN_2', 'AuM_SCN_3', 'AuM_SCN_4', 'AuM_SCN_5', 'AuM_SCN_6', 'AuM_SCN_7', 'AuM_SCN_8', 'AuM_SCN_9', 'AuM_SCN_10', 'AuM_SCN_11', 'AuM_SCN_12', 'AuM_SCN_13', 'AuM_SCN_14', 'AuM_SCN_15', 'AuM_SCN_16', 'AuM_SCN_17', 'AuM_SCN_18', 'AuM_SCN_19', 'AuM_SCN_20', 
                    'AuM_SOCN_0', 'AuM_SOCN_1', 'AuM_SOCN_2', 'AuM_SOCN_3', 'AuM_SOCN_4', 'AuM_SOCN_5', 'AuM_SOCN_6', 'AuM_SOCN_7', 'AuM_SOCN_8', 'AuM_SOCN_9', 'AuM_SOCN_10', 'AuM_SOCN_11', 'AuM_SOCN_12', 'AuM_SOCN_13', 'AuM_SOCN_14', 'AuM_SOCN_15', 'AuM_SOCN_16', 'AuM_SOCN_17', 'AuM_SOCN_18', 'AuM_SOCN_19', 'AuM_SOCN_20', 
                    'AuM_TGCN_0', 'AuM_TGCN_1', 'AuM_TGCN_2', 'AuM_TGCN_3', 'AuM_TGCN_4', 'AuM_TGCN_5', 'AuM_TGCN_6', 'AuM_TGCN_7', 'AuM_TGCN_8', 'AuM_TGCN_9', 'AuM_TGCN_10', 'AuM_TGCN_11', 'AuM_TGCN_12', 'AuM_TGCN_13', 'AuM_TGCN_14', 'AuM_TGCN_15', 'AuM_TGCN_16', 'AuM_TGCN_17', 'AuM_TGCN_18', 'AuM_TGCN_19', 'AuM_TGCN_20', 
                    'AuM_BGCN_0', 'AuM_BGCN_1', 'AuM_BGCN_2', 'AuM_BGCN_3', 'AuM_BGCN_4', 'AuM_BGCN_5', 'AuM_BGCN_6', 'AuM_BGCN_7', 'AuM_BGCN_8', 'AuM_BGCN_9', 'AuM_BGCN_10', 'AuM_BGCN_11', 'AuM_BGCN_12', 'AuM_BGCN_13', 'AuM_BGCN_14', 'AuM_BGCN_15', 'AuM_BGCN_16', 'AuM_BGCN_17', 'AuM_BGCN_18', 'AuM_BGCN_19', 'AuM_BGCN_20', 
                    'AuM_SGCN_0', 'AuM_SGCN_1', 'AuM_SGCN_2', 'AuM_SGCN_3', 'AuM_SGCN_4', 'AuM_SGCN_5', 'AuM_SGCN_6', 'AuM_SGCN_7', 'AuM_SGCN_8', 'AuM_SGCN_9', 'AuM_SGCN_10', 'AuM_SGCN_11', 'AuM_SGCN_12', 'AuM_SGCN_13', 'AuM_SGCN_14', 'AuM_SGCN_15', 'AuM_SGCN_16', 'AuM_SGCN_17', 'AuM_SGCN_18', 'AuM_SGCN_19', 'AuM_SGCN_20', 
                    'AuM_SOGCN_0', 'AuM_SOGCN_1', 'AuM_SOGCN_2', 'AuM_SOGCN_3', 'AuM_SOGCN_4', 'AuM_SOGCN_5', 'AuM_SOGCN_6', 'AuM_SOGCN_7', 'AuM_SOGCN_8', 'AuM_SOGCN_9', 'AuM_SOGCN_10', 'AuM_SOGCN_11', 'AuM_SOGCN_12', 'AuM_SOGCN_13', 'AuM_SOGCN_14', 'AuM_SOGCN_15', 'AuM_SOGCN_16', 'AuM_SOGCN_17', 'AuM_SOGCN_18', 'AuM_SOGCN_19', 'AuM_SOGCN_20', 
                    
                    'PdM_TCN_avg', 'PdM_BCN_avg', 'PdM_SCN_avg', 'PdM_SOCN_avg', 'PdM_TGCN_avg', 'PdM_BGCN_avg', 'PdM_SGCN_avg', 'PdM_SOGCN_avg', 
                    'PdM_TCN_0', 'PdM_TCN_1', 'PdM_TCN_2', 'PdM_TCN_3', 'PdM_TCN_4', 'PdM_TCN_5', 'PdM_TCN_6', 'PdM_TCN_7', 'PdM_TCN_8', 'PdM_TCN_9', 'PdM_TCN_10', 'PdM_TCN_11', 'PdM_TCN_12', 'PdM_TCN_13', 'PdM_TCN_14', 'PdM_TCN_15', 'PdM_TCN_16', 'PdM_TCN_17', 'PdM_TCN_18', 'PdM_TCN_19', 'PdM_TCN_20', 
                    'PdM_BCN_0', 'PdM_BCN_1', 'PdM_BCN_2', 'PdM_BCN_3', 'PdM_BCN_4', 'PdM_BCN_5', 'PdM_BCN_6', 'PdM_BCN_7', 'PdM_BCN_8', 'PdM_BCN_9', 'PdM_BCN_10', 'PdM_BCN_11', 'PdM_BCN_12', 'PdM_BCN_13', 'PdM_BCN_14', 'PdM_BCN_15', 'PdM_BCN_16', 'PdM_BCN_17', 'PdM_BCN_18', 'PdM_BCN_19', 'PdM_BCN_20', 
                    'PdM_SCN_0', 'PdM_SCN_1', 'PdM_SCN_2', 'PdM_SCN_3', 'PdM_SCN_4', 'PdM_SCN_5', 'PdM_SCN_6', 'PdM_SCN_7', 'PdM_SCN_8', 'PdM_SCN_9', 'PdM_SCN_10', 'PdM_SCN_11', 'PdM_SCN_12', 'PdM_SCN_13', 'PdM_SCN_14', 'PdM_SCN_15', 'PdM_SCN_16', 'PdM_SCN_17', 'PdM_SCN_18', 'PdM_SCN_19', 'PdM_SCN_20', 
                    'PdM_SOCN_0', 'PdM_SOCN_1', 'PdM_SOCN_2', 'PdM_SOCN_3', 'PdM_SOCN_4', 'PdM_SOCN_5', 'PdM_SOCN_6', 'PdM_SOCN_7', 'PdM_SOCN_8', 'PdM_SOCN_9', 'PdM_SOCN_10', 'PdM_SOCN_11', 'PdM_SOCN_12', 'PdM_SOCN_13', 'PdM_SOCN_14', 'PdM_SOCN_15', 'PdM_SOCN_16', 'PdM_SOCN_17', 'PdM_SOCN_18', 'PdM_SOCN_19', 'PdM_SOCN_20', 
                    'PdM_TGCN_0', 'PdM_TGCN_1', 'PdM_TGCN_2', 'PdM_TGCN_3', 'PdM_TGCN_4', 'PdM_TGCN_5', 'PdM_TGCN_6', 'PdM_TGCN_7', 'PdM_TGCN_8', 'PdM_TGCN_9', 'PdM_TGCN_10', 'PdM_TGCN_11', 'PdM_TGCN_12', 'PdM_TGCN_13', 'PdM_TGCN_14', 'PdM_TGCN_15', 'PdM_TGCN_16', 'PdM_TGCN_17', 'PdM_TGCN_18', 'PdM_TGCN_19', 'PdM_TGCN_20', 
                    'PdM_BGCN_0', 'PdM_BGCN_1', 'PdM_BGCN_2', 'PdM_BGCN_3', 'PdM_BGCN_4', 'PdM_BGCN_5', 'PdM_BGCN_6', 'PdM_BGCN_7', 'PdM_BGCN_8', 'PdM_BGCN_9', 'PdM_BGCN_10', 'PdM_BGCN_11', 'PdM_BGCN_12', 'PdM_BGCN_13', 'PdM_BGCN_14', 'PdM_BGCN_15', 'PdM_BGCN_16', 'PdM_BGCN_17', 'PdM_BGCN_18', 'PdM_BGCN_19', 'PdM_BGCN_20', 
                    'PdM_SGCN_0', 'PdM_SGCN_1', 'PdM_SGCN_2', 'PdM_SGCN_3', 'PdM_SGCN_4', 'PdM_SGCN_5', 'PdM_SGCN_6', 'PdM_SGCN_7', 'PdM_SGCN_8', 'PdM_SGCN_9', 'PdM_SGCN_10', 'PdM_SGCN_11', 'PdM_SGCN_12', 'PdM_SGCN_13', 'PdM_SGCN_14', 'PdM_SGCN_15', 'PdM_SGCN_16', 'PdM_SGCN_17', 'PdM_SGCN_18', 'PdM_SGCN_19', 'PdM_SGCN_20', 
                    'PdM_SOGCN_0', 'PdM_SOGCN_1', 'PdM_SOGCN_2', 'PdM_SOGCN_3', 'PdM_SOGCN_4', 'PdM_SOGCN_5', 'PdM_SOGCN_6', 'PdM_SOGCN_7', 'PdM_SOGCN_8', 'PdM_SOGCN_9', 'PdM_SOGCN_10', 'PdM_SOGCN_11', 'PdM_SOGCN_12', 'PdM_SOGCN_13', 'PdM_SOGCN_14', 'PdM_SOGCN_15', 'PdM_SOGCN_16', 'PdM_SOGCN_17', 'PdM_SOGCN_18', 'PdM_SOGCN_19', 'PdM_SOGCN_20', 
                    
                    'PtM_TCN_avg', 'PtM_BCN_avg', 'PtM_SCN_avg', 'PtM_SOCN_avg', 'PtM_TGCN_avg', 'PtM_BGCN_avg', 'PtM_SGCN_avg', 'PtM_SOGCN_avg', 
                    'PtM_TCN_0', 'PtM_TCN_1', 'PtM_TCN_2', 'PtM_TCN_3', 'PtM_TCN_4', 'PtM_TCN_5', 'PtM_TCN_6', 'PtM_TCN_7', 'PtM_TCN_8', 'PtM_TCN_9', 'PtM_TCN_10', 'PtM_TCN_11', 'PtM_TCN_12', 'PtM_TCN_13', 'PtM_TCN_14', 'PtM_TCN_15', 'PtM_TCN_16', 'PtM_TCN_17', 'PtM_TCN_18', 'PtM_TCN_19', 'PtM_TCN_20', 
                    'PtM_BCN_0', 'PtM_BCN_1', 'PtM_BCN_2', 'PtM_BCN_3', 'PtM_BCN_4', 'PtM_BCN_5', 'PtM_BCN_6', 'PtM_BCN_7', 'PtM_BCN_8', 'PtM_BCN_9', 'PtM_BCN_10', 'PtM_BCN_11', 'PtM_BCN_12', 'PtM_BCN_13', 'PtM_BCN_14', 'PtM_BCN_15', 'PtM_BCN_16', 'PtM_BCN_17', 'PtM_BCN_18', 'PtM_BCN_19', 'PtM_BCN_20', 
                    'PtM_SCN_0', 'PtM_SCN_1', 'PtM_SCN_2', 'PtM_SCN_3', 'PtM_SCN_4', 'PtM_SCN_5', 'PtM_SCN_6', 'PtM_SCN_7', 'PtM_SCN_8', 'PtM_SCN_9', 'PtM_SCN_10', 'PtM_SCN_11', 'PtM_SCN_12', 'PtM_SCN_13', 'PtM_SCN_14', 'PtM_SCN_15', 'PtM_SCN_16', 'PtM_SCN_17', 'PtM_SCN_18', 'PtM_SCN_19', 'PtM_SCN_20', 
                    'PtM_SOCN_0', 'PtM_SOCN_1', 'PtM_SOCN_2', 'PtM_SOCN_3', 'PtM_SOCN_4', 'PtM_SOCN_5', 'PtM_SOCN_6', 'PtM_SOCN_7', 'PtM_SOCN_8', 'PtM_SOCN_9', 'PtM_SOCN_10', 'PtM_SOCN_11', 'PtM_SOCN_12', 'PtM_SOCN_13', 'PtM_SOCN_14', 'PtM_SOCN_15', 'PtM_SOCN_16', 'PtM_SOCN_17', 'PtM_SOCN_18', 'PtM_SOCN_19', 'PtM_SOCN_20', 
                    'PtM_TGCN_0', 'PtM_TGCN_1', 'PtM_TGCN_2', 'PtM_TGCN_3', 'PtM_TGCN_4', 'PtM_TGCN_5', 'PtM_TGCN_6', 'PtM_TGCN_7', 'PtM_TGCN_8', 'PtM_TGCN_9', 'PtM_TGCN_10', 'PtM_TGCN_11', 'PtM_TGCN_12', 'PtM_TGCN_13', 'PtM_TGCN_14', 'PtM_TGCN_15', 'PtM_TGCN_16', 'PtM_TGCN_17', 'PtM_TGCN_18', 'PtM_TGCN_19', 'PtM_TGCN_20', 
                    'PtM_BGCN_0', 'PtM_BGCN_1', 'PtM_BGCN_2', 'PtM_BGCN_3', 'PtM_BGCN_4', 'PtM_BGCN_5', 'PtM_BGCN_6', 'PtM_BGCN_7', 'PtM_BGCN_8', 'PtM_BGCN_9', 'PtM_BGCN_10', 'PtM_BGCN_11', 'PtM_BGCN_12', 'PtM_BGCN_13', 'PtM_BGCN_14', 'PtM_BGCN_15', 'PtM_BGCN_16', 'PtM_BGCN_17', 'PtM_BGCN_18', 'PtM_BGCN_19', 'PtM_BGCN_20', 
                    'PtM_SGCN_0', 'PtM_SGCN_1', 'PtM_SGCN_2', 'PtM_SGCN_3', 'PtM_SGCN_4', 'PtM_SGCN_5', 'PtM_SGCN_6', 'PtM_SGCN_7', 'PtM_SGCN_8', 'PtM_SGCN_9', 'PtM_SGCN_10', 'PtM_SGCN_11', 'PtM_SGCN_12', 'PtM_SGCN_13', 'PtM_SGCN_14', 'PtM_SGCN_15', 'PtM_SGCN_16', 'PtM_SGCN_17', 'PtM_SGCN_18', 'PtM_SGCN_19', 'PtM_SGCN_20', 
                    'PtM_SOGCN_0', 'PtM_SOGCN_1', 'PtM_SOGCN_2', 'PtM_SOGCN_3', 'PtM_SOGCN_4', 'PtM_SOGCN_5', 'PtM_SOGCN_6', 'PtM_SOGCN_7', 'PtM_SOGCN_8', 'PtM_SOGCN_9', 'PtM_SOGCN_10', 'PtM_SOGCN_11', 'PtM_SOGCN_12', 'PtM_SOGCN_13', 'PtM_SOGCN_14', 'PtM_SOGCN_15', 'PtM_SOGCN_16', 'PtM_SOGCN_17', 'PtM_SOGCN_18', 'PtM_SOGCN_19', 'PtM_SOGCN_20', 
                    
                    'MM_BL_avg', 'MM_BL_std', 'MM_BL_max', 'MM_BL_min', 'MM_BL_num',
                    'AuAu_BL_avg', 'AuAu_BL_std', 'AuAu_BL_max', 'AuAu_BL_min', 'AuAu_BL_num',
                    'AuPd_BL_avg', 'AuPd_BL_std', 'AuPd_BL_max', 'AuPd_BL_min', 'AuPd_BL_num',
                    'AuPt_BL_avg', 'AuPt_BL_std', 'AuPt_BL_max', 'AuPt_BL_min', 'AuPt_BL_num',
                    'PdPd_BL_avg', 'PdPd_BL_std', 'PdPd_BL_max', 'PdPd_BL_min', 'PdPd_BL_num',
                    'PdPt_BL_avg', 'PdPt_BL_std', 'PdPt_BL_max', 'PdPt_BL_min', 'PdPt_BL_num',
                    'PtPt_BL_avg', 'PtPt_BL_std', 'PtPt_BL_max', 'PtPt_BL_min', 'PtPt_BL_num',
                    
                    'AuAu_frac', 'AuPd_frac', 'AuPt_frac', 'PdPd_frac', 'PdPt_frac', 'PtPt_frac', 'N_bond',
                    
                    'MMM_BA1_avg', 'MMM_BA1_std', 'MMM_BA1_max', 'MMM_BA1_min', 'MMM_BA1_num', 
                    'AuAuAu_BA1_avg', 'AuAuAu_BA1_std', 'AuAuAu_BA1_max', 'AuAuAu_BA1_min', 'AuAuAu_BA1_num', 
                    'AuAuPd_BA1_avg', 'AuAuPd_BA1_std', 'AuAuPd_BA1_max', 'AuAuPd_BA1_min', 'AuAuPd_BA1_num', 
                    'AuAuPt_BA1_avg', 'AuAuPt_BA1_std', 'AuAuPt_BA1_max', 'AuAuPt_BA1_min', 'AuAuPt_BA1_num', 
                    'AuPdAu_BA1_avg', 'AuPdAu_BA1_std', 'AuPdAu_BA1_max', 'AuPdAu_BA1_min', 'AuPdAu_BA1_num', 
                    'AuPdPd_BA1_avg', 'AuPdPd_BA1_std', 'AuPdPd_BA1_max', 'AuPdPd_BA1_min', 'AuPdPd_BA1_num', 
                    'AuPdPt_BA1_avg', 'AuPdPt_BA1_std', 'AuPdPt_BA1_max', 'AuPdPt_BA1_min', 'AuPdPt_BA1_num', 
                    'AuPtAu_BA1_avg', 'AuPtAu_BA1_std', 'AuPtAu_BA1_max', 'AuPtAu_BA1_min', 'AuPtAu_BA1_num', 
                    'AuPtPd_BA1_avg', 'AuPtPd_BA1_std', 'AuPtPd_BA1_max', 'AuPtPd_BA1_min', 'AuPtPd_BA1_num', 
                    'AuPtPt_BA1_avg', 'AuPtPt_BA1_std', 'AuPtPt_BA1_max', 'AuPtPt_BA1_min', 'AuPtPt_BA1_num', 
                    'PdAuAu_BA1_avg', 'PdAuAu_BA1_std', 'PdAuAu_BA1_max', 'PdAuAu_BA1_min', 'PdAuAu_BA1_num', 
                    'PdAuPd_BA1_avg', 'PdAuPd_BA1_std', 'PdAuPd_BA1_max', 'PdAuPd_BA1_min', 'PdAuPd_BA1_num', 
                    'PdAuPt_BA1_avg', 'PdAuPt_BA1_std', 'PdAuPt_BA1_max', 'PdAuPt_BA1_min', 'PdAuPt_BA1_num', 
                    'PdPdAu_BA1_avg', 'PdPdAu_BA1_std', 'PdPdAu_BA1_max', 'PdPdAu_BA1_min', 'PdPdAu_BA1_num', 
                    'PdPdPd_BA1_avg', 'PdPdPd_BA1_std', 'PdPdPd_BA1_max', 'PdPdPd_BA1_min', 'PdPdPd_BA1_num', 
                    'PdPdPt_BA1_avg', 'PdPdPt_BA1_std', 'PdPdPt_BA1_max', 'PdPdPt_BA1_min', 'PdPdPt_BA1_num', 
                    'PdPtAu_BA1_avg', 'PdPtAu_BA1_std', 'PdPtAu_BA1_max', 'PdPtAu_BA1_min', 'PdPtAu_BA1_num', 
                    'PdPtPd_BA1_avg', 'PdPtPd_BA1_std', 'PdPtPd_BA1_max', 'PdPtPd_BA1_min', 'PdPtPd_BA1_num', 
                    'PdPtPt_BA1_avg', 'PdPtPt_BA1_std', 'PdPtPt_BA1_max', 'PdPtPt_BA1_min', 'PdPtPt_BA1_num', 
                    'PtAuAu_BA1_avg', 'PtAuAu_BA1_std', 'PtAuAu_BA1_max', 'PtAuAu_BA1_min', 'PtAuAu_BA1_num', 
                    'PtAuPd_BA1_avg', 'PtAuPd_BA1_std', 'PtAuPd_BA1_max', 'PtAuPd_BA1_min', 'PtAuPd_BA1_num', 
                    'PtAuPt_BA1_avg', 'PtAuPt_BA1_std', 'PtAuPt_BA1_max', 'PtAuPt_BA1_min', 'PtAuPt_BA1_num', 
                    'PtPdAu_BA1_avg', 'PtPdAu_BA1_std', 'PtPdAu_BA1_max', 'PtPdAu_BA1_min', 'PtPdAu_BA1_num', 
                    'PtPdPd_BA1_avg', 'PtPdPd_BA1_std', 'PtPdPd_BA1_max', 'PtPdPd_BA1_min', 'PtPdPd_BA1_num', 
                    'PtPdPt_BA1_avg', 'PtPdPt_BA1_std', 'PtPdPt_BA1_max', 'PtPdPt_BA1_min', 'PtPdPt_BA1_num', 
                    'PtPtAu_BA1_avg', 'PtPtAu_BA1_std', 'PtPtAu_BA1_max', 'PtPtAu_BA1_min', 'PtPtAu_BA1_num', 
                    'PtPtPd_BA1_avg', 'PtPtPd_BA1_std', 'PtPtPd_BA1_max', 'PtPtPd_BA1_min', 'PtPtPd_BA1_num', 
                    'PtPtPt_BA1_avg', 'PtPtPt_BA1_std', 'PtPtPt_BA1_max', 'PtPtPt_BA1_min', 'PtPtPt_BA1_num', 
                    
                    'MMM_BA2_avg', 'MMM_BA2_std', 'MMM_BA2_max', 'MMM_BA2_min', 'MMM_BA2_num', 
                    'AuAuAu_BA2_avg', 'AuAuAu_BA2_std', 'AuAuAu_BA2_max', 'AuAuAu_BA2_min', 'AuAuAu_BA2_num', 
                    'AuAuPd_BA2_avg', 'AuAuPd_BA2_std', 'AuAuPd_BA2_max', 'AuAuPd_BA2_min', 'AuAuPd_BA2_num', 
                    'AuAuPt_BA2_avg', 'AuAuPt_BA2_std', 'AuAuPt_BA2_max', 'AuAuPt_BA2_min', 'AuAuPt_BA2_num', 
                    'AuPdAu_BA2_avg', 'AuPdAu_BA2_std', 'AuPdAu_BA2_max', 'AuPdAu_BA2_min', 'AuPdAu_BA2_num', 
                    'AuPdPd_BA2_avg', 'AuPdPd_BA2_std', 'AuPdPd_BA2_max', 'AuPdPd_BA2_min', 'AuPdPd_BA2_num', 
                    'AuPdPt_BA2_avg', 'AuPdPt_BA2_std', 'AuPdPt_BA2_max', 'AuPdPt_BA2_min', 'AuPdPt_BA2_num', 
                    'AuPtAu_BA2_avg', 'AuPtAu_BA2_std', 'AuPtAu_BA2_max', 'AuPtAu_BA2_min', 'AuPtAu_BA2_num', 
                    'AuPtPd_BA2_avg', 'AuPtPd_BA2_std', 'AuPtPd_BA2_max', 'AuPtPd_BA2_min', 'AuPtPd_BA2_num', 
                    'AuPtPt_BA2_avg', 'AuPtPt_BA2_std', 'AuPtPt_BA2_max', 'AuPtPt_BA2_min', 'AuPtPt_BA2_num', 
                    'PdAuAu_BA2_avg', 'PdAuAu_BA2_std', 'PdAuAu_BA2_max', 'PdAuAu_BA2_min', 'PdAuAu_BA2_num', 
                    'PdAuPd_BA2_avg', 'PdAuPd_BA2_std', 'PdAuPd_BA2_max', 'PdAuPd_BA2_min', 'PdAuPd_BA2_num', 
                    'PdAuPt_BA2_avg', 'PdAuPt_BA2_std', 'PdAuPt_BA2_max', 'PdAuPt_BA2_min', 'PdAuPt_BA2_num', 
                    'PdPdAu_BA2_avg', 'PdPdAu_BA2_std', 'PdPdAu_BA2_max', 'PdPdAu_BA2_min', 'PdPdAu_BA2_num', 
                    'PdPdPd_BA2_avg', 'PdPdPd_BA2_std', 'PdPdPd_BA2_max', 'PdPdPd_BA2_min', 'PdPdPd_BA2_num', 
                    'PdPdPt_BA2_avg', 'PdPdPt_BA2_std', 'PdPdPt_BA2_max', 'PdPdPt_BA2_min', 'PdPdPt_BA2_num', 
                    'PdPtAu_BA2_avg', 'PdPtAu_BA2_std', 'PdPtAu_BA2_max', 'PdPtAu_BA2_min', 'PdPtAu_BA2_num', 
                    'PdPtPd_BA2_avg', 'PdPtPd_BA2_std', 'PdPtPd_BA2_max', 'PdPtPd_BA2_min', 'PdPtPd_BA2_num', 
                    'PdPtPt_BA2_avg', 'PdPtPt_BA2_std', 'PdPtPt_BA2_max', 'PdPtPt_BA2_min', 'PdPtPt_BA2_num', 
                    'PtAuAu_BA2_avg', 'PtAuAu_BA2_std', 'PtAuAu_BA2_max', 'PtAuAu_BA2_min', 'PtAuAu_BA2_num', 
                    'PtAuPd_BA2_avg', 'PtAuPd_BA2_std', 'PtAuPd_BA2_max', 'PtAuPd_BA2_min', 'PtAuPd_BA2_num', 
                    'PtAuPt_BA2_avg', 'PtAuPt_BA2_std', 'PtAuPt_BA2_max', 'PtAuPt_BA2_min', 'PtAuPt_BA2_num', 
                    'PtPdAu_BA2_avg', 'PtPdAu_BA2_std', 'PtPdAu_BA2_max', 'PtPdAu_BA2_min', 'PtPdAu_BA2_num', 
                    'PtPdPd_BA2_avg', 'PtPdPd_BA2_std', 'PtPdPd_BA2_max', 'PtPdPd_BA2_min', 'PtPdPd_BA2_num', 
                    'PtPdPt_BA2_avg', 'PtPdPt_BA2_std', 'PtPdPt_BA2_max', 'PtPdPt_BA2_min', 'PtPdPt_BA2_num', 
                    'PtPtAu_BA2_avg', 'PtPtAu_BA2_std', 'PtPtAu_BA2_max', 'PtPtAu_BA2_min', 'PtPtAu_BA2_num', 
                    'PtPtPd_BA2_avg', 'PtPtPd_BA2_std', 'PtPtPd_BA2_max', 'PtPtPd_BA2_min', 'PtPtPd_BA2_num', 
                    'PtPtPt_BA2_avg', 'PtPtPt_BA2_std', 'PtPtPt_BA2_max', 'PtPtPt_BA2_min', 'PtPtPt_BA2_num', 
                    
                    'MMMM_BTneg_avg', 'MMMM_BTneg_std', 'MMMM_BTneg_max', 'MMMM_BTneg_min', 'MMMM_BTneg_num', 'MMMM_BTpos_avg', 'MMMM_BTpos_std', 'MMMM_BTpos_max', 'MMMM_BTpos_min', 'MMMM_BTpos_num', 
                    'AuAuAuAu_BTneg_avg', 'AuAuAuAu_BTneg_std', 'AuAuAuAu_BTneg_max', 'AuAuAuAu_BTneg_min', 'AuAuAuAu_BTneg_num', 'AuAuAuAu_BTpos_avg', 'AuAuAuAu_BTpos_std', 'AuAuAuAu_BTpos_max', 'AuAuAuAu_BTpos_min', 'AuAuAuAu_BTpos_num', 
                    'AuAuAuPd_BTneg_avg', 'AuAuAuPd_BTneg_std', 'AuAuAuPd_BTneg_max', 'AuAuAuPd_BTneg_min', 'AuAuAuPd_BTneg_num', 'AuAuAuPd_BTpos_avg', 'AuAuAuPd_BTpos_std', 'AuAuAuPd_BTpos_max', 'AuAuAuPd_BTpos_min', 'AuAuAuPd_BTpos_num', 
                    'AuAuAuPt_BTneg_avg', 'AuAuAuPt_BTneg_std', 'AuAuAuPt_BTneg_max', 'AuAuAuPt_BTneg_min', 'AuAuAuPt_BTneg_num', 'AuAuAuPt_BTpos_avg', 'AuAuAuPt_BTpos_std', 'AuAuAuPt_BTpos_max', 'AuAuAuPt_BTpos_min', 'AuAuAuPt_BTpos_num', 
                    'AuAuPdAu_BTneg_avg', 'AuAuPdAu_BTneg_std', 'AuAuPdAu_BTneg_max', 'AuAuPdAu_BTneg_min', 'AuAuPdAu_BTneg_num', 'AuAuPdAu_BTpos_avg', 'AuAuPdAu_BTpos_std', 'AuAuPdAu_BTpos_max', 'AuAuPdAu_BTpos_min', 'AuAuPdAu_BTpos_num',  
                    'AuAuPdPd_BTneg_avg', 'AuAuPdPd_BTneg_std', 'AuAuPdPd_BTneg_max', 'AuAuPdPd_BTneg_min', 'AuAuPdPd_BTneg_num', 'AuAuPdPd_BTpos_avg', 'AuAuPdPd_BTpos_std', 'AuAuPdPd_BTpos_max', 'AuAuPdPd_BTpos_min', 'AuAuPdPd_BTpos_num', 
                    'AuAuPdPt_BTneg_avg', 'AuAuPdPt_BTneg_std', 'AuAuPdPt_BTneg_max', 'AuAuPdPt_BTneg_min', 'AuAuPdPt_BTneg_num', 'AuAuPdPt_BTpos_avg', 'AuAuPdPt_BTpos_std', 'AuAuPdPt_BTpos_max', 'AuAuPdPt_BTpos_min', 'AuAuPdPt_BTpos_num', 
                    'AuAuPtAu_BTneg_avg', 'AuAuPtAu_BTneg_std', 'AuAuPtAu_BTneg_max', 'AuAuPtAu_BTneg_min', 'AuAuPtAu_BTneg_num', 'AuAuPtAu_BTpos_avg', 'AuAuPtAu_BTpos_std', 'AuAuPtAu_BTpos_max', 'AuAuPtAu_BTpos_min', 'AuAuPtAu_BTpos_num',  
                    'AuAuPtPd_BTneg_avg', 'AuAuPtPd_BTneg_std', 'AuAuPtPd_BTneg_max', 'AuAuPtPd_BTneg_min', 'AuAuPtPd_BTneg_num', 'AuAuPtPd_BTpos_avg', 'AuAuPtPd_BTpos_std', 'AuAuPtPd_BTpos_max', 'AuAuPtPd_BTpos_min', 'AuAuPtPd_BTpos_num', 
                    'AuAuPtPt_BTneg_avg', 'AuAuPtPt_BTneg_std', 'AuAuPtPt_BTneg_max', 'AuAuPtPt_BTneg_min', 'AuAuPtPt_BTneg_num', 'AuAuPtPt_BTpos_avg', 'AuAuPtPt_BTpos_std', 'AuAuPtPt_BTpos_max', 'AuAuPtPt_BTpos_min', 'AuAuPtPt_BTpos_num', 
                    'AuPdAuAu_BTneg_avg', 'AuPdAuAu_BTneg_std', 'AuPdAuAu_BTneg_max', 'AuPdAuAu_BTneg_min', 'AuPdAuAu_BTneg_num', 'AuPdAuAu_BTpos_avg', 'AuPdAuAu_BTpos_std', 'AuPdAuAu_BTpos_max', 'AuPdAuAu_BTpos_min', 'AuPdAuAu_BTpos_num',  
                    'AuPdAuPd_BTneg_avg', 'AuPdAuPd_BTneg_std', 'AuPdAuPd_BTneg_max', 'AuPdAuPd_BTneg_min', 'AuPdAuPd_BTneg_num', 'AuPdAuPd_BTpos_avg', 'AuPdAuPd_BTpos_std', 'AuPdAuPd_BTpos_max', 'AuPdAuPd_BTpos_min', 'AuPdAuPd_BTpos_num', 
                    'AuPdAuPt_BTneg_avg', 'AuPdAuPt_BTneg_std', 'AuPdAuPt_BTneg_max', 'AuPdAuPt_BTneg_min', 'AuPdAuPt_BTneg_num', 'AuPdAuPt_BTpos_avg', 'AuPdAuPt_BTpos_std', 'AuPdAuPt_BTpos_max', 'AuPdAuPt_BTpos_min', 'AuPdAuPt_BTpos_num', 
                    'AuPdPdAu_BTneg_avg', 'AuPdPdAu_BTneg_std', 'AuPdPdAu_BTneg_max', 'AuPdPdAu_BTneg_min', 'AuPdPdAu_BTneg_num', 'AuPdPdAu_BTpos_avg', 'AuPdPdAu_BTpos_std', 'AuPdPdAu_BTpos_max', 'AuPdPdAu_BTpos_min', 'AuPdPdAu_BTpos_num',  
                    'AuPdPdPd_BTneg_avg', 'AuPdPdPd_BTneg_std', 'AuPdPdPd_BTneg_max', 'AuPdPdPd_BTneg_min', 'AuPdPdPd_BTneg_num', 'AuPdPdPd_BTpos_avg', 'AuPdPdPd_BTpos_std', 'AuPdPdPd_BTpos_max', 'AuPdPdPd_BTpos_min', 'AuPdPdPd_BTpos_num', 
                    'AuPdPdPt_BTneg_avg', 'AuPdPdPt_BTneg_std', 'AuPdPdPt_BTneg_max', 'AuPdPdPt_BTneg_min', 'AuPdPdPt_BTneg_num', 'AuPdPdPt_BTpos_avg', 'AuPdPdPt_BTpos_std', 'AuPdPdPt_BTpos_max', 'AuPdPdPt_BTpos_min', 'AuPdPdPt_BTpos_num', 
                    'AuPdPtAu_BTneg_avg', 'AuPdPtAu_BTneg_std', 'AuPdPtAu_BTneg_max', 'AuPdPtAu_BTneg_min', 'AuPdPtAu_BTneg_num', 'AuPdPtAu_BTpos_avg', 'AuPdPtAu_BTpos_std', 'AuPdPtAu_BTpos_max', 'AuPdPtAu_BTpos_min', 'AuPdPtAu_BTpos_num',  
                    'AuPdPtPd_BTneg_avg', 'AuPdPtPd_BTneg_std', 'AuPdPtPd_BTneg_max', 'AuPdPtPd_BTneg_min', 'AuPdPtPd_BTneg_num', 'AuPdPtPd_BTpos_avg', 'AuPdPtPd_BTpos_std', 'AuPdPtPd_BTpos_max', 'AuPdPtPd_BTpos_min', 'AuPdPtPd_BTpos_num', 
                    'AuPdPtPt_BTneg_avg', 'AuPdPtPt_BTneg_std', 'AuPdPtPt_BTneg_max', 'AuPdPtPt_BTneg_min', 'AuPdPtPt_BTneg_num', 'AuPdPtPt_BTpos_avg', 'AuPdPtPt_BTpos_std', 'AuPdPtPt_BTpos_max', 'AuPdPtPt_BTpos_min', 'AuPdPtPt_BTpos_num', 
                    'AuPtAuAu_BTneg_avg', 'AuPtAuAu_BTneg_std', 'AuPtAuAu_BTneg_max', 'AuPtAuAu_BTneg_min', 'AuPtAuAu_BTneg_num', 'AuPtAuAu_BTpos_avg', 'AuPtAuAu_BTpos_std', 'AuPtAuAu_BTpos_max', 'AuPtAuAu_BTpos_min', 'AuPtAuAu_BTpos_num',  
                    'AuPtAuPd_BTneg_avg', 'AuPtAuPd_BTneg_std', 'AuPtAuPd_BTneg_max', 'AuPtAuPd_BTneg_min', 'AuPtAuPd_BTneg_num', 'AuPtAuPd_BTpos_avg', 'AuPtAuPd_BTpos_std', 'AuPtAuPd_BTpos_max', 'AuPtAuPd_BTpos_min', 'AuPtAuPd_BTpos_num', 
                    'AuPtAuPt_BTneg_avg', 'AuPtAuPt_BTneg_std', 'AuPtAuPt_BTneg_max', 'AuPtAuPt_BTneg_min', 'AuPtAuPt_BTneg_num', 'AuPtAuPt_BTpos_avg', 'AuPtAuPt_BTpos_std', 'AuPtAuPt_BTpos_max', 'AuPtAuPt_BTpos_min', 'AuPtAuPt_BTpos_num', 
                    'AuPtPdAu_BTneg_avg', 'AuPtPdAu_BTneg_std', 'AuPtPdAu_BTneg_max', 'AuPtPdAu_BTneg_min', 'AuPtPdAu_BTneg_num', 'AuPtPdAu_BTpos_avg', 'AuPtPdAu_BTpos_std', 'AuPtPdAu_BTpos_max', 'AuPtPdAu_BTpos_min', 'AuPtPdAu_BTpos_num',  
                    'AuPtPdPd_BTneg_avg', 'AuPtPdPd_BTneg_std', 'AuPtPdPd_BTneg_max', 'AuPtPdPd_BTneg_min', 'AuPtPdPd_BTneg_num', 'AuPtPdPd_BTpos_avg', 'AuPtPdPd_BTpos_std', 'AuPtPdPd_BTpos_max', 'AuPtPdPd_BTpos_min', 'AuPtPdPd_BTpos_num', 
                    'AuPtPdPt_BTneg_avg', 'AuPtPdPt_BTneg_std', 'AuPtPdPt_BTneg_max', 'AuPtPdPt_BTneg_min', 'AuPtPdPt_BTneg_num', 'AuPtPdPt_BTpos_avg', 'AuPtPdPt_BTpos_std', 'AuPtPdPt_BTpos_max', 'AuPtPdPt_BTpos_min', 'AuPtPdPt_BTpos_num', 
                    'AuPtPtAu_BTneg_avg', 'AuPtPtAu_BTneg_std', 'AuPtPtAu_BTneg_max', 'AuPtPtAu_BTneg_min', 'AuPtPtAu_BTneg_num', 'AuPtPtAu_BTpos_avg', 'AuPtPtAu_BTpos_std', 'AuPtPtAu_BTpos_max', 'AuPtPtAu_BTpos_min', 'AuPtPtAu_BTpos_num', 
                    'AuPtPtPd_BTneg_avg', 'AuPtPtPd_BTneg_std', 'AuPtPtPd_BTneg_max', 'AuPtPtPd_BTneg_min', 'AuPtPtPd_BTneg_num', 'AuPtPtPd_BTpos_avg', 'AuPtPtPd_BTpos_std', 'AuPtPtPd_BTpos_max', 'AuPtPtPd_BTpos_min', 'AuPtPtPd_BTpos_num', 
                    'AuPtPtPt_BTneg_avg', 'AuPtPtPt_BTneg_std', 'AuPtPtPt_BTneg_max', 'AuPtPtPt_BTneg_min', 'AuPtPtPt_BTneg_num', 'AuPtPtPt_BTpos_avg', 'AuPtPtPt_BTpos_std', 'AuPtPtPt_BTpos_max', 'AuPtPtPt_BTpos_min', 'AuPtPtPt_BTpos_num', 

                    'PdAuAuAu_BTneg_avg', 'PdAuAuAu_BTneg_std', 'PdAuAuAu_BTneg_max', 'PdAuAuAu_BTneg_min', 'PdAuAuAu_BTneg_num', 'PdAuAuAu_BTpos_avg', 'PdAuAuAu_BTpos_std', 'PdAuAuAu_BTpos_max', 'PdAuAuAu_BTpos_min', 'PdAuAuAu_BTpos_num', 
                    'PdAuAuPd_BTneg_avg', 'PdAuAuPd_BTneg_std', 'PdAuAuPd_BTneg_max', 'PdAuAuPd_BTneg_min', 'PdAuAuPd_BTneg_num', 'PdAuAuPd_BTpos_avg', 'PdAuAuPd_BTpos_std', 'PdAuAuPd_BTpos_max', 'PdAuAuPd_BTpos_min', 'PdAuAuPd_BTpos_num', 
                    'PdAuAuPt_BTneg_avg', 'PdAuAuPt_BTneg_std', 'PdAuAuPt_BTneg_max', 'PdAuAuPt_BTneg_min', 'PdAuAuPt_BTneg_num', 'PdAuAuPt_BTpos_avg', 'PdAuAuPt_BTpos_std', 'PdAuAuPt_BTpos_max', 'PdAuAuPt_BTpos_min', 'PdAuAuPt_BTpos_num', 
                    'PdAuPdAu_BTneg_avg', 'PdAuPdAu_BTneg_std', 'PdAuPdAu_BTneg_max', 'PdAuPdAu_BTneg_min', 'PdAuPdAu_BTneg_num', 'PdAuPdAu_BTpos_avg', 'PdAuPdAu_BTpos_std', 'PdAuPdAu_BTpos_max', 'PdAuPdAu_BTpos_min', 'PdAuPdAu_BTpos_num',  
                    'PdAuPdPd_BTneg_avg', 'PdAuPdPd_BTneg_std', 'PdAuPdPd_BTneg_max', 'PdAuPdPd_BTneg_min', 'PdAuPdPd_BTneg_num', 'PdAuPdPd_BTpos_avg', 'PdAuPdPd_BTpos_std', 'PdAuPdPd_BTpos_max', 'PdAuPdPd_BTpos_min', 'PdAuPdPd_BTpos_num', 
                    'PdAuPdPt_BTneg_avg', 'PdAuPdPt_BTneg_std', 'PdAuPdPt_BTneg_max', 'PdAuPdPt_BTneg_min', 'PdAuPdPt_BTneg_num', 'PdAuPdPt_BTpos_avg', 'PdAuPdPt_BTpos_std', 'PdAuPdPt_BTpos_max', 'PdAuPdPt_BTpos_min', 'PdAuPdPt_BTpos_num', 
                    'PdAuPtAu_BTneg_avg', 'PdAuPtAu_BTneg_std', 'PdAuPtAu_BTneg_max', 'PdAuPtAu_BTneg_min', 'PdAuPtAu_BTneg_num', 'PdAuPtAu_BTpos_avg', 'PdAuPtAu_BTpos_std', 'PdAuPtAu_BTpos_max', 'PdAuPtAu_BTpos_min', 'PdAuPtAu_BTpos_num',  
                    'PdAuPtPd_BTneg_avg', 'PdAuPtPd_BTneg_std', 'PdAuPtPd_BTneg_max', 'PdAuPtPd_BTneg_min', 'PdAuPtPd_BTneg_num', 'PdAuPtPd_BTpos_avg', 'PdAuPtPd_BTpos_std', 'PdAuPtPd_BTpos_max', 'PdAuPtPd_BTpos_min', 'PdAuPtPd_BTpos_num', 
                    'PdAuPtPt_BTneg_avg', 'PdAuPtPt_BTneg_std', 'PdAuPtPt_BTneg_max', 'PdAuPtPt_BTneg_min', 'PdAuPtPt_BTneg_num', 'PdAuPtPt_BTpos_avg', 'PdAuPtPt_BTpos_std', 'PdAuPtPt_BTpos_max', 'PdAuPtPt_BTpos_min', 'PdAuPtPt_BTpos_num', 
                    'PdPdAuAu_BTneg_avg', 'PdPdAuAu_BTneg_std', 'PdPdAuAu_BTneg_max', 'PdPdAuAu_BTneg_min', 'PdPdAuAu_BTneg_num', 'PdPdAuAu_BTpos_avg', 'PdPdAuAu_BTpos_std', 'PdPdAuAu_BTpos_max', 'PdPdAuAu_BTpos_min', 'PdPdAuAu_BTpos_num',  
                    'PdPdAuPd_BTneg_avg', 'PdPdAuPd_BTneg_std', 'PdPdAuPd_BTneg_max', 'PdPdAuPd_BTneg_min', 'PdPdAuPd_BTneg_num', 'PdPdAuPd_BTpos_avg', 'PdPdAuPd_BTpos_std', 'PdPdAuPd_BTpos_max', 'PdPdAuPd_BTpos_min', 'PdPdAuPd_BTpos_num', 
                    'PdPdAuPt_BTneg_avg', 'PdPdAuPt_BTneg_std', 'PdPdAuPt_BTneg_max', 'PdPdAuPt_BTneg_min', 'PdPdAuPt_BTneg_num', 'PdPdAuPt_BTpos_avg', 'PdPdAuPt_BTpos_std', 'PdPdAuPt_BTpos_max', 'PdPdAuPt_BTpos_min', 'PdPdAuPt_BTpos_num', 
                    'PdPdPdAu_BTneg_avg', 'PdPdPdAu_BTneg_std', 'PdPdPdAu_BTneg_max', 'PdPdPdAu_BTneg_min', 'PdPdPdAu_BTneg_num', 'PdPdPdAu_BTpos_avg', 'PdPdPdAu_BTpos_std', 'PdPdPdAu_BTpos_max', 'PdPdPdAu_BTpos_min', 'PdPdPdAu_BTpos_num',  
                    'PdPdPdPd_BTneg_avg', 'PdPdPdPd_BTneg_std', 'PdPdPdPd_BTneg_max', 'PdPdPdPd_BTneg_min', 'PdPdPdPd_BTneg_num', 'PdPdPdPd_BTpos_avg', 'PdPdPdPd_BTpos_std', 'PdPdPdPd_BTpos_max', 'PdPdPdPd_BTpos_min', 'PdPdPdPd_BTpos_num', 
                    'PdPdPdPt_BTneg_avg', 'PdPdPdPt_BTneg_std', 'PdPdPdPt_BTneg_max', 'PdPdPdPt_BTneg_min', 'PdPdPdPt_BTneg_num', 'PdPdPdPt_BTpos_avg', 'PdPdPdPt_BTpos_std', 'PdPdPdPt_BTpos_max', 'PdPdPdPt_BTpos_min', 'PdPdPdPt_BTpos_num', 
                    'PdPdPtAu_BTneg_avg', 'PdPdPtAu_BTneg_std', 'PdPdPtAu_BTneg_max', 'PdPdPtAu_BTneg_min', 'PdPdPtAu_BTneg_num', 'PdPdPtAu_BTpos_avg', 'PdPdPtAu_BTpos_std', 'PdPdPtAu_BTpos_max', 'PdPdPtAu_BTpos_min', 'PdPdPtAu_BTpos_num',  
                    'PdPdPtPd_BTneg_avg', 'PdPdPtPd_BTneg_std', 'PdPdPtPd_BTneg_max', 'PdPdPtPd_BTneg_min', 'PdPdPtPd_BTneg_num', 'PdPdPtPd_BTpos_avg', 'PdPdPtPd_BTpos_std', 'PdPdPtPd_BTpos_max', 'PdPdPtPd_BTpos_min', 'PdPdPtPd_BTpos_num', 
                    'PdPdPtPt_BTneg_avg', 'PdPdPtPt_BTneg_std', 'PdPdPtPt_BTneg_max', 'PdPdPtPt_BTneg_min', 'PdPdPtPt_BTneg_num', 'PdPdPtPt_BTpos_avg', 'PdPdPtPt_BTpos_std', 'PdPdPtPt_BTpos_max', 'PdPdPtPt_BTpos_min', 'PdPdPtPt_BTpos_num', 
                    'PdPtAuAu_BTneg_avg', 'PdPtAuAu_BTneg_std', 'PdPtAuAu_BTneg_max', 'PdPtAuAu_BTneg_min', 'PdPtAuAu_BTneg_num', 'PdPtAuAu_BTpos_avg', 'PdPtAuAu_BTpos_std', 'PdPtAuAu_BTpos_max', 'PdPtAuAu_BTpos_min', 'PdPtAuAu_BTpos_num',  
                    'PdPtAuPd_BTneg_avg', 'PdPtAuPd_BTneg_std', 'PdPtAuPd_BTneg_max', 'PdPtAuPd_BTneg_min', 'PdPtAuPd_BTneg_num', 'PdPtAuPd_BTpos_avg', 'PdPtAuPd_BTpos_std', 'PdPtAuPd_BTpos_max', 'PdPtAuPd_BTpos_min', 'PdPtAuPd_BTpos_num', 
                    'PdPtAuPt_BTneg_avg', 'PdPtAuPt_BTneg_std', 'PdPtAuPt_BTneg_max', 'PdPtAuPt_BTneg_min', 'PdPtAuPt_BTneg_num', 'PdPtAuPt_BTpos_avg', 'PdPtAuPt_BTpos_std', 'PdPtAuPt_BTpos_max', 'PdPtAuPt_BTpos_min', 'PdPtAuPt_BTpos_num', 
                    'PdPtPdAu_BTneg_avg', 'PdPtPdAu_BTneg_std', 'PdPtPdAu_BTneg_max', 'PdPtPdAu_BTneg_min', 'PdPtPdAu_BTneg_num', 'PdPtPdAu_BTpos_avg', 'PdPtPdAu_BTpos_std', 'PdPtPdAu_BTpos_max', 'PdPtPdAu_BTpos_min', 'PdPtPdAu_BTpos_num',  
                    'PdPtPdPd_BTneg_avg', 'PdPtPdPd_BTneg_std', 'PdPtPdPd_BTneg_max', 'PdPtPdPd_BTneg_min', 'PdPtPdPd_BTneg_num', 'PdPtPdPd_BTpos_avg', 'PdPtPdPd_BTpos_std', 'PdPtPdPd_BTpos_max', 'PdPtPdPd_BTpos_min', 'PdPtPdPd_BTpos_num', 
                    'PdPtPdPt_BTneg_avg', 'PdPtPdPt_BTneg_std', 'PdPtPdPt_BTneg_max', 'PdPtPdPt_BTneg_min', 'PdPtPdPt_BTneg_num', 'PdPtPdPt_BTpos_avg', 'PdPtPdPt_BTpos_std', 'PdPtPdPt_BTpos_max', 'PdPtPdPt_BTpos_min', 'PdPtPdPt_BTpos_num', 
                    'PdPtPtAu_BTneg_avg', 'PdPtPtAu_BTneg_std', 'PdPtPtAu_BTneg_max', 'PdPtPtAu_BTneg_min', 'PdPtPtAu_BTneg_num', 'PdPtPtAu_BTpos_avg', 'PdPtPtAu_BTpos_std', 'PdPtPtAu_BTpos_max', 'PdPtPtAu_BTpos_min', 'PdPtPtAu_BTpos_num', 
                    'PdPtPtPd_BTneg_avg', 'PdPtPtPd_BTneg_std', 'PdPtPtPd_BTneg_max', 'PdPtPtPd_BTneg_min', 'PdPtPtPd_BTneg_num', 'PdPtPtPd_BTpos_avg', 'PdPtPtPd_BTpos_std', 'PdPtPtPd_BTpos_max', 'PdPtPtPd_BTpos_min', 'PdPtPtPd_BTpos_num', 
                    'PdPtPtPt_BTneg_avg', 'PdPtPtPt_BTneg_std', 'PdPtPtPt_BTneg_max', 'PdPtPtPt_BTneg_min', 'PdPtPtPt_BTneg_num', 'PdPtPtPt_BTpos_avg', 'PdPtPtPt_BTpos_std', 'PdPtPtPt_BTpos_max', 'PdPtPtPt_BTpos_min', 'PdPtPtPt_BTpos_num', 

                    'PtAuAuAu_BTneg_avg', 'PtAuAuAu_BTneg_std', 'PtAuAuAu_BTneg_max', 'PtAuAuAu_BTneg_min', 'PtAuAuAu_BTneg_num', 'PtAuAuAu_BTpos_avg', 'PtAuAuAu_BTpos_std', 'PtAuAuAu_BTpos_max', 'PtAuAuAu_BTpos_min', 'PtAuAuAu_BTpos_num', 
                    'PtAuAuPd_BTneg_avg', 'PtAuAuPd_BTneg_std', 'PtAuAuPd_BTneg_max', 'PtAuAuPd_BTneg_min', 'PtAuAuPd_BTneg_num', 'PtAuAuPd_BTpos_avg', 'PtAuAuPd_BTpos_std', 'PtAuAuPd_BTpos_max', 'PtAuAuPd_BTpos_min', 'PtAuAuPd_BTpos_num', 
                    'PtAuAuPt_BTneg_avg', 'PtAuAuPt_BTneg_std', 'PtAuAuPt_BTneg_max', 'PtAuAuPt_BTneg_min', 'PtAuAuPt_BTneg_num', 'PtAuAuPt_BTpos_avg', 'PtAuAuPt_BTpos_std', 'PtAuAuPt_BTpos_max', 'PtAuAuPt_BTpos_min', 'PtAuAuPt_BTpos_num', 
                    'PtAuPdAu_BTneg_avg', 'PtAuPdAu_BTneg_std', 'PtAuPdAu_BTneg_max', 'PtAuPdAu_BTneg_min', 'PtAuPdAu_BTneg_num', 'PtAuPdAu_BTpos_avg', 'PtAuPdAu_BTpos_std', 'PtAuPdAu_BTpos_max', 'PtAuPdAu_BTpos_min', 'PtAuPdAu_BTpos_num',  
                    'PtAuPdPd_BTneg_avg', 'PtAuPdPd_BTneg_std', 'PtAuPdPd_BTneg_max', 'PtAuPdPd_BTneg_min', 'PtAuPdPd_BTneg_num', 'PtAuPdPd_BTpos_avg', 'PtAuPdPd_BTpos_std', 'PtAuPdPd_BTpos_max', 'PtAuPdPd_BTpos_min', 'PtAuPdPd_BTpos_num', 
                    'PtAuPdPt_BTneg_avg', 'PtAuPdPt_BTneg_std', 'PtAuPdPt_BTneg_max', 'PtAuPdPt_BTneg_min', 'PtAuPdPt_BTneg_num', 'PtAuPdPt_BTpos_avg', 'PtAuPdPt_BTpos_std', 'PtAuPdPt_BTpos_max', 'PtAuPdPt_BTpos_min', 'PtAuPdPt_BTpos_num', 
                    'PtAuPtAu_BTneg_avg', 'PtAuPtAu_BTneg_std', 'PtAuPtAu_BTneg_max', 'PtAuPtAu_BTneg_min', 'PtAuPtAu_BTneg_num', 'PtAuPtAu_BTpos_avg', 'PtAuPtAu_BTpos_std', 'PtAuPtAu_BTpos_max', 'PtAuPtAu_BTpos_min', 'PtAuPtAu_BTpos_num',  
                    'PtAuPtPd_BTneg_avg', 'PtAuPtPd_BTneg_std', 'PtAuPtPd_BTneg_max', 'PtAuPtPd_BTneg_min', 'PtAuPtPd_BTneg_num', 'PtAuPtPd_BTpos_avg', 'PtAuPtPd_BTpos_std', 'PtAuPtPd_BTpos_max', 'PtAuPtPd_BTpos_min', 'PtAuPtPd_BTpos_num', 
                    'PtAuPtPt_BTneg_avg', 'PtAuPtPt_BTneg_std', 'PtAuPtPt_BTneg_max', 'PtAuPtPt_BTneg_min', 'PtAuPtPt_BTneg_num', 'PtAuPtPt_BTpos_avg', 'PtAuPtPt_BTpos_std', 'PtAuPtPt_BTpos_max', 'PtAuPtPt_BTpos_min', 'PtAuPtPt_BTpos_num', 
                    'PtPdAuAu_BTneg_avg', 'PtPdAuAu_BTneg_std', 'PtPdAuAu_BTneg_max', 'PtPdAuAu_BTneg_min', 'PtPdAuAu_BTneg_num', 'PtPdAuAu_BTpos_avg', 'PtPdAuAu_BTpos_std', 'PtPdAuAu_BTpos_max', 'PtPdAuAu_BTpos_min', 'PtPdAuAu_BTpos_num',  
                    'PtPdAuPd_BTneg_avg', 'PtPdAuPd_BTneg_std', 'PtPdAuPd_BTneg_max', 'PtPdAuPd_BTneg_min', 'PtPdAuPd_BTneg_num', 'PtPdAuPd_BTpos_avg', 'PtPdAuPd_BTpos_std', 'PtPdAuPd_BTpos_max', 'PtPdAuPd_BTpos_min', 'PtPdAuPd_BTpos_num', 
                    'PtPdAuPt_BTneg_avg', 'PtPdAuPt_BTneg_std', 'PtPdAuPt_BTneg_max', 'PtPdAuPt_BTneg_min', 'PtPdAuPt_BTneg_num', 'PtPdAuPt_BTpos_avg', 'PtPdAuPt_BTpos_std', 'PtPdAuPt_BTpos_max', 'PtPdAuPt_BTpos_min', 'PtPdAuPt_BTpos_num', 
                    'PtPdPdAu_BTneg_avg', 'PtPdPdAu_BTneg_std', 'PtPdPdAu_BTneg_max', 'PtPdPdAu_BTneg_min', 'PtPdPdAu_BTneg_num', 'PtPdPdAu_BTpos_avg', 'PtPdPdAu_BTpos_std', 'PtPdPdAu_BTpos_max', 'PtPdPdAu_BTpos_min', 'PtPdPdAu_BTpos_num',  
                    'PtPdPdPd_BTneg_avg', 'PtPdPdPd_BTneg_std', 'PtPdPdPd_BTneg_max', 'PtPdPdPd_BTneg_min', 'PtPdPdPd_BTneg_num', 'PtPdPdPd_BTpos_avg', 'PtPdPdPd_BTpos_std', 'PtPdPdPd_BTpos_max', 'PtPdPdPd_BTpos_min', 'PtPdPdPd_BTpos_num', 
                    'PtPdPdPt_BTneg_avg', 'PtPdPdPt_BTneg_std', 'PtPdPdPt_BTneg_max', 'PtPdPdPt_BTneg_min', 'PtPdPdPt_BTneg_num', 'PtPdPdPt_BTpos_avg', 'PtPdPdPt_BTpos_std', 'PtPdPdPt_BTpos_max', 'PtPdPdPt_BTpos_min', 'PtPdPdPt_BTpos_num', 
                    'PtPdPtAu_BTneg_avg', 'PtPdPtAu_BTneg_std', 'PtPdPtAu_BTneg_max', 'PtPdPtAu_BTneg_min', 'PtPdPtAu_BTneg_num', 'PtPdPtAu_BTpos_avg', 'PtPdPtAu_BTpos_std', 'PtPdPtAu_BTpos_max', 'PtPdPtAu_BTpos_min', 'PtPdPtAu_BTpos_num',  
                    'PtPdPtPd_BTneg_avg', 'PtPdPtPd_BTneg_std', 'PtPdPtPd_BTneg_max', 'PtPdPtPd_BTneg_min', 'PtPdPtPd_BTneg_num', 'PtPdPtPd_BTpos_avg', 'PtPdPtPd_BTpos_std', 'PtPdPtPd_BTpos_max', 'PtPdPtPd_BTpos_min', 'PtPdPtPd_BTpos_num', 
                    'PtPdPtPt_BTneg_avg', 'PtPdPtPt_BTneg_std', 'PtPdPtPt_BTneg_max', 'PtPdPtPt_BTneg_min', 'PtPdPtPt_BTneg_num', 'PtPdPtPt_BTpos_avg', 'PtPdPtPt_BTpos_std', 'PtPdPtPt_BTpos_max', 'PtPdPtPt_BTpos_min', 'PtPdPtPt_BTpos_num', 
                    'PtPtAuAu_BTneg_avg', 'PtPtAuAu_BTneg_std', 'PtPtAuAu_BTneg_max', 'PtPtAuAu_BTneg_min', 'PtPtAuAu_BTneg_num', 'PtPtAuAu_BTpos_avg', 'PtPtAuAu_BTpos_std', 'PtPtAuAu_BTpos_max', 'PtPtAuAu_BTpos_min', 'PtPtAuAu_BTpos_num',  
                    'PtPtAuPd_BTneg_avg', 'PtPtAuPd_BTneg_std', 'PtPtAuPd_BTneg_max', 'PtPtAuPd_BTneg_min', 'PtPtAuPd_BTneg_num', 'PtPtAuPd_BTpos_avg', 'PtPtAuPd_BTpos_std', 'PtPtAuPd_BTpos_max', 'PtPtAuPd_BTpos_min', 'PtPtAuPd_BTpos_num', 
                    'PtPtAuPt_BTneg_avg', 'PtPtAuPt_BTneg_std', 'PtPtAuPt_BTneg_max', 'PtPtAuPt_BTneg_min', 'PtPtAuPt_BTneg_num', 'PtPtAuPt_BTpos_avg', 'PtPtAuPt_BTpos_std', 'PtPtAuPt_BTpos_max', 'PtPtAuPt_BTpos_min', 'PtPtAuPt_BTpos_num', 
                    'PtPtPdAu_BTneg_avg', 'PtPtPdAu_BTneg_std', 'PtPtPdAu_BTneg_max', 'PtPtPdAu_BTneg_min', 'PtPtPdAu_BTneg_num', 'PtPtPdAu_BTpos_avg', 'PtPtPdAu_BTpos_std', 'PtPtPdAu_BTpos_max', 'PtPtPdAu_BTpos_min', 'PtPtPdAu_BTpos_num',  
                    'PtPtPdPd_BTneg_avg', 'PtPtPdPd_BTneg_std', 'PtPtPdPd_BTneg_max', 'PtPtPdPd_BTneg_min', 'PtPtPdPd_BTneg_num', 'PtPtPdPd_BTpos_avg', 'PtPtPdPd_BTpos_std', 'PtPtPdPd_BTpos_max', 'PtPtPdPd_BTpos_min', 'PtPtPdPd_BTpos_num', 
                    'PtPtPdPt_BTneg_avg', 'PtPtPdPt_BTneg_std', 'PtPtPdPt_BTneg_max', 'PtPtPdPt_BTneg_min', 'PtPtPdPt_BTneg_num', 'PtPtPdPt_BTpos_avg', 'PtPtPdPt_BTpos_std', 'PtPtPdPt_BTpos_max', 'PtPtPdPt_BTpos_min', 'PtPtPdPt_BTpos_num', 
                    'PtPtPtAu_BTneg_avg', 'PtPtPtAu_BTneg_std', 'PtPtPtAu_BTneg_max', 'PtPtPtAu_BTneg_min', 'PtPtPtAu_BTneg_num', 'PtPtPtAu_BTpos_avg', 'PtPtPtAu_BTpos_std', 'PtPtPtAu_BTpos_max', 'PtPtPtAu_BTpos_min', 'PtPtPtAu_BTpos_num', 
                    'PtPtPtPd_BTneg_avg', 'PtPtPtPd_BTneg_std', 'PtPtPtPd_BTneg_max', 'PtPtPtPd_BTneg_min', 'PtPtPtPd_BTneg_num', 'PtPtPtPd_BTpos_avg', 'PtPtPtPd_BTpos_std', 'PtPtPtPd_BTpos_max', 'PtPtPtPd_BTpos_min', 'PtPtPtPd_BTpos_num', 
                    'PtPtPtPt_BTneg_avg', 'PtPtPtPt_BTneg_std', 'PtPtPtPt_BTneg_max', 'PtPtPtPt_BTneg_min', 'PtPtPtPt_BTneg_num', 'PtPtPtPt_BTpos_avg', 'PtPtPtPt_BTpos_std', 'PtPtPtPt_BTpos_max', 'PtPtPtPt_BTpos_min', 'PtPtPtPt_BTpos_num', 

                    'q6q6_T_avg', 'q6q6_B_avg', 'q6q6_S_avg', 
                    'q6q6_T_0', 'q6q6_T_1', 'q6q6_T_2', 'q6q6_T_3', 'q6q6_T_4', 'q6q6_T_5', 'q6q6_T_6', 'q6q6_T_7', 'q6q6_T_8', 'q6q6_T_9', 'q6q6_T_10', 'q6q6_T_11', 'q6q6_T_12', 'q6q6_T_13', 'q6q6_T_14', 'q6q6_T_15', 'q6q6_T_16', 'q6q6_T_17', 'q6q6_T_18', 'q6q6_T_19', 'q6q6_T_20', 'q6q6_T_20+', 
                    'q6q6_B_0', 'q6q6_B_1', 'q6q6_B_2', 'q6q6_B_3', 'q6q6_B_4', 'q6q6_B_5', 'q6q6_B_6', 'q6q6_B_7', 'q6q6_B_8', 'q6q6_B_9', 'q6q6_B_10', 'q6q6_B_11', 'q6q6_B_12', 'q6q6_B_13', 'q6q6_B_14', 'q6q6_B_15', 'q6q6_B_16', 'q6q6_B_17', 'q6q6_B_18', 'q6q6_B_19', 'q6q6_B_20', 'q6q6_B_20+', 
                    'q6q6_S_0', 'q6q6_S_1', 'q6q6_S_2', 'q6q6_S_3', 'q6q6_S_4', 'q6q6_S_5', 'q6q6_S_6', 'q6q6_S_7', 'q6q6_S_8', 'q6q6_S_9', 'q6q6_S_10', 'q6q6_S_11', 'q6q6_S_12', 'q6q6_S_13', 'q6q6_S_14', 'q6q6_S_15', 'q6q6_S_16', 'q6q6_S_17', 'q6q6_S_18', 'q6q6_S_19', 'q6q6_S_20', 'q6q6_S_20+', 
                   
                    'FCC', 'HCP', 'ICOS', 'DECA', 
              
                    'Surf_defects_Au', 'Surf_defects_Au_bulk_pack_conc', 'Surf_defects_Au_bulk_pack_ratio', 'Surf_defects_Au_sphere_conc', 'Surf_defects_Au_sphere_ratio', 
                    'Surf_defects_Pd', 'Surf_defects_Pd_bulk_pack_conc', 'Surf_defects_Pd_bulk_pack_ratio', 'Surf_defects_Pd_sphere_conc', 'Surf_defects_Pd_sphere_ratio', 
                    'Surf_defects_Pt', 'Surf_defects_Pt_bulk_pack_conc', 'Surf_defects_Pt_bulk_pack_ratio', 'Surf_defects_Pt_sphere_conc', 'Surf_defects_Pt_sphere_ratio', 
                    'Surf_defects', 'Surf_defects_bulk_pack_conc', 'Surf_defects_bulk_pack_ratio', 'Surf_defects_sphere_conc', 'Surf_defects_sphere_ratio', 
                    'Surf_micros_Au', 'Surf_micros_Au_bulk_pack_conc', 'Surf_micros_Au_bulk_pack_ratio', 'Surf_micros_Au_sphere_conc', 'Surf_micros_Au_sphere_ratio', 
                    'Surf_micros_Pd', 'Surf_micros_Pd_bulk_pack_conc', 'Surf_micros_Pd_bulk_pack_ratio', 'Surf_micros_Pd_sphere_conc', 'Surf_micros_Pd_sphere_ratio', 
                    'Surf_micros_Pt', 'Surf_micros_Pt_bulk_pack_conc', 'Surf_micros_Pt_bulk_pack_ratio', 'Surf_micros_Pt_sphere_conc', 'Surf_micros_Pt_sphere_ratio', 
                    'Surf_micros', 'Surf_micros_bulk_pack_conc', 'Surf_micros_bulk_pack_ratio', 'Surf_micros_sphere_conc', 'Surf_micros_sphere_ratio', 
                    'Surf_facets_Au', 'Surf_facets_Au_bulk_pack_conc', 'Surf_facets_Au_bulk_pack_ratio', 'Surf_facets_Au_sphere_conc', 'Surf_facets_Au_sphere_ratio', 
                    'Surf_facets_Pd', 'Surf_facets_Pd_bulk_pack_conc', 'Surf_facets_Pd_bulk_pack_ratio', 'Surf_facets_Pd_sphere_conc', 'Surf_facets_Pd_sphere_ratio',
                    'Surf_facets_Pt', 'Surf_facets_Pt_bulk_pack_conc', 'Surf_facets_Pt_bulk_pack_ratio', 'Surf_facets_Pt_sphere_conc', 'Surf_facets_Pt_sphere_ratio', 
                    'Surf_facets', 'Surf_facets_bulk_pack_conc', 'Surf_facets_bulk_pack_ratio', 'Surf_facets_sphere_conc', 'Surf_facets_sphere_ratio']
# Features to be added
ADD_FEAT_LIST = ['Vol_bulk_pack', 'Vol_sphere', 
                 'Curve_1-10', 'Curve_11-20', 'Curve_21-30', 'Curve_31-40', 'Curve_41-50', 'Curve_51-60', 'Curve_61-70', 'Curve_71-80', 'Curve_81-90', 'Curve_91-100', 'Curve_101-110', 'Curve_111-120', 'Curve_121-130', 'Curve_131-140', 'Curve_141-150', 'Curve_151-160', 'Curve_161-170', 'Curve_171-180',
                 'Surf_defects_Au', 'Surf_defects_Au_bulk_pack_conc', 'Surf_defects_Au_bulk_pack_ratio', 'Surf_defects_Au_sphere_conc', 'Surf_defects_Au_sphere_ratio', 
                 'Surf_defects_Pd', 'Surf_defects_Pd_bulk_pack_conc', 'Surf_defects_Pd_bulk_pack_ratio', 'Surf_defects_Pd_sphere_conc', 'Surf_defects_Pd_sphere_ratio', 
                 'Surf_defects_Pt', 'Surf_defects_Pt_bulk_pack_conc', 'Surf_defects_Pt_bulk_pack_ratio', 'Surf_defects_Pt_sphere_conc', 'Surf_defects_Pt_sphere_ratio', 
                 'Surf_defects', 'Surf_defects_bulk_pack_conc', 'Surf_defects_bulk_pack_ratio', 'Surf_defects_sphere_conc', 'Surf_defects_sphere_ratio', 
                 'Surf_micros_Au', 'Surf_micros_Au_bulk_pack_conc', 'Surf_micros_Au_bulk_pack_ratio', 'Surf_micros_Au_sphere_conc', 'Surf_micros_Au_sphere_ratio', 
                 'Surf_micros_Pd', 'Surf_micros_Pd_bulk_pack_conc', 'Surf_micros_Pd_bulk_pack_ratio', 'Surf_micros_Pd_sphere_conc', 'Surf_micros_Pd_sphere_ratio', 
                 'Surf_micros_Pt', 'Surf_micros_Pt_bulk_pack_conc', 'Surf_micros_Pt_bulk_pack_ratio', 'Surf_micros_Pt_sphere_conc', 'Surf_micros_Pt_sphere_ratio', 
                 'Surf_micros', 'Surf_micros_bulk_pack_conc', 'Surf_micros_bulk_pack_ratio', 'Surf_micros_sphere_conc', 'Surf_micros_sphere_ratio', 
                 'Surf_facets_Au', 'Surf_facets_Au_bulk_pack_conc', 'Surf_facets_Au_bulk_pack_ratio', 'Surf_facets_Au_sphere_conc', 'Surf_facets_Au_sphere_ratio', 
                 'Surf_facets_Pd', 'Surf_facets_Pd_bulk_pack_conc', 'Surf_facets_Pd_bulk_pack_ratio', 'Surf_facets_Pd_sphere_conc', 'Surf_facets_Pd_sphere_ratio',
                 'Surf_facets_Pt', 'Surf_facets_Pt_bulk_pack_conc', 'Surf_facets_Pt_bulk_pack_ratio', 'Surf_facets_Pt_sphere_conc', 'Surf_facets_Pt_sphere_ratio', 
                 'Surf_facets', 'Surf_facets_bulk_pack_conc', 'Surf_facets_bulk_pack_ratio', 'Surf_facets_sphere_conc', 'Surf_facets_sphere_ratio']


def calcBulkPackVol(row):
    elements = [ELE_COMB[i:i+2] for i in range(0, len(ELE_COMB), 2)]  # Split ELE_COMB into list of 2 alphabets
    totalVol = 0
    for element in elements:
        volEle = row[f"N_{element}"] * (elePropDict[element]['m'] / N_AVOGADRO) / (elePropDict[element]['rho']/A3_PER_M3)
        totalVol += volEle
    return totalVol


def cntSurfSite(row, siteType, element):
    if len(element) > 2:
        elements = [ELE_COMB[i:i+2] for i in range(0, len(ELE_COMB), 2)]
        return sum([row[f"Surf_{siteType}_{ele}"] for ele in elements])
    
    if siteType == 'defects': return sum((row[f"{element}M_SCN_1"], row[f"{element}M_SCN_2"], row[f"{element}M_SCN_3"]))
    elif siteType == 'micros': return sum((row[f"{element}M_SCN_4"], row[f"{element}M_SCN_5"], row[f"{element}M_SCN_6"], row[f"{element}M_SCN_7"]))
    elif siteType == 'facets': return sum((row[f"{element}M_SCN_9"], row[f"{element}M_SCN_10"], row[f"{element}M_SCN_11"]))
    else: raise Exception(f"    {siteType} specified wrongly!")


def calcSurfSiteConc(row, siteType, element, volType):
    if len(element) > 2:
        elements = [ELE_COMB[i:i+2] for i in range(0, len(ELE_COMB), 2)]
        return sum([row[f"Surf_{siteType}_{ele}_{volType}_conc"] for ele in elements])
    return row[f"Surf_{siteType}_{element}"] / row[f"Vol_{volType}"]


def calcSurfSiteRatio(row, siteType, element, volType):  # TODO: Confim with Amanda
    if len(element) > 2:
        elements = [ELE_COMB[i:i+2] for i in range(0, len(ELE_COMB), 2)]
        return sum([row[f"Surf_{siteType}_{ele}_{volType}_ratio"] for ele in elements])
    return row[f"Surf_{siteType}_{element}"] / row['N_atom_total']
    # return row[f"Surf_{siteType}_{element}"] * (elePropDict[element]['m'] / N_AVOGADRO) / (elePropDict[element]['rho'] * (row[f"Vol_{volType}"] / A3_PER_M3))


def dropFeats(df, allHeaders, verbose=False):
    # (*) means could potentially be included if figured out how to work on time-series-like data
    # (**) means could potentially be binned and included like the curvature features
    if verbose: print('    Dropping unused features...')
    preDropColNum = len(df.columns)
    for col in ADD_FEAT_LIST: allHeaders.remove(col)  # Remove the feature names that will be added later
    curvStartColIdx, rdfStartColIdx, sfStartColIdx, baStartColIdx, btStartColIdxd, clStartColIdx = 22, 969, 1821, 2334, 12876, 14372  # To be adjusted if other parts are changed
    dropHeadersList = []

    # - Frame index 
    dropHeadersList.append(5)
    # Add temporary columns for individual curvature degrees
    for i, curvColIdx in enumerate(range(curvStartColIdx, curvStartColIdx+180)): allHeaders.insert(curvColIdx, f"Curve_Deg_{i+1}")
    # - Radial distribution function values (*)
    dropHeadersList.extend(list(range(rdfStartColIdx, rdfStartColIdx+852)))
    # - Structure factors (*)
    dropHeadersList.extend(list(range(sfStartColIdx, sfStartColIdx+513)))
    # - Individual bond angle 1 and bond angle 2 degrees (**)
    baDegCols = []
    skipStatcols, skipColsCntDown = True, 8
    for (i, colIdx) in enumerate(range(baStartColIdx, baStartColIdx+10528)):
        if skipColsCntDown == 0: skipStatcols = False
        if (colIdx-baStartColIdx) % 188 == 0:  # New elemental combination
            skipColsCntDown = 8
            skipStatcols = True
        if skipStatcols: skipColsCntDown -= 1
        else: baDegCols.append(colIdx)
    dropHeadersList.extend(baDegCols)
    # - Individual bond torsion degrees (**)
    dropHeadersList.extend(list(range(btStartColIdxd, btStartColIdxd+362)))
    # - Chain length
    dropHeadersList.extend(list(range(clStartColIdx, clStartColIdx+20)))

    df.drop(df.columns[dropHeadersList], axis=1, inplace=True)
    df = df[df.columns.drop(list(df.filter(regex='Type')))]  # - Types columns
    df = df.apply(pd.to_numeric, errors='coerce')  # Turning data numeric, be careful with 'coerce' option as there is a risk that blatant errors are omitted # TODO Check if anything missed
    df.columns = allHeaders
    # if verbose: print(f"        Number of columns dropped: {preDropColNum - len(df.columns)}")
    return df


def addFeats(df, verbose=False):
    if verbose: print('    Adding new features...')
    # - Volume= Mass / Density, Volume = 4/3*pi*r^3
    df['Vol_bulk_pack'] = df.apply(calcBulkPackVol, axis=1)  # Assuming bulk packing (m^3) TODO: Change to A^3?
    df['Vol_sphere'] = df.apply(lambda row: 3 / 4 * math.pi * row['R_avg']**3, axis=1)  # Geometric volume (A^3)
    # - Curvature
    endVal, curvColIdx = 0, 22
    for i in range(1, 19):
        curvColName = f"Curve_{endVal+1}-{i*10}"
        df[curvColName] = df.iloc[:, curvColIdx:curvColIdx+10].sum(axis=1)
        endVal = i * 10
        curvColIdx += 10
    # Drop individual curvature degrees columns
    curvStartColIdx = 22
    df.drop(df.columns[list(range(curvStartColIdx, curvStartColIdx+180))], axis=1, inplace=True)
    # - Surface characteristics concentration
    for characteristic in ('defects', 'micros', 'facets'):
        for element in ('Au', 'Pd', 'Pt'):
            df[f"Surf_{characteristic}_{element}"] = df.apply(lambda row: cntSurfSite(row, characteristic, element), axis=1)
            for volType in ('bulk_pack', 'sphere'):
                df[f"Surf_{characteristic}_{element}_{volType}_conc"] = df.apply(lambda row: calcSurfSiteConc(row, characteristic, element, volType), axis=1)
                df[f"Surf_{characteristic}_{element}_{volType}_ratio"] = df.apply(lambda row: calcSurfSiteRatio(row, characteristic, element, volType), axis=1)
        df[f"Surf_{characteristic}"] = df.apply(lambda row: cntSurfSite(row, characteristic, ELE_COMB), axis=1)
        for volType in ('bulk_pack', 'sphere'):
            df[f"Surf_{characteristic}_{volType}_conc"] = df.apply(lambda row: calcSurfSiteConc(row, characteristic, ELE_COMB, volType), axis=1)
            df[f"Surf_{characteristic}_{volType}_ratio"] = df.apply(lambda row: calcSurfSiteRatio(row, characteristic, ELE_COMB, volType), axis=1)
    # - Energies
    # df['Formation_E'] = df.apply(lambda row: row['Total_E'] - (row['N_Au']*elePropDict['Au']['bulkE'] + row['N_Pd']*elePropDict['Pd']['bulkE'] + row['N_Pt']*elePropDict['Pt']['bulkE']), axis=1)
    return df


def mergeReformatData(outputMD, verbose=True):
    """
    Concatenate MD output with features extracted from NCPac
    """
    if verbose: print(f"    Concatenating CSV files for nanoparticle {outputMD[0]}...")
    df1 = pd.DataFrame(outputMD).T
    df1.columns = ['ID', 'T', 'P', 'Potential_E', 'Kinetic_E', 'Total_E']
    if os.path.getsize(f"{featSourcePath}/{outputMD[0]}.csv") == 0: 
        print(f"    {outputMD[0]} is a BNP instead of TNP! Skipping...") 
        df = pd.DataFrame(columns=ALL_HEADERS_LIST)
        df.to_csv(f"{featEngPath}/{outputMD[0]}.csv", sep=',', header=True)
        return
    df2 = pd.read_csv(f"{featSourcePath}/{outputMD[0]}.csv", sep=',', header=1, index_col=None)  # usecols, low_memory
    df = pd.concat([df1, df2], axis='columns')
    df.set_index(keys='ID', inplace=True)

    df = dropFeats(df, ALL_HEADERS_LIST.copy(), verbose=verbose)  # Drop unused columns
    df = addFeats(df, verbose=verbose)  # Add new columns

    if verbose: print('    Reordering columns...')
    energyCols = ['Potential_E', 'Kinetic_E', 'Total_E']
    allHeadersOrdered = ALL_HEADERS_LIST.copy()
    for col in energyCols: allHeadersOrdered.remove(col)
    allHeadersOrdered.extend(energyCols)
    df = df[allHeadersOrdered]

    df.to_csv(f"{featEngPath}/{outputMD[0]}.csv", sep=',', header=True)


def runMergeReformatParallel(verbose=False):
    if verbose: print(f"Merging information and reformating data in parallel...")
    outputMDs = []
    with open(MDoutFName, 'r') as f:
        f.readline()
        for outputMD in csv.reader(f): 
            if not exists(f"{featEngPath}/{outputMD[0]}.csv"):
                outputMDs.append(outputMD)
    with Pool() as p: p.map(mergeReformatData, outputMDs)


def concatNPfeats(verbose=False):
    '''Fastest option to concatenate CSV files, almost 2 order of magnitudes faster than Pandas alternative'''
    if verbose: print(f"Concatenating processed feature CSV files...")
    featCSVs = sorted(os.listdir(featEngPath))
    with open(finalDataFName, 'wb') as fout:
        for (i, featCSV) in enumerate(featCSVs):
            if os.path.getsize(f"{featEngPath}/{featCSV}") == 0: 
                print(f"    {featCSV} is a BNP instead of TNP! Skipping...") 
                continue
            with open(f"{featEngPath}/{featCSV}", 'rb') as f:
                if i != 0: next(f)  # Skip header
                fout.write(f.read())


if __name__ == '__main__':
    if runTask == 'mergeReformatData':  # Parallel 
        if not isdir(featEngPath): os.mkdir(featEngPath)
        if runParallel: runMergeReformatParallel(verbose=True)
        else:
            with open(MDoutFName, 'r') as f:
                f.readline()
                for outputMD in csv.reader(f): 
                    if not exists(f"{featEngPath}/{outputMD[0]}.csv"):
                        mergeReformatData(outputMD, verbose=True)
    elif runTask == 'concatNPfeats':  # Serial 
        concatNPfeats(verbose=True)
    elif runTask == 'debug':
        outputMD = ['10448', '2247.7315', '172.04896', '-3821.53' ,'278.33933', '-3543.1907']
        mergeReformatData(outputMD, verbose=True) 
