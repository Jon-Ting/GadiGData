import csv
import math
from multiprocessing import Pool
import os
import pandas as pd


ELEMENTS = ['Au', 'Pt']
runTask = 'concatNPfeats' # 'mergeReformatData'  # 'mergeReformatData' or 'concatNPfeats' or 'debug'
runParallel, verbose = True, True
sourceDirs = ['L10', 'L12', 'RAL','RCS', 'CS']
sourceDirs = ['L10']

PROJECT, USER_NAME = 'q27', 'jt5911'
featEngPath = f"/scratch/{PROJECT}/{USER_NAME}/{''.join(ELEMENTS)}"
MDoutFName = f"{featEngPath}/MDout.csv"
finalDataFName = f"{featEngPath}/{''.join(ELEMENTS)}_nanoparticle_data.csv"

N_AVOGADRO = 6.02214076 * 10**23  # (atoms/mol)
A3_PER_M3 = 10 ** 30  # Angstrom^3 per m^3 (dimensionless)
elePropDict = {'Au': {'rho': 19320, 'm': 0.196967, 'bulkE': 3.81}, 
               'Pt': {'rho': 21450, 'm': 0.195084, 'bulkE': 5.84}, 
               'Pd': {'rho': 12023, 'm': 0.10642, 'bulkE': 3.89}, 
               'Co': {'rho': 8900, 'm': 0.058933, 'bulkE': 4.39}}  # density (kg/m^3), molar mass (kg/mol), cohesive energy per atom for bulk system (eV/atom)

# All features
ALL_HEADERS_LIST = ['T', 'P', 'Potential_E', 'Kinetic_E', 'Total_E', 
                    'N_atom_total', 'N_Ele1', 'N_Ele2', 'N_atom_bulk', 'N_atom_surface', 'Vol_bulk_pack', 'Vol_sphere', 
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
                    
                    'Ele1M_TCN_avg', 'Ele1M_BCN_avg', 'Ele1M_SCN_avg', 'Ele1M_SOCN_avg', 'Ele1M_TGCN_avg', 'Ele1M_BGCN_avg', 'Ele1M_SGCN_avg', 'Ele1M_SOGCN_avg', 
                    'Ele1M_TCN_0', 'Ele1M_TCN_1', 'Ele1M_TCN_2', 'Ele1M_TCN_3', 'Ele1M_TCN_4', 'Ele1M_TCN_5', 'Ele1M_TCN_6', 'Ele1M_TCN_7', 'Ele1M_TCN_8', 'Ele1M_TCN_9', 'Ele1M_TCN_10', 'Ele1M_TCN_11', 'Ele1M_TCN_12', 'Ele1M_TCN_13', 'Ele1M_TCN_14', 'Ele1M_TCN_15', 'Ele1M_TCN_16', 'Ele1M_TCN_17', 'Ele1M_TCN_18', 'Ele1M_TCN_19', 'Ele1M_TCN_20', 
                    'Ele1M_BCN_0', 'Ele1M_BCN_1', 'Ele1M_BCN_2', 'Ele1M_BCN_3', 'Ele1M_BCN_4', 'Ele1M_BCN_5', 'Ele1M_BCN_6', 'Ele1M_BCN_7', 'Ele1M_BCN_8', 'Ele1M_BCN_9', 'Ele1M_BCN_10', 'Ele1M_BCN_11', 'Ele1M_BCN_12', 'Ele1M_BCN_13', 'Ele1M_BCN_14', 'Ele1M_BCN_15', 'Ele1M_BCN_16', 'Ele1M_BCN_17', 'Ele1M_BCN_18', 'Ele1M_BCN_19', 'Ele1M_BCN_20', 
                    'Ele1M_SCN_0', 'Ele1M_SCN_1', 'Ele1M_SCN_2', 'Ele1M_SCN_3', 'Ele1M_SCN_4', 'Ele1M_SCN_5', 'Ele1M_SCN_6', 'Ele1M_SCN_7', 'Ele1M_SCN_8', 'Ele1M_SCN_9', 'Ele1M_SCN_10', 'Ele1M_SCN_11', 'Ele1M_SCN_12', 'Ele1M_SCN_13', 'Ele1M_SCN_14', 'Ele1M_SCN_15', 'Ele1M_SCN_16', 'Ele1M_SCN_17', 'Ele1M_SCN_18', 'Ele1M_SCN_19', 'Ele1M_SCN_20', 
                    'Ele1M_SOCN_0', 'Ele1M_SOCN_1', 'Ele1M_SOCN_2', 'Ele1M_SOCN_3', 'Ele1M_SOCN_4', 'Ele1M_SOCN_5', 'Ele1M_SOCN_6', 'Ele1M_SOCN_7', 'Ele1M_SOCN_8', 'Ele1M_SOCN_9', 'Ele1M_SOCN_10', 'Ele1M_SOCN_11', 'Ele1M_SOCN_12', 'Ele1M_SOCN_13', 'Ele1M_SOCN_14', 'Ele1M_SOCN_15', 'Ele1M_SOCN_16', 'Ele1M_SOCN_17', 'Ele1M_SOCN_18', 'Ele1M_SOCN_19', 'Ele1M_SOCN_20', 
                    'Ele1M_TGCN_0', 'Ele1M_TGCN_1', 'Ele1M_TGCN_2', 'Ele1M_TGCN_3', 'Ele1M_TGCN_4', 'Ele1M_TGCN_5', 'Ele1M_TGCN_6', 'Ele1M_TGCN_7', 'Ele1M_TGCN_8', 'Ele1M_TGCN_9', 'Ele1M_TGCN_10', 'Ele1M_TGCN_11', 'Ele1M_TGCN_12', 'Ele1M_TGCN_13', 'Ele1M_TGCN_14', 'Ele1M_TGCN_15', 'Ele1M_TGCN_16', 'Ele1M_TGCN_17', 'Ele1M_TGCN_18', 'Ele1M_TGCN_19', 'Ele1M_TGCN_20', 
                    'Ele1M_BGCN_0', 'Ele1M_BGCN_1', 'Ele1M_BGCN_2', 'Ele1M_BGCN_3', 'Ele1M_BGCN_4', 'Ele1M_BGCN_5', 'Ele1M_BGCN_6', 'Ele1M_BGCN_7', 'Ele1M_BGCN_8', 'Ele1M_BGCN_9', 'Ele1M_BGCN_10', 'Ele1M_BGCN_11', 'Ele1M_BGCN_12', 'Ele1M_BGCN_13', 'Ele1M_BGCN_14', 'Ele1M_BGCN_15', 'Ele1M_BGCN_16', 'Ele1M_BGCN_17', 'Ele1M_BGCN_18', 'Ele1M_BGCN_19', 'Ele1M_BGCN_20', 
                    'Ele1M_SGCN_0', 'Ele1M_SGCN_1', 'Ele1M_SGCN_2', 'Ele1M_SGCN_3', 'Ele1M_SGCN_4', 'Ele1M_SGCN_5', 'Ele1M_SGCN_6', 'Ele1M_SGCN_7', 'Ele1M_SGCN_8', 'Ele1M_SGCN_9', 'Ele1M_SGCN_10', 'Ele1M_SGCN_11', 'Ele1M_SGCN_12', 'Ele1M_SGCN_13', 'Ele1M_SGCN_14', 'Ele1M_SGCN_15', 'Ele1M_SGCN_16', 'Ele1M_SGCN_17', 'Ele1M_SGCN_18', 'Ele1M_SGCN_19', 'Ele1M_SGCN_20', 
                    'Ele1M_SOGCN_0', 'Ele1M_SOGCN_1', 'Ele1M_SOGCN_2', 'Ele1M_SOGCN_3', 'Ele1M_SOGCN_4', 'Ele1M_SOGCN_5', 'Ele1M_SOGCN_6', 'Ele1M_SOGCN_7', 'Ele1M_SOGCN_8', 'Ele1M_SOGCN_9', 'Ele1M_SOGCN_10', 'Ele1M_SOGCN_11', 'Ele1M_SOGCN_12', 'Ele1M_SOGCN_13', 'Ele1M_SOGCN_14', 'Ele1M_SOGCN_15', 'Ele1M_SOGCN_16', 'Ele1M_SOGCN_17', 'Ele1M_SOGCN_18', 'Ele1M_SOGCN_19', 'Ele1M_SOGCN_20', 
                    
                    'Ele2M_TCN_avg', 'Ele2M_BCN_avg', 'Ele2M_SCN_avg', 'Ele2M_SOCN_avg', 'Ele2M_TGCN_avg', 'Ele2M_BGCN_avg', 'Ele2M_SGCN_avg', 'Ele2M_SOGCN_avg', 
                    'Ele2M_TCN_0', 'Ele2M_TCN_1', 'Ele2M_TCN_2', 'Ele2M_TCN_3', 'Ele2M_TCN_4', 'Ele2M_TCN_5', 'Ele2M_TCN_6', 'Ele2M_TCN_7', 'Ele2M_TCN_8', 'Ele2M_TCN_9', 'Ele2M_TCN_10', 'Ele2M_TCN_11', 'Ele2M_TCN_12', 'Ele2M_TCN_13', 'Ele2M_TCN_14', 'Ele2M_TCN_15', 'Ele2M_TCN_16', 'Ele2M_TCN_17', 'Ele2M_TCN_18', 'Ele2M_TCN_19', 'Ele2M_TCN_20', 
                    'Ele2M_BCN_0', 'Ele2M_BCN_1', 'Ele2M_BCN_2', 'Ele2M_BCN_3', 'Ele2M_BCN_4', 'Ele2M_BCN_5', 'Ele2M_BCN_6', 'Ele2M_BCN_7', 'Ele2M_BCN_8', 'Ele2M_BCN_9', 'Ele2M_BCN_10', 'Ele2M_BCN_11', 'Ele2M_BCN_12', 'Ele2M_BCN_13', 'Ele2M_BCN_14', 'Ele2M_BCN_15', 'Ele2M_BCN_16', 'Ele2M_BCN_17', 'Ele2M_BCN_18', 'Ele2M_BCN_19', 'Ele2M_BCN_20', 
                    'Ele2M_SCN_0', 'Ele2M_SCN_1', 'Ele2M_SCN_2', 'Ele2M_SCN_3', 'Ele2M_SCN_4', 'Ele2M_SCN_5', 'Ele2M_SCN_6', 'Ele2M_SCN_7', 'Ele2M_SCN_8', 'Ele2M_SCN_9', 'Ele2M_SCN_10', 'Ele2M_SCN_11', 'Ele2M_SCN_12', 'Ele2M_SCN_13', 'Ele2M_SCN_14', 'Ele2M_SCN_15', 'Ele2M_SCN_16', 'Ele2M_SCN_17', 'Ele2M_SCN_18', 'Ele2M_SCN_19', 'Ele2M_SCN_20', 
                    'Ele2M_SOCN_0', 'Ele2M_SOCN_1', 'Ele2M_SOCN_2', 'Ele2M_SOCN_3', 'Ele2M_SOCN_4', 'Ele2M_SOCN_5', 'Ele2M_SOCN_6', 'Ele2M_SOCN_7', 'Ele2M_SOCN_8', 'Ele2M_SOCN_9', 'Ele2M_SOCN_10', 'Ele2M_SOCN_11', 'Ele2M_SOCN_12', 'Ele2M_SOCN_13', 'Ele2M_SOCN_14', 'Ele2M_SOCN_15', 'Ele2M_SOCN_16', 'Ele2M_SOCN_17', 'Ele2M_SOCN_18', 'Ele2M_SOCN_19', 'Ele2M_SOCN_20', 
                    'Ele2M_TGCN_0', 'Ele2M_TGCN_1', 'Ele2M_TGCN_2', 'Ele2M_TGCN_3', 'Ele2M_TGCN_4', 'Ele2M_TGCN_5', 'Ele2M_TGCN_6', 'Ele2M_TGCN_7', 'Ele2M_TGCN_8', 'Ele2M_TGCN_9', 'Ele2M_TGCN_10', 'Ele2M_TGCN_11', 'Ele2M_TGCN_12', 'Ele2M_TGCN_13', 'Ele2M_TGCN_14', 'Ele2M_TGCN_15', 'Ele2M_TGCN_16', 'Ele2M_TGCN_17', 'Ele2M_TGCN_18', 'Ele2M_TGCN_19', 'Ele2M_TGCN_20', 
                    'Ele2M_BGCN_0', 'Ele2M_BGCN_1', 'Ele2M_BGCN_2', 'Ele2M_BGCN_3', 'Ele2M_BGCN_4', 'Ele2M_BGCN_5', 'Ele2M_BGCN_6', 'Ele2M_BGCN_7', 'Ele2M_BGCN_8', 'Ele2M_BGCN_9', 'Ele2M_BGCN_10', 'Ele2M_BGCN_11', 'Ele2M_BGCN_12', 'Ele2M_BGCN_13', 'Ele2M_BGCN_14', 'Ele2M_BGCN_15', 'Ele2M_BGCN_16', 'Ele2M_BGCN_17', 'Ele2M_BGCN_18', 'Ele2M_BGCN_19', 'Ele2M_BGCN_20', 
                    'Ele2M_SGCN_0', 'Ele2M_SGCN_1', 'Ele2M_SGCN_2', 'Ele2M_SGCN_3', 'Ele2M_SGCN_4', 'Ele2M_SGCN_5', 'Ele2M_SGCN_6', 'Ele2M_SGCN_7', 'Ele2M_SGCN_8', 'Ele2M_SGCN_9', 'Ele2M_SGCN_10', 'Ele2M_SGCN_11', 'Ele2M_SGCN_12', 'Ele2M_SGCN_13', 'Ele2M_SGCN_14', 'Ele2M_SGCN_15', 'Ele2M_SGCN_16', 'Ele2M_SGCN_17', 'Ele2M_SGCN_18', 'Ele2M_SGCN_19', 'Ele2M_SGCN_20', 
                    'Ele2M_SOGCN_0', 'Ele2M_SOGCN_1', 'Ele2M_SOGCN_2', 'Ele2M_SOGCN_3', 'Ele2M_SOGCN_4', 'Ele2M_SOGCN_5', 'Ele2M_SOGCN_6', 'Ele2M_SOGCN_7', 'Ele2M_SOGCN_8', 'Ele2M_SOGCN_9', 'Ele2M_SOGCN_10', 'Ele2M_SOGCN_11', 'Ele2M_SOGCN_12', 'Ele2M_SOGCN_13', 'Ele2M_SOGCN_14', 'Ele2M_SOGCN_15', 'Ele2M_SOGCN_16', 'Ele2M_SOGCN_17', 'Ele2M_SOGCN_18', 'Ele2M_SOGCN_19', 'Ele2M_SOGCN_20', 

                    'MM_BL_avg', 'MM_BL_std', 'MM_BL_max', 'MM_BL_min', 'MM_BL_num',
                    'Ele1Ele1_BL_avg', 'Ele1Ele1_BL_std', 'Ele1Ele1_BL_max', 'Ele1Ele1_BL_min', 'Ele1Ele1_BL_num',
                    'Ele1Ele2_BL_avg', 'Ele1Ele2_BL_std', 'Ele1Ele2_BL_max', 'Ele1Ele2_BL_min', 'Ele1Ele2_BL_num',
                    'Ele2Ele2_BL_avg', 'Ele2Ele2_BL_std', 'Ele2Ele2_BL_max', 'Ele2Ele2_BL_min', 'Ele2Ele2_BL_num',
                    
                    'Ele1Ele1_frac', 'Ele1Ele2_frac', 'Ele2Ele2_frac', 'N_bond',
                    
                    'MMM_BA1_avg', 'MMM_BA1_std', 'MMM_BA1_max', 'MMM_BA1_min', 'MMM_BA1_num', 
                    'Ele1Ele1Ele1_BA1_avg', 'Ele1Ele1Ele1_BA1_std', 'Ele1Ele1Ele1_BA1_max', 'Ele1Ele1Ele1_BA1_min', 'Ele1Ele1Ele1_BA1_num', 
                    'Ele1Ele1Ele2_BA1_avg', 'Ele1Ele1Ele2_BA1_std', 'Ele1Ele1Ele2_BA1_max', 'Ele1Ele1Ele2_BA1_min', 'Ele1Ele1Ele2_BA1_num', 
                    'Ele1Ele2Ele1_BA1_avg', 'Ele1Ele2Ele1_BA1_std', 'Ele1Ele2Ele1_BA1_max', 'Ele1Ele2Ele1_BA1_min', 'Ele1Ele2Ele1_BA1_num', 
                    'Ele1Ele2Ele2_BA1_avg', 'Ele1Ele2Ele2_BA1_std', 'Ele1Ele2Ele2_BA1_max', 'Ele1Ele2Ele2_BA1_min', 'Ele1Ele2Ele2_BA1_num', 
                    'Ele2Ele1Ele1_BA1_avg', 'Ele2Ele1Ele1_BA1_std', 'Ele2Ele1Ele1_BA1_max', 'Ele2Ele1Ele1_BA1_min', 'Ele2Ele1Ele1_BA1_num', 
                    'Ele2Ele1Ele2_BA1_avg', 'Ele2Ele1Ele2_BA1_std', 'Ele2Ele1Ele2_BA1_max', 'Ele2Ele1Ele2_BA1_min', 'Ele2Ele1Ele2_BA1_num', 
                    'Ele2Ele2Ele1_BA1_avg', 'Ele2Ele2Ele1_BA1_std', 'Ele2Ele2Ele1_BA1_max', 'Ele2Ele2Ele1_BA1_min', 'Ele2Ele2Ele1_BA1_num', 
                    'Ele2Ele2Ele2_BA1_avg', 'Ele2Ele2Ele2_BA1_std', 'Ele2Ele2Ele2_BA1_max', 'Ele2Ele2Ele2_BA1_min', 'Ele2Ele2Ele2_BA1_num', 
                    
                    'MMM_BA2_avg', 'MMM_BA2_std', 'MMM_BA2_max', 'MMM_BA2_min', 'MMM_BA2_num', 
                    'Ele1Ele1Ele1_BA2_avg', 'Ele1Ele1Ele1_BA2_std', 'Ele1Ele1Ele1_BA2_max', 'Ele1Ele1Ele1_BA2_min', 'Ele1Ele1Ele1_BA2_num', 
                    'Ele1Ele1Ele2_BA2_avg', 'Ele1Ele1Ele2_BA2_std', 'Ele1Ele1Ele2_BA2_max', 'Ele1Ele1Ele2_BA2_min', 'Ele1Ele1Ele2_BA2_num', 
                    'Ele1Ele2Ele1_BA2_avg', 'Ele1Ele2Ele1_BA2_std', 'Ele1Ele2Ele1_BA2_max', 'Ele1Ele2Ele1_BA2_min', 'Ele1Ele2Ele1_BA2_num', 
                    'Ele1Ele2Ele2_BA2_avg', 'Ele1Ele2Ele2_BA2_std', 'Ele1Ele2Ele2_BA2_max', 'Ele1Ele2Ele2_BA2_min', 'Ele1Ele2Ele2_BA2_num', 
                    'Ele2Ele1Ele1_BA2_avg', 'Ele2Ele1Ele1_BA2_std', 'Ele2Ele1Ele1_BA2_max', 'Ele2Ele1Ele1_BA2_min', 'Ele2Ele1Ele1_BA2_num', 
                    'Ele2Ele1Ele2_BA2_avg', 'Ele2Ele1Ele2_BA2_std', 'Ele2Ele1Ele2_BA2_max', 'Ele2Ele1Ele2_BA2_min', 'Ele2Ele1Ele2_BA2_num', 
                    'Ele2Ele2Ele1_BA2_avg', 'Ele2Ele2Ele1_BA2_std', 'Ele2Ele2Ele1_BA2_max', 'Ele2Ele2Ele1_BA2_min', 'Ele2Ele2Ele1_BA2_num', 
                    'Ele2Ele2Ele2_BA2_avg', 'Ele2Ele2Ele2_BA2_std', 'Ele2Ele2Ele2_BA2_max', 'Ele2Ele2Ele2_BA2_min', 'Ele2Ele2Ele2_BA2_num', 
                    
                    'MMMM_BTneg_avg', 'MMMM_BTneg_std', 'MMMM_BTneg_max', 'MMMM_BTneg_min', 'MMMM_BTneg_num', 'MMMM_BTpos_avg', 'MMMM_BTpos_std', 'MMMM_BTpos_max', 'MMMM_BTpos_min', 'MMMM_BTpos_num', 
                    'Ele1Ele1Ele1Ele1_BTneg_avg', 'Ele1Ele1Ele1Ele1_BTneg_std', 'Ele1Ele1Ele1Ele1_BTneg_max', 'Ele1Ele1Ele1Ele1_BTneg_min', 'Ele1Ele1Ele1Ele1_BTneg_num', 'Ele1Ele1Ele1Ele1_BTpos_avg', 'Ele1Ele1Ele1Ele1_BTpos_std', 'Ele1Ele1Ele1Ele1_BTpos_max', 'Ele1Ele1Ele1Ele1_BTpos_min', 'Ele1Ele1Ele1Ele1_BTpos_num', 
                    'Ele1Ele1Ele1Ele2_BTneg_avg', 'Ele1Ele1Ele1Ele2_BTneg_std', 'Ele1Ele1Ele1Ele2_BTneg_max', 'Ele1Ele1Ele1Ele2_BTneg_min', 'Ele1Ele1Ele1Ele2_BTneg_num', 'Ele1Ele1Ele1Ele2_BTpos_avg', 'Ele1Ele1Ele1Ele2_BTpos_std', 'Ele1Ele1Ele1Ele2_BTpos_max', 'Ele1Ele1Ele1Ele2_BTpos_min', 'Ele1Ele1Ele1Ele2_BTpos_num', 
                    'Ele1Ele1Ele2Ele1_BTneg_avg', 'Ele1Ele1Ele2Ele1_BTneg_std', 'Ele1Ele1Ele2Ele1_BTneg_max', 'Ele1Ele1Ele2Ele1_BTneg_min', 'Ele1Ele1Ele2Ele1_BTneg_num', 'Ele1Ele1Ele2Ele1_BTpos_avg', 'Ele1Ele1Ele2Ele1_BTpos_std', 'Ele1Ele1Ele2Ele1_BTpos_max', 'Ele1Ele1Ele2Ele1_BTpos_min', 'Ele1Ele1Ele2Ele1_BTpos_num', 
                    'Ele1Ele1Ele2Ele2_BTneg_avg', 'Ele1Ele1Ele2Ele2_BTneg_std', 'Ele1Ele1Ele2Ele2_BTneg_max', 'Ele1Ele1Ele2Ele2_BTneg_min', 'Ele1Ele1Ele2Ele2_BTneg_num', 'Ele1Ele1Ele2Ele2_BTpos_avg', 'Ele1Ele1Ele2Ele2_BTpos_std', 'Ele1Ele1Ele2Ele2_BTpos_max', 'Ele1Ele1Ele2Ele2_BTpos_min', 'Ele1Ele1Ele2Ele2_BTpos_num', 
                    'Ele1Ele2Ele1Ele1_BTneg_avg', 'Ele1Ele2Ele1Ele1_BTneg_std', 'Ele1Ele2Ele1Ele1_BTneg_max', 'Ele1Ele2Ele1Ele1_BTneg_min', 'Ele1Ele2Ele1Ele2_BTneg_num', 'Ele1Ele2Ele1Ele1_BTpos_avg', 'Ele1Ele2Ele1Ele1_BTpos_std', 'Ele1Ele2Ele1Ele1_BTpos_max', 'Ele1Ele2Ele1Ele1_BTpos_min', 'Ele1Ele2Ele1Ele1_BTpos_num', 
                    'Ele1Ele2Ele1Ele2_BTneg_avg', 'Ele1Ele2Ele1Ele2_BTneg_std', 'Ele1Ele2Ele1Ele2_BTneg_max', 'Ele1Ele2Ele1Ele2_BTneg_min', 'Ele1Ele2Ele1Ele2_BTneg_num', 'Ele1Ele2Ele1Ele2_BTpos_avg', 'Ele1Ele2Ele1Ele2_BTpos_std', 'Ele1Ele2Ele1Ele2_BTpos_max', 'Ele1Ele2Ele1Ele2_BTpos_min', 'Ele1Ele2Ele1Ele2_BTpos_num', 
                    'Ele1Ele2Ele2Ele1_BTneg_avg', 'Ele1Ele2Ele2Ele1_BTneg_std', 'Ele1Ele2Ele2Ele1_BTneg_max', 'Ele1Ele2Ele2Ele1_BTneg_min', 'Ele1Ele2Ele2Ele1_BTneg_num', 'Ele1Ele2Ele2Ele1_BTpos_avg', 'Ele1Ele2Ele2Ele1_BTpos_std', 'Ele1Ele2Ele2Ele1_BTpos_max', 'Ele1Ele2Ele2Ele1_BTpos_min', 'Ele1Ele2Ele2Ele1_BTpos_num', 
                    'Ele1Ele2Ele2Ele2_BTneg_avg', 'Ele1Ele2Ele2Ele2_BTneg_std', 'Ele1Ele2Ele2Ele2_BTneg_max', 'Ele1Ele2Ele2Ele2_BTneg_min', 'Ele1Ele2Ele2Ele2_BTneg_num', 'Ele1Ele2Ele2Ele2_BTpos_avg', 'Ele1Ele2Ele2Ele2_BTpos_std', 'Ele1Ele2Ele2Ele2_BTpos_max', 'Ele1Ele2Ele2Ele2_BTpos_min', 'Ele1Ele2Ele2Ele2_BTpos_num', 
                    
                    'Ele2Ele1Ele1Ele1_BTneg_avg', 'Ele2Ele1Ele1Ele1_BTneg_std', 'Ele2Ele1Ele1Ele1_BTneg_max', 'Ele2Ele1Ele1Ele1_BTneg_min', 'Ele2Ele1Ele1Ele1_BTneg_num', 'Ele2Ele1Ele1Ele1_BTpos_avg', 'Ele2Ele1Ele1Ele1_BTpos_std', 'Ele2Ele1Ele1Ele1_BTpos_max', 'Ele2Ele1Ele1Ele1_BTpos_min', 'Ele2Ele1Ele1Ele1_BTpos_num', 
                    'Ele2Ele1Ele1Ele2_BTneg_avg', 'Ele2Ele1Ele1Ele2_BTneg_std', 'Ele2Ele1Ele1Ele2_BTneg_max', 'Ele2Ele1Ele1Ele2_BTneg_min', 'Ele2Ele1Ele1Ele2_BTneg_num', 'Ele2Ele1Ele1Ele2_BTpos_avg', 'Ele2Ele1Ele1Ele2_BTpos_std', 'Ele2Ele1Ele1Ele2_BTpos_max', 'Ele2Ele1Ele1Ele2_BTpos_min', 'Ele2Ele1Ele1Ele2_BTpos_num', 
                    'Ele2Ele1Ele2Ele1_BTneg_avg', 'Ele2Ele1Ele2Ele1_BTneg_std', 'Ele2Ele1Ele2Ele1_BTneg_max', 'Ele2Ele1Ele2Ele1_BTneg_min', 'Ele2Ele1Ele2Ele1_BTneg_num', 'Ele2Ele1Ele2Ele1_BTpos_avg', 'Ele2Ele1Ele2Ele1_BTpos_std', 'Ele2Ele1Ele2Ele1_BTpos_max', 'Ele2Ele1Ele2Ele1_BTpos_min', 'Ele2Ele1Ele2Ele1_BTpos_num', 
                    'Ele2Ele1Ele2Ele2_BTneg_avg', 'Ele2Ele1Ele2Ele2_BTneg_std', 'Ele2Ele1Ele2Ele2_BTneg_max', 'Ele2Ele1Ele2Ele2_BTneg_min', 'Ele2Ele1Ele2Ele2_BTneg_num', 'Ele2Ele1Ele2Ele2_BTpos_avg', 'Ele2Ele1Ele2Ele2_BTpos_std', 'Ele2Ele1Ele2Ele2_BTpos_max', 'Ele2Ele1Ele2Ele2_BTpos_min', 'Ele2Ele1Ele2Ele2_BTpos_num', 
                    'Ele2Ele2Ele1Ele1_BTneg_avg', 'Ele2Ele2Ele1Ele1_BTneg_std', 'Ele2Ele2Ele1Ele1_BTneg_max', 'Ele2Ele2Ele1Ele1_BTneg_min', 'Ele2Ele2Ele1Ele1_BTneg_num', 'Ele2Ele2Ele1Ele1_BTpos_avg', 'Ele2Ele2Ele1Ele1_BTpos_std', 'Ele2Ele2Ele1Ele1_BTpos_max', 'Ele2Ele2Ele1Ele1_BTpos_min', 'Ele2Ele2Ele1Ele1_BTpos_num', 
                    'Ele2Ele2Ele1Ele2_BTneg_avg', 'Ele2Ele2Ele1Ele2_BTneg_std', 'Ele2Ele2Ele1Ele2_BTneg_max', 'Ele2Ele2Ele1Ele2_BTneg_min', 'Ele2Ele2Ele1Ele2_BTneg_num', 'Ele2Ele2Ele1Ele2_BTpos_avg', 'Ele2Ele2Ele1Ele2_BTpos_std', 'Ele2Ele2Ele1Ele2_BTpos_max', 'Ele2Ele2Ele1Ele2_BTpos_min', 'Ele2Ele2Ele1Ele2_BTpos_num', 
                    'Ele2Ele2Ele2Ele1_BTneg_avg', 'Ele2Ele2Ele2Ele1_BTneg_std', 'Ele2Ele2Ele2Ele1_BTneg_max', 'Ele2Ele2Ele2Ele1_BTneg_min', 'Ele2Ele2Ele2Ele1_BTneg_num', 'Ele2Ele2Ele2Ele1_BTpos_avg', 'Ele2Ele2Ele2Ele1_BTpos_std', 'Ele2Ele2Ele2Ele1_BTpos_max', 'Ele2Ele2Ele2Ele1_BTpos_min', 'Ele2Ele2Ele2Ele1_BTpos_num', 
                    'Ele2Ele2Ele2Ele2_BTneg_avg', 'Ele2Ele2Ele2Ele2_BTneg_std', 'Ele2Ele2Ele2Ele2_BTneg_max', 'Ele2Ele2Ele2Ele2_BTneg_min', 'Ele2Ele2Ele2Ele2_BTneg_num', 'Ele2Ele2Ele2Ele2_BTpos_avg', 'Ele2Ele2Ele2Ele2_BTpos_std', 'Ele2Ele2Ele2Ele2_BTpos_max', 'Ele2Ele2Ele2Ele2_BTpos_min', 'Ele2Ele2Ele2Ele2_BTpos_num', 

                    'q6q6_T_avg', 'q6q6_B_avg', 'q6q6_S_avg', 
                    'q6q6_T_0', 'q6q6_T_1', 'q6q6_T_2', 'q6q6_T_3', 'q6q6_T_4', 'q6q6_T_5', 'q6q6_T_6', 'q6q6_T_7', 'q6q6_T_8', 'q6q6_T_9', 'q6q6_T_10', 'q6q6_T_11', 'q6q6_T_12', 'q6q6_T_13', 'q6q6_T_14', 'q6q6_T_15', 'q6q6_T_16', 'q6q6_T_17', 'q6q6_T_18', 'q6q6_T_19', 'q6q6_T_20', 'q6q6_T_20+', 
                    'q6q6_B_0', 'q6q6_B_1', 'q6q6_B_2', 'q6q6_B_3', 'q6q6_B_4', 'q6q6_B_5', 'q6q6_B_6', 'q6q6_B_7', 'q6q6_B_8', 'q6q6_B_9', 'q6q6_B_10', 'q6q6_B_11', 'q6q6_B_12', 'q6q6_B_13', 'q6q6_B_14', 'q6q6_B_15', 'q6q6_B_16', 'q6q6_B_17', 'q6q6_B_18', 'q6q6_B_19', 'q6q6_B_20', 'q6q6_B_20+', 
                    'q6q6_S_0', 'q6q6_S_1', 'q6q6_S_2', 'q6q6_S_3', 'q6q6_S_4', 'q6q6_S_5', 'q6q6_S_6', 'q6q6_S_7', 'q6q6_S_8', 'q6q6_S_9', 'q6q6_S_10', 'q6q6_S_11', 'q6q6_S_12', 'q6q6_S_13', 'q6q6_S_14', 'q6q6_S_15', 'q6q6_S_16', 'q6q6_S_17', 'q6q6_S_18', 'q6q6_S_19', 'q6q6_S_20', 'q6q6_S_20+', 
                   
                    'FCC', 'HCP', 'ICOS', 'DECA', 
              
                    'Surf_defects_Ele1', 'Surf_defects_Ele1_bulk_pack_conc', 'Surf_defects_Ele1_bulk_pack_ratio', 'Surf_defects_Ele1_sphere_conc', 'Surf_defects_Ele1_sphere_ratio', 
                    'Surf_defects_Ele2', 'Surf_defects_Ele2_bulk_pack_conc', 'Surf_defects_Ele2_bulk_pack_ratio', 'Surf_defects_Ele2_sphere_conc', 'Surf_defects_Ele2_sphere_ratio', 
                    'Surf_defects', 'Surf_defects_bulk_pack_conc', 'Surf_defects_bulk_pack_ratio', 'Surf_defects_sphere_conc', 'Surf_defects_sphere_ratio', 
                    'Surf_micros_Ele1', 'Surf_micros_Ele1_bulk_pack_conc', 'Surf_micros_Ele1_bulk_pack_ratio', 'Surf_micros_Ele1_sphere_conc', 'Surf_micros_Ele1_sphere_ratio', 
                    'Surf_micros_Ele2', 'Surf_micros_Ele2_bulk_pack_conc', 'Surf_micros_Ele2_bulk_pack_ratio', 'Surf_micros_Ele2_sphere_conc', 'Surf_micros_Ele2_sphere_ratio', 
                    'Surf_micros', 'Surf_micros_bulk_pack_conc', 'Surf_micros_bulk_pack_ratio', 'Surf_micros_sphere_conc', 'Surf_micros_sphere_ratio', 
                    'Surf_facets_Ele1', 'Surf_facets_Ele1_bulk_pack_conc', 'Surf_facets_Ele1_bulk_pack_ratio', 'Surf_facets_Ele1_sphere_conc', 'Surf_facets_Ele1_sphere_ratio', 
                    'Surf_facets_Ele2', 'Surf_facets_Ele2_bulk_pack_conc', 'Surf_facets_Ele2_bulk_pack_ratio', 'Surf_facets_Ele2_sphere_conc', 'Surf_facets_Ele2_sphere_ratio',
                    'Surf_facets', 'Surf_facets_bulk_pack_conc', 'Surf_facets_bulk_pack_ratio', 'Surf_facets_sphere_conc', 'Surf_facets_sphere_ratio']
# Features to be added
ADD_FEAT_LIST = ['Vol_bulk_pack', 'Vol_sphere', 
                 'Curve_1-10', 'Curve_11-20', 'Curve_21-30', 'Curve_31-40', 'Curve_41-50', 'Curve_51-60', 'Curve_61-70', 'Curve_71-80', 'Curve_81-90', 'Curve_91-100', 'Curve_101-110', 'Curve_111-120', 'Curve_121-130', 'Curve_131-140', 'Curve_141-150', 'Curve_151-160', 'Curve_161-170', 'Curve_171-180',
                 'Surf_defects_Ele1', 'Surf_defects_Ele1_bulk_pack_conc', 'Surf_defects_Ele1_bulk_pack_ratio', 'Surf_defects_Ele1_sphere_conc', 'Surf_defects_Ele1_sphere_ratio', 
                 'Surf_defects_Ele2', 'Surf_defects_Ele2_bulk_pack_conc', 'Surf_defects_Ele2_bulk_pack_ratio', 'Surf_defects_Ele2_sphere_conc', 'Surf_defects_Ele2_sphere_ratio', 
                 'Surf_defects', 'Surf_defects_bulk_pack_conc', 'Surf_defects_bulk_pack_ratio', 'Surf_defects_sphere_conc', 'Surf_defects_sphere_ratio', 
                 'Surf_micros_Ele1', 'Surf_micros_Ele1_bulk_pack_conc', 'Surf_micros_Ele1_bulk_pack_ratio', 'Surf_micros_Ele1_sphere_conc', 'Surf_micros_Ele1_sphere_ratio', 
                 'Surf_micros_Ele2', 'Surf_micros_Ele2_bulk_pack_conc', 'Surf_micros_Ele2_bulk_pack_ratio', 'Surf_micros_Ele2_sphere_conc', 'Surf_micros_Ele2_sphere_ratio', 
                 'Surf_micros', 'Surf_micros_bulk_pack_conc', 'Surf_micros_bulk_pack_ratio', 'Surf_micros_sphere_conc', 'Surf_micros_sphere_ratio', 
                 'Surf_facets_Ele1', 'Surf_facets_Ele1_bulk_pack_conc', 'Surf_facets_Ele1_bulk_pack_ratio', 'Surf_facets_Ele1_sphere_conc', 'Surf_facets_Ele1_sphere_ratio', 
                 'Surf_facets_Ele2', 'Surf_facets_Ele2_bulk_pack_conc', 'Surf_facets_Ele2_bulk_pack_ratio', 'Surf_facets_Ele2_sphere_conc', 'Surf_facets_Ele2_sphere_ratio',
                 'Surf_facets', 'Surf_facets_bulk_pack_conc', 'Surf_facets_bulk_pack_ratio', 'Surf_facets_sphere_conc', 'Surf_facets_sphere_ratio']


def calcBulkPackVol(row):
    totalVol = 0
    for (i, element) in enumerate(ELEMENTS):
        volEle = row[f"N_Ele{i+1}"] * (elePropDict[element]['m'] / N_AVOGADRO) / (elePropDict[element]['rho']/A3_PER_M3)
        totalVol += volEle
    return totalVol


def cntSurfSite(row, siteType, eleIdx):
    if eleIdx < 0:
        return sum([row[f"Surf_{siteType}_Ele{i+1}"] for i in range(len(ELEMENTS))])
    
    if siteType == 'defects': return sum((row[f"Ele{eleIdx}M_SCN_1"], row[f"Ele{eleIdx}M_SCN_2"], row[f"Ele{eleIdx}M_SCN_3"]))
    elif siteType == 'micros': return sum((row[f"Ele{eleIdx}M_SCN_4"], row[f"Ele{eleIdx}M_SCN_5"], row[f"Ele{eleIdx}M_SCN_6"], row[f"Ele{eleIdx}M_SCN_7"]))
    elif siteType == 'facets': return sum((row[f"Ele{eleIdx}M_SCN_9"], row[f"Ele{eleIdx}M_SCN_10"], row[f"Ele{eleIdx}M_SCN_11"]))
    else: raise Exception(f"    {siteType} specified wrongly!")


def calcSurfSiteConc(row, siteType, eleIdx, volType):
    if eleIdx < 0:
        return sum([row[f"Surf_{siteType}_Ele{i+1}_{volType}_conc"] for i in range(len(ELEMENTS))])
    return row[f"Surf_{siteType}_Ele{eleIdx}"] / row[f"Vol_{volType}"]


def calcSurfSiteRatio(row, siteType, element, eleIdx, volType):  # TODO: Confim with Amanda, 'element' not used for now
    if eleIdx < 0:
        return sum([row[f"Surf_{siteType}_Ele{i+1}_{volType}_ratio"] for i in range(len(ELEMENTS))])
    return row[f"Surf_{siteType}_Ele{eleIdx}"] / row['N_atom_total']
    # return row[f"Surf_{siteType}_Ele{eleIdx}"] * (elePropDict[element]['m'] / N_AVOGADRO) / (elePropDict[element]['rho'] * (row[f"Vol_{volType}"] / A3_PER_M3))


def dropFeats(df, allHeaders, verbose=False):
    # (*) means could potentially be included if figured out how to work on time-series-like data
    # (**) means could potentially be binned and included like the curvature features
    if verbose: print('    Dropping unused features...')
    preDropColNum = len(df.columns)
    for col in ADD_FEAT_LIST: allHeaders.remove(col)  # Remove the feature names that will be added later

    # To be adjusted when number of elements changes
    curvStartColIdx, rdfStartColIdx, sfStartColIdx, baStartColIdx, btStartColIdxd, clStartColIdx = 22, 767, 1253, 1766, 5164, 5750
    rdfColNum, baColNum = 486, 3384
    
    dropHeadersList = []
    # - Frame index 
    dropHeadersList.append(5)
    # Add temporary columns for individual curvature degrees
    for i, curvColIdx in enumerate(range(curvStartColIdx, curvStartColIdx+180)): allHeaders.insert(curvColIdx, f"Curve_Deg_{i+1}")
    # - Radial distribution function values (*)
    dropHeadersList.extend(list(range(rdfStartColIdx, rdfStartColIdx+rdfColNum)))
    # - Structure factors (*)
    dropHeadersList.extend(list(range(sfStartColIdx, sfStartColIdx+513)))
    # - Individual bond angle 1 and bond angle 2 degrees (**)
    baDegCols = []
    skipStatcols, skipColsCntDown = True, 8
    for (i, colIdx) in enumerate(range(baStartColIdx, baStartColIdx+baColNum)):
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
        for (i, element) in enumerate(ELEMENTS):
            df[f"Surf_{characteristic}_Ele{i+1}"] = df.apply(lambda row: cntSurfSite(row, characteristic, i+1), axis=1)
            for volType in ('bulk_pack', 'sphere'):
                df[f"Surf_{characteristic}_Ele{i+1}_{volType}_conc"] = df.apply(lambda row: calcSurfSiteConc(row, characteristic, i+1, volType), axis=1)
                df[f"Surf_{characteristic}_Ele{i+1}_{volType}_ratio"] = df.apply(lambda row: calcSurfSiteRatio(row, characteristic, element, i+1, volType), axis=1)
        df[f"Surf_{characteristic}"] = df.apply(lambda row: cntSurfSite(row, characteristic, -1), axis=1)
        for volType in ('bulk_pack', 'sphere'):
            df[f"Surf_{characteristic}_{volType}_conc"] = df.apply(lambda row: calcSurfSiteConc(row, characteristic, -1, volType), axis=1)
            df[f"Surf_{characteristic}_{volType}_ratio"] = df.apply(lambda row: calcSurfSiteRatio(row, characteristic, None, -1, volType), axis=1)
    # - Energies
    # df['Formation_E'] = df.apply(lambda row: row['Total_E'] - (row['N_Ele1']*elePropDict[ELEMENTS[0]]['bulkE'] + row['N_Ele2']*elePropDict[ELEMENTS[1]]['bulkE']), axis=1)
    # df['Cohesive E']
    # df['Surface E']
    return df


def mergeReformatData(outputMD, verbose=True):
    """
    Concatenate MD output with features extracted from NCPac
    """
    if verbose: print(f"    Concatenating CSV files for nanoparticle {outputMD[0]}...")
    df1 = pd.DataFrame(outputMD).T
    df1.columns = ['confID', 'T', 'P', 'Potential_E', 'Kinetic_E', 'Total_E']
    if os.path.getsize(f"{featEngPath}/{outputMD[0]}/od_FEATURESET.csv") == 0: 
        print(f"    *{outputMD[0]} is problematic! Skipping...") 
        df = pd.DataFrame(columns=ALL_HEADERS_LIST)
        df.to_csv(f"{featEngPath}/{outputMD[0]}/{outputMD[0]}.csv", sep=',', header=True)
        return
    df2 = pd.read_csv(f"{featEngPath}/{outputMD[0]}/od_FEATURESET.csv", sep=',', header=1, index_col=None)  # usecols, low_memory
    df = pd.concat([df1, df2], axis='columns')
    df.set_index(keys='confID', inplace=True)

    df = dropFeats(df, ALL_HEADERS_LIST.copy(), verbose=verbose)  # Drop unused columns
    df = addFeats(df, verbose=verbose)  # Add new columns

    if verbose: print('    Reordering columns...')
    energyCols = ['Potential_E', 'Kinetic_E', 'Total_E']
    allHeadersOrdered = ALL_HEADERS_LIST.copy()
    for col in energyCols: allHeadersOrdered.remove(col)
    allHeadersOrdered.extend(energyCols)
    df = df[allHeadersOrdered]

    df.to_csv(f"{featEngPath}/{outputMD[0]}/{outputMD[0]}.csv", sep=',', header=True)


def runMergeReformatParallel(verbose=False):
    if verbose: print(f"Merging information and reformating data in parallel...")
    outputMDs = []
    with open(MDoutFName, 'r') as f:
        f.readline()
        for outputMD in csv.reader(f): 
            if not os.path.exists(f"{featEngPath}/{outputMD[0]}/{outputMD[0]}.csv"):
                outputMDs.append(outputMD)
    with Pool() as p: p.map(mergeReformatData, outputMDs)


def concatNPfeats(verbose=False):
    '''Fastest option to concatenate CSV files, almost 2 order of magnitudes faster than Pandas alternative'''
    if verbose: print(f"Concatenating processed feature CSV files...")
    NPconfs = sorted(os.listdir(featEngPath))
    with open(finalDataFName, 'wb') as fout:
        for (i, NPconf) in enumerate(NPconfs):
            if not os.path.isdir(f"{featEngPath}/{NPconf}"): continue
            if os.path.getsize(f"{featEngPath}/{NPconf}/{NPconf}.csv") == 0: 
                print(f"    *{NPconf} is problematic! Skipping...") 
                continue
            with open(f"{featEngPath}/{NPconf}/{NPconf}.csv", 'rb') as f:
                if i != 0: next(f)  # Skip header
                fout.write(f.read())


if __name__ == '__main__':
    if runTask == 'mergeReformatData':  # Parallel 
        runMergeReformatParallel(verbose=True)
    elif runTask == 'concatNPfeats':  # Serial 
        concatNPfeats(verbose=True)
    elif runTask == 'debug':
        outputMD = ['000015', '273.15', '22.1', '21.1', '-101.1', '-81.0']
        mergeReformatData(outputMD, verbose=True) 
