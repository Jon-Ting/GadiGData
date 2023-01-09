import csv
import math
from multiprocessing import Pool
import os
from os.path import isdir, exists
import pandas as pd


PROJECT, USER_NAME, ELE_COMB = 'p00', 'jt5911', 'AuPtPd'
MDoutFName = f"/scratch/{PROJECT}/{USER_NAME}/{ELE_COMB}/MDout.csv"
featSourcePath = f"/scratch/{PROJECT}/{USER_NAME}/{ELE_COMB}/finalData/Features"
featEngPath = f"/scratch/{PROJECT}/{USER_NAME}/{ELE_COMB}/finalData/FeatEng"
finalDataFName = f"/scratch/{PROJECT}/{USER_NAME}/{ELE_COMB}/finalData/{ELE_COMB}_nanoparticle_data.csv"

allHeaders = ['T', 'P', 'Kinetic_E', 'Potential_E', 'Total_E', 
              'N_atom_total', 'N_Au', 'N_Pt', 'N_Pd', 'N_atom_bulk', 'N_atom_surface', 'Volume', 
              'R_min', 'R_max', 'R_diff', 'R_avg', 'R_std', 'meR_skew', 'R_kurt',
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
              
              'MMM_BA1_sum', 'MMM_BA1_avg', 'MMM_BA1_std', 
              'AuAuAu_BA1_sum', 'AuAuAu_BA1_avg', 'AuAuAu_BA1_std', 'AuAuPd_BA1_sum', 'AuAuPd_BA1_avg', 'AuAuPd_BA1_std', 'AuAuPt_BA1_sum', 'AuAuPt_BA1_avg', 'AuAuPt_BA1_std', 
              'AuPdAu_BA1_sum', 'AuPdAu_BA1_avg', 'AuPdAu_BA1_std', 'AuPdPd_BA1_sum', 'AuPdPd_BA1_avg', 'AuPdPd_BA1_std', 'AuPdPt_BA1_sum', 'AuPdPt_BA1_avg', 'AuPdPt_BA1_std', 
              'AuPtAu_BA1_sum', 'AuPtAu_BA1_avg', 'AuPtAu_BA1_std', 'AuPtPd_BA1_sum', 'AuPtPd_BA1_avg', 'AuPtPd_BA1_std', 'AuPtPt_BA1_sum', 'AuPtPt_BA1_avg', 'AuPtPt_BA1_std', 
              'PdAuAu_BA1_sum', 'PdAuAu_BA1_avg', 'PdAuAu_BA1_std', 'PdAuPd_BA1_sum', 'PdAuPd_BA1_avg', 'PdAuPd_BA1_std', 'PdAuPt_BA1_sum', 'PdAuPt_BA1_avg', 'PdAuPt_BA1_std', 
              'PdPdAu_BA1_sum', 'PdPdAu_BA1_avg', 'PdPdAu_BA1_std', 'PdPdPd_BA1_sum', 'PdPdPd_BA1_avg', 'PdPdPd_BA1_std', 'PdPdPt_BA1_sum', 'PdPdPt_BA1_avg', 'PdPdPt_BA1_std', 
              'PdPtAu_BA1_sum', 'PdPtAu_BA1_avg', 'PdPtAu_BA1_std', 'PdPtPd_BA1_sum', 'PdPtPd_BA1_avg', 'PdPtPd_BA1_std', 'PdPtPt_BA1_sum', 'PdPtPt_BA1_avg', 'PdPtPt_BA1_std', 
              'PtAuAu_BA1_sum', 'PtAuAu_BA1_avg', 'PtAuAu_BA1_std', 'PtAuPd_BA1_sum', 'PtAuPd_BA1_avg', 'PtAuPd_BA1_std', 'PtAuPt_BA1_sum', 'PtAuPt_BA1_avg', 'PtAuPt_BA1_std', 
              'PtPdAu_BA1_sum', 'PtPdAu_BA1_avg', 'PtPdAu_BA1_std', 'PtPdPd_BA1_sum', 'PtPdPd_BA1_avg', 'PtPdPd_BA1_std', 'PtPdPt_BA1_sum', 'PtPdPt_BA1_avg', 'PtPdPt_BA1_std', 
              'PtPtAu_BA1_sum', 'PtPtAu_BA1_avg', 'PtPtAu_BA1_std', 'PtPtPd_BA1_sum', 'PtPtPd_BA1_avg', 'PtPtPd_BA1_std', 'PtPtPt_BA1_sum', 'PtPtPt_BA1_avg', 'PtPtPt_BA1_std', 
              
              'MMM_BA2_sum', 'MMM_BA2_avg', 'MMM_BA2_std', 
              'AuAuAu_BA2_sum', 'AuAuAu_BA2_avg', 'AuAuAu_BA2_std', 'AuAuPd_BA2_sum', 'AuAuPd_BA2_avg', 'AuAuPd_BA2_std', 'AuAuPt_BA2_sum', 'AuAuPt_BA2_avg', 'AuAuPt_BA2_std', 
              'AuPdAu_BA2_sum', 'AuPdAu_BA2_avg', 'AuPdAu_BA2_std', 'AuPdPd_BA2_sum', 'AuPdPd_BA2_avg', 'AuPdPd_BA2_std', 'AuPdPt_BA2_sum', 'AuPdPt_BA2_avg', 'AuPdPt_BA2_std', 
              'AuPtAu_BA2_sum', 'AuPtAu_BA2_avg', 'AuPtAu_BA2_std', 'AuPtPd_BA2_sum', 'AuPtPd_BA2_avg', 'AuPtPd_BA2_std', 'AuPtPt_BA2_sum', 'AuPtPt_BA2_avg', 'AuPtPt_BA2_std', 
              'PdAuAu_BA2_sum', 'PdAuAu_BA2_avg', 'PdAuAu_BA2_std', 'PdAuPd_BA2_sum', 'PdAuPd_BA2_avg', 'PdAuPd_BA2_std', 'PdAuPt_BA2_sum', 'PdAuPt_BA2_avg', 'PdAuPt_BA2_std', 
              'PdPdAu_BA2_sum', 'PdPdAu_BA2_avg', 'PdPdAu_BA2_std', 'PdPdPd_BA2_sum', 'PdPdPd_BA2_avg', 'PdPdPd_BA2_std', 'PdPdPt_BA2_sum', 'PdPdPt_BA2_avg', 'PdPdPt_BA2_std', 
              'PdPtAu_BA2_sum', 'PdPtAu_BA2_avg', 'PdPtAu_BA2_std', 'PdPtPd_BA2_sum', 'PdPtPd_BA2_avg', 'PdPtPd_BA2_std', 'PdPtPt_BA2_sum', 'PdPtPt_BA2_avg', 'PdPtPt_BA2_std', 
             'PtAuAu_BA2_sum', 'PtAuAu_BA2_avg', 'PtAuAu_BA2_std', 'PtAuPd_BA2_sum', 'PtAuPd_BA2_avg', 'PtAuPd_BA2_std', 'PtAuPt_BA2_sum', 'PtAuPt_BA2_avg', 'PtAuPt_BA2_std', 
              'PtPdAu_BA2_sum', 'PtPdAu_BA2_avg', 'PtPdAu_BA2_std', 'PtPdPd_BA2_sum', 'PtPdPd_BA2_avg', 'PtPdPd_BA2_std', 'PtPdPt_BA2_sum', 'PtPdPt_BA2_avg', 'PtPdPt_BA2_std', 
              'PtPtAu_BA2_sum', 'PtPtAu_BA2_avg', 'PtPtAu_BA2_std', 'PtPtPd_BA2_sum', 'PtPtPd_BA2_avg', 'PtPtPd_BA2_std', 'PtPtPt_BA2_sum', 'PtPtPt_BA2_avg', 'PtPtPt_BA2_std', 
              
              'BTneg_avg', 'BTneg_std', 'BTpos_avg', 'BTpos_std', 
              
              'q6q6_T_avg', 'q6q6_B_avg', 'q6q6_S_avg', 
              'q6q6_T_0', 'q6q6_T_1', 'q6q6_T_2', 'q6q6_T_3', 'q6q6_T_4', 'q6q6_T_5', 'q6q6_T_6', 'q6q6_T_7', 'q6q6_T_8', 'q6q6_T_9', 'q6q6_T_10', 'q6q6_T_11', 'q6q6_T_12', 'q6q6_T_13', 'q6q6_T_14', 'q6q6_T_15', 'q6q6_T_16', 'q6q6_T_17', 'q6q6_T_18', 'q6q6_T_19', 'q6q6_T_20', 'q6q6_T_20+', 
              'q6q6_B_0', 'q6q6_B_1', 'q6q6_B_2', 'q6q6_B_3', 'q6q6_B_4', 'q6q6_B_5', 'q6q6_B_6', 'q6q6_B_7', 'q6q6_B_8', 'q6q6_B_9', 'q6q6_B_10', 'q6q6_B_11', 'q6q6_B_12', 'q6q6_B_13', 'q6q6_B_14', 'q6q6_B_15', 'q6q6_B_16', 'q6q6_B_17', 'q6q6_B_18', 'q6q6_B_19', 'q6q6_B_20', 'q6q6_B_20+', 
              'q6q6_S_0', 'q6q6_S_1', 'q6q6_S_2', 'q6q6_S_3', 'q6q6_S_4', 'q6q6_S_5', 'q6q6_S_6', 'q6q6_S_7', 'q6q6_S_8', 'q6q6_S_9', 'q6q6_S_10', 'q6q6_S_11', 'q6q6_S_12', 'q6q6_S_13', 'q6q6_S_14', 'q6q6_S_15', 'q6q6_S_16', 'q6q6_S_17', 'q6q6_S_18', 'q6q6_S_19', 'q6q6_S_20', 'q6q6_S_20+', 
             
              'FCC', 'HCP', 'ICOS', 'DECA', 
              'Surf_defects_mol', 'Surf_micros_mol', 'Surf_facets_mol', 
              'Formation_E']
featAddList = ['Volume', 
               'Curve_1-10', 'Curve_11-20', 'Curve_21-30', 'Curve_31-40', 'Curve_41-50', 'Curve_51-60', 'Curve_61-70', 'Curve_71-80', 'Curve_81-90', 'Curve_91-100', 'Curve_101-110', 'Curve_111-120', 'Curve_121-130', 'Curve_131-140', 'Curve_141-150', 'Curve_151-160', 'Curve_161-170', 'Curve_171-180',
               'Surf_defects_mol', 'Surf_micros_mol', 'Surf_facets_mol', 
               'Formation_E']
featRepeatList = ['Kinetic_E', 'Potential_E', 'Total_E']


def mergeReformatData(outputMD, verbose=True):
    # Concatenate MD output with features extracted from NCPac
    if verbose: print(f"    Concatenating CSV files for nanoparticle {outputMD[0]}...")
    df1 = pd.DataFrame(outputMD).T
    df1.columns = ['confID', 'T', 'P', 'Kinetic_E', 'Potential_E', 'Total_E']
    if os.path.getsize(f"{featSourcePath}/{outputMD[0]}.csv") == 0: 
        print(f"    *{outputMD[0]} is a BNP instead of TNP! Skipping...") 
        df = pd.DataFrame(columns=allHeaders)
        df.to_csv(f"{featEngPath}/{outputMD[0]}.csv", sep=',', header=True)
        return
    df2 = pd.read_csv(f"{featSourcePath}/{outputMD[0]}.csv", sep=',', header=1, index_col=None)  # usecols, low_memory
    df = pd.concat([df1, df2], axis='columns')
    df.set_index(keys='confID', inplace=True)
    headerList = allHeaders.copy()

    # Drop unused columns (TODO: check if all drops)
    # (*) means could potentially be included if figured out how to work on time-series-like data
    # (**) means could potentially be binned and included like the curvature features
    if verbose: print('    Dropping unused columns...')
    preDropColNum = len(df.columns)
    headerDropList = []
    curvStartColIdx, rdfStartColIdx, sfStartColIdx, ba1StartColIdx, btStartColIdxd, clStartColIdx = 22, 969, 1821, 2334, 12754, 13116
    for col in featAddList: headerList.remove(col)  # Remove the feature names that will be added later
    
    headerDropList.append(5)  # - Frame index
    # Add temporary columns for individual curvature degrees
    for i, curvColIdx in enumerate(range(curvStartColIdx, curvStartColIdx+180)): headerList.insert(curvColIdx, f"Curve_Deg_{i+1}")
    headerDropList.extend(list(range(rdfStartColIdx, rdfStartColIdx+852)))  # - Radial distribution function values (*)
    headerDropList.extend(list(range(sfStartColIdx, sfStartColIdx+513)))  # - Structure factors (*)
    # - Individual bond angle 1 and bond angle 2 degrees (**)
    ba1degCols = []
    skip6cols, skipColsCntDown = True, 6
    for (i, colIdx) in enumerate(range(ba1StartColIdx, ba1StartColIdx+10416)):
        if skipColsCntDown == 0: skip6cols = False
        if (colIdx-ba1StartColIdx) % 186 == 0:  # New elemental combination
            skipColsCntDown = 6
            skip6cols = True
        if skip6cols: skipColsCntDown -= 1
        else: ba1degCols.append(colIdx)
    headerDropList.extend(ba1degCols)
    headerDropList.extend(list(range(btStartColIdxd, btStartColIdxd+362)))  # - Individual bond torsion degrees (**)
    headerDropList.extend(list(range(clStartColIdx, clStartColIdx+20)))  # - Chain length
    df.drop(df.columns[headerDropList], axis=1, inplace=True)
    df = df[df.columns.drop(list(df.filter(regex='Type')))]  # - Types columns
    df = df.apply(pd.to_numeric, errors='coerce')  # Turning data numeric, be careful with 'coerce' option as there is a risk that blatant errors are omitted
    df.columns = headerList
    # if verbose: print(f"        Number of columns dropped: {preDropColNum - len(df.columns)}")
    
    # Add new columns
    if verbose: print('    Adding new columns...')
    headerAddList = ['Volume', 'Surf_defects_mol', 'Surf_micros_mol', 'Surf_facets_mol', 'Kinetic_E', 'Potential_E', 'Total_E', 'Formation_E']
    df['Volume'] = df.apply(lambda row: 3 / 4 * math.pi * row['R_avg']**3, axis=1)  # - Volume  # TODO: check formula with George!
    # - Curvature
    endVal, curvStartColIdx = 0, 20
    for i in range(1, 19):
        curvColName = f"Curve_{endVal+1}-{i*10}"
        df[curvColName] = df.iloc[:, curvStartColIdx:curvStartColIdx+10].sum(axis=1)
        endVal = i * 10
        curvStartColIdx += 10
        headerAddList.insert(i, curvColName)
    # Drop individual curvature degrees columns
    df.drop(df.columns[list(range(22, 22+180))], axis=1, inplace=True)
    for i in range(1, 1+180): headerList.remove(f"Curve_Deg_{i}")
    # - Surface characteristics concentration   TODO: confirm formula
    df['Surf_defects_mol'] = df.apply(lambda row: (row['MM_SCN_1'] + row['MM_SCN_2'] + row['MM_SCN_3']) / row['Volume'], axis=1)
    df['Surf_micros_mol'] = df.apply(lambda row: (row['MM_SCN_4'] + row['MM_SCN_5'] + row['MM_SCN_6'] +  row['MM_SCN_7']) / row['Volume'], axis=1)
    df['Surf_facets_mol'] = df.apply(lambda row: (row['MM_SCN_8'] + row['MM_SCN_9'] + row['MM_SCN_10']) / row['Volume'], axis=1)
    # - Formation/Cohesive energy
    E_Au, E_Pd, E_Pt = 1, 1, 1  # TODO: (unit?) To verify the source with George
    df['Formation_E'] = df.apply(lambda row: row['Total_E'] - (row['N_Au']*E_Au + row['N_Pt']*E_Pt + row['N_Pt']*E_Pt), axis=1)  # TODO: get formula

    # Reorder the columns
    if verbose: print('    Reordering columns...')
    headerList.insert(11, 'Volume')
    for i in range(1, 19): headerList.insert(22+i, headerAddList[i])
    for col in featRepeatList: headerList.remove(col)  # Remove the feature names that will be added again
    headerList.extend(headerAddList[-7:])
    df = df[headerList]
    df.to_csv(f"{featEngPath}/{outputMD[0]}.csv", sep=',', header=True)


def runMergeReformatParallel(verbose=False):
    if verbose: print(f"Merging information and reformating data in parallel...")
    outputMDs = []
    with open(MDoutFName, 'r') as f:
        for outputMD in csv.reader(f): 
            if not exists(f"{featEngPath}/{outputMD[0]}.csv"):
                outputMDs.append(outputMD)
    with Pool() as p: p.map(mergeReformatData, outputMDs)


def concatNPfeats(verbose=False):
    '''Fastest option to concatenate CSV files, almost 2 order of magnitudes faster than Pandas alternative'''
    if verbose: print(f"Concatenating processed feature CSV files...")
    featCSVs = sorted(os.listdir(featEngPath))
    featuresDF = pd.DataFrame(columns=allHeaders)
    with open(finalDataFName, 'wb') as fout:
        for (i, featCSV) in enumerate(featCSVs):
            if os.path.getsize(f"{featEngPath}/{featCSV}") == 0: 
                print(f"    *{featCSV} is a BNP instead of TNP! Skipping...") 
                continue
            with open(f"{featEngPath}/{featCSV}", 'rb') as f:
                if i != 0: next(f)  # Skip header
                fout.write(f.read())


if __name__ == '__main__':
    if not isdir(featEngPath): os.mkdir(featEngPath)
    runMergeReformatParallel(verbose=True)  # Parallel
    # concatNPfeats(verbose=True)  # Serial, separate step
