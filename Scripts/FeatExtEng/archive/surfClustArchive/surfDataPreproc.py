#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import os
import math
import sys
import warnings
# import modin.pandas as modpd
import numpy as np
import pandas as pd
import pickle

# General visualisation
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# import plotly.express as px
import seaborn as sns

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
# import umap.plot

# Data processing
from scipy.stats import probplot, shapiro, normaltest, anderson, skew, kurtosis
import sklearn
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer

# Detect polygon convexity
from sympy import Point, Polygon

# Figure setup
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set(color_codes=True)
sns.set(font_scale=1.2)
warnings.filterwarnings("ignore")
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
# rcParams['figure.figsize'] = [8, 5]
# rcParams['figure.dpi'] = 80
rcParams['figure.autolayout'] = True
rcParams['font.style'] = 'normal'
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10

# Print version for reproducibility
print("numpy version {0}".format(np.__version__))
print("pandas version {0}".format(pd.__version__))
print("sklearn version {0}".format(sklearn.__version__))

# Initialise variables and load data
npName, startID, endID = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
testCases = list(str(i).zfill(4) for i in range(2501))
randomSeed = 42
corrMethod = "spearman"
varThreshs, corrThreshs = [0.01, 0.03, 0.05], [0.90, 0.95, 0.99]
# reinitialise = True
# if reinitialise:
#     dfScaledNoLVHCs = np.array([[[None] * len(testCases)] * len(corrThreshs)] * len(varThreshs))
#     dfNoLVHCs = np.array([[[None] * len(testCases)] * len(corrThreshs)] * len(varThreshs))
#     corrMatNoLVHCs = np.array([[[None] * len(testCases)] * len(corrThreshs)] * len(varThreshs)) 
with open(f"pickleFiles/biFeat_dfAlls_{npName}.pickle", 'rb') as f: 
    biFeat_dfAlls = pickle.load(f).copy()


def dfMassage(df, interactive=True):
    """
    Handle missing values, drop duplicates in input DataFrame, one-hot encode certain features, and turn columns datatype into numeric
    input:
        df = input DataFrame
        interactive = Boolean indicator to decide usage of function (display/print) for DataFrame inspection
    output:
        dfNew = processed DataFrame
    """
    # Handle missing values
    if df.isna().any().any():
        print("Missing entries exist!")
        missingNum = df.isna().sum()
        missingNum[missingNum > 0].sort_values(ascending=False)
        print("Missing rows in each column/feature:\n", missingNum)
        df.dropna(axis=1, how='any', thresh=None, inplace=True)
        # df.replace({"csm": {np.nan: df["csm"].max()}, 
        #                "molRMS": {np.nan: df["molRMS"].max()}, 
        #                "Ixx": {np.nan: df["Ixx"].mean()}, # Check if average is appropriate
        #                "Iyy": {np.nan: df["Iyy"].mean()}, 
        #                "Izz": {np.nan: df["Izz"].mean()}, 
        #                "E": {0.0: 1.0}}, 
        #                inplace=True, regex=False, limit=None, method="pad")
    df.replace(to_replace=np.inf, value=1000, inplace=True, regex=False, limit=None, method="pad")  # Check if 1000 is appropriate

    # Drop duplicates
    if df.duplicated().any():
        print("Duplicate entries exists!")
        print("Number of rows before dropping duplicates: ", len(df))
        df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False);
        print("Number of rows after dropping duplicates: ", len(df))
        
    # One-hot encode features
    dummyEle = pd.get_dummies(df['ele'], prefix='ele')
    dfNew = pd.merge(left=df, right=dummyEle, left_index=True, right_index=True)
    
    # To numeric
    if ('Ixx' in df.columns) and ('Iyy' in df.columns) and ('Izz' in df.columns) and ('degenDeg' in df.columns):
        dfNew = dfNew.astype({"x": float, "y": float, "z": float, 
                        "xNorm": float, "yNorm": float, "zNorm": float, 
                        "Ixx": float, "Iyy": float, "Izz": float, 
                        "degenDeg": float})
    
    print("\nMissing values handled, duplicates dropped, one-hot encoded categorical features, feature columns numericised:")
    return dfNew


def dfDropUnusedCols(df):
    """
    Drop unused feature columns (could vary depending on datasets e.g. mono/multi-metallic)
    input:
        df = input DataFrame
    output:
        dfNew = processed DataFrame
    """
    # "x", "y", "z" redundant (included normalised columns)
    # "pg" could only be one-hot encoded when all possibilities are recorded
    # "csm", "molRMS" undefined when "pg" is missing
    # inclusion of "E", "C8", etc. makes clustering difficult (binary features become dominant)
    surfIdx = list(df.columns).index("surf")
    
    if 'pg' in df.columns:
        pgIdx = list(df.columns).index("pg")
        cols2drop = ["x", "y", "z", "ele", "ele_Au", "ele_Pt"] + list(df.columns)[pgIdx:surfIdx]  # Could simply leave "ele" to removal of features with low variance
    elif 'degenDeg' in df.columns:
        degenDegIdx = list(df.columns).index("degenDeg")
        cols2drop = ["x", "y", "z", "ele", "ele_Au", "ele_Pt"] + list(df.columns)[degenDegIdx+1:surfIdx]  # Could simply leave "ele" to removal of features with low variance
    else:
        cols2drop = ["x", "y", "z", "ele", "ele_Au", "ele_Pt"] + list(df.columns)[list(df.columns).index("centParam")+1:surfIdx]  # Could simply leave "ele" to removal of features with low variance
    
    dfNew = df.drop(labels=cols2drop, axis=1, index=None, columns=None, level=None, inplace=False, errors="raise")
    dfNew.rename(columns={'xNorm': 'x', 'yNorm': 'y', 'zNorm': 'z'}, inplace=True)
    return dfNew


# Data-Scaling
def mapGauss(df):
    """
    Map distribution of features to Gaussian distributions via power tranform (parametric, monotonic transformations)
        Box-Cox tranformation (only applicable to strictly positive data)
        Yeo-Johnson transformation
        Quantile transformation (do not transform before train-test split, leaks information otherwise)
    input: 
        df = DataFrame with columns being features
    output:
        dfYJTMap = DataFrame after Yeo-Johnson transformation
        dfQTMap = DataFrame after quantile transformation
    """ 
    YJT = PowerTransformer(method='yeo-johnson', standardize=False, copy=True)
    QT = QuantileTransformer(n_quantiles=1000, output_distribution='normal', ignore_implicit_zeros=False, subsample=100000, random_state=randomSeed, copy=True)
    YJTmappedArr = YJT.fit_transform(df)
    QTmappedArr = QT.fit_transform(df)
    dfYJTMap = pd.concat(objs=[pd.DataFrame(YJTmappedArr, index=df.index, columns=df.columns)], axis=1)
    dfQTMap = pd.concat(objs=[pd.DataFrame(QTmappedArr, index=df.index, columns=df.columns)], axis=1)
    return dfYJTMap, dfQTMap


def normalTest(df, alpha=0.05, verbose=False):
    """
    Assess whether the normality assumption holds for each feature.
        Shapiro-Wilk test quantifies how likely the data is drawn from Gaussian distribution. (W accurate for N > 5000 but not p)
        D'Agostino's K^2 test calculates summary statistics from data to quantify deviation from Gaussian distribution (statistics = sum of square of skewtest's and kurtosistest's z-score)
        Anderson-Darling test evaluates whether a sample comes from one of among many known samples
    input: 
        df = DataFrame with columns being features
        alpha = significance level
        verbose = Boolean indicator for printing output
    output:
        vioList = list of features that violate the normality assumption
        normList = list of features that satisfy the normality assumption
    """ 
    vioList, normList = [], []
    for feat in df.columns:
        
        # Statistical checks (Quantification)
        if verbose: print("\nFeature: {0}".format(feat))
        xArr = df[feat]
        Wstat, WpVal = shapiro(x=xArr)
        Dstat, DpVal = normaltest(a=xArr, axis=None, nan_policy='propagate')
        Astat, AcritVals, AsigLev = anderson(x=xArr, dist='norm')
        if WpVal < alpha or DpVal < alpha or Astat > AcritVals.any():
            if verbose:
                print("  Shapiro-Wilk Test p-value: {0:.3f}\n  D'Agostino's K^2 Test p-value: {1:.3f}".format(WpVal, DpVal))
                print("  Anderson-Barling Test statistics: {0:.3f}".format(Astat))
                for (i, sigLev) in enumerate(AsigLev): 
                    print("    Significance level: {0:.3f}, Critical value: {1:.3f}".format(sigLev, AcritVals[i]))
            if Astat > AcritVals.all():
                vioList.append(feat)
                if verbose: 
                    print("Statistically significant at all significance level, normality assumption violated!")
            else:
                normList.append(feat)
                if verbose: 
                    print("Hypothesis couldn't be rejected!")
    
            # Visual checks (Qualification)
            if verbose: 
                plt.figure(figsize=(12, 5));
                plt.subplot(121);
                sns.histplot(data=df, x=feat, kde=True);
                plt.subplot(122);
                probplot(x=xArr, dist="norm", fit=True, plot=plt, rvalue=True);
                plt.show();
    
    print("  Features checked: ", list(df.columns))
    print("  Normality violated by: ", vioList)
    return vioList, normList


def dfNorm(df, alpha=0.05, verbose=False, interactive=True):
    """
    Scale the dataset after checking normality assumption
    input:
        df = input DataFrame
        alpha = significance level
        verbose = Boolean indicator for output printing
        interactive = Boolean indicator to decide usage of function (display/print) for DataFrame inspection
    output:
        dfScaled = scaled DataFrame
    """
    print("\nCheck normality assumptions prior to transfomation:")
    vioList, normList = normalTest(df, alpha=alpha, verbose=False)
    
    # Transform the distributions of numerical features to Gaussian distributions and perform the same check
    dfYJTMap, dfQTMap = mapGauss(df)
    print("\nCheck normality assumptions after Yeo-Johnson transformation:")
    vioList1, normList1 = normalTest(dfYJTMap, alpha=alpha, verbose=False)
    print("\nCheck normality assumptions after quantile tranformation:")
    vioList2, normList2 = normalTest(dfQTMap, alpha=alpha, verbose=False)
    
    # Normalisation/Standardisation/Robust scaling
    # Could be included for supervised learning pipeline
    scaler = "minMax" if len(vioList1) > 0 or len(vioList2) > 0 else "stand"  # Robust scaling not considered for regularly ordered nanoparticles
    dfScaled = dfScale(df, catCols=["surf"], scaler=scaler)
    print("\nScaling the data, scaler: {0}".format(scaler))
    return dfScaled


def dfScale(df, catCols=None, scaler="minMax"):
    """
    Scale input feature DataFrame using various sklearn scalers
    input:
        df = DataFrame with columns being features
        catCols = list of categorical features, not to be scaled
        scaler = type of sklearn scaler to use
    output:
        Xscaled = scaled DataFrames
    """
    X4keep = df[catCols] if catCols else None
    X4scale = df.drop(X4keep.columns, axis=1, inplace=False) if catCols else df
    if scaler == "minMax":
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(X4scale)
    # elif scaler == "stand":
    #     scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X4stand)
    elif scaler == "robust":
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True).fit(X4scale)
    else:
        raise("\nScaler specified unknown!")
    arrXscaled = scaler.transform(X4scale)
    Xscaled = pd.concat(objs=[pd.DataFrame(arrXscaled, index=df.index, columns=X4scale.columns), X4keep], axis=1)
    Xscaled.index.name = "Index"
    return Xscaled


# Feature selection
def getLowVarCols(df, skipCols=None, varThresh=0.0, autoRemove=False):
    """
    Wrapper for sklearn VarianceThreshold.
    input:
        df = DataFrame with columns being features
        skipCols = columns to be skipped (skipCols is of list type)
        varThresh = low variance threshold for features to be detected
        autoRemove = Boolean indicator for automatic removal of low variance columns
    output:
        df = DataFrame with low variance features removed
        lowVarCols = list of low variance features
    """
    print("\n  Finding features with low-variance, threshold: {0}".format(varThresh))
    try:
        allCols = df.columns
        if skipCols:
            remainCols = allCols.drop(skipCols)
            maxIdx = len(remainCols) - 1
            skippedIdx = [allCols.get_loc(col) for col in skipCols] #store the index of each skipped column

            # Adjust insert location by the number of columns removed (for non-zero insertion locations) to keep relative locations intact
            # This can help cope with the case when the column name in skipCols isn't in accordance with the original order in the dataFrame
            for idx, item in enumerate(skippedIdx):
                if item > maxIdx:
                    diff = item - maxIdx
                    skippedIdx[idx] -= diff
                if item == maxIdx:
                    diff = item - len(skipCols)
                    skippedIdx[idx] -= diff
                if idx == 0:
                    skippedIdx[idx] = item
            skippedVals = df.iloc[:, skippedIdx].values
        else:
            remainCols = allCols

        X = df.loc[:, remainCols].values
        vt = VarianceThreshold(threshold=varThresh)
        vt.fit(X)
        keepColsIdxs = vt.get_support(indices=True)
        keepCols = [remainCols[idx] for idx, _ in enumerate(remainCols) if idx in keepColsIdxs]
        lowVarCols = list(np.setdiff1d(remainCols, keepCols))
        print("    Found {0} low-variance columns.".format(len(lowVarCols)))

        if autoRemove:
            print("    Removing low-variance features...")
            X_removed = vt.transform(X)
            print("    Reassembling the dataframe (with low-variance features removed)...")
            df = pd.DataFrame(data=X_removed, columns=keepCols)
            if skipCols:
                for (i, index) in enumerate(skippedIdx): df.insert(loc=index, column=skipCols[i], value=skippedVals[:, i])
            print("    Succesfully removed low-variance columns: {0}.".format(lowVarCols))
        else:
            print("    No changes have been made to the dataframe.")
    except Exception as e:
        print(e)
        print("    Could not remove low-variance features. Something went wrong.")
    return df, lowVarCols



def getHighCorCols(df, corrThresh=0.95, method="spearman"):
    """
    Compute correlation matrix using pandas
    input: 
        df = input DataFrame with columns being features
        corrThresh = threshold to identify highly correlated features
        method = method to compute DataFrame correlation
    output: 
        corrMat = correlation matrix of all features in input DataFrame
        highCorCols = tuples of highly correlated features
    """ 
    print("\n  Finding features highly-correlated with each other, threshold: {0}".format(corrThresh))
    corrMat = df.corr(method=method, min_periods=1)
    corrMatUpper = corrMat.where(np.triu(np.ones(corrMat.shape), k=1).astype(np.bool))  # Select upper triangle of correlation matrix
    highCorCols = [(row, col) for col in corrMatUpper.columns for row in corrMatUpper.index if corrMatUpper.loc[row, col] > corrThresh]
    print("    Highly correlated columns: {0}".format(highCorCols))
    return corrMat, highCorCols


def plotCorrMat(corrMat, figSize=(8, 8), figName=None):
    """
    Wrapper for sklearn VarianceThreshold.
    input:
        corrMat = correlation matrix of all features in input DataFrame
        figSize = size of figure
        figName = path to save figure
    """
    cmap = sns.diverging_palette(h_neg=210, h_pos=350, s=90, l=50, as_cmap=True)
    cg = sns.clustermap(data=corrMat.abs().mul(100).astype(float), cmap='Blues', metric='correlation',  figsize=figSize)
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0);
    if figName: plt.savefig(figName, dpi=300, bbox_inches='tight')


def autoDrop(highCorCols, verbose=True):
    """
    Automate the selection of highly-correlated features to be dropped. 
    Rank given to each feature is based on its degree of utility/ease of interpretation if it turns out to be correlated with target labels
    'x', 'y', 'z' not very useful even if found to be important
    'avg' values of bond geometries are more useful than 'num' values
    'max', 'min' of bond geometries are hard to control experimentally
    'angParam', 'centParam' are easier to interpret than 'entroParam'
    'disorder' parameters are easier to interpret than 'q' Steinhardt's parameters
    averaged 'q' parameters are more robust to thermal fluctuations, hence more useful than pure 'q' parameters
    q6 > q4 > q2 == q8 == q10 == q12 in usefulness based on literature, thus 'disorder' parameters follow same sequence
    input:
        highCorCols = list of tuples of highly-correlated features
        verbose = Boolean indicator for output printing
    output: 
        cols2drop = list of features to be dropped
    """
    utilityRanking = {'x': 10, 'y': 10, 'z': 10, 'rad': 0, 
                     'blavg': 2, 'blmax': 8, 'blmin': 8, 'blnum': 3, 
                     'bl1avg': 4, 'bl1max': 8.5, 'bl1min': 8.5, 'bl1num': 5, 
                     'bl2avg': 4, 'bl2max': 8.5, 'bl2min': 8.5, 'bl2num': 5, 
                     'bl3avg': 4, 'bl3max': 8.5, 'bl3min': 8.5, 'bl3num': 5, 
                     'bl4avg': 4, 'bl4max': 8.5, 'bl4min': 8.5, 'bl4num': 5,
                     'ba1avg': 2, 'ba1max': 8, 'ba1min': 8, 'ba1num': 3,
                     'ba11avg': 4, 'ba11max': 8.5, 'ba11min': 8.5, 'ba11num': 5, 
                     'ba12avg': 4, 'ba12max': 8.5, 'ba12min': 8.5, 'ba12num': 5,
                     'ba13avg': 4, 'ba13max': 8.5, 'ba13min': 8.5, 'ba13num': 5,
                     'ba14avg': 4, 'ba14max': 8.5, 'ba14min': 8.5, 'ba14num': 5,
                     'ba15avg': 4, 'ba15max': 8.5, 'ba15min': 8.5, 'ba15num': 5, 
                     'ba16avg': 4, 'ba16max': 8.5, 'ba16min': 8.5, 'ba16num': 5,
                     'ba17avg': 4, 'ba17max': 8.5, 'ba17min': 8.5, 'ba17num': 5,
                     'ba18avg': 4, 'ba18max': 8.5, 'ba18min': 8.5, 'ba18num': 5, ### ba1 feature
                     'ba2avg': 2, 'ba2max': 8, 'ba2min': 8, 'ba2num': 3,
                     'ba21avg':4, 'ba21max': 8.5, 'ba21min': 8.5, 'ba21num': 5, 
                     'ba22avg':4, 'ba22max': 8.5, 'ba22min': 8.5, 'ba22num': 5,
                     'ba23avg':4, 'ba23max': 8.5, 'ba23min': 8.5, 'ba23num': 5,
                     'ba24avg':4, 'ba24max': 8.5, 'ba24min': 8.5, 'ba24num': 5,
                     'ba25avg':4, 'ba25max': 8.5, 'ba25min': 8.5, 'ba25num': 5, 
                     'ba26avg':4, 'ba26max': 8.5, 'ba26min': 8.5, 'ba26num': 5,
                     'ba27avg':4, 'ba27max': 8.5, 'ba27min': 8.5, 'ba27num': 5,
                     'ba28avg':4, 'ba28max': 8.5, 'ba28min': 8.5, 'ba28num': 5, ### ba2 feature
                     'btposavg': 2, 'btposmax': 8, 'btposmin': 8, 'btposnum': 3, 
                     'btnegavg': 2, 'btnegmax': 8, 'btnegmin': 8, 'btnegnum': 3, 
                     'cn': 1, 'gcn': 0, 'scn': 3, 'sgcn': 3, 'q6q6': 2, 
                     'Ixx': 5, 'Iyy': 5, 'Izz': 5, 'degenDeg': 6, 
                     'angParam': 4, 'centParam': 4, 'entroParam': 5, 'entroAvgParam': 5.5, 
                     'chi1': 6, 'chi2': 6, 'chi3': 6, 'chi4': 6, 'chi5': 6, 'chi6': 6, 'chi7': 6, 'chi8': 6, 'chi9': 6, 
                     'q2': 5.7, 'q4': 5.6, 'q6': 5.5, 'q8': 5.7, 'q10': 5.7, 'q12': 5.7, 
                     'q2avg': 5.2, 'q4avg': 5.1, 'q6avg': 5, 'q8avg': 5.2, 'q10avg': 5.2, 'q12avg': 5.2, 
                     'disord2': 4.7, 'disord4': 4.6, 'disord6': 4.5, 'disord8': 4.7, 'disord10': 4.7, 'disord12': 4.7, 
                     'disordAvg2': 4.2, 'disordAvg4': 4.1, 'disordAvg6': 4, 'disordAvg8': 4.2, 'disordAvg10': 4.2, 'disordAvg12': 4.2
                     }  # Lower score = Higher rank

    # occurCount = Counter(list(sum(highCorCols, ())))
    print("\n    Sorting all highly-correlated feature pairs based on their minimum and total utility rankings.")
    highCorColsProps = []
    for (col1, col2) in highCorCols:
        rank1, rank2 = utilityRanking[col1], utilityRanking[col2]
        highCorColsProps.append((min(rank1, rank2), rank1 + rank2))
    sortedIdx = sorted(range(len(highCorColsProps)), key=lambda i: highCorColsProps[i])
    highCorCols, highCorColsProps = [highCorCols[i] for i in sortedIdx], [highCorColsProps[i] for i in sortedIdx]
    
    print("\n    Removing one of each highly-correlated feature pairs.")
    cols2drop = []
    for (i, (col1, col2)) in enumerate(highCorCols):
        if verbose: print("      Feature pairs: {0} {1}".format(col1, col2))
        if col1 in cols2drop or col2 in cols2drop:
            if verbose: 
                print("        One of the features is dropped, skip this pair.\n")
            continue
        elif utilityRanking[col1] > utilityRanking[col2]:
            print("        {0} has lower utility score compared to {1}".format(col1, col2))
            cols2drop.append(col1)
        else:
            print("        {0} has lower utility score compared to {1}".format(col2, col1))
            cols2drop.append(col2)
    print("    Feature columns to drop: {0}".format(cols2drop))
    return cols2drop


def varCorrDropCols(X, varThresh=0.01, corrThresh=0.95, figName=None, verbose=True):
    """
    Remove features with low variance and one of the highly-correlated feature pairs
    input:
        X = input scaled DataFrame with each column being feature
        varThresh = threshold below which feature is removed
        corrThresh = threshold above which one of each pair of correlated features is removed
        verbose = Boolean indicator for output printing
    output:
        XNoLVHC = DataFrame with the undesired features removed
        corrMatNoLVHC = computed correlated matrix
    """
    # Remove columns with low variance
    XNoLV, lowVarCols = getLowVarCols(df=X, skipCols=None, varThresh=varThresh, autoRemove=True)  # Using min-max scaled for now
    
    # Remove one of the feature columns that are highly correlated with each other
    corrMatNoLV, highCorCols1 = getHighCorCols(df=XNoLV, corrThresh=corrThresh, method=corrMethod)
    cols2drop = autoDrop(highCorCols1, verbose=verbose)
    XNoLVHC = XNoLV.drop(labels=cols2drop, axis=1, index=None, columns=None, level=None, inplace=False, errors="raise")
    corrMatNoLVHC, highCorCols2 = getHighCorCols(df=XNoLVHC, corrThresh=corrThresh, method=corrMethod)
    plotCorrMat(corrMat=corrMatNoLVHC, figSize=(8, 8), figName=figName)
    return XNoLVHC, corrMatNoLVHC


from sklearn.metrics import mutual_info_score
# mutual_info_score(data["sex"], data["pclass"])


def calRelevance(index, S):
    rel = 0
    # print(S)
    ele = S.iloc[:,index]
    for i in range(S.shape[1]):
        if not i ==index:
            rel += mutual_info_score(ele, S.iloc[:,i])
    return rel

def calRedundancy(fea, S):   
    red = 0
    for i in range(S.shape[1]):
        red += mutual_info_score(fea, S.iloc[:,i])
    return red


def selectNewFea(D,S):
    PotentialGainList = []
    for i in range(D.shape[1]):
        fea = D.iloc[:,i]
        red = calRedundancy(fea,S)
        rel = calRelevance(i,D)
        relSum = 0
        for i in range(D.shape[1]):
            rel = calRelevance(i,D)
            relSum+=rel
        PotentialGain = (rel + relSum) / red
        PotentialGainList.append(PotentialGain)
    maxPGVal = max(PotentialGainList)
    maxPGIdx = PotentialGainList.index(maxPGVal)
    selected = pd.DataFrame(D.iloc[:,maxPGIdx])
    S[selected.columns] = selected
    D = D.drop(D.columns[maxPGIdx],axis=1)

    GainPenaltyFac = calGainPenaltyFac(D,S)
    # print(GainPenaltyFac)
    if GainPenaltyFac >= 1:
        selectNewFea(D,S)
    return D,S,GainPenaltyFac


def calGainPenaltyFac(D,S):
    relSum = 0
    for i in range(D.shape[1]):
            rel = calRelevance(i,D)
            relSum+=rel 
    redSum = 0
    for i in range(S.shape[1]):
            fea = S.iloc[:,i]
            red = calRedundancy(fea,S)
            redSum+=red 
    GainPenaltyFac = relSum/redSum
    return GainPenaltyFac
    

def sortFeaInfoGain(D):
    Selected = []
    relList = []
    for  i in range(D.shape[1]):
        rel = calRelevance(i,D)
        relList.append(rel)
    maxRelVal = max(relList)
    maxRelIdx = relList.index(maxRelVal)
    
    Selected1 = D.iloc[:,maxRelIdx]
    S = pd.DataFrame(Selected1)
    # print(S)
    D = D.drop(D.columns[maxRelIdx],axis=1)

    newD, newS, GainPenaltyFac = selectNewFea(D,S)            
    return newS


# Determine feature number based on the information gain valus
def get_data_radiant(data):
    return np.arctan2(data[:, 1].max() - data[:, 1].min(), data[:, 0].max() - data[:, 0].min())


def find_elbow(data):
    theta = get_data_radiant(data)
    # Make rotation matrix
    co = np.cos(theta)
    si = np.sin(theta)
    rotation_matrix = np.array(((co, -si), (si, co)))

    # Rotate data vector
    rotated_vector = data.dot(rotation_matrix)

    # Return index of elbow
    return np.where(rotated_vector == rotated_vector[:, 1].min())[0][0]


from sklearn.neighbors import NearestNeighbors
import scipy as sp


def get_laplacian_scored(X,k=5,n=5):
    ''' Gets the top k parameters as per locality preservation adherence
        k: top k locality/nn graph preserving features
        n: nearest neighbor graph hyperparameter
        
        sorted_idx[:k]: indices of top k features/columns of X (X is numeric and does not have Nulls or NaNs)
        
        return: selected X with top-k features
    ''' 
    OriginalFeaList = X.columns
    if  type(X) != np.ndarray:
        X = np.asarray(X)
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(X)
    G = nbrs.kneighbors()[1]
    S = []
    for i in range(len(G)):
        s = []
        for j in range(len(G)):
            if j in G[i,:]:
                s.append(np.exp(-1*sp.spatial.distance.euclidean(u=X[i],v=X[j])))
            else:
                s.append(0)
        S.append(s)
    S = np.array(S)
    d1 = np.ones(X.shape[0])
    factor_space = np.linalg.multi_dot([S,d1.transpose()])
    D = np.multiply(np.identity(X.shape[0]),factor_space)
    L = D - S
    f_n = []
    for j in range(X.shape[1]):
        f = X[:,j]
        f_n.append(f - np.multiply(np.linalg.multi_dot([np.linalg.multi_dot([f.transpose(),D]),d1])/np.linalg.multi_dot([np.linalg.multi_dot([d1.transpose(),D]),d1]),d1))
    f_n = np.array(f_n)
    lr = []
    for j in range(f_n.shape[0]):
        lr.append(np.linalg.multi_dot([np.linalg.multi_dot([f_n[j].transpose(),L]),f_n[j]])/np.linalg.multi_dot([np.linalg.multi_dot([f_n[j].transpose(),D]),f_n[j]]))
    sorted_idx = np.argsort(lr)
    XSelected = X[:,sorted_idx[:k]]
    # print(XSelected)
    selectedFeaList = OriginalFeaList[sorted_idx[:k]]
    
    # convert back to pandas dataframe with header
    selectedFeaList = np.reshape(selectedFeaList,(-1,len(selectedFeaList)))
    XselectedDF = pd.DataFrame(np.concatenate([selectedFeaList, XSelected],axis=0))
    header_row = XselectedDF.iloc[0]
    XselectedDFL = pd.DataFrame(XselectedDF.values[1:], columns=header_row)

    return XselectedDFL


# Normalised laplacian (opt in this one)
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def cal_lap_score(features, D, L):
    # if (np.sum(D))==0:
        # print("debug")
    if math.isinf(np.sum(D)):
        features_ = features
    else:
        features_ = features - np.sum((features @ D) / np.sum(D))
    if ((features_ @ D @ features_)==0) or (math.isinf(features_ @ D @ features_)):
        L_score = 0
    # if math.isinf(features_ @ D @ features_):
    #     L_score = 0
    else: L_score = (features_ @ L @ features_) / (features_ @ D @ features_)
    return L_score


def get_k_nearest(dist, k, sample_index):
    # dist is zero means it is the sample itself
    return sorted(
        range(len(dist)),
        key=lambda i: dist[i] if i != sample_index else np.inf
    )[:k] + [sample_index]


def laplacian_score(df_arr, label=None, **kwargs):
    kwargs.setdefault("k_nearest", 5)
    featureNameList = df_arr.columns
    '''
    Construct distance matrix, dist_matrix, using euclidean distance
    '''
    distances = pdist(df_arr, metric='euclidean')
    dist_matrix = squareform(distances)
    del distances
    '''
    Determine the edge of each sample pairs by k nearest neighbor
    '''
    edge_sparse_matrix = pd.DataFrame(
        np.zeros((df_arr.shape[0], df_arr.shape[0])),
        dtype=int
    )
    if label is None:
        for row_id, row in enumerate(dist_matrix):
            k_nearest_id = get_k_nearest(row, kwargs["k_nearest"], row_id)
            edge_sparse_matrix.iloc[k_nearest_id, k_nearest_id] = 1
    else:
        label = np.array(label)
        unique_label = np.unique(label)
        for i in unique_label:
            group_index = np.where(label == i)[0]
            edge_sparse_matrix.iloc[group_index, group_index] = 1
    S = dist_matrix * edge_sparse_matrix
    del dist_matrix, edge_sparse_matrix
    '''
    Calculate the Laplacian graph L
    '''
    D = np.diag(S.sum(axis=1))
    L = D - S
    del S
    '''
    Minimize the Laplacian score
    '''
    features_lap_score = np.apply_along_axis(
        func1d=lambda f: cal_lap_score(f, D, L), axis=0, arr=df_arr
    )
    scoreNP = np.array(features_lap_score)
    nameNP = np.array(featureNameList)
    
    scoreAllNP = np.column_stack([nameNP, scoreNP])
    sortedScore = sorted(scoreAllNP,key=lambda x: x[1])
    # print(sortedScore)
    sortedNames = np.array(sortedScore)[:,0]
    sortedScores = np.array(sortedScore)[:,1]
    plt.scatter(sortedNames, sortedScores, c='DarkBlue')
    
    scoreNPIndex = list(range(len(scoreNP)))
    scoreForElbow = np.column_stack([scoreNPIndex, scoreNP])
    elbowIndex = find_elbow(scoreForElbow)
    selectedFea = df_arr[sortedNames[:elbowIndex+1]]
    
    return selectedFea


def expVarPCA(caseID, varThreshIdx=0, corrThreshIdx=1):
    """
    Calculate and plot the variance explained by PCA components
    input:
        caseID = ID index of nanoparticle of interest
        varThresh = threshold below which feature is removed
        corrThresh = threshold above which one of each pair of correlated features is removed
        * Refer to sklearn package for other input variables
    """
    pca = PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=randomSeed)
    redPCA = pca.fit_transform(dfScaledNoLVHCs[varThreshIdx][corrThreshIdx][caseID].iloc[:, :-1])
    # print("\nExplained Variance Ratio:\n", pca.explained_variance_ratio_)
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance');
    plt.title("Explained Variance Ratio of the Fitted PC Vector, {0} Del Var < {1} Corr > {2}".format(testCases[caseID], 
                                                                                                              varThreshs[varThreshIdx], 
                                                                                                              corrThreshs[corrThreshIdx]));
    plt.savefig("figures/PCAexpVar/{0}_{1}PCA_V{2}C{3}.png".format(npName,testCases[caseID], 
                                                               int(varThreshs[varThreshIdx]*100), 
                                                               int(corrThreshs[corrThreshIdx]*100)), 
                dpi=300, bbox_inches='tight');


def runCase(caseID, verbose=False, interactive=False):
    """
    Run the pipeline for a nanoparticle
    input:
        caseID = ID index of nanoparticle of interest
        verbose = Boolean indicator of whether text outputs should be elaborated 
        interactive = Boolean indicator of whether DataFrames should be displayed (to be investigated in Notebook) or printed (to be stored in .txt file)
    output:
        dfNoLVHC = unscaled DataFrame without highly-correlated and low variance features
        dfScaledNoLVHC = scaled DataFrame without highly-correlated and low variance features
        corrMatNoLVHC = computed correlation matrix
    """
    
    print("Test case: {0}".format(testCases[caseID]))
    # dfDesc(df=biFeat_dfAlls[caseID].copy(), interactive=interactive)
    dfCleaned = dfMassage(df=biFeat_dfAlls[caseID].copy())
    dfDropped = dfDropUnusedCols(df=dfCleaned)
    del dfCleaned
    dfScaled = dfNorm(df=dfDropped, alpha=0.05, verbose=verbose, interactive=interactive)
    Xscaled, y = dfScaled.iloc[:, :-1], dfScaled.loc[:, "surf"]
    del dfScaled
    
    for (i, varThresh) in enumerate(varThreshs[0:1]):
        for (j, corrThresh) in enumerate(corrThreshs[1:2]):
            print("\nRemove features with varThresh: {0}, corrThresh: {1}".format(varThresh, corrThresh))
            varCorrDropColsFigName = "figures/corrMat/{0}_{1}cmatV{2}C{3}.png".format(npName, testCases[caseID], int(varThresh*100), int(corrThresh*100)) 
             
            XScaledNoLVHC, corrMatNoLVHC = varCorrDropCols(X=Xscaled, varThresh=varThresh, corrThresh=corrThresh, figName=varCorrDropColsFigName, verbose=verbose)
            # kvalue = determineFeatureNumber(X=XScaledNoLVHC)
            # kvalue = round(len(XScaledNoLVHC.columns)*0.8)
            XSelected = pd.concat(objs=[laplacian_score(XScaledNoLVHC,None), y], axis=1, join="inner", ignore_index=False)
            
            # we need to return dfScaledNoLVHCs dfNoLVHCs corrMatNoLVHCs outside this loop in the multiprocess pipeline
            # we return XSelected dfNoLVHC corrMatNoLVHC here
            dfNoLVHC = dfDropped[pd.DataFrame(XSelected).columns]
            # dfScaledNoLVHCs[i][j][caseID] = XSelected
            # dfNoLVHCs[i][j][caseID] = dfNoLVHC
            # corrMatNoLVHCs[i][j][caseID] = corrMatNoLVHC
    del dfDropped
    del Xscaled
    if not os.path.isdir(f"pickleFiles/{npName}"): os.mkdir(f"pickleFiles/{npName}")
    with open(f"pickleFiles/{npName}/biFeat_dfScaledNoLVHCs_laplacianScore_{npName}_{caseID}.pickle", 'wb') as f: pickle.dump(XSelected, f)
    with open(f"pickleFiles/{npName}/biFeat_dfNoLVHCs_laplacianScore_{npName}_{caseID}.pickle", 'wb') as f: pickle.dump(dfNoLVHC, f)
    return 'Done'


from multiprocessing import Pool
if __name__ == '__main__':
    caseIDs = list(range(startID, endID))
    remainingCaseIDs = []
    for caseID in caseIDs: 
        if not os.path.exists(f"pickleFiles/{npName}/biFeat_dfNoLVHCs_laplacianScore_{npName}_{caseID}.pickle"): remainingCaseIDs.append(caseID)
    with Pool() as p:
        print(p.map(runCase, remainingCaseIDs))
    #for caseID in remainingCaseIDs: 
    #    runCase(caseID)
    #for (i, varThresh) in enumerate(varThreshs):
        #for (j, corrThresh) in enumerate(corrThreshs):
            #for k in range(len(caseIDs)):
                #XSelected, dfNoLVHC, corrMatNoLVHC = parallelRunResults[k]
                #dfScaledNoLVHCs[i][j][k] = XSelected
                #dfNoLVHCs[i][j][k] = dfNoLVHC
                #corrMatNoLVHCs[i][j][k] = corrMatNoLVHC
