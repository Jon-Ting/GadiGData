def calcFracDim(testCase, repMethod='NPsurf', showPlot=False):
    if repMethod == 'atomCentre': fileName = 'ATOM_CENTRE'
    elif repMethod == 'atomBall' or repMethod == 'sAtomBall': fileName = 'SATOM_BALL'
    elif repMethod == 'NPsurf': fileName = 'NP_SURF'
    else: raise Exception("repMethod unknown!")
    
    with open(f'../../NCPac/featDir/{testCase}_od_FRADIM_{fileName}.csv') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for (i, row) in enumerate(csvReader):
            if i == 0: continue  # Skip the header
            elif i == 1: scaleChange = np.array([float(logScale) for logScale in row[1:]]).reshape(-1,1)  # Get log(1/r)
            elif i == 2: 
                minCountDuplicates = row.count(row[1])  # Number of duplicates of the minimum count
                if repMethod == 'atomCentre':
                    if '  -Infinity' in row:
                        lastInfIdx = len(row) - row[::-1].index('  -Infinity') - 1  # Index of last occurence of 'Infinity' count
                        if minCountDuplicates < lastInfIdx: minCountDuplicates = lastInfIdx
                    maxCountDuplicates = row.count(row[-1])
                    firstPointIdx, lastPointIdx = minCountDuplicates, -(maxCountDuplicates-1)  # Keep one unique value, remove all counts after the plateau
                elif repMethod == 'atomBall' or repMethod == 'NPsurf':
                    firstPointIdx, lastPointIdx = minCountDuplicates + 1, None  # Remove all duplicates (zeros/inf)
                elif repMethod == 'sAtomBall': continue  # Skip 2nd line (as values calculated includes all atoms)
                countChange = np.array([float(logCounts) for logCounts in row[1:]]).reshape(-1,1)[firstPointIdx:lastPointIdx]
                scaleChange = scaleChange[firstPointIdx:lastPointIdx]
            elif i == 3:  # Only for surfaces
                minCountDuplicates = row.count(row[1])
                if repMethod == 'sAtomBall':
                    if '  -Infinity' in row:
                        lastInfIdx = len(row) - row[::-1].index('  -Infinity') - 1  # Index of last occurence of 'Infinity' count
                        minCountDuplicates = lastInfIdx if minCountDuplicates < lastInfIdx else minCountDuplicates + 1  # Remove all duplicates
                    firstPointIdx, lastPointIdx = minCountDuplicates, None
                    countChange = np.array([float(logCounts) for logCounts in row[1:]]).reshape(-1,1)[firstPointIdx:]  # Remove all duplicates (inf)
                    scaleChange = scaleChange[firstPointIdx:lastPointIdx]
            else: print("Unexpected line!")

    print(f"  Fitting linear line to estimate fractal dimension of {testCase} represented as {repMethod}:")
    r2score = 0.0
    firstPointIdx, lastPointIdx = 0, len(scaleChange) - 1
    while r2score < R2_THRESHOLD:
        x, y = scaleChange[firstPointIdx:lastPointIdx], countChange[firstPointIdx:lastPointIdx]
        regModel = sm.OLS(endog=y, exog=sm.add_constant(x)).fit()
        r2score, boxCountDim, slopeConfInt = regModel.rsquared, regModel.params[1], regModel.conf_int(alpha=ALPHA_CONF_INT)[1]
        print(f"    Coefficient of determination (R2): {r2score:.3f}\
              \n    Estimated box-counting dimension (D_Box): {boxCountDim:.3f}\
              \n    {CONF_INT_PERC}% Confidence interval: [{slopeConfInt[0]:.3f}, {slopeConfInt[1]:.3f}]")

        # Visualise the regression model
        yPred = regModel.predict().reshape(-1, 1)
        if showPlot:
            plt.scatter(x, y);
            plt.plot(x, yPred, label='OLS');
            if len(x) > 2:
                predOLS = regModel.get_prediction()
                lowerCIvals, upperCIvals = predOLS.summary_frame()['mean_ci_lower'], predOLS.summary_frame()['mean_ci_upper']
                # if (np.nan not in lowerCIvals) or (np.nan not in upperCIvals):
                plt.plot(x, upperCIvals, 'r--');
                plt.plot(x, lowerCIvals, 'r--');
            plt.xlabel('log(1/r)');
            plt.ylabel('log(N)');
            plt.title(f"R2: {r2score:.3f}, D_Box: {boxCountDim:.3f}, {CONF_INT_PERC}% CI: [{slopeConfInt[0]:.3f}, {slopeConfInt[1]:.3f}]");
            plt.show();
        
        # Decide the next point to remove
        leastSquareErrs = (y - yPred)**2
        if len(y) % 2 == 0: lowerBoundErrSum, upperBoundErrSum = leastSquareErrs[:len(y)//2].sum(), leastSquareErrs[len(y)//2:].sum()
        else: lowerBoundErrSum, upperBoundErrSum = leastSquareErrs[:len(y)//2].sum(), leastSquareErrs[len(y)//2 + 1:].sum()
        if lowerBoundErrSum > upperBoundErrSum: firstPointIdx += 1
        else: lastPointIdx -= 1
    
    return r2score, boxCountDim, slopeConfInt[0], slopeConfInt[1]
