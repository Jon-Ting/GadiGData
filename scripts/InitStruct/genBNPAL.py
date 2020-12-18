# Goal: Generate initial alloy BNP structures for MD simulations
# Author: Jonathan Yik Chang Ting
# Date: 22/10/2020
'''
Note:
- Abbreviations:
    - RAL = randomly distributed alloy
    - RCS = randomly distributed core-shell-like alloy
    - (R)L10 = L1_0 intermetallic alloy (with/without random component)
    - (R)L12 = L1_2 intermetallic alloy (with/without random component)
- To do:
    - FCC is currently being hard-coded for lattice constant retrieval, might need to be flexible
    - Perhaps could add a parameter to control core thickness
'''

import numpy as np
from os.path import isfile, isdir
from numpy.random import seed, rand, RandomState
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from ase.visualize import view
from constants import LMP_DATA_DIR, MNP_DIR, BNP_DIR, RANDOM_DISTRIB_NO, VACUUM_THICKNESS, eleDict, diameterList, shapeList, distribList, ratioList


def dist1D(coord1, coord2, dim):
    """Compute distance between 2 points in one of their real space coordinates"""
    return round(np.sqrt(np.sum((coord2[dim]-coord1[dim]) ** 2)), 3)


def dist3D(coord1, coord2):
    """Compute real space distance between 2 points"""
    return round(np.sqrt(np.sum((coord2 - coord1) ** 2)), 3)


def randConv(obj, element2, ele1Ratio, rseed, prob):
    """Randomly convert elements of atoms until specified ratio is reached"""
    elements = np.array(list(obj.symbols.species()))
    element1 = elements[elements != element2][0]
    ele1Arr, ele2Arr = obj.symbols.search(element1), obj.symbols.search(element2)
    ele2IdealNum = round((100-ele1Ratio) / 100 * len(obj))
    diff = len(ele2Arr) - ele2IdealNum
    (convEleArr, targetEle) = (ele1Arr, element2) if diff < 0 else (ele2Arr, element1)
    randGen = RandomState(rseed)
    probArr = np.array(prob)[convEleArr] / np.array(prob)[convEleArr].sum() if len(prob) > 0 else None
    idxArr = randGen.choice(a=convEleArr, size=abs(diff), replace=False, p=probArr)
    for idx in idxArr: obj[idx].symbol = targetEle
    return obj


def genBNP(obj, element2, shape, ratio, distrib, rseed):
    probList = []
    if distrib == 'RAL':
        seed(rseed)
        randList = rand(len(obj))  # Uniform distribution
        for (i, atom) in enumerate(obj):
            if randList[i] > ratio / 100: atom.symbol = element2

    elif distrib == 'RCS':
        if shape == 'IC':
            massCenter = obj.get_center_of_mass()
            radius = (obj.cell[0][0]-VACUUM_THICKNESS)/2
        else:
            xSlices = set([round(atom[0], 3) for atom in obj.positions])
            ySlices = set([round(atom[1], 3) for atom in obj.positions])
            zSlices = set([round(atom[2], 3) for atom in obj.positions])
            
            # Find the center of each line in the box
            zThreadDict = {(x, y): {'max': 0, 'min': 0, 'mid': []} for x in xSlices for y in ySlices}
            xThreadDict = {(y, z): {'max': 0, 'min': 0, 'mid': []} for y in ySlices for z in zSlices}
            yThreadDict = {(z, x): {'max': 0, 'min': 0, 'mid': []} for z in zSlices for x in xSlices}
            for (atomIdx, atom) in enumerate(obj):
                xCoord, yCoord, zCoord = round(atom.position[0], 3), round(atom.position[1], 3), round(atom.position[2], 3)
                zThreadDict[(xCoord, yCoord)]['mid'].append(zCoord)
                xThreadDict[(yCoord, zCoord)]['mid'].append(xCoord)
                yThreadDict[(zCoord, xCoord)]['mid'].append(yCoord)
            threadDicts = [zThreadDict, xThreadDict, yThreadDict]
            for threadDict in threadDicts:
                emptyDictKeys = []
                for (dim1, dim2) in threadDict.keys():
                    if len(threadDict[(dim1, dim2)]['mid']) == 0: emptyDictKeys.append((dim1, dim2)); continue
                    threadDict[(dim1, dim2)]['max'] = max(threadDict[(dim1, dim2)]['mid'])
                    threadDict[(dim1, dim2)]['min'] = min(threadDict[(dim1, dim2)]['mid'])
                    threadDict[(dim1, dim2)]['mid'] = (threadDict[(dim1, dim2)]['max']+threadDict[(dim1, dim2)]['min']) / 2
                for key in emptyDictKeys: del threadDict[key]
     
        # Alchemical change
        seed(rseed)
        randList = rand(len(obj))
        for (atomIdx, atom) in enumerate(obj):
            if shape == 'IC':
                prob = dist3D(massCenter, atom.position) / radius
            else:
                xCoord, yCoord, zCoord = round(atom.position[0], 3), round(atom.position[1], 3), round(atom.position[2], 3)
                zHalfLen = (zThreadDict[(xCoord, yCoord)]['max']-zThreadDict[(xCoord, yCoord)]['min']) / 2
                xHalfLen = (xThreadDict[(yCoord, zCoord)]['max']-xThreadDict[(yCoord, zCoord)]['min']) / 2
                yHalfLen = (yThreadDict[(zCoord, xCoord)]['max']-yThreadDict[(zCoord, xCoord)]['min']) / 2
                zRelPos = abs(zCoord-zThreadDict[(xCoord, yCoord)]['mid']) / zHalfLen if round(zHalfLen, 3) != 0.0 else abs(zCoord-zThreadDict[(xCoord, yCoord)]['mid'])
                xRelPos = abs(xCoord-xThreadDict[(yCoord, zCoord)]['mid']) / xHalfLen if round(xHalfLen, 3) != 0.0 else abs(xCoord-xThreadDict[(yCoord, zCoord)]['mid'])
                yRelPos = abs(yCoord-yThreadDict[(zCoord, xCoord)]['mid']) / yHalfLen if round(yHalfLen, 3) != 0.0 else abs(yCoord-yThreadDict[(zCoord, xCoord)]['mid'])
                if shape == 'DH':
                    prob = 1 if (round(zRelPos, 3) == 1.0) | (round(zHalfLen, 3) == 0.0) else zRelPos
                else:
                    if (round(zRelPos, 3) == 1.0) | (round(xRelPos, 3) == 1.0) | (round(yRelPos, 3) == 1.0) | (round(zHalfLen, 3) == 0.0) | (round(xHalfLen, 3) == 0.0) | (round(yHalfLen, 3) == 0.0): prob = 1
                    else: prob = (zRelPos+xRelPos+yRelPos) / 3
            if randList[atomIdx] < prob: obj[atomIdx].symbol = element2
            probList.append(prob)

    elif distrib in ['L10', 'L12', 'RL10', 'RL12']:
        lc = eleDict[obj[0].symbol]['lc']['FCC']
        vacOffset = VACUUM_THICKNESS / 2
        for (i, atom) in enumerate(obj):
            yModulo = round((round(obj.positions[i][1], 3) - vacOffset) % lc, 3)
            if (yModulo == 0.0) | (yModulo == lc): atom.symbol = element2
            if distrib == 'L12':
                xModulo = round((round(obj.positions[i][0], 3) - vacOffset) % lc, 3)
                if (xModulo == 0.0) | (xModulo == lc): atom.symbol = element2

    else:
        raise Exception('Specified distribution type unrecognised!')

    if 'R' in distrib: obj = randConv(obj=obj, element2=element2, ele1Ratio=ratio, rseed=rseed, prob=probList)
    return obj


def writeBNP(element1, diameter, shape, ratio, distrib, replace=False, vis=False):
    if not isdir(LMP_DATA_DIR): raise Exception("Can't found directory to store data files!")
    for element2 in eleDict:
        if element2 is element1: continue
        fileNameMNP = '{0}{1}{2}.lmp'.format(element1, diameter, shape)
        mnp = read_lammps_data(fileobj='{0}{1}{2}'.format(LMP_DATA_DIR, MNP_DIR, fileNameMNP), style='atomic', units='metal')
        mnp.set_chemical_symbols(symbols=[element1] * len(mnp))
        for rep in range(RANDOM_DISTRIB_NO):
            if 'R' not in distrib: ratio, rep = '', ''
            fileNameBNP = '{0}{1}{2}{3}{4}{5}{6}.lmp'.format(element1, element2, diameter, shape, ratio, distrib, rep)
            if not replace:
                if isfile(LMP_DATA_DIR + BNP_DIR + fileNameBNP): 
                    print('      {0} already exist, skipping...'.format(fileNameBNP))
                    continue
            bnp = genBNP(obj=mnp.copy(), element2=element2, shape=shape, ratio=ratio, distrib=distrib, rseed=rep)
            write_lammps_data(fileobj='{0}{1}{2}'.format(LMP_DATA_DIR, BNP_DIR, fileNameBNP), atoms=bnp, units='metal', atom_style='atomic')
            print('      Generated {0}, formula: {1}'.format(fileNameBNP, bnp.get_chemical_formula()))
            if vis: view(bnp)
            if 'R' not in distrib: break


def main(replace=False, vis=False):
    print('Generating BNP alloys of:')
    for diameter in diameterList:
        if diameter != 300: continue  # DEBUG
        print('\n  Size {0} Angstrom for:'.format(diameter))
        for element in eleDict:
            if element != 'Pt':  continue  # DEBUG
            print('    Element: {0}'.format(element))
            for shape in shapeList:
                # if shape not in ['DH']:  continue  # DEBUG
                print('    Shape: {0}'.format(shape))
                for ratio in ratioList:
                    # if ratio != 10:  continue  # DEBUG
                    print('    Ratio: {0}'.format(ratio))
                    for distrib in distribList:
                        if (shape in ['DH', 'IC']) & ('L1' in distrib): continue
                        if distrib != 'RCS':  continue  # DEBUG
                        writeBNP(element1=element, diameter=diameter, shape=shape, ratio=ratio, distrib=distrib, replace=replace, vis=vis)
    

if __name__ == '__main__':
    main(replace=False, vis=False)
    print('ALL DONE!')


