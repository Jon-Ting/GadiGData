# Purpose: Store variables for generation of NPs using ASE
# Author: Jonathan Yik Chang Ting
# Date: 19/20/2020

from math import sqrt

LMP_DATA_DIR = '/g/data/q27/jt5911/InitStruct'
MNP_DIR, BNP_DIR = 'MNP/', 'BNP/RCS/'
GOLDEN_RATIO = (1+sqrt(5)) / 2
VACUUM_THICKNESS = 10.0
RANDOM_DISTRIB_NO = 10

# Elements of interest & their lattice parameters
eleDict = {'Co': 
            {'lc': {'FCC': 3.537, 'HCP': 1.621}}, 
           'Pd':
            {'lc': {'FCC': 3.89}}, 
           'Pt': 
            {'lc': {'FCC': 3.92}}, 
           'Au': 
            {'lc': {'FCC': 4.09}}}
'''
- Pd, Pt, Au values were obtained from N. W. Ashcroft and N. D. Mermin, Solid State Physics (Holt, Rinehart, and Winston, New York, 1976.
- The lattice constants are 3.859, 3.912, and 4.065 Angstroms, for the respective FCC metals at 300 K according to W. P. Davey, "Precision Measurements of the Lattice Constants of Twelve Common Metals," Physical Review, vol. 25, (6), pp. 753-761, 1925.
- CoFCC was obtained from Owen, E & Jones, D. (2002). Effect of Grain Size on the Crystal Structure of Cobalt. Proceedings of the Physical Society. Section B. 67. 456. 10.1088/0370-1301/67/6/302. (291 K)
- CoHCP c/a ratio was obtained from F. Ono, H. Maeta. DETERMINATION OF LATTICE PARAMETERS IN HCP COBALT BY USING X-RAY BOND'S METHOD. Journal de Physique Colloques, 1988, 49 (C8), pp.C8-63-C8-64. (4.2 K, 1.6228 at 298 K)
'''

# NP diameters of interest (Angstrom)
# diameterList = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180, 200, 250, 300]  # For generating MNP
# diameterList = [20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 300]  # Ideally generate for all BNPs
diameterList = [20, 30, 40, 50, 60, 70, 80]  # Currrent BNP generation

# Shapes of interest
shapeList = ['CU', 'TH', 'RD', 'OT', 'TO', 'CO', 'DH', 'IC']

# Distributions of interest
distribList = ['RAL', 'RCS', 'L10', 'L12'] #, 'RL10', 'RL12']

# Ratios of interest (A where A + B = 100)
# ratioList = [10, 20, 25, 30, 50, 70, 75, 80, 90]
ratioList = [25, 50, 75]
