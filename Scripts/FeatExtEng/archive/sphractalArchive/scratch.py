from numba import float64, int64, types, typed
from numba.experimental import jitclass
import numpy as np


spec = [
    ('ID', int64),
    ('ele', types.unicode_type),
    ('X', float64),
    ('Y', float64),
    ('Z', float64),
    ('neighs', typed.List.empty_list(types.Array(int64, 1, 'C'))),
    ('isSurf', int64),
    ('avgBondLen', float64)
]

@jitclass(spec)
class Atom(object):
    """Object to store information about any particular atom."""
    def __init__(self, atomIdx, ele, X, Y, Z):
        self.ID = atomIdx
        self.ele = ele
        self.X, self.Y, self.Z = X, Y, Z
        self.neighs = [np.array([i for i in range(0)])]
        self.isSurf = 0
        self.avgBondLen = 0.0

atomIdx = 1
ele = 'Au'
X = float('324.43')
Y = float('43.435325')
Z = float('34.4325')

atom = Atom(atomIdx, ele, X, Y, Z)
atom.isSurf = 1
atom.avgBondLen += 1.3
atom.avgBondLen /= 5
newArr = np.array([12,3,45,6,67,7])
newArrTyped = typed.List()
#newArrTyped.append(np.array(1))
#newArrTyped = typed.List(types.Array(int64, 1, 'C'))
#atom.neighs = newArrTyped
#atom.neighs.append(np.array([2,46,87,9,34]))
#np.append(atom.neighs[0], [454,56,68])
print(atom.neighs)
