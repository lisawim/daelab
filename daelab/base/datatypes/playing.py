import numpy as np

from daelab.base.datatypes.array import Array
from daelab.base.datatypes.semiexplicit import SemiexplicitArray


init = ((2, 4), float)
a = Array(init)

print(a.values)

val = np.array([m for m in range(8)]).reshape((2, 4))

a[:] = val

print(a.values)

init_b = ((2, 4), (3, 4), float)
b = SemiexplicitArray(init_b)
c = SemiexplicitArray(init_b)

val1 = np.array([m for m in range(8)]).reshape((2, 4))
val2 = np.array([m for m in range(12)]).reshape((3, 4))

b.diff[:] = val1
b.alg[:] = val2

print(b)

c[:] = b

print(c)
