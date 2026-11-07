import numpy as np
import operator

from daelab.base.datatypes import Solution, PartitionedContainer, SemiexplicitSolution, IMEXRHS, SemiexplicitRHS


u = Solution(num_nodes=4, nvars=(4, 4))
print("u[1] before overwriting:", u[1])
u[1][0][:] = 1.0
print("u[1] after overwriting:",u[1])
print(float(u[1].sum()))

print("u[2] before overwriting:", u[2])
u[2] = 5.0
print("u[2] after overwriting:",u[2])
v = 2 * u + u
print("v", float(v[1][0, 0]))

num_nodes1 = 3
nvars1 = (4, 4)
dtype1 = float
u1 = Solution(num_nodes=num_nodes1, nvars=nvars1, dtype=dtype1)

for m in range(num_nodes1):
    print("Shape:", u1.flat_node(m).shape)

print(u.as_flat().shape)

print(type(operator.add))

class Pair(PartitionedContainer):
    _components = ("a", "b")
    def __init__(self, num_nodes, nvars_a, nvars_b):
        self.num_nodes = num_nodes
        self.a = Solution(num_nodes, nvars_a)
        self.b = Solution(num_nodes, nvars_b)

p = Pair(num_nodes=2, nvars_a=(1, 3), nvars_b=(2, 3))
p.a[0] = 1.0
a0, b0 = p[0]
print(a0.shape, b0.shape)

p[1] = (2.0, 3.0)
print(p.a[1])
print(p.b[1])

q = 2 * p
print(float(q.a[0][0, 0]), float(q.b[0][0, 0]))

u = SemiexplicitSolution(num_nodes=3, nvars_diff=(2, 4, 5), nvars_alg=(1, 4, 5))
u[2] = (np.ones((2, 4, 5)), 2 * np.ones((1, 4, 5)))
print(u.diff[2][0, 0, 0], u.alg[2][0, 0, 0])

v = 2 * u + u
print(float(v.diff[2][0, 0, 0]), float(v.alg[2][0, 0, 0]))

f = IMEXRHS(num_nodes=4, nvars=(2, 4))
print(f.impl[1].shape)

f.impl[1] = 2 * np.ones((2, 4))
f.expl[1] = np.ones((2, 4))
g1 = f.impl[1] + 10 * f.expl[1]
print(f.impl[1].shape,  f.expl[1].shape, (10 * f.expl[1]).shape, (f.impl[1] + 10 * f.expl[1]).shape, g1.shape)
print(g1, type(g1))
print(f.full_rhs)

f = SemiexplicitRHS(num_nodes=3, nvars_diff=(2, 4), nvars_alg=(1, 4), imex_diff=True)

f.diff.impl[2] = 0.25 * np.ones((2, 4))
f.diff.expl[2] = 0.33 * np.ones((2, 4))

print("f_diff", f.diff[:])
print("f_diff_full_rhs", f.diff.full_rhs[2])