import numpy as np
import operator


class ArithmeticMethods:
    """Class that contains arithmetic operations for datatypes below."""
    def __add__(self, other):
        return self._binary(other, operator.add)
 
    def __sub__(self, other):
        return self._binary(other, operator.sub)
 
    def __mul__(self, other):
        return self._binary(other, operator.mul)
 
    def __truediv__(self, other):
        return self._binary(other, operator.truediv)


class Solution(ArithmeticMethods):
    def __init__(self, num_nodes, nvars, dtype=float):
        self.num_nodes = num_nodes
        self.nvars = nvars
        self.dtype = dtype
        self._comps = [np.zeros(self.nvars, dtype=self.dtype) for _ in range(self.num_nodes)]

    def __getitem__(self, m):
        return self._comps[m]

    def __setitem__(self, m, val):
        """
        Writes val into the solution at node m (in-place, broadcasting allowed).
        A slice, e.g., u[:] = val, writes val into all selected nodes.
        """
        if isinstance(m, slice):
            for node in self._comps[m]:
                node[...] = val
        else:
            self._comps[m][...] = val

    def __repr__(self):
        return (
            f"{type(self).__name__}(num_nodes={self.num_nodes}, "
            f"nvars={self.nvars}, dtype={np.dtype(self.dtype)})"
        )

    def copy(self):
        new = type(self).__new__(type(self))
        new.num_nodes, new.nvars, new.dtype = self.num_nodes, self.nvars, self.dtype
        new._comps = [comp.copy() for comp in self._comps]
        return new

    def _binary(self, other, op):
        new = self.copy()
        if isinstance(other, Solution):
            if other.num_nodes != self.num_nodes or other.nvars != self.nvars:
                raise ValueError("Operands must have the same num_nodes and nvars!")
            new._comps = [op(a, b) for a, b in zip(self._comps, other._comps)]
        else:
            new._comps = [op(a, other) for a in self._comps]
        return new

    def apply_matrix(self, Q):
        """(Q u)[m] = sum_k Q[m, k] * u[k], e.g., Q @ F."""
        new = self.copy()
        for m in range(self.num_nodes):
            new._comps[m][...] = sum(
                Q[m, k] * self._comps[k] for k in range(self.num_nodes)
            )
        return new


class PartitionedContainer(ArithmeticMethods):
    _components = ()

    @property
    def _parts(self):
        return [getattr(self, name) for name in self._components]

    def __getitem__(self, m):
        return tuple(part[m] for part in self._parts)

    def __setitem__(self, m, value):
        if len(value) != len(self._components):
            raise ValueError(f"Expected {len(self._components)} values: {self._components}.")
        for part, val in zip(self._parts, value):
            part[m] = val

    def copy(self):
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        for name in self._components:
            setattr(new, name, getattr(self, name).copy())
        return new
 
    def _binary(self, other, op):
        new = self.copy()
        if isinstance(other, PartitionedContainer):
            if other._components != self._components or other.num_nodes != self.num_nodes:
                raise ValueError("Operands must have the same parts and num_nodes.")
            for name in self._components:
                setattr(new, name, op(getattr(self, name), getattr(other, name)))
        elif isinstance(other, Solution):
            raise TypeError(f"Cannot combine {type(self).__name__} with a Solution.")
        else:  # scalar
            for name in self._components:
                setattr(new, name, op(getattr(self, name), other))
        return new
 
    def apply_matrix(self, Q):
        new = self.copy()
        for name in self._components:
            setattr(new, name, getattr(self, name).apply_matrix(Q))
        return new


class SemiexplicitSolution(PartitionedContainer):
    _components = ("diff", "alg")

    def __init__(self, num_nodes, nvars_diff, nvars_alg, dtype=float):
        if nvars_diff[1:] != nvars_alg[1:]:
            raise ValueError(
                "nvars_diff and nvars_alg must have the same spatial dimensions."
            )

        self.num_nodes = num_nodes
        self.nvars_diff = nvars_diff
        self.nvars_alg = nvars_alg
        self.dtype = dtype

        self.diff = Solution(self.num_nodes, self.nvars_diff, self.dtype)
        self.alg = Solution(self.num_nodes, self.nvars_alg, self.dtype)

    def __repr__(self):
        return (
            f"{type(self).__name__}(num_nodes={self.num_nodes}, "
            f"nvars_diff={self.nvars_diff}, nvars_alg={self.nvars_alg}, dtype={np.dtype(self.dtype)})"
        )


class RHS(Solution):
    """
    Right-hand side of a fully-implicit DAE (residual form F(t, u, u') = 0) at all
    collocation nodes. Same structure as `Solution`: F[m] is the residual at node m
    and has shape nvars.
    """


class IMEXRHS(PartitionedContainer):
    _components = ("impl", "expl")

    def __init__(self, num_nodes, nvars, dtype):
        self.num_nodes = num_nodes
        self.nvars = nvars
        self.dtype = dtype
        self.impl = RHS(num_nodes=self.num_nodes, nvars=self.nvars, dtype=self.dtype)
        self.expl = RHS(num_nodes=self.num_nodes, nvars=self.nvars, dtype=self.dtype)

    @property
    def full_rhs(self):
        return self.impl + self.expl

    def __repr__(self):
        return (
            f"{type(self).__name__}(num_nodes={self.num_nodes}, nvars={self.nvars}, dtype={np.dtype(self.dtype)})"
        )


class SemiexplicitRHS(SemiexplicitSolution):
    def __init__(self, num_nodes, nvars_diff, nvars_alg, dtype=float, imex_diff=False):
        super().__init__(num_nodes, nvars_diff, nvars_alg, dtype)
        self.imex_diff = imex_diff
        self.diff = (
            IMEXRHS(self.num_nodes, self.nvars_diff, self.dtype)
            if self.imex_diff
            else RHS(self.num_nodes, self.nvars_diff, self.dtype)
        )
        self.alg = RHS(self.num_nodes, self.nvars_alg, self.dtype)
