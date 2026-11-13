import numpy as np

from daelab.base.datatypes.arithmetic import ArithmeticMethods
from daelab.base.datatypes.array import _assign, Array
from daelab.base.datatypes.node_list import NodeList


class PartitionedArray(ArithmeticMethods):
    _components = ()

    @property
    def _parts(self):
        return [getattr(self, name) for name in self._components]

    def __iter__(self):
        return iter(self._parts)

    def assign(self, value):
        """
        Writes `value` into the parts (in-place).

        Parameters
        ----------
        value : scalar, tuple or object of the same type
            A scalar is written into all parts, a tuple contains one
            value per part, in the order of ``_components``.
        """
        if isinstance(value, PartitionedArray):
            if value._components != self._components:
                raise ValueError("Operands must have the same parts.")
            values = value._parts
        elif isinstance(value, (int, float, complex, np.number)):
            values = [value] * len(self._components)
        elif isinstance(value, np.ndarray):
            raise TypeError(
                "Expected a scalar, a tuple with one value per part, or an object "
                f"of type {type(self).__name__}, not a numpy array."
            )
        else:
            values = tuple(value)
            if len(values) != len(self._components):
                raise ValueError(
                    f"Expected {len(self._components)} values: "
                    f"{self._components}, got {len(values)} values."
                )
        for part, val in zip(self._parts, values, strict=False):
            _assign(part, val)

    def copy(self):
        """
        Returns an independent copy of the object.

        Returns
        -------
        PartitionedConainer
            Copy of an object including the same parts.
        """
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        for name in self._components:
            setattr(new, name, getattr(self, name).copy())
        return new

    def flat(self):
        return np.concatenate([part.flat() for part in self._parts])

    def _binary(self, other, op):
        """
        Applies an operation ``op`` to ``self`` and ``other`` part by part and
        returns the result as a new object.

        Parameters
        ----------
        other : PartitionedContainer or scalar
            Second operand. A `_Partitioned` object must have the same
            ``_names`` and ``num_nodes``.
        op : callable
            Operation.

        Returns
        -------
        new : Solution
            Result of the same type as ``self`` of performed operation.

        Raises
        ------
        ValueError
            If ``other`` has different parts or a different ``num_nodes``.
        TypeError
            If ``other`` is a numpy array or a `NodeList`.
        """
        new = self.copy()
        if isinstance(other, PartitionedArray):
            if other._components != self._components:
                raise ValueError("Operands must have the same parts.")

            for name in self._components:
                setattr(new, name, op(getattr(self, name), getattr(other, name)))
        elif isinstance(other, (np.ndarray, NodeList)):
            raise TypeError(
                f"Cannot combine {type(self).__name__} with {type(other).__name__}."
            )
        else:  # scalar
            for name in self._components:
                setattr(new, name, op(getattr(self, name), other))
        return new


class IMEXArray(PartitionedArray):
    _components = ("impl", "expl")

    def __init__(self, init):
        nvars, dtype = init
        self.nvars = tuple(nvars)
        self.impl = Array(init)
        self.expl = Array(init)

    @property
    def full_rhs(self):
        return self.impl + self.expl
