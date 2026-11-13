import numpy as np

from daelab.base.datatypes.arithmetic import ArithmeticMethods
from daelab.base.datatypes.array import _assign


def _shapes(init):
    """Shapes (tuples) contained in init, i.e., init without the data type."""
    return tuple(x for x in init if isinstance(x, tuple))


class NodeList(ArithmeticMethods):
    def __init__(self, dtype_node, init, num_nodes):
        self.dtype_node = dtype_node
        self.init = init
        self._nodes = [dtype_node(init) for _ in range(num_nodes + 1)]

    def __len__(self):
        return len(self._nodes)

    def __getitem__(self, m):
        """
        Returns the solution at node ``m``.

        Parameters
        ----------
        m : int or slice
            Index of the collocation node.

        Returns
        -------
        numpy.ndarray
            reference to the solution at node ``m`` of shape
            ``nvars``. If the index is a slice, it returns a list
            of arrays.
        """
        return self._nodes[m]

    def __setitem__(self, m, val):
        """
        Writes ``val`` into the solution at node ``m``.

        Parameters
        ----------
        m : int or slice
            Index of the collocation node.
        val : scalar or array_like
            Values to be written. For scalars, it will be broadcasted
            to ``nvars``. The values are copied, the stored array is not
            replaced.
        """
        if isinstance(m, slice):
            for node in self._nodes[m]:
                _assign(node, val)
        else:
            _assign(self._nodes[m], val)

    def flat_node(self, m):
        """
        Solution at node m as flat vector of length ``np.prod(nvars)``.

        Parameters
        ----------
        m : int
            Index of the collocation nodes.

        Returns
        -------
        numpy.ndarray
            Array of length ``np.prod(nvars)``.
        """
        return self._nodes[m].flat()

    def as_flat(self):
        """
        Returns the solution at all nodes as a matrix of shape
        ``(num_nodes + 1, np.prod(nvars))``.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_nodes + 1, np.prod(nvars))``.
        """
        return np.stack([node.flat() for node in self._nodes])

    def copy(self):
        """
        Returns an independent copy of the object.

        Returns
        -------
        NodeList
            Copy of an object.
        """
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        new._nodes = [node.copy() for node in self._nodes]
        return new

    def _binary(self, other, op):
        """
        Applies an operation ``op`` to ``self`` and ``other`` and returns the
        result.

        Parameters
        ----------
        other : Solution or other datatype
            Second operand. A `Solution` object must have the same
            ``num_nodes`` and ``nvars``.
        op : callable
            Operation.

        Returns
        -------
        new : Solution
            Result of the same type as ``self`` of performed operation.

        Raises
        ------
        ValueError
            If ``other`` has different ``nvars`` or a different ``num_nodes``.
        """
        new = self.copy()
        if isinstance(other, NodeList):
            if len(other) != len(self):
                raise ValueError("Operands must have the same length!")
            if _shapes(other.init) != _shapes(self.init):
                raise ValueError("Operands must have the same nvars.")
            new._nodes = [
                op(a, b) for a, b in zip(self._nodes, other._nodes, strict=False)
            ]
        else:
            new._nodes = [op(a, other) for a in self._nodes]
        return new
