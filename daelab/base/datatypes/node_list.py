import numpy as np

from daelab.base.datatypes.arithmetic import ArithmeticMethods
from daelab.base.datatypes.array import _assign


def _shapes(init):
    """Shapes (tuples) contained in init, i.e., init without the data type."""
    return tuple(x for x in init if isinstance(x, tuple))


class NodeList(ArithmeticMethods):
    """
    Class to create a lise of node objects, e.g., the solution or the right-hand
    side at all collocation nodes.

    The class stores one array of type ``dtype_node`` and corresponding ``init``
    per collocation node. For ``num_nodes`` the list is of length
    ``num_nodes + 1``, where at index ``0``, the initial condition will be
    stored. The operators ``+``, ``-``, ``*``, ``/`` and the unary ``-`` are
    made available through inheritance in this class.

    Parameters
    ----------
    dtype_node : Array, SemiexplicitArray, SemiexplicitIMEXArray, or IMEXArray
        Array datatype at every node.
    init : tuple
        Corresponding init for ``dtype_node``.
    num_nodes : int
        Number of collocation nodes.

    Attributes
    ----------
    dtype_node : Array, SemiexplicitArray, SemiexplicitIMEXArray, or IMEXArray
        Array datatype at every node.
    init : tuple
        Corresponding init for ``dtype_node``.

    Notes
    -----
    - For a ``NodeList`` ``u`` the solution at each collocation node is stored
      in a list that has ``num_nodes + 1`` entries, each of them is of shape
      ``nvars``. For differential and algebraic parts, each entry has shapes
      ``nvars_diff`` and ``nvars_alg``. In ``u[0]``, the initial condition
      ``u0`` is stored.
    - The values of ``u`` at node ``m`` can be accessed using ``u[m + 1]`` and
      ``u[m + 1][i]`` is the i-th unknown at node ``m``. For datatypes using
      different parts ``comp1`` and ``comp2``, the values can be accessed using
      ``u[m + 1].comp1[:]`` and ``u[m + 1].comp2[:]``.
    - ``u[m]`` returns a reference, not a copy. Modifying it, e.g.,
      ``u[m][i][:] = 1.0``, changes the stored solution.
    - Arithmetic operations with another ``NodeList`` requires the same
      ``init``. Scalars and arrays that broadcast to ``nvars`` are applied to
      every node.

    Examples
    --------
    >>> u = NodeList(SemiexplicitArray, ((2, 4, 5), (1, 4, 5), float), 3)
    >>> len(u)
    4
    >>> u[2].diff.shape, u[2].alg.shape
    (2, 4, 5), (1, 4, 5)
    >>> u.flat_node(1).shape
    (60,)
    >>> u.as_flat().shape
    (4, 60)
    """

    def __init__(self, dtype_node, init, num_nodes):
        """Initialization routine."""
        self.dtype_node = dtype_node
        self.init = init
        self._nodes = [dtype_node(init) for _ in range(num_nodes + 1)]

    def __len__(self):
        """
        Returns the length of the list, which is ``num_nodes + 1``.

        Returns
        -------
        int
            Length of list
        """
        return len(self._nodes)

    def __getitem__(self, m):
        """
        Returns the array at node ``m``.

        Parameters
        ----------
        m : int or slice
            Index of the collocation node.

        Returns
        -------
        numpy.ndarray
            Reference to the array at node ``m`` of shape ``nvars``. If the
            index is a slice, it returns a list of arrays.
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
        Array at node ``m`` as flat vector of length ``np.prod(nvars)``.

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
        Returns the entries at all nodes as a matrix of shape
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
        other : NodeList
            Second operand. A `NodeList` object must have the same length and
            ``init``.
        op : callable
            Operation.

        Returns
        -------
        new : NodeList
            Result of the same type as ``self`` of performed operation.

        Raises
        ------
        ValueError
            If ``other`` has different length or a different ``init``.
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
