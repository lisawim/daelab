import numpy as np
import operator

from daelab.base.problems.dae_problem import DAEProblem


class ArithmeticMethods:
    """
    Mixin providing the arithmetic operators for solution and right-hand side
    containers.

    The class defines the basic operators ``+``, ``-``, ``*``, ``/`` and the
    unary ``-``. Each operation is performed in the ``_binary`` method that
    have to be implemented in each subclass.

    Methods that a subclass must implement
    --------------------------------------
    _binary(self, other, op)
        Applies the operation ``op`` to ``self`` and ``other`` and returns the
        result as a new object (representing a copy of ``self``). Here, ``op``
        is one of the functions ``operator.add``, ``operator.sub``,
        ``operator.mul`` and ``operator.truediv``, and ``other`` is either a
        scalar or another object of a compatible type.

    Notes
    -----
    - ``__radd__`` and ``__rmul__`` are supported, so ``2 * u`` and ``2 + u``
      work. Since ``+`` and ``*`` are commutative, they use the same
      implementation as ``__add__`` and ``__mul__``.
    - The unary minus, i.e., negating a value ``val``by ``-val``is realized by
      multiplying it by one.

    Examples
    --------
    >>> import operator
    >>> class Counter(ArithmeticMethods):
    ...     def __init__(self, value):
    ...         self.value = value
    ...
    ...     def _binary(self, other, op):
    ...         return Counter(op(self.value, other))
    >>> (Counter(3) + 2).value
    5
    >>> (2 * Counter(3)).value
    6
    >>> (-Counter(3)).value
    -3
    """

    def __add__(self, other):
        """Returns self + other as a new object."""
        return self._binary(other, operator.add)

    def __radd__(self, other):
        """Returns other + self as a new object."""
        return self._binary(other, operator.add)

    def __sub__(self, other):
        """Returns self - other as a new object."""
        return self._binary(other, operator.sub)

    def __mul__(self, other):
        """Returns self * other as a new object."""
        return self._binary(other, operator.mul)

    def __rmul__(self, other):
        """Returns other * self as a new object."""
        return self._binary(other, operator.mul)

    def __truediv__(self, other):
        """Returns self / other as a new object."""
        return self._binary(other, operator.truediv)

    def __neg__(self):
        """Returns self * (-1)."""
        return self * (-1)


class Solution(ArithmeticMethods):
    """
    Solution datatype of a DAE at all collocation nodes.

    The class stores one numpy array of shape ``nvars`` per collocation node.
    It is used for general fullimplicit DAE problems and represents the parent
    class for the numerical solution for semiexplicit DAEs (see
    ``SemiexplicitSolution``) that have differential and algebraic parts. The
    operators ``+``, ``-``, ``*``, ``/`` and the unary ``-`` are made available
    through inheritance in this class.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes.
    nvars : tuple of integers
        Number of unknowns (first entry), and spatial dimensions shared by all
        unknowns (remaining) entries.
    dtype : data-type, optional
        Data type of the entries. Default is float.

    Attributes
    ----------
    num_nodes : int
        Number of collocation nodes.
    nvars : tuple of integers
        Shape of solution at single collocation node.
    dtype : data-type
        Data type of the entries.

    Notes
    -----
    - The solution at each collocation node is stored in a list that has
      ``num_nodes + 1`` entries, each of them is of shape ``nvars`` and of data
      type ``dtype``. In ``u[0]``, the initial condition ``u0`` is stored. The
      solution ``u`` at node ``m`` can be accessed using ``u[m + 1]`` and
      ``u[m + 1][i]`` is the i-th unknown at node ``m``.
    - ``u[m]`` returns a reference, not a copy. Modifying it, e.g.,
      ``u[m][i][:] = 1.0``, changes the stored solution.
    - Arithmetic operations with another solution of type ``Solution`` requires
      ``num_nodes`` and ``nvars`` of both solutions to be equal. Scalars and
      arrays that broadcast to ``nvars`` are applied to every node.

    Examples
    --------
    >>> u = Solution(num_nodes=4, nvars=(4, 4))
    >>> len(u)
    5
    >>> u[2].shape
    (4, 4)
    >>> u[1][0][:] = 1.0  # writes 1.0 to each entry of unknown at index 0
    >>> u[1].sum()
    4.0
    >>> u[2] = 5.0  # writes 5.0 to each entry of solution at node 2
    >>> v = 2 * u + u
    >>> v[1][0, 0]
    3.0
    >>> u.flat_node(1).shape
    (16,)
    >>> u.as_flat().shape
    (5, 16)
    """

    def __init__(self, num_nodes, nvars, dtype=float):
        """Initialization routine."""

        if isinstance(nvars, int):
            nvars = (nvars,)

        DAEProblem.validate_nvars(nvars)

        self.num_nodes = num_nodes
        self.nvars = nvars
        self.dtype = dtype
        self._comps = [
            np.zeros(self.nvars, dtype=self.dtype) for _ in range(self.num_nodes + 1)
        ]

    def __len__(self):
        """Returns the length of the solution array."""
        return len(self._comps)

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
        return self._comps[m]

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
            for node in self._comps[m]:
                node[...] = val
        else:
            self._comps[m][...] = val

    def __repr__(self):
        """
        Returns the class with inputs as string.

        Returns
        str
            Class description.
        """
        return (
            f"{type(self).__name__}(num_nodes={self.num_nodes}, "
            f"nvars={self.nvars}, dtype={np.dtype(self.dtype)})"
        )

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
        return self._comps[m].reshape(-1)

    def as_flat(self):
        """
        Returns the solution at all nodes as a matrix of shape
        ``(num_nodes, np.prod(nvars))``.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_nodes, np.prod(nvars))``.
        """
        return np.stack([node.reshape(-1) for node in self._comps])

    def copy(self):
        """
        Returns an independent copy of the object.

        Returns
        -------
        Solution
            Copy of an object.
        """
        new = type(self).__new__(type(self))
        new.num_nodes, new.nvars, new.dtype = self.num_nodes, self.nvars, self.dtype
        new._comps = [comp.copy() for comp in self._comps]
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
        if isinstance(other, Solution):
            if other.num_nodes != self.num_nodes or other.nvars != self.nvars:
                raise ValueError("Operands must have the same num_nodes and nvars!")
            new._comps = [
                op(a, b) for a, b in zip(self._comps, other._comps, strict=False)
            ]
        else:
            new._comps = [op(a, other) for a in self._comps]
        return new

    def apply_matrix(self, Q):
        """
        Apply a matrix along the node axis, i.e., (Q u)[m] = sum_k Q[m, k] u[k].

        Parameters
        ----------
        Q : array_like
            Matrix of shape ``(num_nodes, num_nodes)``, e.g., the quadrature
            matrix.

        Returns
        -------
        Solution
            New object of the same type as ``self``.
        """
        new = self.copy()
        for m in range(1, self.num_nodes + 1):
            new._comps[m][...] = sum(
                Q[m - 1, k - 1] * self._comps[k] for k in range(1, self.num_nodes + 1)
            )
        return new
