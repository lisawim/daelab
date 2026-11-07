import numpy as np

from daelab.base.datatypes.solution import ArithmeticMethods, Solution


class PartitionedContainer(ArithmeticMethods):
    """
    Base class for containers that consists of named parts (the components).

    Each component is a `Solution` (or again a `PartitionedContainer`) and is
    available as attribute with the name given in ``_components``. The class
    provides access to all parts at a node, assignment, copying, and
    arithmetic, where every operation is applied part by part. It is the common
    base of `SemiexplicitSolution` (parts ``diff`` and ``alg``) and `IMEXRHS`
    (parts ``impl`` and ``expl``).

    Attributes
    ----------
    num_nodes : int
        Number of collocation nodes.

    Notes
    -----
    A subclass has to
    - define ``_components``,
    - set ``num_nodes`` and one attribute for each name in ``_components`` in
      its ``__init__``. All parts must have the same number of nodes.
    - ``obj[m]`` returns a tuple with the entries of all parts at node ``m``.
      The entries are references, so they can be modified in-place.
    - Arithmetic operations between two containers requires the same
      ``_components`` and ``num_nodes`` (otherwise, ``ValueError`` is raised).
      A scalar is applied to each part. Combining a container with a `Solution`
      raises a ``TypeError``, since it is not clear to which part it refers.
    - Since parts may be containers themselves, partitioned objects can be
      nested.

    Examples
    --------
    >>> class Pair(PartitionedContainer):
    ...     _components = ("a", "b")
    ...
    ...     def __init__(self, num_nodes, nvars_a, nvars_b):
    ...         self.num_nodes = num_nodes
    ...         self.a = Solution(num_nodes, nvars_a)
    ...         self.b = Solution(num_nodes, nvars_b)
    >>> p = Pair(2, (1, 3), (2, 3))
    >>> p.a[0] = 1.0
    >>> a0, b0 = p[0]
    >>> a0.shape, b0.shape
    (1, 3), (2, 3)
    >>> p[1] = (2.0, 3.0)
    >>> p.a[1]
    [[2. 2. 2.]]
    >>> p.b[1]
    [[3. 3. 3.]
    [3. 3. 3.]]
    >>> q = 2 * p
    >>> q.a[0][0, 0], q.b[0][0, 0]
    2.0, 0.0
    >>> r = p.copy()
    >>> r.a[0] = 0.0
    >>> p.a[0][0, 0], r.a[0][0, 0]
    1.0, 0.0
    """

    _components = ()

    @property
    def _parts(self):
        """Returns the list of all components."""
        return [getattr(self, name) for name in self._components]

    def __getitem__(self, m):
        """
        Returns the solution entries of all parts at node ``m``.

        Parameters
        ----------
        m : int
            Index of the collocation node.

        Returns
        -------
        tuple
            One entry per part.
        """
        return tuple(part[m] for part in self._parts)

    def __setitem__(self, m, value):
        """
        Writes ``val`` into solution at all parts at node ``m``.

        Parameters
        ----------
        m : int
            Index of the collocation node.
        value : sequence
            One value per part

        Raises
        ------
        ValueError
            If the number of values does not match the number of parts.
        """
        if len(value) != len(self._components):
            raise ValueError(
                f"Expected {len(self._components)} values: {self._components}."
            )
        for part, val in zip(self._parts, value, strict=False):
            part[m] = val

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
            If ``other`` is a `Solution`.
        """
        new = self.copy()
        if isinstance(other, PartitionedContainer):
            if (
                other._components != self._components
                or other.num_nodes != self.num_nodes
            ):
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
        """
        Apply a matrix along the node axis to each component.

        Parameters
        ----------
        Q : array_like
            Matrix of shape ``(num_nodes, num_nodes)``, e.g., the quadrature
            matrix.

        Returns
        -------
        PartitionedContainer
            New object of the same type as ``self``.
        """
        new = self.copy()
        for name in self._components:
            setattr(new, name, getattr(self, name).apply_matrix(Q))
        return new


class SemiexplicitSolution(PartitionedContainer):
    """
    Solution datatype of a semiexplicit DAE at all collocation nodes.

    For differential and algebraic parts, the class stores one numpy array per
    collocation node as a `Solution` in the corresponding attributes. At each
    node, the numpy array of differential parts is of shape ``nvars_diff`` and
    the numpy array of algebraic parts is of shape ``nvars_alg``. Spatial
    dimensions of both parts needs to be equal (otherwise, a ``ValueError`` is
    raised). The operators ``+``, ``-``, ``*``, ``/`` and the unary ``-`` are
    made available through inheritance in this class.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes.
    nvars : tuple of integers
        Number of unknowns (first entry), and spatial dimensions shared by all
        unknowns (remaining) entries.
    nvars_diff : tuple of integers
        Number of unknowns of differential variables (first entry), and spatial
        dimensions shared by all unknowns (remaining) entries.
    nvars_diff : tuple of integers
        Number of unknowns of algebraic variables (first entry), and spatial
        dimensions shared by all unknowns (remaining) entries.
    dtype : data-type, optional
        Data type of the entries. Default is float.

    Attributes
    ----------
    num_nodes : int
        Number of collocation nodes.
    nvars_diff : tuple of integers
        Shape of solution of differential variables at single collocation node.
    nvars_alg : tuple of integers
        Shape of solution of algebraic variables at single collocation node.
    dtype : data-type
        Data type of the entries.
    diff : Solution
        Differential part of solution.
    alg : Solution
        Algebraic part of solution.

    Notes
    -----
    - For differential and algebraic parts, the solution at each collocation
      node is stored in a list that has ``num_nodes``entries, each of them
      is of shape ``nvars_diff`` or ``nvars_alg`` and of data type ``dtype``.
      Using ``u[m]`` the solution ``u`` at node ``m`` is accessed at both parts,
      ``diff`` and ``alg``. The differential and algebraic parts of the solution
      at node ``m``can be accessed using ``u.diff[m]`` and ``u.alg[m]``,
      respectively.

    Examples
    --------
    >>> u = SemiexplicitSolution(3, (2, 4, 5), (1, 4, 5))
    >>> u.diff[1].shape, u.alg.shape[1]
    (2, 4, 5), (1, 4, 5)
    >>> u.diff[1][0][:] = 1.0
    >>> y, z = u[1]
    >>> y[0, 0, 0], z[0, 0, 0]
    (1.0, 0.0)
    >>> u[2] = (np.ones((2, 4, 5)), 2 * np.ones((1, 4, 5)))
    >>> u.diff[2][0, 0, 0], u.alg[2][0, 0, 0]
    1.0, 2.0
    >>> v = 2 * u + u
    >>> v.diff[2][0, 0, 0], v.alg[2][0, 0, 0]
    3.0, 6.0
    """

    _components = ("diff", "alg")

    def __init__(self, num_nodes, nvars_diff, nvars_alg, dtype=float):
        """Initialization routine."""
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
        """
        Returns the class with inputs as string.

        Returns
        str
            Class description.
        """
        return (
            f"{type(self).__name__}(num_nodes={self.num_nodes}, "
            f"nvars_diff={self.nvars_diff}, "
            f"nvars_alg={self.nvars_alg}, dtype={np.dtype(self.dtype)})"
        )
