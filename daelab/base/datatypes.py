import numpy as np
import operator


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
      ``num_nodes``entries, each of them is of shape ``nvars`` and of data type
      ``dtype``. The solution ``u`` at node ``m`` can be accessed using ``u[m]``
      and ``u[m][i]`` is the i-th unknown at node ``m``.
    - ``u[m]`` returns a reference, not a copy. Modifying it, e.g.,
      ``u[m][i][:] = 1.0``, changes the stored solution.
    - Arithmetic operations with another solution of type ``Solution`` requires
      ``num_nodes`` and ``nvars`` of both solutions to be equal. Scalars and
      arrays that broadcast to ``nvars`` are applied to every node.

    Examples
    --------
    >>> u = Solution(num_nodes=4, nvars=(4, 4))
    >>> len(u)
    4
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
    (4, 16)
    """

    def __init__(self, num_nodes, nvars, dtype=float):
        """Initialization routine."""
        self.num_nodes = num_nodes
        self.nvars = nvars
        self.dtype = dtype
        self._comps = [
            np.zeros(self.nvars, dtype=self.dtype) for _ in range(self.num_nodes)
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
        for m in range(self.num_nodes):
            new._comps[m][...] = sum(
                Q[m, k] * self._comps[k] for k in range(self.num_nodes)
            )
        return new


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
    >>> u = SemiexplicitSolution(
    ...     num_nodes=3, nvars_diff=(2, 4, 5), nvars_alg=(1, 4, 5)
    ... )
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


class RHS(Solution):
    """
    Right-hand side of a fullimplicit DAE problem at all collocation nodes.

    A fully-implicit DAE is given in residual form F(t, u, u') = 0. The class
    stores the residual F at each collocation node. It has the same structure
    as `Solution`, i.e., ``F[m]`` is the residual at node ``m`` and has shape
    ``nvars``, since a DAE has as many equations as unknowns. The separate class
    only marks the object as a right-hand side and distinguishes it from a
    solution.

    All parameters, attributes and methods are inherited from `Solution`.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes.
    nvars : tuple of integers
        Number of unknowns (first entry), and spatial dimensions shared by all
        unknowns (remaining) entries.
    dtype : data-type, optional
        Data type of the entries. Default is float.

    Notes
    -----
    Solutions and right-hand sides can be combined in arithmetic expressions,
    e.g., for SDC. The result has the type of the left operand.

    Examples
    --------
    >>> F = RHS(num_nodes=3, nvars=(2, 4))
    >>> F[1].shape
    (2, 4)
    >>> F[1] = 1.0
    >>> u = Solution(num_nodes=3, nvars=(2, 4))
    >>> u[:] = 2.0
    >>> w = u + 0.5 * F
    >>> type(w).__name__, w[1][0, 0]
    'Solution', 2.5
    >>> type(2 * F).__name__
    'RHS'
    """


class IMEXRHS(PartitionedContainer):
    """
    Right-hand side for imex-splitting for each collocation node.

    The right-hand side of a problem can be splitted into implicit and explicit
    parts. Each component is a `RHS` and is available as attribute ``impl`` and
    ``expl``. The class provides access to all parts at a node, assignment,
    copying, and arithmetic, where every operation is applied part by part.

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
        Number of unknowns (first entry), and spatial dimensions (remaining
        entries) shared by all unknowns.
    dtype : data-type, optional
        Data type of the entries. Default is float.
    impl : RHS
       Right-hand side for implicit treatment.
    expl : RHS
        Right-hand side for implicit treatment.

    Notes
    -----
    - For implicit and explicit parts, the right-hand side at each collocation
      node is stored in a list that has ``num_nodes``entries, each of them is
      of shape ``nvars`` and of data type ``dtype``. Using ``f[m]`` the
      right-hand side ``f`` at node ``m`` is accessed at both parts, ``impl``
      and ``expl``. The implicit and explicit parts of the right-hand side at
      node ``m`` can be accessed using ``f.impl[m]`` and ``f.expl[m]``,
      respectively.

    Examples
    --------
    >>> f = IMEXRHS(num_nodes=4, nvars=(2, 4))
    >>> len(f.impl), len(f.expl)
    4, 4
    >>> f.impl[1].shape
    (2, 4)
    >>> f.impl[1] = 2 * np.ones((2, 4))
    >>> f.expl[1] = np.ones((2, 4))
    >>> g1 = f.impl[1] + 10 * f.expl[1]
    [[12. 12. 12. 12.]
    [12. 12. 12. 12.]]
    >>> f.full_rhs
    RHS(num_nodes=4, nvars=(2, 4), dtype=float64)
    """

    _components = ("impl", "expl")

    def __init__(self, num_nodes, nvars, dtype=float):
        """Initialization routine."""
        self.num_nodes = num_nodes
        self.nvars = nvars
        self.dtype = dtype
        self.impl = RHS(num_nodes=self.num_nodes, nvars=self.nvars, dtype=self.dtype)
        self.expl = RHS(num_nodes=self.num_nodes, nvars=self.nvars, dtype=self.dtype)

    @property
    def full_rhs(self):
        """
        Returns the full right-hand side.

        Returns
        -------
        RHS
            Full right-hand side as sum of implicit and explicit parts.
        """
        return self.impl + self.expl

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


class SemiexplicitRHS(SemiexplicitSolution):
    """
    Right-hand side with splitting into differential and algebraic equation at
    each collocation node.

    Semi-explicit DAEs have differential equations subject to algebraic
    constraints. Each equation type component is a `RHS` and is available as
    attribute ``diff`` and ``alg``. In this class, the right-hand side of the
    differential equation  and the algebraic constraints  at each node are
    stored as a list of numpy arrays, and each array is of shape ``nvars_diff``
    and ``nvars_alg``, respectively. The class provides access to all parts at
    a node, assignment, copying, and arithmetic, where every operation is
    applied part by part. In addition, the right-hand side of the differential
    equation ``f.diff`` can use an imex-splitting by setting ``imex_diff=True``.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes.
    nvars_diff : tuple of integers
        Number of unknowns of differential variables (first entry), and spatial
        dimensions shared by all unknowns (remaining) entries.
    nvars_diff : tuple of integers
        Number of unknowns of algebraic variables (first entry), and spatial
        dimensions shared by all unknowns (remaining) entries.
    dtype : data-type, optional
        Data type of the entries. Default is float.
    imex_diff : bool, optional
        Indicates whether right-hand side of differential equation should use an
        imex splitting. Default is False.

    Attributes
    ----------
    num_nodes : int
        Number of collocation nodes.
    nvars_diff : tuple of integers
        Shape of solution of differential variables at single collocation node.
    nvars_alg : tuple of integers
        Shape of solution of algebraic variables at single collocation node.
    dtype : data-type, optional
        Data type of the entries. Default is float.
    imex_diff : bool, optional
        Flat to choose imex splitting.
    diff : RHS or IMEXRHS
        Right-hand side of differential equation.
    alg : RHS
        Right-hand side of algebraic equation.

    Examples
    --------
    >>> f = SemiexplicitRHS(
    ...     num_nodes=3, nvars_diff=(2, 4), nvars_alg=(1, 4), imex_diff=True
    ... )
    >>> f.diff
    IMEXRHS(num_nodes=3, nvars=(2, 4), dtype=float64)
    >>> f.diff.impl
    RHS(num_nodes=3, nvars=(2, 4), dtype=float64)
    >>> f.diff.impl[2] = 0.25 * np.ones((2, 4))
    >>> f.diff.expl[2] = 0.33 * np.ones((2, 4))
    >>> f.diff.full_rhs[2]
    [[0.58 0.58 0.58 0.58]
    [0.58 0.58 0.58 0.58]]
    """

    def __init__(self, num_nodes, nvars_diff, nvars_alg, dtype=float, imex_diff=False):
        super().__init__(num_nodes, nvars_diff, nvars_alg, dtype)
        self.imex_diff = imex_diff
        self.diff = (
            IMEXRHS(self.num_nodes, self.nvars_diff, self.dtype)
            if self.imex_diff
            else RHS(self.num_nodes, self.nvars_diff, self.dtype)
        )
        self.alg = RHS(self.num_nodes, self.nvars_alg, self.dtype)
