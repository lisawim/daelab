import numpy as np

from daelab.base.datatypes.arithmetic import ArithmeticMethods
from daelab.base.datatypes.array import _assign, Array
from daelab.base.datatypes.node_list import NodeList


class PartitionedArray(ArithmeticMethods):
    """
    Base class for arrays that consists of named parts (the components).

    Each component is a `Array` (or again a `PartitionedArray`) and is
    available as attribute with the name given in ``_components``. The class
    provides access to all parts at a node, assignment, copying, and
    arithmetic, where every operation is applied part by part. It is the common
    base of `SemiexplicitArray` (parts ``diff`` and ``alg``) and `IMEXArray`
    (parts ``impl`` and ``expl``).

    Notes
    -----
    A subclass has to
    - define ``_components``,
    - set one attribute for each name in ``_components`` in its ``__init__``.
    - Arithmetic operations between two arrays requires the same
      ``_components`` (otherwise, ``ValueError`` is raised). A scalar is applied
      to each part. Combining an array with a `np.ndarray` or a `NodeList`
      raises a ``TypeError``, since it is not clear to which part it refers.
    - Since parts may be arrays themselves, partitioned objects can be
      nested.

    Examples
    --------
    >>> class Pair(PartitionedArray):
    ...     _components = ("a", "b")
    ...
    ...     def __init__(self, init):
    ...         nvars, dtype = init
    ...         self.nvars = tuple(nvars)
    ...         self.a = Array(init)
    ...         self.b = Array(init)
    >>> p = Pair(((2, 3), float))
    >>> p.a.shape, p.b.shape
    (2, 3), (2, 3)
    >>> p.assign((2.0, 3.0))
    >>> p.a[:]
    [[2. 2. 2.]
    [2. 2. 2.]]
    >>> p.b[:]
    [[3. 3. 3.]
    [3. 3. 3.]]
    >>> q = 2 * p
    >>> q.a[0, 0], q.b[0, 0]
    4.0, 6.0
    >>> r = p.copy()
    >>> r.a[...] = 0.0
    >>> p.a[0, 0], r.a[0, 0]
    2.0, 0.0
    """

    _components = ()

    @property
    def _parts(self):
        """
        Returns the list of all parts.

        Returns
        -------
        list
            List that contains the parts of a class stored in ``_components``.
        """
        return [getattr(self, name) for name in self._components]

    def __iter__(self):
        """
        Creates an iterator object of the parts list.

        Returns
        -------
            Iterator object to traverse the elements of the parts list.
        """
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
        PartitionedArray
            Copy of an object including the same parts.
        """
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        for name in self._components:
            setattr(new, name, getattr(self, name).copy())
        return new

    def flat(self):
        """
        Returns all parts as a flat vector of shape ``(n,)`` with
        ``n = sum_i prod(nvars_i)``.

        Returns
        -------
        np.ndarray
            Array of shape.
        """
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
    """
    Base class for arrays with imex-splitting.

    The array has the components `impl`` and ``expl`` and is used to split the
    right-hand side of a problem  as ``f = f.impl + f.expl``. Each component
    is a `Array` and is available as attribute ``impl`` and ``expl``. The class
    provides assignment, copying, flattening, and arithmetic, where every
    operation is applied part by part.

    Parameters
    ----------
    init : tuple
        ``(nvars, dtype)`` with nvars = (n_unknowns, *spatial) and the data type
        of the entries.

    Attributes
    ----------
    nvars : tuple of integers
        Number of unknowns (first entry), and spatial dimensions (remaining
        entries) shared by all unknowns.
    impl : Array
       Array for implicit treatment.
    expl : Array
        Array for implicit treatment.

    Examples
    --------
    >>> f = IMEXArray(((2, 4), float))
    >>> len(f.impl), len(f.expl)
    2, 2
    >>> f.impl.shape
    (2, 4)
    >>> f.impl[...] = 2 * np.ones((2, 4))
    >>> f.expl[...] = np.ones((2, 4))
    >>> g1 = f.impl + 10 * f.expl
    [[12. 12. 12. 12.]
    [12. 12. 12. 12.]]
    >>> f.full_rhs
    [[3. 3. 3. 3.]
    [3. 3. 3. 3.]]
    """

    _components = ("impl", "expl")

    def __init__(self, init):
        """Initialization routine."""
        nvars, dtype = init
        self.nvars = tuple(nvars)
        self.impl = Array(init)
        self.expl = Array(init)

    @property
    def full_rhs(self):
        return self.impl + self.expl
