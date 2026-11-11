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

    __array_ufunc__ = None

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
