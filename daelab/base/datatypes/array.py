import numpy as np

from daelab.base.datatypes.arithmetic import ArithmeticMethods


def _assign(target, value):
    """
    Writes `value` into `target` (in-place).

    Parameters
    ----------
    target : np.ndarray or Array
        The array where ``value`` should be written into.
    value : scalar or np.ndarray
        The value to be written into the array.
    """
    if isinstance(target, np.ndarray):
        target[...] = value
    else:
        target.assign(value)


class Array(ArithmeticMethods):
    """
    Base class for arrays of general problems.

    Using the ``init`` parameter ``init = (nvars, dtype)`` that is passed
    a numpy array of shape ``nvars`` and dtype ``dtype`` with zero values is
    initialized. It is stored as attribute ``values`` and represents the
    datatype for one node during the simulation of numerical methods.

    Parameters
    ----------
    init : tuple
        ``(nvars, dtype)`` with nvars = (n_unknowns, *spatial) and the data type
        of the entries.

    Attributes
    ----------
    values : np.ndarray
        The stored array of shape ``nvars``, initialized with zeros.

    Examples
    --------
    >>> a = Array(((2, 8), float))
    >>> a.shape
    (2, 8)
    >>> len(a)  # Length is equal to unknowns
    2
    >>> a[...] = 2.0
    >>> np.all(a == 2.0)
    True
    """

    def __init__(self, init):
        """Initialization routine."""
        nvars, dtype = init
        self.values = np.zeros(nvars, dtype=dtype)

    @property
    def shape(self):
        """
        Property shape of the array.

        Returns
        -------
        tuple
            The shape of the array.
        """
        return self.values.shape

    @property
    def dtype(self):
        """
        Property dtype of the array.

        Returns
        -------
        type
            The dtype of the array.
        """
        return self.values.dtype

    def __len__(self):
        """
        Returns the length of the array that is equal to the number of unknowns.

        Returns
        -------
        int
            Number of unknowns.
        """
        return len(self.values)

    def __getitem__(self, i):
        """
        Returns values at entry ``i`` of the array.

        Parameters
        ----------
        i : int
            Index of the array to access.

        Returns
        -------
        dtype or np.ndarray
            Values at entry ``i``.
        """
        return self.values[i]

    def __setitem__(self, i, val):
        """
        Sets ``val`` at entry ``i`` in the array.

        Parameters
        ----------
        i : int or slice
            Index of the array.
        val : dtype or np.ndarray of dtype
            Value to be set at the entry.
        """
        self.values[i] = val

    def __array__(self, dtype=None, copy=None):
        """
        Copies the array with another dtype.

        Parameters
        ----------
        dtype : type, optional
            Dtype of the array copy.
        copy : bool, optional
            Indicates if the array is copied or the returned array is a view
            of the original array.

        Returns
        -------
        np.ndarray
            If ``copy = False``, the returned array is a view of the original,
            otherwise it is a copy.

        Raises
        ------
        ValueError
            If ``copy`` is set to False and ``dtype`` differs from the dtype
            of ``self``.
        """
        if copy:
            return np.array(self.values, dtype=dtype, copy=True)
        if copy is False and dtype is not None and np.dtype(dtype) != self.dtype:
            raise ValueError("Unable to avoid a copy while converting to this dtype.")
        return np.asarray(self.values, dtype=dtype)

    def copy(self):
        """
        Creates a copy of the same type as the original.

        Returns
        -------
        Array
            The copy with the same values as the original.
        """
        new = type(self).__new__(type(self))
        new.values = self.values.copy()
        return new

    def assign(self, value):
        """
        Writes value into the stored array in-place.

        Parameters
        ----------
        value : Array or other
            The value for assignment.
        """
        self.values[...] = value.values if isinstance(value, Array) else value

    def flat(self):
        """
        Flats values to a flat vector of length ``np.prod(nvars)`` (view if
        possible).

        Returns
        -------
        np.1darray
            The array in C-order.
        """
        return self.values.reshape(-1, order="C")

    def _binary(self, other, op):
        """
        Applies an operation ``op`` to ``self`` and ``other`` and returns the
        result.

        Parameters
        ----------
        other : Array
            Second operand.
        op : callable
            Operation.

        Returns
        -------
        new : Solution
            Result of the same type as ``self`` of performed operation.

        Raises
        ------
        NotImplementedError
            If ``other`` is of type ``ArithmeticMethods`` and not of type
            ``Array``.
        """
        if isinstance(other, ArithmeticMethods) and not isinstance(other, Array):
            raise NotImplementedError
        new = self.copy()
        new.values = op(
            self.values, other.values if isinstance(other, Array) else other
        )
        return new
