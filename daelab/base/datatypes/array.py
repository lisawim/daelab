import numpy as np

from daelab.base.datatypes.arithmetic import ArithmeticMethods


def _assign(target, value):
    """Writes `value` into `target` (in-place)."""
    if isinstance(target, np.ndarray):
        target[...] = value
    else:
        target.assign(value)


class Array(ArithmeticMethods):
    def __init__(self, init):
        nvars, dtype = init
        self.values = np.zeros(nvars, dtype=dtype)

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]

    def __setitem__(self, i, val):
        self.values[i] = val

    def __array__(self, dtype=None, copy=None):
        if copy:
            return np.array(self.values, dtype=dtype, copy=True)
        if copy is False and dtype is not None and np.dtype(dtype) != self.dtype:
            raise ValueError("Unable to avoid a copy while converting to this dtype.")
        return np.asarray(self.values, dtype=dtype)

    def copy(self):
        new = type(self).__new__(type(self))
        new.values = self.values.copy()
        return new

    def assign(self, value):
        """Writes value into the stored array in-place."""
        self.values[...] = value.values if isinstance(value, Array) else value

    def flat(self):
        """Values as flat vector of length prod(nvars) (view if possible)."""
        return self.values.reshape(-1, order="C")

    def _binary(self, other, op):
        if isinstance(other, ArithmeticMethods) and not isinstance(other, Array):
            return NotImplemented
        new = self.copy()
        new.values = op(
            self.values, other.values if isinstance(other, Array) else other
        )
        return new
