from daelab.base.datatypes.array import Array
from daelab.base.datatypes.partitioned import PartitionedArray, IMEXArray


class SemiexplicitArray(PartitionedArray):
    """
    Base class for arrays for semiexplicit DAE problems.

    For differential and algebraic parts, the class stores an array as a
    `Array` in the corresponding attributes. The ``diff``array is of shape
    ``nvars_diff`` and the ``alg`` array is of shape ``nvars_alg``. The parts
    are initialized by passing ``init = (nvars_diff, nvars_alg, dtype)``.
    Spatial dimensions of both parts need to be equal (otherwise, a
    ``ValueError`` is raised). The operators ``+``, ``-``, ``*``, ``/`` and the
    unary ``-`` are made available through inheritance in this class.

    Parameters
    ----------
    init : tuple
        ``(nvars_diff, nvars_alg, dtype)`` with ``nvars_diff`` and ``nvars_alg``
        of the form ``(n_unknowns, *spatial)`` with coinciding spatial
        dimensions, and the data type of the entries.

    Attributes
    ----------
    nvars_diff : tuple of integers
        Shape of array of differential variables.
    nvars_alg : tuple of integers
        Shape of array of algebraic variables.
    dtype : type
        Data type of the entries.
    diff : Array
        Array representing the differential part.
    alg : Array
        Array representing the algebraic part.

    Examples
    --------
    >>> u = SemiexplicitArray(((2, 4, 5), (1, 4, 5), float))
    >>> u.diff.shape, u.alg.shape
    (2, 4, 5), (1, 4, 5)
    >>> u.diff[0][:] = 1.0
    >>> y, z = u
    >>> y[0, 0, 0], z[0, 0, 0]
    (1.0, 0.0)
    """

    _components = ("diff", "alg")
    dtype_diff = Array
    dtype_alg = Array

    def __init__(self, init):
        """Initialization routine."""
        nvars_diff, nvars_alg, dtype = init
        if tuple(nvars_diff[1:]) != tuple(nvars_alg[1:]):
            raise ValueError(
                "nvars_diff and nvars_alg must have the same spatial dimensions."
            )
        self.nvars_diff = tuple(nvars_diff)
        self.nvars_alg = tuple(nvars_alg)
        self.dtype = dtype
        self.diff = self.dtype_diff((self.nvars_diff, dtype))
        self.alg = self.dtype_alg((self.nvars_alg, dtype))


class SemiexplicitIMEXArray(SemiexplicitArray):
    """
    Base class for arrays for semiexplicit DAE problems with splitted
    right-hand side of the differential equation in IMEX fashion.

    For semi-explicit DAEs of the form

    .. math::
        y'(t) = f(y(t), z(t), t),

    .. math::
        0 = g(y(t), z(t), t),

    the right-hand side :math:`f` can be splitted into stiff and nonstiff terms
    leading to the splitting :math:`f = f_I + f_E`, where :math:`f_I` represents
    the terms to be treated implicitly, and :math:`f_E` are the terms that will
    be treated explicitly. Using this class, this splittting is realized through
    ``f = f.impl + f.expl``. In addition, the right-hand sides :math:`f` and
    :math:`g` are represented as arrays under the components ``diff`` and
    ``alg``. Since each component of these parts are a `Array` themselves,
    ``f.diff`` is again nested to realize the IMEX splitting via
    ``f.diff = f.diff.impl + f.diff.expl``. The class provides assignment,
    copying, flattening, and arithmetic, where every operation is applied part
    by part.

    Parameters
    ----------
    init : tuple
        ``(nvars, dtype)`` with nvars = (n_unknowns, *spatial) and the data type
        of the entries.

    Examples
    --------
    >>> f = SemiexplicitIMEXArray(((2, 3, 4), (1, 3, 4), float))
    >>> len(f.alg)
    1
    >>> len(f.diff.impl), len(f.diff.expl)
    2, 2
    >>> f.diff.impl.shape, f.diff.expl.shape, f.alg.shape
    (2, 3, 4), (2, 3, 4), (1, 3, 4)
    >>> f.diff.impl[0, ...] = 0.25 * np.ones((3, 4))
    >>> f.diff.expl[0, ...] = 0.33 * np.ones((3, 4))
    >>> f.diff.full_rhs[0]
    [[0.58 0.58 0.58 0.58]
    [0.58 0.58 0.58 0.58]
    [0.58 0.58 0.58 0.58]]
    """

    dtype_diff = IMEXArray
