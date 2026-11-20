from abc import ABC, abstractmethod

from daelab.base.datatypes.array import Array
from daelab.base.problems.dae_problem import DAEProblem


class FullimplicitDAEProblem(DAEProblem, ABC):
    r"""
    Base class for fully-implicit DAE Problems of the form

    .. math::
        0 = F(t, u(t), u'(t)),

    for :math:`t` in a finite interval. The function :math:`F` is the right-hand
    side of the DAE written as an implicit differential equation. The unknown
    function :math:`u` is the function to be sought. For this DAE type, there
    is no distinction between differential and algebraic variables.

    Parameters
    ----------
    nvars : tuple of integers
        Number of unknowns and spatial dimensions. The first entry is the
        number of unknowns, and the remaining entries are the number of
        spatial degrees of freedom in each dimension shared by all unknowns.
        For example, for nvars = (3, n_x, n_y), the problem has three
        unknowns, with n_x degrees of freedom in x-direction and n_y degrees
        of freedom in y_direction.
    dtype : type
        Datatype of the solution. Default is `float`.
    diff_index : int or None
        Differentiation index of the DAE problem.
    t_span : tuple[float, float] or None
        Time interval `(t0, t_end)` on which the problem is defined.
        If `None`, no specific time interval is prescribed.
    name : str or None
        Name of the problem. If `None`, the class name is used.

    Attributes
    ----------
    init : tuple
        ``(nvars, dtype)`` with nvars = (n_unknowns, *spatial) and the data type
        of the entries.
    nvars : tuple of integers
        Number of unknowns and spatial dimensions.
    is_semiexplicit_dae : bool
        Indicates whether the problem is of semi-explicit form. Set to `False`.

    Raises
    ------
    TypeError
        If `diff_index` is bool or not an integer.
    ValueError
        If `t_span` does not contain exactly two values or
        if `t_end` is not greater than `t0`.
    """

    dtype_u = Array
    dtype_f = Array

    dtype_u = Array
    dtype_f = Array

    def __init__(
        self,
        nvars: tuple[int, ...],
        dtype: type = float,
        diff_index: int | None = None,
        t_span: tuple[float, float] | None = None,
        name: str | None = None,
    ):
        """Initializes a full-implicit DAE Problem."""

        super().__init__(
            dtype=dtype,
            diff_index=diff_index,
            t_span=t_span,
            name=name,
        )

        self.init = (nvars, dtype)
        self.is_semiexplicit_dae = False

        if isinstance(nvars, int):
            nvars = (nvars,)

        self.validate_nvars(nvars)
        self.nvars = nvars

    @abstractmethod
    def F(self, u_dot, u, t):
        """
        Evaluate the DAE function `F(u_dot, u, t) = 0`.

        Parameters
        ----------
        u_dot : Array
            Derivative of the numerical solution.
        u : Array
            Numerical solution.
        t : float
            Current time of numerical solution.

        Returns
        -------
        Array
            Evaluated right-hand side of differential equation
            at current time.

        Raises
        ------
        NotImplementedError
            If the subclasses did not implemented the method.
        """
        raise NotImplementedError("Subclasses must implement F(u_dot, u, t).")

    def u_exact(self, t):
        """
        Computes the exact solution of `u` at time `t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        Array
            Exact solution of `u` at time `t`.

        Raises
        ------
        NotImplementedError
            If the subclasses did not implemented the method.
        """
        raise NotImplementedError("This problem does not provide an exact solution.")
