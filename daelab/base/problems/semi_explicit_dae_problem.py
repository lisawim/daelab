from abc import ABC, abstractmethod

from daelab.base.datatypes.semiexplicit import SemiexplicitArray
from daelab.base.problems.dae_problem import DAEProblem


class SemiexplicitDAEProblem(DAEProblem, ABC):
    r"""
    Base class for semi-explicit DAE Problems of the form

    .. math::
        y'(t) = f(y(t), z(t), t),

    .. math::
        0 = g(y(t), z(t), t),

    for :math:`t` in a finite interval. The function :math:`f` is the right-hand
    side of the differential equations and :math:`g` is the right-hand side of
    the algebraic constraints. The functions :math:`y` and :math:`z` are the
    unknown differential and algebraic variables.

    Parameters
    ----------
    nvars_diff : tuple of integers
        Number of unknown differential variables and spatial dimensions.
    nvars_alg : tuple of integers
        Number of unknown algebraic variables and spatial dimensions.
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
        ``(nvars_diff, nvars_alg, dtype)`` with ``nvars_diff`` and ``nvars_alg``
        of the form ``(n_unknowns, *spatial)`` with coinciding spatial
        dimensions, and the data type of the entries.
    nvars_diff : tuple of integers
        Number of unknown differential variables and spatial dimensions.
    nvars_alg : tuple of integers
        Number of unknown algebraic variables and spatial dimensions.
    is_semiexplicit_dae : bool
       Indicates whether the problem is of semi-explicit form. Set to `True`.

    Raises
    ------
    TypeError
        If `diff_index` is bool or not an integer.
    ValueError
        If `t_span` does not contain exactly two values or
        if `t_end` is not greater than `t0`.
    """

    dtype_u = SemiexplicitArray
    dtype_f = SemiexplicitArray

    def __init__(
        self,
        nvars_diff: tuple[int, ...],
        nvars_alg: tuple[int, ...],
        dtype: type = float,
        diff_index: int | None = None,
        t_span: tuple[float, float] | None = None,
        name: str | None = None,
    ):
        """Initializes a semi-explicit DAE Problem."""
        super().__init__(
            dtype=dtype,
            diff_index=diff_index,
            t_span=t_span,
            name=name,
        )

        self.init = (nvars_diff, nvars_alg, dtype)
        self.is_semiexplicit_dae = True

        if isinstance(nvars_diff, int):
            nvars_diff = (nvars_diff,)

        if isinstance(nvars_alg, int):
            nvars_alg = (nvars_alg,)

        if nvars_diff[1:] != nvars_alg[1:]:
            raise ValueError(
                "Differential and algebraic variables must have the same "
                "spatial dimensions"
            )

        self.validate_nvars(nvars=nvars_diff)
        self.nvars_diff = nvars_diff

        self.validate_nvars(nvars=nvars_alg)
        self.nvars_alg = nvars_alg

    @abstractmethod
    def f(self, u, t):
        """
        Evaluate the differential equation `f(u, t)`.

        Parameters
        ----------
        u : SemiexplicitArray
            Numerical solution of differential and algebraic variables.
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
        raise NotImplementedError("Subclasses must implement f(u, t).")

    @abstractmethod
    def g(self, u, t):
        """
        Evaluate the algebraic equation `g(u, t)`.

        Parameters
        ----------
        u : SemiexplicitArray
            Numerical solution of differential and algebraic variables.
        t : float
            Current time of numerical solution.

        Returns
        -------
        Array
            Evaluated right-hand side of algebraic equation
            at current time.

        Raises
        ------
        NotImplementedError
           If the subclasses did not implemented the method.
        """
        raise NotImplementedError("Subclasses must implement g(u, t).")

    def solve_system(self, factor, u0, t, rhs):
        """
        Solves the implicit system.

        Parameters
        ----------
        factor : float
            Local stepsize or any other factor required for quadrature.
        u0 : SemiexplicitArray
            Initial guess for `u`.
        t : float
            Current time.
        rhs : SemiexplicitArray
            Right-hand side for the implicit system.

        Returns
        -------
        SemiexplicitArray
            Solution `u` of the system.

        Raises
        ------
        NotImplementedError
            If the subclasses did not implemented the method.
        """
        raise NotImplementedError("Subclasses must implement solve_system().")

    def u_exact(self, t):
        """
        Computes the exact solution of `y` and `z` at time `t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        SemiexplicitArray
            Exact solution of `y` and `z` at time `t`.

        Raises
        ------
        NotImplementedError
            If the subclasses did not implemented the method.
        """
        raise NotImplementedError(
            "This problem does not provide an exact solution for u."
        )
