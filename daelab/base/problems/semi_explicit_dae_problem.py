from abc import ABC, abstractmethod

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
        Time interval (t0, t_end) on which the problem is defined.
        If None, no specific time interval is prescribed.
    name : str or None
        Name of the problem. If None, the class name is used.

    Raises
    ------
    TypeError
        If diff_index is bool or not an integer.
    ValueError
        If t_span does not contain exactly two values or
        if t_end is not greater than t0.
    """

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
    def f(self, y, z, t):
        """
        Evaluate the differential equation `f(y, z, t)`.

        Parameters
        ----------
        y : np.ndarray
            Numerical solution of differential variables.
        z : np.ndarray
            Numerical solution of algebraic variables.
        t : float
            Current time of numerical solution.

        Returns
        -------
        float
            Evaluated right-hand side of differential equation
            at current time.

        Raises
        ------
        NotImplementedError
            If the subclasses did not implemented the method.
        """
        raise NotImplementedError("Subclasses must implement f(y, z, t).")

    @abstractmethod
    def g(self, y, z, t):
        """
        Evaluate the algebraic equation `g(y, z, t)`.

        Parameters
        ----------
        y : np.ndarray
            Numerical solution of differential variables.
        z : np.ndarray
            Numerical solution of algebraic variables.
        t : float
            Current time of numerical solution.

        Returns
        -------
        np.ndarray
            Evaluated right-hand side of algebraic equation
            at current time.

        Raises
        ------
        NotImplementedError
           If the subclasses did not implemented the method.
        """
        raise NotImplementedError("Subclasses must implement g(y, z, t).")

    def solve_system(self, factor, y0, z0, t, rhs_diff, rhs_alg):
        """
        Solves the implicit system.

        Parameters
        ----------
        factor : float
            Local stepsize or any other factor required for quadrature.
        y0 : np.ndarray
            Initial guess for `y`.
        z0 : np.ndarray
            Initial guess for `z`.
        t : float
            Current time.
        rhs_diff : np.ndarray
            Right-hand side for the linear system corresponding to the
            differential equation.
        rhs_alg : np.ndarray
            Right-hand side for the linear system corresponding to the
            algebraic equation.

        Returns
        -------
        np.ndarray
            Solution `y` of the system.
        np.ndarray
            Solution `z` of the system.

        Raises
        ------
        NotImplementedError
            If the subclasses did not implemented the method.
        """
        raise NotImplementedError("Subclasses must implement solve_system().")

    def y_exact(self, t):
        """
        Computes the exact solution of `y` at time `t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        np.ndarray
            Exact solution of `y` at time `t`.

        Raises
        ------
        NotImplementedError
            If the subclasses did not implemented the method.
        """
        raise NotImplementedError(
            "This problem does not provide an exact solution for y."
        )

    def z_exact(self, t):
        """
        Computes the exact solution of `z` at time `t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        np.ndarray
            Exact solution of `z` at time `t`.

        Raises
        ------
        NotImplementedError
            If the subclasses did not implemented the method.
        """
        raise NotImplementedError(
            "This problem does not provide an exact solution for z."
        )
