import numpy as np

from daelab.base.problems.semi_explicit_dae_problem import SemiexplicitDAEProblem


class LinearManufacturedDAE(SemiexplicitDAEProblem):
    r"""
    Linear manufactured DAE of semi-explicit form

    .. math::
        y'(t) = -2 y(t) + z(t),

    .. math::
        0 = -2 y(t) - z(t),

    for :math:`t \in [0.0, 1.0]` with differential variable :math:`y` and
    algebraic variable :math:`y`. The exact solution is given by

    .. math::
        (y(t), z(t)) = (e^{-4 t}, -2 e^{-4 t}),

    so initial conditions are obtained by :math:`(y(0.0), z(0.0)) = (y_0, z_0)`.

    Parameters
    ----------
    nvars_diff : tuple of integers, optional
        Number of unknown differential variables. It is set to `(1,)`.
    nvars_alg : tuple of integers, optional
        Number of unknown algebraic variables. It is set to `(1,)`.
    dtype : type, optional.
        Datatype of the values. Default is `float`.
    diff_index : int, optional
        Differentiation index of the DAE. It is given by `1`.
    t_span : tuple, optional
        Time domain where the problem lives. Default is `(0.0, 1.0)`.
        The time interval can be increased, but this also results in an
        exponentially growing solution.
    name : str, optional
        Name of the problem.
    """

    def __init__(
        self,
        nvars_diff=(1,),
        nvars_alg=(1,),
        dtype=float,
        diff_index=1,
        t_span=(0.0, 1.0),
        name="Linear manufactured DAE",
    ):
        """Initialization routine."""
        super().__init__(
            nvars_diff=nvars_diff,
            nvars_alg=nvars_alg,
            dtype=dtype,
            diff_index=diff_index,
            t_span=t_span,
            name=name,
        )

    def f(self, y, z, t):
        """
        Evaluate the differential equation `f(y, z, t)`.

        Parameters
        ----------
        y : float
            Numerical solution of differential variable.
        z : float
            Numerical solution of algebraic variable.
        t : float
            Current time of numerical solution.

        Returns
        -------
        float
            Evaluated right-hand side of differential equation
            at current time.
        """
        return -2 * y + z

    def g(self, y, z, t):
        """
        Evaluate the algebraic equation `g(y, z, t)`.

        Parameters
        ----------
        y : float
            Numerical solution of differential variable.
        z : float
            Numerical solution of algebraic variable.
        t : float
            Current time of numerical solution.

        Returns
        -------
        float
            Evaluated right-hand side of algebraic equation
            at current time.
        """
        return -2 * y - z

    def solve_system(self, factor, y0, z0, t, rhs_diff, rhs_alg):
        """
        Solves the linear implicit system by a direct solve.

        Parameters
        ----------
        factor : float
            Local stepsize or any other factor required for quadrature.
        y0 : float
            Initial guess for `y`.
        z0 : float
            Initial guess for `z`.
        t : float
            Current time.
        rhs_diff : float
            Right-hand side for the linear system corresponding to the
            differential equation.
        rhs_alg : float
            Right-hand side for the linear system corresponding to the
            algebraic equation.

        Returns
        -------
        float
            Solution `y` of the linear system.
        float
            Solution `z` of the linear system.
        """
        b = np.array([rhs_diff, 0.0])
        C = np.array(
            [
                [1 + factor * 2.0, -factor * 1.0],
                [-2.0, -1.0],
            ]
        )

        u = np.linalg.solve(C, b)
        return u[0], u[1]

    def y_exact(self, t):
        """
        Computes the exact solution of `y` at time `t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        float
            Exact solution of `y` at time `t`.
        """
        return np.exp(-4 * t)

    def z_exact(self, t):
        """
        Computes the exact solution of `z` at time `t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        float
            Exact solution of `z` at time `t`.
        """
        return -2 * np.exp(-4 * t)
