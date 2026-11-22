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

    def f(self, u, t):
        """
        Evaluate the differential equation `f(u, t)`.

        Parameters
        ----------
        u : SemiexplicitArray
            Numerical solution of differential and algebraic variable.
        t : float
            Current time of numerical solution.

        Returns
        -------
        Array
            Evaluated right-hand side of differential equation
            at current time.
        """
        f = self.dtype_f(self.init)
        f.diff[:] = -2 * u.diff[:] + u.alg[:]
        return f.diff

    def g(self, u, t):
        """
        Evaluate the algebraic equation `g(u, t)`.

        Parameters
        ----------
        u : SemiexplicitArray
            Numerical solution of differential and algebraic variable.
        t : float
            Current time of numerical solution.

        Returns
        -------
        Array
            Evaluated right-hand side of algebraic equation
            at current time.
        """
        f = self.dtype_f(self.init)
        f.alg[:] = -2 * u.diff[:] - u.alg[:]
        return f.alg

    def solve_system(self, factor, u0, t, rhs):
        """
        Solves the linear implicit system by a direct solve.

        Parameters
        ----------
        factor : float
            Local stepsize or any other factor required for quadrature.
        u0 : SemiexplicitArray
            Initial guess for `u`.
        t : float
            Current time.
        rhs : SemiexplicitArray
            Right-hand side for the linear implicit system.

        Returns
        -------
        SemiexplicitArray
            Solution `u` of the linear system.
        """
        print(rhs.diff[:], rhs.alg[:], rhs.diff[:].shape)
        b = np.r_[rhs.diff[:], 0.0]
        C = np.array(
            [
                [1 + factor * 2.0, -factor * 1.0],
                [-2.0, -1.0],
            ]
        )

        u = np.linalg.solve(C, b)

        solution = self.dtype_u(self.init)
        solution[:] = (u[0], u[1])
        return solution

    def u_exact(self, t):
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
        u = self.dtype_u(self.init)
        u[:] = (np.exp(-4 * t), -2 * np.exp(-4 * t))
        return u
