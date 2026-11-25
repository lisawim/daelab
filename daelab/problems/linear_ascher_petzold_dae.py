import numpy as np

from daelab.base.datatypes.array import Array
from daelab.base.problems.semi_explicit_dae_problem import SemiexplicitDAEProblem


class LinearAscherPetzoldDAE(SemiexplicitDAEProblem):
    """
    Linear index-2 problem of the form

    .. math::
        \begin{aligned}
            x_1'(t) ={}&
                \\left(a - \frac{1}{2-t}\right) x_1(t) \\
                &+ (2-t) a z(t)
                + \frac{3-t}{2-t} e^t.
        \\end{aligned}

    .. math::
        x_2' (t) = \frac{1 - a}{t - 2} x_1 (t) - x_2 (t) + (a - 1) z(t) + 2 e^t,

    .. math::
        0 = (t + 2) x_1 (t) + (t^2 - 4) x_2 (t) - (t^2 + t - 2) e^t,

    for :math:`t \\in [0.0, 1.0]` with differential variables :math:`x_1`,
    :math:`x_2` and algebraic variable :math:`z`. The exact solution is given by

    .. math::
        (x_1(t), x_2(t), z(t)) = (e^t, e^t, -\frac{e^t}{2 - t}),

    so initial conditions are obtained by
    :math:`(x_1(0), x_2(0), z(0)) = (x_{1,0}, x_{2,0}, z_0)`. The problem is
    taken from [1]_.

    Parameters
    ----------
    nvars_diff : tuple of integers, optional
        Number of unknown differential variables. It is set to `(2,)`.
    nvars_alg : tuple of integers, optional
        Number of unknown algebraic variables. It is set to `(1,)`.
    dtype : type, optional.
        Datatype of the values. Default is `float`.
    diff_index : int, optional
        Differentiation index of the DAE. It is given by `2`.
    t_span : tuple, optional
        Time domain where the problem lives. Default is `(0.0, 1.0)`.
    name : str, optional
        Name of the problem.

    Attributes
    ----------
    a : float
        Problem parameter. It is set to `10.0`.

    References
    ----------
    .. [1] U. M. Ascher and L. R. Petzold, Computer Methods for Ordinary
           Differential Equations and Differential-Algebraic Equations, SIAM,
           Philadelphia, PA, 1998.
    """

    def __init__(
        self,
        nvars_diff=(2,),
        nvars_alg=(1,),
        dtype=float,
        diff_index=2,
        t_span=(0.0, 1.0),
        name="Linear Ascher-Petzold DAE",
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

        self.a = 10.0

    def f_diff(self, u, t):
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
        f1 = (
            (self.a - 1.0 / (2.0 - t)) * u.diff[0]
            + (2.0 - t) * self.a * u.alg[0]
            + (3.0 - t) / (2.0 - t) * np.exp(t)
        )
        f2 = (
            (1.0 - self.a) / (t - 2.0) * u.diff[0]
            - u.diff[1]
            + (self.a - 1.0) * u.alg[0]
            + 2.0 * np.exp(t)
        )
        f = Array((self.nvars_diff, self.init[-1]))
        f[:] = (f1, f2)
        return (f1, f2)

    def f_alg(self, u, t):
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
        g_val = (
            (t + 2.0) * u.diff[0]
            + (t**2 - 4.0) * u.diff[1]
            - (t**2 + t - 2.0) * np.exp(t)
        )
        g = Array((self.nvars_alg, self.init[-1]))
        g[:] = g_val
        return g

    def u_exact(self, t):
        """
        Computes the exact solution of `u` at time `t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        SemiexplicitArray
            Exact solution of `u` at time `t`.
        """
        u = self.dtype_u(self.init)
        u.diff[:] = (np.exp(t), np.exp(t))
        u.alg[:] = -np.exp(t) / (2.0 - t)
        return u
