import numpy as np

from daelab.base.semi_explicit_dae_problem import SemiexplicitDAEProblem


class LinearManufacturedDAE(SemiexplicitDAEProblem):
    def __init__(
        self,
        nvars_diff=((1,),),
        nvars_alg=((1,),),
        diff_index=1,
        t_span=(0.0, 1.0),
        name="Linear manufactured DAE",
    ):
        super().__init__(
            nvars_diff=nvars_diff,
            nvars_alg=nvars_alg,
            diff_index=diff_index,
            t_span=t_span,
            name=name,
        )

    def f(self, y, z, t):
        """Evaluate the differential equation `f(y, z, t)`."""
        return -2 * y + z
    
    def g(self, y, z, t):
        """Evaluate the algebraic equation `g(y, z, t)`."""
        return -2 * y - z

    def y_exact(self, t):
        """Returns the exact solution of y."""
        return np.exp(-4 * t)

    def z_exact(self, t):
        """Returns the exact solution of z."""
        return -2 * np.exp(-4 * t)
