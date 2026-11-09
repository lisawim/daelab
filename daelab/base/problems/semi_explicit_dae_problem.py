from abc import ABC, abstractmethod

from daelab.base.problems.dae_problem import DAEProblem


class SemiexplicitDAEProblem(DAEProblem, ABC):
    """Base class for semi-explicit DAE Problems."""

    def __init__(
        self,
        nvars_diff: tuple[int, ...],
        nvars_alg: tuple[int, ...],
        dtype: type = float,
        diff_index: int | None = None,
        t_span: tuple[float, float] | None = None,
        name: str | None = None,
    ):
        """
        Initializes a semi-explicit DAE Problem.

        Parameters
        ----------
        nvars_diff : tuple of integers
            Number of unknown differential variables and spatial dimensions.
        nvars_alg : tuple of integers
            Number of unknown algebraic variables and spatial dimensions.
        dtype : type
            Datatype of the solution. Default is float.
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
        """Evaluate the differential equation `f(y, z, t)`."""
        raise NotImplementedError("Subclasses must implement f(y, z, t).")

    @abstractmethod
    def g(self, y, z, t):
        """Evaluate the algebraic equation `g(y, z, t)`."""
        raise NotImplementedError("Subclasses must implement g(y, z, t).")

    def solve_system(self, factor, y0, z0, t, rhs_diff, rhs_alg):
        """Solves the implicit system."""
        raise NotImplementedError("Subclasses must implement solve_system().")

    def y_exact(self, t):
        raise NotImplementedError(
            "This problem does not provide an exact solution for y."
        )

    def z_exact(self, t):
        raise NotImplementedError(
            "This problem does not provide an exact solution for z."
        )
