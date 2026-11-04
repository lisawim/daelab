from abc import ABC, abstractmethod
from daelab.base.dae_problem import DAEProblem


class FullimplicitDAEProblem(DAEProblem, ABC):
    """Base class for fully-implicit DAE Problems."""
    def __init__(
        self,
        nvars: tuple[tuple[int, ...], ...],
        diff_index: int | None = None,
        t_span: tuple[float, float] | None = None,
        name: str | None = None,
    ):
        """
        Initializes a full-implicit DAE Problem.

        Parameters
        ----------
        nvars : tuple of tuples of integers
            Number of variables and spatial dimensions. For example,
            for nvars = ((n_x, n_y, n_z), (n_x, n_y), (n_x,)), we have len(nvars) = 3
            unknowns. The first unknown has three different spatial directions with
            n_x DoF, n_y DoF and n_z DoF, the second one has n_x DoF and n_y DoF, and
            the third one has n_z DoF.
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
            diff_index=diff_index,
            t_span=t_span,
            name=name,
        )

        self.is_semiexplicit_dae = False

        if isinstance(nvars, int):
            nvars = ((nvars,),)

        self.validate_nvars(nvars)
        self.nvars = nvars

    @abstractmethod
    def F(self, u_dot, u, t):
        """Evaluate the DAE function `F(u_dot, u, t) = 0`"""
        raise NotImplementedError("Subclasses must implement F(u_dot, u, t).")

    def u_exact(self, t):
        """Return the exact solution u(t), if available."""
        raise NotImplementedError(
            "This problem does not provide an exact solution."
        )