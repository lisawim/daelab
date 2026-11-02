from abc import ABC, abstractmethod
from daelab.base.dae_problem import DAEProblem


class FullimplicitDAEProblem(DAEProblem, ABC):
    """Base class for fully-implicit DAE Problems."""

    @abstractmethod
    def F(self, u_dot, u, t):
        """Evaluate the DAE function `F(u_dot, u, t) = 0`"""
        raise NotImplementedError("Subclasses must implement F(u_dot, u, t).")

    def u_exact(self, t):
        """Return the exact solution u(t), if available."""
        raise NotImplementedError(
            "This problem does not provide an exact solution."
        )