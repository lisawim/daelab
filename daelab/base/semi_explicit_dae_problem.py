from abc import ABC, abstractmethod
from daelab.base.dae_problem import DAEProblem


class SemiexplicitDAEProblem(DAEProblem, ABC):
    """Base class for semi-explicit DAE Problems."""

    @abstractmethod
    def f(self, y, z, t):
        """Evaluate the differential equation `f(y, z, t)`."""
        raise NotImplementedError("Subclasses must implement f(y, z, t).")

    @abstractmethod
    def g(self, y, z, t):
        """Evaluate the algebraic equation `g(y, z, t)`."""
        raise NotImplementedError("Subclasses must implement g(y, z, t).")

    def y_exact(self, t):
        raise NotImplementedError(
            "This problem does not provide an exact solution for y."
        )

    def z_exact(self, t):
        raise NotImplementedError(
            "This problem does not provide an exact solution for z."
        )