from daelab.base.datatypes.array import Array
from daelab.base.datatypes.semiexplicit import SemiexplicitArray
from daelab.base.problems.dae_problem import DAEProblem
from daelab.base.methods.sdc import SDCMethod


class TimeStepper:
    def __init__(
        self, problem: DAEProblem, method: SDCMethod, maxiter: int, restol: float
    ):
        self.__method = method
        self.method.setup(problem)
        self.maxiter, self.restol = maxiter, restol

    @property
    def method(self):
        """Getter for the method."""
        return self.__method

    @property
    def problem(self):
        """Getter for the problem."""
        return self.method.problem

    def step(self, u0: Array | SemiexplicitArray, t: float, dt: float):
        """Executes a single time step of size dt."""

        # method.u[0] = u0

        # method.init_node_values(self, u0)

        # Update of the numerical solution at nodes
        # method.update_node_values()

        # Performs a collocation update if the method is not stiffly accurate
        # method.step_update()
        raise NotImplementedError("step is not yet implemented.")
