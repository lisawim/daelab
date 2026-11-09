from daelab.base.methods.sdc import SDCMethod
from daelab.base.problems.dae_problem import DAEProblem
from daelab.base.datatypes.semiexplicit_solution import SemiexplicitSolution
from daelab.base.datatypes.semiexplicit_rhs import SemiexplicitRHS


class ConstrainedSDC(SDCMethod):
    def __init__(
        self,
        num_nodes: int,
        quad_type: str,
        QI_type: str,
        QE_type: str = None,
        node_type: str = "LEGENDRE",
    ):
        """Initialization routine."""
        super().__init__(
            num_nodes=num_nodes,
            quad_type=quad_type,
            QI_type=QI_type,
            QE_type=QE_type,
            node_type=node_type,
        )

    def setup_node_data(self, problem: DAEProblem):
        self.u = SemiexplicitSolution(
            num_nodes=self.num_nodes,
            nvars_diff=problem.nvars_diff,
            nvars_alg=problem.nvars_alg,
            dtype=problem.dtype,
        )
        self.f = SemiexplicitRHS(
            num_nodes=self.num_nodes,
            nvars_diff=problem.nvars_diff,
            nvars_alg=problem.nvars_alg,
            dtype=problem.dtype,
        )

    def compute_defect(self):
        """Computes the defect term using `Q-Q_delta`."""
        # raise NotImplementedError("Subclasses must implement compute_defect().")

        prob = self.problem
        defect = SemiexplicitSolution(
            num_nodes=self.num_nodes,
            nvars_diff=prob.nvars_diff,
            nvars_alg=prob.nvars_alg,
            dtype=prob.dtype,
        )

        defect[1:].diff[:] = self.Q @ self.f[1:].diff[:]
        defect[1:].diff[:] -= self.QI @ self.f[1:].diff[:]
        return defect

    def update_node_values(self):
        """Updates the node values."""

        prob = self.problem

        tau = [t + dt * node for node in self.nodes]

        defect = self.compute_defect()
        defect[1:] += self.u[0].diff[:]

        rhs = SemiexplicitSolution(
            num_nodes=self.num_nodes,
            nvars_diff=prob.nvars_diff,
            nvars_alg=prob.nvars_alg,
            dtype=prob.dtype,
        )
        rhs[:] = defect[:]
        for m in range(self.num_nodes):
            for j in range(1, m + 1):
                rhs[m + 1].diff += dt * self.QI[m, j - 1] * self.f[m + 1].diff[:]

            q = dt * Q[m, m]
            if q != 0:
                self.u[m + 1] = 0.0#solve_system
            else:
                self.u[m + 1] = rhs[m + 1]

            self.f[m + 1].diff[:] = prob.f(self.u[m + 1].diff, self.u[m + 1].alg, tau[m])
            self.f[m + 1].alg[:] = prob.g(self.u[m + 1].diff, self.u[m + 1].alg, tau[m]) 