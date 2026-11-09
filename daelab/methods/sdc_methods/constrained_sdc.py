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
            nvars_diff=problem.nvars,
            nvars_alg=problem.nvars_alg,
            dtype=problem.dtype,
        )
        self.f = SemiexplicitRHS(
            num_nodes=self.num_nodes,
            nvars_diff=problem.nvars_diff,
            nvars_alg=problem.nvars_alg,
            dtype=problem.dtype,
        )