from daelab.base.datatypes.node_list import NodeList
from daelab.base.methods.sdc import SDCMethod


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

    def compute_defect(self):
        """Computes the defect term using `Q-Q_delta`."""

        prob = self.problem

        defect = NodeList(prob.dtype_u, prob.init, self.num_nodes - 1)

        defect[1:].diff[:] = self.Q @ self.f[1:].diff[:]
        defect[1:].diff[:] -= self.QI @ self.f[1:].diff[:]
        return defect

    def update_node_values(self, t: float, dt: float):
        """Updates the node values."""

        prob = self.problem

        tau = [t + dt * node for node in self.nodes]

        defect = self.compute_defect()
        defect[1:] += self.u[0].diff[:]

        for m in range(1, self.num_nodes + 1):
            rhs = prob.dtype_u(prob.init)
            rhs.assign(defect[:])
            for j in range(m):
                rhs.diff[:] += dt * self.QI[m, j] * self.f[m + 1].diff[:]

            q = dt * self.QI[m, m]
            if q != 0:
                self.u[m + 1] = 0.0#solve_system
            else:
                self.u[m + 1] = rhs[:]

            self.f[m + 1].diff[:] = prob.f(self.u[m + 1].diff, self.u[m + 1].alg, tau[m])
            self.f[m + 1].alg[:] = prob.g(self.u[m + 1].diff, self.u[m + 1].alg, tau[m]) 