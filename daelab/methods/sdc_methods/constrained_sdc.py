from daelab.base.datatypes.node_list import NodeList
from daelab.base.methods.sdc import SDCMethod


class ConstrainedSDC(SDCMethod):
    def __init__(
        self,
        num_nodes: int,
        quad_type: str,
        QD_type: str,
        node_type: str = "LEGENDRE",
    ):
        """Initialization routine."""
        super().__init__(
            num_nodes=num_nodes,
            quad_type=quad_type,
            QD_type=QD_type,
            node_type=node_type,
        )

    def compute_defect(self, dt: float):
        """Computes the defect term using `Q-Q_delta`."""

        prob = self.problem

        defect = NodeList(prob.dtype_u, prob.init, self.num_nodes - 1)
        for m in range(self.num_nodes):
            for j in range(self.num_nodes):
                defect[m].diff[:] += (
                    dt * (self.Q[m, j] - self.QD[m, j]) * self.f[j + 1].diff[:]
                )

        return defect

    def update_node_values(self, t: float, dt: float):
        """Updates the node values."""

        prob = self.problem

        tau = [t + dt * node for node in self.nodes]

        defect = self.compute_defect(dt)
        for m in range(self.num_nodes):
            defect[m].diff[:] += self.u[0].diff[:]

        for m in range(self.num_nodes):
            rhs = defect[m].copy()
            for j in range(m):
                rhs.diff[:] += dt * self.QD[m, j] * self.f[j + 1].diff[:]

            q = dt * self.QD[m, m]
            if q != 0:
                self.u[m + 1] = prob.solve_system(q, self.u[m + 1], tau[m], rhs)
            else:
                self.u[m + 1][:] = rhs[:]

            self.f[m + 1] = prob.f(self.u[m + 1], tau[m])

    def compute_residual(self, dt):
        """Computes the residual."""
        prob = self.problem

        res_norm = []
        res = NodeList(prob.dtype_u, prob.init, self.num_nodes - 1)
        for m in range(self.num_nodes):
            res[m].diff[:] += self.u[0].diff[:] - self.u[m + 1].diff[:]
            for j in range(self.num_nodes):
                res[m].diff[:] += dt * self.Q[m, j] * self.f[j + 1].diff[:]

            res_norm.append(abs(res[m].diff[:]))

        residual = max(res_norm)
        return residual
