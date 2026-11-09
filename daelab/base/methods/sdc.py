import numpy as np

from daelab.base.problems.dae_problem import DAEProblem
from daelab.base.datatypes.solution import Solution
from daelab.base.datatypes.rhs import RHS

from qmat import Q_GENERATORS, QDELTA_GENERATORS


class SDCMethod:
    """
    SDC method class for general DAE problems.

    The method is the base class to design different SDC methods.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes.
    quad_type : str
        Quadrature type.
    QI_type : str
        Type of the method. The type refers to the choice of the ``Q_delta``
        matrix leading to an implicit scheme.
    QE_type : str, optional
        Type of the method. The type refers to the choice of the ``Q_delta``
        matrix leading to an explicit scheme. Default is None.
    node_type : str
        Type of nodes. Default is ``"LEGENDRE".``
    """

    def __init__(
        self,
        num_nodes: int,
        quad_type: str,
        QI_type: str,
        QE_type: str = None,
        node_type: str = "LEGENDRE",
    ):
        if num_nodes < 2:
            raise ValueError("num_nodes must be greater than or equal to two!")

        self.num_nodes = num_nodes
        self.quad_type = quad_type
        self.QI_type = QI_type
        self.QE_type = QE_type
        self.node_type = node_type

        self.nodes, self.weights, self.Q = self._generate_coefficients()

        self.QI = self.get_QDelta_implicit(QI_type=self.QI_type)
        self.QE = (
            self.get_QDelta_explicit(QE_type=self.QE_type)
            if self.QE_type is not None
            else None
        )

        self.u = None
        self.f = None

    def setup_node_data(self, problem: DAEProblem):
        self.u = Solution(num_nodes=self.num_nodes, nvars=problem.nvars) #, dtype=problem.dtype
        self.f = RHS(num_nodes=self.num_nodes, nvars=problem.nvars) # , dtype=problem.dtype

    def _generate_coefficients(self):
        """Generate coefficients for SDC method."""

        coll = Q_GENERATORS["coll"](
            nNodes=self.num_nodes,
            nodeType=self.node_type,
            quadType=self.quad_type,
        )

        nodes, weights, Q = coll.genCoeffs()
        return nodes, weights, Q

    def get_QDelta_implicit(self, QI_type):
        QDelta_impl = QDELTA_GENERATORS[QI_type](nodes=self.nodes).getQDelta()

        tri_upper = np.triu_indices_from(QDelta_impl, k=1)
        if np.any(QDelta_impl[tri_upper] != 0):
            raise ValueError("Implicit QDelta matrix must be of lower-triangular form!")

        return QDelta_impl

    def get_QDelta_explicit(self, QE_type):
        QDelta_expl = QDELTA_GENERATORS[QE_type](nodes=self.nodes).getQDelta()

        tri_upper = np.triu_indices_from(QDelta_expl, k=1)
        if any(QDelta_expl[tri_upper] != 0):
            raise ValueError("Explicit QDelta matrix must be of lower-triangular form!")

        diag = QDelta_expl.diagonal()
        if any(diag != 0.0):
            raise ValueError("Explicit QDelta matrix must have zero diagonal entries.")

        return QDelta_expl

    def init_node_values(self):
        """Initializes the values at internal nodes."""

    def compute_defect(self):
        """Computes the defect term using `Q-Q_delta`."""
        raise NotImplementedError("Subclasses must implement compute_defect().")

    def update_node_values(self):
        """Updates the node values."""
        raise NotImplementedError("Subclasses must implement update_node_values().")

    def compute_residual(self):
        """Computes the residual."""

    def step_update(self):
        """Updates the step by computing the value at last node."""
        raise NotImplementedError("Subclasses must implement step_update().")
