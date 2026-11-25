import numpy as np


from daelab.base.datatypes.node_list import NodeList
from daelab.base.methods.method import Method
from daelab.base.problems.dae_problem import DAEProblem

from qmat import Q_GENERATORS, QDELTA_GENERATORS


class SDCMethod(Method):
    """
    SDC method class for general DAE problems.

    The method is the base class to design different SDC methods. For passed
    parameters ``num_nodes``, ``quad_type``, and ``node_type`` it generates
    the weights, nodes, and coefficient of spectral integration matrix ``Q``
    (representing the coefficients of a collocation method). The ``QD_type``
    indicates the choice of the ``QDelta`` matrix and refers to a certain
    SDC method.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes.
    quad_type : str
        Quadrature type.
    QD_type : str
        Type of the method. The type refers to the choice of the ``QDelta``
        matrix leading to an SDC scheme.
    node_type : str
        Type of nodes. Default is ``"LEGENDRE".``

    Attributes
    ----------
    num_nodes : int
        Number of collocation nodes.
    quad_type : str
        Quadrature type.
    QD_type : str
        Type of the SDC method.
    node_type : str
        Type of nodes.
    """

    def __init__(
        self,
        num_nodes: int,
        quad_type: str,
        QD_type: str,
        node_type: str = "LEGENDRE",
    ):
        if num_nodes < 2:
            raise ValueError("num_nodes must be greater than or equal to two!")

        self.num_nodes = num_nodes
        self.quad_type = quad_type
        self.QD_type = QD_type
        self.node_type = node_type

        self.nodes, self.weights, self.Q = self._generate_coefficients()

        self.QD = self.get_QDelta(QD_type=self.QD_type)

        self.u = None
        self.f = None

    @property
    def problem(self):
        """Getter for the problem."""
        return self.__problem

    def setup(self, problem: DAEProblem) -> None:
        """Sets the problem as attribute to the class."""
        self.__problem = problem
        self.setup_node_data()

    def setup_node_data(self):
        """Instantiates the datatypes for the method."""
        prob = self.problem

        self.u = NodeList(prob.dtype_u, prob.init, self.num_nodes)
        self.f = NodeList(prob.dtype_f, prob.init, self.num_nodes)

    def _generate_coefficients(self):
        """Generate coefficients for SDC method."""

        coll = Q_GENERATORS["coll"](
            nNodes=self.num_nodes,
            nodeType=self.node_type,
            quadType=self.quad_type,
        )

        nodes, weights, Q = coll.genCoeffs()
        return nodes, weights, Q

    def get_QDelta(self, QD_type):
        QDelta = QDELTA_GENERATORS[QD_type](nodes=self.nodes).getQDelta()

        tri_upper = np.triu_indices_from(QDelta, k=1)
        if any(QDelta[tri_upper] != 0):
            raise ValueError("QDelta matrix must be of lower-triangular form!")

        return QDelta

    def init_node_values(self):
        """Initializes the values at internal nodes."""
        raise NotImplementedError("Subclasses must implement init_node_values().")

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
