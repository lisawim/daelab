from qmat import Q_GENERATORS, QDELTA_GENERATORS


class SDCMethod:
    def __init__(
        self,
        method_type: str,
        num_nodes: int,
        num_sweeps: int,
        quad_type: str,
        node_type: str="LEGENDRE",
    ):
        if num_nodes < 2:
            raise ValueError("num_nodes must be greater than or equal to two!")

        if num_sweeps < 0:
            raise ValueError("num_sweeps must be greater than or equal to zero!")

        self.num_nodes = num_nodes
        self.num_sweeps = num_sweeps
        self.quad_type = quad_type
        self.node_type = node_type
        self.method_type = method_type

        self._generate_coefficients()

    def _generate_coefficients(self):
        """Generate coefficients for SDC method."""

        coll = Q_GENERATORS["coll"](
            nNodes=self.num_nodes,
            nodeType=self.node_type,
            quadType=self.quad_type,
        )
    
        self.nodes, self.weights, self.Q = coll.genCoeffs()
    
        self.Q_delta = QDELTA_GENERATORS[self.method_type](
            nodes=self.nodes
        ).getQDelta()

    def init_node_values(self):
        """Initializes the values at internal nodes."""

    def compute_defect(self):
        f"""Computes the defect term using `Q-Q_delta`."""

    def update_node_values(self):
        """Updates the node values."""

    def step_update(self):
        """Updates the step by computing the value at last node."""
