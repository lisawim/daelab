import numpy as np

from daelab.base.datatypes.node_list import NodeList
from daelab.base.problems.dae_problem import DAEProblem
from daelab.base.methods.sdc import SDCMethod



class TimeStepper:
    def __init__(self, problem: DAEProblem, method: SDCMethod):
        self.problem = problem
        self.method = method

    def step(self, t: float, u: NodeList, dt: float):
        """Executes a single time step of size dt."""
