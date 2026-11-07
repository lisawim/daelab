from daelab.base.datatypes.rhs import RHS
from daelab.base.datatypes.solution import Solution


def test_RHS_class_is_of_type_Solution():
    rhs = RHS(num_nodes=3, nvars=(128, 128))
    assert isinstance(rhs, Solution)
