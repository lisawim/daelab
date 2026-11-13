import numpy as np
import pytest

from daelab.base.datatypes.node_list import NodeList
from tests.test_base.test_problems.test_full_implicit_dae_problem import (
    DummyImplicitDAE,
)
from daelab.base.methods.sdc import SDCMethod

from qmat import QDELTA_GENERATORS


class DummyQDeltaGenerator:
    def __init__(self, nodes):
        self.nodes = nodes

    def getQDelta(self):
        return np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]])


def test_attributes_are_initialized_correctly():
    num_nodes = 3
    quad_type = "RADAU-RIGHT"
    QI_type = "IE"
    QE_type = "EE"
    node_type = "CHEBY-1"
    sdc = SDCMethod(
        num_nodes=num_nodes,
        quad_type=quad_type,
        QI_type=QI_type,
        QE_type=QE_type,
        node_type=node_type,
    )

    assert sdc.num_nodes == num_nodes
    assert sdc.quad_type == quad_type
    assert sdc.QI_type == QI_type
    assert sdc.QE_type == QE_type
    assert sdc.node_type == node_type
    assert sdc.u is None
    assert sdc.f is None


def test_setup_creates_problem_attribute_solution_and_rhs_with_correct_shapes():
    nvars = (3, 64)
    problem = DummyImplicitDAE(nvars=nvars)

    num_nodes = 3
    sdc = SDCMethod(num_nodes=num_nodes, quad_type="RADAU-RIGHT", QI_type="IE")

    sdc.setup(problem=problem)

    assert isinstance(sdc.u, NodeList)
    assert isinstance(sdc.f, NodeList)
    assert isinstance(sdc.problem, DummyImplicitDAE)
    assert len(sdc.u) == len(sdc.f) == num_nodes + 1
    for u_node in sdc.u:
        assert u_node.shape == nvars
    for f_node in sdc.f:
        assert f_node.shape == nvars


def test_QE_should_be_None_if_QE_type_is_not_set():
    sdc = SDCMethod(num_nodes=2, quad_type="RADAU-RIGHT", QI_type="IE")
    assert sdc.QE is None


def test_should_raise_value_error_if_num_nodes_is_less_than_two():
    err_msg_match = "num_nodes must be greater than or equal to two!"
    with pytest.raises(ValueError, match=err_msg_match):
        SDCMethod(num_nodes=1, quad_type="RADAU-RIGHT", QI_type="IE")


def test_generate_coefficients_should_return_arrays_of_size_num_nodes():
    num_nodes = 2
    sdc = SDCMethod(num_nodes=num_nodes, quad_type="RADAU-RIGHT", QI_type="IE")
    nodes, weights, Q = sdc._generate_coefficients()

    assert isinstance(nodes, np.ndarray)
    assert isinstance(weights, np.ndarray)
    assert isinstance(Q, np.ndarray)

    assert nodes.shape == (num_nodes,)
    assert weights.shape == (num_nodes,)
    assert Q.shape == (num_nodes, num_nodes)


def test_get_QDelta_implicit_should_raise_value_error_if_matrix_is_upper_triangular(
    monkeypatch,
):
    monkeypatch.setitem(QDELTA_GENERATORS, "TRIU", DummyQDeltaGenerator)

    err_msg_match = "Implicit QDelta matrix must be of lower-triangular form!"
    with pytest.raises(ValueError, match=err_msg_match):
        SDCMethod(num_nodes=3, quad_type="RADAU-RIGHT", QI_type="TRIU")


def test_get_QDelta_explicit_should_raise_value_error_if_matrix_is_upper_triangular(
    monkeypatch,
):
    monkeypatch.setitem(QDELTA_GENERATORS, "TRIU", DummyQDeltaGenerator)

    err_msg_match = "Explicit QDelta matrix must be of lower-triangular form!"
    with pytest.raises(ValueError, match=err_msg_match):
        SDCMethod(num_nodes=3, quad_type="RADAU-RIGHT", QI_type="IE", QE_type="TRIU")


def test_get_QDelta_explicit_should_raise_value_error_if_diagonal_entries_are_nonzero():
    err_msg_match = "Explicit QDelta matrix must have zero diagonal entries."
    with pytest.raises(ValueError, match=err_msg_match):
        SDCMethod(num_nodes=3, quad_type="RADAU-RIGHT", QI_type="IE", QE_type="IE")


def test_get_QDelta_implicit_returns_lower_triangular_matrix():
    num_nodes = 3
    quad_type = "RADAU-RIGHT"
    QI_type = "IE"
    sdc = SDCMethod(num_nodes=num_nodes, quad_type=quad_type, QI_type=QI_type)

    QI = sdc.get_QDelta_implicit(QI_type=QI_type)
    assert isinstance(QI, np.ndarray)
    assert QI.shape == (num_nodes, num_nodes)


def test_get_QDelta_explicit_returns_lower_triangular_matrix():
    num_nodes = 3
    quad_type = "RADAU-RIGHT"
    QI_type = "IE"
    QE_type = "EE"
    sdc = SDCMethod(
        num_nodes=num_nodes, quad_type=quad_type, QI_type=QI_type, QE_type=QE_type
    )

    QE = sdc.get_QDelta_explicit(QE_type=QE_type)
    assert isinstance(QE, np.ndarray)
    assert QE.shape == (num_nodes, num_nodes)


def test_compute_defect_should_raise_not_implemented_error():
    sdc = SDCMethod(num_nodes=3, quad_type="RADAU-RIGHT", QI_type="IE")
    err_msg_match = r"Subclasses must implement compute_defect\(\)."
    with pytest.raises(NotImplementedError, match=err_msg_match):
        sdc.compute_defect()


def test_update_node_values_should_raise_not_implemented_error():
    sdc = SDCMethod(num_nodes=3, quad_type="RADAU-RIGHT", QI_type="IE")
    err_msg_match = r"Subclasses must implement update_node_values\(\)."
    with pytest.raises(NotImplementedError, match=err_msg_match):
        sdc.update_node_values()


def test_step_update_should_raise_not_implemented_error():
    sdc = SDCMethod(num_nodes=3, quad_type="RADAU-RIGHT", QI_type="IE")
    err_msg_match = r"Subclasses must implement step_update\(\)."
    with pytest.raises(NotImplementedError, match=err_msg_match):
        sdc.step_update()
