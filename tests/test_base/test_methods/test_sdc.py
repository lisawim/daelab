import numpy as np
import pytest

from daelab.base.methods.sdc import SDCMethod


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


def test_QE_should_be_None_if_QE_type_is_not_set():
    sdc = SDCMethod(num_nodes=2, quad_type="RADAU-RIGHT", QI_type="IE")
    assert sdc.QE is None


def test_should_raise_value_error_if_num_nodes_is_less_than_two():
    err_msg_match = "num_nodes must be greater than or equal to two!"
    with pytest.raises(ValueError, match=err_msg_match):
        SDCMethod(num_nodes=1, quad_type="RADAU-RIGHT", QI_type="IE")


def test_generate_coefficients_should_return_arraysof_size_num_nodes():
    num_nodes = 2
    sdc = SDCMethod(num_nodes=num_nodes, quad_type="RADAU-RIGHT", QI_type="IE")
    nodes, weights, Q = sdc._generate_coefficients()

    assert isinstance(nodes, np.ndarray)
    assert isinstance(weights, np.ndarray)
    assert isinstance(Q, np.ndarray)

    assert nodes.shape == (num_nodes,)
    assert weights.shape == (num_nodes,)
    assert Q.shape == (num_nodes, num_nodes)


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
