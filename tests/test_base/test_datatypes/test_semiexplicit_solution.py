import numpy as np
import pytest

from daelab.base.datatypes.solution import Solution
from daelab.base.datatypes.semiexplicit_solution import (
    PartitionedContainer,
    SemiexplicitSolution,
)


class DummyContainer(PartitionedContainer):
    _components = ("comp1", "comp2")

    def __init__(self):
        self.comp1 = np.arange(6).reshape(3, 2)
        self.comp2 = np.arange(3).reshape(3, 1)
        self.num_nodes = 3


class OtherContainer(PartitionedContainer):
    _components = ("u", "z")


def test_parts_return_all_components():
    con = DummyContainer()

    assert con._parts == [con.comp1, con.comp2]


def test_getitem_returns_parts_at_nodes():
    con = DummyContainer()

    m = 1
    assert np.array_equal(con[m][0], con.comp1[m])
    assert np.array_equal(con[m][1], con.comp2[m])


def test_setitem_sets_all_parts():
    con = DummyContainer()

    con[1] = (np.array([10, 11]), np.array([20]))

    assert np.array_equal(con.comp1[1], [10, 11])
    assert np.array_equal(con.comp2[1], [20])


def test_setitem_should_raise_value_error_for_wrong_number_of_values():
    con = DummyContainer()

    with pytest.raises(ValueError, match="Expected 2 values"):
        con[1] = (np.array([10, 11]),)


def test_copy_creates_independent_container():
    con = DummyContainer()

    con_copy = con.copy()

    assert con_copy is not con
    assert con_copy.comp1 is not con.comp1
    assert con_copy.comp1 is not con.comp2

    assert np.array_equal(con_copy.comp1, con.comp1)
    assert np.array_equal(con_copy.comp2, con.comp2)


def test_binary_adds_two_containers():
    con1 = DummyContainer()
    con2 = DummyContainer()

    sum = con1._binary(con2, np.add)

    assert np.array_equal(sum.comp1, con1.comp1 + con2.comp1)
    assert np.array_equal(sum.comp2, con1.comp2 + con2.comp2)


def test_binary_adds_container_and_scalar():
    con = DummyContainer()

    sum = con._binary(6, np.add)

    assert np.array_equal(sum.comp1, con.comp1 + 6)
    assert np.array_equal(sum.comp2, con.comp2 + 6)


def test_binary_should_raise_value_error_if_components_are_different():
    con1 = DummyContainer()
    other_con = OtherContainer()

    other_con.u = np.zeros((3, 1))
    other_con.z = np.zeros((2, 2))

    with pytest.raises(
        ValueError, match="Operands must have the same parts and num_nodes."
    ):
        con1._binary(other_con, np.add)


def test_binary_should_raise_value_error_if_num_nodes_are_different():
    con1 = DummyContainer()
    other_con = OtherContainer()

    other_con.num_nodes = 4

    with pytest.raises(
        ValueError, match="Operands must have the same parts and num_nodes."
    ):
        con1._binary(other_con, np.add)


def test_binary_should_raise_type_error_if_datatypes_are_different():
    con = DummyContainer()
    u = Solution(num_nodes=3, nvars=(1,))

    with pytest.raises(
        TypeError, match="Cannot combine DummyContainer with a Solution."
    ):
        con._binary(u, np.add)


# Wieso ist das Ergebnis nicht gleich Q @ u?
def test_apply_matrix_is_applied_to_all_components():
    u = SemiexplicitSolution(num_nodes=2, nvars_diff=(2, 2), nvars_alg=(2, 2))

    Q = np.array([[1, 0], [2, 0]])

    for m in range(u.num_nodes):
        u.diff[m] = (m + 3) * np.ones(u.nvars_diff)
        u.alg[m] = (m + 5) * np.ones(u.nvars_alg)

    res = u.apply_matrix(Q)

    for m in range(u.num_nodes):
        expected_diff = sum(Q[m, k] * u.diff[k] for k in range(u.num_nodes))
        expected_alg = sum(Q[m, k] * u.alg[k] for k in range(u.num_nodes))

        assert np.array_equal(res.diff[m], expected_diff)
        assert np.array_equal(res.alg[m], expected_alg)


def test_function_repr_returns_correct_output_SemiexplicitSolution():
    num_nodes = 3
    nvars = (128, 128, 256)
    dtype = float
    u = SemiexplicitSolution(
        num_nodes=num_nodes, nvars_diff=nvars, nvars_alg=nvars, dtype=dtype
    )

    assert (
        str(u) == f"SemiexplicitSolution({num_nodes=}, "
        f"nvars_diff={nvars}, nvars_alg={nvars}, dtype={np.dtype(dtype)})"
    )
