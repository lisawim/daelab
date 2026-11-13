import numpy as np
import operator
import pytest

from daelab.base.datatypes.array import Array
from daelab.base.datatypes.node_list import _shapes, NodeList
from daelab.base.datatypes.semiexplicit import SemiexplicitArray

NVARS_DIFF = (2, 3, 4)
NVARS_ALG = (1, 3, 4)
INIT = (NVARS_DIFF, NVARS_ALG, float)
NUM_NODES = 3


def diff_values():
    return 42.0 + np.arange(24, dtype=float).reshape(NVARS_DIFF)


def alg_values():
    return 6.0 + np.arange(12, dtype=float).reshape(NVARS_ALG)


@pytest.fixture
def zero_node_list():
    return NodeList(SemiexplicitArray, INIT, NUM_NODES)


@pytest.fixture
def filled_node_list():
    u = NodeList(SemiexplicitArray, INIT, NUM_NODES)
    for m in range(len(u)):
        u[m].diff.values[...] = (m + 1) + diff_values()
        u[m].alg.values[...] = 2 * m + alg_values()
    return u


@pytest.fixture
def array_node_list():
    u = NodeList(Array, (NVARS_DIFF, float), NUM_NODES)
    for m in range(len(u)):
        u[m].values[...] = 100.0 * m + np.arange(24, dtype=float).reshape(NVARS_DIFF)
    return u


@pytest.mark.parametrize("init", [((2, 3, 4), float), ((2, 4, 4), (1, 4, 4), complex)])
def test_shapes_returns_tuple_only_with_shapes(init):
    assert _shapes(init) == init[:-1], (
        "Shapes does not match with shapes contained in init."
    )


def test_init_creates_independent_zero_initialized_nodes_of_given_type(zero_node_list):
    nodes = list(zero_node_list)

    assert len(zero_node_list) == len(nodes) == NUM_NODES + 1
    assert all(type(node) is SemiexplicitArray for node in nodes)
    assert len({id(node) for node in nodes}) == len(nodes), (
        "Nodes are not distinct objects."
    )

    for node in nodes:
        assert node.diff.shape == NVARS_DIFF
        assert node.alg.shape == NVARS_ALG
        assert not np.any(node.diff.values) and not np.any(node.alg.values)

    assert not np.shares_memory(nodes[0].diff.values, nodes[1].diff.values)
    assert zero_node_list.dtype_node is SemiexplicitArray
    assert zero_node_list.init == INIT


def test_getitem_returns_stored_node_that_can_be_modified_in_place(filled_node_list):
    node = filled_node_list[1]
    node.diff[:] = 34.0

    assert filled_node_list[1] is node
    assert np.all(filled_node_list[1].diff.values == 34.0), (
        "Diff values at first node have not been set to 34.0."
    )
    assert np.array_equal(filled_node_list[1].alg.values, 2 + alg_values())
    assert np.array_equal(filled_node_list[0].diff.values, 1 + diff_values())


def test_getitem_returns_array_with_nvars(zero_node_list):
    assert zero_node_list[1].diff.shape == (2, 3, 4), (
        "Shape of diff array at second node is incorrect."
    )
    assert zero_node_list[1].alg.shape == (1, 3, 4), (
        "Shape of alg array at second node is incorrect."
    )


@pytest.mark.parametrize("index", [4, -5])
def test_getitem_should_raise_index_error_if_index_is_out_of_range(
    filled_node_list, index
):
    with pytest.raises(IndexError):
        filled_node_list[index]


def test_setitem_writes_values_into_existing_node_only(zero_node_list):
    node = zero_node_list[2]
    diff_val, alg_val = diff_values(), alg_values()

    zero_node_list[2] = (diff_val, alg_val)
    diff_val[:] = -1.0

    assert zero_node_list[2] is node
    assert np.array_equal(node.diff.values, diff_values())
    assert np.array_equal(node.alg.values, alg_values())
    for m in (0, 1, 3):
        assert not np.any(zero_node_list[m].diff.values)
        assert not np.any(zero_node_list[m].alg.values)


def test_setitem_with_slice_writes_into_selected_nodes(zero_node_list):
    zero_node_list[1:3] = 5.0

    for m, expected in enumerate([0.0, 5.0, 5.0, 0.0]):
        assert np.all(zero_node_list[m].diff.values == expected)
        assert np.all(zero_node_list[m].alg.values == expected)

    zero_node_list[:] = 2.0

    for m in range(len(zero_node_list)):
        assert np.all(zero_node_list[m].diff.values == 2.0)
        assert np.all(zero_node_list[m].alg.values == 2.0)


@pytest.mark.parametrize("index", [4, -5])
def test_setitem_should_raise_index_error_if_index_is_out_of_range(
    zero_node_list, index
):
    with pytest.raises(IndexError):
        zero_node_list[index] = 1.0


def test_flat_node_returns_flattened_view_of_node(array_node_list):
    for m in range(len(array_node_list)):
        print("Type:", type(array_node_list[m]))
        flat = array_node_list.flat_node(m)

        assert flat.shape == (24,)
        assert np.array_equal(flat, array_node_list[m].values.reshape(-1))
        assert np.shares_memory(flat, array_node_list[m].values)


def test_as_flat_returns_matrix_with_one_flattened_row_per_node_as_copy(
    array_node_list,
):
    flat = array_node_list.as_flat()

    assert flat.shape == (NUM_NODES + 1, 24)
    for m in range(len(array_node_list)):
        assert np.array_equal(flat[m], array_node_list.flat_node(m))
        assert not np.shares_memory(flat, array_node_list[m].values)

    flat[...] = 0.0
    assert np.any(array_node_list[1].values), (
        "Writing into the matrix changed the nodes."
    )


def test_copy_creates_independent_node_list(filled_node_list):
    copied = filled_node_list.copy()

    assert type(copied) is NodeList
    assert copied is not filled_node_list
    assert len(copied) == len(filled_node_list)
    assert copied.dtype_node is filled_node_list.dtype_node
    assert copied.init == filled_node_list.init

    for m in range(len(copied)):
        assert copied[m] is not filled_node_list[m]
        assert np.array_equal(copied[m].diff.values, filled_node_list[m].diff.values)
        assert np.array_equal(copied[m].alg.values, filled_node_list[m].alg.values)
        assert not np.shares_memory(
            copied[m].diff.values, filled_node_list[m].diff.values
        )
        assert not np.shares_memory(
            copied[m].alg.values, filled_node_list[m].alg.values
        )

    copied[0].diff[:] = -1.0
    assert np.array_equal(filled_node_list[0].diff.values, 1 + diff_values())


def test_binary_with_scalar_applies_operation_to_each_node(filled_node_list):
    result = filled_node_list._binary(1.0, operator.sub)

    assert type(result) is NodeList
    assert result is not filled_node_list
    for m in range(len(result)):
        assert np.array_equal(result[m].diff.values, (m + 1) + diff_values() - 1.0)
        assert np.array_equal(result[m].alg.values, 2 * m + alg_values() - 1.0)
        assert np.array_equal(filled_node_list[m].diff.values, (m + 1) + diff_values())
        assert np.array_equal(filled_node_list[m].alg.values, 2 * m + alg_values())


def test_binary_with_node_list_applies_operation_nodewise_to_new_object(
    filled_node_list,
):
    other = NodeList(SemiexplicitArray, INIT, NUM_NODES)
    other[:] = 2.0

    result = filled_node_list._binary(other, operator.sub)

    assert type(result) is NodeList
    assert result is not filled_node_list and result is not other
    for m in range(len(result)):
        assert np.array_equal(result[m].diff.values, (m + 1) + diff_values() - 2.0)
        assert np.array_equal(result[m].alg.values, 2 * m + alg_values() - 2.0)
        assert not np.shares_memory(
            result[m].diff.values, filled_node_list[m].diff.values
        )
        assert not np.shares_memory(
            result[m].alg.values, filled_node_list[m].alg.values
        )
        assert np.all(other[m].diff.values == 2.0)
        assert np.array_equal(filled_node_list[m].diff.values, (m + 1) + diff_values())
        assert np.array_equal(filled_node_list[m].alg.values, 2 * m + alg_values())


def test_binary_allows_node_lists_with_different_dtype(filled_node_list):
    other = NodeList(SemiexplicitArray, (NVARS_DIFF, NVARS_ALG, np.float32), NUM_NODES)
    other[:] = 1.0

    result = filled_node_list._binary(other, operator.add)

    for m in range(len(result)):
        assert np.array_equal(result[m].diff.values, (m + 1) + diff_values() + 1.0)
        assert np.array_equal(result[m].alg.values, 2 * m + alg_values() + 1.0)


@pytest.mark.parametrize(
    "init, num_nodes, match",
    [
        (INIT, NUM_NODES + 1, "same length!"),
        (((2, 3, 5), (1, 3, 5), float), NUM_NODES, "same nvars."),
    ],
)
def test_binary_should_raise_value_error_if_operands_do_not_match(
    filled_node_list, init, num_nodes, match
):
    other = NodeList(SemiexplicitArray, init, num_nodes)
    err_msg_match = "Operands must have the " + match
    with pytest.raises(ValueError, match=err_msg_match):
        filled_node_list._binary(other, operator.add)
