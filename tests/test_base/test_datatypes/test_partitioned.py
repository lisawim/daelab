import numpy as np
import operator
import pytest

from daelab.base.datatypes.node_list import NodeList
from daelab.base.datatypes.partitioned import (
    PartitionedArray,
    IMEXArray,
)
from daelab.base.datatypes.semiexplicit import SemiexplicitArray, SemiexplicitIMEXArray

NVARS_DIFF = (2, 3, 4)
NVARS_ALG = (1, 3, 4)
OPERATORS = [operator.add, operator.sub, operator.mul, operator.truediv]


def diff_values():
    return np.arange(24, dtype=float).reshape(NVARS_DIFF)


def alg_values():
    return 100.0 + np.arange(12, dtype=float).reshape(NVARS_ALG)


@pytest.fixture
def zero_array():
    return SemiexplicitArray((NVARS_DIFF, NVARS_ALG, float))


@pytest.fixture
def filled_array():
    a = SemiexplicitArray((NVARS_DIFF, NVARS_ALG, float))
    a.diff.values[...] = diff_values()
    a.alg.values[...] = alg_values()
    return a


@pytest.fixture
def zero_imex_array():
    return SemiexplicitIMEXArray((NVARS_DIFF, NVARS_ALG, float))


@pytest.fixture
def filled_imex_array():
    a = SemiexplicitIMEXArray((NVARS_DIFF, NVARS_ALG, float))
    a.diff.impl.values[...] = diff_values()
    a.diff.expl.values[...] = -diff_values()
    a.alg.values[...] = alg_values()
    return a


class DummyArray(PartitionedArray):
    _components = ("comp1", "comp2")

    def __init__(self):
        self.comp1 = np.arange(6).reshape(3, 2)
        self.comp2 = np.arange(3).reshape(3, 1)
        self.num_nodes = 3


class OtherArray(PartitionedArray):
    _components = ("u", "z")


def test_parts_and_iteration_return_components_in_order():
    a = DummyArray()

    assert len(a._parts) == 2
    assert a._parts[0] is a.comp1 and a._parts[1] is a.comp2

    comp1, comp2 = a
    assert comp1 is a.comp1 and comp2 is a.comp2


def test_assign_scalar_writes_all_parts(zero_imex_array):
    zero_imex_array.assign(3.0)

    assert np.all(zero_imex_array.diff.impl.values == 3.0)
    assert np.all(zero_imex_array.diff.expl.values == 3.0)
    assert np.all(zero_imex_array.alg.values == 3.0)

    a = DummyArray()
    a.assign(7.0)

    assert np.all(a.comp1 == 7.0)
    assert np.all(a.comp2 == 7.0)


def test_assign_sequence_writes_each_part(zero_array):
    zero_array.assign((diff_values(), alg_values()))

    assert np.array_equal(zero_array.diff.values, diff_values())
    assert np.array_equal(zero_array.alg.values, alg_values())

    a = DummyArray()
    comp1, comp2 = a.comp1, a.comp2

    a.assign((np.full((3, 2), 10), np.full((3, 1), 20)))

    assert a.comp1 is comp1 and a.comp2 is comp2
    assert np.all(comp1 == 10)
    assert np.all(comp2 == 20)


def test_assign_same_type_writes_nested_parts_and_stays_independent(
    zero_imex_array, filled_imex_array
):
    zero_imex_array.assign(filled_imex_array)
    zero_imex_array.assign(-1.0)

    assert np.array_equal(filled_imex_array.diff.impl.values, diff_values())
    assert np.array_equal(filled_imex_array.diff.expl.values, -diff_values())
    assert np.array_equal(filled_imex_array.alg.values, alg_values())

    a, other = DummyArray(), DummyArray()
    other.comp1[...] = 10
    other.comp2[...] = 20

    a.assign(other)
    other.comp1[...] = -1
    other.comp2[...] = -1

    assert np.all(a.comp1 == 10)
    assert np.all(a.comp2 == 20)


def test_assign_writes_in_place_into_existing_parts(zero_array):
    diff, alg = zero_array.diff, zero_array.alg
    diff_array, alg_array = diff.values, alg.values

    zero_array.assign((diff_values(), alg_values()))

    assert zero_array.diff is diff and zero_array.alg is alg
    assert zero_array.diff.values is diff_array and zero_array.alg.values is alg_array


def test_assign_should_raise_value_error_if_components_differ():
    a = DummyArray()

    err_msg_match = "Operands must have the same parts."
    with pytest.raises(ValueError, match=err_msg_match):
        a.assign(OtherArray())


def test_assign_should_raise_type_error_if_value_is_np_array(zero_array):
    err_msg_match = "Expected a scalar, a tuple with one value per part, or "
    f"an object of type {type(zero_array).__name__}, not a numpy array."
    with pytest.raises(TypeError, match=err_msg_match):
        zero_array.assign(np.ones((2, 3, 4)))


@pytest.mark.parametrize("vals", [(1.0,), (1.0, 2.0, 3.0)])
def test_assign_should_raise_value_error_if_length_of_values_differ(vals, zero_array):
    err_msg_match = f"Expected {len(zero_array._components)} values: "
    f"{zero_array._components}, got {len(vals)} values."
    with pytest.raises(ValueError, match=err_msg_match):
        zero_array.assign(vals)


def test_copy_creates_independent_array():
    a = DummyArray()

    a_copy = a.copy()

    assert type(a_copy) is type(a), "Copied array is not an instance of DummyArray."
    assert a_copy is not a
    assert a_copy.comp1 is not a.comp1
    assert a_copy.comp1 is not a.comp2
    assert a_copy.num_nodes == a.num_nodes
    for name in a._components:
        original, copied = getattr(a, name), getattr(a_copy, name)
        assert copied is not original
        assert not np.shares_memory(copied, original)
        assert np.array_equal(copied, original)


def test_copy_returns_independent_object_with_equal_values(filled_array):
    original_diff = filled_array.diff.values.copy()
    original_alg = filled_array.alg.values.copy()
    arr = filled_array.copy()

    assert type(arr) is type(filled_array), (
        "Copied array is not an instance of SemiexplicitArray."
    )
    assert arr is not filled_array
    assert arr.diff.shape == filled_array.diff.shape == NVARS_DIFF, (
        "Copied array does not have the correct shape in diff part."
    )
    assert arr.alg.shape == filled_array.alg.shape == NVARS_ALG, (
        "Copied array does not have the correct shape in alg part."
    )
    assert arr.diff.dtype == filled_array.diff.dtype == np.dtype(float), (
        " Copied array does not have the correct dtype."
    )
    assert np.array_equal(arr.diff.values, original_diff)
    assert np.array_equal(arr.alg.values, original_alg)
    assert not np.shares_memory(arr.diff.values, filled_array.diff.values), (
        "Diff part: Copied array shares memory with the original."
    )
    assert not np.shares_memory(arr.alg.values, filled_array.alg.values), (
        "Alg part: Copied array shares memory with the original."
    )

    arr.diff[...] = -1.0
    arr.alg[...] = 6.0
    assert np.array_equal(filled_array.diff.values, original_diff), (
        "Diff part: Original arrays have been erroneously overwritten."
    )
    assert np.array_equal(filled_array.alg.values, original_alg), (
        "Alg part: Original arrays have been erroneously overwritten."
    )


@pytest.mark.parametrize(
    "nvars", [((1,), (2,)), ((2, 4), (1, 4)), ((2, 3, 4), (1, 3, 4))]
)
def test_flat_has_length_prod_of_nvars(nvars):
    init = (nvars[0], nvars[1], float)
    a = SemiexplicitArray(init)

    flat = a.flat()
    flat_shape = (np.prod(nvars[0]) + np.prod(nvars[1]),)
    assert flat.shape == flat_shape, "Flattened array does not have correct shape."


def test_flat_is_one_dimensional_numpy_array():
    a = SemiexplicitArray(((2, 3, 4), (1, 3, 4), float))
    flat = a.flat()

    assert isinstance(flat, np.ndarray)
    assert flat.ndim == 1


def test_flat_has_values_in_c_order():
    a = SemiexplicitArray((NVARS_DIFF, NVARS_ALG, float))
    a.diff.values[...] = np.arange(np.prod(NVARS_DIFF)).reshape(NVARS_DIFF)
    a.alg.values[...] = np.arange(
        np.prod(NVARS_DIFF), np.prod(NVARS_ALG) + np.prod(NVARS_DIFF)
    ).reshape(NVARS_ALG)

    assert np.array_equal(a.flat(), np.arange(36))
    assert a.flat().flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("op", OPERATORS)
def test_binary_should_raise_value_error_if_other_object_has_different_parts(op):
    a = DummyArray()

    with pytest.raises(ValueError, match="Operands must have the same parts."):
        a._binary(OtherArray(), op)


@pytest.mark.parametrize("op", OPERATORS)
@pytest.mark.parametrize(
    "other", [np.ones((3, 2)), NodeList(IMEXArray, (NVARS_ALG, float), 3)]
)
def test_binary_should_raise_type_error_if_other_object_is_np_array_or_nodelist(
    other, op
):
    a = DummyArray()

    err_msg_match = f"Cannot combine {type(a).__name__} with {type(other).__name__}."
    with pytest.raises(TypeError, match=err_msg_match):
        a._binary(other, op)


@pytest.mark.parametrize("op", OPERATORS)
def test_binary_returns_new_object_and_leaves_operands_unchanged(filled_array, op):
    other = SemiexplicitArray((NVARS_DIFF, NVARS_ALG, float))
    other.diff.values[...] = filled_array.diff + 1.0
    other.alg.values[...] = filled_array.alg + 42.0

    original_diff = filled_array.diff.values.copy()
    other_original_diff = other.diff.values.copy()
    original_alg = filled_array.alg.values.copy()
    other_original_alg = other.alg.values.copy()

    result = filled_array._binary(other, op)

    assert result is not filled_array and result is not other
    assert not np.shares_memory(result.diff.values, filled_array.diff.values)
    assert not np.shares_memory(result.diff.values, other.diff.values)
    assert not np.shares_memory(result.alg.values, filled_array.alg.values)
    assert not np.shares_memory(result.alg.values, other.alg.values)
    assert np.array_equal(filled_array.diff.values, original_diff)
    assert np.array_equal(other.diff.values, other_original_diff)
    assert np.array_equal(filled_array.alg.values, original_alg)
    assert np.array_equal(other.alg.values, other_original_alg)


@pytest.mark.parametrize("op", OPERATORS)
def test_binary_with_scalar_applies_operation_to_values(filled_array, op):
    result = filled_array._binary(2.0, op)

    assert np.array_equal(result.diff.values, op(filled_array.diff.values, 2.0)), (
        "op is not applied to all values of diff part."
    )
    assert np.array_equal(result.alg.values, op(filled_array.alg.values, 2.0)), (
        "op is not applied to all values of diff part."
    )
