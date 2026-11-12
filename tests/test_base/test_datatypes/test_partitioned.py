import numpy as np
import operator
import pytest

from daelab.base.datatypes.array import _assign
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


class Recorder:
    """Target that is not a numpy array and records calls of assign."""

    def __init__(self):
        self.calls = []

    def assign(self, value):
        self.calls.append(value)


def test_assign_to_ndarray_writes_values_in_place():
    target = np.zeros((2, 3, 4))
    view = target[1]

    val = np.arange(24, dtype=float).reshape(2, 3, 4)

    _assign(target, val)

    assert np.array_equal(target, val)
    assert np.array_equal(val[1], view)


def test_assign_to_other_target_calls_its_assign_method():
    target = Recorder()
    val = object()

    _assign(target, val)

    assert target.calls == [val]


def test_parts_return_all_components():
    a = DummyArray()

    assert a._parts == [a.comp1, a.comp2]


def test_assign_should_raise_value_error_if_components_differ():
    a = DummyArray()
    value = OtherArray()

    err_msg_match = "Operands must have the same parts."
    with pytest.raises(ValueError, match=err_msg_match):
        a.assign(value)


def test_copy_creates_independent_array():
    a = DummyArray()

    a_copy = a.copy()

    assert type(a_copy) is type(a), "Copied array is not an instance of DummyArray."
    assert a_copy is not a
    assert a_copy.comp1 is not a.comp1
    assert a_copy.comp1 is not a.comp2

    assert not np.shares_memory(a_copy, a), (
        "Copied array shares memory with the original"
    )

    assert np.array_equal(a_copy.comp1, a.comp1)
    assert np.array_equal(a_copy.comp2, a.comp2)


def test_assign_scalar_writes_all_parts(zero_imex_array):
    zero_imex_array.assign(3.0)

    assert np.all(zero_imex_array.diff.impl.values == 3.0)
    assert np.all(zero_imex_array.diff.expl.values == 3.0)
    assert np.all(zero_imex_array.alg.values == 3.0)


def test_assign_sequence_writes_each_part(zero_array):
    zero_array.assign((diff_values(), alg_values()))

    assert np.array_equal(zero_array.diff.values, diff_values())
    assert np.array_equal(zero_array.alg.values, alg_values())


def test_assign_same_type_writes_nested_parts_and_stays_independent(
    zero_imex_array, filled_imex_array
):
    zero_imex_array.assign(filled_imex_array)
    zero_imex_array.assign(-1.0)

    assert np.array_equal(filled_imex_array.diff.impl.values, diff_values())
    assert np.array_equal(filled_imex_array.diff.expl.values, -diff_values())
    assert np.array_equal(filled_imex_array.alg.values, alg_values())


def test_assign_writes_in_place_into_existing_parts(zero_array):
    diff, alg = zero_array.diff, zero_array.alg
    diff_array, alg_array = diff.values, alg.values

    zero_array.assign((diff_values(), alg_values()))

    assert zero_array.diff is diff and zero_array.alg is alg
    assert zero_array.diff.values is diff_array and zero_array.alg.values is alg_array


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


@pytest.mark.parametrize("op", OPERATORS)
def test_binary_should_raise_value_error_if_other_object_has_different_parts(op):
    a = DummyArray()
    other = OtherArray()

    other.u = np.zeros((3, 1))
    other.z = np.zeros((2, 2))

    with pytest.raises(ValueError, match="Operands must have the same parts."):
        a._binary(other, op)


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
