import numpy as np
import operator
import pytest

from daelab.base.datatypes.arithmetic import ArithmeticMethods
from daelab.base.datatypes.array import Array

OPERATORS = [operator.add, operator.sub, operator.mul, operator.truediv]


def test_class_initializes_array_of_desired_shape_and_dtype():
    nvars = (1, 64, 64)
    init = (nvars, complex)
    u = Array(init=init)

    assert u.shape == nvars, "Array does not have the correct shape."
    assert u.dtype == complex, "Array does not have the correct dtype."
    assert isinstance(u.values, np.ndarray), "Values is not a numpy array."


@pytest.mark.parametrize("nvars", [(1,), (3,), (2, 4), (3, 4, 5)])
def test_shape_equals_nvars(nvars):
    init = (nvars, float)
    u = Array(init=init)

    assert u.shape == nvars, "Property shape does not return the correct shape."
    assert u.shape == u.values.shape, (
        "Property shape does not return the correct shape as the numpy property."
    )


@pytest.mark.parametrize("nvars", [(1,), (3,), (2, 4), (3, 4, 5)])
def test_shape_is_a_tuple(nvars):
    init = (nvars, float)
    u = Array(init=init)

    assert isinstance(u.shape, tuple), "Shape of array is not a tuple."


def test_shape_should_raise_attribute_error_if_it_is_set():
    init = ((2, 4), float)
    u = Array(init=init)

    with pytest.raises(AttributeError):
        u.shape = (4, 2)


@pytest.mark.parametrize("datatype", [int, float, complex])
def test_dtype_equals_set_datatype(datatype):
    init = ((2, 4), datatype)
    u = Array(init=init)

    assert u.dtype == datatype, "Property shape does not return the correct shape."
    assert u.dtype == u.values.dtype, (
        "Property dtype does not return the correct dtype as the numpy property."
    )


@pytest.mark.parametrize("datatype", [int, float, complex])
def test_dtype_is_numpy_dtype_of_requested_type(datatype):
    init = ((2, 4), float)
    u = Array(init=init)

    assert isinstance(u.dtype, np.dtype), "Datatype of array is not a type."


def test_dtype_should_raise_attribute_error_if_it_is_set():
    init = ((2, 4), float)
    u = Array(init=init)

    with pytest.raises(AttributeError):
        u.dtype = complex


@pytest.mark.parametrize("nvars", [(1,), (3, 4), (2, 3, 4)])
def test_len_is_number_of_unknowns(nvars):
    init = (nvars, float)
    u = Array(init=init)
    assert len(u) == nvars[0], "Number of unknowns is not equal to length of array."


@pytest.fixture
def filled_array():
    nvars = (2, 3, 4)
    init = (nvars, float)
    u = Array(init=init)
    u.values[...] = np.arange(np.prod(nvars)).reshape(nvars)
    return u


def test_getitem_returns_unknown_with_spatial_dimension(filled_array):
    assert filled_array[1].shape == (3, 4), (
        "Spatial dimension of second unknown is incorrect."
    )


@pytest.mark.parametrize("ind", [0, 1, -1, slice(None), slice(0, 1), (0, 1)])
def test_getitem_matches_indexing_of_values(filled_array, ind):
    assert np.array_equal(filled_array[ind], filled_array.values[ind])


@pytest.mark.parametrize("index", [0, 1, slice(None), (1, slice(None), 2)])
def test_getitem_returns_reference_to_values(filled_array, index):
    assert np.shares_memory(filled_array[index], filled_array.values), (
        "Arrays does not share the memory."
    )


def test_getitem_allows_in_place_modifications(filled_array):
    filled_array[0][:] = 100.0
    assert np.all(filled_array.values[0] == 100.0), (
        "Values of first unknown have not been set to 100.0."
    )
    assert not np.any(filled_array.values[1] == 100.0), (
        "At least on value of the second unknown has been erroneously set to 100.0."
    )


@pytest.mark.parametrize("index", [2, -3])
def test_getitem_should_raise_index_error_if_index_is_out_of_range(filled_array, index):
    with pytest.raises(IndexError):
        filled_array[index]


@pytest.fixture
def zero_array():
    nvars = (2, 3, 4)
    init = (nvars, float)
    u = Array(init=init)
    return u


@pytest.mark.parametrize(
    "ind, val",
    [
        (0, 1.0),
        (1, 2.0),
        (-1, 3.0),
        (slice(None), 4.0),
        ((0, 1), 5.0),
        ((1, slice(None), 2), 6.0),
        ((Ellipsis, 0), 7.0),
    ],
)
def test_setitem_writes_only_at_index(zero_array, ind, val):
    expected = np.zeros((2, 3, 4))
    expected[ind] = val

    zero_array[ind] = val

    assert np.array_equal(zero_array, expected), (
        "val has not been written into the array."
    )


def test_setitem_writes_array_into_unknowns(zero_array):
    val = np.arange(12).reshape(3, 4)
    zero_array[1] = val

    assert np.array_equal(zero_array[1], val), (
        "val has not been written into the array."
    )
    assert np.all(zero_array[0] == 0.0), "First unknown includes nonzero values."


def test_setitem_does_not_replace_values_array(zero_array):
    ref = zero_array.values
    zero_array[0] = 5.0

    assert zero_array.values is ref


def test_setitem_copies_input(zero_array):
    val = np.ones((3, 4))
    zero_array[0] = val

    val[:] = 42.0

    assert np.all(zero_array[0] == 1.0), (
        "Values in array have been changed after writing."
    )


@pytest.mark.parametrize("val", [np.ones((4, 3)), np.ones((2, 4))])
def test_setitem_should_raise_value_error_if_shape_does_not_match(zero_array, val):
    with pytest.raises(ValueError):
        zero_array[0] = val


@pytest.mark.parametrize("ind", [2, -3])
def test_setitem_should_raise_index_error_if_index_is_out_of_range(zero_array, ind):
    with pytest.raises(IndexError):
        zero_array[ind] = 1.0


def test_asarray_returns_numpy_array_with_values(filled_array):
    arr = np.asarray(filled_array)

    assert isinstance(arr, np.ndarray), "arr is not a numpy array."
    assert arr.shape == (2, 3, 4), "arr does not have the correct shape."
    assert np.array_equal(arr, filled_array.values), (
        "Entries of both arrays are different."
    )


def test_asarray_returns_view_of_values(filled_array):
    arr = np.asarray(filled_array)

    assert np.shares_memory(arr, filled_array.values), (
        "Both arrays do not share the same memory."
    )
    arr[0, 0, 0] = -1.0
    assert filled_array.values[0, 0, 0] == -1.0, "Entry has not been changed."


def np_array_returns_copy(filled_array):
    arr = np.array(filled_array)

    assert not np.shares_memory(arr, filled_array.values), (
        "Both arrays shares the same memory."
    )
    arr[...] = 0.0
    assert np.array_equal(filled_array.values, np.arange(24).reshape(2, 3, 4)), (
        "filled_array has changed after writing into arr."
    )


def test_array_with_copy_true_returns_copy(filled_array):
    arr = filled_array.__array__(copy=True)

    assert not np.shares_memory(filled_array.values, arr), (
        "After copying arrays still share the same memory."
    )


def test_array_with_copy_false_returns_view(filled_array):
    arr = filled_array.__array__(copy=False)

    assert np.shares_memory(filled_array.values, arr), (
        "Arrays does not share the same memory."
    )


def test_asarray_with_dtype_converts_without_changing_values(filled_array):
    arr = np.asarray(filled_array, dtype=np.float32)

    assert arr.dtype == np.float32, "Dtype of array is incorrect."
    assert np.array_equal(arr, filled_array.values), (
        "Array has not been converted successfully."
    )
    assert filled_array.values.dtype == np.float64, (
        "filled_array does not have the original dtype."
    )


def test_array_should_raise_value_error_if_copy_is_false_but_dtype_differs(
    filled_array,
):
    err_msg_match = "Unable to avoid a copy while converting to this dtype."
    with pytest.raises(ValueError, match=err_msg_match):
        filled_array.__array__(dtype=np.float32, copy=False)


@pytest.mark.parametrize("datatype", [int, float, complex])
def test_asarray_has_requested_dtype(datatype):
    init = ((2, 3), datatype)
    u = Array(init=init)

    assert np.asarray(u).dtype == np.dtype(datatype), (
        "asarray does not have requested dtype."
    )


def test_copy_returns_independent_object_with_equal_values(filled_array):
    original = filled_array.values.copy()
    arr = filled_array.copy()

    assert type(arr) is type(filled_array), "Copied array is not an instance of Array."
    assert arr is not filled_array
    assert arr.shape == filled_array.shape == (2, 3, 4), (
        " Copied array does not have the correct shape."
    )
    assert arr.dtype == filled_array.dtype == np.dtype(float), (
        " Copied array does not have the correct dtype."
    )
    assert np.array_equal(arr.values, original)
    assert not np.shares_memory(arr.values, filled_array.values), (
        "Copied array shares memory with the original"
    )

    arr[...] = -1.0
    assert np.array_equal(filled_array.values, original), (
        "Original arrays have been erroneously overwritten."
    )


@pytest.mark.parametrize("val", [0.0, 3.5, -1])
def test_assign_scalar_writes_all_entries(zero_array, val):
    zero_array.assign(val)

    assert np.all(zero_array.values == val), "val isn't written to all entries."


def test_assign_numpy_array_writes_values(zero_array):
    val = np.arange(24, dtype=float).reshape(2, 3, 4)

    zero_array.assign(val)

    assert np.array_equal(zero_array.values, val), (
        "numpy array is not written into array."
    )


def test_assign_array_writes_values_of_other_array(zero_array, filled_array):
    zero_array.assign(filled_array)

    assert np.array_equal(zero_array.values, filled_array.values), (
        "Array is not written into other array."
    )


def test_assign_array_does_not_share_memory_with_other_array(zero_array, filled_array):
    original = filled_array.values.copy()

    zero_array.assign(filled_array)
    zero_array[...] = -1.0

    assert not np.shares_memory(zero_array.values, filled_array.values), (
        "Arrays share the same memory after assigning an array into another."
    )
    assert np.array_equal(filled_array.values, original), (
        "The array that was written to has changed from the original."
    )


def test_assign_does_not_replace_values_array(zero_array):
    ref = zero_array.values

    zero_array.assign(5.0)

    assert zero_array.values is ref


def test_assign_copies_input(zero_array):
    val = np.ones((2, 3, 4))

    zero_array.assign(val)
    val[:] = 9.0

    assert np.all(zero_array.values == 1.0), (
        "Values in array have been changed after assigning."
    )


def test_assign_broadcasts_array_of_lower_dimension(zero_array):
    zero_array.assign(np.arange(4.0))

    assert np.array_equal(zero_array.values, np.broadcast_to(np.arange(4.0), (2, 3, 4)))


@pytest.mark.parametrize("val", [np.ones((3, 2)), np.ones((2, 3, 5))])
def test_assign_should_raise_value_error_if_shape_does_not_match(zero_array, val):
    with pytest.raises(ValueError):
        zero_array.assign(val)


def test_assign_should_raise_value_error_if_other_array_has_different_shape(zero_array):
    with pytest.raises(ValueError):
        zero_array.assign(Array(((2, 3, 5), float)))


@pytest.mark.parametrize("nvars", [(1,), (3,), (2, 4), (2, 3, 4)])
def test_flat_has_length_prod_of_nvars(nvars):
    a = Array((nvars, float))

    assert a.flat().shape == (np.prod(nvars),), (
        "Flattened array does not have correct shape."
    )


def test_flat_is_one_dimensional_numpy_array(filled_array):
    flat = filled_array.flat()

    assert isinstance(flat, np.ndarray), "Flattened array is not a numpy array."
    assert flat.ndim == 1, "Flattened array does not have dimension one."


def test_flat_has_values_in_c_order(filled_array):
    assert np.array_equal(filled_array.flat(), np.arange(24))


def test_flat_reshaped_gives_values_back(filled_array):
    flat = filled_array.flat()

    assert np.array_equal(flat.reshape(filled_array.shape), filled_array.values), (
        "Reshaping of flattened array does not lead to original array."
    )


def test_flat_returns_view_of_values(filled_array):
    flat = filled_array.flat()

    assert np.shares_memory(flat, filled_array.values), "Flattened array is not a view."
    flat[:] = 0.0
    assert np.all(filled_array.values == 0.0), "All entries must be zero."


class OtherArithmetic(ArithmeticMethods):
    def _binary(self, other, op):
        return "other"


class MyArray(Array):
    pass


@pytest.mark.parametrize("op", OPERATORS)
def test_binary_with_scalar_applies_operation_to_values(filled_array, op):
    result = filled_array._binary(2.0, op)

    assert np.array_equal(result.values, op(filled_array.values, 2.0)), (
        "op is not applied to all values."
    )


@pytest.mark.parametrize("op", OPERATORS)
def test_binary_with_array_applies_operation_to_values(filled_array, op):
    other = Array(((2, 3, 4), float))
    other.values[...] = filled_array.values + 1.0

    result = filled_array._binary(other, op)

    assert np.array_equal(result.values, op(filled_array.values, other.values))


@pytest.mark.parametrize("op", OPERATORS)
def test_binary_with_numpy_array_broadcasts(filled_array, op):
    other = np.arange(1.0, 5.0)

    result = filled_array._binary(other, op)

    assert np.array_equal(result.values, op(filled_array.values, other))


def test_binary_returns_new_object_and_leaves_operands_unchanged(filled_array):
    other = Array(((2, 3, 4), float))
    other.values[...] = filled_array.values + 1.0
    original, other_original = filled_array.values.copy(), other.values.copy()

    result = filled_array._binary(other, operator.add)

    assert result is not filled_array and result is not other
    assert not np.shares_memory(result.values, filled_array.values)
    assert not np.shares_memory(result.values, other.values)
    assert np.array_equal(filled_array.values, original)
    assert np.array_equal(other.values, other_original)


def test_binary_keeps_type_of_subclass():
    u = MyArray(((2, 3), float))

    assert type(u._binary(1.0, operator.add)) is MyArray, (
        "After operation type of array has changed."
    )


def test_binary_result_dtype_follows_numpy_promotion():
    u = Array(((2, 3), int))

    result = u._binary(0.5, operator.add)

    assert result.dtype == np.dtype(float)
    assert u.dtype == np.dtype(int)


def test_binary_returns_not_implemented_for_other_arithmetic_type(filled_array):
    assert filled_array._binary(OtherArithmetic(), operator.add) is NotImplemented


def test_binary_should_raise_value_error_if_shapes_do_not_match(filled_array):
    with pytest.raises(ValueError):
        filled_array._binary(np.ones((3, 2)), operator.add)
