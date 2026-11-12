import numpy as np
import pytest

from daelab.base.datatypes.array import Array


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
def array():
    nvars = (2, 3, 4)
    init = (nvars, float)
    u = Array(init=init)
    u.values[...] = np.arange(np.prod(nvars)).reshape(nvars)
    return u


def test_getitem_returns_unknown_with_spatial_dimension(array):
    assert array[1].shape == (2, 3), "Spatial dimension of second unknown is incorrect."


@pytest.mark.parametrize("ind", [0, 1, -1, slice(None), slice(0, 1), (0, 1)])
def test_getitem_matches_indexing_of_values(array, ind):
    assert np.array_equal(array[ind] == array.values[ind])