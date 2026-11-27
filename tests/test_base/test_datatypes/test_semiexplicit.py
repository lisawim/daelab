import numpy as np
import pytest

from daelab.base.datatypes.array import Array
from daelab.base.datatypes.partitioned import IMEXArray
from daelab.base.datatypes.semiexplicit import SemiexplicitArray, SemiexplicitIMEXArray

INIT = ((2, 3, 4), (1, 3, 4), float)


class OtherArray(Array):
    pass


class CustomPartsArray(SemiexplicitArray):
    dtype_diff = OtherArray
    dtype_alg = OtherArray


class OnlyDiffCustom(SemiexplicitArray):
    dtype_diff = OtherArray


def test_part_types_default_to_array():
    assert SemiexplicitArray.dtype_diff is Array
    assert SemiexplicitArray.dtype_alg is Array


def test_semiexplicitarray_is_initialized_correctly():
    u = SemiexplicitArray(INIT)

    assert u.dtype_diff is Array
    assert u.dtype_alg is Array
    assert u.nvars_diff == (2, 3, 4)
    assert u.nvars_alg == (1, 3, 4)
    assert u.dtype == np.dtype(float)
    assert type(u.diff) is Array and type(u.alg) is Array


def test_init_creates_parts_of_the_declared_part_types():
    u = CustomPartsArray(INIT)

    assert type(u.diff) is OtherArray
    assert type(u.alg) is OtherArray


def test_init_creates_parts_of_the_declared_part_types_separately():
    u = OnlyDiffCustom(INIT)

    assert type(u.diff) is OtherArray
    assert type(u.alg) is Array


def test_semiexplicitarray_should_raise_value_error_if_spatial_dimensions_differ():
    err_msg_match = "nvars_diff and nvars_alg must have the same spatial dimensions."
    with pytest.raises(ValueError, match=err_msg_match):
        SemiexplicitArray(((2, 3, 4), (1, 4, 5), float))


def test_imex_container_splits_only_diff_part():
    assert SemiexplicitIMEXArray.dtype_diff is IMEXArray
    assert SemiexplicitIMEXArray.dtype_alg is Array


def test_diff_part_of_semiexplicitimexarray_is_of_type_imexarray():
    u = SemiexplicitIMEXArray(INIT)
    assert type(u.diff) is IMEXArray and type(u.alg) is Array
