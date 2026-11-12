import numpy as np
import pytest

from daelab.base.datatypes.array import Array
from daelab.base.datatypes.partitioned import IMEXArray
from daelab.base.datatypes.semiexplicit import SemiexplicitArray, SemiexplicitIMEXArray


def test_semiexplicitarray_is_initialized_correctly():
    u = SemiexplicitArray(((2, 3, 4), (1, 3, 4), float))

    assert u.nvars_diff == (2, 3, 4)
    assert u.nvars_alg == (1, 3, 4)
    assert u.dtype == np.dtype(float)
    assert type(u.diff) is Array and type(u.alg) is Array


def test_semiexplicitarray_should_raise_value_error_if_spatial_dimensions_differ():
    err_msg_match = "nvars_diff and nvars_alg must have the same spatial dimensions."
    with pytest.raises(ValueError, match=err_msg_match):
        SemiexplicitArray(((2, 3, 4), (1, 4, 5), float))


def test_diff_part_of_semiexplicitimexarray_is_of_type_imexarray():
    u = SemiexplicitIMEXArray(((2, 3, 4), (1, 3, 4), float))
    assert type(u.diff) is IMEXArray and type(u.alg) is Array
