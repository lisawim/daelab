import numpy as np

from daelab.base.datatypes.imex_rhs import IMEXRHS
from daelab.base.datatypes.rhs import RHS


def test_vars_are_assigned_correctly_in_IMEXRHS():
    num_nodes = 4
    nvars = (128, 128)
    dtype = float
    imex_rhs = IMEXRHS(num_nodes=num_nodes, nvars=nvars, dtype=dtype)

    assert imex_rhs.num_nodes == num_nodes
    assert imex_rhs.nvars == nvars
    assert isinstance(imex_rhs.impl, RHS)
    assert isinstance(imex_rhs.expl, RHS)


def test_full_rhs_should_return_the_complete_rhs():
    num_nodes = 4
    nvars = (128, 128)
    dtype = float
    imex_rhs = IMEXRHS(num_nodes=num_nodes, nvars=nvars, dtype=dtype)

    imex_rhs.impl[1] = 2 * np.ones(nvars)
    imex_rhs.expl[1] = 3 * np.ones(nvars)

    imex_rhs.impl[2] = 1.67 * np.ones(nvars)
    imex_rhs.expl[2] = 5 * np.ones(nvars)

    assert np.array_equal(imex_rhs.full_rhs[1], imex_rhs.impl[1] + imex_rhs.expl[1])
    assert np.array_equal(imex_rhs.full_rhs[2], imex_rhs.impl[2] + imex_rhs.expl[2])


def test_function_repr_returns_correct_output_IMEXRHS():
    num_nodes = 4
    nvars = (128, 128)
    dtype = float
    imex_rhs = IMEXRHS(num_nodes=num_nodes, nvars=nvars, dtype=dtype)

    assert str(imex_rhs) == f"IMEXRHS({num_nodes=}, {nvars=}, dtype={np.dtype(dtype)})"
