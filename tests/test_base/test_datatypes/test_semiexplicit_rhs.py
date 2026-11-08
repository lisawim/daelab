import numpy as np

from daelab.base.datatypes.rhs import RHS
from daelab.base.datatypes.imex_rhs import IMEXRHS
from daelab.base.datatypes.semiexplicit_rhs import SemiexplicitRHS


def test_vars_are_assigned_correctly_in_SemiexplicitRHS():
    num_nodes = 4
    nvars = (128, 128)
    dtype = float
    imex_diff = False
    semiexplicit_rhs = SemiexplicitRHS(
        num_nodes=num_nodes,
        nvars_diff=nvars,
        nvars_alg=nvars,
        dtype=dtype,
        imex_diff=imex_diff,
    )

    assert semiexplicit_rhs.num_nodes == num_nodes
    assert semiexplicit_rhs.nvars_diff == nvars
    assert semiexplicit_rhs.nvars_alg == nvars
    assert semiexplicit_rhs.dtype == dtype
    assert semiexplicit_rhs.imex_diff == imex_diff
    assert isinstance(semiexplicit_rhs.diff, RHS)
    assert isinstance(semiexplicit_rhs.alg, RHS)


def test_SemiExplicitRHS_can_use_imex_splitting():
    num_nodes = 4
    nvars = (128, 128)
    dtype = float
    imex_diff = True
    se_imex_rhs = SemiexplicitRHS(
        num_nodes=num_nodes,
        nvars_diff=nvars,
        nvars_alg=nvars,
        dtype=dtype,
        imex_diff=imex_diff,
    )

    assert isinstance(se_imex_rhs.diff, IMEXRHS)
    assert isinstance(se_imex_rhs.alg, RHS)

    se_imex_rhs.diff.impl[1] = 2.5 * np.ones(nvars)
    se_imex_rhs.diff.expl[1] = 6 * np.ones(nvars)

    se_imex_rhs.diff.impl[2] = 0.25 * np.ones(nvars)
    se_imex_rhs.diff.expl[2] = 0.33 * np.ones(nvars)

    assert np.array_equal(
        se_imex_rhs.diff.full_rhs[1],
        se_imex_rhs.diff.impl[1] + se_imex_rhs.diff.expl[1],
    )
    assert np.array_equal(
        se_imex_rhs.diff.full_rhs[2],
        se_imex_rhs.diff.impl[2] + se_imex_rhs.diff.expl[2],
    )
