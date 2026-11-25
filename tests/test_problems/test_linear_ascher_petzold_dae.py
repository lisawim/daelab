import numpy as np
import pytest

from daelab.base.datatypes.semiexplicit import SemiexplicitArray
from daelab.problems.linear_ascher_petzold_dae import LinearAscherPetzoldDAE


def test_attributes_are_initialized_by_default_after_instantiation():
    problem = LinearAscherPetzoldDAE()

    assert problem.dtype_u is SemiexplicitArray, (
        "dtype_u is not of SemiexplicitArray type."
    )
    assert problem.dtype_f is SemiexplicitArray, (
        "dtype_f is not of SemiexplicitArray type."
    )

    assert problem.nvars_diff == (2,), "nvars_diff is not set to (2,)!"
    assert problem.nvars_alg == (1,), "nvars_alg is not set to (1,)!"
    assert problem.dtype == float, "dtype is not set to float!"
    assert problem.init == (problem.nvars_diff, problem.nvars_alg, problem.dtype), (
        "init is not correctly set!"
    )
    assert problem.is_semiexplicit_dae, "is_semiexplicit_dae is not set to True!"
    assert problem.diff_index == 2, "diff_index is not initialized as two!"
    assert problem.t_span == (0.0, 1.0), "t_span is not initialized as (0.0, 1.0)!"
    assert problem.name == "Linear Ascher-Petzold DAE", (
        "name is not initialized as 'Ascher-Petzold'!"
    )


@pytest.mark.parametrize("t", np.random.uniform(0.0, 1.0, size=100))
def test_f_diff_returns_correct_rhs(t):
    problem = LinearAscherPetzoldDAE()

    u = problem.dtype_u(problem.init)
    x, z = (np.exp(t), np.exp(t)), -np.exp(t) / (2.0 - t)
    u[:] = (x, z)

    expected = (np.exp(t), np.exp(t))
    np.testing.assert_allclose(problem.f_diff(u, t), expected, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("t", np.random.uniform(0.0, 1.0, size=100))
def test_f_alg_returns_zero(t):
    problem = LinearAscherPetzoldDAE()

    u = problem.dtype_u(problem.init)
    x, z = (np.exp(t), np.exp(t)), -np.exp(t) / (2.0 - t)
    u[:] = (x, z)
    np.testing.assert_allclose(problem.f_alg(u, t), 0.0, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_u_exact_returns_correct_solution(t):
    problem = LinearAscherPetzoldDAE()

    u = problem.u_exact(t)
    x, z = u

    np.testing.assert_allclose(x, (np.exp(t), np.exp(t)))
    np.testing.assert_allclose(z, -np.exp(t) / (2 - t))


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_f_returns_f_diff_and_f_alg(t):
    problem = LinearAscherPetzoldDAE()

    u = problem.u_exact(t)
    f = problem.f(u, t)
    f_diff = problem.f_diff(u, t)
    f_alg = problem.f_alg(u, t)

    np.testing.assert_allclose(f.diff, f_diff)
    np.testing.assert_allclose(f.alg, f_alg)
    assert f.diff.shape == problem.init[0]
    assert f.alg.shape == problem.init[1]
    # assert f.diff.shape == f_diff.shape == problem.init[0]
    # assert f.alg.shape == f_alg.shape == problem.init[1]
