import numpy as np
import pytest

from daelab.problems.linear_manufactured_dae import LinearManufacturedDAE


def test_attributes_are_initialized_by_default_after_instantiation():
    problem = LinearManufacturedDAE()

    assert problem.nvars_diff == problem.nvars_alg == (1,), (
        "nvars_diff and nvars_alg are not set to (1,)!"
    )
    assert problem.dtype == float, "dtype is not set to float!"
    assert problem.init == (problem.nvars_diff, problem.nvars_alg, problem.dtype), (
        "init is not correctly set!"
    )
    assert problem.is_semiexplicit_dae, "is_semiexplicit_dae is not set to True!"
    assert problem.diff_index == 1, "diff_index is not initialized as one!"
    assert problem.t_span == (0.0, 1.0), "t_span is not initialized as (0.0, 1.0)!"
    assert problem.name == "Linear manufactured DAE", (
        "name is not initialized as 'Linear manufactured DAE'!"
    )


@pytest.mark.parametrize(
    "y_z, t, expected",
    [
        ((1.0, -2.0), 0.0, -4.0),
        ((np.exp(-2.0), -2.0 * np.exp(-2.0)), 0.5, -4.0 * np.exp(-2.0)),
        ((np.exp(-4.0), -2.0 * np.exp(-4.0)), 1.0, -4.0 * np.exp(-4.0)),
    ],
)
def test_f_returns_correct_rhs(y_z, t, expected):
    problem = LinearManufacturedDAE()

    u = problem.dtype_u(problem.init)
    u[:] = y_z

    np.testing.assert_allclose(problem.f(u, t), expected)


@pytest.mark.parametrize(
    "y_z, t",
    [
        ((1.0, -2.0), 0.0),
        ((np.exp(-2.0), -2.0 * np.exp(-2.0)), 0.5),
        ((np.exp(-4.0), -2.0 * np.exp(-4.0)), 1.0),
    ],
)
def test_g_returns_correct_rhs(y_z, t):
    problem = LinearManufacturedDAE()

    u = problem.dtype_u(problem.init)
    u[:] = y_z
    np.testing.assert_allclose(problem.g(u, t), 0.0)


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_u_exact_returns_correct_solution(t):
    problem = LinearManufacturedDAE()

    u = problem.u_exact(t)
    y, z = u

    np.testing.assert_allclose(y, np.exp(-4 * t))
    np.testing.assert_allclose(z, -2 * np.exp(-4 * t))


def test_solve_system_accurately_solves_the_node_system():
    prob = LinearManufacturedDAE()

    t0 = 0.0
    u0 = prob.u_exact(t=t0)
    y0, z0 = u0

    u0, rhs = prob.dtype_u(prob.init), prob.dtype_u(prob.init)
    u0[:], rhs[:] = (y0, z0), (y0, z0)

    dt = 0.01
    u = prob.solve_system(factor=dt, u0=u0, t=t0, rhs=rhs)
    y, z = u

    u1 = prob.u_exact(t=t0 + dt)
    y1, z1 = u1

    np.testing.assert_allclose(y, y1, atol=7.5e-4, rtol=0)
    np.testing.assert_allclose(z, z1, atol=1.5e-3, rtol=0)


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_exact_solution_is_consistent_with_dae(t):
    problem = LinearManufacturedDAE()

    u = problem.u_exact(t)

    y_dot = -4 * np.exp(-4 * t)

    # Consistency checked for y_dot; consistency for g is already checked above
    np.testing.assert_allclose(y_dot, problem.f(u, t))
