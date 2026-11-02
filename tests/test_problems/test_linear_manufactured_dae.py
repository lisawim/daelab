import numpy as np
import pytest

from daelab.problems.linear_manufactured_dae import LinearManufacturedDAE


def test_attributes_are_initialized_by_default_after_instantiation():
    problem = LinearManufacturedDAE()

    assert problem.diff_index == 1, "diff_index is not initialized as one!"
    assert problem.t_span == (0.0, 1.0), "t_span is not initialized as (0.0, 1.0)!"
    assert problem.name == "Linear manufactured DAE", "name is not initialized as 'Linear manufactured DAE'!"


@pytest.mark.parametrize(
    "y, z, t, expected",
    [
        (1.0, -2.0, 0.0, -4.0),
        (np.exp(-2.0), -2.0 * np.exp(-2.0), 0.5, -4.0 * np.exp(-2.0)),
        (np.exp(-4.0), -2.0 * np.exp(-4.0), 1.0, -4.0 * np.exp(-4.0)),
    ],
)
def test_f_returns_correct_rhs(y, z, t, expected):
    problem = LinearManufacturedDAE()

    assert problem.f(y, z, t) == pytest.approx(expected), "Computed rhs does not match with expected rhs!"


@pytest.mark.parametrize(
    "y, z, t",
    [
        (1.0, -2.0, 0.0),
        (np.exp(-2.0), -2.0 * np.exp(-2.0), 0.5),
        (np.exp(-4.0), -2.0 * np.exp(-4.0), 1.0),
    ],
)
def test_g_returns_correct_rhs(y, z, t):
    problem = LinearManufacturedDAE()

    assert problem.g(y, z, t) == 0.0, "Computed rhs does not match with expected rhs!"


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_y_exact_returns_correct_solution(t):
    problem = LinearManufacturedDAE()

    assert problem.y_exact(t) == np.exp(-4 * t), "Computed solution does not match with expected solution!"


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_z_exact_returns_correct_solution(t):
    problem = LinearManufacturedDAE()

    assert problem.z_exact(t) == -2 * np.exp(-4 * t), "Computed solution does not match with expected solution!"


@pytest.mark.parametrize("t", [0.0, 0.5, 1.0])
def test_exact_solution_is_consistent_with_dae(t):
    problem = LinearManufacturedDAE()

    y, z = problem.y_exact(t), problem.z_exact(t)

    y_dot = -4 * np.exp(-4 * t)

    # Consistency is checked for y_dot; consistency for g is already checked above.
    assert y_dot == problem.f(y, z, t), "y_dot does not match with the rhs f!"
