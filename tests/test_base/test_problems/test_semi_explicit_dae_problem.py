import pytest

from daelab.base.problems.semi_explicit_dae_problem import SemiexplicitDAEProblem


class IncompleteSemiexplicitDAE(SemiexplicitDAEProblem):
    """Dummy DAE problem without implementations of f and g."""


class DummySemiexplicitDAE(SemiexplicitDAEProblem):
    """Dummy DAE problem with implementations of f and g."""

    def f(self, u, t):
        return -u.diff[:] + u.alg[:]

    def g(self, u, t):
        return u.diff[:] + u.alg[:] - 1


class DummySemiexplicitDAEWithExactSolution(DummySemiexplicitDAE):
    """Dummy DAE problem with an exact solution."""

    def u_exact(self, t):
        return 1.0


class DummyMissingF(SemiexplicitDAEProblem):
    """Dummy DAE problem without implementation of f."""

    def g(self, u, t):
        return u.diff[:] + u.alg[:] - 1


class DummyMissingG(SemiexplicitDAEProblem):
    """Dummy DAE problem without implementation of g."""

    def f(self, u, t):
        return -u.diff[:] + u.alg[:]


def test_attribute_init_should_be_tuple_of_dtype_and_nvars(
    nvars_diff=(1,), nvars_alg=(3,)
):
    problem = DummySemiexplicitDAE(
        nvars_diff=nvars_diff, nvars_alg=nvars_alg, dtype=complex
    )

    assert problem.init == (nvars_diff, nvars_alg, complex), (
        "init is not set to (nvars, dtype)."
    )


def test_attribute_is_semiexplicit_dae_should_set_to_True(nvars=(1,)):
    problem = DummySemiexplicitDAE(nvars_diff=nvars, nvars_alg=nvars)

    assert problem.is_semiexplicit_dae, "is_semiexplicit_dae is not set to True!"


def test_nvars_attributes_as_integer_is_transformed_into_correct_shape():
    nvars = 1
    dae_problem = DummySemiexplicitDAE(nvars_diff=nvars, nvars_alg=nvars)

    assert dae_problem.nvars_diff == (nvars,), (
        "nvars_diff is not transformed into the correct shape!"
    )
    assert dae_problem.nvars_alg == (nvars,), (
        "nvars_alg is not transformed into the correct shape!"
    )


def test_should_raise_value_error_if_diff_alg_vars_have_different_spatial_dimensions():
    nvars_diff = (2, 128, 256)
    nvars_alg = (1, 128, 128)

    err_msg = (
        "Differential and algebraic variables must have the same spatial dimensions"
    )
    with pytest.raises(ValueError, match=err_msg):
        DummySemiexplicitDAE(nvars_diff=nvars_diff, nvars_alg=nvars_alg)


def test_raises_type_error_without_f_and_g(nvars=(1,)):
    with pytest.raises(TypeError):
        IncompleteSemiexplicitDAE(nvars_diff=nvars, nvars_alg=nvars)


def test_abstract_f_raises_not_implemented_error_when_called(nvars=(1,)):
    problem = DummySemiexplicitDAE(nvars_diff=nvars, nvars_alg=nvars)

    with pytest.raises(
        NotImplementedError, match=r"Subclasses must implement f\(u, t\)\."
    ):
        super(DummySemiexplicitDAE, problem).f(None, 0.0)


def test_abstract_g_raises_not_implemented_error_when_called(nvars=(1,)):
    problem = DummySemiexplicitDAE(nvars_diff=nvars, nvars_alg=nvars)

    with pytest.raises(
        NotImplementedError, match=r"Subclasses must implement g\(u, t\)\."
    ):
        super(DummySemiexplicitDAE, problem).g(None, 0.0)


def test_instantiation_is_successful(nvars=(1,)):
    problem = DummySemiexplicitDAE(nvars_diff=nvars, nvars_alg=nvars)

    assert isinstance(problem, SemiexplicitDAEProblem), (
        "Problem was not an instance of SemiexplicitDAEProblem!"
    )


def test_raises_type_error_without_f(nvars=(1,)):
    with pytest.raises(TypeError):
        DummyMissingF(nvars_diff=nvars, nvars_alg=nvars)


def test_raises_type_error_without_g(nvars=(1,)):
    with pytest.raises(TypeError):
        DummyMissingG(nvars_diff=nvars, nvars_alg=nvars)


def test_raises_not_implemented_error_if_methods_not_implemented(nvars=(1,)):
    problem = DummySemiexplicitDAE(nvars_diff=nvars, nvars_alg=nvars)

    u0 = problem.dtype_u(problem.init)
    u0[:] = (1.5, 2.0)

    rhs = problem.dtype_u(problem.init)
    rhs[:] = (0.25, 0.2)

    with pytest.raises(
        NotImplementedError, match=r"Subclasses must implement solve_system\(\)."
    ):
        problem.solve_system(1.0, u0, 0.5, rhs)

    with pytest.raises(
        NotImplementedError,
        match="This problem does not provide an exact solution for u.",
    ):
        problem.u_exact(0.0)


def test_u_exact_can_be_overridden_in_subclass(nvars=(1,)):
    problem = DummySemiexplicitDAEWithExactSolution(nvars_diff=nvars, nvars_alg=nvars)

    assert problem.u_exact(0.0) == 1.0, (
        "u_exact was not successfully overriden in subclass!"
    )
