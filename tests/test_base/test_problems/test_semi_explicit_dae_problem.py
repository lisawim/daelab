import numpy as np
import pytest

from daelab.base.datatypes.array import Array
from daelab.base.datatypes.partitioned import IMEXArray
from daelab.base.datatypes.semiexplicit import SemiexplicitArray, SemiexplicitIMEXArray
from daelab.base.problems.semi_explicit_dae_problem import SemiexplicitDAEProblem


class OtherArray(Array):
    pass


class CustomPartsArray(SemiexplicitArray):
    dtype_diff = OtherArray
    dtype_alg = OtherArray


class IncompleteSemiexplicitDAE(SemiexplicitDAEProblem):
    """Dummy DAE problem without implementations of f_diff and f_alg."""


class DummySemiexplicitDAE(SemiexplicitDAEProblem):
    """Dummy DAE problem with implementations of f_diff and f_alg."""

    def f_diff(self, u, t):
        return -u.diff[:] + u.alg[:]

    def f_alg(self, u, t):
        return u.diff[:] + u.alg[:] - 1


class ProblemWithCustomParts(DummySemiexplicitDAE):
    dtype_u = CustomPartsArray


class ProblemWithImexRhs(DummySemiexplicitDAE):
    dtype_f = SemiexplicitIMEXArray


class DummySemiexplicitDAEWithExactSolution(DummySemiexplicitDAE):
    """Dummy DAE problem with an exact solution."""

    def u_exact(self, t):
        return 1.0


class DummyMissingFDiff(SemiexplicitDAEProblem):
    """Dummy DAE problem without implementation of f_diff."""

    def f_alg(self, u, t):
        return u.diff[:] + u.alg[:] - 1


class DummyMissingFAlg(SemiexplicitDAEProblem):
    """Dummy DAE problem without implementation of f_alg."""

    def f_diff(self, u, t):
        return -u.diff[:] + u.alg[:]


def test_attribute_init_should_be_tuple_of_dtype_and_nvars(
    nvars_diff=(1,), nvars_alg=(3,)
):
    problem = DummySemiexplicitDAE(
        nvars_diff=nvars_diff, nvars_alg=nvars_alg, dtype=complex
    )

    assert problem.init == (nvars_diff, nvars_alg, complex), (
        "init is not set to (nvars_diff, nvars_alg, dtype)."
    )
    assert problem.init_diff == (nvars_diff, complex), (
        "init_diff is not set to (nvars_diff, dtype)."
    )
    assert problem.init_alg == (nvars_alg, complex), (
        "init_alg is not set to (nvars_alg, dtype)."
    )


def test_dtype_diff_and_dtype_alg_are_array_by_default():
    problem = DummySemiexplicitDAE((2,), (1,))

    assert problem.dtype_diff is Array
    assert problem.dtype_alg is Array


def test_dtype_diff_and_dtype_alg_follow_dtype_u():
    problem = ProblemWithCustomParts((2,), (1,))

    assert problem.dtype_diff is OtherArray
    assert problem.dtype_alg is OtherArray


def test_dtype_diff_and_dtype_alg_do_not_depend_on_dtype_f():
    problem = ProblemWithImexRhs((2,), (1,))

    assert problem.dtype_diff is Array
    assert type(problem.dtype_f(problem.init).diff) is IMEXArray


def test_dtype_diff_and_dtype_alg_create_zero_arrays_with_shape_of_the_parts():
    problem = DummySemiexplicitDAE((2, 3), (1, 3), dtype=float)

    d, a = problem.dtype_diff(problem.init_diff), problem.dtype_alg(problem.init_alg)

    assert d.shape == (2, 3) and a.shape == (1, 3)
    assert d.dtype == np.dtype(float) and a.dtype == np.dtype(float)

    assert not np.any(d.values) and not np.any(a.values)


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


def test_raises_type_error_without_f_diff_and_f_alg(nvars=(1,)):
    with pytest.raises(TypeError):
        IncompleteSemiexplicitDAE(nvars_diff=nvars, nvars_alg=nvars)


def test_abstract_f_diff_raises_not_implemented_error_when_called(nvars=(1,)):
    problem = DummySemiexplicitDAE(nvars_diff=nvars, nvars_alg=nvars)

    with pytest.raises(
        NotImplementedError, match=r"Subclasses must implement f_diff\(u, t\)\."
    ):
        super(DummySemiexplicitDAE, problem).f_diff(None, 0.0)


def test_abstract_f_alg_raises_not_implemented_error_when_called(nvars=(1,)):
    problem = DummySemiexplicitDAE(nvars_diff=nvars, nvars_alg=nvars)

    with pytest.raises(
        NotImplementedError, match=r"Subclasses must implement f_alg\(u, t\)\."
    ):
        super(DummySemiexplicitDAE, problem).f_alg(None, 0.0)


def test_instantiation_is_successful(nvars=(1,)):
    problem = DummySemiexplicitDAE(nvars_diff=nvars, nvars_alg=nvars)

    assert isinstance(problem, SemiexplicitDAEProblem), (
        "Problem was not an instance of SemiexplicitDAEProblem!"
    )


def test_raises_type_error_without_f_diff(nvars=(1,)):
    with pytest.raises(TypeError):
        DummyMissingFDiff(nvars_diff=nvars, nvars_alg=nvars)


def test_raises_type_error_without_f_alg(nvars=(1,)):
    with pytest.raises(TypeError):
        DummyMissingFAlg(nvars_diff=nvars, nvars_alg=nvars)


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
