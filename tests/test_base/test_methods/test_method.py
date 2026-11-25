import pytest

from daelab.base.methods.method import Method
from daelab.base.methods.sdc import SDCMethod
from daelab.base.problems.dae_problem import DAEProblem
from daelab.problems.linear_manufactured_dae import LinearManufacturedDAE
from tests.test_base.test_problems.test_semi_explicit_dae_problem import (
    DummySemiexplicitDAE,
)


class IncompleteMethod(Method):
    """Class without setup method."""


class DummyMethod(Method):
    """Dummy class with setup method."""

    def setup(self, problem: DAEProblem):
        self.problem = problem


def test_raises_type_error_without_setup(nvars=(1,)):
    with pytest.raises(TypeError):
        IncompleteMethod()


def test_method_stores_problem_as_None_after_instantiation():
    method = DummyMethod()
    assert method.problem is None, "problem is not stored as None."


def test_abstract_setup_raises_not_implemented_error_when_called():
    method = DummyMethod()
    problem = LinearManufacturedDAE()

    match = r"Subclasses must implement setup\(problem\)\."
    with pytest.raises(NotImplementedError, match=match):
        super(DummyMethod, method).setup(problem)


def test_bind_should_return_independent_copies():
    method = DummyMethod()
    prob_a = LinearManufacturedDAE()
    prob_b = DAEProblem(diff_index=1, t_span=(0.0, 1.0))

    bound_a = method.bind(problem=prob_a)
    bound_b = method.bind(problem=prob_b)

    assert bound_a is not bound_b


def test_method_should_create_independent_node_lists_for_different_problems():
    sdc = SDCMethod(num_nodes=3, quad_type="RADAU-RIGHT", QD_type="IE")
    prob_a = LinearManufacturedDAE()
    prob_b = DummySemiexplicitDAE(nvars_diff=(1,), nvars_alg=(1,))

    sdc_a = sdc.bind(problem=prob_a)
    sdc_b = sdc.bind(problem=prob_b)

    assert sdc_a.u is not sdc_b.u
    assert sdc_a.f is not sdc_b.f


def test_require_setup_should_raise_runtime_error_if_problem_is_still_None():
    method = DummyMethod()

    match = (
        r"DummyMethod is not set up. Use TimeStepper\(method, problem\)"
        r" or method.bind\(problem\)\."
    )
    with pytest.raises(RuntimeError, match=match):
        method._require_setup()


def test_require_setup_does_not_raise_if_problem_is_set():
    method = DummyMethod()
    problem = LinearManufacturedDAE()

    assert method.problem is None

    bound = method.bind(problem)

    assert method.problem is None
    assert bound.problem is not None

    bound._require_setup()
