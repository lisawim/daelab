import pytest

from daelab.base.methods.sdc import SDCMethod
from daelab.base.problems.full_implicit_dae_problem import FullimplicitDAEProblem
from daelab.simulation.time_stepper import TimeStepper
from tests.test_base.test_problems.test_full_implicit_dae_problem import (
    DummyImplicitDAE,
)


class AnotherDummyImplicitDAE(FullimplicitDAEProblem):
    """Another dummy DAE problem with an implementation of F."""

    def F(self, u_dot, u, t):
        return u_dot - u + t


def test_method_should_independent_node_lists_for_different_problems():
    sdc = SDCMethod(num_nodes=3, quad_type="RADAU-RIGHT", QD_type="IE")

    a = TimeStepper(
        problem=DummyImplicitDAE(nvars=(3, 64)), method=sdc, maxiter=1, restol=1e-12
    )
    b = TimeStepper(
        problem=AnotherDummyImplicitDAE(nvars=(3, 64)),
        method=sdc,
        maxiter=1,
        restol=1e-12,
    )

    assert a.method is not b.method

    assert a.method.u is not b.method.u
    assert a.method.f is not b.method.f


def test_method_property():
    sdc = SDCMethod(num_nodes=3, quad_type="RADAU-RIGHT", QD_type="IE")
    problem = AnotherDummyImplicitDAE(nvars=(3, 64))

    stepper = TimeStepper(problem=problem, method=sdc, maxiter=1, restol=1e-12)
    assert stepper.method is not sdc
    assert isinstance(stepper.method, SDCMethod)


def test_problem_property():
    sdc = SDCMethod(num_nodes=3, quad_type="RADAU-RIGHT", QD_type="IE")
    problem = AnotherDummyImplicitDAE(nvars=(3, 64))

    stepper = TimeStepper(problem=problem, method=sdc, maxiter=1, restol=1e-12)
    assert stepper.method.problem is stepper.problem
    assert stepper.problem is problem
    assert isinstance(stepper.problem, AnotherDummyImplicitDAE)


def test_current_time_stepper_should_raise_not_implemented_error_when_called():
    sdc = SDCMethod(num_nodes=3, quad_type="RADAU-RIGHT", QD_type="IE")
    problem = AnotherDummyImplicitDAE(nvars=(3, 64))

    stepper = TimeStepper(problem=problem, method=sdc, maxiter=1, restol=1e-12)
    match = "step is not yet implemented."
    prob = stepper.problem
    with pytest.raises(NotImplementedError, match=match):
        stepper.step(prob.dtype_u(prob.init), 0.0, 0.0)
