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
    sdc = SDCMethod(num_nodes=3, quad_type="RADAU-RIGHT", QI_type="IE")

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
