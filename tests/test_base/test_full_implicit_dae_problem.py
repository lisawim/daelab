import pytest

from daelab.base.full_implicit_dae_problem import FullimplicitDAEProblem


class DummyImplicitDAE(FullimplicitDAEProblem):
    """Dummy DAE problem with an implementation of F."""

    def F(self, u_dot, u, t):
        return u_dot + u


class DummyImplicitDAEWithExactSolution(DummyImplicitDAE):
    """Dummy DAE problem with an exact solution."""

    def u_exact(self, t):
        return 1.0


def test_raises_not_implemented_error_during_instatiation_without_F():
    problem = DummyImplicitDAE()

    with pytest.raises(NotImplementedError):
        super(DummyImplicitDAE, problem).F(None, None, 0.0)


def test_instance_with_F_is_successful():
    problem = DummyImplicitDAE()

    assert isinstance(problem, FullimplicitDAEProblem)


def test_raises_notimplemented_error_if_u_exact_not_implemented():
    problem = DummyImplicitDAE()

    with pytest.raises(NotImplementedError):
        problem.u_exact(0.0)


def test_u_exact_can_be_overridden():
    problem = DummyImplicitDAEWithExactSolution()

    assert problem.u_exact(0.0) == 1.0
