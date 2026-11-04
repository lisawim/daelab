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


def test_attribute_is_semiexplicit_dae_should_set_to_False(nvars=((1,),)):
    problem = DummyImplicitDAE(nvars=nvars)

    assert problem.is_semiexplicit_dae == False, "is_semiexplicit_dae is not set to False!"


def test_raises_not_implemented_error_during_instatiation_without_F(nvars=((1,),)):
    problem = DummyImplicitDAE(nvars=nvars)

    with pytest.raises(NotImplementedError, match=r"Subclasses must implement F\(u_dot, u, t\)\."):
        super(DummyImplicitDAE, problem).F(None, None, 0.0)


def test_instance_with_F_is_successful(nvars=((1,),)):
    problem = DummyImplicitDAE(nvars=nvars)

    assert isinstance(problem, FullimplicitDAEProblem), "Problem was not an instance of FullimplicitDAEProblem!"


def test_raises_notimplemented_error_if_u_exact_not_implemented(nvars=((1,),)):
    problem = DummyImplicitDAE(nvars=nvars)

    with pytest.raises(NotImplementedError, match="This problem does not provide an exact solution."):
        problem.u_exact(0.0)


def test_u_exact_can_be_overridden_in_subclass(nvars=((1,),)):
    problem = DummyImplicitDAEWithExactSolution(nvars=nvars)

    assert problem.u_exact(0.0) == 1.0
