import pytest


def test_initialize_dae_problem_correctly():
    from daelab.base.dae_problem import DAEProblem

    t_span = (0.0, 1.0)
    name = "DAE"
    dae_problem = DAEProblem(t_span=t_span, name=name)

    assert dae_problem.t0 == t_span[0], "t_span is not initialized correctly!"
    assert dae_problem.t_end == t_span[1], "t_span is not initialized correctly!"
    assert t_span == dae_problem.t_span, "t_span is not initialized correctly!"
    assert name == dae_problem.name, "name is not initialized correctly!"

def test_value_error_invalid_t_span_is_raised_correctly():
    from daelab.base.dae_problem import DAEProblem

    with pytest.raises(ValueError, match="t_span is a tuple containing two values!"):
        DAEProblem(t_span=(1.0, 5.0, 2.0))

    with pytest.raises(ValueError, match="t_end must be greater than t0!"):
        DAEProblem(t_span=(5.0, 2.0))

def test_values_are_correctly_by_default():
    from daelab.base.dae_problem import DAEProblem

    dae_problem = DAEProblem()

    assert dae_problem.name == "DAEProblem", "Name is not initialized by default!"
    assert dae_problem.t0 == None, "t0 is not initialized by default!"
    assert dae_problem.t_end == None, "t_end is not initialized by default!"
    assert dae_problem.t_span == None, "t_span is not initialized by default!"
