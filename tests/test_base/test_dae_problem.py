import pytest

from daelab.base.dae_problem import DAEProblem


def test_initialize_dae_problem_correctly():
    diff_index = 1
    t_span = (0.0, 1.0)
    name = "DAE"
    dae_problem = DAEProblem(diff_index=diff_index, t_span=t_span, name=name)

    assert dae_problem.diff_index == diff_index, "diff_index is not intialized correctly!"
    assert dae_problem.t0 == t_span[0], "t_span is not initialized correctly!"
    assert dae_problem.t_end == t_span[1], "t_span is not initialized correctly!"
    assert t_span == dae_problem.t_span, "t_span is not initialized correctly!"
    assert name == dae_problem.name, "name is not initialized correctly!"


def test_validate_nvars_accepts_valid_input():
    DAEProblem.validate_nvars((1, 12, 128))


def test_validate_nvars_rejects_invalid_input():
    with pytest.raises(ValueError, match="nvars must contain only positive integers."):
        DAEProblem.validate_nvars((0, 0, 0))


def test_validate_nvars_should_raise_type_error_if_nvars_is_not_a_tuple():
    with pytest.raises(TypeError, match="nvars must be a tuple."):
        DAEProblem.validate_nvars(nvars=42)


def test_validate_nvars_should_raise_value_error_if_nvars_is_empty_tuple():
    with pytest.raises(ValueError, match="nvars must not be empty."):
        DAEProblem.validate_nvars(nvars=())


@pytest.mark.parametrize(
    "nvars",
    [
        (2, 0, 128),
        (5, -1),
        (3, 256, -1),
    ],
)
def test_validate_nvars_should_raise_value_error_if_single_dimension_is_nonpositive_integer(nvars):
    with pytest.raises(ValueError, match="nvars must contain only positive integers."):
        DAEProblem.validate_nvars(nvars=nvars)


def test_errors_invalid_diff_index_is_raised_correctly():
    with pytest.raises(TypeError, match="diff_index must be an integer or None!"):
        DAEProblem(diff_index=True)

    with pytest.raises(TypeError, match="diff_index must be an integer or None!"):
        DAEProblem(diff_index=1.0)

    with pytest.raises(ValueError, match="diff_index must be equal to or greater than one!"):
        DAEProblem(diff_index=0)


def test_value_error_invalid_t_span_is_raised_correctly():
    with pytest.raises(ValueError, match="t_span is a tuple containing two values!"):
        DAEProblem(t_span=(1.0, 5.0, 2.0))

    with pytest.raises(ValueError, match="t_end must be greater than t0!"):
        DAEProblem(t_span=(5.0, 2.0))


def test_values_are_correctly_by_default():
    dae_problem = DAEProblem()

    assert dae_problem.diff_index == None, "diff_index is not initialized by default!"
    assert dae_problem.name == "DAEProblem", "Name is not initialized by default!"
    assert dae_problem.t0 == None, "t0 is not initialized by default!"
    assert dae_problem.t_end == None, "t_end is not initialized by default!"
    assert dae_problem.t_span == None, "t_span is not initialized by default!"
