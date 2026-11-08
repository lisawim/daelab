import numpy as np
import operator
import pytest

from daelab.base.datatypes.solution import ArithmeticMethods, Solution


class DummyMethod(ArithmeticMethods):
    def _binary(self, other, op):
        return op, other


def test_add_calls_binary_with_add():
    assert DummyMethod() + 2 == (operator.add, 2)


def test_radd_calls_binary_with_add():
    assert 3 + DummyMethod() == (operator.add, 3)


def test_sub_calls_binary_with_sub():
    assert DummyMethod() - 5 == (operator.sub, 5)


def test_mul_calls_binary_with_mul():
    assert DummyMethod() * 2.5 == (operator.mul, 2.5)


def test_rmul_calls_binary_with_mul():
    assert 2.5 * DummyMethod() == (operator.mul, 2.5)


def test_truediv_calls_binary_with_truediv():
    assert DummyMethod() / 4 == (operator.truediv, 4)


def test_neg_multiplies_with_minus_one():
    assert -DummyMethod() == (operator.mul, -1)


def test_vars_are_assigned_correctly_in_solution():
    num_nodes = 4
    nvars = (1, 64, 64)
    u = Solution(num_nodes=num_nodes, nvars=nvars)

    assert u.num_nodes == num_nodes
    assert u.nvars == nvars
    assert len(u._comps) == num_nodes
    for val in u._comps:
        assert val.shape == nvars

    assert isinstance(u, ArithmeticMethods)


def test_nvars_integer_is_converted_into_tuple():
    num_nodes = 4
    nvars = 2
    u = Solution(num_nodes=num_nodes, nvars=nvars)

    assert isinstance(u.nvars, tuple)


@pytest.mark.parametrize("num_nodes", [2, 3, 4, 5])
def test_len_method_should_return_num_nodes(num_nodes):
    nvars = (1, 64, 64)
    u = Solution(num_nodes=num_nodes, nvars=nvars)

    assert len(u) == num_nodes


@pytest.mark.parametrize(
    "nvars",
    [(1, 2, 4), (32, 64), (512, 128, 128)],
)
def test_setitem_writes_correct_values_into_node(nvars):
    num_nodes = 4
    nvars = nvars
    u = Solution(num_nodes=num_nodes, nvars=nvars)
    u[1] = 4 * np.ones(nvars)
    u[2] = 3.14 * np.ones(nvars)

    assert np.array_equal(u[1], 4 * np.ones(nvars))
    assert np.array_equal(u[2], 3.14 * np.ones(nvars))


@pytest.mark.parametrize(
    "nvars",
    [(1, 2, 4), (32, 64), (512, 128, 128)],
)
def test_getitem_returns_array_of_shape_nvars(nvars):
    num_nodes = 4
    u = Solution(num_nodes=num_nodes, nvars=nvars)

    assert u[1].shape == nvars


def test_function_repr_returns_correct_output():
    num_nodes = 3
    nvars = (128, 128, 256)
    dtype = float
    u = Solution(num_nodes=num_nodes, nvars=nvars, dtype=dtype)

    assert str(u) == f"Solution({num_nodes=}, {nvars=}, dtype={np.dtype(dtype)})"


@pytest.mark.parametrize("num_nodes", [2, 3, 4])
@pytest.mark.parametrize("nvars", [(1,), (3,), (2, 4), (3, 4, 5)])
def test_flat_node_is_flattened_view_of_node(num_nodes, nvars):
    u = Solution(num_nodes=num_nodes, nvars=nvars, dtype=float)
    rng = np.random.default_rng(0)
    for m in range(num_nodes):
        u[m] = rng.random(nvars)

    for m in range(num_nodes):
        flat = u.flat_node(m)

        assert flat.shape == (np.prod(nvars),)
        assert np.array_equal(flat, u[m].reshape(-1))
        assert np.array_equal(flat.reshape(nvars), u[m])
        assert np.shares_memory(flat, u[m])


@pytest.mark.parametrize("num_nodes", [2, 3, 4])
@pytest.mark.parametrize("nvars", [(1,), (3,), (2, 4), (3, 4, 5)])
def test_as_flat_has_one_flattened_row_per_node(num_nodes, nvars):
    u = Solution(num_nodes=num_nodes, nvars=nvars, dtype=float)
    rng = np.random.default_rng(0)
    for m in range(num_nodes):
        u[m] = rng.random(nvars)

    flat = u.as_flat()

    assert flat.shape == (num_nodes, np.prod(nvars))
    for m in range(num_nodes):
        assert np.array_equal(flat[m], u[m].reshape(-1))
        assert np.array_equal(flat[m].reshape(nvars), u[m])


def test_as_flat_returns_copy():
    num_nodes = 3
    nvars = (16, 16)
    u = Solution(num_nodes=num_nodes, nvars=nvars)
    u[:] = 1.0

    flat = u.as_flat()
    flat[:] = 5.0

    assert all(np.all(u[m] == 1.0) for m in range(3))
    assert not np.shares_memory(flat, u[0])


def test_making_a_copy_gives_copied_instance_with_equal_attributes_to_original():
    num_nodes = 3
    nvars = (128, 128, 256)
    dtype = float
    u = Solution(num_nodes=num_nodes, nvars=nvars, dtype=dtype)

    u_copy = u.copy()
    assert isinstance(u_copy, Solution)
    assert u_copy.num_nodes == u.num_nodes
    assert u_copy.nvars == u.nvars
    assert u_copy.dtype == u.dtype
    assert len(u_copy._comps) == len(u._comps)


def test_apply_matrix_is_applied_to_values_at_all_nodes():
    num_nodes = 3
    nvars = (3, 3)
    dtype = float
    u = Solution(num_nodes=num_nodes, nvars=nvars, dtype=dtype)

    Q = np.array([[1, 0, 5], [2, 0, 3], [1, 1, 1]])

    for m in range(u.num_nodes):
        u[m] = (m + 2) * np.ones(nvars)

    res = u.apply_matrix(Q)

    for m in range(u.num_nodes):
        expected = sum(Q[m, k] * u._comps[k] for k in range(u.num_nodes))

        assert np.array_equal(res[m], expected)

    assert isinstance(res, Solution)


def test_solution_binary_should_raise_value_error_if_num_nodes_are_different():
    u1 = Solution(num_nodes=3, nvars=(2, 2))
    other = Solution(num_nodes=4, nvars=(2, 2))

    with pytest.raises(
        ValueError, match="Operands must have the same num_nodes and nvars!"
    ):
        u1._binary(other, np.add)


def test_solution_binary_should_raise_type_error_if_nvars_are_different():
    u1 = Solution(num_nodes=3, nvars=(2, 2))
    other = Solution(num_nodes=3, nvars=(4, 64))

    with pytest.raises(
        ValueError, match="Operands must have the same num_nodes and nvars!"
    ):
        u1._binary(other, np.add)


def test_binary_adds_two_solutions():
    num_nodes = 3
    nvars = (2, 2)
    u1 = Solution(num_nodes=num_nodes, nvars=nvars)
    u2 = Solution(num_nodes=num_nodes, nvars=nvars)

    for m in range(num_nodes):
        u1[m] = (m + 2) * np.ones(nvars)
        u2[m] = (m + 4) * np.ones(nvars)

    sum = u1._binary(u2, np.add)

    for m in range(num_nodes):
        assert np.array_equal(sum[m], u1[m] + u2[m])


def test_binary_adds_solution_and_scalar():
    num_nodes = 3
    nvars = (2, 2)
    u = Solution(num_nodes=num_nodes, nvars=nvars)

    for m in range(num_nodes):
        u[m] = (m + 2) * np.ones(nvars)

    sum = u._binary(6, np.add)

    for m in range(num_nodes):
        assert np.array_equal(sum[m], u[m] + 6)
