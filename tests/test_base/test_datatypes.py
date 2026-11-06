import numpy as np
import operator
import pytest

from daelab.base.datatypes import (
    ArithmeticMethods,
    Solution,
    PartitionedContainer,
    SemiexplicitSolution,
    RHS,
    IMEXRHS,
    SemiexplicitRHS,
)


class DummyMethod(ArithmeticMethods):
    def _binary(self, other, op):
        return op, other


def test_add_calls_binary_with_add():
    assert DummyMethod() + 2 == (operator.add, 2)


def test_sub_calls_binary_with_sub():
    assert DummyMethod() - 5 == (operator.sub, 5)


def test_mul_calls_binary_with_mul():
    assert 2.5 * DummyMethod() == (operator.mul, 2.5)


def test_truediv_calls_binary_with_truediv():
    assert DummyMethod() / 4 == (operator.truediv, 4)


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


def test_making_a_copy_gives_copied_instance_with_equal_attributes_to_original():
    num_nodes = 3
    nvars = (128, 128, 256)
    dtype = float
    u = Solution(num_nodes=num_nodes, nvars=nvars, dtype=dtype)

    u_copy = u.copy()
    assert type(u_copy) == Solution
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
        expected = sum(
            Q[m, k] * u._comps[k] for k in range(u.num_nodes)
        )

        assert np.array_equal(res[m], expected)

    assert type(res) == Solution


def test_solution_binary_should_raise_value_error_if_num_nodes_are_different():
    u1 = Solution(num_nodes=3, nvars=(2, 2))
    other = Solution(num_nodes=4, nvars=(2, 2))

    with pytest.raises(ValueError, match="Operands must have the same num_nodes and nvars!"):
        u1._binary(other, np.add)


def test_solution_binary_should_raise_type_error_if_nvars_are_different():
    u1 = Solution(num_nodes=3, nvars=(2, 2))
    other = Solution(num_nodes=3, nvars=(4, 64))

    with pytest.raises(ValueError, match="Operands must have the same num_nodes and nvars!"):
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


class DummyContainer(PartitionedContainer):
    _components = ("comp1", "comp2")

    def __init__(self):
        self.comp1 = np.arange(6).reshape(3, 2)
        self.comp2 = np.arange(3).reshape(3, 1)
        self.num_nodes = 3


class OtherContainer(PartitionedContainer):
    _components = ("u", "z")


def test_parts_return_all_components():
    con = DummyContainer()

    assert con._parts == [con.comp1, con.comp2]


def test_getitem_returns_parts_at_nodes():
    con = DummyContainer()

    m = 1
    assert np.array_equal(con[m][0], con.comp1[m])
    assert np.array_equal(con[m][1], con.comp2[m])


def test_setitem_sets_all_parts():
    con = DummyContainer()

    con[1] = (np.array([10, 11]), np.array([20]))

    assert np.array_equal(con.comp1[1], [10, 11])
    assert np.array_equal(con.comp2[1], [20])


def test_setitem_should_raise_value_error_for_wrong_number_of_values():
    con = DummyContainer()

    with pytest.raises(ValueError, match="Expected 2 values"):
        con[1] = (np.array([10, 11]),)


def test_copy_creates_independent_container():
    con = DummyContainer()

    con_copy = con.copy()

    assert con_copy is not con
    assert con_copy.comp1 is not con.comp1
    assert con_copy.comp1 is not con.comp2

    assert np.array_equal(con_copy.comp1, con.comp1)
    assert np.array_equal(con_copy.comp2, con.comp2)

def test_binary_adds_two_containers():
    con1 = DummyContainer()
    con2 = DummyContainer()

    sum = con1._binary(con2, np.add)

    assert np.array_equal(sum.comp1, con1.comp1 + con2.comp1)
    assert np.array_equal(sum.comp2, con1.comp2 + con2.comp2)


def test_binary_adds_container_and_scalar():
    con = DummyContainer()

    sum = con._binary(6, np.add)

    assert np.array_equal(sum.comp1, con.comp1 + 6)
    assert np.array_equal(sum.comp2, con.comp2 + 6)


def test_binary_should_raise_value_error_if_components_are_different():
    con1 = DummyContainer()
    other_con = OtherContainer()

    other_con.u = np.zeros((3, 1))
    other_con.z = np.zeros((2, 2))

    with pytest.raises(ValueError, match="Operands must have the same parts and num_nodes."):
        con1._binary(other_con, np.add)


def test_binary_should_raise_value_error_if_num_nodes_are_different():
    con1 = DummyContainer()
    other_con = OtherContainer()

    other_con.num_nodes = 4

    with pytest.raises(ValueError, match="Operands must have the same parts and num_nodes."):
        con1._binary(other_con, np.add)


def test_binary_should_raise_type_error_if_datatypes_are_different():
    con = DummyContainer()
    u = Solution(num_nodes=3, nvars=(1,))

    with pytest.raises(TypeError, match=f"Cannot combine DummyContainer with a Solution."):
        con._binary(u, np.add)

# Wieso ist das Ergebnis nicht gleich Q @ u?
def test_apply_matrix_is_applied_to_all_components():
    u = SemiexplicitSolution(num_nodes=2, nvars_diff=(2, 2), nvars_alg=(2, 2))

    Q = np.array([[1, 0], [2, 0]])

    for m in range(u.num_nodes):
        u.diff[m] = (m + 3) * np.ones(u.nvars_diff)
        u.alg[m] = (m + 5) * np.ones(u.nvars_alg)

    
    res = u.apply_matrix(Q)

    for m in range(u.num_nodes):
        expected_diff = sum(
            Q[m, k] * u.diff[k] for k in range(u.num_nodes)
        )
        expected_alg = sum(
            Q[m, k] * u.alg[k] for k in range(u.num_nodes)
        )

        assert np.array_equal(res.diff[m], expected_diff)
        assert np.array_equal(res.alg[m], expected_alg)


def test_function_repr_returns_correct_output():
    num_nodes = 3
    nvars = (128, 128, 256)
    dtype = float
    u = SemiexplicitSolution(num_nodes=num_nodes, nvars_diff=nvars, nvars_alg=nvars, dtype=dtype)

    assert str(u) == f"SemiexplicitSolution({num_nodes=}, nvars_diff={nvars}, nvars_alg={nvars}, dtype={np.dtype(dtype)})"


def test_RHS_class_is_of_type_Solution():
    rhs = RHS(num_nodes=3, nvars=(128, 128))
    assert isinstance(rhs, Solution)


def test_vars_are_assigned_correctly_in_IMEXRHS():
    num_nodes = 4
    nvars = (128, 128)
    dtype = float
    imex_rhs = IMEXRHS(num_nodes=num_nodes, nvars=nvars, dtype=dtype)

    assert imex_rhs.num_nodes == num_nodes
    assert imex_rhs.nvars == nvars
    assert type(imex_rhs.impl) == RHS
    assert type(imex_rhs.expl) == RHS


def test_full_rhs_should_return_the_complete_rhs():
    num_nodes = 4
    nvars = (128, 128)
    dtype = float
    imex_rhs = IMEXRHS(num_nodes=num_nodes, nvars=nvars, dtype=dtype)

    imex_rhs.impl[1] = 2 * np.ones(nvars)
    imex_rhs.expl[1] = 3 * np.ones(nvars)

    imex_rhs.impl[2] = 1.67 * np.ones(nvars)
    imex_rhs.expl[2] = 5 * np.ones(nvars)

    assert np.array_equal(imex_rhs.full_rhs[1], imex_rhs.impl[1] + imex_rhs.expl[1])
    assert np.array_equal(imex_rhs.full_rhs[2], imex_rhs.impl[2] + imex_rhs.expl[2])


def test_function_repr_returns_correct_output_IMEXRHS():
    num_nodes = 4
    nvars = (128, 128)
    dtype = float
    imex_rhs = IMEXRHS(num_nodes=num_nodes, nvars=nvars, dtype=dtype)
    
    assert str(imex_rhs) == f"IMEXRHS({num_nodes=}, {nvars=}, dtype={np.dtype(dtype)})"


def test_vars_are_assigned_correctly_in_SemiexplicitRHS():
    num_nodes = 4
    nvars = (128, 128)
    dtype = float
    imex_diff = False
    semiexplicit_rhs = SemiexplicitRHS(
        num_nodes=num_nodes,
        nvars_diff=nvars,
        nvars_alg=nvars,
        dtype=dtype,
        imex_diff=imex_diff,
    )

    assert semiexplicit_rhs.num_nodes == num_nodes
    assert semiexplicit_rhs.nvars_diff == nvars
    assert semiexplicit_rhs.nvars_alg == nvars
    assert semiexplicit_rhs.dtype == dtype
    assert semiexplicit_rhs.imex_diff == imex_diff
    assert type(semiexplicit_rhs.diff) == RHS
    assert type(semiexplicit_rhs.alg) == RHS


def test_SemiExplicitRHS_can_use_imex_splitting():
    num_nodes = 4
    nvars = (128, 128)
    dtype = float
    imex_diff = True
    se_imex_rhs = SemiexplicitRHS(
        num_nodes=num_nodes,
        nvars_diff=nvars,
        nvars_alg=nvars,
        dtype=dtype,
        imex_diff=imex_diff,
    )

    assert type(se_imex_rhs.diff) == IMEXRHS
    assert type(se_imex_rhs.alg) == RHS

    se_imex_rhs.diff.impl[1] = 2.5 * np.ones(nvars)
    se_imex_rhs.diff.expl[1] = 6 * np.ones(nvars)

    se_imex_rhs.diff.impl[2] = 0.25 * np.ones(nvars)
    se_imex_rhs.diff.expl[2] = 0.33 * np.ones(nvars)

    assert np.array_equal(
        se_imex_rhs.diff.full_rhs[1], se_imex_rhs.diff.impl[1] + se_imex_rhs.diff.expl[1]
    )
    assert np.array_equal(
        se_imex_rhs.diff.full_rhs[2], se_imex_rhs.diff.impl[2] + se_imex_rhs.diff.expl[2]
    )





