import numpy as np
import pytest

from daelab.base.datatypes.node_list import NodeList
from daelab.base.datatypes.semiexplicit import SemiexplicitArray
from daelab.base.methods.sdc import SDCMethod
from daelab.methods.sdc_methods.constrained_sdc import ConstrainedSDC
from daelab.problems.linear_manufactured_dae import LinearManufacturedDAE
from tests.test_base.test_methods.test_sdc import DummyDAE


def test_init_passes_parameters_to_method():
    kwargs = dict(num_nodes=4, quad_type="RADAU-RIGHT", QD_type="IE")
    sdc_c, sdc = ConstrainedSDC(**kwargs), SDCMethod(**kwargs)

    assert isinstance(sdc_c, SDCMethod)
    assert sdc_c.num_nodes == 4
    assert np.array_equal(sdc_c.nodes, sdc.nodes)
    assert np.array_equal(sdc_c.Q, sdc.Q)
    assert np.array_equal(sdc_c.QD, sdc.QD)


@pytest.mark.parametrize("QD_type", ["IE", "EE"])
def test_compute_defect_correctly_computes_defect(QD_type, t=0.0, dt=0.5):
    problem = LinearManufacturedDAE()

    num_nodes = 3
    sdc_c = ConstrainedSDC(
        num_nodes=num_nodes, quad_type="RADAU-RIGHT", QD_type=QD_type
    )

    sdc_c.setup(problem=problem)

    tau = [t + dt * node for node in sdc_c.nodes]

    sdc_c.u[0] = problem.u_exact(t)
    sdc_c.f[0] = problem.f(sdc_c.u[0], t)

    # Initialization of provisional solution by spreading it to each node
    for m in range(1, sdc_c.num_nodes + 1):
        sdc_c.u[m] = problem.u_exact(tau[m - 1])
        sdc_c.f[m] = problem.f(sdc_c.u[m], tau[m - 1])

    Q, QD = sdc_c.Q, sdc_c.QD

    _u, f = sdc_c.u, sdc_c.f

    defect = sdc_c.compute_defect(dt)
    assert isinstance(defect, NodeList)
    for m in range(len(defect)):
        assert isinstance(defect[m], SemiexplicitArray)
        assert defect[m].diff.shape == problem.nvars_diff
        assert defect[m].alg.shape == problem.nvars_alg

    for m in range(sdc_c.num_nodes):
        expected = (
            dt * (Q[m, 0] - QD[m, 0]) * f[1].diff[:]
            + dt * (Q[m, 1] - QD[m, 1]) * f[2].diff[:]
            + dt * (Q[m, 2] - QD[m, 2]) * f[3].diff[:]
        )
        assert np.allclose(defect[m].diff, expected)
        assert np.allclose(defect[m].alg, 0.0)


@pytest.fixture
def problem_and_method(t=0.0, dt=0.1):
    problem = LinearManufacturedDAE()

    num_nodes = 5
    sdc_c = ConstrainedSDC(num_nodes=num_nodes, quad_type="RADAU-RIGHT", QD_type="IE")

    sdc_c.setup(problem=problem)

    return problem, sdc_c


def test_update_node_values_correctly_updates_values_at_nodes(
    problem_and_method, t=0.0, dt=0.1
):
    problem, sdc_c = problem_and_method

    sdc_c.u[0] = problem.u_exact(t)
    sdc_c.init_node_values(t, dt)

    for _ in range(2 * sdc_c.num_nodes):
        sdc_c.update_node_values(t, dt)

    u = sdc_c.u

    u_ex = problem.u_exact(t + dt)
    assert np.allclose(u[-1].diff[:], u_ex.diff[:], rtol=0.0, atol=3e-13)
    assert np.allclose(u[-1].alg[:], u_ex.alg[:], rtol=0.0, atol=3e-13)


def test_compute_residual_is_max_norm_over_nodes_and_components(t=0.0, dt=0.5):
    problem = DummyDAE()

    num_nodes = 3
    sdc_c = ConstrainedSDC(num_nodes=num_nodes, quad_type="RADAU-RIGHT", QD_type="IE")

    sdc_c.setup(problem=problem)

    tau = [t + dt * node for node in sdc_c.nodes]

    sdc_c.u[0] = problem.u_exact(t)
    sdc_c.f[0] = problem.f(sdc_c.u[0], t)

    # Initialization of provisional solution by spreading it to each node
    for m in range(1, sdc_c.num_nodes + 1):
        sdc_c.u[m] = problem.u_exact(tau[m - 1])
        sdc_c.f[m] = problem.f(sdc_c.u[m], tau[m - 1])

    residual = sdc_c.compute_residual(dt)

    u0 = problem.u_exact(t)

    expected_res_norm = []
    expected_res = NodeList(problem.dtype_u, problem.init, sdc_c.num_nodes - 1)
    for m in range(sdc_c.num_nodes):
        expected_res[m].diff[:] += u0.diff[:] - sdc_c.u[m + 1].diff[:]
        for j in range(sdc_c.num_nodes):
            expected_res[m].diff[:] += dt * sdc_c.Q[m, j] * sdc_c.f[j + 1].diff[:]

        expected_res_norm.append(abs(expected_res[m].diff[:]))

    expected_residual = max(expected_res_norm)

    assert np.isclose(residual, expected_residual, rtol=0.0, atol=1e-14)
    for m in range(sdc_c.num_nodes):
        assert np.isclose(expected_res[m].alg, 0.0, rtol=0.0, atol=1e-14)
