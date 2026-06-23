

import numpy as np


from triqs.gfs import MeshDLRImTime


from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

from triqs.utility import mpi
from triqs_xca.block_sparse_solver import BlockSparseSolver


from common import plot_comparison, test_convergence_rate, Dummy


def pyed_two_level_solution(mesh_tau, t=-1.0, e0=0.0, e1=0.0, U=2.0):

    from triqs.operators import c, c_dag, n

    e0 -= U/2
    e1 -= U/2

    H = e0 * n('0', 0) + e1 * n('0', 1) + U * n('0', 0) * n('0', 1) + \
        t * ( c_dag('0', 0) * c('0', 2) + c_dag('0', 2) * c('0', 0) ) + \
        t * ( c_dag('0', 1) * c('0', 3) + c_dag('0', 3) * c('0', 1) )
        
    fundamental_operators = [c('0', 0), c('0', 1), c('0', 2), c('0', 3)]
    ed = TriqsExactDiagonalization(H, fundamental_operators, mesh_tau.beta)

    from triqs.gfs import Gf
    G_tau = Gf(mesh=mesh_tau, target_shape=[])
    ed.set_g2_tau(G_tau, c('0', 0), c_dag('0', 0))

    Chi_tau = Gf(mesh=mesh_tau, target_shape=[])
    ed.set_g2_tau(Chi_tau, n('0', 0), n('0', 0))

    d = Dummy()
    d.G_tau = G_tau
    d.Chi_tau = Chi_tau
    return d


def xca_two_level_solution(mesh_tau, t=-1.0, e0=0.0, e1=0.0, U=2.0, sigma_order=1, spgf_order=None, verbose=False):

    if spgf_order is None: spgf_order = sigma_order

    m = mesh_tau

    from triqs.operators import n, c, c_dag

    e0 -= U/2
    e1 -= U/2

    S = BlockSparseSolver(
        H_loc=e0 * n('0', 0) + e1 * n('0', 1) + U * n('0', 0) * n('0', 1), 
        gf_struct=[['0', 2]],
        beta=m.beta, w_max=m.w_max, eps=m.eps,
        conserved_operators=[],
        )

    S.Delta_tau['0'].data[:, 0, 0] = -0.5 * t**2
    S.Delta_tau['0'].data[:, 1, 1] = -0.5 * t**2

    S.solve(max_order=sigma_order, spgf_max_order=spgf_order, 
            tol=1e-12, maxiter=20, verbose=True)

    S.G_tau_ref = S.eval_one_time_correlator(
        S.G, max_order=spgf_order, ops_tau=[c('0', 0)], ops_0=[c_dag('0', 0)])

    np.testing.assert_array_almost_equal(S.G_tau['0'][0, 0].data, S.G_tau_ref[0, 0].data)

    S.Chi_tau = S.eval_one_time_correlator(
        S.G, max_order=spgf_order, ops_tau=[n('0', 0)], ops_0=[n('0', 0)])

    d = Dummy()
    d.S = S
    d.sigma_order = sigma_order
    d.spgf_order = spgf_order
    d.G_tau = S.G_tau['0'][0, 0].copy()
    d.Chi_tau = S.Chi_tau[0, 0].copy()
    return d


if __name__ == "__main__":

    m_dlr = MeshDLRImTime(beta=2.3, statistic='Fermion', eps=1e-12, w_max=8.0)

    #plot_comparison(m_dlr, pyed_two_level_solution, xca_two_level_solution, max_order=3)
    test_convergence_rate(m_dlr, pyed_two_level_solution, xca_two_level_solution, label='two_level', max_order=4)