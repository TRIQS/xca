

import numpy as np


from triqs.gfs import MeshDLRImTime


from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

from triqs.utility import mpi
from triqs_xca.block_sparse_solver import BlockSparseSolver


from common import plot_comparison, test_convergence_rate, Dummy


def analytic_solution(mesh_tau, t=-1.0, e0=0.1):

    from triqs.gfs import Gf, make_gf_dlr_imfreq, inverse, iOmega_n, SemiCircular, make_gf_dlr_imtime, make_gf_dlr
    G_tau = Gf(mesh=mesh_tau, target_shape=[])
    G_iw = make_gf_dlr_imfreq(G_tau)
    Delta_iw = G_iw.copy()

    Delta_iw << t**2 * inverse(iOmega_n)

    G_iw << inverse(iOmega_n - e0 - Delta_iw)
    G_tau = make_gf_dlr_imtime(G_iw)

    Chi_tau = Gf(mesh=mesh_tau, target_shape=[])

    G_dlr = make_gf_dlr(G_tau)
    beta = mesh_tau.beta
    n_exp = -G_dlr(beta)
    for tau in mesh_tau:
        Chi_tau[tau] = -G_tau[tau] * G_dlr(beta-tau) - n_exp**2

    d = Dummy()
    d.G_tau = G_tau
    d.Chi_tau = Chi_tau
    return d


def xca_two_level_solution(
        mesh_tau, t=-1.0, e0=0.1, sigma_order=1, 
        spgf_order=None, verbose=False, conserved_operators=[]):

    if spgf_order is None: spgf_order = sigma_order

    m = mesh_tau

    from triqs.operators import n, c, c_dag

    S = BlockSparseSolver(
        H_loc=e0 * (n('0', 0) + n('0', 1)), 
        gf_struct=[['0', 2]],
        beta=m.beta, w_max=m.w_max, eps=m.eps,
        conserved_operators=conserved_operators,
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


def xca_two_level_solution_dense(
        mesh_tau, t=-1.0, e0=0.1, sigma_order=1, 
        spgf_order=None, verbose=False):

    return xca_two_level_solution(
        mesh_tau, t=t, e0=e0, sigma_order=sigma_order,
        spgf_order=spgf_order, verbose=verbose, conserved_operators=[])


def xca_two_level_solution_block_sparse(
        mesh_tau, t=-1.0, e0=0.1, sigma_order=1, 
        spgf_order=None, verbose=False):

    return xca_two_level_solution(
        mesh_tau, t=t, e0=e0, sigma_order=sigma_order,
        spgf_order=spgf_order, verbose=verbose, conserved_operators='automatic')


if __name__ == "__main__":

    m_dlr = MeshDLRImTime(beta=2.3, statistic='Fermion', eps=1e-12, w_max=8.0)

    #plot_comparison(m_dlr, analytic_two_level_solution, xca_two_level_solution_dense, max_order=3)
    #test_convergence_rate(m_dlr, analytic_solution, xca_two_level_solution_dense, label='two_level', max_order=3, do_test=True)
    test_convergence_rate(m_dlr, analytic_solution, xca_two_level_solution_block_sparse, label='two_level', max_order=4, do_test=True)