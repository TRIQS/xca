"""
Test asymptotic convergence of the single particle Green's function
and the density-density correlation function for a non-interacting 
n-level AIM with discrete bath.

The convergence is tested for both the dense and block-sparse diagram evaluators.

Author: Hugo U. R. Strand (2026) 
"""

import numpy as np


from triqs.gfs import MeshDLRImTime


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


def xca_n_level_solution(
        mesh_tau, N=2,t=-1.0, e0=0.1, sigma_order=1, 
        spgf_order=None, verbose=False, conserved_operators=[]):

    if spgf_order is None: spgf_order = sigma_order

    m = mesh_tau

    from triqs.operators import n, c, c_dag

    H_loc = e0 * sum(n('0', i) for i in range(N))

    S = BlockSparseSolver(
        H_loc=H_loc, gf_struct=[['0', N]],
        beta=m.beta, w_max=m.w_max, eps=m.eps,
        conserved_operators=conserved_operators,
        )

    for idx in range(N):
        S.Delta_tau['0'].data[:, idx, idx] = -0.5 * t**2

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

    m_dlr = MeshDLRImTime(beta=2.3, statistic='Fermion', eps=1e-12, w_max=8.0, symmetrize=False)

    #plot_comparison(m_dlr, analytic_two_level_solution, xca_two_level_solution_dense, max_order=3)

    for N in range(1, 2+1):
        for conserved_operators in [[], 'automatic']:

            xca_nlvl = lambda mesh_tau, t, sigma_order, verbose : \
                xca_n_level_solution(mesh_tau, N=N, t=t, sigma_order=sigma_order, conserved_operators=conserved_operators, verbose=verbose)

            label = f'{N}_level'

            if conserved_operators == 'automatic':
                label += '_block_sparse'
            else:
                label += '_dense'

            test_convergence_rate(m_dlr, analytic_solution, xca_nlvl, 
                                  label=label, max_order=3, do_test=True, verbose=False)
