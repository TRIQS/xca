

import numpy as np


from triqs.gfs import MeshDLRImTime


from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

from triqs.utility import mpi
from triqs_xca.block_sparse_solver import BlockSparseSolver


from common import plot_comparison, test_convergence_rate, Dummy


def pyed_dimer_solution(mesh_tau, t=-1.0, e0=0.1):

    from triqs.operators import c, c_dag, n

    H = e0 * n('0', 0) + t * ( c_dag('0', 0) * c('0', 1) + c_dag('0', 1) * c('0', 0) )
    fundamental_operators = [c('0', 0), c('0', 1)]
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


def xca_dimer_solution(mesh_tau, t=-1.0, e0 = 0.1, sigma_order=1, spgf_order=None, verbose=False, conserved_operators=[]):

    if spgf_order is None: spgf_order = sigma_order

    m = mesh_tau

    from triqs.operators import n

    S = BlockSparseSolver(
        H_loc=e0 * n('0', 0), gf_struct=[['0', 1]],
        beta=m.beta, w_max=m.w_max, eps=m.eps,
        conserved_operators=conserved_operators)

    S.Delta_tau['0'].data[:] = -0.5 * t**2

    S.solve(max_order=sigma_order, spgf_max_order=spgf_order, 
            tol=1e-12, maxiter=20, verbose=True)

    S.Chi_tau = S.eval_one_time_correlator(
        S.G, max_order=spgf_order, ops_tau=[n('0', 0)], ops_0=[n('0', 0)])

    d = Dummy()
    d.S = S
    d.sigma_order = sigma_order
    d.spgf_order = spgf_order
    d.G_tau = S.G_tau['0'][0, 0].copy()
    d.Chi_tau = S.Chi_tau[0, 0].copy()
    return d


def xca_dimer_solution_dense(mesh_tau, t=-1.0, e0=0.1, sigma_order=1, spgf_order=None, verbose=False):
    return xca_dimer_solution(mesh_tau, t=t, e0=e0, sigma_order=sigma_order, spgf_order=spgf_order, verbose=verbose, conserved_operators=[])


def xca_dimer_solution_block_sparse(mesh_tau, t=-1.0, e0=0.1, sigma_order=1, spgf_order=None, verbose=False):
    return xca_dimer_solution(mesh_tau, t=t, e0=e0, sigma_order=sigma_order, spgf_order=spgf_order, verbose=verbose, conserved_operators='automatic')


if __name__ == "__main__":

    m_dlr = MeshDLRImTime(beta=2.3, statistic='Fermion', eps=1e-12, w_max=4.0)

    #plot_comparison(m_dlr, pyed_dimer_solution, xca_dimer_solution_dense, max_order=5)
    test_convergence_rate(m_dlr, pyed_dimer_solution, xca_dimer_solution_dense, label='dimer', max_order=5)
    #test_convergence_rate(m_dlr, pyed_dimer_solution, xca_dimer_solution_block_sparse, label='dimer', max_order=5)    