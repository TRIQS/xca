"""

Test the one-time correlator computation by comparison with 
the single particle Green's function evaluator.

"""

import numpy as np


from triqs.operators import n, c, c_dag
from triqs.gfs import make_gf_dlr_imfreq, make_gf_dlr_imtime, SemiCircular, make_gf_imtime


from triqs_xca.block_sparse_solver import BlockSparseSolver


def test_one_time_correlator(verbose, conserved_operators):
    
    # -- Parameters
    
    U = 3.0
    B = 1.0
    mu = U/2 + 0.4
    soc = 0.1 - 0.2j

    beta = 2.0
    eps = 1e-10
    w_max = 6.0

    # -- Local Hamiltonian
    
    gf_struct = [['0', 2]]

    N_tot = n('0', 0) + n('0', 1)
    
    H_loc = -mu * N_tot + U * n('0', 0) * n('0', 1) + B * (n('0', 0) - n('0', 1)) \
        + soc * c_dag('0', 0) * c('0', 1) + np.conj(soc) *c_dag('0', 1) * c('0', 0)

    S = BlockSparseSolver(H_loc, beta, w_max, eps, gf_struct=gf_struct,
        conserved_operators=conserved_operators, # Triggers no symmetries using DenseDiagramEvaluator
        )

    Delta_w = make_gf_dlr_imfreq(S.Delta_tau['0'])
    Delta_w << SemiCircular(1.0)
    S.Delta_tau['0'] << make_gf_dlr_imtime(Delta_w)

    max_order = 1

    S.solve(max_order=max_order, tol=1e-8, verbose=True)

    G_tau = S.G_tau['0']

    G_tau_ref = S.eval_one_time_correlator(
        S.G, max_order=max_order, 
        ops_tau=[c('0', 0), c('0', 1)], 
        ops_0=[c_dag('0', 0), c_dag('0', 1)])
    
    np.testing.assert_array_almost_equal(G_tau.data, G_tau_ref.data)

    if verbose:
        from triqs.plot.mpl_interface import oplot, plt
        oplot(make_gf_imtime(G_tau, n_tau=400).real, '-', label='G_tau')
        oplot(make_gf_imtime(G_tau_ref, n_tau=400).real, '--', label='one-time correlator')
        plt.show()


if __name__ == '__main__':
    verbose = False
    test_one_time_correlator(verbose=verbose, conserved_operators=[]) # Test DenseDiagramEvaluator
    test_one_time_correlator(verbose=verbose, conserved_operators='automatic') # Test BlockSparseDiagramEvaluator
