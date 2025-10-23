""" Author: Hugo U. R. Strand (2025) """

import numpy as np
from mpi4py import MPI

from triqs.operators import c, c_dag

from triqs_xca.solver import Solver

from triqs_xca.dense import NCA_dense, OCA_dense


def test_block_sparsity_NCA_dense(verbose=True):
    
    beta = 10.0
    t = 0.5
    ek = 0.1
    mu = 1/3

    H = -mu * ( c_dag(0,0) * c(0,0) + c_dag(0,1) * c(0,1) )
    fundamental_operators = [ c(0,i) for i in range(2) ]

    S = Solver(beta=beta, lamb=100., eps=1e-10, H_loc=H, fundamental_operators=fundamental_operators)

    delta_iaa = t**2 * S.fd.free_greens(beta, np.diag([ek, ek]))
    S.set_hybridization(delta_iaa, compress=False)

    Sigma_iaa_NCA_ref = S.calc_Sigma(max_order=1)
    Sigma_iaa_OCA_ref = S.calc_Sigma(max_order=2)
    Sigma_iaa_OCA_ref -= Sigma_iaa_NCA_ref

    # -- Solve using block-sparsity in dense formulation
    
    F     = np.array(S.F,     dtype=complex)
    F_dag = np.array(S.F_dag, dtype=complex)
    
    delta_iaa_refl = S.ito.reflect(delta_iaa)

    print('--> NCA_dense')
    Sigma_iaa_NCA = NCA_dense(delta_iaa, delta_iaa_refl, S.G0_iaa, F, F_dag)
    print('--> OCA_dense')
    Sigma_iaa_OCA = OCA_dense(delta_iaa, S.ito, beta, S.G0_iaa, F, F_dag)
    print('--> done.')

    diff_NCA = np.max(np.abs(Sigma_iaa_NCA - Sigma_iaa_NCA_ref))
    print(f'diff_NCA = {diff_NCA:2.2E}')

    diff_OCA = np.max(np.abs(Sigma_iaa_OCA - Sigma_iaa_OCA_ref))
    print(f'diff_OCA = {diff_OCA:2.2E}')
    
    if verbose:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 8))
        subp = [3, 1, 1]

        plt.subplot(*subp); subp[-1] += 1
        
        plt.plot(S.tau_i, delta_iaa[:, 0, 0].flatten().real, 'o-')
        plt.ylabel(r'$\Delta(\tau)$')
        plt.xlabel(r'$\tau$')

        plt.subplot(*subp); subp[-1] += 1

        for i in range(4):
            plt.plot(S.tau_i, Sigma_iaa_NCA_ref[:, i, i].real, '+-')
            plt.plot(S.tau_i, Sigma_iaa_NCA[:, i, i].real, 'x-')

        plt.ylabel(r'$\Sigma_{ii}^{(NCA)}(\tau)$')
        plt.xlabel(r'$\tau$')

        plt.plot([], [], '+-', color='gray', label='Z. Huang')
        plt.plot([], [], 'x-', color='gray', label='NCA_dense')    
        plt.legend(loc='best')

        plt.subplot(*subp); subp[-1] += 1

        for i in range(4):
            plt.plot(S.tau_i, Sigma_iaa_OCA_ref[:, i, i].real, '+-')
            plt.plot(S.tau_i, Sigma_iaa_OCA[:, i, i].real, 'x-')

        plt.ylabel(r'$\Sigma_{ii}^{(OCA)}(\tau)$')
        plt.xlabel(r'$\tau$')

        plt.plot([], [], '+-', color='gray', label='Z. Huang')
        plt.plot([], [], 'x-', color='gray', label='OCA_dense')
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.show()


    np.testing.assert_array_almost_equal(Sigma_iaa_NCA, Sigma_iaa_NCA_ref)
    np.testing.assert_array_almost_equal(Sigma_iaa_OCA, Sigma_iaa_OCA_ref)

        
if __name__ == "__main__":
        
    test_block_sparsity_NCA_dense(verbose=False)
