################################################################################
#
# triqs_xca: Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2025 by H. U.R. Strand
#
# triqs_xca is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# triqs_xca is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# triqs_xca. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################


r""" Test diagram evaluation for simple case that can be treated analytically.

Letting the local Hamiltonian being H_loc = 0 gives a degenerate atomic
pseudo-particle propagator

G(\tau) = - exp(-alpha * tau)

with alpha = log(N) / beta where N is the size of the local Hilbert space.

Since G is a simple exponential it factorizes out of all self-energy and
single particle diagrams.

The hybridization is set to Delta(\tau) = -0.5

(corresponding to a bath level at zero energy)

In this case the diagram integrations become simple polynomials in tau

e.g. the OCA diagram has the form

\Sigma_{OCA} = \sum_{abcd} \int_{\tau}^\beta d\tau_1 \int_0^{\tau_1} d\tau_2
  \Delta_{ab}(\tau - \tau_2) \Delta_{cd}(\tau_1 - 0) *
  [ O_a G(\tau - \tau_1) O_c G(\tau_1 - \tau_2) O_b G(\tau_2 - 0) O_d ]

which simplifies to
 
\Sigma_{OCA} = (-1/2)^2 G(\tau) ( \sum_{abcd} [O_a O_c O_b O_d] )
    \int_{\tau}^\beta d\tau_1 \int_0^{\tau_1} d\tau_2

where the integrals produce the polynomial

\int_{\tau}^\beta d\tau_1 \int_0^{\tau_1} d\tau_2 = \tau^2 / 2

The sum over operators give a combinatorical factor

A = \sum_{abcd} [O_a O_c O_b O_d] = (n - 1) n

that depends on the number of orbitals n.

So in total the OCA self-energy has the form

\Sigma_{OCA} = (-1/2)^2 G(\tau) n (n - 1) \tau^2 / 2

Below this analysis is performed also for the single particle Green's function
diagrams up to third order.

"""


import numpy as np

import triqs.utility.mpi as mpi
from triqs.gf import make_gf_dlr_imtime, make_gf_dlr_imfreq, inverse, iOmega_n
from triqs.operators import c, c_dag, Operator

from triqs_xca.solver import Solver, is_root
from triqs_xca.triqs_solver import TriqsSolver


class AnalyticDiagrams:

    def __init__(self, n, N, tau, beta):

        print(f'n = {n}, N = {N}')

        I_n = np.eye(n)[None, :, :] 
        I_N = np.eye(N)[None, :, :]

        tau = tau[:, None, None]

        alpha = np.log(N) / beta    
        self.G0_iaa = -np.exp(-alpha * tau) * I_N

        self.Sigma_iaa_nca = -1 * (-0.5) * self.G0_iaa * n

        self.dSigma_iaa_oca = (-0.5)**2 * (tau**2/2) * self.G0_iaa * n * (n - 1)    
        self.dSigma_iaa_tca = (-0.5)**3 * tau**4/24 * self.G0_iaa * (-self.__func_K(n))

        self.Sigma_iaa_oca = self.Sigma_iaa_nca + self.dSigma_iaa_oca
        self.Sigma_iaa_tca = self.Sigma_iaa_oca + self.dSigma_iaa_tca

        self.g_iaa_nca = -0.5 * I_n + 0 * tau

        self.dg_iaa_oca = - 0.5 * (beta - tau) * (tau - 0) * (n - 1) / 2 * I_n

        C_1 = 0.5 * self.__func_C_1(n)
        C_2 = 0.5 * self.__func_C_2(n)
        C_3 = C_2

        g_tca_contrib_1 = (beta**2 - 2*beta*tau + tau**2) * tau**2/4 * C_1
        g_tca_contrib_2 = (beta**3 - 3*beta**2*tau + 3*beta*tau**2 - tau**3) * tau / 6 * C_2 
        g_tca_contrib_3 = (beta - tau) * tau**3 / 6 * C_3

        self.dg_iaa_tca = - (0.5)**2 * ( g_tca_contrib_1 + g_tca_contrib_2 + g_tca_contrib_3) * I_n

        self.g_iaa_oca = self.g_iaa_nca + self.dg_iaa_oca
        self.g_iaa_tca = self.g_iaa_oca + self.dg_iaa_tca


    def __func_K(self, n): return n * (4*n**2 - 9*n + 4)
    def __func_C_1(self, n): return (n - 2) * (n + 0) + (n - 1) * (n - 2)     
    def __func_C_2(self, n): return (n - 1) * (n - 1)

        
def test_n_fermions(n, verbose):

    gf_struct = [('0', n)]
    h_int = 0.0 * Operator()

    analytic_diagram_cf(h_int, gf_struct, verbose)
    
    
def analytic_diagram_cf(h_int, gf_struct, verbose):

    S = TriqsSolver(beta=2.0, gf_struct=gf_struct, eps=1e-12, w_max=10.0)

    for bidx, delta_tau in S.Delta_tau:
        delta_w = make_gf_dlr_imfreq(delta_tau)
        delta_w << inverse(iOmega_n)
        delta_tau[:] = make_gf_dlr_imtime(delta_w)
    
    S.solve(h_int=h_int, order=1, tol=1e-9, maxiter=0)

    g_iaa_nca = S.S.calc_spgf(max_order=1, verbose=True)
    g_iaa_oca = S.S.calc_spgf(max_order=2, verbose=True)
    g_iaa_tca = S.S.calc_spgf(max_order=3, verbose=True)

    dg_iaa_oca = g_iaa_oca - g_iaa_nca
    dg_iaa_tca = g_iaa_tca - g_iaa_oca
    
    Sigma_iaa_nca = S.S.calc_Sigma(max_order=1, verbose=True)
    Sigma_iaa_oca = S.S.calc_Sigma(max_order=2, verbose=True)
    Sigma_iaa_tca = S.S.calc_Sigma(max_order=3, verbose=True)

    dSigma_iaa_oca = Sigma_iaa_oca - Sigma_iaa_nca
    dSigma_iaa_tca = Sigma_iaa_tca - Sigma_iaa_oca
    
    def test_equal_diagonal(arr):
        assert(len(arr.shape) == 3)
        assert(arr.shape[1] == arr.shape[2])
        m = arr.shape[-1]

        off_diag = arr.copy()
        for i in range(m):
            np.testing.assert_array_almost_equal(arr[:, i, i], arr[:, 0, 0])
            off_diag[:, i, i] -= arr[:, i, i]

        np.testing.assert_array_almost_equal(off_diag, np.zeros_like(off_diag))

    
    for arr in [g_iaa_nca, g_iaa_oca, Sigma_iaa_nca, Sigma_iaa_oca]:
        test_equal_diagonal(arr)
    
    # -- Analytic solution

    n = len(S.S.fundamental_operators)
    N = S.S.H_mat.shape[0]
    
    ref = AnalyticDiagrams(n, N, S.S.tau_i, S.beta)
    
    
    if verbose and is_root():
    
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        
        subp = [3, 3, 1]

        plt.subplot(*subp); subp[-1] += 1

        for i in range(n):
            plt.plot(S.S.tau_i, S.Delta_tau['0'].data[:, i, i].flatten().real, 'x-')
        plt.ylim([-.75, 0])
        plt.ylabel(r'$\Delta(\tau)$')

        plt.subplot(*subp); subp[-1] += 1

        #for i in range(N):
        for i in [0]:
            plt.plot(S.S.tau_i, S.S.G0_iaa[:, i, i].real, 'x-', label='G0')
            plt.plot(S.S.tau_i, ref.G0_iaa[:, i, i].real, '+-', label='G0 ref')

        plt.legend(loc='best')
        plt.ylabel(r'$\hat{G}(\tau)$')

        plt.subplot(*subp); subp[-1] += 1
        
        plt.subplot(*subp); subp[-1] += 1

        #for i in range(n):
        for i in [0]:
            plt.plot(S.S.tau_i, g_iaa_nca[:, i, i].flatten().real, '-', label='nca')
            plt.plot(S.S.tau_i, ref.g_iaa_nca[:, i, i].flatten().real, 'x', label='nca (ref)')

        plt.legend(loc='best')
        #plt.ylim([-.75, 0])
        plt.ylabel(r'$g(\tau)$')

        plt.subplot(*subp); subp[-1] += 1

        #for i in range(n):
        for i in [0]:
            plt.plot(S.S.tau_i, dg_iaa_oca[:, i, i].flatten().real, '-', label='oca')
            plt.plot(S.S.tau_i, ref.dg_iaa_oca[:, i, i].flatten().real, '+', label='oca (ref)')

        plt.legend(loc='best')
        #plt.ylim([-.75, 0])
        plt.ylabel(r'$g(\tau)$')

        plt.subplot(*subp); subp[-1] += 1

        #for i in range(n):
        for i in [0]:
            plt.plot(S.S.tau_i, dg_iaa_tca[:, i, i].flatten().real, '-', label='tca')
            plt.plot(S.S.tau_i, ref.dg_iaa_tca[:, i, i].flatten().real, '+', label='tca (ref)')

        plt.legend(loc='best')
        #plt.ylim([-.75, 0])
        plt.ylabel(r'$g(\tau)$')
        
        plt.subplot(*subp); subp[-1] += 1

        #for i in range(N):
        for i in [0]:
            plt.plot(S.S.tau_i, Sigma_iaa_nca[:, i, i].real, 'x-', label='nca')
            plt.plot(S.S.tau_i, ref.Sigma_iaa_nca[:, i, i].real, '+-', label='nca ref')
            #plt.plot(TS.S.tau_i, TS.G_tau[0].data.flatten().real, 'x-', label='triqs')
        plt.legend(loc='best')
        plt.ylabel(r'$\hat{\Sigma}(\tau)$')

        plt.subplot(*subp); subp[-1] += 1

        #for i in range(N):
        for i in [0]:
            plt.plot(S.S.tau_i, dSigma_iaa_oca[:, i, i].real, 'x-', label='oca')
            plt.plot(S.S.tau_i, ref.dSigma_iaa_oca[:, i, i].real, '+-', label='oca ref')
            
        plt.legend(loc='best')
        plt.ylabel(r'$\hat{\Sigma}_{OCA}(\tau)$')

        plt.subplot(*subp); subp[-1] += 1

        #for i in range(N):
        for i in [0]:
            plt.plot(S.S.tau_i, dSigma_iaa_tca[:, i, i].real, 'x-', label='tca')
            plt.plot(S.S.tau_i, ref.dSigma_iaa_tca[:, i, i].real, '+-', label='tca ref')
            
        plt.legend(loc='best')
        plt.ylabel(r'$\hat{\Sigma}_{TCA}(\tau)$')
        
        plt.tight_layout()
        plt.show()


    if is_root():
        np.testing.assert_array_almost_equal(S.S.G0_iaa, S.S.G_iaa)
        np.testing.assert_array_almost_equal(ref.G0_iaa, S.S.G0_iaa)

        np.testing.assert_array_almost_equal(Sigma_iaa_nca, ref.Sigma_iaa_nca)
        np.testing.assert_array_almost_equal(Sigma_iaa_oca, ref.Sigma_iaa_oca)
        np.testing.assert_array_almost_equal(Sigma_iaa_tca, ref.Sigma_iaa_tca)

        np.testing.assert_array_almost_equal(g_iaa_nca, ref.g_iaa_nca)
        np.testing.assert_array_almost_equal(g_iaa_oca, ref.g_iaa_oca) 
        np.testing.assert_array_almost_equal(g_iaa_tca, ref.g_iaa_tca)


        
def test_prefactors():

    nK = [1, 2, 3, 4, 5, 6]
    K = [-1, 4, 39, 128, 295, 564]

    pK = np.polyfit(nK, K, 3)
    print(f'pK = {pK}')

    K_err = np.max(np.abs(np.polyval(pK, nK) - K))
    print(f'K_err = {K_err}')
    
    nC = [1, 2, 3, 4, 5]
    C1 = [-1, 0, 5, 14, 27]
    C2 = [0, 1, 4, 9, 16]

    p1 = np.polyfit(nC, C1, 2)
    p2 = np.polyfit(nC, C2, 2)

    print(f'p1 = {p1}')
    print(f'p2 = {p2}')
    
    
    for n in [6]:
        print(f'n = {n}')
        print(f'C1 = {np.polyval(p1, n)}')
        print(f'C2 = {np.polyval(p2, n)}')

    
    import matplotlib.pyplot as plt

    plt.plot(nK, K, 's', label='K')
    plt.plot(nC, C1, 'o', label='C1')
    plt.plot(nC, C2, '>', label='C2')

    x = np.linspace(0, 5.5, num=400)
    
    plt.plot(x, np.polyval(pK, x), ':', label='K fit')
    plt.plot(x, np.polyval(p1, x), ':', label='C1 fit')
    plt.plot(x, np.polyval(p2, x), ':', label='C2 fit')

    plt.plot(x, func_K(x), '-', label='K func')
    plt.plot(x, func_C_1(x), '-', label='C1 func')
    plt.plot(x, func_C_2(x), '-', label='C2 func')
    
    plt.legend(loc='best')
    plt.show()
        

if __name__ == '__main__':

    #test_prefactors(); exit()

    for n in range(1, 4):
        test_n_fermions(n=n, verbose=False)

