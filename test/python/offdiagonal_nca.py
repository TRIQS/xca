################################################################################
#
# triqs_soehyb: Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2023 by H. U.R. Strand
#
# triqs_soehyb is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# triqs_soehyb is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# triqs_soehyb. If not, see <http://www.gnu.org/licenses/>.
#
################################################################################

from itertools import product
import numpy as np

from triqs.operators import c, c_dag
from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization


from triqs_soehyb.pycppdlr import build_dlr_rf
from triqs_soehyb.pycppdlr import ImTimeOps

from triqs_soehyb.impurity import Fastdiagram


def NCA_Sigma_ref_impl(ito, beta, tau_i, G_iaa, delta_iaa, F, F_dag):

    def interp(g_xaa, tau_j):
        eval = lambda t : ito.coefs2eval(g_xaa, np.real(t)/beta)
        return np.vectorize(eval, signature='()->(m,m)')(tau_j)
    
    Sigma_iaa = np.zeros_like(G_iaa)

    delta_xaa = ito.vals2coefs(delta_iaa)
    delta_iaa_rev = interp(delta_xaa, beta - tau_i)
    
    n = F.shape[0]
    for i, j in product(range(n), repeat=2):
        Sigma_iaa -= np.matmul(F_dag[i], np.matmul(G_iaa, F[j])) * delta_iaa[:, i, j][:, None, None]
        Sigma_iaa -= np.matmul(F[j], np.matmul(G_iaa, F_dag[i])) * delta_iaa_rev[:, i, j][:, None, None]

    return Sigma_iaa


def test_off_diagonal_NCA_Sigma(verbose=False):

    beta = 1.
    lamb = 1000.
    eps = 1e-12

    dlr_rf = build_dlr_rf(lamb, eps)
    ito = ImTimeOps(lamb, dlr_rf)

    e0 = +1.0
    e1 = -1.0
    v = 0.75 + 0.75j

    H = e0 * c_dag(0,0) * c(0,0) + e1 * c_dag(0,1) * c(0,1) + v * c_dag(0,0) * c(0,1) + np.conj(v) * c_dag(0,1) * c(0,0)
    fundamental_operators = [ c(0,i) for i in range(2) ]
    ed = TriqsExactDiagonalization(H, fundamental_operators, beta)

    F = np.array([np.array(
        ed.rep.sparse_operators.c_dag[idx].T.conj().todense())
                  for idx in range(len(fundamental_operators)) ])

    F_dag = np.array([np.array(
        ed.rep.sparse_operators.c_dag[idx].todense())
                      for idx in range(len(fundamental_operators)) ])

    fd = Fastdiagram(beta, lamb, ito, F, F_dag)

    t = 1.0
    vp = 1.0 + 2.0j
    e0, e1 = 0.5, -2.0
    ek = np.array([[e0, vp], [np.conj(vp), e1]], dtype=complex)
    delta_iaa = t**2 * fd.free_greens(beta, ek)
    
    fd.hyb_init(delta_iaa)
    fd.hyb_decomposition()

    H_mat = np.array(ed.ed.H.todense())
    G_iaa = fd.free_greens_ppsc(beta, H_mat)

    tau_i = fd.get_it_actual()

    Sigma_iaa = fd.Sigma_calc(G_iaa, 'NCA')

    Sigma_ref_iaa = NCA_Sigma_ref_impl(ito, beta, tau_i, G_iaa, delta_iaa, F, F_dag)

    def interp(g_xaa, tau_j):
        eval = lambda t : ito.coefs2eval(g_xaa, np.real(t)/beta)
        return np.vectorize(eval, signature='()->(m,m)')(tau_j)

    tau_f = np.linspace(0, beta, num=1000)

    delta_xaa = ito.vals2coefs(delta_iaa)
    delta_faa = interp(delta_xaa, tau_f)

    G_xaa = ito.vals2coefs(G_iaa)
    G_faa = interp(G_xaa, tau_f)

    Sigma_xaa = ito.vals2coefs(Sigma_iaa)
    Sigma_faa = interp(Sigma_xaa, tau_f)

    Sigma_ref_xaa = ito.vals2coefs(Sigma_ref_iaa)
    Sigma_ref_faa = interp(Sigma_ref_xaa, tau_f)

    if False:
        from pyppsc.dlr import dlr

        from pyppsc.ppsc import PseudoParticleInteraction
        from pyppsc.ppsc import PseudoParticleResponseFunction
        from pyppsc.ppsc import PseudoParticleStrongCoupling

        d = dlr(lamb=lamb, eps=eps)
        tau_i = d.get_tau(beta)
        delta_iaa = interp(delta_xaa, tau_i)
        n = delta_iaa.shape[-1]
        prange = product(range(n), repeat=2)
        ppints = [ PseudoParticleInteraction(F_dag[i], delta_iaa[:, i, j].reshape(len(d), 1, 1), F[j]) for i, j in prange ]
        ppchis = []
        ppsc = PseudoParticleStrongCoupling(ed, d, ppints, ppchis, eta=0., cplx_oca=True, cplx_tca=True)
        G0_iaa = ppsc.calc_G0_tau()
        Sigma_iaa = ppsc.calc_Sigma_iaa(G0_iaa)
        Sigma_xaa = d.dlr_from_tau(Sigma_iaa)
        Sigma_ref2_faa = d.eval_dlr_tau(Sigma_xaa, tau_f, beta)

        np.testing.assert_array_almost_equal(Sigma_ref_faa, Sigma_ref2_faa)

    if verbose:
        import matplotlib.pyplot as plt

        subp = [3, 2, 1]

        plt.subplot(*subp); subp[-1] += 1
        for i, j in product(range(2), repeat=2):
            plt.plot(tau_f, -delta_faa[:, i, j].real)

        plt.subplot(*subp); subp[-1] += 1
        for i, j in product(range(2), repeat=2):
            plt.plot(tau_f, -delta_faa[:, i, j].imag)

        plt.subplot(*subp); subp[-1] += 1
        for i, j in product(range(4), repeat=2):
            plt.plot(tau_f, -G_faa[:, i, j].real)

        plt.subplot(*subp); subp[-1] += 1
        for i, j in product(range(4), repeat=2):
            plt.plot(tau_f, -G_faa[:, i, j].imag)

        plt.subplot(*subp); subp[-1] += 1
        for i, j in product(range(4), repeat=2):
            plt.plot(tau_f, -Sigma_faa[:, i, j].real, '-k')
            plt.plot(tau_f, -Sigma_ref_faa[:, i, j].real, ':r', lw=2)
            #plt.plot(tau_f, -Sigma_ref2_faa[:, i, j].real, '--g')

        plt.subplot(*subp); subp[-1] += 1
        for i, j in product(range(4), repeat=2):
            plt.plot(tau_f, -Sigma_faa[:, i, j].imag, '-k')
            plt.plot(tau_f, -Sigma_ref_faa[:, i, j].imag, ':r', lw=2)
            #plt.plot(tau_f, -Sigma_ref2_faa[:, i, j].imag, '--g')

        plt.tight_layout()
        plt.show()

    np.testing.assert_array_almost_equal(Sigma_faa, Sigma_ref_faa)


if __name__ == '__main__':
    test_off_diagonal_NCA_Sigma(verbose=False)
