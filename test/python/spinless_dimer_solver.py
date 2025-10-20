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

import numpy as np

from mpi4py import MPI

from triqs.gf import Gf, MeshImTime
from triqs.operators import c, c_dag
from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

from triqs_soehyb.pycppdlr import build_dlr_rf
from triqs_soehyb.pycppdlr import ImTimeOps

from triqs_soehyb.impurity import Fastdiagram
from triqs_soehyb.solver import Solver, is_root


def spinless_dimer_ed(ntau=500, beta=1.0, t=1.0, ek=0.0, mu=0.01):
    
    H = - mu * c_dag(0, 0) * c(0, 0) + \
        t * (c_dag(0,0) * c(0,1) + c_dag(0,1) * c(0,0) ) + \
        ek * c_dag(0,1) * c(0,1)
    
    fundamental_operators = [ c(0,i) for i in range(2) ]
    ed = TriqsExactDiagonalization(H, fundamental_operators, beta)

    mesh = MeshImTime(beta, 'Fermion', ntau)
    g_tau = Gf(name=r'$g$', mesh=mesh, target_shape=(1,1))
    ed.set_g2_tau(g_tau[0,0], c(0,0), c_dag(0,0))
    return g_tau    


def calc_eta0(H, beta):
    E = np.linalg.eigvals(H)
    E0 = np.min(E.real)
    E -= E0
    Z = np.sum(np.exp(-beta*E))
    eta0 = E0 - np.log(np.real(Z)) / beta
    return eta0


def calc_spinless_dimer(
        beta = 1.0,
        t=1.0,
        ek=0.0,
        mu=0.01,
        lamb=200.,
        eps=1e-12,
        ppsc_tol=1e-9,
        ppsc_maxiter=100,
        order=1,
        mix=1.0,
        verbose=False,
        tau_j=None,
        ):

    print(f'Order: {order}')

    H = -mu * c_dag(0,0) * c(0,0)
    fundamental_operators = [ c(0,i) for i in range(1) ]

    S = Solver(beta, lamb, eps, H, fundamental_operators)

    delta_iaa = t**2 * S.fd.free_greens(beta, np.array([[ek]]))
    S.set_hybridization(delta_iaa)
    
    S.solve(order, tol=ppsc_tol, maxiter=ppsc_maxiter, mix=mix)

    g_iaa = S.calc_spgf(order)
    
    class Dummy():
        def __init__(self):
            pass

    d = Dummy()
    d.G_iaa = S.G_iaa
    d.g_iaa = g_iaa
    d.tau_i = S.tau_i
    d.fd = S.fd
    d.ito = S.ito
    d.order = order

    assert( tau_j is not None )

    d.tau_j = tau_j
    
    d.g_xaa = S.ito.vals2coefs(d.g_iaa)
    d.G_xaa = S.ito.vals2coefs(d.G_iaa)

    def interp(g_xaa, tau_j):
        eval = lambda t : S.ito.coefs2eval(g_xaa, t)
        return np.vectorize(eval, signature='()->(m,m)')(tau_j)
    
    d.g_jaa = interp(d.g_xaa, tau_j)
    d.G_jaa = interp(d.G_xaa, tau_j)

    return d


def test_spinless_dimer(verbose=False):
    
    #opts = dict(beta=1.0, t=1.0, ek=0., mu=-0.01) 
    opts = dict(beta=1.0, t=0.5, ek=0., mu=-0.01) 
        
    g_ed = spinless_dimer_ed(**opts)
    tau_j = np.array([float(t) for t in g_ed.mesh])

    ppsc_opts = dict(
        lamb=10.,
        eps=1e-12,
        ppsc_tol=1e-12,
        ppsc_maxiter=100,
        mix=1.0,
        verbose=False,
        tau_j=tau_j,
        )

    res = [ calc_spinless_dimer(order=order, **opts, **ppsc_opts) for order in [1, 3] ]
    nca = res[0]
    tca = res[1]
    
    #nca = calc_spinless_dimer(order=1, **opts, **ppsc_opts)
    #oca = calc_spinless_dimer(order=2, **opts, **ppsc_opts)    
    #tca = calc_spinless_dimer(order=3, **opts, **ppsc_opts)
    #fca = calc_spinless_dimer(order=4, **opts, **ppsc_opts)
    
    if verbose and is_root():
        import itertools
        from triqs.plot.mpl_interface import oplot, oplotr, oploti, plt
        plt.figure(figsize=(12, 4))
        subp = [1, 4, 1]

        for i, j in itertools.product(range(1), repeat=2):
            plt.subplot(*subp); subp[-1] += 1
            plt.title(f'g_{i}{j}')

            for r in res:
                plt.plot(r.tau_j, r.g_jaa[:, i, j], '-', label=f'Order {r.order}')
            #plt.plot(oca.tau_j, oca.g_jaa[:, i, j], '-', label=f'OCA')
            #plt.plot(tca.tau_j, tca.g_jaa[:, i, j], '-', label=f'TCA')
            #plt.plot(fca.tau_j, fca.g_jaa[:, i, j], '-', label=f'FCA')
            
            oplotr(g_ed[i, j], '--', label='ed')
            plt.legend()
            plt.grid(True)
            plt.ylabel(r'$G(\tau)$')
            plt.xlabel(r'$\tau$')
            plt.title(None)
            #plt.ylim(top=0)

        for i, j in itertools.product(range(1), repeat=2):
            plt.subplot(*subp); subp[-1] += 1
            plt.title(f'g_{i}{j}')
            for r in res:
                plt.plot(tau_j, np.abs(r.g_jaa[:, i, j] - g_ed.data[:, i, j].real), '-', label=f'Order {r.order}')
            #plt.plot(tau_j, np.abs(nca.g_jaa[:, i, j] - g_ed.data[:, i, j].real), '-', label='NCA')
            #plt.plot(tau_j, np.abs(oca.g_jaa[:, i, j] - g_ed.data[:, i, j].real), '-', label='OCA')
            #plt.plot(tau_j, np.abs(tca.g_jaa[:, i, j] - g_ed.data[:, i, j].real), '-', label='TCA')
            plt.legend()
            plt.semilogy([], [])
            plt.grid(True)
            plt.ylabel(r'$|G_{XCA} - G_{exact}|$')
            plt.xlabel(r'$\tau$')
            plt.title(None)

        plt.subplot(*subp); subp[-1] += 1
        for r in res:
            plt.plot(r.tau_i, -r.G_iaa[:, 0, 0].real, '.--', label='G_0')
            plt.plot(r.tau_i, -r.G_iaa[:, 1, 1].real, '.-', label='G_1')
        plt.grid(True)
        plt.ylabel(r'$\hat{G}(\tau)$')
        plt.xlabel(r'$\tau$')
        plt.legend()

        plt.subplot(*subp); subp[-1] += 1
        nca, tca = res[0], res[1]

        plt.plot(nca.tau_i, -nca.G_iaa[:, 0, 0].real + tca.G_iaa[:, 0, 0].real,
                 label='dG_0')
        plt.plot(nca.tau_i, -nca.G_iaa[:, 1, 1].real + tca.G_iaa[:, 1, 1].real,
                 label='dG_1')
        plt.grid(True)
        plt.ylabel(r'$\hat{G}(\tau)$')
        plt.xlabel(r'$\tau$')
        plt.legend()
            
        plt.tight_layout()
        plt.savefig('figure_spinless_dimer.pdf')
        plt.show()

    #np.testing.assert_array_almost_equal(nca.g_iaa, oca.g_iaa)

    np.testing.assert_array_almost_equal(nca.g_jaa.flatten(), g_ed.data[:, 0, 0].real, decimal=2)
    np.testing.assert_array_almost_equal(tca.g_jaa.flatten(), g_ed.data[:, 0, 0].real, decimal=4)

    
if __name__ == '__main__':
    
    test_spinless_dimer(verbose=False)
