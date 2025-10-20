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
from triqs_soehyb.solver import Sigma_calc_loop, G_calc_loop, is_root


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

    dlr_rf = build_dlr_rf(lamb, eps)
    ito = ImTimeOps(lamb, dlr_rf)

    H = -mu * c_dag(0,0) * c(0,0)
    fundamental_operators = [ c(0,i) for i in range(1) ]
    ed = TriqsExactDiagonalization(H, fundamental_operators, beta)

    mat_c_dag = np.array(ed.rep.sparse_operators.c_dag[0].todense()) 
    mat_c = mat_c_dag.T.conj()

    F = np.array([mat_c])
    F_dag = np.array([mat_c_dag])
    
    fd = Fastdiagram(beta, lamb, ito, F, F_dag)

    delta_iaa = t**2 * fd.free_greens(beta, np.array([[ek]]))
    
    fd.hyb_init(delta_iaa)
    fd.hyb_decomposition()

    H_mat = np.array(ed.ed.H.todense())
    I_mat = np.eye(H_mat.shape[0])
    
    G0_iaa = fd.free_greens_ppsc(beta, H_mat)
    G_iaa = G0_iaa.copy()
    
    tau_i = fd.get_it_actual()
    eta = 0.

    order_str = { 1 : 'NCA', 2 : 'OCA', 3 : 'TCA' }[order]
    
    for ppsc_iter in range(ppsc_maxiter):

        Sigma_iaa = fd.Sigma_calc(G_iaa, order_str)

        Sigma_iaa_ref = Sigma_calc_loop(fd, G_iaa, order)
        diff = np.max(np.abs(Sigma_iaa - Sigma_iaa_ref))
        if is_root():
            print('-'*72)
            print(f'Sigma_iaa diff : {diff:2.2E}')
            print('-'*72)
        np.testing.assert_array_almost_equal(Sigma_iaa, Sigma_iaa_ref)
        #exit()
        
        G_iaa_new = fd.time_ordered_dyson(beta, H_mat, eta, Sigma_iaa)

        Z = fd.partition_function(G_iaa_new)
        deta = np.log(Z) / beta
        G_iaa_new[:] *= np.exp(-tau_i * deta)[:, None, None]
        eta += deta

        ppsc_diff = np.max(np.abs(G_iaa - G_iaa_new))

        G_iaa = mix*G_iaa_new + (1-mix)*G_iaa 

        if is_root():
            print(f'PPSC: iter = {ppsc_iter:3d} diff = {ppsc_diff:2.2E}')
        if ppsc_diff < ppsc_tol: break

    g_iaa = fd.G_calc(G_iaa, order_str)
    n_orb = g_iaa.shape[-1]
    g_iaa_ref = G_calc_loop(fd, G_iaa, order, n_orb)

    diff = np.max(np.abs(g_iaa - g_iaa_ref))
    if is_root():
        print('-'*72)
        print(f'g_iaa diff : {diff:2.2E}')
        print('-'*72)
    np.testing.assert_array_almost_equal(g_iaa, g_iaa_ref)
    #exit()
    
    class Dummy():
        def __init__(self):
            pass

    d = Dummy()
    d.G_iaa = G_iaa
    d.g_iaa = g_iaa
    d.tau_i = tau_i
    d.fd = fd
    d.ito = ito

    assert( tau_j is not None )

    d.tau_j = tau_j
    
    d.g_xaa = ito.vals2coefs(d.g_iaa)
    d.G_xaa = ito.vals2coefs(d.G_iaa)

    def interp(g_xaa, tau_j):
        eval = lambda t : ito.coefs2eval(g_xaa, t)
        return np.vectorize(eval, signature='()->(m,m)')(tau_j)
    
    d.g_jaa = interp(d.g_xaa, tau_j)
    d.G_jaa = interp(d.G_xaa, tau_j)

    return d


def test_spinless_dimer(verbose=False):
    
    opts = dict(beta=1.0, t=1.0, ek=0., mu=-0.01) 
        
    g_ed = spinless_dimer_ed(**opts)
    tau_j = np.array([float(t) for t in g_ed.mesh])

    ppsc_opts = dict(
        lamb=10.,
        eps=1e-10,
        ppsc_tol=1e-8,
        ppsc_maxiter=100,
        mix=1.0,
        verbose=False,
        tau_j=tau_j,
        )
    
    nca = calc_spinless_dimer(order=1, **opts, **ppsc_opts)
    oca = calc_spinless_dimer(order=2, **opts, **ppsc_opts)    
    tca = calc_spinless_dimer(order=3, **opts, **ppsc_opts)
    
    if verbose:
        import itertools
        from triqs.plot.mpl_interface import oplot, oplotr, oploti, plt
        plt.figure(figsize=(12, 4))
        subp = [1, 4, 1]

        for i, j in itertools.product(range(1), repeat=2):
            plt.subplot(*subp); subp[-1] += 1
            plt.title(f'g_{i}{j}')

            plt.plot(nca.tau_j, nca.g_jaa[:, i, j], '-', label=f'NCA')
            #plt.plot(oca.tau_j, oca.g_jaa[:, i, j], '-', label=f'OCA')
            plt.plot(tca.tau_j, tca.g_jaa[:, i, j], '-', label=f'TCA')
            
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
            plt.plot(tau_j, np.abs(nca.g_jaa[:, i, j] - g_ed.data[:, i, j].real), '-', label='NCA')
            #plt.plot(tau_j, np.abs(oca.g_jaa[:, i, j] - g_ed.data[:, i, j].real), '-', label='OCA')
            plt.plot(tau_j, np.abs(tca.g_jaa[:, i, j] - g_ed.data[:, i, j].real), '-', label='TCA')
            plt.legend()
            plt.semilogy([], [])
            plt.grid(True)
            plt.ylabel(r'$|G_{XCA} - G_{exact}|$')
            plt.xlabel(r'$\tau$')
            plt.title(None)

        plt.subplot(*subp); subp[-1] += 1
        for d in [nca, tca]:
            plt.plot(d.tau_i, -d.G_iaa[:, 0, 0].real, '.--', label='G_0')
            plt.plot(d.tau_i, -d.G_iaa[:, 1, 1].real, '.-', label='G_1')
        plt.grid(True)
        plt.ylabel(r'$\hat{G}(\tau)$')
        plt.xlabel(r'$\tau$')
        plt.legend()

        plt.subplot(*subp); subp[-1] += 1
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

    np.testing.assert_array_almost_equal(nca.g_iaa, oca.g_iaa)

    np.testing.assert_array_almost_equal(nca.g_jaa.flatten(), g_ed.data[:, 0, 0].real, decimal=2)
    np.testing.assert_array_almost_equal(tca.g_jaa.flatten(), g_ed.data[:, 0, 0].real, decimal=4)

    
if __name__ == '__main__':
    
    test_spinless_dimer(verbose=False)
