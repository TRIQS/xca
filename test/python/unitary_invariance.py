################################################################################
#
# triqs_soehyb: Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2024 by H. U.R. Strand
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

"""
Single particle basis unitary invariance test.

The PPSC equations are invariant with respect to any
unitary transform of the single particle basis.

This test solves a simple problem in two single particle basises
and checks that the result is the same by trasforming the result
with the inverse unitary transform.
"""


import itertools
import numpy as np

from mpi4py import MPI

from triqs.gf import Gf, MeshImTime
from triqs.operators import c, c_dag

from pyed.OperatorUtils import operator_single_particle_transform

from triqs_soehyb.pycppdlr import build_dlr_rf
from triqs_soehyb.pycppdlr import ImTimeOps
from triqs_soehyb.impurity import Fastdiagram
from triqs_soehyb.solver import Solver


def get_Hamiltonian(eps1=-0.1, t = 0.5 + 0.5j, U = 1.0):
    
    n_up = c_dag(0,0) * c(0,0)
    n_do = c_dag(0,1) * c(0,1)

    H = eps1 * ( n_up + n_do ) + U * n_up * n_do + \
        t * c_dag(0,0)*c(0,1) + np.conj(t) * c_dag(0,1)*c(0,0)

    return H


def get_ppsc_soe_gf(H, delta_iaa, beta, order=1, ntau=100):
    
    lamb, eps = 100., 1e-12

    dlr_rf = build_dlr_rf(lamb, eps)
    ito = ImTimeOps(lamb, dlr_rf)
    
    fundamental_operators = [c(0,0), c(0, 1)]
    
    S = Solver(beta, lamb, eps, H, fundamental_operators)

    S.set_hybridization(
        delta_iaa, compress=True, Hermitian=True, verbose=True)
        
    S.solve(order, tol=1e-12, maxiter=1, mix=1., update_eta_exact=True, verbose=True)

    g_iaa = S.calc_spgf(order)

    g_xaa = ito.vals2coefs(g_iaa)
    
    mesh = MeshImTime(beta, 'Fermion', ntau)
    g_tau = Gf(mesh=mesh, target_shape=(2, 2))

    tau_j = np.array([ float(t) for t in g_tau.mesh ])

    def interp(g_xaa, tau_j):
        eval = lambda t : ito.coefs2eval(g_xaa, t/beta)
        return np.vectorize(eval, signature='()->(m,m)')(tau_j)
    
    g_jaa = interp(g_xaa, tau_j)

    g_tau.data[:] = g_jaa
    
    return g_tau


def test_unitary_symmetry_for_ppsc(order=1, verbose=False):

    # -- Complex unitary transform
    
    U = np.array([
        [1., 1j],
        [1, -1j]
        ]) / np.sqrt(2)

    np.testing.assert_almost_equal(U @ U.T.conj(), np.eye(2))

    # -- Setup the same two state problem in two single particle bases
    
    beta = 10.
    fundamental_operators = [c(0,0), c(0, 1)]

    lamb, eps = 100., 1e-12
    dlr_rf = build_dlr_rf(lamb, eps)
    ito = ImTimeOps(lamb, dlr_rf)
    fd = Fastdiagram(beta, lamb, ito, np.empty((0, 0, 0)), np.empty((0, 0, 0)))
    
    H1 = get_Hamiltonian()
    H2 = operator_single_particle_transform(H1, U, fundamental_operators)

    V = 0.2
    h = np.array([
        [0.1,  0.1],
        [0.1, -0.1],
        ])
    
    delta1_iaa = V**2 * fd.free_greens(beta, h)
    delta2_iaa = np.einsum('ab,tbc,cd->tad', U.T.conj(), delta1_iaa, U)

    # -- Use ppsc-soe to compute the single particle Green's function for the two systems
    
    g1_tau = get_ppsc_soe_gf(H1, delta1_iaa, beta, order=order)
    g2_tau = get_ppsc_soe_gf(H2, delta2_iaa, beta, order=order)

    # -- Rotate the 2nd result back to original basis
    
    g2_tau.data[:] = np.einsum('ab,tbc,cd->tad', U, g2_tau.data, U.T.conj())

    # -- Compare solutions
    
    diff = np.max(np.abs(g1_tau.data - g2_tau.data))
    print('='*72)
    print('='*72)
    print(f'order = {order}, diff = {diff}')
    print('='*72)
    print('='*72)
    
    if verbose:
        from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti
        plt.figure(figsize=(6, 8))
        subp = [4, 2, 1]
        for i, j in itertools.product(range(2), repeat=2):
            plt.subplot(*subp); subp[-1] += 1
            oplotr(g1_tau[i, j], 'g-', label='Re')
            oploti(g1_tau[i, j], 'g--', label='Im')
            oplotr(g2_tau[i, j], 'b-', label='Re (rot)')
            oploti(g2_tau[i, j], 'b--', label='Im (rot)')

        for i, j in itertools.product(range(2), repeat=2):
            plt.subplot(*subp); subp[-1] += 1
            oplotr(g1_tau[i, j]-g2_tau[i, j], 'g-', label='Re')
            oploti(g1_tau[i, j]-g2_tau[i, j], 'g--', label='Im')
            plt.ylabel('g1 - g2')
            
        plt.tight_layout()
        plt.show()

    np.testing.assert_array_almost_equal(g1_tau.data, g2_tau.data)
        
        
if __name__ == '__main__':

    for order in [1, 2, 3]:
        test_unitary_symmetry_for_ppsc(order=order, verbose=False)
