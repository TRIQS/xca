################################################################################
#
# triqs_soehyb: Sum-Of-Exponentials bold HYBridization expansion impurity solver
#
# Copyright (C) 2025 by H. U.R. Strand
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

""" Test TRIQS solver API against low-level numpy solver API """


import numpy as np

from mpi4py import MPI

from triqs.gf import make_gf_dlr_imtime, make_gf_dlr_imfreq, inverse, iOmega_n
from triqs.operators import c, c_dag

from triqs_soehyb.solver import Solver
from triqs_soehyb.triqs_solver import TriqsSolver


def test_triqs_solver_one_fermion(verbose):
    
    mu = 1/3
    beta = 10.0
    ek = 0.5
    t = 0.25

    w_max = 100.0
    eps = 1e-12

    lamb = beta * w_max

    order = 1

    H_loc = -mu * c_dag('bl',0) * c('bl',0)

    # -- Solve using Triqs solver API
    
    TS = TriqsSolver(beta=beta, gf_struct=[('bl', 1)], eps=eps, w_max=w_max)

    for bidx, delta_tau in TS.Delta_tau:
        delta_w = make_gf_dlr_imfreq(delta_tau)
        delta_w << t**2 * inverse(iOmega_n - ek)
        delta_tau[:] = make_gf_dlr_imtime(delta_w)
    
    TS.solve(h_int=H_loc, order=order, tol=1e-9)

    print(TS.G_tau)

    # -- Reference solution using low level numpy api

    fundamental_operators = [ c('bl',i) for i in range(1) ]

    S = Solver(beta, lamb, eps, H_loc, fundamental_operators, verbose=True)

    delta_iaa = t**2 * S.fd.free_greens(beta, np.array([[ek]]))
    S.set_hybridization(delta_iaa)

    S.solve(order, tol=1e-9)

    g_iaa = S.calc_spgf(order)

    if verbose:
    
        import matplotlib.pyplot as plt

        subp = [2, 1, 1]

        plt.subplot(*subp); subp[-1] += 1

        plt.plot(S.tau_i, delta_iaa.flatten().real, '.-', label='numpy')
        plt.plot(TS.S.tau_i, TS.Delta_tau[0].data.flatten().real, 'x-', label='triqs')
        plt.legend(loc='best')

        plt.subplot(*subp); subp[-1] += 1

        plt.plot(S.tau_i, g_iaa.flatten().real, '.-', label='numpy')
        plt.plot(TS.S.tau_i, TS.G_tau[0].data.flatten().real, 'x-', label='triqs')
        plt.legend(loc='best')

        plt.tight_layout()
        plt.show()

    # -- Compare solution from Triqs solver and low level solver
    
    np.testing.assert_array_almost_equal(TS.Delta_tau['bl'].data, delta_iaa)
    np.testing.assert_array_almost_equal(TS.G_tau['bl'].data, g_iaa)
        

if __name__ == '__main__':

    test_triqs_solver_one_fermion(verbose=False)
