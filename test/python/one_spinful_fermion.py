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

"""
Solve interacting single spinful fermion with local Hubbard U,
hybridized with a discrete bath with one state.

The solution is compared with reference results from NESSi-ppsc
in the zeroth, first, and second order (atomic, NCA, and OCA) expansion order.
"""

import numpy as np

from mpi4py import MPI

import triqs.utility.mpi as mpi

import h5py

from triqs.operators import c, c_dag, n
from triqs.operators.util import N_op

from triqs.gf import make_gf_dlr_imtime, make_gf_dlr_imfreq, SemiCircular, inverse, iOmega_n

from triqs_soehyb.triqs_solver import TriqsSolver

from itertools import product


class Dummy():
    def __init__(self): pass

    
def load_nessi_ppsc(path, data_file, input_file):

    from ReadCNTR import read_input_file
    from ReadCNTRhdf5 import read_imp_h5file
    
    imp_filename = f'{path}/{data_file}'
    print('--> Loading:', imp_filename)

    d = read_imp_h5file(imp_filename)
    d.imp.parm =  read_input_file(f'{path}/{input_file}')

    d.imp.tau = np.linspace(0.0, d.imp.beta[0], num=d.imp.ntau[0]+1)
    d.imp.t = np.arange(-1, d.imp.nt[0] + 1) * d.imp.h
    
    return d


def solve_one_spinful_fermion(
        beta=10.0, U=1.0, mu=0.0, V=0.1, eps1=-0.1, eps2=+0.1,
        order=1, eps=1e-12, w_max=40.0,
        dmft_maxiter=1, dmft_tol=1e-9,
        ):

    #mu = mu + U/2
    
    norb = 1
    spin_names = ('up','do')
    orb_names = list(range(norb))
    
    gf_struct = [ (s + f"_{n}", 1) for s, n in product(spin_names, orb_names) ]

    H_int = U * n('up_0', 0) * n('do_0', 0)
    N_tot = N_op(spin_names, norb, off_diag=False)
    H_loc = H_int - mu * N_tot + eps1 * N_tot
    
    S = TriqsSolver(beta=beta, gf_struct=gf_struct, eps=eps, w_max=w_max)

    for bidx, delta_tau in S.Delta_tau:
        delta_w = make_gf_dlr_imfreq(delta_tau)
        delta_w << V**2 * inverse(iOmega_n - eps2)
        delta_tau[:] = make_gf_dlr_imtime(delta_w)
    
    for iter in range(1, dmft_maxiter+1):
        
        S.solve(h_int=H_loc, order=order, tol=1e-9, maxiter=100, update_eta_exact=False)

        dmft_diff = np.max(np.abs((S.G_tau['up_0'] - S.Delta_tau['up_0']).data))

        if mpi.is_master_node():
            print('='*72)
            print(f'DMFT: iter = {iter} diff = {dmft_diff:2.2E}')
            print('='*72)
            print()

        if dmft_diff < dmft_tol or iter == dmft_maxiter: break
        
        for bidx, delta_tau in S.Delta_tau:
            delta_tau[:] = S.G_tau[bidx]

    S.g_iaa_nca = S.S.calc_spgf(max_order=1)
            
    from triqs.gf import make_gf_imtime
    S.G_tau_fine = make_gf_imtime(S.G_tau, n_tau=801)
    S.Delta_tau_fine = make_gf_imtime(S.Delta_tau, n_tau=801)

    S.H_int = H_int

    return S


def test_one_spinful_fermion(verbose=True):
    
    nessi = Dummy()
    
    filename = 'nessi_ppsc_one_spinful_fermion.ref.h5'
    print(f'--> Loading: {filename}')
    with h5py.File(filename, 'r') as fd:

        def read_grp(grp):
            data = Dummy()
            data.imp = Dummy()
            data.imp.tau = np.array(grp['tau'])
            data.d = Dummy()
            data.d.mat = np.array(grp['delta_tau'])
            data.g = Dummy()
            data.g.mat = np.array(grp['g_tau'])
            return data
            
        nessi.atom = read_grp(fd['atom'])
        nessi.nca = read_grp(fd['nca'])
        nessi.oca = read_grp(fd['oca'])

    #exit()
    
    if False:
        path = '/Users/hugstr/dev/ppsc/examples/one_spinful_fermion/'

        nessi.atom = load_nessi_ppsc(path, 'data_ppsc_V0.h5', 'input_param_V0.txt')
        nessi.nca = load_nessi_ppsc(path, 'data_ppsc_nca.h5', 'input_param_nca.txt')
        nessi.oca = load_nessi_ppsc(path, 'data_ppsc_oca.h5', 'input_param_oca.txt')

    soehyb = Dummy()
    
    soehyb.atom = solve_one_spinful_fermion(V=0.0, order=1)
    soehyb.nca = solve_one_spinful_fermion(order=1)
    soehyb.oca = solve_one_spinful_fermion(order=2)

    if verbose:
        #import matplotlib.pyplot as plt

        from triqs.plot.mpl_interface import oplot, oplotr, oploti, plt

        subp = [2, 2, 1]

        plt.figure(figsize=(8, 8))

        plt.subplot(*subp); subp[-1] += 1
        plt.plot(nessi.atom.imp.tau, nessi.atom.d.mat.flatten().real, '-', label='atom nessi')

        plt.plot(nessi.atom.imp.tau, nessi.atom.d.mat.flatten().real, '-', label='atom nessi')
        plt.plot(nessi.nca.imp.tau, nessi.nca.d.mat.flatten().real, '--', label='nca nessi')
        plt.plot(nessi.oca.imp.tau, nessi.oca.d.mat.flatten().real, ':', label='oca nessi')

        plt.plot(soehyb.nca.S.tau_i, soehyb.nca.delta_iaa[:, 0, 0].real, '+', label='soehyb nca')
        plt.plot(soehyb.oca.S.tau_i, soehyb.oca.delta_iaa[:, 0, 0].real, 'x', label='soehyb oca')

        plt.ylabel(r'$\Delta(\tau)$')
        plt.legend(loc='best')

        plt.subplot(*subp); subp[-1] += 1

        delta_nca_diff = nessi.nca.d.mat.flatten().real - soehyb.nca.Delta_tau_fine['up_0'].data.flatten().real
        plt.plot(nessi.nca.imp.tau, delta_nca_diff)

        plt.ylabel(r'Diff in $\Delta(\tau)$')
        #plt.legend(loc='best')

        plt.subplot(*subp); subp[-1] += 1
        plt.plot(nessi.atom.imp.tau, nessi.atom.g.mat.flatten().real, '-', label='atom nessi')
        plt.plot(nessi.nca.imp.tau, nessi.nca.g.mat.flatten().real, '--', label='nca nessi')
        plt.plot(nessi.oca.imp.tau, nessi.oca.g.mat.flatten().real, ':', label='oca nessi')

        plt.plot(soehyb.atom.S.tau_i, soehyb.atom.g_iaa[:, 0, 0].real, '.', label='soehyb atom')
        plt.plot(soehyb.nca.S.tau_i, soehyb.nca.g_iaa[:, 0, 0].real, '+', label='soehyb nca')
        plt.plot(soehyb.oca.S.tau_i, soehyb.oca.g_iaa[:, 0, 0].real, 'x', label='soehyb oca')
        plt.plot(soehyb.oca.S.tau_i, soehyb.oca.g_iaa_nca[:, 0, 0].real, 'x', label='soehyb oca (g@nca)')
        
        #oplotr(soehyb.nca.G_tau_fine['up_0'], label='soehyb nca')

        plt.ylabel(r'$g(\tau)$')
        plt.legend(loc='best')

        plt.subplot(*subp); subp[-1] += 1

        g_atom_diff = nessi.atom.g.mat.flatten().real - soehyb.atom.G_tau_fine['up_0'].data.flatten().real
        g_nca_diff = nessi.nca.g.mat.flatten().real - soehyb.nca.G_tau_fine['up_0'].data.flatten().real
        g_oca_diff = nessi.oca.g.mat.flatten().real - soehyb.oca.G_tau_fine['up_0'].data.flatten().real

        plt.plot(nessi.atom.imp.tau, g_atom_diff, label='atom diff')
        plt.plot(nessi.nca.imp.tau, g_nca_diff, label='nca diff')
        plt.plot(nessi.oca.imp.tau, g_oca_diff, label='oca diff')

        plt.ylabel(r'Diff in $g(\tau)$')
        plt.legend(loc='best')


        plt.tight_layout()
        plt.show()


        
    delta_diff = nessi.atom.d.mat.flatten().real - soehyb.atom.Delta_tau_fine['up_0'].data.flatten().real

    g_atom_diff = nessi.atom.g.mat.flatten().real - soehyb.atom.G_tau_fine['up_0'].data.flatten().real
    g_nca_diff = nessi.nca.g.mat.flatten().real - soehyb.nca.G_tau_fine['up_0'].data.flatten().real
    g_oca_diff = nessi.oca.g.mat.flatten().real - soehyb.oca.G_tau_fine['up_0'].data.flatten().real

    print('='*72)
    print('Comparing SoE-HYB with NESSi-ppsc reference solution')
    
    print(f' delta diff = {np.max(np.abs(delta_diff)):2.2E}')
    
    print(f'atom g diff = {np.max(np.abs(g_atom_diff)):2.2E}')
    print(f' nca g diff = {np.max(np.abs(g_nca_diff)):2.2E}')
    print(f' oca g diff = {np.max(np.abs(g_oca_diff)):2.2E}')
    print('='*72)
    
    np.testing.assert_array_almost_equal(
        nessi.atom.d.mat.flatten().real, soehyb.atom.Delta_tau_fine['up_0'].data.flatten().real)

    np.testing.assert_array_almost_equal(
        nessi.atom.g.mat.flatten().real, soehyb.atom.G_tau_fine['up_0'].data.flatten().real)

    np.testing.assert_array_almost_equal(
        nessi.nca.g.mat.flatten().real, soehyb.nca.G_tau_fine['up_0'].data.flatten().real)

    np.testing.assert_array_almost_equal(
        nessi.oca.g.mat.flatten().real, soehyb.oca.G_tau_fine['up_0'].data.flatten().real)
         

if __name__ == '__main__':

    test_one_spinful_fermion(verbose=False)
