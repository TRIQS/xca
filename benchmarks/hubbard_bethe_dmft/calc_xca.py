
"""
Solve Hubbard model on the Bethe graph with DMFT using the XCA impurity solver at 1st order.

Using both the TRIQS solver and the block sparse solver.
"""

import numpy as np

import triqs.utility.mpi as mpi

from h5 import HDFArchive

from triqs.operators import n

from triqs.gf import make_gf_dlr_imtime, make_gf_dlr_imfreq, SemiCircular

from triqs_xca.triqs_solver import TriqsSolver


def solve_one_spinful_fermion_triqs_solver(
        beta=10.0, U=6.0, mu=0.0, 
        order=1, eps=1e-12, w_max=10.0,
        dmft_maxiter=40, dmft_tol=1e-9,
        ppsc_maxiter=40, ppsc_tol=1e-9,
        ):

    spin_names = ['up', 'do']
    gf_struct = [(s, 1) for s in spin_names]

    H_loc = U * n('up', 0) * n('do', 0) - (mu + U/2) * (n('up', 0) + n('do', 0))
    
    S = TriqsSolver(beta=beta, gf_struct=gf_struct, eps=eps, w_max=w_max)

    Delta_w = make_gf_dlr_imfreq(S.Delta_tau)

    for spin in spin_names:
        Delta_w[spin] << SemiCircular(1.0)
        S.Delta_tau[spin] << make_gf_dlr_imtime(Delta_w[spin])
    
    for iter in range(1, dmft_maxiter+1):
        
        S.solve(h_int=H_loc, order=order, tol=ppsc_tol, maxiter=ppsc_maxiter, update_eta_exact=False)

        dmft_diff = np.max(np.abs((S.G_tau['up'] - S.Delta_tau['up']).data))

        if mpi.is_master_node():
            print('='*72)
            print(f'DMFT: iter = {iter} diff = {dmft_diff:2.2E}')
            print('='*72)
            print()

        if dmft_diff < dmft_tol or iter == dmft_maxiter: break
        
        for spin in spin_names:
            S.Delta_tau[spin] << S.G_tau[spin]

    S.g_iaa_nca = S.S.calc_spgf(max_order=1)
            
    from triqs.gf import make_gf_imtime
    S.G_tau_fine = make_gf_imtime(S.G_tau, n_tau=801)
    S.Delta_tau_fine = make_gf_imtime(S.Delta_tau, n_tau=801)

    S.H_loc = H_loc

    return S


def solve_one_spinful_fermion_block_sparse_solver(
        beta=10.0, U=6.0, mu=0.0, 
        order=1, eps=1e-12, w_max=20.0,
        dmft_maxiter=40, dmft_tol=1e-6,
        ppsc_maxiter=40, ppsc_tol=1e-8,
        ):

    spin_names = ['up', 'do']
    gf_struct = [(s, 1) for s in spin_names]

    H_loc = U * n('up', 0) * n('do', 0) - (mu + U/2) * (n('up', 0) + n('do', 0))
    
    from triqs_xca.block_sparse_solver import BlockSparseSolver

    S = BlockSparseSolver(H_loc, beta, w_max, eps, gf_struct=gf_struct)
    S.order = order

    Delta_w = make_gf_dlr_imfreq(S.Delta_tau)

    for spin in spin_names:
        Delta_w[spin] << SemiCircular(1.0)
        S.Delta_tau[spin] << make_gf_dlr_imtime(Delta_w[spin])
    
    for iter in range(1, dmft_maxiter+1):
        
        S.solve(max_order=order, tol=ppsc_tol, maxiter=ppsc_maxiter, mix=1.0, dlr_polefitting=True, verbose=True)

        n_up_exp = S.expectation_value(n('up', 0)).real
        print(f'<n_up> = {n_up_exp:2.4f}')

        dmft_diff = np.max(np.abs((S.G_tau['up'] - S.Delta_tau['up']).data))

        if mpi.is_master_node():
            print('='*72)
            print(f'DMFT: iter = {iter} diff = {dmft_diff:2.2E}')
            print('='*72)
            print()

        if dmft_diff < dmft_tol or iter == dmft_maxiter: break
        
        for spin in spin_names:
            S.Delta_tau[spin] << S.G_tau[spin]

    S.H_loc = H_loc

    return S


if __name__ == '__main__':

    S = solve_one_spinful_fermion_triqs_solver()
    S = solve_one_spinful_fermion_block_sparse_solver()


