
"""
Solve Hubbard model on the Bethe graph with DMFT using the XCA impurity solver at 1st order.

Using both the TRIQS solver and the block sparse solver.
"""

import numpy as np

import triqs.utility.mpi as mpi

from h5 import HDFArchive

from triqs.gfs import make_gf_dlr_imtime, make_gf_dlr_imfreq, make_gf_dlr, SemiCircular, Gf


def solve_one_spinful_fermion_block_sparse_solver(
        beta=10.0, U=6.0, mu=0.0, order=1,
        eps=1e-10, w_max=6.0,
        dmft_maxiter=20, dmft_tol=1e-5, 
        ppsc_maxiter=20, ppsc_tol=1e-8,
        S_old=None, delta_mix=1.0, ppsc_mix=1.0,
        hyb_tol=1e-8, dlr_symmetrize=False,
        ):
    
    spin_names = ['up', 'do']
    gf_struct = [(s, 1) for s in spin_names]

    from triqs.operators import n
    H_loc = U * n('up', 0) * n('do', 0) - (mu + U/2) * (n('up', 0) + n('do', 0))
    
    from triqs_xca.block_sparse_solver import BlockSparseSolver
    S = BlockSparseSolver(H_loc, beta, w_max, eps, gf_struct=gf_struct, dlr_symmetrize=dlr_symmetrize)

    S.order = order
    S.H_loc = H_loc
    S.U = U
    S.beta = beta
    S.mu = mu

    if S_old is None:
        Delta_w = make_gf_dlr_imfreq(S.Delta_tau)
        for spin in spin_names:
            Delta_w[spin] << SemiCircular(1.0)
            S.Delta_tau[spin] << make_gf_dlr_imtime(Delta_w[spin])
    else:
        for spin in spin_names:
            assert(S.Delta_tau[spin].mesh == S_old.Delta_tau[spin].mesh)
            S.Delta_tau[spin] << S_old.Delta_tau[spin]
        for bidx, g in S_old.G:
            assert(S.G[bidx].mesh == S_old.G[bidx].mesh)
            S.G[bidx] << g
        S.eta = S_old.eta0 + S_old.eta - S.eta0 # Adjust eta keeping in mind that it is relative to eta0 (that can change between runs)

    for iter in range(1, dmft_maxiter+1):

        S.solve(max_order=order, tol=ppsc_tol, hyb_tol=hyb_tol,
                maxiter=ppsc_maxiter, mix=ppsc_mix,
                verbose=True, normalization='classic')

        S.n_up_exp = S.expectation_value(n('up', 0)).real
        S.n_do_exp = S.expectation_value(n('do', 0)).real
        S.docc_exp = S.expectation_value(n('up', 0)*n('do', 0)).real

        if mpi.is_master_node():
            print(f'<n_up> = {S.n_up_exp:2.4f}')
            print(f'<n_do> = {S.n_do_exp:2.4f}')
            print(f'<n_up n_do> = {S.docc_exp:2.4f}')

        dmft_diff = np.max(np.abs((S.G_tau['up'] - S.Delta_tau['up']).data))

        if mpi.is_master_node():
            print('='*72)
            print(f'DMFT: iter = {iter} diff = {dmft_diff:2.2E}')
            print('='*72)
            print()

        if dmft_diff < dmft_tol or iter == dmft_maxiter: break
        
        for spin in spin_names:
            S.Delta_tau[spin] << delta_mix * S.G_tau[spin] + (1 - delta_mix) * S.Delta_tau[spin]

    S.dmft_iter = iter
    S.dmft_diff = dmft_diff

    return S


class ListDummy():
    def __init__(self, data): self.data = data
    def __getattr__(self, key): return np.array([ getattr(d, key) for d in self.data ])
    

def calc_linear_sweep_both_ways(Us, beta=100.0, order=1):

    orders_Us = [(order, Us), (order, Us[::-1])]

    S = None
    Sss = []
    for order, Us in orders_Us:
        Ss = []
        for idx, U in enumerate(Us):
                
            S = solve_one_spinful_fermion_block_sparse_solver(
                U=U, beta=beta, order=order, S_old=S, 
                eps=1e-10, w_max=6.0,
                dmft_maxiter=40, dmft_tol=1e-5, ppsc_maxiter=20, ppsc_tol=1e-8,
                delta_mix=1.0, ppsc_mix=1.0)

            Ss.append(S)

        Sss.append(Ss)

        if mpi.is_master_node():
            r = ListDummy(Ss)
            print(f'U = {r.U}')
            print(f'<n_up> = {r.n_up_exp}')
            print(f'<n_do> = {r.n_do_exp}')
            print(f'<n_up n_do> = {r.docc_exp}')

    if mpi.is_master_node():
        filename = f'data_hubbard_bethe_dmft_order{order}_beta{beta}_Umin{Us.min()}_Umax{Us.max()}.h5'
        print(f'-> Storing: {filename}')
        with HDFArchive(filename, 'w') as A:
            A['Sss'] = Sss


if __name__ == '__main__':

    # -- 1st order transition already at order 1.

    # beta = 50
    calc_linear_sweep_both_ways(Us=np.linspace(3.4, 3.6, num=20), beta=50.0, order=1)

    # beta = 100
    #calc_linear_sweep_both_ways(Us=np.linspace(3.4, 3.8, num=20), beta=100.0, order=1)

    # beta = 200
    #calc_linear_sweep_both_ways(Us=np.linspace(3.4, 3.8, num=20), beta=200.0, order=1)
