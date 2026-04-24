
"""
Solve Hubbard model on the Bethe graph with DMFT using the XCA impurity solver at 1st order.

Using both the TRIQS solver and the block sparse solver.
"""

from matplotlib.pylab import beta
import numpy as np

import triqs.utility.mpi as mpi

from h5 import HDFArchive

from triqs.operators import n

from triqs.gf import make_gf_dlr_imtime, make_gf_dlr_imfreq, make_gf_dlr, SemiCircular

from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti


def solve_one_spinful_fermion_triqs_solver(
        beta=25.0, U=6.0, mu=0.0, 
        order=1, eps=1e-12, w_max=10.0,
        dmft_maxiter=40, dmft_tol=1e-9,
        ppsc_maxiter=40, ppsc_tol=1e-9,
        ):

    spin_names = ['up', 'do']
    gf_struct = [(s, 1) for s in spin_names]

    H_loc = U * n('up', 0) * n('do', 0) - (mu + U/2) * (n('up', 0) + n('do', 0))
    
    from triqs_xca.triqs_solver import TriqsSolver
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
        beta=10.0, U=6.0, mu=0.0, order=1,
        eps=1e-10, w_max=6.0,
        dmft_maxiter=20, dmft_tol=1e-5, ppsc_maxiter=20, ppsc_tol=1e-8,
        S_old=None, delta_mix=1.0, ppsc_mix=1.0,
        block_sparse_solver=True,
        plot_each_iter=False,
        ):

    spin_names = ['up', 'do']
    gf_struct = [(s, 1) for s in spin_names]

    H_loc = U * n('up', 0) * n('do', 0) - (mu + U/2) * (n('up', 0) + n('do', 0))
    
    if block_sparse_solver:
        from triqs_xca.block_sparse_solver import BlockSparseSolver
        S = BlockSparseSolver(H_loc, beta, w_max, eps, gf_struct=gf_struct)
    else:
        from triqs_xca.triqs_solver import TriqsSolver
        S = TriqsSolver(beta=beta, gf_struct=gf_struct, eps=eps, w_max=w_max)

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

            for t in S.Delta_tau[spin].mesh:
                S.Delta_tau[spin] << S_old.Delta_tau[spin]

        if block_sparse_solver:
            for bidx, g in S_old.G:
                assert(S.G[bidx].mesh == S_old.G[bidx].mesh)
                S.G[bidx] << g
            S.eta = S_old.eta0 + S_old.eta - S.eta0 # Adjust eta keeping in mind that it is relative to eta0 (that can change between runs)
        else:
            S.S.G_iaa = S_old.S.G_iaa.copy()

    if plot_each_iter:
        Delta_iaa, Delta_yaa = get_Delta_ir(S)
        G_iaa, G_yaa = get_G_ir(S)

        plt.figure(figsize=(6, 10))
        subp_org = [6, 1, 1]

        subp = subp_org.copy()

        plt.subplot(*subp); subp[-1] += 1
        c = plt.plot([], [])[0].get_color()
        plt.title(f'U = {U}, beta = {beta}, order = {order}')
        oplotr(S.Delta_tau, 'o', label=None, color='r')
        plt.plot(ir.tau_i, Delta_iaa[:, 0, 0].real, '.-', color='r')
        plt.grid(True)

        plt.subplot(*subp); subp[-1] += 1
        if block_sparse_solver:
            oplotr(S.G, label=None, color=c)
        else:
            plt.plot(S.S.tau_i, S.S.G_iaa.reshape(len(S.S.tau_i), -1), color=c)
        plt.plot(ir.tau_i, G_iaa.real, '.-', color='r')
        plt.grid(True)

        plt.subplot(*subp); subp[-1] += 1
        plt.plot(np.abs(Delta_yaa.real), 'o', color='r')
        plt.ylabel(r'$\Delta(\tau)$ IR coeffs.')
        plt.xlabel(r'IR order')
        plt.semilogy([], [])
        plt.grid(True)

        plt.subplot(*subp); subp[-1] += 1
        plt.plot(np.abs(G_yaa.real), 'o', color='r')
        plt.ylabel(r'$G(\tau)$ IR coeffs.')
        plt.xlabel(r'IR order')
        plt.semilogy([], [])
        plt.grid(True)

    for iter in range(1, dmft_maxiter+1):

        if block_sparse_solver:        
            S.solve(max_order=order, tol=ppsc_tol, delta_tol=1e-5, maxiter=ppsc_maxiter, mix=ppsc_mix, dlr_polefitting=True, verbose=True, normalization='classic')
        else:
            S.solve(h_int=H_loc, order=order, tol=ppsc_tol, maxiter=ppsc_maxiter, update_eta_exact=False, compress_hybridization=False)

        if block_sparse_solver:
            S.n_up_exp = S.expectation_value(n('up', 0)).real
            S.n_do_exp = S.expectation_value(n('do', 0)).real
            S.docc_exp = S.expectation_value(n('up', 0)*n('do', 0)).real
        else:
            S.n_up_exp = S.S.get_expectation_value(n('up', 0)).real
            S.n_do_exp = S.S.get_expectation_value(n('do', 0)).real
            S.docc_exp = S.S.get_expectation_value(n('up', 0)*n('do', 0)).real

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

        if plot_each_iter:
            Delta_iaa, Delta_yaa = get_Delta_ir(S)
            G_iaa, G_yaa = get_G_ir(S)

            subp = subp_org.copy()
            plt.subplot(*subp); subp[-1] += 1
            oplotr(S.Delta_tau, label=None, color=c)
            plt.plot(ir.tau_i, Delta_iaa[:, 0, 0].real, '.-', color=c)

            plt.subplot(*subp); subp[-1] += 1
            if block_sparse_solver:
                oplotr(S.G, label=None, color=c)
            else:
                plt.plot(S.S.tau_i, S.S.G_iaa.reshape(len(S.S.tau_i), -1), color=c)
            plt.plot(ir.tau_i, G_iaa.real, '.-', color=c)

            plt.subplot(*subp); subp[-1] += 1
            plt.plot(np.abs(Delta_yaa.real), '.', color=c)
            plt.ylabel(r'$\Delta(\tau)$ IR coeffs.')
            plt.xlabel(r'IR order')
            plt.semilogy([], [])

            plt.subplot(*subp); subp[-1] += 1
            plt.plot(np.abs(G_yaa.real), '.', color=c)
            plt.ylabel(r'$G(\tau)$ IR coeffs.')
            plt.xlabel(r'IR order')
            plt.semilogy([], [])
            plt.grid(True)

            plt.subplot(*subp); subp[-1] += 1
            plt.plot(iter, np.max(np.abs(Delta_yaa[-1])), 'x', color=c)
            plt.plot(iter, np.max(np.abs(G_yaa[-1])), '+', color=c)
            plt.ylabel(r'Highest IR coeff')
            plt.xlabel(r'DMFT iteration')
            plt.semilogy([], [])
            plt.grid(True)

            if False:
                plt.subplot(*subp); subp[-1] += 1
                plt.plot(iter, S.Z_pre_norm, 'x', color=c)
                plt.plot(iter, S.Z_post_norm, '+', color=c)
                plt.ylabel(r'Partition function Z')
                plt.xlabel(r'DMFT iteration')
                plt.grid(True)

            plt.subplot(*subp); subp[-1] += 1
            plt.plot(iter, S.eta, 'x', color=c)
            plt.ylabel(r'eta')
            plt.xlabel(r'DMFT iteration')
            plt.grid(True)

        if dmft_diff < dmft_tol or iter == dmft_maxiter: break
        
        # -- Impose spin SU(2) symmetry
        S.G_tau['up'] << 0.5 * (S.G_tau['up'] + S.G_tau['do'])
        S.G_tau['do'] << S.G_tau['up']

        for spin in spin_names:
            S.Delta_tau[spin] << delta_mix * S.G_tau[spin] + (1 - delta_mix) * S.Delta_tau[spin]
            #S.Delta_tau[spin].data[:] = delta_mix * S.G_tau[spin].data + (1 - delta_mix) * S.Delta_tau[spin].data

    if plot_each_iter:
        Delta_iaa, Delta_yaa = get_Delta_ir(S)
        G_iaa, G_yaa = get_G_ir(S)

        c = 'm'
        subp = subp_org.copy()
        plt.subplot(*subp); subp[-1] += 1
        oplotr(S.Delta_tau, label=None, color=c)
        plt.plot(ir.tau_i, Delta_iaa[:, 0, 0].real, '.-', color=c)

        plt.subplot(*subp); subp[-1] += 1
        if block_sparse_solver:
            oplotr(S.G, label=None, color=c)
        else:
            plt.plot(S.S.tau_i, S.S.G_iaa.reshape(len(S.S.tau_i), -1), color=c)
        plt.plot(ir.tau_i, G_iaa.real, '.-', color=c)

        plt.subplot(*subp); subp[-1] += 1
        plt.plot(np.abs(Delta_yaa.real), 'x', color=c, alpha=.5)
        plt.ylabel(r'$\Delta(\tau)$ IR coeffs.')
        plt.xlabel(r'IR order')
        plt.semilogy([], [])

        plt.subplot(*subp); subp[-1] += 1
        plt.plot(np.abs(G_yaa.real), 'x', color=c, alpha=.5)
        plt.ylabel(r'$G(\tau)$ IR coeffs.')
        plt.xlabel(r'IR order')
        plt.semilogy([], [])
        plt.grid(True)
        plt.tight_layout()

    S.dmft_iter = iter
    S.dmft_diff = dmft_diff

    return S


class ListDummy():
    def __init__(self, data): self.data = data
    def __getattr__(self, key): return np.array([ getattr(d, key) for d in self.data ])
    

if __name__ == '__main__':

    #S = solve_one_spinful_fermion_triqs_solver()

    beta = 100.0
    #Us = np.linspace(3., 6., num=20)
    #Us = np.linspace(3., 4., num=20) # For order 1 the transition/crossover is here for beta=25.
    #Us = np.linspace(3.3, 3.7, num=20) # For order 1 the transition/crossover is here for beta=25.
    Us = np.linspace(3.4, 3.8, num=20) # For order 1 the transition/crossover is here for beta=25.

    orders_Us = [(1, Us), (1, Us[::-1])]

    rs = []
    S = None
    for order, Us in orders_Us:
        Ss = []
        for idx, U in enumerate(Us):
                
            S = solve_one_spinful_fermion_block_sparse_solver(
                U=U, beta=beta, order=order, S_old=S, block_sparse_solver=True,
                eps=1e-10, w_max=6.0,
                dmft_maxiter=40, dmft_tol=1e-5, ppsc_maxiter=20, ppsc_tol=1e-8,
                delta_mix=1.0, ppsc_mix=1.0)

            Ss.append(S)

        r = ListDummy(Ss)
        rs.append(r)

        if mpi.is_master_node():
            print(f'U = {r.U}')
            print(f'<n_up> = {r.n_up_exp}')
            print(f'<n_do> = {r.n_do_exp}')
            print(f'<n_up n_do> = {r.docc_exp}')


    #plt.tight_layout()
    #plt.show(); exit()

    if mpi.is_master_node():
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 9))
        subp = [3, 1, 1]

        plt.subplot(*subp); subp[-1] += 1
        for r in rs:
            plt.plot(r.U, r.docc_exp, 'o-', label=f'order = {r.order[0]}, $\\beta$ = {r.beta[0]}')

        #plt.ylim(bottom=0)
        plt.legend(loc='best')
        plt.xlabel('U')
        plt.ylabel(r'$\langle n_\uparrow n_\downarrow \rangle$')

        plt.subplot(*subp); subp[-1] += 1
        for r in rs:
            plt.plot(r.U, r.dmft_iter, 'o-')
        plt.xlabel('U')
        plt.ylabel('DMFT iterations')

        plt.subplot(*subp); subp[-1] += 1
        for r in rs:
            plt.plot(r.U, r.dmft_diff, 'o-')
        plt.xlabel('U')
        plt.ylabel('DMFT sc error')
        plt.semilogy([], [])
        plt.grid(True)


        plt.tight_layout()
        plt.show()