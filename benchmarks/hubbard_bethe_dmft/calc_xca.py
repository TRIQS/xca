
"""
Solve Hubbard model on the Bethe graph with DMFT using the XCA impurity solver at 1st order.

Using both the TRIQS solver and the block sparse solver.
"""

import numpy as np

import triqs.utility.mpi as mpi

from h5 import HDFArchive

from triqs.operators import n

from triqs.gf import make_gf_dlr_imtime, make_gf_dlr_imfreq, make_gf_dlr, SemiCircular, Gf
from triqs_xca.block_sparse_solver import atomic_pseudo_particle_greens_function


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


def solve_one_spinful_fermion_block_sparse_solver_debug(
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


def solve_one_spinful_fermion_block_sparse_solver(
        beta=10.0, U=6.0, mu=0.0, order=1,
        eps=1e-10, w_max=6.0,
        dmft_maxiter=20, dmft_tol=1e-5, 
        ppsc_maxiter=20, ppsc_tol=1e-8,
        S_old=None, delta_mix=1.0, ppsc_mix=1.0,
        dlr_polefitting=True, delta_tol=1e-8, dlr_symmetrize=False,
        ):

    spin_names = ['up', 'do']
    gf_struct = [(s, 1) for s in spin_names]

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

        S.solve(max_order=order, tol=ppsc_tol, delta_tol=delta_tol,
                maxiter=ppsc_maxiter, mix=ppsc_mix, dlr_polefitting=dlr_polefitting, 
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
        
        # -- Impose spin SU(2) symmetry
        #S.G_tau['up'] << 0.5 * (S.G_tau['up'] + S.G_tau['do'])
        #S.G_tau['do'] << S.G_tau['up']

        for spin in spin_names:
            S.Delta_tau[spin] << delta_mix * S.G_tau[spin] + (1 - delta_mix) * S.Delta_tau[spin]

    S.dmft_iter = iter
    S.dmft_diff = dmft_diff

    return S


def solve_one_spinful_fermion_block_sparse_solver_adiabatic(
        beta=10.0, U=6.0, mu=0.0, order=1,
        eps=1e-10, w_max=6.0,
        dmft_maxiter=20, dmft_tol=1e-5, 
        ppsc_maxiter=20, ppsc_tol=1e-8,
        delta_mix=1.0, ppsc_mix=1.0,
        S_old=None, dx_vec=None, dx=0.1,
        ):

    if False:
        spin_names = ['up', 'do']
        gf_struct = [(s, 1) for s in spin_names]
        H_loc = U * n('up', 0) * n('do', 0) - (mu + U/2) * (n('up', 0) + n('do', 0))
        
        from triqs_xca.block_sparse_solver import BlockSparseSolver
        S = BlockSparseSolver(H_loc, beta, w_max, eps, gf_struct=gf_struct)

        S.order = order
        S.H_loc = H_loc
        S.U = U
        S.beta = beta
        S.mu = mu

    if S_old is None:
        # -- Run one fwd iteration solve to get started
        S_old = solve_one_spinful_fermion_block_sparse_solver(
            beta=beta, U=U, mu=mu, order=order,
            eps=eps, w_max=w_max,
            dmft_maxiter=dmft_maxiter, dmft_tol=dmft_tol, 
            ppsc_maxiter=ppsc_maxiter, ppsc_tol=ppsc_tol,
            S_old=None, delta_mix=delta_mix, ppsc_mix=ppsc_mix, dlr_polefitting=False, dlr_symmetrize=False)

    # -- TODO: Implement .copy() for BlockSparseSolver to avoid this hack        
    from triqs_xca.block_sparse_solver import BlockSparseSolver
    d = S_old.__reduce_to_dict__()
    S = BlockSparseSolver.__factory_from_dict__('Name?', d)


    class ConverterIRDLR:

        def __init__(self, S, n_cut):
            self.n_cut = n_cut
            self.mesh_tau = S.mesh_tau

            from pydlr.pyir import IrDlr
            m = self.mesh_tau
            self.ir = IrDlr(lamb=m.w_max*m.beta, eps=m.eps)
            self.tau_i = self.ir.get_tau(m.beta)
            self.n_ir_max = self.ir.T_yx.shape[0]

            if n_cut is not None:
                assert(n_cut <= self.n_ir_max//2)


        def to_ir(self, g_tau, ph_sym=False):

            g_dlr = make_gf_dlr(g_tau)
            
            #g_i = np.vectorize(g_dlr.__call__, otypes=[complex], signature='(n)->(n,m,m)')(self.tau_i.real)

            g_i = np.zeros([len(self.ir)] + list(g_dlr.target_shape), dtype=complex)
            for idx, tau in enumerate(self.tau_i):
                g_i[idx] = g_dlr(tau.real)

            g_x = self.ir.dlr_from_tau(g_i)
            g_y = self.ir.ir_from_dlr(g_x)

            if False:
                print(f'g_i.shape = {g_i.shape}')
                print(f'g_y.shape = {g_y.shape}')

            if ph_sym:
                g_y = g_y[0::2]
                if False: print(f'g_y.shape = {g_y.shape} (ph_sym=True)')
                
            return g_y


        def from_ir(self, g_y_in, ph_sym=False):
            
            n_ir = g_y_in.shape[0]

            if False:
                print(f'n_ir = {n_ir}')
                print(f'n_ir_max = {self.n_ir_max}')

            assert n_ir <= self.n_ir_max, f'IR order of input {n_ir} exceeds IR basis size {self.n_ir_max}'

            g_y = np.zeros([self.n_ir_max] + list(g_y_in.shape[1:]), dtype=g_y_in.dtype)

            if ph_sym:
                g_y[0:2*n_ir:2] = g_y_in
            else:
                g_y[:n_ir] = g_y_in

            G_tau = Gf(mesh=self.mesh_tau, target_shape=g_y.shape[1:])

            tau = np.array([ float(t) for t in self.mesh_tau ])

            #g_x = self.ir.dlr_from_ir(g_y)
            #G_tau.data[:] = self.ir.eval_dlr_tau(g_x, tau, self.mesh_tau.beta)

            G_tau.data[:] = self.ir.eval_ir_tau(g_y, tau, self.mesh_tau.beta)

            return G_tau


        def get_x(self, S):

            g_y = self.to_ir(S.G_tau['up'], ph_sym=True)
            G0_y = self.to_ir(S.G['0'])
            G2_y = self.to_ir(S.G['2'])

            if self.n_cut is not None:            
                g_y = g_y[:self.n_cut]
                G0_y = G0_y[:self.n_cut*2]
                G2_y = G2_y[:self.n_cut*2]

            if False:
                print(f'n_cut = {self.n_cut}')
                print(f'n_ir = {len(self.ir)}')
                print(f'n_ir//2 = {len(self.ir)//2}')
                print(f'g_y.shape = {g_y.shape}')
                print(f'G0_y.shape = {G0_y.shape}')
                print(f'G2_y.shape = {G2_y.shape}')

            # U, eta, Delta_up, G0, G1
            x = np.array([], dtype=float)
            x = np.append(x, [S.U.real, S.eta.real])
            x = np.append(x, g_y.real)
            x = np.append(x, G0_y.real)
            x = np.append(x, G2_y.real)

            return x
        

        def set_x(self, S, x):

            S.U = x[0]
            S.eta = x[1]

            terms = [
                (S.Delta_tau['up'].data, True),
                #S.Delta_tau['do'].data,
                (S.G['0'].data, False),
                #S.G['1'].data,
                (S.G['2'].data, False),
                #S.G['3'].data,
                ]

            sidx = 2
            for term, ph_sym in terms:

                if self.n_cut is not None:
                    n_ir = self.n_cut if ph_sym else self.n_cut * 2
                else:
                    n_ir = self.n_ir_max//2 if ph_sym else self.n_ir_max

                shape = [n_ir] + list(term.shape[1:])
                N = np.prod(shape)
                y = x[sidx : sidx + N].reshape(shape)
                term[:] = self.from_ir(y, ph_sym=ph_sym).data
                sidx += N

            S.Delta_tau['do'].data[:] = S.Delta_tau['up'].data
            S.G['1'].data[:] = S.G['0'].data
            S.G['3'].data[:] = S.G['2'].data

            if True:
                # Update H_loc and atom_diag for change in U
                S.H_loc = S.U * n('up', 0) * n('do', 0) - (S.mu + S.U/2) * (n('up', 0) + n('do', 0))
                from triqs.atom_diag import AtomDiag
                S.atom_diag = AtomDiag(S.H_loc, S.fundamental_operators) 
                S.G0, S.eta0 = atomic_pseudo_particle_greens_function(S.atom_diag, S.beta, S.mesh_tau)

                # FIXME! Get ito from mesh_tau
                from triqs_xca.pycppdlr import build_dlr_rf
                from triqs_xca.pycppdlr import ImTimeOps
                from triqs_xca.dlr_dyson_ppsc import DysonItPPSC
                ito = ImTimeOps(S.w_max * S.beta, build_dlr_rf(S.w_max * S.beta, S.eps, S.dlr_symmetrize), symmetrize=S.dlr_symmetrize) 
                S.dysons = [DysonItPPSC(S.beta, ito, G0_block.data) for _, G0_block in S.G0]
                    



    def get_x_dlr(S):

        if False:
            print('--> get_x')

            g_up_y = conv.to_ir(S.G_tau['up'], ph_sym=True)

            g_up_y = g_up_y[:25] # Cut off for testing

            print(f'g_up_y.shape = {g_up_y.shape}')
            G_up_tau_ref = conv.from_ir(g_up_y, ph_sym=True)
            print(f'g_up_y = {g_up_y}')

            diff = np.max(np.abs(S.G_tau['up'].data - G_up_tau_ref.data))
            print(f'diff = {diff:2.2E}')
            np.testing.assert_array_almost_equal(S.G_tau['up'].data, G_up_tau_ref.data)
            exit()

        # U, eta, Delta_up, G0, G1
        x = np.array([], dtype=float)

        x = np.append(x, [S.U.real, S.eta.real])
        #x = np.append(x, [S.U.real])

        diff_up_do = np.max(np.abs(S.G_tau["up"].data - S.G_tau["do"].data))
        diff_0_1 = np.max(np.abs(S.G["0"].data - S.G["1"].data))
        diff_2_3 = np.max(np.abs(S.G["2"].data - S.G["3"].data))

        diff_tol = 1e-8

        if diff_up_do > diff_tol or diff_0_1 > diff_tol or diff_2_3 > diff_tol:
            print('WARNING: Significant spin or block asymmetry detected in get_x:')
            print(f'G_tau up/do diff = {diff_up_do:2.2E}')
            print(f'G[0]/G[1] diff = {diff_0_1:2.2E}')
            print(f'G[2]/G[3] diff = {diff_2_3:2.2E}')

        x = np.append(x, S.G_tau['up'].data.real.flatten())
        #x = np.append(x, S.G_tau['do'].data.real.flatten())

        if S.dlr_symmetrize:
            # Check symmetric tau grid
            tau_i = np.array([ float(t) for t in S.G_tau['up'].mesh])
            n_tau = len(tau_i)
            beta = S.G_tau['up'].mesh.beta
            print(f'tau_i = {tau_i}')
            np.testing.assert_array_almost_equal(tau_i[:n_tau//2], beta - tau_i[n_tau//2:][::-1])
            exit()

        # DEBUG!
        #x = np.append(x, S.Delta_tau['up'].data.real.flatten())
        #x = np.append(x, S.Delta_tau['do'].data.real.flatten())
        #np.testing.assert_array_almost_equal(S.G_tau['do'].data, S.G_tau['up'].data)

        x = np.append(x, S.G['0'].data.real.flatten())
        #x = np.append(x, S.G['1'].data.real.flatten())
        #np.testing.assert_array_almost_equal(S.G['1'].data, S.G['0'].data)
        x = np.append(x, S.G['2'].data.real.flatten())
        #x = np.append(x, S.G['3'].data.real.flatten())
        #np.testing.assert_array_almost_equal(S.G['3'].data, S.G['2'].data)


        return x

    def set_x_dlr(S, x):

        S.U = x[0]
        #print(f'Set U = {S.U:2.4f}')
        S.eta = x[1]
        sidx = 2

        #sidx = 1

        terms = [
            S.Delta_tau['up'].data,
            #S.Delta_tau['do'].data,
            S.G['0'].data,
            #S.G['1'].data,
            S.G['2'].data,
            #S.G['3'].data,
            ]

        for term in terms:
            shape = term.shape
            N = np.prod(shape)
            term[:] = x[sidx : sidx + N].reshape(shape)
            sidx += N

        S.Delta_tau['do'].data[:] = S.Delta_tau['up'].data
        S.G['1'].data[:] = S.G['0'].data
        S.G['3'].data[:] = S.G['2'].data

        if True:
            # Update H_loc and atom_diag for change in U
            S.H_loc = S.U * n('up', 0) * n('do', 0) - (S.mu + S.U/2) * (n('up', 0) + n('do', 0))
            from triqs.atom_diag import AtomDiag
            S.atom_diag = AtomDiag(S.H_loc, S.fundamental_operators) 
            S.G0, S.eta0 = atomic_pseudo_particle_greens_function(S.atom_diag, S.beta, S.mesh_tau)

            # FIXME! Get ito from mesh_tau
            from triqs_xca.pycppdlr import build_dlr_rf
            from triqs_xca.pycppdlr import ImTimeOps
            from triqs_xca.dlr_dyson_ppsc import DysonItPPSC
            ito = ImTimeOps(S.w_max * S.beta, build_dlr_rf(S.w_max * S.beta, S.eps, S.dlr_symmetrize), symmetrize=S.dlr_symmetrize) 
            S.dysons = [DysonItPPSC(S.beta, ito, G0_block.data) for _, G0_block in S.G0]
    

    if False:
        set_x, get_x = set_x_dlr, get_x_dlr
    else:
        conv = ConverterIRDLR(S, n_cut=None)
        set_x, get_x = conv.set_x, conv.get_x

    # set direction for initial guess
    x_old = get_x(S)

    if dx_vec is None:
        dx_vec = np.zeros_like(x_old)
        dx_vec[0] = dx # Change U by dx

    x0 = x_old + dx_vec
    #x0 = x_old

    def root_function(x):
        
        set_x(S, x)
        
        S.solve(max_order=order, tol=1e-20, maxiter=1,
                dlr_polefitting=False, verbose=False, normalization='classic')
        
        x_new = get_x(S)

        root = x_new - x
        diff = np.max(np.abs(root))
        #print(f'diff = {diff:2.2E}')
        
        norm = np.linalg.norm(x_new - x_old)
        dist = norm - dx # Ensure dx distance from previous sol
        root[0] = dist # DEBUG

        #print(f'dx = {dx}, root[0] = {root[0]:2.2E}, root[1] = {root[1]:2.2E}')
        #root[0] = x[0] - (x_old[0] + dx) # Ensure dx distance from previous sol
        
        #print(f'max(abs(root)) = {np.max(np.abs(root)):2.2E}')
        return root

    if True:        
        from scipy.optimize import root
        
        sol = root(
            root_function, x0, tol=1e-6,
            #method='df-sane', options=dict(sigma_eps=1e-3, sigma_0=0.01, disp=True, maxfev=200, line_search='cheng'))
            method='hybr', options=dict(maxfev=300))
        
        print(sol)
        x = sol.x
        set_x(S, x)
        dx_vec = x - x_old

        S.dmft_iter = sol.nfev
        S.dmft_diff = np.max(np.abs(sol.fun))

    else:
        x = x0

    if FAlse:
        root = root_function(x)
        print(f'Plotting: max(abs(root)) = {np.max(np.abs(root)):2.2E}')

        from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti
        subp = [4, 2, 1]

        plt.figure(figsize=(6, 12))
        plt.subplot(*subp); subp[-1] += 1
        oplotr(S_old.G, 'x')

        plt.subplot(*subp); subp[-1] += 1
        oplotr(S.G, 'x')

        plt.subplot(*subp); subp[-1] += 1
        oploti(S_old.G, 'x')

        plt.subplot(*subp); subp[-1] += 1
        oploti(S.G, 'x')

        plt.subplot(*subp); subp[-1] += 1
        oplotr(S_old.G - S.G, 'x')

        plt.subplot(*subp); subp[-1] += 1
        oploti(S_old.G - S.G, 'x')

        plt.subplot(*subp); subp[-1] += 1
        oplotr(S.Delta_tau - S.G_tau, 'x')

        plt.subplot(*subp); subp[-1] += 1
        oploti(S.Delta_tau - S.G_tau, 'x')

        plt.tight_layout()
        plt.show()

        #exit()

    S.n_up_exp = S.expectation_value(n('up', 0)).real
    S.n_do_exp = S.expectation_value(n('do', 0)).real
    S.docc_exp = S.expectation_value(n('up', 0)*n('do', 0)).real

    return S, dx_vec


class ListDummy():
    def __init__(self, data): self.data = data
    def __getattr__(self, key): return np.array([ getattr(d, key) for d in self.data ])
    

def calc_linear_sweep_both_ways(Us, beta=100.0, order=1):

    orders_Us = [(1, Us), (1, Us[::-1])]

    S = None
    Sss = []
    for order, Us in orders_Us:
        Ss = []
        for idx, U in enumerate(Us):
                
            S = solve_one_spinful_fermion_block_sparse_solver(
                U=U, beta=beta, order=order, S_old=S, 
                #block_sparse_solver=True,
                eps=1e-10, w_max=6.0,
                dmft_maxiter=40, dmft_tol=1e-5, ppsc_maxiter=20, ppsc_tol=1e-8,
                delta_mix=1.0, ppsc_mix=1.0, dlr_polefitting=False)

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
        with HDFArchive(filename, 'w') as A:
            A['Sss'] = Sss


def calc_adiabatic(U0, dx, Umax=None, Umin=None, beta=100.0, order=1, max_adiabatic_iter=2, root_tol=1e-6):

    S = None
    dx_vec = None
    Ss = []

    from adiabatic_delta import solve_one_spinful_fermion_block_sparse_solver_adiabatic_delta

    for iter in range(max_adiabatic_iter):
        S_new, dx_vec_new = solve_one_spinful_fermion_block_sparse_solver_adiabatic_delta(
            U=U0, beta=beta, order=order,
            eps=1e-10, w_max=6.0,
            dmft_maxiter=100, dmft_tol=1e-9, ppsc_maxiter=40, ppsc_tol=1e-9,
            root_tol=root_tol,
            delta_mix=1.0, ppsc_mix=1.0, dx=dx, dx_vec=dx_vec, S_old=S)

        print(f'Adiabatic iter = {iter} U = {S_new.U:2.4f} dU = {dx_vec_new[0]:2.2E} dmft_diff = {S_new.dmft_diff:2.2E}')

        if dx == 0.:
            print(f'WARNING: dx = 0. No adiabatic change. Stopping iteration.')
            break

        if S_new.dmft_diff > root_tol:
            #print(f'WARNING: Adiabatic step not converged giving up')
            dx *= 0.1
            dx_vec *= 0.1
            print(f'WARNING: Adiabatic step not converged reducing dx to {dx:2.2E} and trying again')
            #print(f'norm(dx_vec) = {np.linalg.norm(dx_vec_new):2.2E}')

            if np.abs(dx) < root_tol:
                print(f'WARNING: dx = {dx:2.2E} below root_tol = {root_tol:2.2E}. Stopping iteration.')
                break

            # Skip the current S_new and dx_vec_new and try again with smaller dx
            continue

        S = S_new
        dx_vec = dx_vec_new
        Ss.append(S)

        if Umax is not None and S.U >= Umax:
            print(f'Reached U = {S.U:2.4f} >= Umax = {Umax:2.4f}. Stopping adiabatic iteration.')
            break
        if Umin is not None and S.U <= Umin:
            print(f'Reached U = {S.U:2.4f} <= Umin = {Umin:2.4f}. Stopping adiabatic iteration.')
            break

    if mpi.is_master_node():
        filename = f'data_hubbard_bethe_dmft_order{order}_beta{beta}_U0{U0}.h5'
        print(f'Saving: {filename}')
        with HDFArchive(filename, 'w') as A:
            A['Sss'] = [Ss]


if __name__ == '__main__':

    # -- 1st order transition already at order 1.
    #calc_linear_sweep_both_ways(Us=np.linspace(3.4, 3.6, num=80), beta=50.0, order=1)
    #calc_linear_sweep_both_ways(Us=np.linspace(3.4, 3.6, num=20), beta=50.0, order=1)

    #calc_linear_sweep_both_ways(Us=np.linspace(3.4, 3.8, num=20), beta=100.0, order=1)

    #calc_adiabatic(U0=3.4, Umax=3.8, dx=0.01, beta=100.0, order=1, max_adiabatic_iter=200)
    #calc_adiabatic(U0=3.8, Umin=3.4, dx=-0.01, beta=100.0, order=1, max_adiabatic_iter=1)

    #calc_linear_sweep_both_ways(Us=np.linspace(3.4, 3.8, num=20), beta=200.0, order=1)
    calc_adiabatic(U0=3.4, Umax=3.8, dx=0.01, beta=200.0, order=1, max_adiabatic_iter=200)
    #calc_adiabatic(U0=3.8, Umin=3.4, dx=-0.01, beta=200.0, order=1, max_adiabatic_iter=200)
