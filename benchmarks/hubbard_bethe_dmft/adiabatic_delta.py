

import copy
import numpy as np

import triqs.utility.mpi as mpi

from triqs.gf import make_gf_dlr_imtime, make_gf_dlr_imfreq, make_gf_dlr, SemiCircular, Gf

from calc_xca import solve_one_spinful_fermion_block_sparse_solver

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

        update_U(S, S.U)


    def get_x_delta(self, S):

        g_y = self.to_ir(S.G_tau['up'], ph_sym=True)

        if self.n_cut is not None:            
            g_y = g_y[:self.n_cut]

        # U, Delta_up
        x = np.array([], dtype=float)
        x = np.append(x, [S.U.real])
        x = np.append(x, g_y.real)

        return x
    

    def set_x_delta(self, S, x):

        S.U = x[0]

        terms = [
            (S.Delta_tau['up'].data, True),
            ]

        sidx = 1
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

        update_U(S, S.U)        


def get_x_dlr(S):

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

    update_U(S, S.U)


def get_x_dlr_delta(S):

    # U, Delta_up
    x = np.array([], dtype=float)

    x = np.append(x, [S.U.real])

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

    return x


def set_x_dlr_delta(S, x):

    S.U = x[0]
    sidx = 1

    terms = [
        S.Delta_tau['up'].data,
        ]

    for term in terms:
        shape = term.shape
        N = np.prod(shape)
        term[:] = x[sidx : sidx + N].reshape(shape)
        sidx += N

    S.Delta_tau['do'].data[:] = S.Delta_tau['up'].data

    update_U(S, S.U)    


def update_U(S, U):
    S.U = U
    # Update H_loc and atom_diag for change in U
    from triqs.operators import n
    S.H_loc = S.U * n('up', 0) * n('do', 0) - (S.mu + S.U/2) * (n('up', 0) + n('do', 0))

    from triqs.atom_diag import AtomDiag
    S.atom_diag = AtomDiag(S.H_loc, S.fundamental_operators) 
    from triqs_xca.block_sparse_solver import atomic_pseudo_particle_greens_function
    S.G0, S.eta0 = atomic_pseudo_particle_greens_function(S.atom_diag, S.beta, S.mesh_tau)

    # FIXME! Get ito from mesh_tau
    from triqs_xca.pycppdlr import build_dlr_rf
    from triqs_xca.pycppdlr import ImTimeOps
    from triqs_xca.dlr_dyson_ppsc import DysonItPPSC
    ito = ImTimeOps(S.w_max * S.beta, build_dlr_rf(S.w_max * S.beta, S.eps, S.dlr_symmetrize), symmetrize=S.dlr_symmetrize) 
    S.dysons = [DysonItPPSC(S.beta, ito, G0_block.data) for _, G0_block in S.G0]
                    

def solve_one_spinful_fermion_block_sparse_solver_adiabatic_delta(
        beta=10.0, U=6.0, mu=0.0, order=1,
        eps=1e-10, w_max=6.0,
        dmft_maxiter=40, dmft_tol=1e-5, 
        ppsc_maxiter=20, ppsc_tol=1e-8,
        delta_mix=1.0, ppsc_mix=1.0,
        S_old=None, dx_vec=None, dx=0.1,
        root_tol=1e-6,
        ):

    if S_old is None:
        # -- Run one fwd iteration solve to get started
        S_old = solve_one_spinful_fermion_block_sparse_solver(
            beta=beta, U=U, mu=mu, order=order,
            eps=eps, w_max=w_max,
            dmft_maxiter=dmft_maxiter, dmft_tol=dmft_tol, 
            ppsc_maxiter=ppsc_maxiter, ppsc_tol=ppsc_tol,
            S_old=None, delta_mix=delta_mix, ppsc_mix=ppsc_mix, dlr_polefitting=False, dlr_symmetrize=False)

    # -- TODO: Implement .copy() for BlockSparseSolver to avoid this hack        
    #from triqs_xca.block_sparse_solver import BlockSparseSolver
    #d = S_old.__reduce_to_dict__()
    #S = BlockSparseSolver.__factory_from_dict__('Name?', d)
    S = copy.deepcopy(S_old)

    if False:
        set_x, get_x = set_x_dlr, get_x_dlr
        #set_x, get_x = set_x_dlr_delta, get_x_dlr_delta
    else:
        conv = ConverterIRDLR(S, n_cut=20) # Metallic
        #conv = ConverterIRDLR(S, n_cut=None) # Insulating
        set_x, get_x = conv.set_x, conv.get_x
        #set_x, get_x = conv.set_x_delta, conv.get_x_delta

    # set direction for initial guess
    x_old = get_x(S)

    if dx_vec is None:
        print(f'No dx_vec provided, using dx = {dx} for initial guess direction.')
        dx_vec = np.zeros_like(x_old)
        dx_vec[0] = dx # Change U by dx

    x0 = x_old + dx_vec
    #x0 = x_old

    def root_function(x):
        
        set_x(S, x)
        
        S.solve(
            max_order=order, 
            tol=1e-20, maxiter=1, # for set/get_x_dlr
            #tol=1e-8, maxiter=30, # for set/get_x_dlr_delta
            dlr_polefitting=False, verbose=False, normalization='classic')
        
        x_new = get_x(S)

        root = x_new - x
        #diff = np.max(np.abs(root))
        #print(f'diff = {diff:2.2E}')
        
        diff = x_new - x_old
        #diff[1:] *= 10.
        norm = np.linalg.norm(diff)
        dist = norm - np.abs(dx) # Ensure dx distance from previous sol
        root[0] = dist # DEBUG

        #print(f'dx = {dx}, root[0] = {root[0]:2.2E}, root[1] = {root[1]:2.2E}')
        #root[0] = x[0] - (x_old[0] + dx) # Ensure dx distance from previous sol
        
        #print(f'max(abs(root)) = {np.max(np.abs(root)):2.2E}')
        return root

    if True:        
        from scipy.optimize import root

        r = root_function(x0)
        print(f'Before root solve: dx = {dx}, norm(dx_vec) = {np.linalg.norm(dx_vec):2.2E}, max(abs(root)) = {np.max(np.abs(r)):2.2E}')

        sol = root(
            root_function, x0,
            #method='df-sane', options=dict(fatol=1e-6, sigma_eps=1e-3, sigma_0=0.01, disp=True, maxfev=200, line_search='cheng'))
            #method='hybr', options=dict(maxfev=1000, xtol=root_tol))
            #method='df-sane', options=dict(maxfev=300, fatol=root_tol, disp=True))
            method='krylov', options=dict(maxiter=10, xtol=root_tol, disp=True))
        
        print(sol)
        x = sol.x
        set_x(S, x)
        dx_vec = x - x_old

        S.dmft_iter = sol.nfev
        S.dmft_diff = np.max(np.abs(sol.fun))

        r = root_function(x)
        print(f'After root solve: dx = {dx}, norm(dx_vec) = {np.linalg.norm(dx_vec):2.2E}, max(abs(root)) = {np.max(np.abs(r)):2.2E}')


    else:
        x = x0

    if False:
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

    from triqs.operators import n
    S.n_up_exp = S.expectation_value(n('up', 0)).real
    S.n_do_exp = S.expectation_value(n('do', 0)).real
    S.docc_exp = S.expectation_value(n('up', 0)*n('do', 0)).real

    return S, dx_vec
