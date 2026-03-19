import numpy as np

from collections import defaultdict

import triqs.utility.mpi as mpi

from triqs.gf import Gf, MeshDLRImTime, BlockGf, make_gf_dlr, make_gf_dlr_imfreq
from triqs.atom_diag import AtomDiag

from adapol.anacont import anacont_triqs
from adapol.fit_utils_dlr import polefitting_dlr

from .diag import all_connected_pairings

from .dlr_dyson_ppsc import DysonItPPSC

from . import DiagramEvaluator
from .dense import DenseDiagramEvaluator

from .module import trace
from .module import convolve_ppsc as conv


def is_root():
    return mpi.is_master_node()


def scatter_array_over_ranks(arr):
    size = mpi.size
    rank = mpi.rank
    arr_rank = np.array_split(np.array(arr), size, axis=0)[rank]
    return arr_rank


class BlockSparseSolver(object):
    
    def __init__(self, H_loc, beta, w_max, eps, gf_struct, conserved_operators='automatic'):

        self.H_loc = H_loc
        self.gf_struct = gf_struct
        self.beta = beta
        self.w_max = w_max
        self.eps = eps
        self.conserved_operators = conserved_operators

        self.fundamental_operators = fundamental_operators_from_gf_struct(gf_struct)
        self.use_dense_solver = (self.conserved_operators == []) # Without symmetries, use the dense solver

        if is_root():
            print(logo())
            print()
            print(f'use_dense_solver = {self.use_dense_solver}')
            print(f'conserved_operators = {self.conserved_operators}')
        
        self.eta = 0. # Pseudo particle chemical potential (shift of the pseudo particle energies).

        self.mesh_tau = MeshDLRImTime(beta=self.beta, statistic='Fermion', w_max=self.w_max, eps=self.eps)

        self.Delta_tau = BlockGf(mesh=self.mesh_tau, gf_struct=self.gf_struct, name='Delta_tau')

        print(f'self.H_loc = {self.H_loc}')
        print(f'self.fundamental_operators = {self.fundamental_operators}')
        print(f'self.conserved_operators = {self.conserved_operators}')

        if self.conserved_operators == 'automatic':
            # Uses autopartition algorithm for symmetry discovery, when no conserved operators are provided. 
            # This is the default, and recommended, usage.
            self.ad = AtomDiag(self.H_loc, self.fundamental_operators) 
        else:
            self.ad = AtomDiag(self.H_loc, self.fundamental_operators, self.conserved_operators)

        if is_root(): 
            print(f'mesh: {self.mesh_tau}')
            print_atom_diag_info(self.ad)

        self.G0 = atomic_pseudo_particle_greens_function(self.ad, self.beta, self.mesh_tau)
        self.G = self.G0.copy()
        self.Sigma = self.get_zero_pseudo_particle_propagator()

        # FIXME! Get ito from mesh_tau
        from .pycppdlr import build_dlr_rf
        from .pycppdlr import ImTimeOps
        ito = ImTimeOps(w_max * beta, build_dlr_rf(w_max * beta, eps)) 

        self.dysons = [DysonItPPSC(self.beta, ito, G0_block.data) for _, G0_block in self.G0]
    

    def fit_hybridization(self, tol=None, use_polefitting_dlr=False):

        self.tol_adapol = self.eps if tol is None else tol

        if use_polefitting_dlr:
            #Delta_dlr = make_gf_dlr(self.Delta_tau)
            #Delta_dlr_dense = self.__from_blockgf_to_dense(Delta_dlr)
            #w_dlr = np.array([ float(x) for x in Delta_dlr.mesh ])
            
            #pole_weights, poles, fit_error = polefitting_dlr(
            #    Delta_dlr_dense.data, w_dlr, self.beta, eps=self.tol_adapol, statistics="Fermion", verbose=True)

            from adapol.fit_utils_dlr import polefitting_dlr_triqs
            Delta_tau_dense = self.__from_blockgf_to_dense(self.Delta_tau)
            pole_weights, poles, fit_error = polefitting_dlr_triqs(
                Delta_tau_dense, eps=self.tol_adapol, statistics="Fermion", verbose=True)

            pole_weights *= -1. # FIXME! Why is this necessary? Is there a sign convention issue in polefitting_dlr?

        else:
            Delta_iw = make_gf_dlr_imfreq(self.Delta_tau)
            Delta_iw_dense = self.__from_blockgf_to_dense(Delta_iw)
            _, fit_error, poles, pole_weights = anacont_triqs(Delta_iw_dense, tol=self.tol_adapol, debug=True)
            #_, fit_error, poles_ref, pole_weights_ref = anacont_triqs(Delta_iw_dense, tol=self.tol_adapol, debug=True)

        #print(f'poles     = {poles}')
        #print(f'poles_ref = {poles_ref}')
        #print(f'pole_weights     =\n{pole_weights}')
        #print(f'pole_weights_ref =\n{pole_weights_ref}')
        
        self.set_hybridization_poles_and_coefficients(poles, pole_weights)

        self.hyb.fit_error = fit_error


    def __from_blockgf_to_dense(self, G):

        for b, g in G:
            assert( len(g.target_shape) == 2)
            assert( g.target_shape[0] == g.target_shape[1] )

        norb = sum([ g.target_shape[0] for b, g in G ])
        
        G_dense = Gf(mesh=G.mesh, target_shape=[norb]*2)
        
        sidx = 0
        for b, g in G:
            size = g.target_shape[0]
            G_dense.data[:, sidx:sidx+size, sidx:sidx+size] = g.data
            sidx += size

        return G_dense


    def __from_dense_to_blockgf(self, G_dense, gf_struct):

        G = BlockGf(mesh=G_dense.mesh, gf_struct=gf_struct)

        sidx = 0
        for b, g in G:
            size = g.target_shape[0]
            g.data[:] = G_dense.data[:, sidx:sidx+size, sidx:sidx+size]
            sidx += size

        return G


    def set_hybridization_poles_and_coefficients(self, poles, coefficients):

        self.hyb = PoleRepresentation(poles, coefficients)

        if is_root():
            print(f'hyb poles = {len(self.hyb.poles)}')
            #print(f'hyb.poles = {self.hyb.poles}')
            #print(f'hyb.coefficients =\n{self.hyb.coefficients}')

        self.init_diagram_evaluator() # FIXME! Evaluator takes hyb poles and coeffs in constructor


    def init_diagram_evaluator(self):
        if is_root(): print(f'Initializing diagram evaluator with use_dense_solver = {self.use_dense_solver}')
        if self.use_dense_solver:
            self.d = DenseDiagramEvaluator(self.hyb.poles, self.hyb.coefficients, self.mesh_tau, self.ad)
        else:
            self.d = DiagramEvaluator(self.hyb.poles, self.hyb.coefficients, self.mesh_tau, self.ad)
        if is_root(): print(f'done.')


    def solve(self, max_order, tol=1e-7, maxiter=10, mix=1., delta_tol=None):

        self.max_order = max_order
        self.delta_tol = delta_tol if delta_tol is not None else 0.1 * tol

        self.fit_hybridization(tol=tol)

        for iter in range(1, maxiter+1):

            self.Sigma = self.eval_pseudo_particle_self_energy(self.G, self.max_order)

            #G_new = self.solve_dyson(self.Sigma, self.eta)
            #G_new = self.normalize_pseudo_particle_gf(G_new)

            self.solve_ppsc_chempot_newton(tol=10*self.eps)

            G_new = self.solve_dyson(self.Sigma, self.eta)
            Z = self.partition_function_from_ppgf(G_new)

            if np.abs(Z - 1) > 10*self.eps:
                if is_root(): print(f'solve_ppsc_chempot_adiabatic_ode')
                self.solve_ppsc_chempot_adiabatic_ode(tol=10*self.eps)
                self.solve_ppsc_chempot_newton(tol=10*self.eps)
            
            G_new = self.solve_dyson(self.Sigma, self.eta)

            Z = self.partition_function_from_ppgf(G_new)

            diff_G = max_abs_diff_BlockGf(G_new, self.G)

            if is_root(): print(f'iter = {iter}, diff_G = {diff_G:2.2E}, Z-1 = {Z-1:+2.2E}')

            self.G = mix * G_new + (1 - mix) * self.G

            if diff_G < tol:
                if is_root(): print(f'Converged after {iter} iterations with diff_G = {diff_G:2.2E} < tol = {tol:2.2E}')
                break

        G_tau = self.eval_single_particle_greens_function(self.G, max_order=self.max_order)
        self.G_tau = self.__from_dense_to_blockgf(G_tau, self.gf_struct)


    def normalize_pseudo_particle_gf(self, G):
        """ Normalize the pseudo particle Green's function by updating the pseudo particle chemical potential eta, such that the partition function Z is equal to 1. """

        Z = self.partition_function_from_ppgf(G)

        deta = np.log(np.abs(Z)) / self.beta

        def energy_shift_ppgf(G, deta):
            G_new = G.copy()
            tau = np.array([ float(t) for t in G.mesh ])
            for bidx, g in G_new:
                g.data[:] *= np.exp(-tau * deta)[:, None, None]
            return G_new

        G_new = energy_shift_ppgf(G, deta)

        Z_new = self.partition_function_from_ppgf(G_new)

        if is_root(): print(f'  Update eta = {self.eta:2.2E}, Z = {Z:2.2E}, Z_new-1 = {Z_new-1:+2.2E}')

        self.eta += deta

        return G_new


    def pseudo_particle_greens_function(self):
        return self.G


    def partition_function_from_ppgf(self, G):
        """ Partition function of the impurity model, computed from the pseudo particle Green's function as

        ..math::
            Z = -\\mathrm{Tr}[ G(\\tau=\\beta) ]
        
        """
        
        Z = -trace_dlr_imtime_BlockGf(G)
        assert(Z.imag < 1e-12)
        Z = Z.real

        return Z


    def partition_function(self):
        return self.partition_function_from_ppgf(self.G)


    def solve_dyson(self, Sigma, eta):

        assert type(Sigma) is BlockGf, 'Sigma must be a BlockGf'

        G = self.get_zero_pseudo_particle_propagator()

        for dyson, (bidx, sigma_b) in zip(self.dysons, Sigma):
            G[bidx].data[:] = dyson.solve(sigma_b.data, eta)

        return G


    def pseudo_particle_self_energy(self):
        return self.Sigma


    def eval_pseudo_particle_self_energy(self, G, max_order):

        self.Sigma = self.get_zero_pseudo_particle_propagator()

        for order in range(1, max_order+1):
            self.Sigma += self.eval_pseudo_particle_self_energy_order(G, order)

        return self.Sigma


    def eval_pseudo_particle_self_energy_order(self, G, order):

        Sigma = self.get_zero_pseudo_particle_propagator()
        
        for sign, topology in all_connected_pairings(order):
            topology = np.array(topology, dtype=np.int32)
            Sigma +=  pow(-1, order) * sign * \
                self.eval_pseudo_particle_self_energy_topology_loop(G, topology) # FIXME! Signs
            
        return Sigma
    
    
    def eval_pseudo_particle_self_energy_topology(self, G, topology):
        order = len(topology)
        return pow(-1, order+1) * self.d.compute_self_energy(G, topology) # FIXME! Sign convention.


    def eval_pseudo_particle_self_energy_topology_loop(self, G, topology):
        order = len(topology)
        Sigma = self.get_zero_pseudo_particle_propagator()

        n_max = self.d.get_num_self_energy_backbones(topology)
        n_vec = scatter_array_over_ranks(np.arange(n_max))

        for n in n_vec:
            Sigma += pow(-1, order+1) * self.d.compute_self_energy(G, topology, n) # FIXME! Sign convention.

        for bidx, sigma_b in Sigma:
            sigma_b.data[:] = mpi.all_reduce(sigma_b.data)

        return Sigma
    

    def get_zero_pseudo_particle_propagator(self):
        return zero_pseudo_particle_propagator(self.ad, self.mesh_tau)


    def get_zero_single_particle_greens_function(self):
        spgf = Gf(mesh=self.mesh_tau, target_shape=[len(self.fundamental_operators)]*2)
        spgf.data[:] = 0.
        return spgf


    def single_particle_greens_function(self, max_order):
        return self.eval_single_particle_greens_function(self.G, max_order=max_order)


    def eval_single_particle_greens_function(self, G, max_order):
        self.spgf = self.get_zero_single_particle_greens_function()

        for order in range(1, max_order+1):
            self.spgf += self.eval_single_particle_greens_function_order(G, order)
        
        return self.spgf


    def eval_single_particle_greens_function_order(self, G, order):

        spgf = self.get_zero_single_particle_greens_function()

        for sign, topology in all_connected_pairings(order):
            topology = np.array(topology, dtype=np.int32)
            spgf += pow(-1, order) * sign * \
                self.eval_single_particle_greens_function_topology_loop(G, topology)

        return spgf


    def eval_single_particle_greens_function_topology(self, G, topology):

        spgf = self.get_zero_single_particle_greens_function()

        spgf.data[:] = self.d.compute_single_ptcle_gf(G, topology) # FIXME! return triqs::gfs::gf

        return spgf


    def eval_single_particle_greens_function_topology_loop(self, G, topology):
        spgf = self.get_zero_single_particle_greens_function()

        n_max = self.d.get_num_single_ptcle_gf_backbones(topology)
        n_vec = scatter_array_over_ranks(np.arange(n_max))

        for n in n_vec:
            spgf.data[:] += self.d.compute_single_ptcle_gf(G, topology, n) # FIXME! return triqs::gfs::gf

        spgf.data[:] = mpi.all_reduce(spgf.data)

        return spgf


    def Z_alpha(self, alpha, eta):
        r""" Partition function with Sigma scaled by alpha 
        Solving
        
        .. math::
            (1 - G_0\ast*(\alpha \cdot \Sigma - \eta)\ast*)G = G_0

            Z_\alpha = - \textrm{Tr} \left[ G_\alpha (\beta) \right]
        
        """
        if type(eta) == np.ndarray: eta = eta.item()
        
        Sigma = self.pseudo_particle_self_energy()
        Ga = self.solve_dyson(alpha**2 * Sigma, eta)
        Za = self.partition_function_from_ppgf(Ga).real
        return Za    


    def dZ_alpha_deta(self, alpha, eta):
        r""" Derivative of partition function Z_\alpha with respect to
        the pseudo-particle chemical potential eta. 
        
        .. math::
            \frac{d Z_\alpha}{d \eta} = - \textr{Tr} \left[G_\alpha G_\alpha \right]
        """
        if type(eta) == np.ndarray: eta = eta.item()

        Sigma = self.pseudo_particle_self_energy()
        G = self.solve_dyson(alpha**2 * Sigma, eta)
        dZ_deta = -trace(conv(G, G)).real
        return dZ_deta


    def deta_dalpha(self, alpha, eta):
        r""" Analytic derivative of :math:`\eta(\alpha)`

        .. math::
            \frac{d \eta}{d \alpha} = 
                - \textr{Tr} \left[ G_\alpha \ast \Sigma \ast G_\alpha \right]
                / \textrm{Tr} \left[ G_\alpha \ast G_\alpha \right]

        """
        Sigma = self.pseudo_particle_self_energy()
        Ga = self.solve_dyson(alpha**2 * Sigma, eta)
        TrGaGa = -trace(conv(Ga, Ga)).real
        TrGaSigmaGa = -trace(conv(Ga, conv(Sigma, Ga)))
        deta_dalpha = - 2 * alpha * TrGaSigmaGa / TrGaGa
        return deta_dalpha.real


    def d2eta_dalpha_deta(self, alpha, eta):
        r""" Higher order derivative (used for stiff ODE solver)
        
        .. math:: 
            \frac{d}{d\eta} \left (\frac{d \eta}{d \alpha})
            =
            - \textr{Tr} \left[ G^2_\alpha \ast \Sigma \ast G_\alpha \right]
            / \textrm{Tr} \left[ G_\alpha \ast G_\alpha \right]
            +
            \textr{Tr} \left[ G_\alpha \ast \Sigma \ast G^2_\alpha \right]
            / \textrm{Tr} \left[ G_\alpha \ast G_\alpha \right]
            - 
            2 \textrm{Tr} \left[ G_\alpha \ast G_\alpha \ast G_\alpha \right]
            / \textrm{Tr} \left[ G_\alpha \ast G_\alpha \right]^2

        """

        print('--> d2eta_dalpha_deta')

        Sigma = self.pseudo_particle_self_energy()

        Ga = self.solve_dyson(alpha**2 * Sigma, eta)
        
        Ga2 = conv(Ga, Ga)
        Ga3 = conv(Ga, Ga2)
        
        TrGa2 = -trace(Ga2)
        TrGa3 = -trace(Ga3)

        TrGaSigmaGa  = -trace(conv(Ga,  conv(Sigma, Ga )))
        TrGa2SigmaGa = -trace(conv(Ga2, conv(Sigma, Ga )))
        TrGaSigmaGa2 = -trace(conv(Ga,  conv(Sigma, Ga2)))
        
        d2eta_dalpha_deta = -2*alpha * ((TrGa2SigmaGa + TrGaSigmaGa2)/TrGa2 - 2*TrGaSigmaGa*TrGa3/TrGa2**2)
        
        return d2eta_dalpha_deta.real
    
    
    def solve_ppsc_chempot_adiabatic_ode(self, tol=1e-9, method='DOP853'):

        func = lambda t, y : self.deta_dalpha(t, y[0])
        jac = lambda t, y : np.array([self.d2eta_dalpha_deta(t, y[0])]).reshape(1, 1)

        from scipy.integrate import solve_ivp

        sol = solve_ivp(
            func, (0., 1.), 
            np.array([0.]), 
            #jac=jac,
            atol=tol, 
            method=method,
            dense_output=True,
            )
        
        self.eta = sol.y[0, -1]
        G = self.solve_dyson(self.Sigma, self.eta)
        Z = self.partition_function_from_ppgf(G)
        
        if is_root(): print(f'PPSC: Chempot eta from adiabatic ODE: Z-1={Z-1:+2.2E}')
        return sol


    def solve_ppsc_chempot_newton(self, tol=1e-9):

        f = lambda eta : self.Z_alpha(1., eta) - 1
        fprime = lambda eta : self.dZ_alpha_deta(1., eta)

        from scipy.optimize import root_scalar

        sol = root_scalar(
            f, x0=self.eta, 
            fprime=fprime,
            method='newton',
            xtol=tol,
            )
        
        self.eta = sol.root

        G = self.solve_dyson(self.Sigma, self.eta)
        Z = self.partition_function_from_ppgf(G)

        #if is_root(): print(f'PPSC: Newton eta = {self.eta:+2.2E} Z-1 = {Z-1:+2.2E}')

        return sol


    def __eq__(self, obj):

        for key in self.__dict__.keys():
            if key not in obj.__dict__.keys():
                if key != 'd': # -- Skip the diagram evaluator
                    return False
        
        for key in self.__dict__.keys():
            if key not in self.__skip_keys():
                a = getattr(self, key)
                b = getattr(obj, key)                
                if type(a) == np.ndarray:
                    if not np.array_equal(a, b):
                        print(f'BlockSparseSolver: __eq__: attribute {key} differ')
                        return False
                elif type(a) is Gf:
                    if not np.array_equal(a.data, b.data):
                        print(f'BlockSparseSolver: __eq__: attribute {key} differ')
                        return False
                elif type(a) is BlockGf:
                    for bidx, g in a:
                        g_b = b[bidx]
                        if not np.array_equal(g.data, g_b.data):
                            print(f'BlockSparseSolver: __eq__: attribute {key} differ in block {bidx}')
                            return False
                else:
                    if not a == b:
                        print(f'BlockSparseSolver: __eq__: attribute {key} differ')
                        return False

        return True

    
    def __skip_keys(self):
        return ['d', 'dysons', 'ad']


    def __reduce_to_dict__(self):
        d = self.__dict__.copy()
        keys = set(d.keys()).intersection(self.__skip_keys())
        for key in keys: del d[key]
        return d


    @classmethod
    def __factory_from_dict__(cls, name, d):
        arg_keys = ['H_loc', 'beta', 'w_max', 'eps', 'gf_struct', 'conserved_operators']
        #argv_keys = ['verbose']
        argv_keys = []
        #verbose = d['verbose']
        #d['verbose'] = False # -- Suppress printouts on reconstruction from dict
        ret = cls(*[ d[key] for key in arg_keys ],
                  **{ key : d[key] for key in argv_keys })
        ret.__dict__.update(d)
        #ret.verbose = verbose
        return ret


def logo():
    """ https://patorjk.com/software/taag/#p=display&f=Red+Phoenix&t=XCA """
    return r"""____  ____________     _____
\   \/  /\_   ___ \   /  _  \
 \     / /    \  \/  /  /_\  \
 /     \ \     \____/    |    \
/___/\  \ \______  /\____|__  /
      \_/        \/         \/  [github.com/TRIQS/xca]"""


class PoleRepresentation(object):
    def __init__(self, poles, coefficients):
        self.poles = poles
        self.coefficients = coefficients


    def __eq__(self, obj):
        return np.array_equal(self.poles, obj.poles) and \
            np.array_equal(self.coefficients, obj.coefficients)


    def __reduce_to_dict__(self):
        d = self.__dict__.copy()
        return d


    @classmethod
    def __factory_from_dict__(cls, name, d):
        arg_keys = ['poles', 'coefficients']
        argv_keys = []
        ret = cls(*[ d[key] for key in arg_keys ],
                  **{ key : d[key] for key in argv_keys })
        ret.__dict__.update(d)
        return ret


def hamiltonian_matrix(ad):

    H_mat = np.zeros([ad.full_hilbert_space_dim]*2, dtype=float)

    for sidx in range(ad.n_subspaces):
        H_block = hamiltonian_matrix_block(ad, sidx)
        fidx = ad.fock_states[sidx]
        bidx = np.ix_(fidx, fidx)
        H_mat[bidx] = H_block

    return H_mat


def hamiltonian_matrix_blocks(ad):
  
    H_blocks = []

    for sidx in range(ad.n_subspaces):
        H_block = hamiltonian_matrix_block(ad, sidx)
        H_blocks.append(H_block)

    return H_blocks


def hamiltonian_matrix_block(ad, sidx):

    U = ad.unitary_matrices[sidx]
    E = ad.energies[sidx]
    H_block_diag = np.diag(E + ad.gs_energy)
    H_block = U @ H_block_diag @ U.T.conj()

    return H_block


def zero_pseudo_particle_propagator_dense(ad, mesh_tau):

    G_tau = Gf(mesh=mesh_tau, target_shape=[ad.full_hilbert_space_dim]*2)
    G_tau.data[:] = 0.
    
    return G_tau


def zero_pseudo_particle_propagator(ad, mesh_tau):

    G_tau_blocks = []
    
    for sidx in range(ad.n_subspaces):
        G_tau = Gf(mesh=mesh_tau, target_shape=[ad.get_subspace_dims()[sidx]]*2)
        G_tau.data[:] = 0.
        G_tau_blocks.append(G_tau)
        
    G_tau = BlockGf(block_list=G_tau_blocks)

    return G_tau
        

def atomic_pseudo_particle_greens_function_dense(ad, beta, mesh_tau):
    G_b_tau = atomic_pseudo_particle_greens_function(ad, beta, mesh_tau)
    G_tau = pseudo_particle_block_gf_to_dense(G_b_tau, ad)
    return G_tau


def atomic_pseudo_particle_greens_function(ad, beta, mesh_tau):

    G_tau_blocks = []

    Z0 = np.sum([ np.sum(np.exp(-beta * E)) for E in ad.energies ])
    eta0 = -np.log(Z0) / beta

    tau = np.array([ float(t) for t in mesh_tau ])
    
    for sidx in range(ad.n_subspaces):

        G_tau = Gf(mesh=mesh_tau, target_shape=[ad.get_subspace_dims()[sidx]]*2)
        
        U = ad.unitary_matrices[sidx]
        E = ad.energies[sidx]

        G_diag_tau = -np.exp(-tau[:, None]*(E - eta0)[None, :])
        G_tau.data[:] = np.matmul(U, G_diag_tau[:, :, None] * U.T.conj()[None, :, :])

        G_tau_blocks.append(G_tau)
        
    G_tau = BlockGf(block_list=G_tau_blocks)

    return G_tau


def print_atom_diag_info(ad, verbose=False):

    if False:
        print(r"""   _____    __                   ________   .___    _____     ________
    /  _  \ _/  |_  ____    _____  \______ \  |   |  /  _  \   /  _____/
    /  /_\  \\   __\/  _ \  /     \  |    |  \ |   | /  /_\  \ /   \  ___
    /    |    \|  | (  <_> )|  Y Y  \ |    `   \|   |/    |    \\    \_\  \
    \____|__  /|__|  \____/ |__|_|  //_______  /|___|\____|__  / \______  /
            \/                    \/         \/              \/         \/  """)

    print(f'Triqs: AtomDiag')
    print(f'full_hilbert_space_dim = {ad.full_hilbert_space_dim}')
    print(f'n_subspaces = {ad.n_subspaces}')

    hist = defaultdict(int)
    for dim in ad.get_subspace_dims():
        hist[dim] += 1

    subspace_dims = np.array(list(hist.keys()))
    num_subspaces_per_dim = np.array(list(hist.values()))

    sidx = np.argsort(subspace_dims)

    print(f'subspace_dims = {subspace_dims[sidx]}')
    print(f'num_subspaces_per_dim = {num_subspaces_per_dim[sidx]}')

    print(f'gs_energy = {ad.gs_energy}')
    emin = np.min([np.min(x) for x in ad.energies])
    emax = np.max([np.max(x) for x in ad.energies])
    print(f'energies min/max = {emin:+2.2E} / {emax:+2.2E}')

    if verbose:
        print(f'subspace_dims = {ad.get_subspace_dims()}')
        print(f'fops = {ad.fops}')
        print(f'quantum_numbers = {ad.quantum_numbers}')
        print(f'fock_states = {ad.fock_states}')
        print(f'unitary_matrics = {ad.unitary_matrices}')
        print(f'energies = {ad.energies}')
        print(f'energies + gs_energy = {[ e + ad.gs_energy for e in ad.energies]}')
        print(ad)


def max_abs_diff_BlockGf(G1, G2):
    return np.max([ np.max(np.abs(g1.data - g2.data)) for (_, g1), (_, g2) in zip(G1, G2) ])


def pseudo_particle_block_gf_to_dense(block_gf, ad):

    mesh = block_gf[0].mesh
    dense_gf = Gf(mesh=mesh, target_shape=[ad.full_hilbert_space_dim]*2)

    for sidx in range(ad.n_subspaces):
        fidx = ad.fock_states[sidx]
        bidx = np.ix_(range(len(mesh)), fidx, fidx)
        dense_gf.data[bidx] = block_gf[sidx].data

    return dense_gf    


def get_max_abs_trapz_block_gf(G):

    max_abs = -float('inf')
    for b, g in G:
        g_dlr = make_gf_dlr(g)
        beta = g.mesh.beta

        max_abs_block = np.max(np.abs(np.diag(g_dlr(0.) + g_dlr(beta)))) / 2

        if max_abs_block > max_abs:
            max_abs = max_abs_block

    return max_abs


def get_max_abs_block_gf(G):

    max_abs = -float('inf')
    for b, g in G:
        max_abs_block = np.max(np.abs(g.data))
        if max_abs_block > max_abs:
            max_abs = max_abs_block

    return max_abs


def trace_dlr_imtime_BlockGf(G_dlr_imtime_BlockGf):

    G_dlr_coeff_BlockGf = make_gf_dlr(G_dlr_imtime_BlockGf)

    def block_trace(g_dlr_coeff): 
        return np.trace(g_dlr_coeff(g_dlr_coeff.mesh.beta))

    return np.sum([block_trace(g) for _, g in G_dlr_coeff_BlockGf])


def fundamental_operators_from_gf_struct(gf_struct):
    fundamental_operators = []
    for s, n in gf_struct:
        fundamental_operators += [ (s, i) for i in range(n) ]
    return fundamental_operators


# -- Register Solver in Triqs formats

from h5.formats import register_class
register_class(BlockSparseSolver)
register_class(PoleRepresentation)