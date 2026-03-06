
import numpy as np

import triqs.utility.mpi as mpi

from triqs.gf import Gf, MeshDLRImTime, BlockGf, make_gf_dlr, make_gf_dlr_imfreq
from triqs.atom_diag import AtomDiag

from adapol.anacont import anacont_triqs

from .diag import all_connected_pairings

from .dlr_dyson_ppsc import DysonItPPSC

from . import DiagramEvaluator
from .dense import DenseDiagramEvaluator


class BlockSparseSolver(object):
    
    def __init__(self, H_loc, beta, w_max, eps, gf_struct, conserved_operators=None):

        self.H_loc = H_loc
        self.gf_struct = gf_struct
        self.beta = beta
        self.w_max = w_max
        self.eps = eps
        self.conserved_operators = conserved_operators

        self.fundamental_operators = fundamental_operators_from_gf_struct(gf_struct)

        self.eta = 0. # Pseudo particle chemical potential (shift of the pseudo particle energies).

        self.mesh_tau = MeshDLRImTime(beta=self.beta, statistic='Fermion', w_max=self.w_max, eps=self.eps)

        self.Delta_tau = BlockGf(mesh=self.mesh_tau, gf_struct=self.gf_struct)

        if self.conserved_operators is None:
            # Uses autopartition algorithm for symmetry discovery, when no conserved operators are provided. 
            # This is the default, and recommended, usage.
            self.ad = AtomDiag(self.H_loc, self.fundamental_operators) 
            self.conserved_operators = 'automatic'
        else:
            self.ad = AtomDiag(self.H_loc, self.fundamental_operators, self.conserved_operators)

        self.use_dense_solver = (self.conserved_operators == []) # Without symmetries, use the dense solver

        print_atom_diag_info(self.ad)

        self.G0 = atomic_pseudo_particle_greens_function(self.ad, self.beta, self.mesh_tau)
        self.G = self.G0.copy()
        self.Sigma = self.get_zero_pseudo_particle_propagator()

        # FIXME! Get ito from mesh_tau
        from .pycppdlr import build_dlr_rf
        from .pycppdlr import ImTimeOps
        ito = ImTimeOps(w_max * beta, build_dlr_rf(w_max * beta, eps)) 

        self.dysons = [DysonItPPSC(self.beta, ito, G0_block.data) for _, G0_block in self.G0]

        print(logo())
        print()
        print(f'use_dense_solver = {self.use_dense_solver}')
        print(f'conserved_operators = {self.conserved_operators}')
        print()
    

    def fit_hybridization(self, tol=None):

        self.tol_adapol = self.eps if tol is None else tol

        Delta_iw = make_gf_dlr_imfreq(self.Delta_tau)

        Delta_iw_dense = self.__from_blockgf_to_dense(Delta_iw)

        _, fit_error, poles, pole_weights = anacont_triqs(Delta_iw_dense, tol=self.tol_adapol, debug=True)

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
            g.data[:] = g_dense.data[:, sidx:sidx+size, sidx:sidx+size]
            sidx += size

        return G


    def set_hybridization_poles_and_coefficients(self, poles, coefficients):

        self.hyb = Dummy()
        self.hyb.poles = poles
        self.hyb.coefficients = coefficients


    def init_diagram_evaluator(self):
        
        if self.use_dense_solver:
            self.d = DenseDiagramEvaluator(self.hyb.poles, self.hyb.coefficients, self.mesh_tau, self.ad)
        else:
            self.d = DiagramEvaluator(
                self.beta, self.w_max * self.beta, self.eps, # -- Todo: Get this info from self.G
                self.hyb.poles, self.hyb.coefficients,
                self.G, self.ad)


    def solve(self, max_order, tol=1e-7, maxiter=10, mix=1.):

        for iter in range(1, maxiter+1):

            self.Sigma = self.pseudo_particle_self_energy(self.G, max_order)
            G_new = self.solve_dyson(self.Sigma, self.eta)
            G_new = self.normalize_pseudo_particle_gf(G_new)
            Z = self.partition_function_from_ppgf(G_new)

            diff_G = max_abs_diff_BlockGf(G_new, self.G)

            print(f'iter = {iter}, diff_G = {diff_G:2.2E}, Z-1 = {Z-1:2.2E}')

            self.G = mix * G_new + (1 - mix) * self.G

            if diff_G < tol:
                print(f'Converged after {iter} iterations with diff_G = {diff_G:2.2E} < tol = {tol:2.2E}')
                break


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

        print(f'Updated eta = {self.eta:2.2E} to normalize Z = {Z:2.2E} to 1.')
        print(f'After normalization, Z_new-1 = {Z_new-1:2.2E}')

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


    def pseudo_particle_self_energy(self, G, max_order):

        self.Sigma = self.get_zero_pseudo_particle_propagator()

        for order in range(1, max_order+1):
            self.Sigma += self.pseudo_particle_self_energy_order(G, order)

        return self.Sigma


    def pseudo_particle_self_energy_order(self, G, order):

        Sigma = self.get_zero_pseudo_particle_propagator()
        
        for sign, topology in all_connected_pairings(order):
            topology = np.array(topology, dtype=np.int32)
            Sigma +=  pow(-1, order) * sign * self.pseudo_particle_self_energy_topology(G, topology) # FIXME! Signs
            
        return Sigma
    
    
    def pseudo_particle_self_energy_topology(self, G, topology):
        order = len(topology)
        return pow(-1, order+1) * self.d.compute_self_energy(G, topology) # FIXME! Sign convention.


    def pseudo_particle_self_energy_topology_loop(self, G, topology):

        order = len(topology)
        Sigma = self.get_zero_pseudo_particle_propagator()

        for n in range(self.d.get_num_self_energy_backbones(topology)):
            Sigma += pow(-1, order+1) * self.d.compute_self_energy(G, topology, n) # FIXME! Sign convention.

        return Sigma
    

    def get_zero_pseudo_particle_propagator(self):
        return zero_pseudo_particle_propagator(self.ad, self.mesh_tau)


    def get_zero_single_particle_greens_function(self):
        spgf = Gf(mesh=self.mesh_tau, target_shape=[len(self.fundamental_operators)]*2)
        spgf.data[:] = 0.
        return spgf


    def single_particle_greens_function(self, G, max_order):
        self.spgf = self.get_zero_single_particle_greens_function()

        for order in range(1, max_order+1):
            self.spgf += self.single_particle_greens_function_order(G, order)
        
        return self.spgf


    def single_particle_greens_function_order(self, G, order):

        spgf = self.get_zero_single_particle_greens_function()

        for sign, topology in all_connected_pairings(order):
            topology = np.array(topology, dtype=np.int32)
            spgf += pow(-1, order) * sign * self.single_particle_greens_function_topology(G, topology)

        return spgf


    def single_particle_greens_function_topology(self, G, topology):

        spgf = self.get_zero_single_particle_greens_function()

        spgf.data[:] = self.d.compute_single_ptcle_gf(G, topology) # FIXME! return triqs::gfs::gf

        return spgf


    def single_particle_greens_function_topology_loop(self, G, topology):
        spgf = self.get_zero_single_particle_greens_function()

        for n in range(self.d.get_num_single_ptcle_gf_backbones(topology)):
            spgf.data[:] += self.d.compute_single_ptcle_gf(G, topology, n) # FIXME! return triqs::gfs::gf

        return spgf


def logo():
    """ https://patorjk.com/software/taag/#p=display&f=Red+Phoenix&t=XCA """
    return r"""____  ____________     _____
\   \/  /\_   ___ \   /  _  \
 \     / /    \  \/  /  /_\  \
 /     \ \     \____/    |    \
/___/\  \ \______  /\____|__  /
      \_/        \/         \/  [github.com/TRIQS/xca]"""


class Dummy(object):
    def __init__(self):
        pass


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

    print(r"""   _____    __                   ________   .___    _____     ________
  /  _  \ _/  |_  ____    _____  \______ \  |   |  /  _  \   /  _____/
 /  /_\  \\   __\/  _ \  /     \  |    |  \ |   | /  /_\  \ /   \  ___
/    |    \|  | (  <_> )|  Y Y  \ |    `   \|   |/    |    \\    \_\  \
\____|__  /|__|  \____/ |__|_|  //_______  /|___|\____|__  / \______  /
        \/                    \/         \/              \/         \/  """)

    print(ad)
    if verbose:
        print(f'full_hilbert_space_dim = {ad.full_hilbert_space_dim}')
        print(f'n_subspaces = {ad.n_subspaces}')
        print(f'get_subspace_dims = {ad.get_subspace_dims()}')
        print(f'fops = {ad.fops}')
        print(f'quantum_numbers = {ad.quantum_numbers}')
        print(f'fock_states = {ad.fock_states}')
        print(f'unitary_matrics = {ad.unitary_matrices}')
        print(f'gs_energy = {ad.gs_energy}')
        print(f'energies = {ad.energies}')
        print(f'energies + gs_energy = {[ e + ad.gs_energy for e in ad.energies]}')
    print('-'*72)


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