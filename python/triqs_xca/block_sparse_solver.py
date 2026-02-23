
import numpy as np

import triqs.utility.mpi as mpi

from triqs.gf import Gf, MeshDLRImTime, BlockGf
from triqs.atom_diag import AtomDiag

from adapol import anacont as adapol_anacont

from .diag import all_connected_pairings

from . import DiagramEvaluator


class BlockSparseSolver(object):

    
    def __init__(self, H_loc, fundamental_operators, beta, w_max, eps, conserved_operators=[]):

        self.H_loc = H_loc
        self.fundamental_operators = fundamental_operators
        self.beta = beta
        self.w_max = w_max
        self.eps = eps
        self.conserved_operators = conserved_operators

        self.mesh_tau = MeshDLRImTime(beta=self.beta, statistic='Fermion', w_max=self.w_max, eps=self.eps)

        self.ad = AtomDiag(self.H_loc, self.fundamental_operators, self.conserved_operators)
        print_atom_diag_info(self.ad)

        self.G0 = atomic_pseudo_particle_greens_function(self.ad, self.beta, self.mesh_tau)
        self.G = self.G0.copy()
        self.Sigma = self.get_zero_pseudo_particle_propagator()

        print(logo())


    def set_hybridization(self, Delta_iw, eps=None): # FIXME! Delta_tau on DLR mesh

        self.tol_adapol = self.eps if eps is None else eps

        iwn = np.array([ complex(iw) for iw in Delta_iw.mesh ])
        
        func, fitting_error, pol, weight = adapol_anacont(Delta_iw.data, iwn, tol=self.tol_adapol)
        
        self.set_hybridization_poles_and_coefficients(pol, weight)


    def set_hybridization_poles_and_coefficients(self, poles, coefficients):

        self.hyb = Dummy()
        self.hyb.poles = poles
        self.hyb.coefficients = coefficients


    def init_diagram_evaluator(self):
        
        self.d = DiagramEvaluator(
            self.beta, self.w_max * self.beta, self.eps, # -- Todo: Get this info from self.G
            self.hyb.poles, self.hyb.coefficients,
            self.G, self.ad)

    def pseudo_particle_greens_function(self):
        return self.G


    def pseudo_particle_self_energy(self, max_order):
        self.Sigma = self.get_zero_pseudo_particle_propagator()
        for order in range(1, max_order+1):
            self.Sigma += self.pseudo_particle_self_energy_order(order)

        return self.Sigma
            
    def pseudo_particle_self_energy_order(self, order):

        Sigma = self.get_zero_pseudo_particle_propagator()
        
        for sign, topology in all_connected_pairings(order):
            print(f'topology = {topology}')
            topology = np.array(topology, dtype=np.int32)
            Sigma +=  pow(-1, order) * sign * self.pseudo_particle_self_energy_topology(topology)
            
        return Sigma
    
    
    def pseudo_particle_self_energy_topology(self, topology):
        return self.d.compute_self_energy(topology)


    def pseudo_particle_self_energy_topology_loop(self, topology):

        Sigma = self.get_zero_pseudo_particle_propagator()

        for n in range(self.d.get_num_self_energy_backbones(topology)):
            Sigma += self.d.compute_self_energy(topology, n)

        return Sigma
    

    def get_zero_pseudo_particle_propagator(self):
        return zero_pseudo_particle_propagator(self.ad, self.mesh_tau)


    def get_zero_single_particle_greens_function(self):
        spgf = Gf(mesh=self.mesh_tau, target_shape=[len(self.fundamental_operators)]*2)
        spgf.data[:] = 0.
        return spgf


    def single_particle_greens_function(self, max_order):
        print('--> single_particle_greens_function')
        print(f'max_order = {max_order}')
        self.spgf = self.get_zero_single_particle_greens_function()
        print(self.spgf)

        for order in range(1, max_order+1):
            self.spgf += self.single_particle_greens_function_order(order)
        
        return self.spgf


    def single_particle_greens_function_order(self, order):
        print('--> single_particle_greens_function_order')
        print(f'order = {order}')

        spgf = self.get_zero_single_particle_greens_function()

        for sign, topology in all_connected_pairings(order):
            print(f'topology = {topology}')
            topology = np.array(topology, dtype=np.int32)
            spgf += pow(-1, order) * sign * self.single_particle_greens_function_topology(topology)

        return spgf


    def single_particle_greens_function_topology(self, topology):
        spgf = self.get_zero_single_particle_greens_function()
        spgf.data[:] = self.d.compute_single_ptcle_gf(topology)
        return spgf


    def single_particle_greens_function_topology_loop(self, topology):
        spgf = self.get_zero_single_particle_greens_function()

        for n in range(self.d.get_num_single_ptcle_gf_backbones(topology)):
            spgf.data[:] += self.d.compute_single_ptcle_gf(topology, n)

        return spgf




def logo():
    """ https://patorjk.com/software/taag/#p=display&f=Red+Phoenix&t=XCA """
    return r"""____  ____________     _____
\   \/  /\_   ___ \   /  _  \
 \     / /    \  \/  /  /_\  \
 /     \ \     \____/    |    \
/___/\  \ \______  /\____|__  /
      \_/        \/         \/  [github.com/TRIQS/xca]

    [[Block-Sparse]]"""


class Dummy(object):
    def __init__(self):
        pass


def hamiltonian_matrix(ad):

    H_mat = np.zeros([ad.full_hilbert_space_dim]*2, dtype=float)

    for sidx in range(ad.n_subspaces):
        U = ad.unitary_matrices[sidx]
        E = ad.energies[sidx]
        H_block_diag = np.diag(E + ad.gs_energy)
        H_block = U @ H_block_diag @ U.T.conj()
        fidx = ad.fock_states[sidx]
        bidx = np.ix_(fidx, fidx)
        H_mat[bidx] = H_block

    return H_mat


def zero_pseudo_particle_propagator(ad, mesh_tau):

    G_tau_blocks = []
    
    for sidx in range(ad.n_subspaces):
        G_tau = Gf(mesh=mesh_tau, target_shape=[ad.get_subspace_dims()[sidx]]*2)
        G_tau.data[:] = 0.
        G_tau_blocks.append(G_tau)
        
    G_tau = BlockGf(block_list=G_tau_blocks)

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


def print_atom_diag_info(ad):

    print(r"""   _____    __                   ________   .___    _____     ________
  /  _  \ _/  |_  ____    _____  \______ \  |   |  /  _  \   /  _____/
 /  /_\  \\   __\/  _ \  /     \  |    |  \ |   | /  /_\  \ /   \  ___
/    |    \|  | (  <_> )|  Y Y  \ |    `   \|   |/    |    \\    \_\  \
\____|__  /|__|  \____/ |__|_|  //_______  /|___|\____|__  / \______  /
        \/                    \/         \/              \/         \/  """)

    print(ad)
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


def pseudo_particle_block_gf_to_dense(block_gf, ad):

    mesh = block_gf[0].mesh
    dense_gf = Gf(mesh=mesh, target_shape=[ad.full_hilbert_space_dim]*2)

    for sidx in range(ad.n_subspaces):
        fidx = ad.fock_states[sidx]
        bidx = np.ix_(range(len(mesh)), fidx, fidx)
        dense_gf.data[bidx] = block_gf[sidx].data

    return dense_gf    
