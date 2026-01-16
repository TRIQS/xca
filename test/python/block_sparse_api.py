import matplotlib.pyplot as plt
from triqs.operators import c, c_dag, n
from triqs.operators.util.hamiltonians import h_int_kanamori, make_operator_real
from triqs.atom_diag import *
import numpy as np
from itertools import product
from triqs.gf import Gf, MeshDLRImTime, BlockGf
from adapol import anacont
from scipy.integrate import quad
from triqs_xca import DiagramEvaluator

def Kw(w, v):
    return 1 / ( v - w)
def semicircular(x):
    return 2 * np.sqrt(1 - x**2) / np.pi
def make_Delta_with_cont_spec_mat( Z, rho, a=-1.0, b=1.0, r0=0.5, eps=1e-12):
    Delta = np.zeros((Z.shape[0]), dtype=np.complex128)
    for n in range(len(Z)):
        def f(w):
            return Kw(w , Z[n]) * rho(w)

        # f = lambda w: Kw(w-en[i],Z[n])*rho(w)
        Delta[n] = quad(
            f, a, b, epsabs=eps, epsrel=eps, complex_func=True
        )[0]
    T = np.array([
        [1, r0, 0, 0],
        [r0, 1, 0, 0],
        [0, 0, 1, r0],
        [0, 0, r0, 1]])

    return Delta[:, None, None] * T[None, :, :]
def get_full_h_atomic(ad):

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

# ===== Parameters =====
beta = 2.0
eps = 1e-6
lamb = 20.0 * beta

# ===== Construct AtomDiag object =====
# Definition of a 2-orbital atom
spin_names = ('up','dn')
n_orb = 2
# Set of fundamental operators
fops = [(sn,on) for sn, on in product(spin_names,range(n_orb))]

# Numbers of particles with spin up/down
N_up = n('up',0) + n('up',1)
N_dn = n('dn',0) + n('dn',1)

# Construct Hubbard-Kanamori Hamiltonian
U = 3.0 * np.ones((2,2))
Uprime = 2.0 * np.ones((2,2))
J_hund = 0.5

H = h_int_kanamori(spin_names, n_orb, U, Uprime, J_hund, off_diag=True) # not sure about off_diag

# Add chemical potential
H += -4.0 * (N_up + N_dn)

# Split using the total number of particles, N = N_up + N_dn
ad = AtomDiag(H, fops, [N_up+N_dn])

# Get blocks of Hamiltonian and construct G_ppsc as atomic propagator
H_mat = get_full_h_atomic(ad)
H_evals = []
H_evecs = []
tr_exp_minusbetaH = 0
for s, state in enumerate(ad.fock_states):
    H_block = H_mat[state][:, state].reshape(len(state), len(state))
    evals, evecs = np.linalg.eigh(H_block)
    H_evals.append(evals)
    H_evecs.append(evecs)
    tr_exp_minusbetaH += np.sum(np.exp(-beta * evals))
eta_0 = np.log(tr_exp_minusbetaH) / beta
G_blocks = []
tau_mesh = MeshDLRImTime(beta = beta, statistic = 'Fermion', w_max = lamb / beta, eps = eps)
for s, state in enumerate(ad.fock_states):
    G_block = Gf(mesh = tau_mesh, target_shape = [len(state), len(state)])
    for tau in tau_mesh:
        G_block[tau] = H_evecs[s] @ np.diag(-np.exp(-beta * tau * (H_evals[s] + eta_0))) @ H_evecs[s].T.conj()
    G_blocks.append(G_block)

# Construct G_ppsc as the atomic propagator
G_ppsc = BlockGf(block_list = G_blocks)

# ===== Hybridization function fitting =====
N = 55
Z = 1j *(np.linspace(-N, N, N + 1)) * np.pi / beta
Delta = make_Delta_with_cont_spec_mat(Z, semicircular, a=-1.0, b=1.0, r0=0.5, eps=eps)
Np = 4
func, fitting_error, pol, weight = anacont(Delta, Z, tol=eps)

d = DiagramEvaluator(beta, lamb, eps, pol, weight, G_ppsc, ad)

topology = [[0, 2], [1, 3]]

# Serial diagram evaluation
Sigma_ppsc_ser = d.compute_self_energy(topology)

# Parallel diagram evaluation
N = d.get_num_backbones(topology)

par_blocks = []
for b in range(ad.n_subspaces):
    par_block = Gf(mesh = tau_mesh, target_shape = G_ppsc[b].target_shape)
    par_blocks.append(par_block)
Sigma_ppsc_par = BlockGf(block_list = par_blocks)

for n in range(N):
    Sigma_ppsc_par += d.compute_self_energy(topology, n)

# Test that serial and parallel evaluations give the same result
for tau in tau_mesh:
    assert(np.max(np.abs(Sigma_ppsc_ser[0][tau] - Sigma_ppsc_par[0][tau])) < 1e-13)