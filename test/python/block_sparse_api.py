

from mpi4py import MPI

from itertools import product

import numpy as np
from scipy.integrate import quad

from triqs.gf import Gf, MeshDLRImTime, BlockGf
from triqs.operators import n
from triqs.operators.util.hamiltonians import h_int_kanamori, make_operator_real
from triqs.atom_diag import AtomDiag

from adapol import anacont

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
a = -1.0
b = +1.0
r0 = 0.5

beta = 2.0
eps = 1e-12
lamb = 20.0 * beta
w_max = lamb / beta

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

if False:
    print(dir(ad))
    print(f'full_hilbert_space_dim = {ad.full_hilbert_space_dim}')
    print(f'n_subspaces = {ad.n_subspaces}')
    print(f'get_subspace_dims = {ad.get_subspace_dims()}')
    print(f'fock_states = {ad.fock_states}')
    print(f'gs_energy = {ad.gs_energy}')
    print(f'energies = {ad.energies}')
    print(f'get_eigenvalue = {ad.get_eigenvalue(0, 0)}')
    print(f'h_atomic = {ad.h_atomic}')
    print(f'fops = {ad.fops}')

    op_idx = 0
    sp_idx = 0
    print(f'c_connection = {ad.c_connection(op_idx, sp_idx)}')
    print(f'c_matrix = \n{ad.c_matrix(op_idx, sp_idx)}')
    exit()

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
        #G_block[tau] = H_evecs[s] @ np.diag(-np.exp(-beta * tau * (H_evals[s] + eta_0))) @ H_evecs[s].T.conj() # Bug! extra factor of _beta_
        G_block[tau] = H_evecs[s] @ np.diag(-np.exp(-tau * (H_evals[s] + eta_0))) @ H_evecs[s].T.conj()
    G_blocks.append(G_block)

# Construct G_ppsc as the atomic propagator
G_ppsc = BlockGf(block_list = G_blocks)

# ===== Hybridization function fitting =====

N = 55
Z = 1j *(np.linspace(-N, N, N + 1)) * np.pi / beta
Delta = make_Delta_with_cont_spec_mat(Z, semicircular, a=a, b=b, r0=r0, eps=eps)
Np = 4 # unused?
func, fitting_error, pol, weight = anacont(Delta, Z, tol=eps)

from triqs.gf import MeshDLRImFreq

mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=lamb/beta, eps=eps)
Delta_w = Gf(mesh=mesh_w, target_shape=[2]*2)
iwn = np.array([ complex(x) for x in mesh_w ])
Delta_w.data[:] = make_Delta_with_cont_spec_mat(iwn, semicircular, a=a, b=b, r0=r0, eps=eps)[:, :2, :2]

from triqs.gf import make_gf_dlr_imtime, make_gf_dlr

Delta_tau = make_gf_dlr_imtime(Delta_w)
Delta_dlr = make_gf_dlr(Delta_w)

# ===== Evaluate diagrams! =====

d = DiagramEvaluator(beta, lamb, eps, pol, weight, G_ppsc, ad)

topology = np.array([[0, 2], [1, 3]], dtype=np.int32)

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


# ===== Construct dense solver and compare =====

gf_struct = [ ['up', 2], ['dn', 2] ]

from triqs_xca.triqs_solver import TriqsSolver
S = TriqsSolver(beta=beta, gf_struct=gf_struct, eps=eps, w_max=w_max)

#print(pol)
#print(weight)

# Low level hybridization API
#S.S.fd.hyb_init(delta_iaa, poledlrflag=False)
#S.S.fd.copy_aaa_result(pol, weight)
#eps_svd = 1e-12
#S.S.fd.hyb_decomposition(poledlrflag=False, eps=eps_svd)

# High level hybridization API (no control over compression)
S.Delta_tau['up'] << Delta_tau
S.Delta_tau['dn'] << Delta_tau

S.solve(h_int=H, order=2, tol=eps, maxiter=0)

#from triqs_xca.solver import Sigma_calc_loop
#max_order = 2
#Sigma_dense = Sigma_calc_loop(S.S.fd, S.S.G0_iaa, max_order)

#from triqs_xca.diag import all_connected_pairings
#for order in range(1, max_order+1):
#    for sign, diag in all_connected_pairings(order):
#        print(sign, diag)

sign = -1
order = 2
diag = [(0, 2), (1, 3)]
n_diags = S.S.fd.number_of_diagrams(order)

print(f'n_diags = {n_diags}')

diag_vec = np.vstack([ np.array(pair, dtype=np.int32) for pair in diag ])
diag_idx_vec = np.arange(n_diags, dtype=np.int32)
Sigma_iaa = pow(-1, order) * sign * S.S.fd.Sigma_calc_group(S.S.G0_iaa, diag_vec, diag_idx_vec)

np.testing.assert_array_almost_equal(H_mat, S.S.H_mat)

def ppsc_BlockGf_to_dense(block_gf, ad):
    
    # Get mesh
    mesh = block_gf[0].mesh
    print(mesh)
    # make dense
    target_shape = [ad.full_hilbert_space_dim]*2
    dense_gf = Gf(mesh=mesh, target_shape=target_shape)

    for sidx in range(ad.n_subspaces):
        #U = ad.unitary_matrices[sidx]
        #E = ad.energies[sidx]
        #H_block_diag = np.diag(E + ad.gs_energy)
        #H_block = U @ H_block_diag @ U.T.conj()

        fidx = ad.fock_states[sidx]
        bidx = np.ix_(range(len(mesh)), fidx, fidx)

        dense_gf.data[bidx] = block_gf[sidx].data

    return dense_gf
    
G_ppsc_dense = ppsc_BlockGf_to_dense(G_ppsc, ad)
Sigma_ppsc_dense = ppsc_BlockGf_to_dense(Sigma_ppsc_ser, ad)

np.testing.assert_array_almost_equal(G_ppsc_dense.data, S.S.G0_iaa)
#np.testing.assert_array_almost_equal(Sigma_ppsc_dense.data, Sigma_iaa)
    
# ===== Vizualize =====

if False:

    from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti

    plt.figure(figsize=(8, 8))
    subp = [4, 2, 1]

    plt.subplot(*subp); subp[-1] += 1
    for i,j in product(range(4), repeat=2):
        if not (Delta[:, i, j].real == 0.).all():
            plt.plot(Z.imag, Delta[:, i, j].real, label=f'i,j = {i},{j}')
    plt.legend()
    plt.xlabel(r'$\omega_n$')
    plt.ylabel(r'Re[$\Delta(i\omega_n)$]')

    plt.subplot(*subp); subp[-1] += 1
    for i,j in product(range(4), repeat=2):
        if not (Delta[:, i, j].imag == 0.).all():
            plt.plot(Z.imag, Delta[:, i, j].imag, label=f'i,j = {i},{j}')
    plt.legend()
    plt.ylabel(r'Im[$\Delta(i\omega_n)$]')
    plt.xlabel(r'$\omega_n$')

    plt.subplot(*subp); subp[-1] += 1
    oplot(G_ppsc)
    plt.gca().get_legend().remove()
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$G_0(\tau)$ bs')

    plt.subplot(*subp); subp[-1] += 1
    for i,j in product(range(16), repeat=2):
        data = S.S.G0_iaa[:, i, j].real
        if not (data == 0).all():
            plt.plot(S.S.tau_i, data, 'o-')

    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$G_0(\tau)$ zh')

    plt.subplot(*subp); subp[-1] += 1
    oplot(Sigma_ppsc_ser)
    plt.gca().get_legend().remove()

    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\Sigma_{OCA}(\tau)$ bs')

    plt.subplot(*subp); subp[-1] += 1
    for i,j in product(range(16), repeat=2):
        data = Sigma_iaa[:, i, j].real
        if not (data == 0).all():
            plt.plot(S.S.tau_i, data, 'o-')

    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$\Sigma_{OCA}(\tau)$ zh')
            
    plt.subplot(*subp); subp[-1] += 1
    for i,j in product(range(16), repeat=2):
        data = Sigma_iaa[:, i, j].real - Sigma_ppsc_dense.data[:, i, j].real
        if not (data == 0).all():
            plt.plot(S.S.tau_i, data, 'o-')

    plt.xlabel(r'$\tau$')
    plt.ylabel(r'Difference $\Sigma_{OCA}(\tau)$')
            
    plt.tight_layout()
    plt.show()
