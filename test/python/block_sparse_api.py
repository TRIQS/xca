

from itertools import product

import numpy as np
from scipy.integrate import quad

import triqs.utility.mpi as mpi

from triqs.gf import Gf, MeshDLRImTime
from triqs.operators.util.hamiltonians import h_int_kanamori, make_operator_real

from adapol import anacont


from triqs_xca.triqs_solver import TriqsSolver

from triqs_xca.block_sparse_solver import BlockSparseSolver

from triqs_xca.block_sparse_solver import hamiltonian_matrix, pseudo_particle_block_gf_to_dense


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


def test_oca_diagram_cf_block_sparse_and_dense(beta=2.0, verbose=False):

    print('='*72)
    print('='*72)
    print(f'beta = {beta}')
    print('='*72)
    print('='*72)
    
    # -- Parameters
    
    a = -1.0
    b = +1.0
    r0 = 0.5
    mu = 4.0

    eps = 1e-12
    lamb = 20.0 * beta
    w_max = lamb / beta

    
    # -- Local Hamiltonian
    
    n_orb = 2
    spin_names = ('up','dn')

    fops = [ (sn,on) for sn, on in product(spin_names, range(n_orb)) ]

    from triqs.operators import n
    N_up = n('up',0) + n('up',1)
    N_dn = n('dn',0) + n('dn',1)

    U = 3.0 * np.ones((2,2))
    Uprime = 2.0 * np.ones((2,2))
    J_hund = 0.5

    H = h_int_kanamori(spin_names, n_orb, U, Uprime, J_hund, off_diag=True) # not sure about off_diag

    H += -mu * (N_up + N_dn)

    
    # -- Hybridization function and adapol fit

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

    
    # -- Self-energy diagram topology

    topology = np.array([[0, 2], [1, 3]], dtype=np.int32)    
    sign = -1
    order = len(topology)    

    
    # -- Block sparse solver

    BSS = BlockSparseSolver(H, fops, beta, w_max, eps, conserved_operators=[N_up + N_dn])

    #BSS.set_hybridization(Delta_w)
    BSS.set_hybridization_poles_and_coefficients(pol, weight)

    BSS.init_diagram_evaluator()

    G_ppsc = BSS.pseudo_particle_greens_function()
    G_ppsc_dense = pseudo_particle_block_gf_to_dense(G_ppsc, BSS.ad)

    Sigma_ppsc = BSS.pseudo_particle_self_energy_topology(topology)
    Sigma_ppsc_dense = pseudo_particle_block_gf_to_dense(Sigma_ppsc, BSS.ad)

    # -- Test the loop version of Sigma calc
    Sigma_ppsc_loop = BSS.pseudo_particle_self_energy_topology_loop(topology)
    Sigma_ppsc_loop_dense = pseudo_particle_block_gf_to_dense(Sigma_ppsc_loop, BSS.ad)

    spgf = BSS.single_particle_greens_function_topology(topology)
    spgf_loop = BSS.single_particle_greens_function_topology_loop(topology)


    # -- Dense solver

    S = TriqsSolver(beta=beta, gf_struct=[['up', 2], ['dn', 2]], eps=eps, w_max=w_max)

    S.Delta_tau['up'] << Delta_tau
    S.Delta_tau['dn'] << Delta_tau

    S.solve(h_int=H, order=2, tol=eps, maxiter=0, compress_hybridization=True)
    
    Sigma_iaa = pow(-1, order) * sign * S.S.calc_Sigma_topology(topology)

    spgf_iaa = S.S.calc_spgf_toplogy(topology)
    
    # -- Compare block-sparse and dense results
    
    G_diff = np.max(np.abs(G_ppsc_dense.data - S.S.G0_iaa))
    print(f'G_diff = {G_diff:2.2E}')
    
    Sigma_diff = np.max(np.abs(Sigma_ppsc_dense.data - Sigma_iaa))
    print(f'Sigma_diff = {Sigma_diff:2.2E}')

    spgf_diff = np.max(np.abs(spgf.data - spgf_iaa))
    print(f'spgf_diff = {spgf_diff:2.2E}')
    
    # -- Vizualize

    if verbose:

        from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti

        plt.figure(figsize=(8, 6))
        subp = [2, 2, 1]

        plt.subplot(*subp); subp[-1] += 1

        oplot(G_ppsc, label=None)
        
        for i,j in product(range(16), repeat=2):
            plt.plot(S.S.tau_i, S.S.G0_iaa[:, i, j].real, '+-')

        plt.plot([], [], 'x', color='gray', label='Block-sparse (PR)')
        plt.plot([], [], '+-', color='gray', label='Dense (ZH)')
        plt.legend()
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$G_0(\tau)$')
        
        plt.subplot(*subp); subp[-1] += 1

        oplot(Sigma_ppsc_ser, label=None)
        for i,j in product(range(16), repeat=2):
            plt.plot(S.S.tau_i, Sigma_iaa[:, i, j].real, '+-')

        plt.plot([], [], 'x', color='gray', label='Block-sparse (PR)')
        plt.plot([], [], '+-', color='gray', label='Dense (ZH)')
        plt.legend()
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$\Sigma_{OCA}(\tau)$')

        plt.subplot(*subp); subp[-1] += 1
        for i,j in product(range(16), repeat=2):
            data = Sigma_iaa[:, i, j].real - Sigma_ppsc_dense.data[:, i, j].real
            if not (data == 0).all():
                plt.plot(S.S.tau_i, data, 'o-')

        plt.xlabel(r'$\tau$')
        plt.ylabel(r'Difference $\Sigma_{OCA}(\tau)$')

        # -- Hybridization function
        
        if False:
            plt.subplot(*subp); subp[-1] += 1
            for i,j in product(range(4), repeat=2):
                if not (Delta[:, i, j].real == 0.).all():
                    plt.plot(Z.imag, Delta[:, i, j].real, label=f'i,j = {i},{j}')
            plt.legend()
            plt.xlabel(r'$\omega_n$')
            plt.ylabel(r'Re[$\Delta(i\omega_n)$]')
        else:
            assert( (Delta.real == 0.0).all() )

        plt.subplot(*subp); subp[-1] += 1
        for i,j in product(range(4), repeat=2):
            if not (Delta[:, i, j].imag == 0.).all():
                plt.plot(Z.imag, Delta[:, i, j].imag, label=f'i,j = {i},{j}')
        plt.legend()
        plt.ylabel(r'Im[$\Delta(i\omega_n)$]')
        plt.xlabel(r'$\omega_n$')

        plt.tight_layout()        
        plt.show()


    # -- Asserts

    H_mat = hamiltonian_matrix(BSS.ad)    
    np.testing.assert_array_almost_equal(H_mat, S.S.H_mat)
    
    np.testing.assert_array_almost_equal(G_ppsc_dense.data, S.S.G0_iaa)

    np.testing.assert_array_almost_equal(Sigma_ppsc_dense.data, Sigma_iaa)
    np.testing.assert_array_almost_equal(Sigma_ppsc_loop_dense.data, Sigma_ppsc_dense.data)

    np.testing.assert_array_almost_equal(spgf.data, spgf_iaa)
    np.testing.assert_array_almost_equal(spgf_loop.data, spgf.data)
    
        
if __name__ == '__main__':

    test_oca_diagram_cf_block_sparse_and_dense(beta=1.0, verbose=False)
    test_oca_diagram_cf_block_sparse_and_dense(beta=2.0, verbose=False)
     
