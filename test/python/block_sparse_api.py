

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


class Dummy():
    def __init__(self): pass


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
    lamb = 5.0 * beta
    w_max = lamb / beta

    # -- Local Hamiltonian
    
    n_orb = 2
    spin_names = ('up', 'dn')
    gf_struct = [['up', 2], ['dn', 2]]

    fops = [ (sn, on) for sn, on in product(spin_names, range(n_orb)) ]

    from triqs.operators import n
    
    N_up = n('up', 0) + n('up', 1)
    N_dn = n('dn', 0) + n('dn', 1)

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

    
    # -- Block sparse solver

    BSS = BlockSparseSolver(
        H, fops, beta, w_max, eps,
        #conserved_operators=[],
        #conserved_operators=[N_up + N_dn],
        conserved_operators=[N_up, N_dn],
        )

    #BSS.set_hybridization(Delta_w)
    BSS.set_hybridization_poles_and_coefficients(pol, weight)

    BSS.init_diagram_evaluator()

    
    # -- Dense solver

    S = TriqsSolver(beta=beta, gf_struct=gf_struct, eps=eps, w_max=w_max)

    for sidx in spin_names:
        S.Delta_tau[sidx] << Delta_tau

    S.solve(h_int=H, order=2, tol=eps, maxiter=0, compress_hybridization=True)


    # -- Compare pseudo particle Green's function

    G_S = S.S.G0_iaa
    
    G_BSS = pseudo_particle_block_gf_to_dense(
        BSS.pseudo_particle_greens_function(), BSS.ad)

    G_diff = np.max(np.abs(G_BSS.data - G_S))
    print(f'G_diff = {G_diff:2.2E}')

    
    # -- Compare self-energy topologies

    from triqs_xca.diag import all_connected_pairings

    results = dict()
    
    #for order in [1, 2, 3]:
    for order in [1, 2]:
        print(f'order = {order}')
        
        for sign, topology in all_connected_pairings(order):
            
            print(f'  topology = {topology}')

            d = Dummy()
            d.order = order
            d.topology = topology
            d.sign = sign
            d.tau = S.S.tau_i
             
            topology = np.array(topology, dtype=np.int32)

            import time

            t1 = time.time()
            
            d.Sigma_S = S.S.calc_Sigma_topology(topology)

            t2 = time.time()

            d.Sigma_BSS = pseudo_particle_block_gf_to_dense(
                BSS.pseudo_particle_self_energy_topology(topology), BSS.ad)

            t3 = time.time()

            print(f'    Sigma time ZH ({t2 - t1} s) BS ({t3 - t2} s)')
            
            #Sigma_BSS_loop = pseudo_particle_block_gf_to_dense(
            #    BSS.pseudo_particle_self_energy_topology_loop(topology), BSS.ad)
            #np.testing.assert_array_almost_equal(d.Sigma_BSS.data, Sigma_BSS_loop.data)
            
            d.Sigma_diff = np.max(np.abs(d.Sigma_BSS.data - d.Sigma_S))
            print(f'    Sigma_diff = {d.Sigma_diff:2.2E}')

            t1 = time.time()
            d.spgf_S = S.S.calc_spgf_toplogy(topology)
            t2 = time.time()
            d.spgf_BSS = BSS.single_particle_greens_function_topology(topology)
            t3 = time.time()

            print(f'    spgf time ZH ({t2 - t1} s) BS ({t3 - t2} s)')
            
            #spgf_BSS_loop = BSS.single_particle_greens_function_topology_loop(topology)
            #np.testing.assert_array_almost_equal(d.spgf_BSS.data, spgf_BSS_loop.data)
            
            d.spgf_diff = np.max(np.abs(d.spgf_BSS.data - d.spgf_S))
            print(f'    spgf_diff = {d.spgf_diff:2.2E}')

            results[tuple(d.topology)] = d


    # -- Vizualize

    if verbose:

        from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti

        plt.figure(figsize=(6, 8))
        subp = [3, 1, 1]

        plt.subplot(*subp); subp[-1] += 1

        oplot(G_BSS, label=None)
        
        for i,j in product(range(16), repeat=2):
            plt.plot(S.S.tau_i, G_S[:, i, j].real, '+-')

        plt.plot([], [], 'x', color='gray', label='Block-sparse (PR)')
        plt.plot([], [], '+-', color='gray', label='Dense (ZH)')
        plt.legend()
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$G_0(\tau)$')

        # -- Hybridization function
        
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
        
        plt.tight_layout()        

        
        plt.figure(figsize=(14, 12))
        subp = [6, 4, 1]

        for topology, t in results.items():
            
            # -- Self-energy

            plt.subplot(*subp); subp[-1] += 1

            oplot(t.Sigma_BSS, label=None)
            for i,j in product(range(t.Sigma_S.shape[-1]), repeat=2):
                plt.plot(t.tau, t.Sigma_S[:, i, j].real, '+-')

            plt.plot([], [], 'x', color='gray', label='Block-sparse (PR)')
            plt.plot([], [], '+-', color='gray', label='Dense (ZH)')
            plt.legend(fontsize=7)
            plt.xlabel(r'$\tau$')
            plt.ylabel(r'$\Sigma_{' + f'{t.topology}' + r'}(\tau)$')

            # -- Difference in Self-energy

            plt.subplot(*subp); subp[-1] += 1
            for i,j in product(range(t.Sigma_S.shape[-1]), repeat=2):
                data = t.Sigma_S[:, i, j].real - t.Sigma_BSS.data[:, i, j].real
                if not (data == 0).all():
                    plt.plot(t.tau, data, 'o-')

            plt.xlabel(r'$\tau$')
            plt.ylabel(r'Difference' + '\n' + r'$\Sigma_{' + f'{t.topology}' + r'}(\tau)$')

            # -- spgf

            plt.subplot(*subp); subp[-1] += 1

            oplot(t.spgf_BSS, label=None)
            for i,j in product(range(t.spgf_S.shape[-1]), repeat=2):
                plt.plot(t.tau, t.spgf_S[:, i, j].real, '+-')

            plt.plot([], [], 'x', color='gray', label='Block-sparse (PR)')
            plt.plot([], [], '+-', color='gray', label='Dense (ZH)')
            plt.legend(fontsize=7)
            plt.xlabel(r'$\tau$')
            plt.ylabel(r'$g_{' + f'{t.topology}' + r'}(\tau)$')

            # -- Difference in spgf

            plt.subplot(*subp); subp[-1] += 1
            for i,j in product(range(t.spgf_S.shape[-1]), repeat=2):
                data = t.spgf_S[:, i, j].real - t.spgf_BSS.data[:, i, j].real
                if not (data == 0).all():
                    plt.plot(t.tau, data, 'o-')

            plt.xlabel(r'$\tau$')
            plt.ylabel(r'Difference' + '\n' + r'$g_{' + f'{t.topology}' + r'}(\tau)$')
            

        plt.tight_layout()
        plt.savefig('figure_xca_top_cf.pdf')
        plt.show()


    H_mat = hamiltonian_matrix(BSS.ad)    
    np.testing.assert_array_almost_equal(H_mat, S.S.H_mat)
    
    np.testing.assert_array_almost_equal(G_BSS.data, G_S)

    for topology, t in results.items():
        print(f'topology = {t.topology}')
        np.testing.assert_array_almost_equal(t.Sigma_BSS.data, t.Sigma_S)
        np.testing.assert_array_almost_equal(t.spgf_BSS.data, t.spgf_S)


if __name__ == '__main__':

    test_oca_diagram_cf_block_sparse_and_dense(beta=2.0, verbose=False)
     
