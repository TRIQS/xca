import numpy as np

from itertools import product

from triqs.gf import Gf, MeshDLRImTime, inverse, iOmega_n
from triqs.operators.util.hamiltonians import h_int_kanamori, make_operator_real

from triqs_xca.triqs_solver import TriqsSolver
from triqs_xca.block_sparse_solver import BlockSparseSolver
from triqs_xca.block_sparse_solver import hamiltonian_matrix, pseudo_particle_block_gf_to_dense


class Dummy():
    def __init__(self): pass


def test_diagrams_cf_block_sparse_and_dense(e1=-1.5, beta=2.0, conserved_operators='none', verbose=False):

    print('='*72)
    print('='*72)
    print(f'beta = {beta}')
    print(f'e1 = {e1}')
    print(f'conserved_operators = {conserved_operators}')
    print('='*72)
    print('='*72)
    
    # -- Parameters
    
    a = -1.0
    b = +1.0
    r0 = 0.5
    mu = 0.3
    U = 3.0

    eps = 1e-12
    w_max = 20.0

    # -- Local Hamiltonian
    
    gf_struct = [['0', 2]]

    from triqs.operators import n

    N_0 = n('0', 0)
    N_1 = n('0', 1)
    N_op = N_0 + N_1
    
    H = -mu * N_op + U * n('0', 0) * n('0', 1)

    print(H)
    print(N_op)

    conserved_operators = dict(
        none=[],
        total_density=[N_op],
        individual_density=[N_0, N_1],
        automatic='automatic',
        )[conserved_operators]
    
    print(f'conserved_operators = {conserved_operators}')
    
    # -- Hybridization function and adapol fit

    from triqs.gf import MeshDLRImFreq

    mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps)
    Delta_w = Gf(mesh=mesh_w, target_shape=[2]*2)
    iwn = np.array([ complex(x) for x in mesh_w ])

    Delta_w << inverse(iOmega_n - e1)

    from triqs.gf import make_gf_dlr_imtime, make_gf_dlr

    Delta_tau = make_gf_dlr_imtime(Delta_w)
    Delta_dlr = make_gf_dlr(Delta_w)


    # -- Dense solver

    S = TriqsSolver(beta=beta, gf_struct=gf_struct, eps=eps, w_max=w_max)

    S.Delta_tau['0'] << Delta_tau

    S.solve(h_int=H, order=2, tol=eps, maxiter=0, compress_hybridization=True)

    
    # -- Block sparse solver

    BSS = BlockSparseSolver(
        H, beta, w_max, eps, gf_struct=gf_struct,
        conserved_operators=conserved_operators, # calling DiagramEvaluator with a list of operators segfaults!
        )

    BSS.Delta_tau['0'] << Delta_tau

    BSS.fit_hybridization(tol=eps)


    # -- Compare pseudo particle Green's function

    G_S = S.S.G0_iaa
    
    G_BSS = BSS.pseudo_particle_greens_function()
    G_BSS = pseudo_particle_block_gf_to_dense(G_BSS, BSS.ad)

    G_diff = np.max(np.abs(G_BSS.data - G_S))
    print(f'G_diff = {G_diff:2.2E}')

    Z = BSS.partition_function()
    print(f'Z = {Z}')
    np.testing.assert_almost_equal(Z, 1.0)

    G_DYSON_BSS = BSS.solve_dyson(BSS.Sigma, BSS.eta) # Solving Dyson with zero self-energy
    G_DYSON_BSS = pseudo_particle_block_gf_to_dense(G_DYSON_BSS, BSS.ad)
    np.testing.assert_array_almost_equal(G_DYSON_BSS.data, G_S)

    results_by_order = dict()
    

    # -- Compare self-energy topologies

    from triqs_xca.diag import all_connected_pairings

    results = dict()
    
    for order in [1, 2, 3]:
        print(f'order = {order}')
        
        d = Dummy()
        d.order = order
        
        d.Sigma_S = S.S.calc_Sigma(d.order)
        d.spgf_S = S.S.calc_spgf(d.order)

        d.Sigma_BSS = pseudo_particle_block_gf_to_dense(BSS.eval_pseudo_particle_self_energy(BSS.G, d.order), BSS.ad)
        d.spgf_BSS = BSS.eval_single_particle_greens_function(BSS.G, d.order)

        results_by_order[order] = d

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

            d.Sigma_BSS = BSS.eval_pseudo_particle_self_energy_topology(BSS.G, topology)
            d.Sigma_BSS = pseudo_particle_block_gf_to_dense(d.Sigma_BSS, BSS.ad)

            t3 = time.time()

            print(f'    Sigma time ZH ({t2 - t1} s) BS ({t3 - t2} s)')
            
            d.Sigma_diff = np.max(np.abs(d.Sigma_BSS.data - d.Sigma_S))
            print(f'    Sigma_diff = {d.Sigma_diff:2.2E}')

            t1 = time.time()
            d.spgf_S = S.S.calc_spgf_toplogy(topology)
            t2 = time.time()
            d.spgf_BSS = BSS.eval_single_particle_greens_function_topology(BSS.G, topology) 
            t3 = time.time()

            print(f'    spgf time ZH ({t2 - t1} s) BS ({t3 - t2} s)')
            
            d.spgf_diff = np.max(np.abs(d.spgf_BSS.data - d.spgf_S))
            print(f'    spgf_diff = {d.spgf_diff:2.2E}')

            results[tuple(d.topology)] = d


    # -- Vizualize

    if verbose:

        from triqs.plot.mpl_interface import oplot, plt, oplotr, oploti

        plt.figure(figsize=(6, 8))
        subp = [4, 1, 1]

        plt.subplot(*subp); subp[-1] += 1

        oplot(G_BSS, label=None)
        
        for i,j in product(range(G_S.shape[-1]), repeat=2):
            plt.plot(S.S.tau_i, G_S[:, i, j].real, '+-')

        plt.plot([], [], 'x', color='gray', label='Block-sparse (PR)')
        plt.plot([], [], '+-', color='gray', label='Dense (ZH)')
        plt.legend()
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$G_0(\tau)$')

        # -- Hybridization function
        plt.subplot(*subp); subp[-1] += 1
        oplot(S.Delta_tau)
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$\Delta(\tau)$]')
        
        plt.subplot(*subp); subp[-1] += 1
        oplotr(Delta_w)
        plt.legend()
        plt.xlabel(r'$\omega_n$')
        plt.ylabel(r'Re[$\Delta(i\omega_n)$]')

        plt.subplot(*subp); subp[-1] += 1
        oploti(Delta_w)
        plt.legend()
        plt.ylabel(r'Im[$\Delta(i\omega_n)$]')
        plt.xlabel(r'$\omega_n$')
        
        plt.tight_layout()        

        
        plt.figure(figsize=(14, 12))
        subp = [6, 4, 1]
        #subp = [34, 4, 1]

        print(len(results))
        #exit()

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

        # -- Analysis of 3rd order topolgy
        
        topology = ((0, 3), (1, 4), (2, 5))
        t = results[topology]

        plt.show()

    if not verbose:

        H_mat = hamiltonian_matrix(BSS.ad)    
        np.testing.assert_array_almost_equal(H_mat, S.S.H_mat)

        np.testing.assert_array_almost_equal(G_BSS.data, G_S)

        for order, t in results_by_order.items():
            print(f'order = {order}')
            np.testing.assert_array_almost_equal(t.Sigma_BSS.data, t.Sigma_S)
            np.testing.assert_array_almost_equal(t.spgf_BSS.data, t.spgf_S)
            print('  Passed')

        for topology, t in results.items():
            print(f'topology = {t.topology}')
            np.testing.assert_array_almost_equal(t.Sigma_BSS.data, t.Sigma_S)
            np.testing.assert_array_almost_equal(t.spgf_BSS.data, t.spgf_S)


if __name__ == '__main__':

    ops = [
        'none', 
        'total_density',
        'individual_density',
        'automatic',
        ]
    
    for e1 in [+1.5, -1.5]:
        for op in ops:
            test_diagrams_cf_block_sparse_and_dense(
                e1=e1, beta=2.0, conserved_operators=op, verbose=False)
     
