r""" Analytic test of diagrammatics for a single Fermion
(and comparison to Z. Huang's code)

Using a hybridization function with a single pole $\omega$

..math::
    \Delta(\tau) = K(\tau, \omega)

and degenerate atomic states giving a pseudo particle Green's function
that is proportional to the identity with a single exponential decay

..math:: 
    G(\tau) = - 2^{-\tau / \beta}

with $G(\beta) = -1/2$.

In this case it is possible to derive analytic expressions for the self energy
and single particle Green's function diagrams, see separate notebook in the examples folder.

"""

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


def get_analytic_solution():

    """ For details see separate notebook in the examples folder """

    import sympy as sp

    class Dummuy():
        def __init__(self): pass

    d = Dummy()

    t, t1, t2, t3, t4 = sp.symbols(r'\tau \tau_1 \tau_2 \tau_3 \tau_4', positive=True)
    b, w = sp.symbols(r'\beta \omega', nonzero=True)
    K = lambda t, w : -sp.exp(-w*t)/(1 + sp.exp(-b*w))

    # -- ppgf
    
    G = -sp.exp(-sp.ln(2) / b * t).simplify()
    d.Gfunc = sp.lambdify([t, b], G)

    # -- 1st order Sigma
    
    Sigma_01_00 = K(t, -w) * G
    Sigma_01_11 = K(t, w) * G 

    d.Sfunc_01_00 = sp.lambdify([t, b, w], Sigma_01_00)
    d.Sfunc_01_11 = sp.lambdify([t, b, w], Sigma_01_11)

    # -- 3rd order Sigma

    I4 = sp.integrate(sp.exp(+w*t4), (t4, 0, t3))
    I3 = sp.integrate(sp.exp(-w*t3) * I4, (t3, 0, t2)) 
    I2 = sp.integrate(sp.exp(+w*t2) * I3, (t2, 0, t1)).simplify()
    I1 = sp.integrate(sp.exp(-w*t1) * I2, (t1, 0, t))

    Sigma_031425_00 = G * K(0, -w)**2 * K(0, w) * sp.exp(w*t) * I1
    Sigma_031425_00 = Sigma_031425_00.simplify()

    d.Sfunc_031425_00 = sp.lambdify([t, b, w], Sigma_031425_00)

    I4 = sp.integrate(sp.exp(-w*t4), (t4, 0, t3))
    I3 = sp.integrate(sp.exp(+w*t3) * I4, (t3, 0, t2)) 
    I2 = sp.integrate(sp.exp(-w*t2) * I3, (t2, 0, t1)).simplify()
    I1 = sp.integrate(sp.exp(+w*t1) * I2, (t1, 0, t))

    Sigma_031425_11 = G * K(0, w)**2 * K(0, -w) * sp.exp(-w*t) * I1
    Sigma_031425_11 = Sigma_031425_11.simplify()

    d.Sfunc_031425_11 = sp.lambdify([t, b, w], Sigma_031425_11)

    # -- 3rd order spgf

    IL = sp.integrate(sp.exp(w*(t1 - t2)), (t2, t, t1), (t1, t, b))
    IR = sp.integrate(sp.exp(w*(-t3 + t4)), (t4, 0, t3), (t3, 0, t))
    
    spgf_031425 = K(0, w) * K(0, -w) * IL * IR / 2
    spgf_031425 = spgf_031425.simplify()

    d.spgf_func_031425 = sp.lambdify([t, b, w], spgf_031425)
    
    return d



class Dummy():
    def __init__(self): pass


def Kw(w, v):
    return 1 / ( v - w)


def semicircular(x):
    return 2 * np.sqrt(1 - x**2) / np.pi


def make_Delta_with_cont_spec_mat( Z, rho, a=-1.0, b=1.0, eps=1e-12):
    Delta = np.zeros((Z.shape[0]), dtype=np.complex128)

    for n in range(len(Z)):
        def f(w):
            return Kw(w , Z[n]) * rho(w)

        Delta[n] = quad(
            f, a, b, epsabs=eps, epsrel=eps, complex_func=True
        )[0]
        
    T = np.array([[1.]])

    return Delta[:, None, None] * T[None, :, :]


def test_oca_diagram_cf_block_sparse_and_dense(
        e1=0.8, beta=2.0, conserved_operators='none', verbose=False, dense=False):

    print('='*72)
    print('='*72)
    print(f'beta = {beta}')
    print('='*72)
    print('='*72)
    
    # -- Parameters
    
    a = -1.0
    b = +1.0
    r0 = 0.5
    mu = 0.0
    
    eps = 1e-12
    #lamb = 20.0 * beta
    lamb = 200.0 * beta
    
    w_max = lamb / beta

    # -- Local Hamiltonian
    
    gf_struct = [['0', 1]]

    fops = [ ('0', 0) ]

    from triqs.operators import n
    
    N_op = n('0', 0) 
    
    H = -mu * N_op

    print(H)
    print(N_op)

    conserved_operators = dict(
        none=[],
        total_density=[N_op],
        )[conserved_operators]
    
    print(f'conserved_operators = {conserved_operators}')    
    
    # -- Hybridization function and adapol fit

    if True:
        N = 55
        Z = 1j *(np.linspace(-N, N, N + 1)) * np.pi / beta
        #Delta = make_Delta_with_cont_spec_mat(Z, semicircular, a=a, b=b, eps=eps)
        #Delta = (1./Z).reshape(len(Z), 1, 1)
        Delta = (1./(Z - e1)).reshape(len(Z), 1, 1)
        #Delta = (1./(Z - e1) + 1./(Z - e2)).reshape(len(Z), 1, 1)
        Np = 4 # unused?
        func, fitting_error, pol, weight = anacont(Delta, Z, tol=eps)
        print(f'pol = {pol}')
        print(f'weight = {weight}')
    else:
        pol = np.array([0.])
        weight = np.array([[[1.]]], dtype=complex)

    from triqs.gf import MeshDLRImFreq

    mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=lamb/beta, eps=eps)
    Delta_w = Gf(mesh=mesh_w, target_shape=[1]*2)
    iwn = np.array([ complex(x) for x in mesh_w ])

    #Delta_w.data[:] = make_Delta_with_cont_spec_mat(iwn, semicircular, a=a, b=b, eps=eps)
    #Delta_w.data[:, 0, 0] = 1./iwn
    Delta_w.data[:, 0, 0] = 1./(iwn - e1)
    #Delta_w.data[:, 0, 0] = 1./(iwn - e1) + 1./(iwn - e2)

    from triqs.gf import make_gf_dlr_imtime, make_gf_dlr

    Delta_tau = make_gf_dlr_imtime(Delta_w)
    Delta_dlr = make_gf_dlr(Delta_w)


    # -- Dense solver

    S = TriqsSolver(beta=beta, gf_struct=gf_struct, eps=eps, w_max=w_max)

    S.Delta_tau['0'] << Delta_tau

    S.solve(h_int=H, order=2, tol=eps, maxiter=0, compress_hybridization=True)

    
    # -- Block sparse solver
    
    BSS = BlockSparseSolver(
        H, fops, beta, w_max, eps,
        conserved_operators=conserved_operators,
        dense=dense,
        )

    #BSS.set_hybridization(Delta_w)
    BSS.set_hybridization_poles_and_coefficients(pol, weight)

    print(pol.shape)
    print(weight.shape)
    
    BSS.init_diagram_evaluator() # -- This calls the DiagramEvaluator constructor.

    # -- Compare pseudo particle Green's function

    G_S = S.S.G0_iaa
    
    G_BSS = BSS.pseudo_particle_greens_function()
    if not dense: 
        G_BSS = pseudo_particle_block_gf_to_dense(G_BSS, BSS.ad)

    G_diff = np.max(np.abs(G_BSS.data - G_S))
    print(f'G_diff = {G_diff:2.2E}')
    np.testing.assert_array_almost_equal(G_BSS.data, G_S)

    Z = BSS.partition_function()
    print(f'Z = {Z}')
    np.testing.assert_almost_equal(Z, 1.0)

    G_DYSON_BSS = BSS.solve_dyson(BSS.Sigma, BSS.eta) # Solving Dyson with zero self-energy

    if not dense: 
        G_DYSON_BSS = pseudo_particle_block_gf_to_dense(G_DYSON_BSS, BSS.ad)
    np.testing.assert_array_almost_equal(G_DYSON_BSS.data, G_S)

    # -- Compare self-energy topologies

    from triqs_xca.diag import all_connected_pairings

    results = dict()

    orders_with_signs_and_topologies = [
        (1, [(1, [(0, 1)])]), # 1st order (NCA) diagram
        # 2nd order diags are zero, skip
        (3, [(1, [(0, 3), (1, 4), (2, 5)])]), # Non-zero 3rd order diagram
        ]
    
    for order in [1, 2, 3]:
        print(f'order = {order}')
        for sign, topology in all_connected_pairings(order):
    #for order, signs_and_topologies in orders_with_signs_and_topologies:
    #    print(f'signs_and_topologies = {signs_and_topologies}')
    #    for sign, topology in signs_and_topologies:
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

            d.Sigma_BSS = BSS.pseudo_particle_self_energy_topology(topology)
            if not dense:
                d.Sigma_BSS = pseudo_particle_block_gf_to_dense(d.Sigma_BSS, BSS.ad)

            #d.Sigma_BSS.data[:] *= sign # FIXME! Different sign convention?!?
            d.Sigma_BSS.data[:] *= -pow(-1, d.order) # FIXME! Different sign convention?!?

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
        oplot(Delta_tau)
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$\Delta(\tau)$]')
        
        plt.subplot(*subp); subp[-1] += 1
        for i,j in product(range(Delta.shape[-1]), repeat=2):
            if not (Delta[:, i, j].real == 0.).all():
                plt.plot(Z.imag, Delta[:, i, j].real, label=f'i,j = {i},{j}')
        plt.legend()
        plt.xlabel(r'$\omega_n$')
        plt.ylabel(r'Re[$\Delta(i\omega_n)$]')

        plt.subplot(*subp); subp[-1] += 1
        for i,j in product(range(Delta.shape[-1]), repeat=2):
            if not (Delta[:, i, j].imag == 0.).all():
                plt.plot(Z.imag, Delta[:, i, j].imag, label=f'i,j = {i},{j}')
        plt.legend()
        plt.ylabel(r'Im[$\Delta(i\omega_n)$]')
        plt.xlabel(r'$\omega_n$')
        
        plt.tight_layout()        

        
        plt.figure(figsize=(14, 12))
        #subp = [2, 4, 1]
        subp = [6, 4, 1]
        #subp = [34, 4, 1]

        print(len(results))
        #exit()

        for topology, t in results.items():
            
            # -- Self-energy

            plt.subplot(*subp); subp[-1] += 1

            #oplotr(t.Sigma_BSS_block, label=None)
            oplot(t.Sigma_BSS, label=None)
            for i,j in product(range(t.Sigma_S.shape[-1]), repeat=2):
            #for i,j in [(0,0), (1,1)]:
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

        # -- Analysis of 1st order topolgy

        topology = ((0, 1),)
        #if True:
        if topology in results.keys():
            t = results[topology]

            # -- Analytic soltuion

            ana = get_analytic_solution()
            G_anal = ana.Gfunc(t.tau, beta)
            Sigma_anal_00 = ana.Sfunc_01_00(t.tau, beta, e1)
            Sigma_anal_11 = ana.Sfunc_01_11(t.tau, beta, e1)

            if False:
                a = np.log(2) / beta
                def K(tau, omega): return -np.exp(-omega * tau) / (1 + np.exp(-beta * omega))
                G_anal = - np.exp(-a * t.tau)
                Sigma_anal1 = np.exp(-a * t.tau) * -K(t.tau, e1)
                Sigma_anal2 = np.exp(-a * t.tau) * -K(beta - t.tau, e1)

                #spgf_anal = 0.5**2 * np.exp(-a * beta) * (beta**2 - 2*beta*t.tau + t.tau**2) * t.tau**2 / 4
                #spgf_BS_ref = -0.5**2 * np.exp(-a * beta) * (beta - t.tau) * t.tau**2 / 2 # -- Found in BS code

            plt.figure(figsize=(6, 6))
            subp = [2, 2, 1]

            #i, j = (0, 0)
            #for i,j in [(0, 0), (1, 1)]:
            for i,j in [(0, 0)]:
                plt.subplot(*subp); subp[-1] += 1
                if i == 0 : plt.title(f'topology {topology}')

                plt.plot(S.S.tau_i, G_S[:, i, j].real, '+', label='ZH numeric')
                plt.plot(S.S.tau_i, G_BSS.data[:, i, j].real, '+', label='BS numeric')
                plt.plot(t.tau, G_anal, '-', label='analytic')

                plt.legend()
                plt.xlabel(r'$\tau$')
                plt.ylabel(r'$G(\tau)$')        

            plt.subplot(*subp); subp[-1] += 1
            if False:
                #oplot(t.spgf_BSS, label=None)
                for i,j in [(0, 0)]:
                #for i,j in product(range(t.spgf_S.shape[-1]), repeat=2):
                    plt.subplot(*subp); subp[-1] += 1
                    plt.plot(t.tau, t.spgf_S[:, i, j].real, '+', label='ZH numeric')
                    #plt.plot(t.tau, t.spgf_S[:, i, j].imag, '+-')
                    plt.plot(t.tau, t.spgf_BSS.data[:, i, j].real, 'x', label='BS numeric')
                    #plt.plot(t.tau, spgf_anal, '-', label='analytic')
                    #plt.plot(t.tau, spgf_BS_ref, '--', label='expression matching BS numeric')
                    plt.xlabel(r'$\tau$')
                    plt.ylabel(r'$g(\tau)$')
                    plt.legend()

            for i,j in [(0, 0), (1, 1)]:
                plt.subplot(*subp); subp[-1] += 1
            #for i,j in product(range(t.Sigma_S.shape[-1]), repeat=2):
                plt.plot(t.tau, t.Sigma_S[:, i, j].real, '+', label='ZH numeric')
                plt.plot(t.tau, t.Sigma_BSS.data[:, i, j].real, 'x', label='BS numeric')
                #plt.plot(t.tau, t.Sigma_BSS[0][:, i, j].real, 'x')
                #plt.plot(t.tau, t.Sigma_S[:, i, j].imag, '+')
                if i == 0:
                    plt.plot(t.tau, Sigma_anal_00, '-', label='analytic')
                else:
                    plt.plot(t.tau, Sigma_anal_11, '-', label='analytic')

                plt.xlabel(r'$\tau$')
                plt.ylabel(r'$\Sigma_{' + f'{i},{j}' + r'}(\tau)$')
                plt.legend()

            plt.tight_layout()
            plt.savefig('figure_xca_one_fermion_1st_order_topology_analytic_cf.pdf')
        

        # -- Analysis of 3rd order topolgy
        
        topology = ((0, 3), (1, 4), (2, 5))
        if topology in results.keys():
        #if True:
            t = results[topology]

            # -- Analytic soltuion
            
            ana = get_analytic_solution()
            G_anal = ana.Gfunc(t.tau, beta)

            Sigma_anal_00 = ana.Sfunc_031425_00(t.tau, beta, e1)
            Sigma_anal_11 = ana.Sfunc_031425_11(t.tau, beta, e1)

            spgf_anal = ana.spgf_func_031425(t.tau, beta, e1)
            
            #a = np.log(2) / beta
            #G_anal = - np.exp(-a * t.tau)

            #Sigma_anal = 0.5**3 * np.exp(-a * t.tau) * t.tau**4 / (2*3*4)
            
            #spgf_anal = 0.5**2 * np.exp(-a * beta) * (beta**2 - 2*beta*t.tau + t.tau**2) * t.tau**2 / 4
            #spgf_BS_ref = -0.5**2 * np.exp(-a * beta) * (beta - t.tau) * t.tau**2 / 2 # -- Found in BS code

            plt.figure(figsize=(6, 6))
            subp = [2, 2, 1]

            #i, j = (0, 0)
            #for i,j in [(0, 0), (1, 1)]:
            for i,j in [(0, 0)]:
                plt.subplot(*subp); subp[-1] += 1
                if i == 0 : plt.title(f'topology {topology}')

                plt.plot(S.S.tau_i, G_S[:, i, j].real, '+', label='ZH numeric')
                plt.plot(S.S.tau_i, G_BSS.data[:, i, j].real, '+', label='BS numeric')
                plt.plot(t.tau, G_anal, '-', label='analytic')

                plt.legend()
                plt.xlabel(r'$\tau$')
                plt.ylabel(r'$G(\tau)$')        

            if True:
                #oplot(t.spgf_BSS, label=None)
                for i,j in [(0, 0)]:
                #for i,j in product(range(t.spgf_S.shape[-1]), repeat=2):
                    plt.subplot(*subp); subp[-1] += 1
                    plt.plot(t.tau, t.spgf_S[:, i, j].real, '+', label='ZH numeric')
                    #plt.plot(t.tau, t.spgf_S[:, i, j].imag, '+-')
                    plt.plot(t.tau, t.spgf_BSS.data[:, i, j].real, 'x', label='BS numeric')
                    plt.plot(t.tau, spgf_anal, '-', label='analytic')
                    #plt.plot(t.tau, spgf_BS_ref, '--', label='expression matching BS numeric')
                    plt.xlabel(r'$\tau$')
                    plt.ylabel(r'$g(\tau)$')
                    plt.legend()

            for i,j in [(0, 0), (1, 1)]:
            #for i,j in product(range(t.Sigma_S.shape[-1]), repeat=2):
                plt.subplot(*subp); subp[-1] += 1
                plt.plot(t.tau, t.Sigma_S[:, i, j].real, '+', label='ZH numeric')
                plt.plot(t.tau, t.Sigma_BSS.data[:, i, j].real, 'x', label='BS numeric')
                #plt.plot(t.tau, t.Sigma_BSS[0][:, i, j].real, 'x')
                #plt.plot(t.tau, t.Sigma_S[:, i, j].imag, '+')
                if i == 0:
                    plt.plot(t.tau, Sigma_anal_00, '-', label='analytic')
                else:
                    plt.plot(t.tau, Sigma_anal_11, '-', label='analytic')

                plt.xlabel(r'$\tau$')
                plt.ylabel(r'$\Sigma(\tau)$')
                plt.legend()

            plt.tight_layout()
            plt.savefig('figure_xca_one_fermion_3rd_order_topology_analytic_cf.pdf')

        plt.show()

    if not verbose:

        H_mat = hamiltonian_matrix(BSS.ad)    
        np.testing.assert_array_almost_equal(H_mat, S.S.H_mat)

        np.testing.assert_array_almost_equal(G_BSS.data, G_S)

        for topology, t in results.items():
            print(f'topology = {t.topology}')
            np.testing.assert_array_almost_equal(t.Sigma_BSS.data, t.Sigma_S)
            np.testing.assert_array_almost_equal(t.spgf_BSS.data, t.spgf_S)


if __name__ == '__main__':

    ops = [
        'none',
        'total_density', 
        ]

    for e1 in [+0.8, -0.8]:
        
        test_oca_diagram_cf_block_sparse_and_dense(
            e1=e1, beta=2.0, dense=True, verbose=False)
        
        for op in ops:
            test_oca_diagram_cf_block_sparse_and_dense(
                e1=e1, beta=2.0, conserved_operators=op, verbose=False)
