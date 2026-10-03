r""" Analytic test of diagrammatics for a single Fermion

Using a hybridization function with a single pole $\omega$

.. math::
    \Delta(\tau) = K(\tau, \omega)

and degenerate atomic states giving a pseudo particle Green's function
that is proportional to the identity with a single exponential decay

.. math::
    G(\tau) = - 2^{-\tau / \beta}

with $G(\beta) = -1/2$.

In this case it is possible to derive analytic expressions for the self energy
and single particle Green's function diagrams, derived in examples/one_fermion_analytical_solutions.ipynb.

"""

import numpy as np

from functools import lru_cache
from types import SimpleNamespace

from triqs.gfs import make_gf_dlr_imtime
from triqs.gfs import Gf, MeshDLRImFreq, iOmega_n, inverse
from triqs.operators import n

from triqs_xca import Solver
from triqs_xca.solver import hamiltonian_matrix, pseudo_particle_block_gf_to_dense
from triqs_xca.diag import all_connected_pairings


@lru_cache(maxsize=None)
def get_analytic_solution():

    """ Lambdified analytic expressions, as attributes of the returned namespace """

    import sympy as sp

    d = SimpleNamespace()

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


def test_diagrams_vs_analytic_one_fermion(
        e1=0.8, beta=2.0, conserved_operators='none', verbose=False):

    print('='*72)
    print(f'e1 = {e1}, beta = {beta}, conserved_operators = {conserved_operators}')
    print('='*72)

    # -- Parameters

    mu = 0.0
    eps = 1e-12
    w_max = 10.0

    # Agreement with the analytic expressions is limited by the hybridization fit tolerance
    fit_tol = 100*eps
    atol = fit_tol

    # -- Local Hamiltonian

    gf_struct = [['0', 1]]

    N_op = n('0', 0)

    H = -mu * N_op

    conserved_operators = dict(
        none=[],
        total_density=[N_op],
        automatic='automatic',
        )[conserved_operators]

    mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps, symmetrize=False)
    Delta_w = Gf(mesh=mesh_w, target_shape=[1]*2)

    Delta_w << inverse(iOmega_n - e1)

    Delta_tau = make_gf_dlr_imtime(Delta_w)

    # -- Block sparse solver

    BSS = Solver(
        H, beta, w_max, eps, gf_struct=gf_struct,
        conserved_operators=conserved_operators,
        )

    BSS.Delta_tau['0'] << Delta_tau

    BSS.fit_hybridization(tol=fit_tol, compression=True, verbose=verbose)
    BSS.init_diagram_evaluator()

    ana = get_analytic_solution()

    to_dense = lambda g: pseudo_particle_block_gf_to_dense(g, BSS.atom_diag)

    def assert_off_diag_zero(data, name):
        off_diag = data * (1 - np.eye(data.shape[-1]))
        np.testing.assert_allclose(off_diag, 0, rtol=0, atol=atol, err_msg=f'{name} off-diagonal')

    def assert_diag(data, diag_ref, name):
        """ Compare the diagonal of data (tau, state, state) with reference functions
        of tau for each state, and check that the off-diagonal elements vanish """
        for i, ref in enumerate(diag_ref):
            np.testing.assert_allclose(
                data[:, i, i], ref, rtol=0, atol=atol, err_msg=f'{name} state {i}')
        assert_off_diag_zero(data, name)

    # -- Hamiltonian and pseudo particle Green's function (degenerate atomic states)

    H_mat = hamiltonian_matrix(BSS.atom_diag)
    np.testing.assert_allclose(H_mat, np.zeros((2, 2)), rtol=0, atol=1e-14)

    G_BSS = to_dense(BSS.pseudo_particle_greens_function())
    tau = np.array([float(t) for t in G_BSS.mesh])

    G_anal = ana.Gfunc(tau, beta)
    assert_diag(G_BSS.data, [G_anal]*2, 'G')

    Z = BSS.partition_function()
    print(f'Z = {Z}')
    np.testing.assert_allclose(Z, 1.0, rtol=0, atol=atol)

    # Dyson equation with zero self-energy returns the non-interacting propagator
    G_dyson = to_dense(BSS.solve_dyson(BSS.Sigma, BSS.eta))
    assert_diag(G_dyson.data, [G_anal]*2, 'G Dyson')

    # -- Self-energy and single-particle Green's function topologies

    # Analytic references (Sigma state 0, Sigma state 1) per topology
    Sigma_anal = {
        ((0, 1),): [
            ana.Sfunc_01_00(tau, beta, e1),
            ana.Sfunc_01_11(tau, beta, e1),
            ],
        ((0, 3), (1, 4), (2, 5)): [
            ana.Sfunc_031425_00(tau, beta, e1),
            ana.Sfunc_031425_11(tau, beta, e1),
            ],
        }

    # Analytic reference spgf (single orbital, 1x1) per topology
    spgf_anal = {
        ((0, 3), (1, 4), (2, 5)): ana.spgf_func_031425(tau, beta, e1),
        }

    results = dict()

    Sigma_sum = 0
    spgf_sum = 0

    for order in [1, 2, 3]:
        print(f'order = {order}')

        for sign, topology in all_connected_pairings(order):
            print(f'  topology = {topology}')
            topo = np.array(topology, dtype=np.int32)

            Sigma_raw = to_dense(BSS.eval_pseudo_particle_self_energy_topology(BSS.G, topo)).data
            spgf_raw = BSS.eval_single_particle_greens_function_topology(BSS.G, topo).data

            # Compensate for the sign factor internal to the evaluators
            Sigma = sign * Sigma_raw
            spgf = sign * spgf_raw
            results[tuple(topology)] = (Sigma, spgf)

            # Number conservation: no off-diagonal elements for any topology
            assert_off_diag_zero(Sigma, f'Sigma {topology}')

            if tuple(topology) in Sigma_anal:
                assert_diag(Sigma, Sigma_anal[tuple(topology)], f'Sigma {topology}')
                print('    Sigma matches analytic solution')

            if tuple(topology) in spgf_anal:
                np.testing.assert_allclose(
                    spgf[:, 0, 0], spgf_anal[tuple(topology)], rtol=0, atol=atol,
                    err_msg=f'spgf {topology}')
                print('    spgf matches analytic solution')

            Sigma_sum = Sigma_sum - Sigma_raw
            spgf_sum = spgf_sum + (-1)**order * spgf_raw

        # -- Order sums, cumulative up to the given order, are the sums over the topologies

        Sigma_BSS = to_dense(BSS.eval_pseudo_particle_self_energy(BSS.G, order)).data
        spgf_BSS = BSS.eval_single_particle_greens_function(BSS.G, order).data

        np.testing.assert_allclose(Sigma_BSS, Sigma_sum, rtol=0, atol=1e-12)
        np.testing.assert_allclose(spgf_BSS, spgf_sum, rtol=0, atol=1e-12)

        if order == 1:
            # The order sum carries the opposite sign of the topology contributions
            assert_diag(Sigma_BSS, [-s for s in Sigma_anal[((0, 1),)]], 'Sigma order 1')

        print('  Passed')

    # -- Visualize

    if verbose:

        from triqs.plot.mpl_interface import plt

        plt.figure(figsize=(6, 10))
        subp = [3, 2, 1]

        plt.subplot(*subp); subp[-1] += 1
        plt.plot(tau, G_BSS.data[:, 0, 0].real, 'x', label='numeric')
        plt.plot(tau, G_anal, '-', label='analytic')
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'$G(\tau)$')
        plt.legend()

        for topology, anal in Sigma_anal.items():
            Sigma, spgf = results[topology]

            for i in range(2):
                plt.subplot(*subp); subp[-1] += 1
                plt.plot(tau, Sigma[:, i, i].real, 'x', label='numeric')
                plt.plot(tau, anal[i], '-', label='analytic')
                plt.xlabel(r'$\tau$')
                plt.ylabel(r'$\Sigma_{' + f'{topology}, {i}' + r'}(\tau)$')
                plt.legend(fontsize=7)

            if topology in spgf_anal:
                plt.subplot(*subp); subp[-1] += 1
                plt.plot(tau, spgf[:, 0, 0].real, 'x', label='numeric')
                plt.plot(tau, spgf_anal[topology], '-', label='analytic')
                plt.xlabel(r'$\tau$')
                plt.ylabel(r'$g_{' + f'{topology}' + r'}(\tau)$')
                plt.legend(fontsize=7)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    ops = [
        'none',
        'total_density',
        'automatic',
        ]

    for e1 in [+0.8, -0.8]:
        for op in ops:
            test_diagrams_vs_analytic_one_fermion(
                e1=e1, beta=2.0, conserved_operators=op, verbose=False)
