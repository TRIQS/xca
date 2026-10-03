""" Bethe-lattice self-consistency loop, Delta(tau) = t^2 G(tau), with the block-sparse solver.

Two analytic references:

* U = 0: the loop converges to the semicircular Green's function of half bandwidth D = 2t, known in closed
  form. The pseudo-particle expansion truncated at order n is an expansion in the hybridization, so the
  distance to the semicircle must be small and fall rapidly with the order.

* U > 0 at half filling, H = U n_up n_do - U/2 (n_up + n_do):
  - Small t: the double occupancy approaches the atomic value 1/(2 + 2 exp(beta U / 2)) with a deviation
    c t^2 + O(t^4). Every order of the expansion contains the full t^2 term, and c is the second-order
    hopping coefficient of a two-site cluster (the Bethe lattice is the limit of many neighbours with
    z t_b^2 = t^2, and at second order the bonds act independently), obtained here by exact diagonalization.
  - Any t: the particle-hole symmetry of the Hamiltonian and of the semicircular bath is preserved by the
    pseudo-particle equations at every order, so the converged solution must be particle-hole and spin
    symmetric, with n = 1/2, equal empty and double occupancy, and the exact normalization
    G(0+) + G(beta-) = -1. The fixed point of the loop must not depend on the initial hybridization. """

import numpy as np

from triqs.gfs import MeshDLRImFreq, Gf, SemiCircular, inverse, iOmega_n
from triqs.gfs import make_gf_dlr, make_gf_dlr_imtime
from triqs.operators import n

from triqs_xca import Solver


w_max, eps = 10.0, 1e-10

spin_names = ['up', 'do']
gf_struct = [[spin, 1] for spin in spin_names]

dmft_tol = 1e-8
dmft_maxiter = 100
ppsc_tol = 1e-10
hyb_tol = 1e-9 # Hybridization pole compression tolerance, 10 eps (the DLR representation is accurate to eps)


def semicircle_g_tau(tau, beta, D):
    """ G(tau) = -int de A(e) e^{-tau e} / (1 + e^{-beta e}) for the semicircular A(e) of half bandwidth D,
    by Gauss-Chebyshev quadrature of the second kind (exact weight sqrt(1 - x^2)). """
    N = 200
    k = np.arange(1, N + 1)
    x = np.cos(k * np.pi / (N + 1))
    w = 2.0 / (N + 1) * np.sin(k * np.pi / (N + 1))**2
    e = D * x
    tau = np.asarray(tau, dtype=float)
    return -np.sum(w[None, :] * np.exp(-np.outer(tau, e)) / (1.0 + np.exp(-beta * e))[None, :], axis=1)


def initial_hybridization(kind, mesh_w, t):
    Delta_w = Gf(mesh=mesh_w, target_shape=[1, 1])
    if kind == 'semicircle':
        Delta_w << t**2 * SemiCircular(2 * t)
    elif kind == 'pole':
        # Particle-hole asymmetric start, a pole at 0.7 instead of the symmetric bath
        Delta_w << t**2 * inverse(iOmega_n - 0.7)
    else:
        raise ValueError(kind)
    return make_gf_dlr_imtime(Delta_w)


def solve_bethe_loop(U, order, beta, t, init='semicircle'):
    """ Iterate solver and Delta = t^2 G until Delta is stationary. Returns the solver and its operators. """
    n_up, n_do = n('up', 0), n('do', 0)
    H = U * n_up * n_do - 0.5 * U * (n_up + n_do)

    S = Solver(H, beta, w_max, eps, gf_struct, verbose=False)

    mesh_w = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=w_max, eps=eps, symmetrize=False)
    Delta_tau = initial_hybridization(init, mesh_w, t)
    for spin in spin_names:
        S.Delta_tau[spin] << Delta_tau

    for iter in range(1, dmft_maxiter + 1):
        S.solve(max_order=order, tol=ppsc_tol, hyb_tol=hyb_tol, maxiter=100, verbose=False)

        diff = 0.0
        for spin in spin_names:
            Delta_new = t**2 * S.G_tau[spin].data
            diff = max(diff, np.max(np.abs(Delta_new - S.Delta_tau[spin].data)))
            S.Delta_tau[spin].data[:] = Delta_new

        if diff < dmft_tol:
            break

    assert diff < dmft_tol, f'Bethe loop not converged after {iter} iterations: diff = {diff:2.2E}'
    S.n_dmft_iter = iter

    return S, n_up, n_do


def two_site_double_occupancy(U, beta, t_b):
    """ Double occupancy of site 1 of the half-filled Hubbard dimer with hopping t_b, by exact diagonalization. """
    M = 4 # Modes: 1up, 1do, 2up, 2do
    a, Z, I = np.array([[0.0, 1.0], [0.0, 0.0]]), np.diag([1.0, -1.0]), np.eye(2)
    c = []
    for k in range(M):
        op = np.eye(1)
        for j in range(M):
            op = np.kron(op, Z if j < k else (a if j == k else I))
        c.append(op)
    nn = [x.T @ x for x in c]

    H = sum(U * nn[2*i] @ nn[2*i + 1] - 0.5 * U * (nn[2*i] + nn[2*i + 1]) for i in range(2))
    for s in range(2):
        hop = c[s].T @ c[2 + s]
        H = H - t_b * (hop + hop.T)

    E, V = np.linalg.eigh(H)
    p = np.exp(-beta * (E - E.min()))
    p /= p.sum()
    return np.sum(p * np.einsum('ij,ik,kj->j', V, nn[0] @ nn[1], V))


def atomic_double_occupancy(U, beta):
    """ Half-filled isolated atom: weights 1, exp(beta U / 2), exp(beta U / 2), 1 for empty, up, down, double. """
    return 1.0 / (2.0 + 2.0 * np.exp(0.5 * beta * U))


def hopping_coefficient(U, beta, h=1e-3):
    """ c in d = d_atom + c t^2 + O(t^4), from the two-site cluster; the finite-difference error is O(h^2 c). """
    return (two_site_double_occupancy(U, beta, h) - atomic_double_occupancy(U, beta)) / h**2


def test_semicircle_at_zero_interaction(verbose=False):

    beta = 2.0
    t = 0.3 # Bethe lattice hopping, half bandwidth D = 2 t
    D = 2 * t

    # Truncation error bounds per order of the hybridization expansion
    bounds = {1: 3e-2, 2: 5e-4, 3: 2e-5}

    errors = {}
    for order, bound in bounds.items():
        S, n_up, n_do = solve_bethe_loop(U=0.0, order=order, beta=beta, t=t)

        tau = np.array([float(x) for x in S.G_tau['up'].mesh])
        g_ref = semicircle_g_tau(tau, beta, D)

        errors[order] = 0.0
        for spin in spin_names:
            g = S.G_tau[spin].data[:, 0, 0]
            np.testing.assert_allclose(g.imag, 0.0, atol=1e-12)
            errors[order] = max(errors[order], np.max(np.abs(g.real - g_ref)))
        assert errors[order] < bound, f'order {order}: |g - g_semicircle| = {errors[order]:2.2E} > {bound:2.2E}'

        np.testing.assert_allclose(S.expectation_value(n_up), 0.5, atol=1e-8)
        np.testing.assert_allclose(S.expectation_value(n_do), 0.5, atol=1e-8)

        print(f'U = 0, order {order}: |g - g_semicircle| = {errors[order]:2.2E}, DMFT iterations = {S.n_dmft_iter}')

    # Each order must gain at least an order of magnitude
    for order in [2, 3]:
        assert errors[order] < 0.1 * errors[order - 1]

    if verbose:
        plot_semicircle_comparison(S, g_ref, tau)


def test_atomic_limit_expansion(U=2.0):

    beta = 4.0
    d_atom = atomic_double_occupancy(U, beta)
    c = hopping_coefficient(U, beta)
    np.testing.assert_allclose(two_site_double_occupancy(U, beta, 0.0), d_atom, atol=1e-14)

    # The next term of the expansion in t^2 has a coefficient of the size of c
    t_values = [0.1, 0.05]

    dev = {}
    for order in [1, 2]:
        for t in t_values:
            S, n_up, n_do = solve_bethe_loop(U=U, order=order, beta=beta, t=t)
            d = S.expectation_value(n_up * n_do).real
            dev[(order, t)] = d - d_atom

            # Agreement with the leading term up to O(t^4), margin 2 c t^4
            assert abs(dev[(order, t)] - c * t**2) < 2 * c * t**4, \
                f'order {order}, t = {t}: d - d_atom = {dev[(order, t)]:2.4E}, c t^2 = {c * t**2:2.4E}'

            print(f'U = {U}, order {order}, t = {t}: d = {d:.6f}, (d - d_atom) / (c t^2) = {dev[(order, t)] / (c * t**2):.5f}')

        # Halving t reduces the deviation from the atomic limit by 4 up to O(t^2) relative corrections
        ratio = dev[(order, t_values[0])] / dev[(order, t_values[1])]
        assert abs(ratio - 4.0) < 0.1, f'order {order}: deviation ratio for halved t = {ratio:.4f}'


def test_half_filled_hubbard_symmetries(U=2.0, verbose=False):

    # The 'pole' start is particle-hole asymmetric, so the symmetries below only hold if the loop
    # converges to the symmetric fixed point
    beta = 4.0
    t = 0.5
    tau_uniform = np.linspace(0.0, beta, 41)

    g_fixed_point = {}
    for order in [1, 2]:
        for init in ['semicircle', 'pole']:
            S, n_up, n_do = solve_bethe_loop(U=U, order=order, beta=beta, t=t, init=init)

            G_up, G_do = S.G_tau['up'], S.G_tau['do']
            np.testing.assert_allclose(G_up.data, G_do.data, atol=1e-12) # Spin symmetry
            np.testing.assert_allclose(G_up.data.imag, 0.0, atol=1e-12)

            # Particle-hole symmetry G(tau) = G(beta - tau), on the DLR interpolant at uniform tau
            G_dlr = make_gf_dlr(G_up)
            g = np.array([G_dlr(x)[0, 0] for x in tau_uniform]).real
            np.testing.assert_allclose(g, g[::-1], atol=1e-9)

            # Normalization G(0+) + G(beta-) = -1
            np.testing.assert_allclose(g[0] + g[-1], -1.0, atol=1e-9)

            # Half filling, and equal weight of the empty and doubly occupied state
            n_exp = S.expectation_value(n_up)
            d_exp = S.expectation_value(n_up * n_do)
            e_exp = S.expectation_value((1 - n_up) * (1 - n_do))
            np.testing.assert_allclose(n_exp, 0.5, atol=1e-8)
            np.testing.assert_allclose(d_exp, e_exp, atol=1e-8)
            assert 0.0 < d_exp.real < 0.25 # Below the uncorrelated value n_up n_do = 1/4

            print(f'U = {U}, order {order}, init {init}: d = {d_exp.real:.6f}, DMFT iterations = {S.n_dmft_iter}')

            g_fixed_point[(order, init)] = S.G_tau['up'].data.copy()

        # Same fixed point from both starting hybridizations
        np.testing.assert_allclose(
            g_fixed_point[(order, 'semicircle')], g_fixed_point[(order, 'pole')], atol=1e-6)

    # The order is honoured, the two truncations are different approximations
    assert np.max(np.abs(g_fixed_point[(1, 'semicircle')] - g_fixed_point[(2, 'semicircle')])) > 1e-4

    if verbose:
        plot_hubbard(g_fixed_point, S)


def plot_semicircle_comparison(S, g_ref, tau):
    from triqs.plot.mpl_interface import plt
    if plt.get_backend().lower() == 'agg': return

    plt.figure(figsize=(6, 4))
    plt.plot(tau, S.G_tau['up'].data[:, 0, 0].real, 'x', label='block-sparse')
    plt.plot(tau, g_ref, '-', label='semicircle')
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$G(\tau)$')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hubbard(g_fixed_point, S):
    from triqs.plot.mpl_interface import plt
    if plt.get_backend().lower() == 'agg': return

    tau = np.array([float(x) for x in S.G_tau['up'].mesh])
    plt.figure(figsize=(6, 4))
    for (order, init), g in g_fixed_point.items():
        plt.plot(tau, g[:, 0, 0].real, 'x', label=f'order {order}, {init}')
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$G(\tau)$')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    test_semicircle_at_zero_interaction(verbose=False)
    test_atomic_limit_expansion()
    test_half_filled_hubbard_symmetries(verbose=False)
