""" Topology-resolved evaluators of the block-sparse solver against independent references.

The local Hamiltonian is a two-orbital Kanamori model, with and without a complex intra-spin orbital hopping,
coupled to a discrete bath given directly as poles and coefficients, so no hybridization fit enters.

* The atomic pseudo-particle propagator G0 and the local Hamiltonian are checked against exact
  diagonalization of the local problem.
* The order one (NCA) self-energy and single-particle Green's function are checked against their closed forms
  in the Fock basis, built from the fermion matrices of the exact diagonalization.
* Every topology up to order two of the self-energy and the single-particle Green's function is compared with
  the dense evaluator (no symmetry blocks) on the same pole set. A second model uses orbital-diagonal
  hybridization with the automatic symmetry partition, which has finer blocks, and a third small model
  reaches order three.
* The order and total sums of the solver agree with the sums over the topologies.

Orders above one are thus checked as dense-versus-block-sparse evaluator parity and sum consistency, with
independent references only at order one (closed form) and in the atomic limit (exact diagonalization). The
order two manual dense reference is in dense_manual_NCA_OCA.py.

Quantities are compared in the Fock basis, where the block-sparse and dense pseudo-particle propagators are
directly comparable, and the complex hopping and hybridization phases make a transposed or conjugated index
visible. """

import numpy as np

from scipy.linalg import expm

from triqs.gfs import make_gf_dlr
from triqs.operators import n, c, c_dag
from triqs.operators.util.hamiltonians import h_int_kanamori

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

from triqs_xca import Solver
from triqs_xca.solver import hamiltonian_matrix_block
from triqs_xca.solver import pseudo_particle_block_gf_to_dense as to_fock
from triqs_xca.diag import all_connected_pairings


# w_max exceeds the width of the many-body spectrum of the local Hamiltonians below
beta, w_max, eps = 2.0, 8.0, 1e-8

# The two evaluators run the same DLR arithmetic and differ only in the order of floating point operations
rtol_evaluators = 1e-12

# The closed form self-energy uses the exact hybridization, while the evaluators represent products of
# propagators to the DLR accuracy eps, a few times worse for a propagator dressed by a Dyson step, hence
# 100 eps. The closed form single-particle Green's function uses the same DLR representation of the
# propagator as the evaluators, so it agrees to rounding.
rtol_sigma_closed_form = 100 * eps
rtol_spgf_closed_form = 1e-12


def make_model(kind):
    """ Local Hamiltonian, Green's function structure, symmetry specification, bath poles and the
    hybridization coefficients (V^dagger V)_{ab} for each pole, block diagonal in the Green's function blocks. """

    rng = np.random.default_rng(1234)
    poles = np.array([-0.7, 0.2, 0.9])

    if kind in ('kanamori_mixed', 'kanamori_diagonal'):

        gf_struct = [['up', 2], ['dn', 2]]
        N_up = n('up', 0) + n('up', 1)
        N_dn = n('dn', 0) + n('dn', 1)

        H = h_int_kanamori(('up', 'dn'), 2, 3.0 * np.ones((2, 2)), 2.0 * np.ones((2, 2)), 0.5, off_diag=True)
        H += -4.0 * (N_up + N_dn)

        coefficients = np.zeros((len(poles), 4, 4), dtype=complex)

        if kind == 'kanamori_mixed':
            # Complex hopping, with different phases for the two spins, and orbital mixing hybridization.
            # Only N_up and N_dn are conserved, and the hybridization is block diagonal in their sectors.
            for s, t in (('up', 0.3 * np.exp(0.7j)), ('dn', 0.2 * np.exp(-0.4j))):
                H += t * c_dag(s, 0) * c(s, 1) + np.conj(t) * c_dag(s, 1) * c(s, 0)
            conserved_operators = [N_up, N_dn]
            for p in range(len(poles)):
                for s in range(2):
                    v = 0.4 * (rng.normal(size=2) + 1j * rng.normal(size=2))
                    coefficients[p, 2 * s:2 * s + 2, 2 * s:2 * s + 2] = np.outer(v.conj(), v)
        else:
            # Orbital-diagonal hybridization, so the symmetries found by the automatic partition are kept
            conserved_operators = 'automatic'
            for p in range(len(poles)):
                coefficients[p] = np.diag(rng.uniform(0.05, 0.3, size=4))

    elif kind == 'dimer':

        gf_struct = [['a', 2]]
        N = n('a', 0) + n('a', 1)
        t = 0.4 * np.exp(0.5j)
        H = 0.3 * N + 1.5 * n('a', 0) * n('a', 1) + t * c_dag('a', 0) * c('a', 1) + np.conj(t) * c_dag('a', 1) * c('a', 0)
        conserved_operators = [N]
        v = 0.5 * (rng.normal(size=(len(poles), 2)) + 1j * rng.normal(size=(len(poles), 2)))
        coefficients = np.array([np.outer(v[p].conj(), v[p]) for p in range(len(poles))])

    else:
        raise ValueError(kind)

    return dict(H=H, gf_struct=gf_struct, conserved_operators=conserved_operators,
                poles=poles, coefficients=coefficients)


def make_solver(model, use_blocks):
    """ Block-sparse solver, or the dense one when use_blocks is False, with the bath set by poles. """

    S = Solver(
        H_loc=model['H'], beta=beta, w_max=w_max, eps=eps, gf_struct=model['gf_struct'],
        conserved_operators=model['conserved_operators'] if use_blocks else [], verbose=False)

    assert S.use_dense_solver == (not use_blocks)

    S.set_hybridization_poles_and_coefficients(model['poles'], model['coefficients'])
    S.init_diagram_evaluator()

    return S


def tau_nodes(S):
    return np.array([float(t) for t in S.mesh_tau])


def fock_hamiltonian(ad):
    """ Local Hamiltonian in the Fock basis, keeping a complex part. """

    H = np.zeros([ad.full_hilbert_space_dim] * 2, dtype=complex)
    for sidx in range(ad.n_subspaces):
        fidx = ad.fock_states[sidx]
        H[np.ix_(fidx, fidx)] = hamiltonian_matrix_block(ad, sidx)

    return H


def exact_diagonalization(S, model):
    """ Fermion matrices c^dagger_a and c_a in the Fock basis of the solver's operator order. """

    fops = [c(*idx) for idx in S.fundamental_operators]
    ed = TriqsExactDiagonalization(model['H'], fops, beta)

    cdag = [np.array(m.todense()) for m in ed.rep.sparse_operators.c_dag]
    cann = [m.conj().T for m in cdag]

    return np.array(ed.ed.H.todense()), cdag, cann


def block_gf_from_fock(G_fock, S):
    """ Block propagator of the solver S from a Fock basis propagator, which must have no weight between the
    blocks. """

    ad = S.atom_diag
    G = S.get_zero_pseudo_particle_propagator()
    inside = np.zeros(G_fock.shape[1:], dtype=bool)

    for sidx in range(ad.n_subspaces):
        fidx = ad.fock_states[sidx]
        G[sidx].data[:] = G_fock[:, fidx[:, None], fidx[None, :]]
        inside[np.ix_(fidx, fidx)] = True

    assert not np.any(G_fock[:, ~inside]), 'propagator couples different blocks'

    return G


def dressed_propagator(Sd, Sb, order=2):
    """ Propagator dressed by a Dyson step with the block-sparse self-energy of the given order, for both solvers.
    It has weight between degenerate states and is not diagonal in the eigenbasis. """

    Sigma = Sb.eval_pseudo_particle_self_energy(Sb.G0, order)
    Gb = Sb.solve_dyson(Sigma, Sb.eta)
    Gd = block_gf_from_fock(to_fock(Gb, Sb.atom_diag).data, Sd)

    return Gd, Gb


def max_abs(x):
    return np.max(np.abs(x))


def assert_close(x, ref, rtol, what):
    """ Agreement relative to the largest element of the reference, which must not vanish. """

    scale = max_abs(ref)
    assert scale > 1e-8, f'vacuous comparison, {what} vanishes'
    err = max_abs(x - ref)
    assert err <= rtol * scale, f'{what}: max|diff| = {err:2.2E} exceeds {rtol:1.0E} * {scale:2.2E}'

    return err / scale


def test_atomic_limit_vs_exact_diagonalization(verbose=False):
    """ The Hamiltonian, the ground state shift and the atomic propagator
    G0(tau) = -exp(-tau (H - E_gs - eta0)) of both evaluators, with eta0 = -ln(sum exp(-beta (E - E_gs))) / beta,
    against exact diagonalization. The complex hopping makes the Hamiltonian complex. """

    model = make_model('kanamori_mixed')

    for use_blocks in (False, True):
        S = make_solver(model, use_blocks)
        ad = S.atom_diag
        H_ed, _, _ = exact_diagonalization(S, model)

        assert np.max(np.abs(H_ed.imag)) > 0.1, 'vacuous test, the Hamiltonian is real'
        assert_close(fock_hamiltonian(ad), H_ed, 1e-12, 'Hamiltonian')

        E = np.linalg.eigvalsh(H_ed)
        eta0 = -np.log(np.sum(np.exp(-beta * (E - E[0])))) / beta

        np.testing.assert_allclose(ad.gs_energy, E[0], atol=1e-12)
        np.testing.assert_allclose(S.eta0, eta0, atol=1e-12)

        G0_ref = np.array([-expm(-t * (H_ed - (E[0] + eta0) * np.eye(len(E)))) for t in tau_nodes(S)])
        err = assert_close(to_fock(S.G0, ad).data, G0_ref, 1e-11, f'G0 (blocks: {use_blocks})')

        # With this shift Tr[G0(beta)] = -1, the normalization of the pseudo-particle propagator
        np.testing.assert_allclose(S.partition_function(), 1.0, atol=eps)

        if verbose:
            print(f'blocks = {use_blocks}, n_subspaces = {ad.n_subspaces}, '
                  f'rel. err. G0 = {err:2.2E}, eta0 = {S.eta0:+2.6f}')


def nca_self_energy_closed_form(G, tau, poles, coefficients, cdag, cann):
    """ The order one self-energy, in the Fock basis, of the propagator G on the nodes tau,

    X(tau) = sum_ab [ c^dag_a G(tau) c_b Delta_ab(tau) + c_a G(tau) c^dag_b Delta_ba(beta - tau) ],

    with the pole representation Delta(tau) = sum_p K(tau, w_p) (V^dag V)_p of the hybridization and the
    kernel K(tau, w) = -exp(-w tau) / (1 + exp(-beta w)). """

    def Delta(t):
        K = -np.exp(-np.outer(t, poles)) / (1 + np.exp(-beta * poles))[None, :]
        return np.einsum('tp,pab->tab', K, coefficients)

    D_fwd, D_bwd = Delta(tau), Delta(beta - tau)
    X = np.zeros_like(G)

    for a in range(len(cdag)):
        for b in range(len(cdag)):
            X += D_fwd[:, a, b, None, None] * np.einsum('ij,tjk,kl->til', cdag[a], G, cann[b])
            X += D_bwd[:, b, a, None, None] * np.einsum('ij,tjk,kl->til', cann[a], G, cdag[b])

    return X


def nca_spgf_closed_form(G_of_tau, tau, cdag, cann):
    """ The order one single-particle Green's function g_ab(tau) = Tr[ c_a G(tau) c^dag_b G(beta - tau) ]
    of a pseudo-particle propagator evaluated at tau and beta - tau by the callable G_of_tau. """

    g = np.zeros((len(tau), len(cdag), len(cdag)), dtype=complex)

    for k, t in enumerate(tau):
        G_fwd, G_bwd = G_of_tau(t), G_of_tau(beta - t)
        for a in range(len(cdag)):
            for b in range(len(cdag)):
                g[k, a, b] = np.trace(cann[a] @ G_fwd @ cdag[b] @ G_bwd)

    return g


def test_nca_closed_form(verbose=False):
    """ The order one topology of both evaluators against the closed forms, on the atomic propagator and on a
    propagator dressed by a Dyson step. The closed form of the self-energy is linear in the propagator, so a
    dressed one tests more of the block structure, the single-particle Green's function needs the propagator
    at beta - tau, taken by DLR interpolation. """

    model = make_model('kanamori_mixed')
    Sd, Sb = make_solver(model, False), make_solver(model, True)
    _, cdag, cann = exact_diagonalization(Sb, model)
    tau = tau_nodes(Sb)

    topology = np.array([(0, 1)], dtype=np.int32)
    Gd_dressed, Gb_dressed = dressed_propagator(Sd, Sb)

    for label, S, G_blocks in (
            ('dense, bare', Sd, Sd.G0), ('block-sparse, bare', Sb, Sb.G0),
            ('dense, dressed', Sd, Gd_dressed), ('block-sparse, dressed', Sb, Gb_dressed)):

        G_fock = to_fock(G_blocks, S.atom_diag)

        X = to_fock(S.eval_pseudo_particle_self_energy_topology(G_blocks, topology), S.atom_diag).data
        X_ref = nca_self_energy_closed_form(
            G_fock.data, tau, model['poles'], model['coefficients'], cdag, cann)
        err_X = assert_close(X, X_ref, rtol_sigma_closed_form, f'NCA self-energy ({label})')

        G_dlr = make_gf_dlr(G_fock)
        g = S.eval_single_particle_greens_function_topology(G_blocks, topology).data
        g_ref = nca_spgf_closed_form(lambda t: np.array(G_dlr(t)), tau, cdag, cann)
        err_g = assert_close(g, g_ref, rtol_spgf_closed_form, f'NCA single-particle Green\'s function ({label})')

        if verbose:
            print(f'{label}: rel. err. Sigma = {err_X:2.2E}, g = {err_g:2.2E}')

    # The transposed orbital index is a different function, since the hybridization and hopping are complex
    g_T = np.transpose(g_ref, (0, 2, 1))
    assert max_abs(g_ref - g_T) > 1e-3 * max_abs(g_ref), 'vacuous test, g is symmetric in the orbital indices'


def compare_with_dense(kind, max_order, verbose=False):
    """ Self-energy and single-particle Green's function of every connected topology up to max_order,
    block-sparse against dense, on the atomic propagator and on a dressed one. Returns the results for plotting. """

    model = make_model(kind)
    Sd, Sb = make_solver(model, False), make_solver(model, True)

    assert Sd.atom_diag.n_subspaces == 1
    assert Sb.atom_diag.n_subspaces > 1, 'vacuous test, no block structure'

    # The pseudo-particle propagators of the two evaluators agree to rounding
    G0_d, G0_b = to_fock(Sd.G0, Sd.atom_diag).data, to_fock(Sb.G0, Sb.atom_diag).data
    assert_close(G0_b, G0_d, rtol_evaluators, 'G0')

    results = []

    for label, (Gd, Gb) in (('bare', (Sd.G0, Sb.G0)), ('dressed', dressed_propagator(Sd, Sb))):
        for order in range(1, max_order + 1):
            for _, topology in all_connected_pairings(order):

                topo = np.array(topology, dtype=np.int32)

                Sigma_d = to_fock(Sd.eval_pseudo_particle_self_energy_topology(Gd, topo), Sd.atom_diag).data
                Sigma_b = to_fock(Sb.eval_pseudo_particle_self_energy_topology(Gb, topo), Sb.atom_diag).data
                spgf_d = Sd.eval_single_particle_greens_function_topology(Gd, topo).data
                spgf_b = Sb.eval_single_particle_greens_function_topology(Gb, topo).data

                what = f'{kind}, {label} G, topology {topology}'
                err_Sigma = assert_close(Sigma_b, Sigma_d, rtol_evaluators, f'Sigma, {what}')
                err_spgf = assert_close(spgf_b, spgf_d, rtol_evaluators, f'spgf, {what}')

                if verbose:
                    print(f'{what}: rel. err. Sigma = {err_Sigma:2.2E}, spgf = {err_spgf:2.2E}')

                results.append(dict(
                    label=label, topology=topology, tau=tau_nodes(Sb),
                    Sigma_b=Sigma_b, Sigma_d=Sigma_d, spgf_b=spgf_b, spgf_d=spgf_d))

    return results


def test_topologies_kanamori_mixed_hybridization(verbose=False):
    """ Orbital mixing hybridization and complex hopping, with the N_up and N_dn blocks. """
    return compare_with_dense('kanamori_mixed', 2, verbose)


def test_topologies_kanamori_automatic_partition(verbose=False):
    """ Orbital-diagonal hybridization, where the automatic partition has more and smaller blocks. """
    return compare_with_dense('kanamori_diagonal', 2, verbose)


def test_topologies_dimer_order_three(verbose=False):
    """ All four third order topologies on a small model. """
    return compare_with_dense('dimer', 3, verbose)


def test_order_and_total_sums(verbose=False):
    """ The solver level sums equal the sums over topologies: the self-energy of one order is minus the
    sum of the topologies of that order, and the single-particle Green's function has the extra
    factor (-1)^order. """

    model = make_model('dimer')
    max_order = 3

    Sd, Sb = make_solver(model, False), make_solver(model, True)
    Gd, Gb = dressed_propagator(Sd, Sb)

    for tag, S, G in (('dense', Sd, Gd), ('block-sparse', Sb, Gb)):

        Sigma_sum, spgf_sum = 0., 0.
        for order in range(1, max_order + 1):
            for _, topology in all_connected_pairings(order):
                topo = np.array(topology, dtype=np.int32)
                Sigma_sum = Sigma_sum - to_fock(S.eval_pseudo_particle_self_energy_topology(G, topo), S.atom_diag).data
                spgf_sum = spgf_sum + (-1)**order * S.eval_single_particle_greens_function_topology(G, topo).data

        Sigma = to_fock(S.eval_pseudo_particle_self_energy(G, max_order), S.atom_diag).data
        spgf = S.eval_single_particle_greens_function(G, max_order).data

        assert_close(Sigma, Sigma_sum, rtol_evaluators, f'summed self-energy ({tag})')
        assert_close(spgf, spgf_sum, rtol_evaluators, f'summed single-particle Green\'s function ({tag})')


def plot_results(results, title):
    """ Block-sparse (crosses) against dense (lines) per topology, for the atomic propagator.
    Plots are shown only with an interactive backend, so nothing happens on a machine without a display. """

    import matplotlib.pyplot as plt

    results = [r for r in results if r['label'] == 'bare']
    fig, axes = plt.subplots(len(results), 2, figsize=(8, 2.5 * len(results)), squeeze=False)

    for row, r in zip(axes, results):
        for ax, key, name in zip(row, ('Sigma', 'spgf'), ('Sigma', 'g')):
            dense, sparse = r[key + '_d'], r[key + '_b']
            for i in range(dense.shape[1]):
                for j in range(dense.shape[2]):
                    if max_abs(dense[:, i, j]) > 1e-10:
                        ax.plot(r['tau'], dense[:, i, j].real, '-')
                        ax.plot(r['tau'], sparse[:, i, j].real, 'x', color='gray')
            ax.set_xlabel(r'$\tau$')
            ax.set_ylabel(f'{name} {r["topology"]}', fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig('figure_block_sparse_api_' + title.replace(' ', '_') + '.pdf')

    if plt.get_backend().lower() != 'agg':
        plt.show()


if __name__ == '__main__':

    verbose = False

    test_atomic_limit_vs_exact_diagonalization(verbose=verbose)
    test_nca_closed_form(verbose=verbose)

    results = test_topologies_kanamori_mixed_hybridization(verbose=verbose)
    test_topologies_kanamori_automatic_partition(verbose=verbose)
    test_topologies_dimer_order_three(verbose=verbose)

    test_order_and_total_sums(verbose=verbose)

    if verbose:
        plot_results(results, 'kanamori mixed')
