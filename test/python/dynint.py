""" Test dynamic interaction expansion for a single fermionic level coupled to a bosonic mode.

Author: Hugo U. R. Strand, 2026 """

import numpy as np

import triqs.utility.mpi as mpi

from triqs.gfs import Gf, MeshDLRImTime, make_gf_dlr_imfreq, make_gf_dlr_imtime, make_gf_dlr, inverse, iOmega_n, make_gf_imtime


from pyed.SparseExactDiagonalization import SparseExactDiagonalization
from pyed.SparseMatrixFockStates import SparseMatrixFermiBoseCreationOperators


from triqs_xca import Solver


def get_ed_ref(eps0, g, omega0, mesh_f_tau, mesh_b_tau, Nb_max=10):

    ops = SparseMatrixFermiBoseCreationOperators(Nf=1, Nb=1, Nb_max=Nb_max)
    
    c, b = ops.c_dag[0].getH(), ops.b_dag[0].getH()
    nf, nb = c.getH() * c, b.getH() * b

    H = eps0 * nf + g*(b + b.getH())*nf + omega0 * nb

    ed = SparseExactDiagonalization(H, mesh_f_tau.beta)

    tau_f = np.array([float(t) for t in mesh_f_tau])
    g_tau = Gf(mesh=mesh_f_tau, target_shape=[1, 1])
    g_tau.data[:, 0, 0] = ed.get_tau_greens_function_component(tau_f, c, c.getH())

    # Sign convention of chi_nn(tau) = <T n(tau) n(0)> as returned by
    # Solver.eval_one_time_correlator, cf. benchmarks/convergence_order/dynint.py
    tau_b = np.array([float(t) for t in mesh_b_tau])
    chi_tau = Gf(mesh=mesh_b_tau, target_shape=[1, 1])
    chi_tau.data[:, 0, 0] = ed.get_tau_greens_function_component(tau_b, nf, nf)

    return g_tau, chi_tau


def make_solver(beta=2.1, eps0=-0.1, g=0.1, omega0=1., w_max=2.0, eps=1e-12,
                conserved_operators=[], solver_class=Solver):

    """ AIM with a single fermionic level coupled to a bosonic mode with linear coupling,
    i.e. a retarded interaction given by the bosonic propagator. Returns the solver and the
    chemical potential mu giving half-filling at eps0 = 0. """

    mu = -g**2 / omega0 # For half-filling at eps0 = 0

    from triqs.operators import n

    S = solver_class(
        H_loc=(eps0 - mu) * n('0', 0),
        beta=beta, w_max=w_max, eps=eps, gf_struct=[['0', 1]],
        conserved_operators=conserved_operators)

    S.Delta_tau['0'].data[:] = 0.

    f_mesh = S.mesh_tau
    b_mesh = MeshDLRImTime(beta=f_mesh.beta, statistic='Boson', eps=f_mesh.eps, w_max=f_mesh.w_max, symmetrize=False)

    D0_tau = Gf(mesh=b_mesh, target_shape=[1, 1])
    D0_iw = make_gf_dlr_imfreq(D0_tau)
    D0_iw << -2 * g**2 * omega0 * inverse(omega0**2 - iOmega_n*iOmega_n)
    D0_iw << 0.5 * D0_iw # FIXME! Compensate for double number of Sigma diagrams for retarded interactions
    D0_tau << make_gf_dlr_imtime(D0_iw)

    S.set_dynamic_interactions(dynint_ops=[n('0', 0)], dynint_tau=D0_tau)

    return S, mu


def solve_dynint_one_fermion(
        beta=2.1, eps0=-0.1, g=0.1, omega0=1., w_max=2.0, eps=1e-12,
        order=1, conserved_operators=[], hyb_comp=True, solver_class=Solver):

    """ Solve the one-fermion dynint model and return the fields rather than the error norms, for
    tests that need a pointwise comparison. Contains no plotting, the verbose branch stays in
    test_dynint_one_fermion. """

    from triqs.operators import n

    S, mu = make_solver(beta=beta, eps0=eps0, g=g, omega0=omega0, w_max=w_max, eps=eps,
                        conserved_operators=conserved_operators, solver_class=solver_class)

    f_mesh = S.mesh_tau

    S.solve(max_order=order, spgf_max_order=1, maxiter=8, tol=1e-8, verbose=True, hyb_comp=hyb_comp)

    chi_tau = S.eval_one_time_correlator(
        S.G, max_order=order, ops_tau=[n('0', 0)], ops_0=[n('0', 0)])

    # ED reference on the solver's imaginary time mesh, where both G_tau and chi_tau live
    g_tau_ed_0, chi_tau_ed_0 = get_ed_ref(eps0 - mu, 0.0, omega0, f_mesh, f_mesh, Nb_max=10)
    g_tau_ed, chi_tau_ed = get_ed_ref(eps0 - mu, g, omega0, f_mesh, f_mesh, Nb_max=10)

    return S, chi_tau, g_tau_ed, chi_tau_ed, g_tau_ed_0, chi_tau_ed_0


def test_dynint_one_fermion(
        beta=2.1, eps0=-0.1, g=0.1, omega0=1., w_max=2.0, eps=1e-12,
        order=1, verbose=False, conserved_operators=[], hyb_comp=True):

    """" Solve AIM with single fermionic level coupled to a bosonic mode
    with linear coupling and retarded interaction given by the bosonic propagator.
    Compare to ED reference solution.

    Note that the retarded interaction does not contribute to the
    single particle Green's function diagrams (they are zero for all orders),
    so the single particle Green's function only tests the pseudo particle
    self-energy diagrams.

    The density-density susceptibility chi_nn(tau) = <T n(tau) n(0)>, on the other hand,
    does get contributions from the retarded interaction at every order above the first,
    and is evaluated here at the same order as the self-energy expansion.

    Returns the maximal deviation from the ED reference of the single particle Green's
    function and of chi_nn.
    """

    S, chi_tau, g_tau_ed, chi_tau_ed, g_tau_ed_0, chi_tau_ed_0 = solve_dynint_one_fermion(
        beta=beta, eps0=eps0, g=g, omega0=omega0, w_max=w_max, eps=eps,
        order=order, conserved_operators=conserved_operators, hyb_comp=hyb_comp)

    if verbose:
        from triqs.plot.mpl_interface import oplot, plt

        plt.figure(figsize=(6, 8))
        subp = [2, 1, 1]

        plt.subplot(*subp); subp[-1] += 1
        oplot(make_gf_imtime(S.G_tau, n_tau=100).real, '-', label='xca')
        oplot(make_gf_imtime(g_tau_ed, n_tau=100).real, ':', label='ed')
        oplot(make_gf_imtime(g_tau_ed_0, n_tau=100).real, ':', label='ed (g=0)')
        plt.ylabel(r'$G(\tau)$')

        plt.subplot(*subp); subp[-1] += 1
        oplot(chi_tau[0, 0].real, '-', label='xca')
        oplot(chi_tau_ed[0, 0].real, ':', label='ed')
        oplot(chi_tau_ed_0[0, 0].real, ':', label='ed (g=0)')
        plt.ylabel(r'$\chi_{nn}(\tau)$')

        plt.tight_layout()
        plt.show()

    g_error = np.max(np.abs(S.G_tau['0'].data - g_tau_ed.data))
    chi_error = np.max(np.abs(chi_tau.data - chi_tau_ed.data))

    return g_error, chi_error


def test_dynint_chi(
        beta=2.1, eps0=-0.1, g=0.4, omega0=1., w_max=2.0, eps=1e-12,
        order=2, verbose=False, conserved_operators=[]):

    """ Density-density susceptibility chi_nn(tau) = <T n(tau) n(0)> of the same model.

    Unlike the single particle Green's function, chi_nn does get contributions from the
    dynamical interaction at every order above the first, so it exercises the interaction
    vertices on the internal lines of the correlator diagrams.

    The dense diagram evaluator can evaluate chi_nn in two ways:

      A) as a component of the "single particle" correlator. The dynamical interaction
         operators are appended to the field operators as additional flavours, so the
         (n_hyb + i, n_hyb + j) component of compute_single_ptcle_gf() is <T O_i(tau) O_j(0)>.

      B) directly, through compute_one_time_correlator(ops_tau=[O_i], ops_0=[O_j]).

    Both sum the same backbone diagrams over the same flat indices, so they must agree
    topology by topology. This pins down the number of interaction operators, n_int, being
    handed to the CorrelatorBackbone: without it the interaction vertices on the internal
    lines are counted as fermionic when the permutation parity is computed, and path A
    comes out with the opposite sign for the topologies that carry an interaction line
    (here the second order one, and two of the four at third order).

    Path A used to be wrong for that reason. It did not affect the solver's own observables
    - eval_one_time_correlator uses path B, and eval_single_particle_greens_function
    discards the interaction components of path A - so this check reaches past the solver
    and calls the diagram evaluator directly.
    """

    from triqs.operators import n
    from triqs_xca.diag import all_connected_pairings

    S, mu = make_solver(beta=beta, eps0=eps0, g=g, omega0=omega0, w_max=w_max, eps=eps,
                        conserved_operators=conserved_operators)
    S.solve(max_order=order, spgf_max_order=order, maxiter=8, tol=1e-8, verbose=False, hyb_comp=True)

    # Path A reads component (n_hyb, n_hyb) of the single-particle correlator, which the block-sparse
    # spgf of shape (r, n_hyb, n_hyb) does not have, so the A/B cross-check stays dense-only
    two_path = S.use_dense_solver

    f_mesh = S.mesh_tau

    # -- chi_nn against the ED reference, evaluated on the solver's imaginary time mesh

    chi = S.eval_one_time_correlator(
        S.G, max_order=order, ops_tau=[n('0', 0)], ops_0=[n('0', 0)])

    g_tau_ed, chi_tau_ed = get_ed_ref(eps0 - mu, g, omega0, f_mesh, f_mesh, Nb_max=10)

    chi_error = np.max(np.abs(chi.data[:, 0, 0] - chi_tau_ed.data[:, 0, 0]))

    if mpi.is_master_node():
        print(f'chi_nn error vs ED: {chi_error:2.2E}')

    # -- chi_nn evaluated in two ways, topology by topology

    n_hyb = len(S.fundamental_operators)
    op = n('0', 0)

    errors = []
    for o in range(1, order + 1) if two_path else []:
        for sign, topology in all_connected_pairings(o):
            topology = np.array(topology, dtype=np.int32)

            f_ix_vec = np.arange(
                S.d.get_num_single_ptcle_gf_backbones(topology), dtype=np.int32)

            chi_flavour = S.d.compute_single_ptcle_gf(
                S.G, topology, f_ix_vec)[:, n_hyb, n_hyb]
            chi_direct = S.d.compute_one_time_correlator(
                S.G, [op], [op], S.atom_diag, topology, f_ix_vec)[:, 0, 0]

            error = np.max(np.abs(chi_flavour - chi_direct))
            errors.append(error)

            if mpi.is_master_node():
                print(f'O{o} topology {topology.tolist()}: '
                      f'|chi_A - chi_B| = {error:2.2E}, |chi_A + chi_B| = '
                      f'{np.max(np.abs(chi_flavour + chi_direct)):2.2E}')

    if verbose and mpi.is_master_node():
        from triqs.plot.mpl_interface import oplot, plt
        oplot(chi[0, 0].real, '-', label='xca')
        oplot(chi_tau_ed[0, 0].real, ':', label='ed')
        plt.ylabel(r'$\chi_{nn}(\tau)$')
        plt.show()

    assert chi_error < 0.1 * g**2, \
        f'chi_nn deviates from the ED reference by {chi_error:2.2E}'

    if two_path:
        assert np.max(errors) < 1e-10, \
            f'chi_nn from compute_single_ptcle_gf and from compute_one_time_correlator ' \
            f'disagree by {np.max(errors):2.2E}'
    else:
        assert not errors, 'the A/B cross-check must not run on the block-sparse path'


def test_convergence_rate(verbose=False, conserved_operators=[], orders=[1, 2]):

    """ Test convergence rate of the dynamic interaction expansion
    by comparing to ED reference solution for a single fermionic level
    coupled to a bosonic mode.

    At order = 1 we expect a convergence rate of 2 (error ~ g^4), and
    at order = 2 we expect a convergence rate of 3 (error ~ g^6).

    The density-density susceptibility chi_nn, evaluated with the one time correlator api
    at the same order as the self-energy expansion, converges one rate slower: order
    instead of order + 1. The correlator expansion starts at the bare bubble, so
    truncating it at max_order = m leaves a leading neglected vertex correction of
    O(g^(2m)), while the self-energy at order m is accurate to O(g^(2(m+1))). Evaluating
    chi_nn at max_order = order + 1 instead does recover the rate of the single particle
    Green's function."""

    g2s = np.logspace(-1.5, -0.5, 3)

    g_errss, chi_errss = [], []
    for order in orders:
        g_errs = np.zeros_like(g2s)
        chi_errs = np.zeros_like(g2s)
        for i, g2 in enumerate(g2s):
            g = np.sqrt(g2)
            g_errs[i], chi_errs[i] = test_dynint_one_fermion(
                g=g, order=order, conserved_operators=conserved_operators)
        g_errss.append(g_errs)
        chi_errss.append(chi_errs)

    def convergence_rate(errs):
        return (np.log(errs[:-1] / errs[1:]) / np.log(g2s[:-1] / g2s[1:]))[0]

    if mpi.is_master_node():
        # Compute convergence rates
        g_rates, chi_rates = [], []
        for order, g_errs, chi_errs in zip(orders, g_errss, chi_errss):
            g_rates.append(convergence_rate(g_errs))
            chi_rates.append(convergence_rate(chi_errs))
            print(f'Order {order} convergence rates: G={g_rates[-1]}, Chi={chi_rates[-1]}')

    if verbose and mpi.is_master_node():
        import matplotlib.pyplot as plt
        for i, (order, g_errs, chi_errs) in enumerate(zip(orders, g_errss, chi_errss)):
            plt.loglog(g2s, g_errs, 'o-', color=f'C{i}', label=f'O{order} G')
            plt.loglog(g2s, chi_errs, 's--', color=f'C{i}', label=f'O{order} Chi')
        plt.xlabel('$g^2$')
        plt.ylabel('Error')
        plt.legend(loc='best')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    if mpi.is_master_node():
        # Test convergence rates
        for label, rates, offset in [('G', g_rates, 1), ('Chi', chi_rates, 0)]:
            for order, rate in zip(orders, rates):
                expected = order + offset
                diff = np.abs(rate - expected)
                assert( diff < 0.2 ), \
                    f'Expected {label} convergence rate of {expected} for order {order}, but got {rate}, diff {diff}'

    # return the rates for test_dynint_block_sparse_convergence, empty off the master node
    if not mpi.is_master_node():
        return {'G': {}, 'Chi': {}}
    return {'G': dict(zip(orders, g_rates)), 'Chi': dict(zip(orders, chi_rates))}


def test_dynint_block_sparse_convergence(verbose=False):

    """ The block-sparse expansion must converge at the same rate as the dense one.

    Order 2 only, comparing the measured convergence rates of the two evaluator paths rather than
    their absolute values. """

    # order 2 only, order 1 exercises no crossing topology
    rates_dense = test_convergence_rate(conserved_operators=[], orders=[2])
    rates_bs    = test_convergence_rate(conserved_operators='automatic', orders=[2])

    if verbose:
        print(f'dense rates: {rates_dense}')
        print(f'bs    rates: {rates_bs}')

    for key in ('G', 'Chi'):
        for order in rates_dense[key]:
            d, b = rates_dense[key][order], rates_bs[key][order]
            assert abs(b - d) < 1e-6, \
                f'{key} convergence rate at order {order} differs: dense {d:.6f} vs block-sparse {b:.6f}'


def test_dynint_h5_roundtrip(verbose=False):

    """ A solved dynint solver must survive h5 write/read and copy, on both evaluator paths.

    The restored object is also re-solved and must give the same G_tau, since __eq__ compares the
    stored dictionary and cannot see the evaluator, which is rebuilt from scratch. """

    import copy, tempfile, os
    from h5 import HDFArchive

    for conserved_operators in ([], 'automatic'):
        label = 'dense' if conserved_operators == [] else 'block-sparse'

        S, mu = make_solver(conserved_operators=conserved_operators)
        S.solve(max_order=2, spgf_max_order=1, maxiter=4, tol=1e-8, verbose=False, hyb_comp=False)
        G_ref = S.G_tau['0'].data.copy()

        assert S.has_dynamic_interactions, f'{label}: vacuous test, no dynamical interaction is set'

        filename = os.path.join(tempfile.mkdtemp(), 'dynint_roundtrip.h5')
        with HDFArchive(filename, 'w') as A: A['S'] = S
        with HDFArchive(filename, 'r') as A: S_h5 = A['S']

        S_copy = copy.deepcopy(S)

        for restored, how in ((S_h5, 'h5'), (S_copy, 'deepcopy')):
            assert S == restored, f'{label}: solver does not compare equal after {how}'
            assert restored.has_dynamic_interactions, f'{label}: dynamical interactions lost in {how}'
            assert len(restored.dynint_ops) == len(S.dynint_ops), f'{label}: dynint_ops lost in {how}'

            # the evaluator is in __skip_keys and has to be rebuilt by the restored object
            restored.solve(max_order=2, spgf_max_order=1, maxiter=4, tol=1e-8, verbose=False, hyb_comp=False)
            err = np.max(np.abs(restored.G_tau['0'].data - G_ref))
            assert err < 1e-12, f'{label}: re-solving after {how} changed G_tau by {err:2.2E}'

        if verbose:
            print(f'{label}: h5 and deepcopy round-trip OK')


def test_dynint_block_sparse(verbose=False):

    """ The block-sparse evaluator must reproduce the dense one with dynamical interactions.

    The same model is solved with conserved_operators=[] (dense evaluator) and 'automatic' (block-sparse
    evaluator), and both are compared to the ED reference and to each other. Order 2 is needed to exercise
    the permutation parity, both G and chi are asserted since the retarded interaction does not enter G,
    and hyb_comp=False keeps the adapol fit tolerance out of the comparison. """

    from triqs.operators import n

    # verbose=False on the inner calls, their verbose flag ends in a blocking plt.show()
    kwargs = dict(order=2, hyb_comp=False, verbose=False)

    g_dense, chi_dense = test_dynint_one_fermion(conserved_operators=[], **kwargs)
    g_bs, chi_bs       = test_dynint_one_fermion(conserved_operators='automatic', **kwargs)

    if verbose:
        print(f'dense: g_error = {g_dense:2.2E}  chi_error = {chi_dense:2.2E}')
        print(f'bs   : g_error = {g_bs:2.2E}  chi_error = {chi_bs:2.2E}')

    # both paths must reproduce ED to within the order-2 truncation error
    assert g_dense < 1e-6, f'dense G is not at the ED reference: {g_dense:2.2E}'
    assert chi_dense < 1e-4, f'dense chi is not at the ED reference: {chi_dense:2.2E}'
    assert g_bs < 1e-6, f'block-sparse G is not at the ED reference: {g_bs:2.2E}'
    assert chi_bs < 1e-4, f'block-sparse chi is not at the ED reference: {chi_bs:2.2E}'

    # the two evaluators compute the same diagrams and must agree up to the summation order
    assert abs(g_bs - g_dense) < 1e-12, \
        f'block-sparse and dense G errors differ: {g_bs:2.6E} vs {g_dense:2.6E}'
    assert abs(chi_bs - chi_dense) < 1e-12, \
        f'block-sparse and dense chi errors differ: {chi_bs:2.6E} vs {chi_dense:2.6E}'


class _ReversedPoleSolver(Solver):

    """ Reverses the fitted pole order, an exact-arithmetic invariant that probes the conditioning of
    the pole set without a reference or a second evaluator. """

    def fit_hybridization(self, *args, **kwargs):
        super().fit_hybridization(*args, **kwargs)
        poles = np.asarray(self.hyb.poles)[::-1].copy()
        coeffs = np.ascontiguousarray(np.asarray(self.hyb.coefficients)[::-1])
        tol, compression, fit_error = self.hyb.tol, self.hyb.compression, self.hyb.fit_error
        self.set_hybridization_poles_and_coefficients(poles, coeffs)  # rebuilds self.hyb
        self.hyb.tol, self.hyb.compression, self.hyb.fit_error = tol, compression, fit_error
        if self.has_dynamic_interactions:
            self.dynint_coeffs = np.ascontiguousarray(np.asarray(self.dynint_coeffs)[::-1])


def _dlr_window_slack(eps):

    """ Relative slack on the DLR window that adapol allows for a pole physically at w_max, mirrors
    adapol.triqs._dlr_window_slack. If adapol changes it, the window asserts below fail loudly. """

    return max(1e-9, 50.0 * eps)


def test_dynint_hyb_comp_pole_window(verbose=False):

    """ The compressed pole set must stay inside the DLR window.

    The AAA pole locations are unconstrained, and a pole at |beta*omega| > Lambda is not representable
    in a DLR basis built for Lambda, so every vals2coefs round trip silently loses it. adapol's
    approx_gf_dlr_tol drops such poles and refits. The bound is Lambda itself, since the uncompressed
    DLR fallback reaches 0.9995 * Lambda. """

    S, mu = make_solver()
    beta = S.mesh_tau.beta
    Lambda = beta * S.mesh_tau.w_max

    S.fit_hybridization(tol=1e-9, compression=True, verbose=False)

    bw = beta * np.asarray(S.hyb.poles)
    worst = np.max(np.abs(bw))

    # the uncompressed DLR fallback is inside the window by construction, so check that compression ran
    assert S.hyb.compression, 'vacuous: compression did not run, so the window bound is trivial'
    assert S.hyb.fit_error is not None and S.hyb.fit_error < 1e-9, \
        f'vacuous: adapol did not produce a fit (fit_error = {S.hyb.fit_error})'
    assert len(bw) == 2, \
        f'vacuous: expected the 3-pole adapol fit to be filtered to 2, got {len(bw)} poles -- ' \
        f'either compression was skipped or the adapol DLR window filter did not fire'

    if verbose:
        print(f'pole window: max|beta*omega| = {worst:2.4E}  Lambda = {Lambda:2.4E}  '
              f'ratio = {worst / Lambda:2.4E}  n_poles = {len(bw)}')

    assert worst <= Lambda * (1 + 1e-10), \
        f'compressed pole set leaves the DLR window: max|beta*omega| = {worst:2.4E} vs ' \
        f'Lambda = {Lambda:2.4E} ({worst / Lambda:.1f}x). Poles beyond Lambda are not ' \
        f'representable in this basis. beta*omega = {np.sort(bw)}'


def test_dynint_hyb_comp_edge_pole_survives(verbose=False):

    """ A physical pole sitting exactly on the window edge must survive adapol's DLR window filter.

    adapol's pole locations are not exact, so a mode at w_max may come back slightly outside the
    window, while the smallest artefact overshoot measured is 1.8e-03, and the slack of the filter
    sits between them. omega0 == w_max is the natural choice for the mode frequency. """

    # eps is swept to check that the slack holds for both loose and tight DLR accuracy
    for beta, w_max, eps in ((2.1, 1.0, 1e-12), (1.0, 1.0, 1e-6), (2.1, 0.5, 1e-6),
                             (0.3, 1.0, 1e-8), (5.0, 2.0, 1e-10)):
        S, mu = make_solver(beta=beta, w_max=w_max, omega0=w_max, eps=eps)
        Lambda = beta * w_max
        slack = _dlr_window_slack(eps)
        S.fit_hybridization(tol=1e-9, compression=True, verbose=False)

        bw = beta * np.asarray(S.hyb.poles)
        weight = np.abs(np.asarray(S.dynint_coeffs)).reshape(len(bw), -1).max(axis=1)
        dominant = weight > 0.5 * weight.max()

        if verbose:
            print(f'edge pole: beta={beta} w_max={w_max} eps={eps:.0e} slack={slack:.1e} '
                  f'n_poles={len(bw)} max_ratio={np.max(np.abs(bw)) / Lambda:.12f} '
                  f'rel_weight={np.sort(weight / weight.max())}')

        assert len(bw) >= 2, \
            f'beta={beta}, w_max={w_max}, eps={eps:.0e}: the filter left {len(bw)} pole(s). The ' \
            f'physical mode is AT the window edge and must survive -- the window slack ' \
            f'({slack:.1e}) is too tight.'
        assert np.sum(dominant) >= 2, \
            f'beta={beta}, w_max={w_max}, eps={eps:.0e}: only {np.sum(dominant)} dominant pole(s) ' \
            f'survived; the +-omega0 pair carries the physics and both must be kept'
        # the survivors are inside the window to within the slack
        assert np.max(np.abs(bw)) <= Lambda * (1 + slack), \
            f'beta={beta}, w_max={w_max}, eps={eps:.0e}: a pole at ratio ' \
            f'{np.max(np.abs(bw)) / Lambda:.12f} survived the filter'


def test_dynint_block_sparse_hyb_comp(verbose=False):

    """ Dense and block-sparse agree pointwise with hybridization compression on, at full accuracy.

    The comparison is max|G_bs - G_dense| rather than a difference of error norms, which two solvers
    wrong in different directions can agree on. The uncompressed block-sparse solve is the accuracy
    reference, since agreement alone would accept two evaluators that are wrong together. """

    kw = dict(order=3, hyb_comp=True)
    Sd, chi_d, g_ed, chi_ed, _, _ = solve_dynint_one_fermion(conserved_operators=[], **kw)
    Sb, chi_b, *_                 = solve_dynint_one_fermion(conserved_operators='automatic', **kw)
    Su, chi_u, *_                 = solve_dynint_one_fermion(conserved_operators='automatic',
                                                             order=3, hyb_comp=False)

    dG = np.max(np.abs(Sb.G_tau['0'].data - Sd.G_tau['0'].data))
    dchi = np.max(np.abs(chi_b.data - chi_d.data))
    g_dense = np.max(np.abs(Sd.G_tau['0'].data - g_ed.data))
    g_bs = np.max(np.abs(Sb.G_tau['0'].data - g_ed.data))
    g_uncomp = np.max(np.abs(Su.G_tau['0'].data - g_ed.data))

    if verbose:
        print(f'pointwise : max|G_bs - G_dense| = {dG:2.3E}  max|chi_bs - chi_dense| = {dchi:2.3E}')
        print(f'accuracy  : g_dense = {g_dense:2.6E}  g_bs = {g_bs:2.6E}  '
              f'g_uncompressed = {g_uncomp:2.6E}')

    assert dG < 1e-13, \
        f'compressed: dense and block-sparse disagree pointwise, max|G_bs - G_dense| = {dG:2.3E}'
    assert dchi < 1e-13, \
        f'compressed: dense and block-sparse disagree pointwise, max|chi_bs - chi_dense| = {dchi:2.3E}'
    assert abs(g_dense - g_uncomp) < 1e-13, \
        f'compression costs the DENSE evaluator accuracy: {g_dense:2.6E} against the ' \
        f'uncompressed {g_uncomp:2.6E}. Agreement between evaluators is not enough -- both can be ' \
        f'wrong together.'
    assert abs(g_bs - g_uncomp) < 1e-13, \
        f'compression costs the BLOCK-SPARSE evaluator accuracy: {g_bs:2.6E} against the ' \
        f'uncompressed {g_uncomp:2.6E}'


def test_dynint_hyb_comp_pole_relabeling(verbose=False):

    """ Reversing the fitted pole order must not move G, on both evaluators at order 3. Relabeling is
    exactly invariant, so any movement is the conditioning of the pole set. At order 2 the reversal
    is invisible even with a defect, so order 3 is required. """

    kw = dict(order=3, hyb_comp=True)
    for conserved_operators, tag in (([], 'dense'), ('automatic', 'block-sparse')):
        S, chi, *_ = solve_dynint_one_fermion(conserved_operators=conserved_operators, **kw)
        R, chi_R, *_ = solve_dynint_one_fermion(conserved_operators=conserved_operators,
                                                solver_class=_ReversedPoleSolver, **kw)

        # check that the reversal happened, reversing a symmetric pair with equal residues is a numerical identity
        assert not np.array_equal(np.asarray(R.hyb.poles), np.asarray(S.hyb.poles)), \
            f'{tag}: the pole order was not actually reversed, so this test proves nothing'
        dG = np.max(np.abs(R.G_tau['0'].data - S.G_tau['0'].data))
        dchi = np.max(np.abs(chi_R.data - chi.data))
        scale = np.max(np.abs(S.G_tau['0'].data))

        if verbose:
            print(f'relabeling/{tag}: max|dG| = {dG:2.3E}  max|dchi| = {dchi:2.3E}  '
                  f'max|G| = {scale:2.3E}')

        assert dG < 1e-13, \
            f'{tag}: reversing the pole order moved G by {dG:2.3E} against max|G| = {scale:2.3E}. ' \
            f'Relabeling is an exact-arithmetic invariant, so this is the conditioning of the ' \
            f'pole set, not a diagram error.'
        assert dchi < 1e-13, \
            f'{tag}: reversing the pole order moved chi by {dchi:2.3E}'



if __name__ == '__main__':
    test_convergence_rate(verbose=False)
    test_dynint_chi(verbose=False)
    test_dynint_chi(verbose=False, conserved_operators='automatic')
    test_dynint_block_sparse(verbose=True)
    test_dynint_h5_roundtrip(verbose=True)
    test_dynint_block_sparse_convergence(verbose=True)
    test_dynint_hyb_comp_pole_window(verbose=True)
    test_dynint_hyb_comp_edge_pole_survives(verbose=True)
    test_dynint_block_sparse_hyb_comp(verbose=True)
    test_dynint_hyb_comp_pole_relabeling(verbose=True)
