r""" Acceptance test for a transverse retarded interaction, S^+ D(tau) S^-, against ED.

This pins the dynamical-interaction index convention, which the density-density tests cannot see
since n^dag = n.

Model
-----
A spinful impurity (n_orb orbitals, zero hybridization) coupled to one bosonic mode by the spin-flip
coupling

    H = H_loc + omega0 b^dag b + g ( b^dag S^- + b S^+ ),   S^+ = sum_m c^dag_{up m} c_{do m}

Integrating out the boson with G = (d_tau + omega0)^{-1} = <T b(tau) b^dag(0)>_0 gives

    S_eff = - g^2 \int\int  S^+(tau) G(tau-tau') S^-(tau')
          = - g^2 \int\int_{tau>tau'} [ S^+(tau) G(s) S^-(tau') + S^-(tau) G(-s) S^+(tau') ]

so, with the library convention D_ij <-> O_i^dag(tau) O_j(0),

    D_00(tau) = -g^2 (1 + n_B) e^{-omega0 tau}      paired with O_0 = S^-  (i.e. S^+ ... S^-)
    D_11(tau) = -g^2  n_B      e^{+omega0 tau}      paired with O_1 = S^+  (i.e. S^- ... S^+)

and D_11(tau) = D_00(beta - tau) exactly. In Matsubara D_00(i nu) = g^2 / (i nu - omega0) and
D_11(i nu) = -g^2 / (i nu + omega0), whose sum is the total kernel of the density-density test in dynint.py.

Encodings, every bosonic line being enumerated forward and backward: a single entry dynint_ops = [S^-]
with D = D_00 and no halving, two entries dynint_ops = [S^-, S^+] with D = 0.5 * diag(D_00, D_11), and
the naive 0.5 * diag(D_00, D_00), which is wrong and asserted to be wrong.

Author: Hugo U. R. Strand, 2026 """

import numpy as np

from triqs.gfs import Gf, MeshDLRImTime, make_gf_dlr_imfreq, make_gf_dlr_imtime, inverse, iOmega_n
from triqs.operators import n, c, c_dag

from pyed.SparseExactDiagonalization import SparseExactDiagonalization
from pyed.SparseMatrixFockStates import SparseMatrixFermiBoseCreationOperators

from triqs_xca import Solver


# ---------------------------------------------------------------------------- model helpers

def S_plus(n_orb):  return sum(c_dag('up', m) * c('do', m) for m in range(n_orb))
def S_minus(n_orb): return sum(c_dag('do', m) * c('up', m) for m in range(n_orb))
def H_local(eps, h=0.0):
    """ orbital energies eps[m] plus a Zeeman splitting h (up: +h, do: -h). """
    return sum((e + h) * n('up', m) + (e - h) * n('do', m) for m, e in enumerate(eps))


def get_ed_ref_transverse(eps, g, omega0, mesh_tau, Nb_max=10, h=0.0):

    """ ED reference for the transverse model.  Flavour order matches the triqs fundamental
    operators of gf_struct = [['up', n_orb], ['do', n_orb]]: up 0..n_orb-1 then do 0..n_orb-1.
    Returns G_up, G_do, chi_zz, chi_pm on mesh_tau, all in the pyed sign convention
    -<A(tau) B(0)>, which is what Solver returns for both G_tau and
    eval_one_time_correlator. """

    n_orb = len(eps)
    ops = SparseMatrixFermiBoseCreationOperators(Nf=2 * n_orb, Nb=1, Nb_max=Nb_max)
    cd = ops.c_dag
    cc = [x.getH() for x in cd]
    b = ops.b_dag[0].getH()

    up, do = list(range(n_orb)), list(range(n_orb, 2 * n_orb))
    Sp = sum((cd[u] * cc[d] for u, d in zip(up, do)), 0 * ops.I)
    Sm = Sp.getH()
    nf = lambda i: cd[i] * cc[i]

    H = sum(((e + h) * nf(u) + (e - h) * nf(d) for e, u, d in zip(eps, up, do)), 0 * ops.I)
    H = H + omega0 * b.getH() * b + g * (b.getH() * Sm + b * Sp)

    ed = SparseExactDiagonalization(H, mesh_tau.beta)
    tau = np.array([float(t) for t in mesh_tau])

    Sz = 0.5 * (sum((nf(u) for u in up), 0 * ops.I) - sum((nf(d) for d in do), 0 * ops.I))

    def gf(A, B):
        g_ = Gf(mesh=mesh_tau, target_shape=[1, 1])
        g_.data[:, 0, 0] = ed.get_tau_greens_function_component(tau, A, B)
        return g_

    return gf(cc[up[0]], cd[up[0]]), gf(cc[do[0]], cd[do[0]]), gf(Sz, Sz), gf(Sp, Sm)


def make_transverse_solver(eps, g, omega0, beta=2.1, w_max=2.0, dlr_eps=1e-12,
                           encoding='single-minus', conserved_operators=[], h=0.0):

    """ Zero-hybridization spinful impurity with the transverse retarded interaction. """

    n_orb = len(eps)
    S = Solver(
        H_loc=H_local(eps, h), beta=beta, w_max=w_max, eps=dlr_eps,
        gf_struct=[['up', n_orb], ['do', n_orb]],
        conserved_operators=conserved_operators)

    for bl, _ in S.Delta_tau:
        S.Delta_tau[bl].data[:] = 0.

    b_mesh = MeshDLRImTime(beta=beta, statistic='Boson', eps=S.mesh_tau.eps,
                           w_max=S.mesh_tau.w_max, symmetrize=False)

    D00 = Gf(mesh=b_mesh, target_shape=[1, 1])          # -g^2 <T b(tau) b^dag(0)>_0
    iw = make_gf_dlr_imfreq(D00)
    iw << g**2 * inverse(iOmega_n - omega0)
    D00 << make_gf_dlr_imtime(iw)

    D11 = Gf(mesh=b_mesh, target_shape=[1, 1])          # -g^2 <T b^dag(tau) b(0)>_0 = D00(beta-tau)
    iw = make_gf_dlr_imfreq(D11)
    iw << -g**2 * inverse(iOmega_n + omega0)
    D11 << make_gf_dlr_imtime(iw)

    Sp, Sm = S_plus(n_orb), S_minus(n_orb)

    if encoding == 'single-minus':                      # full kernel, NO halving
        D, ops = Gf(mesh=b_mesh, target_shape=[1, 1]), [Sm]
        D.data[:] = D00.data
    elif encoding == 'single-plus':
        D, ops = Gf(mesh=b_mesh, target_shape=[1, 1]), [Sp]
        D.data[:] = D11.data
    elif encoding in ('two-reflected', 'two-naive'):    # 0.5 for the forward/backward doubling
        D, ops = Gf(mesh=b_mesh, target_shape=[2, 2]), [Sm, Sp]
        D.data[:] = 0.
        D.data[:, 0, 0] = 0.5 * D00.data[:, 0, 0]
        D.data[:, 1, 1] = 0.5 * (D00 if encoding == 'two-naive' else D11).data[:, 0, 0]
    else:
        raise ValueError(f'unknown encoding {encoding}')

    S.set_dynamic_interactions(dynint_ops=ops, dynint_tau=D)
    return S, D00, D11


def _solve(S, order):
    S.solve(max_order=order, spgf_max_order=order, maxiter=30, tol=1e-12,
            verbose=False, hyb_comp=True)


def _errors(S, order, n_orb, ref):
    g_up_ed, g_do_ed, chi_zz_ed, chi_pm_ed = ref
    w = np.array([0.5] * n_orb + [-0.5] * n_orb)
    ops_n = [n('up', m) for m in range(n_orb)] + [n('do', m) for m in range(n_orb)]
    chi = S.eval_one_time_correlator(S.G, order, ops_n, ops_n)
    chi_zz = np.einsum('tij,i,j->t', chi.data, w, w)
    chi_pm = S.eval_one_time_correlator(S.G, order, [S_plus(n_orb)], [S_minus(n_orb)])
    return dict(
        G_up=np.max(np.abs(S.G_tau['up'].data[:, 0, 0] - g_up_ed.data[:, 0, 0])),
        G_do=np.max(np.abs(S.G_tau['do'].data[:, 0, 0] - g_do_ed.data[:, 0, 0])),
        chi_zz=np.max(np.abs(chi_zz - chi_zz_ed.data[:, 0, 0])),
        chi_pm=np.max(np.abs(chi_pm.data[:, 0, 0] - chi_pm_ed.data[:, 0, 0])))


# ------------------------------------------------------------------- 1. encoding acceptance

def test_transverse_encodings(eps=[0.025], h=-0.125, g=0.1, omega0=1.0, order=2,
                              conserved_operators=[], verbose=False):

    """ One spinful level, transverse interaction. Asserts that the three equivalent encodings agree
    with each other exactly and with ED, and that the naive 0.5*diag(D00, D00) does not. """

    eps, n_orb = list(eps), len(eps)
    S0, _, _ = make_transverse_solver(eps, g, omega0, encoding='single-minus',
                                      conserved_operators=conserved_operators, h=h)
    ref = get_ed_ref_transverse(eps, g, omega0, S0.mesh_tau, Nb_max=12, h=h)
    ref0 = get_ed_ref_transverse(eps, 0.0, omega0, S0.mesh_tau, Nb_max=12, h=h)

    # -- non-vacuity: the interaction must move the observables far more than the tolerances
    effect = {k: np.max(np.abs(a.data - b.data)) for k, a, b in
              zip(['G_up', 'chi_zz', 'chi_pm'], [ref[0], ref[2], ref[3]],
                  [ref0[0], ref0[2], ref0[3]])}
    assert effect['G_up'] > 1e-3 and effect['chi_zz'] > 1e-4 and effect['chi_pm'] > 1e-3, \
        f'the retarded interaction barely moves the ED reference, the test is vacuous: {effect}'

    errs = {}
    for encoding in ['single-minus', 'single-plus', 'two-reflected', 'two-naive']:
        S = S0 if encoding == 'single-minus' else make_transverse_solver(
            eps, g, omega0, encoding=encoding, conserved_operators=conserved_operators, h=h)[0]
        _solve(S, order)
        errs[encoding] = _errors(S, order, n_orb, ref)
        if verbose:
            print(f'{encoding:>14s}: ' + '  '.join(f'{k} {v:2.2E}' for k, v in errs[encoding].items()))

    for encoding in ['single-minus', 'single-plus', 'two-reflected']:
        e = errs[encoding]
        assert e['G_up'] < 1e-7 and e['G_do'] < 1e-7, f'{encoding} G vs ED: {e}'
        assert e['chi_zz'] < 1e-5, f'{encoding} chi_zz vs ED: {e}'
        assert e['chi_pm'] < 1e-5, f'{encoding} chi+- vs ED: {e}'

    # the three encodings must be the same calculation, not merely both accurate
    for encoding in ['single-plus', 'two-reflected']:
        for k in errs[encoding]:
            assert abs(errs[encoding][k] - errs['single-minus'][k]) < 1e-12, \
                f'{encoding} and single-minus differ in {k}'

    # and the naive diag(D00, D00) must be caught, not silently accepted
    assert errs['two-naive']['G_up'] > 1e-4, \
        'the naive 0.5*diag(D00, D00) encoding no longer fails - reflected weights may have ' \
        'stopped being necessary, or this test stopped reaching the interaction'

    return errs


# --------------------------------------------- 2. index convention, read off the Fock states

def test_transverse_sigma_convention(eps=[0.025], h=-0.125, g=0.3, omega0=1.0, verbose=False):

    """ Fock-state route to the index convention, independent of the ED comparison.

    At order 1 with zero hybridization the pseudo-particle self-energy is one interaction line,

        Sigma_aa(tau) = - D(tau) * G_bb(tau) * |<b|O|a>|^2 ,

    so with dynint_ops = [S^-] and the asymmetric kernel D00(tau) ~ e^{-omega0 tau} the decaying kernel
    must land on |up> and the reflected, growing one on |do>, i.e. D_00 <-> S^+(tau) ... S^-(0). UP and
    DO are identified from the pseudo-particle decay rates, which the Zeeman splitting h separates, the
    Fock ordering itself is pinned by dynint_spin_flip.py::test_fock_index_convention. """

    # |up> and |do> are told apart by their pseudo-particle decay rates, which requires h != 0
    assert h != 0.
    S, D00, D11 = make_transverse_solver(eps, g, omega0, encoding='single-minus', h=h)
    S.solve(max_order=1, spgf_max_order=1, maxiter=30, tol=1e-12, verbose=False, hyb_comp=True)

    Gb = [x for _, x in S.G][0]
    Sb = [x for _, x in S.Sigma][0]
    tau = np.array([float(t) for t in Gb.mesh])
    order_ = np.argsort(tau)
    tau, Gd, Sd = tau[order_], Gb.data.real[order_], Sb.data.real[order_]

    # identify the pp indices from the propagator decay rates E_a (up to a common shift eta)
    i, j = 1, len(tau) - 2
    E = -np.log(np.abs(Gd[j].diagonal() / Gd[i].diagonal())) / (tau[j] - tau[i])
    E = E - E.min()
    UP, DO = int(np.argmin(E)), int(np.argmax(E))       # h < 0 so |up> is the pp ground state
    assert UP != DO

    beta = S.mesh_tau.beta
    nB = 1. / (np.exp(beta * omega0) - 1.)
    d00 = -g**2 * (1 + nB) * np.exp(-omega0 * tau)
    d11 = -g**2 * nB * np.exp(+omega0 * tau)

    e_right = max(np.max(np.abs(Sd[:, UP, UP] + d00 * Gd[:, DO, DO])),
                  np.max(np.abs(Sd[:, DO, DO] + d11 * Gd[:, UP, UP])))
    e_swapped = np.max(np.abs(Sd[:, UP, UP] + d11 * Gd[:, DO, DO]))
    if verbose:
        print(f'Sigma convention: right {e_right:2.2E}  swapped {e_swapped:2.2E}')

    assert e_right < 1e-11, \
        f'dynint_ops=[S^-] no longer means S^+(tau) D_00 S^-(0): first order Sigma deviates ' \
        f'from -D00(tau) G_pp,do(tau) by {e_right:2.2E}'
    assert e_swapped > 1e-3, 'the swapped assignment is not distinguishable, test is vacuous'

    return e_right, e_swapped


# --------------------------------------------------------- 3. the transverse chi+- vertex arm

def test_transverse_chi_pm(eps=[-0.1, 0.15], g=0.1, omega0=1.0, order=2,
                           conserved_operators=[], verbose=False):

    """ chi+-(tau) = -<T S^+(tau) S^-(0)> against ED.

    A single spinful level has no order-2 correlator contribution to chi+-, so two spinful orbitals with
    the interaction on the total spin are used to reach the vertex correction. This also exercises the
    fermion-parity classification, S^+/S^- must be treated as bosonic by compute_one_time_correlator. """

    n_orb = len(eps)
    assert n_orb >= 2, 'chi+- needs >= 2 spinful orbitals to see the order-2 vertex'

    S, _, _ = make_transverse_solver(eps, g, omega0, encoding='single-minus',
                                     conserved_operators=conserved_operators)
    _solve(S, order)
    ref = get_ed_ref_transverse(eps, g, omega0, S.mesh_tau, Nb_max=10)
    ref0 = get_ed_ref_transverse(eps, 0.0, omega0, S.mesh_tau, Nb_max=10)

    chi_pm_ed = ref[3].data[:, 0, 0]
    effect = np.max(np.abs(chi_pm_ed - ref0[3].data[:, 0, 0]))

    Sp, Sm = S_plus(n_orb), S_minus(n_orb)
    chi1 = S.eval_one_time_correlator(S.G, 1, [Sp], [Sm]).data[:, 0, 0]
    chi2 = S.eval_one_time_correlator(S.G, order, [Sp], [Sm]).data[:, 0, 0]

    e1, e2 = np.max(np.abs(chi1 - chi_pm_ed)), np.max(np.abs(chi2 - chi_pm_ed))
    vertex = np.max(np.abs(chi2 - chi1))
    if verbose:
        print(f'chi+-: order1 {e1:2.2E}  order{order} {e2:2.2E}  vertex {vertex:2.2E}  '
              f'effect {effect:2.2E}  |chi+-| {np.max(np.abs(chi_pm_ed)):2.4f}')

    assert effect > 1e-3, 'the interaction barely moves chi+-, the test is vacuous'
    assert vertex > 1e-4, \
        'the order-2 correlator contribution to chi+- vanishes, so this test does not reach ' \
        'the transverse vertex'
    assert e2 < 1e-4, f'chi+- deviates from ED by {e2:2.2E}'
    assert e2 < 0.02 * e1, f'order {order} did not improve on order 1 ({e2:2.2E} vs {e1:2.2E})'

    return e1, e2, vertex


if __name__ == '__main__':
    N_up = sum(n('up', m) for m in range(2))
    N_do = sum(n('do', m) for m in range(2))
    test_transverse_sigma_convention(verbose=True)
    test_transverse_encodings(verbose=True)
    test_transverse_encodings(verbose=True, conserved_operators=[n('up', 0), n('do', 0)])
    test_transverse_chi_pm(verbose=True)
    test_transverse_chi_pm(verbose=True, conserved_operators=[N_up, N_do])
