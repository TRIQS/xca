""" Test the slot semantics of the dynamical interaction coefficients with the non-hermitian operators
S+ and S- on the dense evaluator, without an ED reference: dynint_ops = [S-] with the weight in slot
(0, 0) means S+(tau) ... S-(0), and a single entry with the full weight equals two entries with
0.5 * diag(w, w_reflected). The self-energy is evaluated directly on the bare pseudo-particle propagator,
so there is no self-consistency and no hybridization fit. """

import numpy as np

from triqs.operators import n, c, c_dag

from triqs_xca import Solver
from triqs_xca.diag import all_connected_pairings

beta, w_max, eps = 2.0, 5.0, 1e-10

# Fock state indices, asserted in test_fock_index_convention()
FOCK_EMPTY, FOCK_UP, FOCK_DO, FOCK_UPDO = 0, 1, 2, 3


def S_minus():
    return c_dag('do', 0) * c('up', 0)


def S_plus():
    return c_dag('up', 0) * c('do', 0)


def make_evaluator(dynint_ops, dynint_coeffs, poles, hyb_coeffs=None, e_up=-0.3, e_do=0.7):
    """ Single spinful level on the dense path, with the pole set and the interaction coefficients given
    directly rather than fitted, so that a +-symmetric pole pair can be used. e_up != e_do makes the four
    Fock states non-degenerate. """
    S = Solver(H_loc=e_up * n('up', 0) + e_do * n('do', 0),
                          beta=beta, w_max=w_max, eps=eps,
                          gf_struct=[['up', 1], ['do', 1]],
                          conserved_operators=[], verbose=False)
    p = len(poles)
    if hyb_coeffs is None:
        hyb_coeffs = np.zeros((p, 2, 2), dtype=complex)
    S.set_hybridization_poles_and_coefficients(np.array(poles, dtype=float), hyb_coeffs)
    if dynint_ops:
        S.has_dynamic_interactions = True
        S.dynint_ops = dynint_ops
        S.dynint_coeffs = np.ascontiguousarray(np.asarray(dynint_coeffs, dtype=complex))
    S.init_diagram_evaluator()
    return S


def sigma_of_order(S, order):
    """ The self-energy of one expansion order on the bare pseudo-particle propagator, since the S+/S-
    interaction contributes nothing at order 2 on this model. The only factor is the overall minus of the
    solver, the topology parity is already carried by Backbone::get_parity() inside C++. """
    out = None
    for _sign, topology in all_connected_pairings(order):
        X = S.d.compute_self_energy(S.G0, np.array(topology, dtype=np.int32))
        v = -np.array([np.asarray(b.data) for _, b in X][0])  # one subspace on the dense path
        out = v if out is None else out + v
    return out


def peak(sigma, i):
    return np.max(np.abs(sigma[:, i, i]))


# ----------------------------------------------------------------------------------------------
# calibration


def test_fock_index_convention():
    """ Pin the Fock state indexing of the dense pseudo-particle objects: indexed by Fock state and not by
    eigenstate, with bit 0 of the Fock integer being 'up'. Both follow from the support of the order-1
    self-energy of a density interaction. """
    poles = [1.3, -1.3]
    d = np.zeros((2, 1, 1), dtype=complex)
    d[:, 0, 0] = [0.61, -0.37]

    for op, label, expected in [(n('up', 0), 'n_up', {FOCK_UP, FOCK_UPDO}),
                                (n('do', 0), 'n_do', {FOCK_DO, FOCK_UPDO})]:
        sigma = sigma_of_order(make_evaluator([op], d, poles), 1)
        assert sigma.shape[1:] == (4, 4), f'{label}: expected a 4x4 single-subspace self-energy'
        support = {i for i in range(4) if peak(sigma, i) > 1e-10}
        assert support == expected, \
            f'{label}: self-energy supported on Fock states {sorted(support)}, expected {sorted(expected)}'
        offdiag = np.max(np.abs(sigma - np.einsum('tii->ti', sigma)[:, :, None] * np.eye(4)[None, :, :]))
        assert offdiag < 1e-12, f'{label}: self-energy is not diagonal in the Fock basis, max offdiag {offdiag:.2e}'


# ----------------------------------------------------------------------------------------------
# 1. slot semantics


def test_slot_semantics_of_a_single_interaction_entry(verbose=False):
    """ dynint_ops = [S-] with the weight in slot (0, 0) means S+(tau) ... S-(0).

    A coefficient entry pairs an operator with the dagger of an operator, F_dags(n_hyb + i) = O_i^dag, so
    with O_0 = S- the diagonal entry multiplies S+ ... S- and the order-1 self-energy is dominant on |up>,
    and on |do> for [S+]. Both spin states carry weight, since the backward direction of the interaction
    line supplies the hermitian partner with the reflected kernel. The density-density tests cannot see a
    transposed slot since n = n^dag, and the literal transpose D_ij -> D_ji is unobservable for the
    physical diag(D_{+-}, D_{-+}). """
    poles = [1.3, -1.3]
    d = np.zeros((2, 1, 1), dtype=complex)
    d[:, 0, 0] = [0.61, -0.37]

    s_minus = sigma_of_order(make_evaluator([S_minus()], d, poles), 1)
    s_plus = sigma_of_order(make_evaluator([S_plus()], d, poles), 1)

    up_m, do_m = peak(s_minus, FOCK_UP), peak(s_minus, FOCK_DO)
    up_p, do_p = peak(s_plus, FOCK_UP), peak(s_plus, FOCK_DO)
    if verbose:
        print(f'[S-]: |Sigma| on |up> = {up_m:.6e}, on |do> = {do_m:.6e}, ratio {up_m / do_m:.4f}')
        print(f'[S+]: |Sigma| on |up> = {up_p:.6e}, on |do> = {do_p:.6e}, ratio {do_p / up_p:.4f}')

    # The S_z = 0 states are untouched: S+ and S- annihilate both of them.
    for sigma, label in [(s_minus, '[S-]'), (s_plus, '[S+]')]:
        for i in (FOCK_EMPTY, FOCK_UPDO):
            assert peak(sigma, i) < 1e-12, f'{label}: Sigma is nonzero on the S_z = 0 Fock state {i}'

    # check that both spin states carry weight
    for v, name in [(up_m, '[S-] on |up>'), (do_m, '[S-] on |do>'),
                    (up_p, '[S+] on |up>'), (do_p, '[S+] on |do>')]:
        assert v > 1e-3, f'vacuous test: {name} is {v:.2e}'

    assert up_m > 1.5 * do_m, \
        f'[S-] must be dominant on |up>, got |up> = {up_m:.6e} and |do> = {do_m:.6e} -- the slot ' \
        f'convention has been transposed: D~_00 is being read as S-(tau) ... S+(0)'
    assert do_p > 1.5 * up_p, \
        f'[S+] must be dominant on |do>, got |do> = {do_p:.6e} and |up> = {up_p:.6e}'

    # [S+] on the model with e_up and e_do exchanged is the up <-> do mirror image of [S-], exactly
    mirror = sigma_of_order(make_evaluator([S_plus()], d, poles, e_up=0.7, e_do=-0.3), 1)
    perm = [FOCK_EMPTY, FOCK_DO, FOCK_UP, FOCK_UPDO]
    err = np.max(np.abs(s_minus - mirror[:, perm][:, :, perm]))
    assert err < 1e-13, \
        f'[S-] and the up<->do mirror of [S+] differ by {err:.3e}; the two operators are not ' \
        f'entering the same slot the same way'
    return up_m / do_m, do_p / up_p


# ----------------------------------------------------------------------------------------------
# 2. single-entry vs two-entry


def reflected_weights(w, poles):
    """ DLR weights of D(beta - tau) given the weights of D(tau). Reflection in tau negates every pole, so
    the reflected function is representable on the same pole set only if it is closed under negation, and
    the reflected weights are then the pole-negation permutation of w. """
    poles = np.asarray(poles, dtype=float)
    idx = []
    for p in poles:
        matches = np.flatnonzero(np.abs(poles + p) < 1e-12)
        assert len(matches) == 1, \
            f'pole set is not closed under negation: {-p} is not in {poles}. The two-entry ' \
            f'encoding needs a +-symmetric pole set; there is no exact reflection otherwise.'
        idx.append(matches[0])
    return np.asarray(w)[idx]


def test_interaction_contributes_at_the_order_under_test(verbose=False):
    """ Guard for the test below: the S+/S- interaction contributes nothing to the order-2 pseudo-particle
    self-energy on this model, so the encoding comparison has to be made at an order where the
    interaction is present. """
    poles = [1.3, -1.3]
    d = np.zeros((2, 1, 1), dtype=complex)
    d[:, 0, 0] = [0.61, -0.37]
    hyb = np.zeros((2, 2, 2), dtype=complex)
    hyb[0] = np.diag([0.35, 0.22])
    hyb[1] = np.diag([-0.18, 0.27])

    S_int = make_evaluator([S_minus()], d, poles, hyb)
    S_ferm = make_evaluator([], None, poles, hyb)

    influence = {}
    for order in (1, 2, 3):
        influence[order] = np.max(np.abs(sigma_of_order(S_int, order) - sigma_of_order(S_ferm, order)))
        if verbose:
            print(f'order {order}: max|Sigma_[S-] - Sigma_no_dynint| = {influence[order]:.6e}')

    assert influence[1] > 1e-3, 'the S+/S- interaction does not reach the order-1 self-energy'
    # a difference of two independently enumerated sums, so not exactly zero
    assert influence[2] < 1e-15, \
        'the S+/S- interaction contributes at order 2 on this fixture'
    # The order-2 vanishing is a property of this fixture: on one spinful orbital every crossing diagram
    # with two interaction lines contains S+S+ or S-S-, and the mixed diagram needs a spin-off-diagonal
    # hybridization. On two spinful orbitals the transverse interaction does reach order 2.
    assert influence[3] > 1e-5, \
        'the S+/S- interaction does not reach the order-3 self-energy either, so the order-2 zero above ' \
        'is not a selection rule but missing plumbing'
    return influence


def test_single_entry_equals_two_entry(verbose=False):
    """ The three encodings of one physical S+ D_{+-} S- + S- D_{-+} S+ interaction,

      A  dynint_ops = [S-],      dynint_coeffs = w                       (single entry, no halving)
      B  dynint_ops = [S-, S+],  dynint_coeffs = 0.5 * diag(w, w_refl)   (two entries)
      C  dynint_ops = [S-, S+],  dynint_coeffs = 0.5 * diag(w, w)        (two entries, wrong)

    with A == B and A != C. A single entry generates both hermitian partners, since the backward direction
    of the interaction line carries the kernel of D(beta - tau) and KMS gives D_{+-}(beta - tau) = D_{-+}(tau).
    A second operator therefore double counts, which the 0.5 removes, and it must carry the reflected
    weights. diag(w, w) asserts D_{-+} = D_{+-}, which only holds for a tau-symmetric D. """
    poles = [1.3, -1.3]
    w = np.array([0.61, -0.37])
    w_refl = reflected_weights(w, poles)
    assert not np.allclose(w, w_refl), \
        'vacuous test: the reflected weights equal the unreflected ones, so B and C coincide'

    dA = np.zeros((2, 1, 1), dtype=complex)
    dA[:, 0, 0] = w
    dB = np.zeros((2, 2, 2), dtype=complex)
    dB[:, 0, 0], dB[:, 1, 1] = 0.5 * w, 0.5 * w_refl
    dC = np.zeros((2, 2, 2), dtype=complex)
    dC[:, 0, 0], dC[:, 1, 1] = 0.5 * w, 0.5 * w

    ops1, ops2 = [S_minus()], [S_minus(), S_plus()]
    order = 1   # see test_interaction_contributes_at_the_order_under_test
    A = sigma_of_order(make_evaluator(ops1, dA, poles), order)
    B = sigma_of_order(make_evaluator(ops2, dB, poles), order)
    C = sigma_of_order(make_evaluator(ops2, dC, poles), order)

    scale = np.max(np.abs(A))
    err_B, err_C = np.max(np.abs(A - B)), np.max(np.abs(A - C))
    if verbose:
        print(f'max|A| = {scale:.6e}, |A - B| = {err_B:.6e}, |A - C| = {err_C:.6e}')

    assert scale > 1e-2, f'vacuous test: max|Sigma| = {scale:.2e}'
    assert err_B < 1e-9 * max(scale, 1.0), \
        f'single-entry and reflected two-entry disagree: |A - B| = {err_B:.3e}, max|A| = {scale:.3e}'
    assert err_C > 0.1 * scale, \
        f'the NAIVE 0.5 * diag(w, w) agrees with the single-entry encoding to {err_C:.3e}. It must ' \
        f'not: it asserts D_{{-+}} = D_{{+-}}. Either the pole set stopped being asymmetric under ' \
        f'reflection, or the second coefficient slot is being ignored.'
    return scale, err_B, err_C


if __name__ == '__main__':
    test_fock_index_convention()
    print('ratios (S-, S+):', test_slot_semantics_of_a_single_interaction_entry(verbose=True))
    print('influence per order:', test_interaction_contributes_at_the_order_under_test(verbose=True))
    print('A/B/C:', test_single_entry_equals_two_entry(verbose=True))
    print('OK')
