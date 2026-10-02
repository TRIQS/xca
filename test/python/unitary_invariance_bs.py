""" Single-particle basis unitary invariance of the dense and block-sparse evaluators.

The PPSC equations are invariant under a unitary transform of the single-particle basis: solving the same
problem in two bases and rotating one result back must give the same Green's function, exactly and without
an ED reference. unitary_invariance.py asserts this for the legacy Solver path, this file for the two
evaluators the dynamical-interaction port uses, on the same fixture.

The discriminant is the rotation, not the hybridization. The backward branch of the zero-vertex line
computes sum_{mu,kap} DeltaR_{mu,kap} c_mu G c^dag_kap, which rotates to V^T DeltaR V* where the rotated
problem computes V^dag DeltaR V. The two agree for every Delta exactly when conj(V) = e^{i theta} V, i.e.
when V is a global phase times a real matrix, so a real rotation is blind to an untransposed backward
index whatever Delta is, and the control below is the same Delta under a real rotation. The non-vacuity
check is that V^dag V* is not proportional to the identity, which is the actual condition. """

import numpy as np

from triqs.gfs import Gf, MeshDLRImTime
from triqs.operators import c, c_dag

from pyed.OperatorUtils import operator_single_particle_transform

from triqs_xca import Solver


beta, w_max, eps = 10.0, 10.0, 1e-12

# Order 1 has only the vertex-0 line, so a failure localises to it, and order 2 pins the endpoint line
# against the interior lines it has to match
orders = (1, 2)


def hamiltonian(eps1=-0.1, t=0.5 + 0.5j, U=1.0):
    n_up = c_dag('0', 0) * c('0', 0)
    n_do = c_dag('0', 1) * c('0', 1)
    return eps1 * (n_up + n_do) + U * n_up * n_do \
        + t * c_dag('0', 0) * c('0', 1) + np.conj(t) * c_dag('0', 1) * c('0', 0)


def greens_function(H, delta_tau_data, conserved_operators, order):
    """ G_tau of one basis, at fixed order, with the hybridization given directly. """
    S = Solver(H_loc=H, beta=beta, w_max=w_max, eps=eps,
                          gf_struct=[['0', 2]], conserved_operators=conserved_operators,
                          verbose=False)
    S.Delta_tau['0'].data[:] = delta_tau_data
    # maxiter=1, invariance must hold at every iteration
    S.solve(max_order=order, spgf_max_order=order, maxiter=1, tol=1e-12,
            verbose=False, hyb_comp=False)
    return np.array(S.G_tau['0'].data)


def free_hybridization(S_like_mesh, h, V=0.2):
    """ Delta(tau) = V^2 * G_free(tau; h), built on the solver's own DLR mesh. """
    g = Gf(mesh=S_like_mesh, target_shape=(2, 2))
    tau = np.array([float(t) for t in g.mesh])
    w, U = np.linalg.eigh(h)
    # G_free(tau) = -U diag(e^{-tau w} / (1 + e^{-beta w})) U^dag
    expo = np.exp(-np.outer(tau, w)) / (1.0 + np.exp(-beta * w))[None, :]
    g.data[:] = -np.einsum('ab,tb,cb->tac', U, expo, U.conj())
    return V**2 * g.data


def _mesh():
    S = Solver(H_loc=hamiltonian(), beta=beta, w_max=w_max, eps=eps,
                          gf_struct=[['0', 2]], conserved_operators=[], verbose=False)
    return S.Delta_tau['0'].mesh


def _check(rotation, conserved_operators, tag, order, verbose=False):
    """ Solve in two bases related by `rotation`, rotate the second back, compare. """
    fops = [c('0', 0), c('0', 1)]
    U = rotation

    np.testing.assert_allclose(U @ U.conj().T, np.eye(2), atol=1e-14,
                               err_msg='the rotation is not unitary')

    H1 = hamiltonian()
    H2 = operator_single_particle_transform(H1, U, fops)

    h = np.array([[0.1, 0.1], [0.1, -0.1]])
    d1 = free_hybridization(_mesh(), h)
    d2 = np.einsum('ab,tbc,cd->tad', U.conj().T, d1, U)

    g1 = greens_function(H1, d1, conserved_operators, order)
    g2 = greens_function(H2, d2, conserved_operators, order)
    g2_back = np.einsum('ab,tbc,cd->tad', U, g2, U.conj().T)

    diff = np.max(np.abs(g1 - g2_back))
    scale = np.max(np.abs(g1))
    if verbose:
        print(f'{tag}: max|g1 - U g2 U^dag| = {diff:.4e}   max|g1| = {scale:.4e}   rel = {diff/scale:.4e}')
    return diff, scale, d2


def test_unitary_invariance_real_rotation(verbose=False):
    """ Control. A real rotation is covariant with or without the index defect, so a red here means the
    harness is broken and not the evaluator. It catches a wrong direction of the Delta, H or g2 transforms,
    but not a conjugation-only error, since a real rotation is its own conjugate. """
    th = 0.4
    V = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=complex)
    assert np.max(np.abs(V - V.conj())) < 1e-14, 'this control must use a REAL rotation'

    for order in orders:
        for conserved_operators, tag in (([], 'dense'), ('automatic', 'block-sparse')):
            diff, scale, _ = _check(V, conserved_operators, f'control/{tag}', order, verbose)
            assert diff < 1e-10 * scale, \
                f'{tag}, order {order}: the real-rotation control is not covariant ' \
                f'({diff:.4e} vs scale {scale:.4e}) -- the harness is broken, not the evaluator'


def test_unitary_invariance_complex_rotation(verbose=False):
    """ A complex rotation exposes an untransposed backward index. The legacy Solver path passes the
    identical assertion, since fastdiagram.cpp transposes the reflected hybridization at construction. """
    U = np.array([[1.0, 1j], [1.0, -1j]]) / np.sqrt(2)

    # the blind rotations are a global phase times a real matrix, for which U^dag U* is proportional to the identity
    M = U.conj().T @ U.conj()
    assert np.max(np.abs(M - np.trace(M) / U.shape[0] * np.eye(U.shape[0]))) > 0.1, \
        'vacuous test: the rotation is a global phase times a real matrix, so it cannot see the ' \
        'backward-index defect'

    # measure both evaluators at both orders before asserting
    bad = {}
    for order in orders:
        for conserved_operators, tag in (([], 'dense'), ('automatic', 'block-sparse')):
            diff, scale, d2 = _check(U, conserved_operators, f'complex/{tag}', order, verbose)
            if not diff < 1e-10 * scale:
                bad[f'{tag} order {order}'] = (diff, scale)
            # the asymmetry of the rotated hybridization is incidental, not the discriminant
            if verbose:
                asym = np.max(np.abs(d2 - np.transpose(d2, (0, 2, 1))))
                print(f'    (rotated Delta asymmetry {asym:.3e} -- incidental, not the discriminant)')

    assert not bad, \
        'single-particle unitary invariance is violated under a COMPLEX rotation: ' \
        + '; '.join(f'{k} by {d:.4e} against max|g| = {s_:.4e}' for k, (d, s_) in bad.items()) \
        + '. Check that the zero-vertex hybridization line transposes its coefficient index on the ' \
          'backward branch and that the interior lines agree with it. An order-1 failure points at the ' \
          'endpoint line, an order-2-only failure at the interior lines.'


if __name__ == '__main__':
    test_unitary_invariance_real_rotation(verbose=True)
    test_unitary_invariance_complex_rotation(verbose=True)
