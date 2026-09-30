""" Test the ordering of the block-sparse single-particle Green's function, which must be indexed by
orbital and not by (symmetry set, orbital) when the symmetry sets interleave with the orbital order. """

import numpy as np

from triqs.operators import n, c, c_dag

from triqs_xca.block_sparse_solver import BlockSparseSolver


def make_solver(conserved_operators, beta=2.0, norb=2, U=1.0, mu=0.25, V=0.1, hyb=-0.3):

    H = sum(U * n('up', i) * n('do', i)
            + mu * (n('up', i) + n('do', i))
            + V * (c_dag('up', i) * c('do', i) + c_dag('do', i) * c('up', i))
            for i in range(norb))

    gf_struct = [('up', norb), ('do', norb)]

    S = BlockSparseSolver(H_loc=H, beta=beta, w_max=20.0, eps=1e-8, gf_struct=gf_struct,
                          conserved_operators=conserved_operators, verbose=False)

    # orbital-diagonal hybridization
    for _, g in S.Delta_tau:
        for i in range(norb):
            g.data[:, i, i] = hyb

    # order 1 and a fixed number of iterations suffice, the permutation is in the external legs
    S.solve(max_order=1, verbose=False, hyb_comp=False, maxiter=5)
    return S


def test_spgf_ordering():

    S_dense = make_solver([])            # no symmetries -> the dense evaluator
    S_bs    = make_solver('automatic')   # autopartitioning -> several interleaving symmetry sets

    assert S_dense.use_dense_solver
    assert not S_bs.use_dense_solver

    # check that the symmetry sets, grouped by c_connection rows, interleave with the orbital order
    ad = S_bs.atom_diag
    rows = [tuple(int(ad.c_connection(o, s)) for s in range(ad.n_subspaces))
            for o in range(len(S_bs.fundamental_operators))]
    labels, seen = [], {}
    for row in rows:
        labels.append(seen.setdefault(row, len(seen)))
    assert len(set(labels)) > 1, f'vacuous test: a single symmetry set, labels {labels}'
    assert labels != sorted(labels), \
        f'vacuous test: symmetry sets do not interleave with the orbital order, labels {labels}'

    for b, g_bs in S_bs.G_tau:
        g_dense = S_dense.G_tau[b]
        scale = np.max(np.abs(g_dense.data))
        assert scale > 1e-3, f'vacuous test: G_tau[{b}] is zero'
        err = np.max(np.abs(g_bs.data - g_dense.data))
        print(f'G_tau[{b}]: max|bs - dense| = {err:.3e}, max|dense| = {scale:.3e}')
        assert err < 1e-8 * scale, \
            f'block-sparse G_tau[{b}] disagrees with the dense reference by {err:.3e}'

    # The two orbitals are decoupled, so the intra-spin inter-orbital component is exactly zero.
    for b, g in S_bs.G_tau:
        offdiag = np.max(np.abs(g.data[:, 0, 1]))
        diag = np.max(np.abs(g.data[:, 0, 0]))
        print(f'G_tau[{b}]: max|G[0,1]| = {offdiag:.3e}, max|G[0,0]| = {diag:.3e}')
        assert diag > 1e-3, f'vacuous test: G_tau[{b}][0,0] is zero, nothing to compare against'
        assert offdiag < 1e-10 * diag, \
            f'G_tau[{b}][0,1] = {offdiag:.3e} is nonzero, but the two orbitals are decoupled'


if __name__ == '__main__':
    test_spgf_ordering()
