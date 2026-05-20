"""

Test the one-time correlator computation by comparison with 
the single particle Green's function evaluator.

"""

import numpy as np


from triqs.operators import n, c, c_dag, dagger

from triqs.gfs import make_gf_dlr_imfreq, make_gf_dlr_imtime, SemiCircular, make_gf_imtime


from triqs_xca.block_sparse_solver import BlockSparseSolver


def get_dens_mat(S, fops):
    dens_mat = np.zeros((len(fops), len(fops)), dtype=complex)
    for i, op_i in enumerate(fops):
        for j, op_j in enumerate(fops):
            dens_mat[i, j] = S.expectation_value(dagger(op_i) * op_j)
    return dens_mat


def get_ed_dens_mat(H_loc, fops, beta):
    from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization
    ed = TriqsExactDiagonalization(H_loc, fops, beta)
    dens_mat_ref = np.zeros((len(fops), len(fops)), dtype=complex)
    for i, op_i in enumerate(fops):
        for j, op_j in enumerate(fops):
            dens_mat_ref[i, j] = ed.get_expectation_value(dagger(op_i) * op_j)
    return dens_mat_ref


def test_expectation_value(conserved_operators, cf_pyed=False):
    
    # -- Parameters
    
    B = 1.0
    mu = 0.4
    soc = 0.1 - 0.2j

    beta = 2.0
    eps = 1e-10
    w_max = 6.0

    # -- Local Hamiltonian
    
    gf_struct = [['0', 2]]

    h_ij = np.array([
        [-mu + B, soc],
        [np.conj(soc), -mu - B]
        ])

    fops = [c('0', 0), c('0', 1)]

    H_loc = sum([ dagger(op_i) * h_ij[i, j] * op_j for i, op_i in enumerate(fops) for j, op_j in enumerate(fops) ])

    S = BlockSparseSolver(H_loc, beta, w_max, eps, gf_struct=gf_struct,
        conserved_operators=conserved_operators)
    dens_mat = get_dens_mat(S, fops)

    if cf_pyed:
        dens_mat_ed = get_ed_dens_mat(H_loc, fops, beta)
        print(f'dens_mat = \n', dens_mat)
        print(f'dens_mat_ed = \n', dens_mat_ed)
        np.testing.assert_array_almost_equal(dens_mat, dens_mat_ed)

    # Single particle unitary tranformation

    U = np.array([ [1, 1 + 1.j], [1 - 1.j, -1] ]) / np.sqrt(3)

    np.testing.assert_array_almost_equal(np.eye(2), U @ U.conj().T)
    np.testing.assert_array_almost_equal(np.eye(2), U.conj().T @ U)

    # Compute density matrix in transformed basis

    #h_ij_t = U.conj().T @ h_ij @ U
    h_ij_t = U.conj() @ h_ij @ U.T

    print(f'h_ij = \n', h_ij)
    print(f'eigenvalues = \n', np.linalg.eigvalsh(h_ij))
    print(f'h_ij_t = \n', h_ij_t)
    print(f'eigenvalues_t = \n', np.linalg.eigvalsh(h_ij_t))

    H_loc_t = sum([ dagger(op_i) * h_ij_t[i, j] * op_j for i, op_i in enumerate(fops) for j, op_j in enumerate(fops) ])
    S_t = BlockSparseSolver(H_loc_t, beta, w_max, eps, gf_struct=gf_struct,
        conserved_operators=conserved_operators)
    dens_mat_t = get_dens_mat(S_t, fops)

    if cf_pyed:
        dens_mat_t_ed = get_ed_dens_mat(H_loc_t, fops, beta)
        print(f'dens_mat_t = \n', dens_mat_t)
        print(f'dens_mat_t_ed = \n', dens_mat_t_ed)
        np.testing.assert_array_almost_equal(dens_mat_t, dens_mat_t_ed)

    dens_mat_invt = U @ dens_mat_t @ U.conj().T

    print(f'dens_mat_t = \n', dens_mat_t)
    print(f'dens_mat_invt = \n', dens_mat_invt)
    print(f'dens_mat = \n', dens_mat)

    np.testing.assert_array_almost_equal(dens_mat, dens_mat_invt)


if __name__ == '__main__':
    test_expectation_value(conserved_operators=[]) # Test DenseDiagramEvaluator
    test_expectation_value(conserved_operators='automatic') # Test BlockSparseDiagramEvaluator
