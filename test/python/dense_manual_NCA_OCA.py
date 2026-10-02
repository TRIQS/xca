""" Author: Hugo U. R. Strand (2025) """

import numpy as np

from triqs.operators import c, c_dag

from cppdlr.pycppdlr import ImTimeOps, build_dlr_rf

from triqs_xca.dense import NCA_dense, OCA_dense
from triqs_xca.block_sparse_solver import BlockSparseSolver
from triqs_xca.block_sparse_solver import hamiltonian_matrix_block, pseudo_particle_block_gf_to_dense


def full_operator_matrices(ad):
    """ Annihilation and creation operators in the Fock basis of the full Hilbert space,
    arrays of shape (norb, dim, dim) indexed [target, source]. """

    dim = ad.full_hilbert_space_dim
    norb = len(list(ad.fops))

    def full_matrix(oidx, connection, matrix):
        op_mat = np.zeros([dim]*2, dtype=complex)
        for s1 in range(ad.n_subspaces):
            s2 = connection(oidx, s1)
            if s2 < 0: continue
            block = ad.unitary_matrices[s2] @ matrix(oidx, s1) @ ad.unitary_matrices[s1].T.conj()
            op_mat[np.ix_(ad.fock_states[s2], ad.fock_states[s1])] = block
        return op_mat

    F = np.array([ full_matrix(i, ad.c_connection, ad.c_matrix) for i in range(norb) ])
    F_dag = np.array([ full_matrix(i, ad.cdag_connection, ad.cdag_matrix) for i in range(norb) ])

    return F, F_dag


def test_block_sparsity_NCA_dense(verbose=False):

    beta = 10.0
    t = 0.5
    mu = 1/3
    w_max = 10.0
    eps = 1e-10

    # Hopping and interaction make the local Hamiltonian non-diagonal in the Fock basis, so the
    # Fock-basis operators and propagators are not trivially the eigenbasis ones.
    t_loc = 0.2 + 0.1j
    U = 1.0
    n_0 = c_dag('0', 0) * c('0', 0)
    n_1 = c_dag('0', 1) * c('0', 1)
    H = -mu * (n_0 + n_1) + U * n_0 * n_1 \
        + t_loc * c_dag('0', 0) * c('0', 1) + np.conj(t_loc) * c_dag('0', 1) * c('0', 0)
    gf_struct = [['0', 2]]

    # A complex hermitian, non-symmetric bath with Delta_ab != Delta_ba, which makes the test sensitive to the
    # hybridization index order of the manual reference evaluators. A real hermitian matrix is symmetric and can not see it.
    ek = np.array([[0.1, 0.3 + 0.2j], [0.3 - 0.2j, -0.15]])

    for conserved_operators in ['automatic', []]:

        S = BlockSparseSolver(H, beta, w_max, eps, gf_struct=gf_struct,
                              conserved_operators=conserved_operators, verbose=False)

        # Delta(tau) = t^2 G_free(tau; ek), with G_free(tau) = -exp(-tau ek)/(1 + exp(-beta ek)) on the solver mesh
        tau = np.array([ float(x) for x in S.mesh_tau ])
        e, V = np.linalg.eigh(ek)
        occ = np.exp(-np.outer(tau, e)) / (1. + np.exp(-beta * e))[None, :]
        delta_iaa = -t**2 * np.einsum('ab,tb,cb->tac', V, occ, V.conj())
        S.Delta_tau['0'].data[:] = delta_iaa

        S.fit_hybridization(compression=False, verbose=False)
        S.init_diagram_evaluator()

        # -- Block-sparse solver reference, per order
        # The manual evaluators return the diagrams without the sign (-1)^order of the self-energy.

        Sigma_NCA_ref = pseudo_particle_block_gf_to_dense(
            S.eval_pseudo_particle_self_energy(S.G, 1), S.atom_diag).data.copy()
        Sigma_OCA_ref = pseudo_particle_block_gf_to_dense(
            S.eval_pseudo_particle_self_energy(S.G, 2), S.atom_diag).data - Sigma_NCA_ref

        # -- Solve using block-sparsity in dense formulation

        H_mat = np.zeros([S.atom_diag.full_hilbert_space_dim]*2, dtype=complex)
        for sidx in range(S.atom_diag.n_subspaces):
            fidx = S.atom_diag.fock_states[sidx]
            H_mat[np.ix_(fidx, fidx)] = hamiltonian_matrix_block(S.atom_diag, sidx)
        F, F_dag = full_operator_matrices(S.atom_diag)
        G0_iaa = pseudo_particle_block_gf_to_dense(S.G0, S.atom_diag).data.copy()

        # Operator matrices consistent with the Hamiltonian and the canonical anticommutation relations
        dim = H_mat.shape[0]
        for i in range(2):
            for j in range(2):
                np.testing.assert_allclose(F[i] @ F_dag[j] + F_dag[j] @ F[i], (i == j) * np.eye(dim), atol=1e-13)
                np.testing.assert_allclose(F[i] @ F[j] + F[j] @ F[i], 0., atol=1e-13)
        n_mat = [ F_dag[i] @ F[i] for i in range(2) ]
        H_ops = -mu * (n_mat[0] + n_mat[1]) + U * n_mat[0] @ n_mat[1] \
            + t_loc * F_dag[0] @ F[1] + np.conj(t_loc) * F_dag[1] @ F[0]
        np.testing.assert_allclose(H_ops, H_mat, atol=1e-13)

        # The dense evaluators take the hybridization and its reflection (beta - tau) on the DLR imaginary time nodes
        ito = ImTimeOps(w_max * beta, build_dlr_rf(w_max * beta, eps, S.dlr_symmetrize), symmetrize=S.dlr_symmetrize)
        # (the nodes are in relative format, negative for tau > beta/2)
        tau_ito = np.where(ito.get_itnodes() < 0, ito.get_itnodes() + 1., ito.get_itnodes()) * beta
        np.testing.assert_allclose(tau_ito, tau, atol=1e-12)

        delta_iaa = np.ascontiguousarray(delta_iaa, dtype=complex)
        delta_iaa_refl = ito.reflect(delta_iaa)

        print(f'conserved_operators = {conserved_operators}: evaluating NCA_dense and OCA_dense')
        Sigma_NCA = pow(-1, 1) * NCA_dense(delta_iaa, delta_iaa_refl, G0_iaa, F, F_dag)
        Sigma_OCA = pow(-1, 2) * OCA_dense(delta_iaa, ito, beta, G0_iaa, F, F_dag)

        diff_NCA = np.max(np.abs(Sigma_NCA - Sigma_NCA_ref))
        print(f'conserved_operators = {conserved_operators}, diff_NCA = {diff_NCA:2.2E}')

        diff_OCA = np.max(np.abs(Sigma_OCA - Sigma_OCA_ref))
        print(f'conserved_operators = {conserved_operators}, diff_OCA = {diff_OCA:2.2E}')

        assert np.max(np.abs(Sigma_NCA_ref)) > 1e-3, 'vacuous test: the NCA self-energy vanishes'
        assert np.max(np.abs(Sigma_OCA_ref)) > 1e-4, 'vacuous test: the OCA self-energy vanishes'

        # Both evaluators sum the same DLR hybridization lines, so they agree to round-off
        np.testing.assert_allclose(Sigma_NCA, Sigma_NCA_ref, atol=1e-12)
        np.testing.assert_allclose(Sigma_OCA, Sigma_OCA_ref, atol=1e-12)

        if verbose:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 8))
            subp = [3, 1, 1]

            plt.subplot(*subp); subp[-1] += 1

            plt.plot(tau, delta_iaa[:, 0, 0].flatten().real, 'o-')
            plt.ylabel(r'$\Delta(\tau)$')
            plt.xlabel(r'$\tau$')

            plt.subplot(*subp); subp[-1] += 1

            for i in range(dim):
                plt.plot(tau, Sigma_NCA_ref[:, i, i].real, '+-')
                plt.plot(tau, Sigma_NCA[:, i, i].real, 'x-')

            plt.ylabel(r'$\Sigma_{ii}^{(NCA)}(\tau)$')
            plt.xlabel(r'$\tau$')

            plt.plot([], [], '+-', color='gray', label='BlockSparseSolver')
            plt.plot([], [], 'x-', color='gray', label='NCA_dense')
            plt.legend(loc='best')

            plt.subplot(*subp); subp[-1] += 1

            for i in range(dim):
                plt.plot(tau, Sigma_OCA_ref[:, i, i].real, '+-')
                plt.plot(tau, Sigma_OCA[:, i, i].real, 'x-')

            plt.ylabel(r'$\Sigma_{ii}^{(OCA)}(\tau)$')
            plt.xlabel(r'$\tau$')

            plt.plot([], [], '+-', color='gray', label='BlockSparseSolver')
            plt.plot([], [], 'x-', color='gray', label='OCA_dense')
            plt.legend(loc='best')

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":

    test_block_sparsity_NCA_dense(verbose=False)
