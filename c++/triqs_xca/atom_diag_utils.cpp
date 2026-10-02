#include <stdexcept>

#include "triqs_xca/atom_diag_utils.hpp"

namespace triqs_xca::atom_diag {

  namespace {

    template <bool IsComplex>
    nda::matrix<dcomplex> get_full_operator_matrix_impl(const triqs_atom_diag_t<IsComplex> &ad, int oidx, bool is_creation) {
      int dim                      = ad.get_full_hilbert_space_dim();
      nda::matrix<dcomplex> op_mat = nda::zeros<dcomplex>(dim, dim);

      for (int s1 = 0; s1 < ad.n_subspaces(); ++s1) {
        // Get connection to target subspace
        long s2 = is_creation ? ad.cdag_connection(oidx, s1) : ad.c_connection(oidx, s1);
        if (s2 < 0) continue;

        // Get matrix block in eigenbasis
        auto block_mat = is_creation ? ad.cdag_matrix(oidx, s1) : ad.c_matrix(oidx, s1);

        // Transform to Fock basis
        auto U1                              = ad.get_unitary_matrix(s1);
        auto U2                              = ad.get_unitary_matrix(s2);
        nda::matrix<dcomplex> block_mat_fock = U2 * block_mat * nda::conj(nda::transpose(U1));

        // Get Fock states
        auto f1 = ad.get_fock_states(s1);
        auto f2 = ad.get_fock_states(s2);

        // Copy block into full matrix
        for (int i = 0; i < f2.size(); ++i) {
          for (int j = 0; j < f1.size(); ++j) { op_mat(f2[i], f1[j]) = block_mat_fock(i, j); }
        }
      }

      return op_mat;
    }

  } // namespace

  using nda::linalg::matmul;

  using cppdlr::_;

  nda::matrix<dcomplex> get_full_h_atomic(const triqs_atom_diag &ad) {
    int dim    = ad.get_full_hilbert_space_dim();
    auto H_mat = nda::zeros<dcomplex>(dim, dim);

    for (int sidx = 0; sidx < ad.n_subspaces(); ++sidx) {
      auto U        = ad.get_unitary_matrix(sidx);
      auto energies = ad.get_energies()[sidx];

      // Create diagonal energy matrix
      nda::matrix<dcomplex> H_block_diag = nda::zeros<dcomplex>(energies.size(), energies.size());
      for (int i = 0; i < energies.size(); ++i) { H_block_diag(i, i) = energies[i] + ad.get_gs_energy(); }

      // Transform to Fock basis: H_block = U @ H_block_diag @ U^dagger
      nda::matrix<dcomplex> H_block = U * H_block_diag * nda::conj(nda::transpose(U));

      // Get Fock states for this subspace
      auto fock_states = ad.get_fock_states(sidx);

      // Copy block into full matrix
      for (int i = 0; i < fock_states.size(); ++i) {
        for (int j = 0; j < fock_states.size(); ++j) { H_mat(fock_states[i], fock_states[j]) = H_block(i, j); }
      }
    }

    return H_mat;
  }

  nda::matrix<dcomplex> get_full_operator_matrix(const triqs_atom_diag_t<true> &ad, int oidx, bool is_creation) {
    return get_full_operator_matrix_impl(ad, oidx, is_creation);
  }

  nda::matrix<dcomplex> get_full_operator_matrix(const triqs_atom_diag_t<false> &ad, int oidx, bool is_creation) {
    return get_full_operator_matrix_impl(ad, oidx, is_creation);
  }

  nda::array<dcomplex, 3> get_tensor_in_atom_diag_subspace(nda::array_const_view<dcomplex, 3> tensor_full, int subspace_index,
                                                           triqs_atom_diag const &ad) {
    // Permute a tensor from the full Hilbert space to the Fock state ordered basis

    int r = tensor_full.extent(0);

    std::vector<unsigned long> H_perm;
    auto fock_states = ad.get_fock_states(subspace_index);
    for (auto state : fock_states) H_perm.push_back(state);

    int N_sub                               = fock_states.size();
    nda::array<dcomplex, 3> tensor_subspace = nda::zeros<dcomplex>(r, N_sub, N_sub);

    for (int t = 0; t < r; ++t) {
      for (int i = 0; i < N_sub; ++i) {
        for (int j = 0; j < N_sub; ++j) { tensor_subspace(t, i, j) = tensor_full(t, H_perm[i], H_perm[j]); }
      }
    }

    return tensor_subspace;
  }

} // namespace triqs_xca::atom_diag
