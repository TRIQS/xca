#include "triqs_xca/atom_diag.hpp"

namespace triqs_xca::atom_diag {

  namespace {

    template <bool IsComplex>
    std::optional<nda::matrix<dcomplex>> get_operator_block_impl(const triqs_atom_diag_t<IsComplex> &ad, int oidx, bool is_creation, int s1) {
      // Get connection to target subspace
      long s2 = is_creation ? ad.cdag_connection(oidx, s1) : ad.c_connection(oidx, s1);
      if (s2 < 0) return std::nullopt;

      // Get matrix block in eigenbasis
      auto block_mat = is_creation ? ad.cdag_matrix(oidx, s1) : ad.c_matrix(oidx, s1);

      // Transform to Fock basis
      auto U1                              = ad.get_unitary_matrix(s1);
      auto U2                              = ad.get_unitary_matrix(s2);
      nda::matrix<dcomplex> block_mat_fock = U2 * block_mat * nda::conj(nda::transpose(U1));

      return block_mat_fock;
    }

  } // namespace

  std::optional<nda::matrix<dcomplex>> get_operator_block(const triqs_atom_diag_t<true> &ad, int oidx, bool is_creation, int s) {
    return get_operator_block_impl(ad, oidx, is_creation, s);
  }

  std::optional<nda::matrix<dcomplex>> get_operator_block(const triqs_atom_diag_t<false> &ad, int oidx, bool is_creation, int s) {
    return get_operator_block_impl(ad, oidx, is_creation, s);
  }

  nda::matrix<dcomplex> get_hamiltonian_block(const triqs_atom_diag &ad, int s) {
    auto U        = ad.get_unitary_matrix(s);
    auto energies = ad.get_energies()[s];

    // Create diagonal energy matrix
    nda::matrix<dcomplex> H_block_diag = nda::zeros<dcomplex>(energies.size(), energies.size());
    for (int i = 0; i < energies.size(); ++i) { H_block_diag(i, i) = energies[i] + ad.get_gs_energy(); }

    // Transform to Fock basis: H_block = U @ H_block_diag @ U^dagger
    nda::matrix<dcomplex> H_block = U * H_block_diag * nda::conj(nda::transpose(U));

    return H_block;
  }

} // namespace triqs_xca::atom_diag
