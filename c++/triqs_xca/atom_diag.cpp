#include <stdexcept>
#include <string>

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

    template <bool IsComplex>
    op_block_mat get_op_mat_impl(const triqs_atom_diag_t<IsComplex> &ad, triqs::operators::many_body_operator_complex const &op) {
      int nb = ad.n_subspaces();
      op_block_mat op_mat{nda::array<long, 1>(nb), std::vector<nda::matrix<dcomplex>>(nb)};
      op_mat.connection() = -1;

      for (int b = 0; b < nb; ++b) {
        for (auto const &term : op) {
          auto [bb, mat] = ad.get_matrix_element_of_monomial(term.monomial, b);
          if (bb == -1) continue;

          nda::matrix<dcomplex> contribution = term.coef * nda::matrix<dcomplex>{mat};
          if (op_mat.connection(b) == -1) {
            op_mat.connection(b) = bb;
            op_mat.block_mat[b]  = contribution;
          } else if (op_mat.connection(b) != bb) {
            throw std::runtime_error("get_op_mat: the monomials of the operator take subspace " + std::to_string(b) + " to different subspaces, "
                                     + std::to_string(op_mat.connection(b)) + " and " + std::to_string(bb));
          } else {
            op_mat.block_mat[b] += contribution;
          }
        }
      }
      return op_mat;
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

  op_block_mat get_op_mat(const triqs_atom_diag_t<true> &ad, triqs::operators::many_body_operator_complex const &op) {
    return get_op_mat_impl(ad, op);
  }

  op_block_mat get_op_mat(const triqs_atom_diag_t<false> &ad, triqs::operators::many_body_operator_complex const &op) {
    return get_op_mat_impl(ad, op);
  }

} // namespace triqs_xca::atom_diag
