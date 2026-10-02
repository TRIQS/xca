#include "triqs_xca/dense/atom_diag.hpp"

namespace triqs_xca::dense::atom_diag {

  using triqs_xca::atom_diag::get_hamiltonian_block;
  using triqs_xca::atom_diag::get_operator_block;

  namespace {

    template <bool IsComplex>
    nda::matrix<dcomplex> get_full_operator_matrix_impl(const triqs_atom_diag_t<IsComplex> &ad, int oidx, bool is_creation) {
      int dim                      = ad.get_full_hilbert_space_dim();
      nda::matrix<dcomplex> op_mat = nda::zeros<dcomplex>(dim, dim);

      for (int s1 = 0; s1 < ad.n_subspaces(); ++s1) {
        auto block = get_operator_block(ad, oidx, is_creation, s1);
        if (!block) continue;

        long s2 = is_creation ? ad.cdag_connection(oidx, s1) : ad.c_connection(oidx, s1);

        // Get Fock states
        auto f1 = ad.get_fock_states(s1);
        auto f2 = ad.get_fock_states(s2);

        // Copy block into full matrix
        for (int i = 0; i < f2.size(); ++i) {
          for (int j = 0; j < f1.size(); ++j) { op_mat(f2[i], f1[j]) = (*block)(i, j); }
        }
      }

      return op_mat;
    }

    template <bool IsComplex>
    std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> get_operators_dense_impl(const triqs_atom_diag_t<IsComplex> &ad) {

      int norb = ad.get_fops().size();
      int N    = ad.get_full_hilbert_space_dim();

      // Get full operator matrices
      nda::array<dcomplex, 3> Fs{norb, N, N};
      nda::array<dcomplex, 3> Fdags{norb, N, N};

      for (int oidx = 0; oidx < norb; ++oidx) {
        Fs(oidx, cppdlr::_, cppdlr::_)    = get_full_operator_matrix(ad, oidx, false);
        Fdags(oidx, cppdlr::_, cppdlr::_) = get_full_operator_matrix(ad, oidx, true);
      }
      return {Fs, Fdags};
    }

  } // namespace

  std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> get_operators_dense(const triqs_atom_diag_t<true> &ad) {
    return get_operators_dense_impl(ad);
  }

  std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> get_operators_dense(const triqs_atom_diag_t<false> &ad) {
    return get_operators_dense_impl(ad);
  }

  FSet get_operators_dense(const triqs_atom_diag_t<true> &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs) {
    auto [Fs, Fdags] = get_operators_dense_impl(ad);
    return {Fs, Fdags, hyb_coeffs};
  }

  FSet get_operators_dense(const triqs_atom_diag_t<false> &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs) {
    auto [Fs, Fdags] = get_operators_dense_impl(ad);
    return {Fs, Fdags, hyb_coeffs};
  }

  nda::matrix<dcomplex> get_full_h_atomic(const triqs_atom_diag &ad) {
    int dim    = ad.get_full_hilbert_space_dim();
    auto H_mat = nda::zeros<dcomplex>(dim, dim);

    for (int sidx = 0; sidx < ad.n_subspaces(); ++sidx) {
      auto H_block = get_hamiltonian_block(ad, sidx);

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

} // namespace triqs_xca::dense::atom_diag
