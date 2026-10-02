#include "triqs_xca/dense/atom_diag_utils.hpp"

namespace triqs_xca::atom_diag {

  namespace {

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

} // namespace triqs_xca::atom_diag
