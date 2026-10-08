#include <string>

#include <gtest/gtest.h>

#include <nda/nda.hpp>

#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/block_sparse/manual/sigma.hpp>
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/dense/manual/sigma.hpp>
#include <triqs_xca/dense/manual/spgf.hpp>
#include <triqs_xca/hyb.hpp>

#include "test_utils/block_sparse.hpp"
#include "test_utils/dense.hpp"

using nda::dcomplex;
using nda::range;

namespace block_sparse = triqs_xca::block_sparse;
namespace dense        = triqs_xca::dense;
namespace test_utils   = triqs_xca::test_utils;

/**
 * @file manual_routines.cpp
 *
 * @brief Tests of the manual reference routines in dense::manual and block_sparse::manual
 *
 * @details The routines take generic nda views, and some pass them to cppdlr routines that reshape their argument, so a view that is not
 * C-contiguous has to give the same result as a contiguous copy. The model has two orbitals and a non-symmetric complex hybridization, so
 * that the matrix structure of the hybridization is visible.
 */

namespace {

  constexpr double beta = 2.0, Lambda = 40.0, eps = 1.0e-10;
  constexpr int n_quad = 6;

  // a copy of a in every other column of a wider array, with filler values in the skipped columns
  nda::array<dcomplex, 3> spread(nda::array_const_view<dcomplex, 3> a) {
    nda::array<dcomplex, 3> wide(a.extent(0), a.extent(1), 2 * a.extent(2));
    wide                                                       = dcomplex(99.0, -7.0);
    wide(range::all, range::all, range(0, 2 * a.extent(2), 2)) = a;
    return wide;
  }

  // a copy of a with the time index reversed, which reversed() turns back into a view of the values of a with a negative stride
  nda::array<dcomplex, 3> reversed_copy(nda::array_const_view<dcomplex, 3> a) { return a(range(a.extent(0) - 1, -1, -1), range::all, range::all); }
  nda::array_const_view<dcomplex, 3> reversed(nda::array<dcomplex, 3> const &rev) {
    return rev(range(rev.extent(0) - 1, -1, -1), range::all, range::all);
  }

  // the values spread by spread(), as a view with a non-unit last stride
  nda::array_const_view<dcomplex, 3> every_other(nda::array<dcomplex, 3> const &wide) {
    return wide(range::all, range::all, range(0, wide.extent(2), 2));
  }

  void expect_same(nda::array_const_view<dcomplex, 3> from_view, nda::array_const_view<dcomplex, 3> from_copy, std::string const &what) {
    SCOPED_TRACE(what);
    ASSERT_GT(nda::max_element(nda::abs(from_copy)), 1.0e-3) << "vacuous test: the result vanishes";
    EXPECT_EQ(nda::max_element(nda::abs(from_view - from_copy)), 0.0);
  }

} // namespace

/**
 * @brief Check that the manual routines that pass a caller view to a reshaping cppdlr routine give the same result for views of those inputs
 * with a skipped column or a reversed time index as for contiguous copies
 */
TEST(ManualRoutines, strided_inputs) {
  auto m     = test_utils::two_fermion_model_helper(beta, Lambda, eps);
  auto itops = m.G_ppsc[0].mesh().dlr_it();
  int p = 2, n = 2;

  nda::vector<double> hyb_poles = {1.3, -0.8};
  nda::array<dcomplex, 3> hyb_coeffs(p, n, n);
  for (int l = 0; l < p; ++l)
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) hyb_coeffs(l, i, j) = dcomplex(0.5 - 0.2 * l + 0.15 * (i - j), 0.1 * (i + 2 * j));

  auto hyb                    = triqs_xca::hyb::coefs2vals(beta, itops, hyb_coeffs, hyb_poles);
  auto Gt                     = test_utils::get_tensor_in_full_hilbert_space(m.G_ppsc, m.ad);
  auto [Fs, F_dags]           = dense::atom_diag::get_operators(m.ad);
  auto Fq                     = std::get<0>(block_sparse::atom_diag::get_operators(m.ad, hyb_coeffs));
  auto Fs_bs                  = test_utils::make_correlator_ops(Fq, n).first;
  auto hyb_dlr                = itops.vals2coefs(hyb);
  auto hyb_refl_dlr           = itops.vals2coefs(itops.reflect(hyb));
  auto Gt_dlr                 = itops.vals2coefs(Gt);
  auto [hyb_w, hyb_dlr_w]     = std::pair{spread(hyb), spread(hyb_dlr)};
  auto [hyb_refl_dlr_w, Gt_w] = std::pair{spread(hyb_refl_dlr), spread(Gt_dlr)};
  auto [hyb_r, hyb_dlr_r]     = std::pair{reversed_copy(hyb), reversed_copy(hyb_dlr)};
  auto [hyb_refl_dlr_r, Gt_r] = std::pair{reversed_copy(hyb_refl_dlr), reversed_copy(Gt_dlr)};
  ASSERT_FALSE(every_other(hyb_w).indexmap().is_contiguous());
  ASSERT_LT(reversed(hyb_r).indexmap().strides()[0], 0);

  auto full = [&](block_sparse::BlockDiagOpFun const &S) { return test_utils::get_tensor_in_full_hilbert_space(S, m.ad); };
  for (bool skip_column : {true, false}) {
    auto v = [&](nda::array<dcomplex, 3> const &w, nda::array<dcomplex, 3> const &rev) { return skip_column ? every_other(w) : reversed(rev); };
    std::string kind = skip_column ? "skipped column, " : "reversed time, ";
    expect_same(dense::manual::sigma_oca(v(hyb_w, hyb_r), itops, beta, Gt, Fs, F_dags), dense::manual::sigma_oca(hyb, itops, beta, Gt, Fs, F_dags),
                kind + "dense sigma_oca");
    expect_same(dense::manual::sigma_oca_tpz(v(hyb_w, hyb_r), itops, beta, Gt, Fs, n_quad),
                dense::manual::sigma_oca_tpz(hyb, itops, beta, Gt, Fs, n_quad), kind + "dense sigma_oca_tpz");
    expect_same(dense::manual::sigma_o3_tpz(v(hyb_w, hyb_r), itops, beta, Gt, Fs, n_quad),
                dense::manual::sigma_o3_tpz(hyb, itops, beta, Gt, Fs, n_quad), kind + "dense sigma_o3_tpz");
    expect_same(dense::manual::spgf_oca_tpz(v(hyb_dlr_w, hyb_dlr_r), v(hyb_refl_dlr_w, hyb_refl_dlr_r), itops, beta, v(Gt_w, Gt_r), Fs, n_quad),
                dense::manual::spgf_oca_tpz(hyb_dlr, hyb_refl_dlr, itops, beta, Gt_dlr, Fs, n_quad), kind + "dense spgf_oca_tpz");
    expect_same(dense::manual::spgf_o3_tpz(v(hyb_dlr_w, hyb_dlr_r), v(hyb_refl_dlr_w, hyb_refl_dlr_r), itops, beta, v(Gt_w, Gt_r), Fs, n_quad),
                dense::manual::spgf_o3_tpz(hyb_dlr, hyb_refl_dlr, itops, beta, Gt_dlr, Fs, n_quad), kind + "dense spgf_o3_tpz");
    expect_same(full(block_sparse::manual::sigma_oca(v(hyb_w, hyb_r), itops, beta, m.G_bdof, Fs_bs)),
                full(block_sparse::manual::sigma_oca(hyb, itops, beta, m.G_bdof, Fs_bs)), kind + "block-sparse sigma_oca");
    expect_same(full(block_sparse::manual::sigma_oca(v(hyb_w, hyb_r), hyb_poles, itops, beta, m.G_bdof, Fq)),
                full(block_sparse::manual::sigma_oca(hyb, hyb_poles, itops, beta, m.G_bdof, Fq)), kind + "block-sparse sigma_oca with poles");
  }
}
