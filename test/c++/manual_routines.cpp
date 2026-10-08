#include <string>

#include <gtest/gtest.h>

#include <nda/nda.hpp>

#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/block_sparse/manual/sigma.hpp>
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/dense/manual/sigma.hpp>
#include <triqs_xca/dense/manual/spgf.hpp>
#include <triqs_xca/hyb.hpp>
#include <triqs_xca/topology.hpp>

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
 * C-contiguous has to give the same result as a contiguous copy. The trapezoidal routines are checked against the evaluator. The model has
 * two orbitals and a non-symmetric complex hybridization, so that the matrix structure of the hybridization is visible.
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

  // two fermions with a non-symmetric complex hybridization, in the layouts of both manual families
  struct Model {
    test_utils::FermionModelData base = test_utils::two_fermion_model_helper(beta, Lambda, eps);
    cppdlr::imtime_ops itops          = base.G_ppsc[0].mesh().dlr_it();
    nda::vector<double> hyb_poles     = {1.3, -0.8};
    nda::array<dcomplex, 3> hyb_coeffs;
    nda::array<dcomplex, 3> hyb, Gt, Fs, F_dags;

    Model() : hyb_coeffs(2, 2, 2) {
      for (int l = 0; l < 2; ++l)
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 2; ++j) hyb_coeffs(l, i, j) = dcomplex(0.5 - 0.2 * l + 0.15 * (i - j), 0.1 * (i + 2 * j));
      hyb                  = triqs_xca::hyb::coefs2vals(beta, itops, hyb_coeffs, hyb_poles);
      Gt                   = test_utils::get_tensor_in_full_hilbert_space(base.G_ppsc, base.ad);
      std::tie(Fs, F_dags) = dense::atom_diag::get_operators(base.ad);
    }
  };

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
  Model md;
  auto &m          = md.base;
  auto &itops      = md.itops;
  auto &hyb_poles  = md.hyb_poles;
  auto &hyb_coeffs = md.hyb_coeffs;
  auto &hyb        = md.hyb;
  auto &Gt         = md.Gt;
  auto &Fs         = md.Fs;
  auto &F_dags     = md.F_dags;
  int n            = 2;

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

/**
 * @brief Check the trapezoidal routines against the evaluator with a non-symmetric two-orbital hybridization
 *
 * @details A hybridization line enters as hyb(tau_later - tau_earlier, index at the later vertex, index at the earlier vertex), and a
 * backward line as the transpose of the reflected hybridization, which only a non-symmetric hybridization tells apart. The trapezoidal
 * results for n_quad and 2 n_quad are Richardson-extrapolated on the coarse grid, which removes the O(n_quad^-2) error, so that the
 * tolerances are well below the error of a wrong index convention.
 */
TEST(ManualRoutines, trapezoidal_routines_match_the_evaluator) {
  Model m;

  auto mesh         = m.base.G_ppsc[0].mesh();
  auto G_dense      = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>{std::vector{triqs::gfs::gf<triqs::mesh::dlr_imtime>(mesh, m.Gt)}};
  auto D            = dense::DiagramEvaluator(m.hyb_poles, m.hyb_coeffs, mesh, m.base.ad);
  auto hyb_dlr      = m.itops.vals2coefs(m.hyb);
  auto hyb_refl_dlr = m.itops.vals2coefs(m.itops.reflect(m.hyb));
  auto Gt_dlr       = m.itops.vals2coefs(m.Gt);

  // tpz(n_q) is a trapezoidal routine on n_q intervals; the spgf routines leave the end points of the grid at zero
  auto check = [&](std::string const &what, auto const &tpz, nda::array<int, 2> const &topology, nda::array<dcomplex, 3> ref, int n_q,
                   bool skip_end_points, double rel_tol) {
    SCOPED_TRACE(what);
    auto coarse = tpz(n_q), fine = tpz(2 * n_q);
    auto extrapolated = nda::make_regular((4.0 * fine(range(0, 2 * n_q + 1, 2), range::all, range::all) - coarse) / 3.0);
    ref *= triqs_xca::topology::topology_parity(topology);
    auto ref_eq  = test_utils::eval_eq(m.itops, ref, n_q);
    auto points  = skip_end_points ? range(1, n_q) : range(0, n_q + 1);
    double scale = nda::max_element(nda::abs(ref_eq(points, range::all, range::all)));
    ASSERT_GT(scale, 1.0e-3) << "vacuous test: the reference vanishes";
    EXPECT_LE(nda::max_element(nda::abs(extrapolated(points, range::all, range::all) - ref_eq(points, range::all, range::all))), rel_tol * scale);
  };

  nda::array<int, 2> oca = {{0, 2}, {1, 3}}, o3 = {{0, 3}, {1, 4}, {2, 5}};
  check(
     "sigma_oca_tpz", [&](int n_q) { return dense::manual::sigma_oca_tpz(m.hyb, m.itops, beta, m.Gt, m.Fs, n_q); }, oca,
     D.compute_self_energy(G_dense, oca)[0].data(), 8, false, 2.0e-3);
  check(
     "spgf_oca_tpz", [&](int n_q) { return dense::manual::spgf_oca_tpz(hyb_dlr, hyb_refl_dlr, m.itops, beta, Gt_dlr, m.Fs, n_q); }, oca,
     D.compute_single_ptcle_gf(G_dense, oca), 8, true, 2.0e-3);
  check(
     "sigma_o3_tpz", [&](int n_q) { return dense::manual::sigma_o3_tpz(m.hyb, m.itops, beta, m.Gt, m.Fs, n_q); }, o3,
     D.compute_self_energy(G_dense, o3)[0].data(), 4, false, 0.06);
  check(
     "spgf_o3_tpz", [&](int n_q) { return dense::manual::spgf_o3_tpz(hyb_dlr, hyb_refl_dlr, m.itops, beta, Gt_dlr, m.Fs, n_q); }, o3,
     D.compute_single_ptcle_gf(G_dense, o3), 4, true, 0.06);
}
