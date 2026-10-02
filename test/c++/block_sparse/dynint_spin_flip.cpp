#include <algorithm>
#include <string>

#include <gtest/gtest.h>

#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/block_sparse/atom_diag.hpp>
#include <triqs_xca/block_sparse/block_op.hpp>
#include <triqs_xca/block_sparse/diagram_evaluator.hpp>
#include <triqs_xca/dense/diagram_evaluator.hpp>
#include <triqs_xca/hyb.hpp>
#include <triqs_xca/block_sparse/dynint.hpp>
#include <triqs_xca/topology.hpp>

#include "test_utils/block_sparse.hpp"
#include "test_utils/dense_comparison.hpp"
#include "test_utils/parallel_atom_diag_check.hpp"

using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;
using nda::dcomplex;

using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::many_body_operator_real;

using triqs_xca::block_sparse::atom_diag::get_operators;
using triqs_xca::block_sparse::BlockDiagOpFun;
using triqs_xca::block_sparse::BlockOpSymQuartet;
using triqs_xca::block_sparse::DiagramEvaluator;
namespace dense = triqs_xca::dense;
using triqs_xca::hyb::get_extended_coefficients;
using triqs_xca::block_sparse::dynint::get_operators_and_interactions;
namespace test_utils = triqs_xca::test_utils;

/**
 * @file block_sparse/dynint_spin_flip.cpp
 *
 * @brief Dense against block-sparse with the non-block-diagonal interaction operators S+ and S-
 *
 * @details Density operators are block diagonal and hermitian, while S+ and S- map (N_up, N_do) -> (N_up +- 1, N_do -+ 1), so on the
 * block-sparse side they exercise the connection-row and inverse-map machinery of resolve_dynint_op(), with F_dags(n_hyb + i) != Fs(n_hyb + i).
 * Both evaluators use the same convention for the meaning of slot (i, j) of dynint_coeffs, so this file only checks that the block-sparse
 * path reproduces the dense one, the physical convention is checked against ED in test/python/dynint_transverse.py.
 */

namespace {

  constexpr double beta = 2.0, Lambda = 40.0, eps = 1.0e-10;
  constexpr int p_poles = 2;

  // tolerance for the comparison with the dense reference
  constexpr double sigma_tol = 1.0e-13;
  constexpr double corr_tol  = 1.0e-13;

  struct SpinFlipModel {
    triqs::atom_diag::atom_diag<true> ad;      // sym_ops = {N_up, N_do}: the block-sparse side
    triqs::atom_diag::atom_diag<true> ad_flat; // sym_ops = {}: the dense side
    nda::vector<double> hyb_poles;
    nda::array<dcomplex, 3> hyb_coeffs;
    nda::array<dcomplex, 3> dynint_coeffs;
    std::vector<many_body_operator_real> dynint_ops;
  };

  many_body_operator_real S_minus(int norb) {
    many_body_operator_real op;
    for (int i = 0; i < norb; ++i) op += c_dag("do", i) * c("up", i);
    return op;
  }

  many_body_operator_real S_plus(int norb) {
    many_body_operator_real op;
    for (int i = 0; i < norb; ++i) op += c_dag("up", i) * c("do", i);
    return op;
  }

  /**
   * @brief sz_resolved_atom_diag_helper and its single-subspace twin, with dynint_ops = [S-, S+]
   *
   * @details On this fixture S+ and S- are single-target operators, since N_up and N_do are separately conserved. The diagonal dynint_coeffs
   * have different weights per slot and per pole, so that swapping the two interaction flavours or the two poles changes the answer, and
   * the off-diagonal is zero since S- and S+ land in different symmetry sets.
   */
  SpinFlipModel spin_flip_model(int norb) {
    auto ad      = test_utils::sz_resolved_atom_diag_helper(norb, true);
    auto ad_flat = test_utils::sz_resolved_atom_diag_helper(norb, false);

    int n_hyb     = static_cast<int>(ad.get_fops().size());
    auto labels_f = std::get<1>(get_operators(ad, nda::zeros<dcomplex>(p_poles, n_hyb, n_hyb)));
    auto hyb      = test_utils::sym_set_diagonal_hyb(labels_f, p_poles);

    std::vector<many_body_operator_real> ops{S_minus(norb), S_plus(norb)};

    auto d = nda::zeros<dcomplex>(p_poles, 2, 2);
    for (int l = 0; l < p_poles; ++l) {
      d(l, 0, 0) = 0.61 - 0.07 * l;
      d(l, 1, 1) = 0.43 + 0.11 * l;
    }

    nda::vector<double> hyb_poles = {1.3, -0.8};
    return {ad, ad_flat, hyb_poles, hyb, d, ops};
  }

} // namespace

/**
 * @brief Check that S+ and S- land in two distinct symmetry sets of their own and construct an evaluator with the expected n / n_hyb / n_int
 */
TEST(BlockSparseDynintSpinFlip, spin_flip_ops_construct_a_valid_evaluator) {

  for (int norb : {1, 2}) {
    SCOPED_TRACE("norb = " + std::to_string(norb));
    auto m    = spin_flip_model(norb);
    int n_hyb = static_cast<int>(m.ad.get_fops().size());
    ASSERT_EQ(n_hyb, 2 * norb);

    auto [Fq, labels] = get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops);
    auto ext          = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);

    // S- and S+ have different connection rows and must land in two distinct sets of their own
    ASSERT_EQ(labels.size(), n_hyb + 2);
    EXPECT_NE(labels(n_hyb), labels(n_hyb + 1)) << "S- and S+ share a symmetry set, which would make the cross-set guard blind";
    for (int i = 0; i < n_hyb; ++i) {
      EXPECT_NE(labels(i), labels(n_hyb)) << "fermionic flavour " << i << " merged with the S- set";
      EXPECT_NE(labels(i), labels(n_hyb + 1)) << "fermionic flavour " << i << " merged with the S+ set";
    }

    DiagramEvaluator D(beta, Lambda, eps, m.hyb_poles, ext, Fq, /*n_int=*/2);
    EXPECT_EQ(D.n_hyb, n_hyb);
    EXPECT_EQ(D.n_int, 2);
    EXPECT_EQ(D.n, n_hyb + 2);
  }
}

/**
 * @brief Compare the self-energy to the dense evaluator on the crossing order-2 topology, with norb = 2 so that the connected subspaces have
 * dimension > 1
 */
TEST(BlockSparseDynintSpinFlip, self_energy_matches_dense_with_spin_flip_interactions) {

  auto m = spin_flip_model(/*norb=*/2);
  ASSERT_NO_FATAL_FAILURE(test_utils::assert_parallel_atom_diags(m.ad, m.ad_flat));

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  ASSERT_EQ(triqs_xca::topology::topology_parity(topology), -1) << "vacuous test: topology is not crossing";

  int n_int = static_cast<int>(m.dynint_ops.size());

  // check that some subspace has dimension > 1, otherwise the non-block-diagonal blocks are all 1x1
  auto dims = m.ad.get_subspace_dims();
  ASSERT_GT(*std::max_element(dims.begin(), dims.end()), 1);

  auto itops = imtime_ops(Lambda, build_dlr_rf(Lambda, eps));
  auto Gt    = test_utils::ad_to_atom_prop(m.ad, beta, itops);

  auto Fq  = std::get<0>(get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops));
  auto ext = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);
  DiagramEvaluator D(beta, Lambda, eps, m.hyb_poles, ext, Fq, n_int);

  // check that the interaction flavours enter the backbone enumeration
  auto Fq_ferm = std::get<0>(get_operators(m.ad, m.hyb_coeffs));
  DiagramEvaluator D_ferm(beta, Lambda, eps, m.hyb_poles, m.hyb_coeffs, Fq_ferm);
  ASSERT_GT(D.get_num_self_energy_backbones(topology), D_ferm.get_num_self_energy_backbones(topology)) << "vacuous test: n_ext == n_hyb";

  auto Sigma = BlockDiagOpFun(D.compute_self_energy(Gt, topology));

  auto G_flat = test_utils::ad_to_atom_prop(m.ad_flat, beta, Lambda, eps);
  dense::DiagramEvaluator D_dense(m.hyb_poles, m.hyb_coeffs, G_flat[0].mesh(), m.ad_flat, m.dynint_ops, m.dynint_coeffs);
  ASSERT_EQ(D_dense.n_int, n_int);
  auto Sigma_dense = D_dense.compute_self_energy(G_flat, topology);

  auto [err, scale] = test_utils::compare_sigma_with_dense(Sigma, Sigma_dense, m.ad);
  ASSERT_GT(scale, 0.01) << "vacuous test: the self-energy is too small for an absolute tolerance";
  EXPECT_LE(err, sigma_tol) << "max|Sigma_dense| = " << scale << ", max|bs - dense| = " << err;

  // check that the interaction changes the self-energy, otherwise both sides are the purely fermionic answer
  auto Sigma_ferm         = BlockDiagOpFun(D_ferm.compute_self_energy(Gt, topology));
  double dynint_influence = 0.0;
  for (int b = 0; b < Sigma.get_num_block_cols(); ++b)
    dynint_influence = std::max(dynint_influence, nda::max_element(nda::abs(Sigma.get_block(b) - Sigma_ferm.get_block(b))));
  ASSERT_GT(dynint_influence, 1.0e-6) << "vacuous test: the S+/S- interaction does not change the self-energy at all";
}

/**
 * @brief Compare the single-particle Green's function to the dense evaluator on the leading n_hyb x n_hyb window
 */
TEST(BlockSparseDynintSpinFlip, spgf_matches_dense_with_spin_flip_interactions) {

  auto m = spin_flip_model(/*norb=*/2);
  ASSERT_NO_FATAL_FAILURE(test_utils::assert_parallel_atom_diags(m.ad, m.ad_flat));

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  int n_hyb                   = static_cast<int>(m.ad.get_fops().size());
  int n_int                   = static_cast<int>(m.dynint_ops.size());

  auto G_bs = test_utils::ad_to_atom_prop(m.ad, beta, Lambda, eps);
  auto G_fl = test_utils::ad_to_atom_prop(m.ad_flat, beta, Lambda, eps);

  auto Fq  = std::get<0>(get_operators_and_interactions(m.ad, m.hyb_coeffs, m.dynint_coeffs, m.dynint_ops));
  auto ext = get_extended_coefficients(m.hyb_coeffs, m.dynint_coeffs);
  DiagramEvaluator D(beta, Lambda, eps, m.hyb_poles, ext, Fq, n_int);
  dense::DiagramEvaluator D_dense(m.hyb_poles, m.hyb_coeffs, G_fl[0].mesh(), m.ad_flat, m.dynint_ops, m.dynint_coeffs);

  auto Fq_ferm = std::get<0>(get_operators(m.ad, m.hyb_coeffs));
  DiagramEvaluator D_ferm(beta, Lambda, eps, m.hyb_poles, m.hyb_coeffs, Fq_ferm);
  ASSERT_GT(D.get_num_single_ptcle_gf_backbones(topology), D_ferm.get_num_single_ptcle_gf_backbones(topology)) << "vacuous test: n_ext == n_hyb";

  auto spgf_dense = D_dense.compute_single_ptcle_gf(G_fl, topology);
  ASSERT_GE(spgf_dense.extent(1), n_hyb);

  auto [err, scale] = test_utils::compare_leading_block(D.compute_single_ptcle_gf(G_bs, topology), spgf_dense, n_hyb);
  ASSERT_GT(scale, 1.0e-3) << "vacuous test: the single-particle Green's function is too small";
  EXPECT_LE(err, corr_tol) << "max|dense| = " << scale << ", max|bs - dense| = " << err;

  // check that the interaction changes the spgf
  auto spgf_ferm        = D_ferm.compute_single_ptcle_gf(G_bs, topology);
  auto [d_err, d_scale] = test_utils::compare_leading_block(D.compute_single_ptcle_gf(G_bs, topology), spgf_ferm, n_hyb);
  ASSERT_GT(d_err, 1.0e-8) << "vacuous test: the S+/S- interaction does not change the spgf at all";
}

/**
 * @brief Check that any off-diagonal entry of dynint_coeffs between S- and S+, symmetric or not, is rejected as a cross-set coefficient
 */
TEST(BlockSparseDynintSpinFlip, rejects_off_diagonal_interaction_coefficients_between_S_minus_and_S_plus) {

  auto m = spin_flip_model(/*norb=*/2);

  auto try_build = [&](nda::array<dcomplex, 3> const &d) {
    try {
      auto q = get_operators_and_interactions(m.ad, m.hyb_coeffs, d, m.dynint_ops);
      return std::string{};
    } catch (std::invalid_argument const &e) { return std::string{e.what()}; }
  };

  // the diagonal fixture must be accepted
  EXPECT_EQ(try_build(m.dynint_coeffs), std::string{}) << "the diagonal two-entry encoding must be accepted";

  for (auto const &[name, d01, d10] :
       std::vector<std::tuple<std::string, dcomplex, dcomplex>>{{"asymmetric off-diagonal", 0.31, 0.17}, {"symmetric off-diagonal", 0.31, 0.31}}) {
    SCOPED_TRACE(name);
    auto d     = nda::array<dcomplex, 3>{m.dynint_coeffs};
    d(0, 0, 1) = d01;
    d(0, 1, 0) = d10;
    auto msg   = try_build(d);
    EXPECT_NE(msg.find("couples different symmetry sets"), std::string::npos) << "message was: " << msg;
  }
}
