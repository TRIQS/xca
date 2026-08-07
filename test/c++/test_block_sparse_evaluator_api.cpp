#include <gtest/gtest.h>

#include <triqs_xca/atom_diag_utils.hpp>

#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs_xca/hyb.hpp>

#include "block_sparse_utils.hpp"

using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;

using triqs_xca::dense::DenseDiagramEvaluator;

using triqs_xca::block_sparse::DiagramEvaluator;

using triqs_xca::atom_diag::ad_to_atom_prop;
using triqs_xca::atom_diag::get_full_h_atomic;

/**
 * @file test_block_sparse_evaluator_api.cpp
 *
 * @brief Tests that the several ways of calling a DiagramEvaluator all describe the same problem
 *
 * @details Diagram evaluators can be constructed in two ways in C++, either from an atom_diag object, which derives the field operators and the 
 * block structure itself and takes the propagator as a block_gf, or from a BlockOpSymQuartet assembled by the caller, which takes the propagator as 
 * a BlockDiagOpFun. Separately, the backbone evaluation loop can be left to the evaluator or driven by the caller one flat index at a time.
 *
 * These are consistency checks between code paths rather than checks of the diagram itself: the routes build the same Hybridization from the same
 * coefficients and sum the same backbones, so they should agree to machine precision.
 *
 * Every test here uses the same model, the two-band Kanamori atom with a two-pole discrete-bath hybridization, and takes its atom_diag from
 * two_band_atom_diag_helper, so the subspaces, and hence the block ordering, are identical throughout.
 */

/**
 * @brief Check that the two DiagramEvaluator constructors give the same self-energy
 *
 * @details The BlockOpSymQuartet constructor is C2PY_IGNORE and predates the atom_diag one, but is still used internally, so the two must agree on
 * the same model.
 */
TEST(EvaluatorAPI, OCA_constructor_equivalence) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_reflect] = discrete_bath_helper(beta, Lambda, eps);
  auto hyb_coeffs               = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};

  // --- atom_diag constructor, taking the atomic propagator as a block_gf ---
  auto ad      = two_band_atom_diag_helper();
  auto G0_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);
  DiagramEvaluator D(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  BlockDiagOpFun OCA_result(D.compute_self_energy(G0_ppsc, topology));

  // --- old constructor, taking field operators as a BlockOpSymQuartet and the propagator as a BlockDiagOpFun ---
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);
  DiagramEvaluator DDE(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  BlockDiagOpFun OCA_result_2(DDE.compute_self_energy(Gt, topology));

  // the two constructors must assemble the same hybridization ...
  ASSERT_LE(nda::max_element(nda::abs(D.hyb.values - DDE.hyb.values)), 1.0e-15);
  ASSERT_EQ(D.hyb.poles, DDE.hyb.poles);

  // ... and hence the same self-energy, block by block
  ASSERT_EQ(OCA_result.get_num_block_cols(), OCA_result_2.get_num_block_cols());
  for (int i = 0; i < OCA_result.get_num_block_cols(); i++) {
    SCOPED_TRACE("block " + std::to_string(i));
    EXPECT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - OCA_result_2.get_block(i))), 1.0e-15);
  }
}

/**
 * @brief Check that summing the backbones one at a time reproduces the built-in self-energy diagram loop
 *
 * @details compute_self_energy has an overload taking a single flat backbone index, which is how a caller drives the diagram loop itself, e.g. to
 * distribute it. Summing that overload over all get_num_self_energy_backbones indices must reproduce the overload that loops internally, so their
 * difference is checked to vanish. Both sides come from the same evaluator, so this is self-consistency of the loop bookkeeping and is independent
 * of the model.
 */
TEST(EvaluatorAPI, manual_loop) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_reflect] = discrete_bath_helper(beta, Lambda, eps);
  auto hyb_coeffs               = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  // Kanamori atom and its atomic propagator as a block_gf
  auto ad      = two_band_atom_diag_helper();
  auto G0_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  DiagramEvaluator D(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto OCA_result = D.compute_self_energy(G0_ppsc, topology); // evaluate self-energy using built-in loop

  for (int f = 0; f < D.get_num_self_energy_backbones(topology); ++f) {
    OCA_result -= D.compute_self_energy(G0_ppsc, topology, f); // subtract off self-energy evaluation using manual loop
  }
  for (int i = 0; i < ad.n_subspaces(); ++i) {
    SCOPED_TRACE("block " + std::to_string(i));
    EXPECT_LE(nda::max_element(nda::abs(OCA_result[i].data())), 1.0e-15);
  }
}

/**
 * @brief Check that the two DiagramEvaluator constructors give the same single-particle Green's function
 *
 * @details The single-particle analogue of OCA_constructor_equivalence above, with a comparison against the dense evaluator appended: unlike the
 * self-energy, whose block-sparse-vs-dense comparison is made in test_block_sparse_sparsity_invariance.cpp, the single-particle Green's function is
 * a dense object in orbital space and is compared directly.
 */
TEST(EvaluatorAPI, gf_constructor_equivalence) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_reflect] = discrete_bath_helper(beta, Lambda, eps);
  auto hyb_coeffs               = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  // Kanamori atom and its atomic propagator as a block_gf
  auto ad      = two_band_atom_diag_helper();
  auto G0_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};

  // --- atom_diag constructor, taking the atomic propagator as a block_gf ---
  DiagramEvaluator D(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto OCA_gf = D.compute_single_ptcle_gf(G0_ppsc, topology);

  // --- old constructor, taking field operators as a BlockOpSymQuartet and the propagator as a BlockDiagOpFun ---
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);
  DiagramEvaluator D2(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto OCA_gf_2 = D2.compute_single_ptcle_gf(Gt, topology);

  // the two constructors must assemble the same hybridization ...
  ASSERT_LE(nda::max_element(nda::abs(D.hyb.values - D2.hyb.values)), 1.0e-15);
  ASSERT_EQ(D.hyb.poles, D2.hyb.poles);

  // ... and hence the same Green's function
  EXPECT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_2)), 1.0e-15);

  // --- dense evaluation, over the full Hilbert space as a single block ---
  auto G0t_dense = Hmat_to_Gtmat(get_full_h_atomic(ad), beta, cppdlr::rel2abs(itops.get_itnodes()));
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G0_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G0_ppsc[0].mesh(), G0t_dense)};
  auto G0_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G0_dense_blocks);
  DenseDiagramEvaluator D_dense(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto OCA_gf_dense = D_dense.compute_single_ptcle_gf(G0_ppsc_dense, topology);
  EXPECT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_dense)), 1.0e-15);
}

/**
 * @brief Check that summing the backbones one at a time reproduces the built-in single-particle Green's function diagram loop
 *
 * @details The single-particle analogue of manual_loop above, summing the flat-index overload of compute_single_ptcle_gf over all
 * get_num_single_ptcle_gf_backbones indices. The manual sum runs on a fresh evaluator, so a diagram loop that left state behind would show up here.
 */
TEST(EvaluatorAPI, gf_manual_loop) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_reflect] = discrete_bath_helper(beta, Lambda, eps);
  auto hyb_coeffs               = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  // Kanamori atom and its atomic propagator as a block_gf
  auto ad      = two_band_atom_diag_helper();
  auto G0_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  DiagramEvaluator D(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto OCA_gf = D.compute_single_ptcle_gf(G0_ppsc, topology); // evaluate Green's function using built-in loop

  // ... and again, driving the backbone loop from the caller, one flat index at a time, on a fresh evaluator
  DiagramEvaluator D2(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto OCA_gf_2 = nda::make_regular(0 * OCA_gf);
  for (int f = 0; f < D.get_num_single_ptcle_gf_backbones(topology); ++f) { OCA_gf_2 += D2.compute_single_ptcle_gf(G0_ppsc, topology, f); }

  EXPECT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_2)), 1.0e-15);
}
