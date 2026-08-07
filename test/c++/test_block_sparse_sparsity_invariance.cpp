#include <gtest/gtest.h>

#include <triqs_xca/atom_diag_utils.hpp>

#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs_xca/hyb.hpp>

#include "block_sparse_utils.hpp"

using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;

using triqs_xca::dense::DenseDiagramEvaluator;
using triqs_xca::dense::DenseFSet;

using triqs_xca::block_sparse::DiagramEvaluator;

using triqs_xca::atom_diag::ad_to_atom_prop;
using triqs_xca::atom_diag::get_full_h_atomic;
using triqs_xca::atom_diag::get_operators;
using triqs_xca::atom_diag::get_operators_dense;
using triqs_xca::atom_diag::get_tensor_in_atom_diag_subspace;

/**
 * @file test_block_sparse_sparsity_invariance.cpp
 *
 * @brief Tests that diagram evaluation does not depend on how the Hilbert space is partitioned
 *
 * @details The block-sparse evaluator should produce consistent results regardless of what symmetries are used. In particular, we cover the 
 * following structures:
 *
 * - using every available symmetry using the autopartitioning algorithm of atom_diag; 
 * - labeling the atom_diag subspaces by the total particle number alone rather than by everything autopartitioning finds; and 
 * - discarding the block structure entirely, i.e. a single block spanning the whole Hilbert space and a single symmetry set holding every flavor,
 *   which reduces the block-sparse evaluator to a dense one.
 *
 * Each is checked for both the self-energy and the single-particle correlator, and in each case the dense evaluator, which sums the same backbones
 * over the full Hilbert space, serves as the reference.
 */

/**
 * @brief Check that the block-sparse OCA self-energy is unchanged when the block structure is discarded
 *
 * @details The block-sparse evaluator is run twice on the same model: once with the block structure of the atom_diag subspaces, and once with a
 * trivial sparsity pattern, i.e. a single block spanning the whole Hilbert space and a single symmetry set holding all four flavors. The two must
 * agree, and both must agree with the dense evaluator. The model is the two-band Kanamori atom and a two-pole discrete-bath hybridization.
 */
TEST(SparsityInvariance, OCA_trivial_sparsity) {
  // DLR generation
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-6;
  auto dlr_rf   = build_dlr_rf(Lambda, eps);
  auto itops    = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_refl] = discrete_bath_helper(beta, Lambda, eps);
  auto hyb_coeffs            = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  int n                       = 4; // number of flavors
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, n);

  // --- block-sparse evaluation, using the block structure of the atom_diag subspaces ---
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto OCA_result = BlockDiagOpFun(D.compute_self_energy(Gt, topology));

  // --- dense evaluation ---
  // The old dense constructor divides the poles it is given by beta internally, so it takes dlr_rf where the block-sparse one takes dlr_rf / beta.
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);
  auto Fset                               = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs);
  DenseDiagramEvaluator DDE(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  DDE.eval_self_energy(Gt_dense, B);
  auto OCA_dense_result = DDE.Sigma;

  // --- block-sparse evaluation with trivial sparsity: one block, one symmetry set ---
  auto [Gt_triv, Fq_triv] = trivial_sparsity_helper(Gt_dense, Fs_dense, F_dags_dense, hyb_coeffs, n);
  DiagramEvaluator D_triv(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq_triv);
  auto OCA_trivial_bs = BlockDiagOpFun(D_triv.compute_self_energy(Gt_triv, topology));

  // Both references live on the full Hilbert space, so they are projected onto each atom_diag subspace before being compared block by block.
  auto ad = two_band_atom_diag_helper();
  for (int i = 0; i < OCA_result.get_num_block_cols(); i++) {
    SCOPED_TRACE("block " + std::to_string(i));
    auto result_dense_block = get_tensor_in_atom_diag_subspace(OCA_dense_result, i, ad);
    EXPECT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - result_dense_block)), 10 * eps);

    auto result_trivial_bs_block = get_tensor_in_atom_diag_subspace(OCA_trivial_bs.get_block(0), i, ad);
    EXPECT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - result_trivial_bs_block)), 10 * eps);
  }
}

/**
 * @brief Check that the block-sparse OCA correlator is unchanged when the block structure is discarded
 *
 * @details The correlator analogue of OCA_trivial_sparsity above: eval_correlator is run with the block structure of the atom_diag subspaces, with a
 * trivial sparsity pattern (a single block spanning the whole Hilbert space, and a single symmetry set holding all four flavors), and with the dense
 * evaluator, on the two-band Kanamori atom with the two-pole discrete-bath hybridization. All three must agree.
 *
 * Unlike the self-energy, the correlator is already a dense object in orbital space, so the three results are compared directly rather than block by
 * block.
 */
TEST(SparsityInvariance, OCA_correlator_trivial_sparsity) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;
  int n         = 4; // number of flavors

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_refl] = discrete_bath_helper(beta, Lambda, eps);
  auto hyb_coeffs            = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  // backbone
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = CorrelatorBackbone(topology, n);

  // --- block-sparse evaluation, using the block structure of the atom_diag subspaces ---
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto [mu_ops, kap_ops] = make_correlator_ops(Fq, n);
  auto OCA_result        = D.eval_correlator(Gt, B, mu_ops, kap_ops);

  // --- dense evaluation ---
  // The old dense constructor divides the poles it is given by beta internally, so it takes dlr_rf where the block-sparse one takes dlr_rf / beta.
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);
  auto Fset                               = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs);
  DenseDiagramEvaluator D_dense(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  auto OCA_dense_result = D_dense.eval_correlator(Gt_dense, B, Fs_dense, F_dags_dense);

  // --- block-sparse evaluation with trivial sparsity: one block, one symmetry set ---
  auto [Gt_triv, Fq_triv] = trivial_sparsity_helper(Gt_dense, Fs_dense, F_dags_dense, hyb_coeffs, n);
  DiagramEvaluator D_triv(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq_triv);
  auto [mu_ops_triv, kap_ops_triv] = make_correlator_ops(Fq_triv, n);
  auto OCA_trivial_bs              = D_triv.eval_correlator(Gt_triv, B, mu_ops_triv, kap_ops_triv);

  EXPECT_LE(nda::max_element(nda::abs(OCA_result - OCA_dense_result)), 1.0e-15);
  EXPECT_LE(nda::max_element(nda::abs(OCA_result - OCA_trivial_bs)), 1.0e-15);
}

/**
 * @brief Compare block-sparse and dense OCA self-energies for the spin-flip fermion model
 *
 * @param[in] use_particle_number_sym if true, the atom_diag subspaces are labeled by the particle
 * number N, so that all field operators share a single symmetry set; if false, the subspaces come
 * from autopartitioning alone and the field operators are spread over several symmetry sets.
 */
static void check_spin_flip_fermion(bool use_particle_number_sym) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  int norb             = 2;
  int nn               = 2 * norb; // 2 * number of orbitals
  auto [hyb, hyb_refl] = discrete_bath_spin_flip_helper(beta, Lambda, eps, nn);
  auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs

  // set up the spin-flip model, either from the particle number as a quantum number or by autopartitioning
  auto ad = spin_flip_atom_diag_helper(norb, use_particle_number_sym);

  // compute the atomic propagator and generate creation/annihilation operators in block-sparse storage
  auto dlr_it_abs           = cppdlr::rel2abs(itops.get_itnodes());
  auto Gt                   = ad_to_atom_prop(ad, beta, itops);
  auto Gt_block_sizes       = Gt.get_block_sizes();
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, nn);
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto result = BlockDiagOpFun(D.compute_self_energy(Gt, topology));

  // get dense Gt, field operators
  auto Gt_dense = Hmat_to_Gtmat(get_full_h_atomic(ad), beta, dlr_it_abs);
  auto Fset     = get_operators_dense(ad, hyb_coeffs);

  DenseDiagramEvaluator DDE(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  DDE.eval_self_energy_by_pairs(Gt_dense, B);
  auto result_dense = DDE.Sigma;

  for (int i = 0; i < Gt_block_sizes.size(); i++) {
    SCOPED_TRACE("block " + std::to_string(i));
    auto result_dense_block = get_tensor_in_atom_diag_subspace(result_dense, i, ad);
    ASSERT_LE(nda::max_element(nda::abs(result.get_block(i) - result_dense_block)), eps);
  }
}

TEST(SparsityInvariance, spin_flip_fermion) { check_spin_flip_fermion(true); } // self-energies match when using just total particle number symmetry

TEST(SparsityInvariance, spin_flip_fermion_sym_sets) { check_spin_flip_fermion(false); } // self-energies match when using all available symmetries

/**
 * @brief Compare block-sparse and dense OCA correlators for the spin-flip fermion model
 *
 * @param[in] use_particle_number_sym if true, the atom_diag subspaces are labeled by the particle
 * number N, so that all field operators share a single symmetry set; if false, the subspaces come
 * from autopartitioning alone and the field operators are spread over several symmetry sets.
 */
static void check_spin_flip_fermion_correlator(bool use_particle_number_sym) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  int norb             = 2;
  int nn               = 2 * norb; // 2 * number of orbitals
  auto [hyb, hyb_refl] = discrete_bath_spin_flip_helper(beta, Lambda, eps, nn);
  auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs

  // set up the spin-flip model, either from the particle number as a quantum number or by autopartitioning
  auto ad = spin_flip_atom_diag_helper(norb, use_particle_number_sym);

  // compute atomic propagator
  auto dlr_it_abs = cppdlr::rel2abs(itops.get_itnodes());
  auto Gt         = ad_to_atom_prop(ad, beta, itops);

  // generate creation/annihilation operators in block-sparse storage
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);
  auto [mu_ops, kap_ops]    = make_correlator_ops(Fq, nn);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = CorrelatorBackbone(topology, nn);
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto result = D.eval_correlator(Gt, B, mu_ops, kap_ops);

  // compare to dense backbone result
  auto Gt_dense = Hmat_to_Gtmat(get_full_h_atomic(ad), beta, dlr_it_abs);
  auto Fset     = get_operators_dense(ad, hyb_coeffs);
  DenseDiagramEvaluator D_dense(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  auto result_dense = D_dense.eval_correlator(Gt_dense, B, Fset.Fs, Fset.F_dags);

  ASSERT_LE(nda::max_element(nda::abs(result - result_dense)), 1.0e-15);
}

TEST(SparsityInvariance, spin_flip_fermion_correlator) {
  check_spin_flip_fermion_correlator(true);
} // spgfs match when using total particle number symmetry

TEST(SparsityInvariance, spin_flip_fermion_correlator_sym_sets) {
  check_spin_flip_fermion_correlator(false);
} // spgfs match when using available symmetries
