#include <gtest/gtest.h>

#include <nda/algorithms.hpp>
#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/atom_diag_utils.hpp>

#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs_xca/block_sparse_manual.hpp>
#include <triqs_xca/block_sparse_manual_gf.hpp>
#include <triqs_xca/hyb.hpp>

#include <triqs_xca/strong_cpl.hpp>

#include "block_sparse_utils.hpp"

using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::many_body_operator_complex;
using triqs::operators::n;

using triqs_xca::dense::DenseDiagramEvaluator;
using triqs_xca::dense::DenseFSet;

using triqs_xca::block_sparse::DiagramEvaluator;
using triqs_xca::block_sparse::NCA_dense;
using triqs_xca::block_sparse::NCA_gf_dense;
using triqs_xca::block_sparse::OCA_dense;
using triqs_xca::block_sparse::OCA_gf_dense;

using triqs_xca::block_sparse::eval_eq;
using triqs_xca::block_sparse::third_order_tpz;

using triqs_xca::atom_diag::ad_to_atom_prop;
using triqs_xca::atom_diag::get_full_h_atomic;
using triqs_xca::atom_diag::get_operators;
using triqs_xca::atom_diag::get_operators_dense;
using triqs_xca::atom_diag::get_tensor_in_atom_diag_subspace;

/**
 * @file test_one_fermion_se_spgf_all_evals.cpp
 * 
 * @brief Tests of self-energy and single-particle Green's function diagram evaluators for models with one spinless fermion 
 * 
 * @details Each test covers a different hybridization that is a linear combination of zero, one, or two K(\tau, \omega) at different values of the 
 * poles \omega. All the tests compare the self-energy/single-particle Green's function computed analytically at first- (NCA), second- (OCA), and 
 * third order to the results of several evaluation routines. For NCA and OCA, there are comparisons to the following evaluators:
 *
 * - the `compute_self_energy/single_particle_gf` routine of DenseDiagramEvaluator
 * - the `compute_self_energy_by_pairs` routine of DenseDiagramEvaluator, which has no single-particle Green's function analogue
 * - the `compute_self_energy/single_particle_gf` routine of the DiagramEvaluator, which takes advantage of block-sparsity
 * - `N/OCA_(gf_)dense`, a routine that can only evaluate N/OCA with dense matmuls
 * - `N/OCA_(gf_)bs`, a routine that can only evaluate N/OCA taking advantage of block-sparsity
 * 
 * The last two routines above are not wrapped and exist just for testing purposes. For third-order diagrams, there are no analogues to these last 
 * two evaluators, as we are testing that the (Dense)DiagramEvaluator routines do in fact work at arbitrary order.
 */

/**
 * @brief Test computation of several diagrams for a spinless fermion with a constant hybridization
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams by comparing analytical calculations carried out
 * in examples/one_fermion_analytical_solutions.ipynb to the results of calls to the DiagramEvaluator compute_self_energy routine.
 */
TEST(one_fermion, const_hyb_se) {
  // Generate DLR imaginary-time object
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;
  auto dlr_rf   = build_dlr_rf(Lambda, eps);
  auto itops    = imtime_ops(Lambda, dlr_rf);
  int r         = itops.rank();

  auto one_fermion_model = one_fermion_model_helper(beta, Lambda, eps);
  auto &hyb_coeffs       = one_fermion_model.hyb_coeffs;
  auto &hyb_poles        = one_fermion_model.hyb_poles;
  auto &ad               = one_fermion_model.ad;
  auto &G0_ppsc          = one_fermion_model.G_ppsc;
  auto &G0_bdof          = one_fermion_model.G_bdof;

  // Check that G0_ppsc is correct by comparing to analytical expression
  auto dlr_it = itops.get_itnodes();
  auto G0_ana = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double t  = rel2abs(dlr_it(i));
    G0_ana(i) = -exp(-t * std::numbers::ln2);
  }
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) { ASSERT_LE(nda::max_element(nda::abs(G0_bdof.get_block(b)(_, 0, 0) - G0_ana)), eps); }

  // Set up the dense diagram evaluator, which sums the same backbones as the block-sparse one but over
  // the full Hilbert space, so its self-energy carries the same overall sign convention
  auto dlr_it_abs               = cppdlr::rel2abs(dlr_it);
  auto H_dense                  = get_full_h_atomic(ad);
  auto Gt_dense                 = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
  auto [Fs_dense, F_dags_dense] = get_operators_dense(ad);
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G0_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G0_ppsc[0].mesh(), Gt_dense)};
  auto G0_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G0_dense_blocks);
  DenseDiagramEvaluator D_dense(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);

  // Set up the block-sparse diagram evaluator and its field operators
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);

  // ----- NCA test -----
  nda::array<int, 2> topology1 = {{0, 1}};
  // Dense diagram evaluator
  auto nca_se_dde = D_dense.compute_self_energy(G0_ppsc_dense, topology1);
  ASSERT_LE(max_offdiag(nca_se_dde[0].data()), eps);
  // Dense diagram evaluator, evaluating the backbones in pairs
  auto nca_se_pairs = D_dense.compute_self_energy_by_pairs(G0_ppsc_dense, topology1);
  ASSERT_LE(nda::max_element(nda::abs(nca_se_dde[0].data() - nca_se_pairs[0].data())), eps);
  // Block-sparse diagram evaluator
  auto nca_se     = D.compute_self_energy(G0_ppsc, topology1);
  auto nca_se_ana = nda::zeros<double>(r);
  nca_se_ana      = -G0_ana / 2; // self-energy NCA contribution computed analytically
  // Compare computed and expected NCA
  ASSERT_LE(nda::max_element(nda::abs(nca_se[0].data()(_, 0, 0) - nca_se_ana)), eps);
  ASSERT_LE(nda::max_element(nda::abs(nca_se[1].data()(_, 0, 0) - nca_se_ana)), eps);
  // Compare with the dense diagram evaluator
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto nca_se_dde_block = get_tensor_in_atom_diag_subspace(nca_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() - nca_se_dde_block)), eps);
  }
  // Compare with manual dense NCA evaluator
  auto nca_se_dense = NCA_dense(D.hyb.values, D.hyb.values_reflect, Gt_dense, Fs_dense, F_dags_dense);
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto nca_se_dense_block = get_tensor_in_atom_diag_subspace(nca_se_dense, b, ad);
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() + nca_se_dense_block)), eps);
  }
  ASSERT_LE(max_offdiag(nca_se_dense), eps);
  // Compare with manual block-sparse NCA evaluator
  auto nca_se_manual = NCA_bs(D.hyb.values, D.hyb.values_reflect, G0_bdof, Fq);
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    // NCA_bs and DiagramEvaluator::compute_self_energy use opposite overall sign conventions for the
    // self-energy (cf. the "-Sigma_Diagram_calc" negation in test_block_sparse_NCA_manual.cpp), hence "+".
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() + nca_se_manual.get_block(b))), eps);
  }

  // ----- OCA test -----
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  // Dense diagram evaluator
  auto oca_se_dde = D_dense.compute_self_energy(G0_ppsc_dense, topology2);
  ASSERT_LE(max_offdiag(oca_se_dde[0].data()), eps);
  // Dense diagram evaluator, evaluating the backbones in pairs
  auto oca_se_pairs = D_dense.compute_self_energy_by_pairs(G0_ppsc_dense, topology2);
  ASSERT_LE(nda::max_element(nda::abs(oca_se_dde[0].data() - oca_se_pairs[0].data())), eps);
  // Block-sparse diagram evaluator
  auto oca_se = D.compute_self_energy(G0_ppsc, topology2);
  // OCA contribution should be identically zero
  for (int i = 0; i < 2; ++i) { ASSERT_LE(nda::max_element(nda::abs(oca_se[i].data()(_, 0, 0))), eps); }
  // Compare with the dense diagram evaluator
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto oca_se_dde_block = get_tensor_in_atom_diag_subspace(oca_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_dde_block)), eps);
  }
  // Compare with manual dense OCA evaluator
  auto oca_se_dense =
     OCA_dense(D.hyb.values, D.hyb.coeffs, D.hyb.values_reflect, D.hyb.coeffs, D.hyb.poles, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto oca_se_dense_block = get_tensor_in_atom_diag_subspace(oca_se_dense, b, ad);
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_dense_block)), eps);
  }
  ASSERT_LE(max_offdiag(oca_se_dense), eps);
  // Compare with manual block-sparse OCA evaluator
  auto oca_se_manual = OCA_bs(D.hyb.values, D.hyb.poles, itops, beta, G0_bdof, Fq);
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    // Unlike NCA (odd order), OCA (even order) does not need the sign flip; the (-1) per hybridization
    // line in NCA_bs/OCA_bs cancels against the extra line at second order.
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_manual.get_block(b))), eps);
  }

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the diagram evaluators are compared here.
  nda::array<int, 2> topology = {{0, 3}, {1, 4}, {2, 5}};
  // Dense diagram evaluator
  auto third_order_se_dde = D_dense.compute_self_energy(G0_ppsc_dense, topology);
  ASSERT_LE(max_offdiag(third_order_se_dde[0].data()), eps);
  // Dense diagram evaluator, evaluating the backbones in pairs
  auto third_order_se_pairs = D_dense.compute_self_energy_by_pairs(G0_ppsc_dense, topology);
  ASSERT_LE(nda::max_element(nda::abs(third_order_se_dde[0].data() - third_order_se_pairs[0].data())), eps);
  // Block-sparse diagram evaluator
  auto third_order_se     = D.compute_self_energy(G0_ppsc, topology);
  auto third_order_se_ana = nda::zeros<double>(r);
  double t                = 0;
  double bt4              = 0;
  for (int i = 0; i < r; ++i) {
    t                     = rel2abs(dlr_it(i)); // t = tau / beta
    bt4                   = beta * t;
    bt4                   = bt4 * bt4;
    bt4                   = bt4 * bt4;
    third_order_se_ana(i) = bt4 * exp(-t * std::numbers::ln2) / 192.0;
  }
  ASSERT_LE(nda::max_element(nda::abs(third_order_se[0].data()(_, 0, 0) - third_order_se_ana)), eps);
  ASSERT_LE(nda::max_element(nda::abs(third_order_se[1].data()(_, 0, 0) - third_order_se_ana)), eps);
  // Compare with the dense diagram evaluator
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto third_order_se_dde_block = get_tensor_in_atom_diag_subspace(third_order_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(third_order_se[b].data() - third_order_se_dde_block)), eps);
  }
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from a single pole
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams.
 */
TEST(one_fermion, one_hyb_pole_se) {
  double beta   = 1.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;

  // Generate DLR imaginary-time object
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();

  auto one_fermion_model = one_fermion_model_helper(beta, Lambda, eps, 0.8);
  auto &hyb_coeffs       = one_fermion_model.hyb_coeffs;
  auto &hyb_poles        = one_fermion_model.hyb_poles;
  auto &ad               = one_fermion_model.ad;
  auto &G0_ppsc          = one_fermion_model.G_ppsc;
  auto &G0_bdof          = one_fermion_model.G_bdof;
  auto dlr_it            = itops.get_itnodes();
  auto G0_ana            = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double t  = rel2abs(dlr_it(i));
    G0_ana(i) = -exp(-t * std::numbers::ln2);
  }
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) { ASSERT_LE(nda::max_element(nda::abs(G0_bdof.get_block(b)(_, 0, 0) - G0_ana)), eps); }

  // set up the dense diagram evaluator, which sums the same backbones as the block-sparse one but over
  // the full Hilbert space, so its self-energy carries the same overall sign convention
  auto dlr_it_abs                 = cppdlr::rel2abs(dlr_it);
  auto H0_dense                   = get_full_h_atomic(ad);
  auto Gt0_dense                  = Hmat_to_Gtmat(H0_dense, beta, dlr_it_abs);
  auto [Fs0_dense, F0_dags_dense] = get_operators_dense(ad);
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G0_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G0_ppsc[0].mesh(), Gt0_dense)};
  auto G0_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G0_dense_blocks);
  DenseDiagramEvaluator D_dense(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);

  // set up the block-sparse diagram evaluator and its field operators
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);

  // NCA
  nda::array<int, 2> topology1 = {{0, 1}};
  // dense diagram evaluator
  auto nca_se_dde = D_dense.compute_self_energy(G0_ppsc_dense, topology1);
  ASSERT_LE(max_offdiag(nca_se_dde[0].data()), eps);
  // dense diagram evaluator, evaluating the backbones in pairs
  auto nca_se_pairs = D_dense.compute_self_energy_by_pairs(G0_ppsc_dense, topology1);
  ASSERT_LE(nda::max_element(nda::abs(nca_se_dde[0].data() - nca_se_pairs[0].data())), eps);
  // block-sparse diagram evaluator
  auto nca_se     = D.compute_self_energy(G0_ppsc, topology1);
  auto nca_se_ana = nda::zeros<double>(r, 2);
  double t        = 0;
  double om       = hyb_poles(0);
  double tom      = 0;
  for (int i = 0; i < r; ++i) {
    t                = rel2abs(dlr_it(i)); // t = tau / beta
    tom              = t * om;
    nca_se_ana(i, 0) = exp(-t * std::numbers::ln2) * exp(tom) / (exp(beta * om) + 1);
    nca_se_ana(i, 1) = exp(-t * std::numbers::ln2) * exp(-tom) / (exp(-beta * om) + 1);
  }
  ASSERT_LE(nda::max_element(nda::abs(nca_se[0].data()(_, 0, 0) - nca_se_ana(_, 0))), eps);
  ASSERT_LE(nda::max_element(nda::abs(nca_se[1].data()(_, 0, 0) - nca_se_ana(_, 1))), eps);
  // compare with the dense diagram evaluator
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto nca_se_dde_block = get_tensor_in_atom_diag_subspace(nca_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() - nca_se_dde_block)), eps);
  }
  // compare with manual dense NCA evaluator
  auto nca_se_dense = NCA_dense(D.hyb.values, D.hyb.values_reflect, Gt0_dense, Fs0_dense, F0_dags_dense);
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto nca_se_dense_block = get_tensor_in_atom_diag_subspace(nca_se_dense, b, ad);
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() + nca_se_dense_block)), eps);
  }
  ASSERT_LE(max_offdiag(nca_se_dense), eps);
  // compare with manual block-sparse NCA evaluator
  auto nca_se_manual = NCA_bs(D.hyb.values, D.hyb.values_reflect, G0_bdof, Fq);
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    // NCA_bs and DiagramEvaluator::compute_self_energy use opposite overall sign conventions for the
    // self-energy (cf. the "-Sigma_Diagram_calc" negation in test_block_sparse_NCA_manual.cpp), hence "+".
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() + nca_se_manual.get_block(b))), eps);
  }

  // OCA
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  // dense diagram evaluator
  auto oca_se_dde = D_dense.compute_self_energy(G0_ppsc_dense, topology2);
  ASSERT_LE(max_offdiag(oca_se_dde[0].data()), eps);
  // dense diagram evaluator, evaluating the backbones in pairs
  auto oca_se_pairs = D_dense.compute_self_energy_by_pairs(G0_ppsc_dense, topology2);
  ASSERT_LE(nda::max_element(nda::abs(oca_se_dde[0].data() - oca_se_pairs[0].data())), eps);
  // block-sparse diagram evaluator
  auto oca_se = D.compute_self_energy(G0_ppsc, topology2);
  ASSERT_LE(nda::max_element(nda::abs(oca_se[0].data()(_, 0, 0))), eps);
  ASSERT_LE(nda::max_element(nda::abs(oca_se[1].data()(_, 0, 0))), eps);
  // compare with the dense diagram evaluator
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto oca_se_dde_block = get_tensor_in_atom_diag_subspace(oca_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_dde_block)), eps);
  }
  // compare with manual dense OCA evaluator
  auto oca_se_dense =
     OCA_dense(D.hyb.values, D.hyb.coeffs, D.hyb.values_reflect, D.hyb.coeffs, D.hyb.poles, itops, beta, Gt0_dense, Fs0_dense, F0_dags_dense);
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto oca_se_dense_block = get_tensor_in_atom_diag_subspace(oca_se_dense, b, ad);
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_dense_block)), eps);
  }
  ASSERT_LE(max_offdiag(oca_se_dense), eps);
  // compare with manual block-sparse OCA evaluator
  auto oca_se_manual = OCA_bs(D.hyb.values, D.hyb.poles, itops, beta, G0_bdof, Fq);
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    // Unlike NCA (odd order), OCA (even order) does not need the sign flip; the (-1) per hybridization
    // line in NCA_bs/OCA_bs cancels against the extra line at second order.
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_manual.get_block(b))), eps);
  }

  // third order; there are no manual third-order evaluators, so only the diagram evaluators are compared
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  // dense diagram evaluator
  auto third_order_se_dde = D_dense.compute_self_energy(G0_ppsc_dense, topology3);
  ASSERT_LE(max_offdiag(third_order_se_dde[0].data()), eps);
  // dense diagram evaluator, evaluating the backbones in pairs
  auto third_order_se_pairs = D_dense.compute_self_energy_by_pairs(G0_ppsc_dense, topology3);
  ASSERT_LE(nda::max_element(nda::abs(third_order_se_dde[0].data() - third_order_se_pairs[0].data())), eps);
  // block-sparse diagram evaluator
  auto third_order_se     = D.compute_self_energy(G0_ppsc, topology3);
  auto third_order_se_ana = nda::zeros<double>(r, 2);
  double denom            = om * (exp(beta * om) + 1);
  denom                        = denom * denom * denom;
  denom                        = 2 * om * denom;
  for (int i = 0; i < r; ++i) {
    t                        = rel2abs(dlr_it(i)); // t = tau / beta
    tom                      = t * om;
    third_order_se_ana(i, 0) = (tom * tom * exp(tom) - 4 * tom * exp(tom) - 2 * tom + 6 * exp(tom) - 6) * exp(beta * om);
    third_order_se_ana(i, 1) += (tom * tom + 4 * tom + 2 * (tom - 3) * exp(tom) + 6) * exp(om * (2 * beta - t));
    third_order_se_ana(i, _) *= exp(-t * std::numbers::ln2) / denom;
  }
  ASSERT_LE(nda::max_element(nda::abs(third_order_se_ana(_, 0) - third_order_se[0].data()(_, 0, 0))), eps);
  ASSERT_LE(nda::max_element(nda::abs(third_order_se_ana(_, 1) - third_order_se[1].data()(_, 0, 0))), eps);
  // compare with the dense diagram evaluator
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto third_order_se_dde_block = get_tensor_in_atom_diag_subspace(third_order_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(third_order_se[b].data() - third_order_se_dde_block)), eps);
  }
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from two poles
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams for a
 * two-pole hybridization Delta(tau) = K(tau, omega_1) + 2*K(tau, omega_2).
 */
TEST(one_fermion, two_hyb_poles_se) {
  double beta   = 1.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;

  // Generate DLR imaginary-time object (unsymmetrized, as in the one-pole test above)
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();

  // Two-pole hybridization: weight 1 at omega_1 = 0.6, weight 2 at omega_2 = -0.9
  int p    = 2;
  int norb = 1;
  nda::array<dcomplex, 3> hyb_coeffs(p, norb, norb);
  hyb_coeffs(0, 0, 0) = 1.0;
  hyb_coeffs(1, 0, 0) = 2.0;
  nda::vector<double> hyb_poles(p);
  hyb_poles(0) = 0.6;
  hyb_poles(1) = -0.9;

  // Trivial atomic Hamiltonian H = 0, one orbital -- same as one_fermion_model_helper
  using triqs::operators::many_body_operator_complex;
  using triqs::operators::n;
  many_body_operator_complex H;
  double mu = 0.0;
  many_body_operator_complex N;
  N = n("0", 0);
  H = -mu * N;

  triqs::atom_diag::fundamental_operator_set fop_set;
  fop_set.insert("0", 0);
  auto ad = triqs::atom_diag::atom_diag<true>(H, fop_set);

  auto G0_ppsc = triqs_xca::atom_diag::ad_to_atom_prop(ad, beta, Lambda, eps);
  auto G0_bdof = BlockDiagOpFun(G0_ppsc);

  // Check that G0_ppsc is correct by comparing to analytical expression (same as the other two tests --
  // independent of the hybridization since H = 0 in all three)
  auto dlr_it = itops.get_itnodes();
  auto G0_ana = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double t  = rel2abs(dlr_it(i));
    G0_ana(i) = -exp(-t * std::numbers::ln2);
  }
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) { ASSERT_LE(nda::max_element(nda::abs(G0_bdof.get_block(b)(_, 0, 0) - G0_ana)), eps); }

  // Set up the dense diagram evaluator, which sums the same backbones as the block-sparse one but over
  // the full Hilbert space, so its self-energy carries the same overall sign convention
  auto dlr_it_abs                 = cppdlr::rel2abs(dlr_it);
  auto H0_dense                   = get_full_h_atomic(ad);
  auto Gt0_dense                  = Hmat_to_Gtmat(H0_dense, beta, dlr_it_abs);
  auto [Fs0_dense, F0_dags_dense] = get_operators_dense(ad);
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G0_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G0_ppsc[0].mesh(), Gt0_dense)};
  auto G0_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G0_dense_blocks);
  DenseDiagramEvaluator D_dense(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);

  // Set up the block-sparse diagram evaluator and its field operators
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);

  // ----- NCA test -----
  // First order is linear in Delta, so the analytic reference is just the coefficient-weighted sum of
  // the single-pole NCA formula over the two poles.
  nda::array<int, 2> topology1 = {{0, 1}};
  // compare with the dense diagram evaluator
  auto nca_se_dde = D_dense.compute_self_energy(G0_ppsc_dense, topology1);
  ASSERT_LE(max_offdiag(nca_se_dde[0].data()), eps);
  // dense diagram evaluator, evaluating the backbones in pairs
  auto nca_se_pairs = D_dense.compute_self_energy_by_pairs(G0_ppsc_dense, topology1);
  ASSERT_LE(nda::max_element(nda::abs(nca_se_dde[0].data() - nca_se_pairs[0].data())), eps);
  // block-sparse diagram evaluator
  auto nca_se = D.compute_self_energy(G0_ppsc, topology1);

  auto nca_se_ana = nda::zeros<double>(r, 2);
  double t        = 0;
  double om1      = hyb_poles(0);
  double om2      = hyb_poles(1);
  double c1       = hyb_coeffs(0, 0, 0).real();
  double c2       = hyb_coeffs(1, 0, 0).real();
  for (int i = 0; i < r; ++i) {
    t                = rel2abs(dlr_it(i)); // t = tau / beta
    nca_se_ana(i, 0) = c1 * exp(-t * std::numbers::ln2) * exp(t * om1) / (exp(beta * om1) + 1)
       + c2 * exp(-t * std::numbers::ln2) * exp(t * om2) / (exp(beta * om2) + 1);
    nca_se_ana(i, 1) = c1 * exp(-t * std::numbers::ln2) * exp(-t * om1) / (exp(-beta * om1) + 1)
       + c2 * exp(-t * std::numbers::ln2) * exp(-t * om2) / (exp(-beta * om2) + 1);
  }
  ASSERT_LE(nda::max_element(nda::abs(nca_se[0].data()(_, 0, 0) - nca_se_ana(_, 0))), eps);
  ASSERT_LE(nda::max_element(nda::abs(nca_se[1].data()(_, 0, 0) - nca_se_ana(_, 1))), eps);
  // compare with the dense diagram evaluator
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto nca_se_dde_block = get_tensor_in_atom_diag_subspace(nca_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() - nca_se_dde_block)), eps);
  }
  // compare with manual dense NCA evaluator
  auto nca_se_dense = NCA_dense(D.hyb.values, D.hyb.values_reflect, Gt0_dense, Fs0_dense, F0_dags_dense);
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto nca_se_dense_block = get_tensor_in_atom_diag_subspace(nca_se_dense, b, ad);
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() + nca_se_dense_block)), eps);
  }
  ASSERT_LE(max_offdiag(nca_se_dense), eps);
  // compare with manual block-sparse NCA evaluator
  auto nca_se_manual = NCA_bs(D.hyb.values, D.hyb.values_reflect, G0_bdof, Fq);
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    // NCA_bs and DiagramEvaluator::compute_self_energy use opposite overall sign conventions for the
    // self-energy (cf. the "-Sigma_Diagram_calc" negation in test_block_sparse_NCA_manual.cpp), hence "+".
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() + nca_se_manual.get_block(b))), eps);
  }

  // ----- OCA test -----
  // Still identically zero: the combinatorial argument (creation/annihilation operators must alternate
  // for a single fermion level) doesn't depend on how many poles Delta has.
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  // dense diagram evaluator
  auto oca_se_dde = D_dense.compute_self_energy(G0_ppsc_dense, topology2);
  ASSERT_LE(max_offdiag(oca_se_dde[0].data()), eps);
  // dense diagram evaluator, evaluating the backbones in pairs
  auto oca_se_pairs = D_dense.compute_self_energy_by_pairs(G0_ppsc_dense, topology2);
  ASSERT_LE(nda::max_element(nda::abs(oca_se_dde[0].data() - oca_se_pairs[0].data())), eps);
  // block-sparse diagram evaluator
  auto oca_se = D.compute_self_energy(G0_ppsc, topology2);
  ASSERT_LE(nda::max_element(nda::abs(oca_se[0].data()(_, 0, 0))), eps);
  ASSERT_LE(nda::max_element(nda::abs(oca_se[1].data()(_, 0, 0))), eps);
  // compare with the dense diagram evaluator
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto oca_se_dde_block = get_tensor_in_atom_diag_subspace(oca_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_dde_block)), eps);
  }
  // compare with manual dense OCA evaluator
  auto oca_se_dense =
     OCA_dense(D.hyb.values, D.hyb.coeffs, D.hyb.values_reflect, D.hyb.coeffs, D.hyb.poles, itops, beta, Gt0_dense, Fs0_dense, F0_dags_dense);
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto oca_se_dense_block = get_tensor_in_atom_diag_subspace(oca_se_dense, b, ad);
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_dense_block)), eps);
  }
  ASSERT_LE(max_offdiag(oca_se_dense), eps);
  // compare with manual block-sparse OCA evaluator
  auto oca_se_manual = OCA_bs(D.hyb.values, D.hyb.poles, itops, beta, G0_bdof, Fq);
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    // Unlike NCA (odd order), OCA (even order) does not need the sign flip; the (-1) per hybridization
    // line in NCA_bs/OCA_bs cancels against the extra line at second order.
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_manual.get_block(b))), eps);
  }

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the diagram evaluators are compared here.
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  // dense diagram evaluator
  auto third_order_se_dde = D_dense.compute_self_energy(G0_ppsc_dense, topology3);
  ASSERT_LE(max_offdiag(third_order_se_dde[0].data()), eps);
  // dense diagram evaluator, evaluating the backbones in pairs
  auto third_order_se_pairs = D_dense.compute_self_energy_by_pairs(G0_ppsc_dense, topology3);
  ASSERT_LE(nda::max_element(nda::abs(third_order_se_dde[0].data() - third_order_se_pairs[0].data())), eps);
  // block-sparse diagram evaluator
  auto third_order_se = D.compute_self_energy(G0_ppsc, topology3);

  auto se00_coeffs = itops.vals2coefs(third_order_se[0].data()(_, 0, 0));
  auto se11_coeffs = itops.vals2coefs(third_order_se[1].data()(_, 0, 0));

  // Reference values for Sigma_00(tau), Sigma_11(tau) at beta=1, omega_1=0.6, omega_2=-0.9, computed from
  // the closed form in one_fermion_two_poles_analytical_solutions.ipynb and cross-validated there against
  // independent brute-force nested quadrature of the undecomposed diagram integral.
  std::vector<double> tau_pts  = {0.1, 0.3, 0.5, 0.7, 0.9};
  std::vector<double> se00_ref = {0.000014094454748, 0.000890679147681, 0.005485107140040, 0.017223807607182, 0.039435093044707};
  std::vector<double> se11_ref = {0.000010136868438, 0.000699571262738, 0.004714452501594, 0.016214271857205, 0.040647750033025};

  for (size_t k = 0; k < tau_pts.size(); ++k) {
    dcomplex se00_val = itops.coefs2eval(se00_coeffs, tau_pts[k]);
    dcomplex se11_val = itops.coefs2eval(se11_coeffs, tau_pts[k]);
    ASSERT_LE(std::abs(se00_val - se00_ref[k]), eps);
    ASSERT_LE(std::abs(se11_val - se11_ref[k]), eps);
  }

  // compare with the dense diagram evaluator
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) {
    auto third_order_se_dde_block = get_tensor_in_atom_diag_subspace(third_order_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(third_order_se[b].data() - third_order_se_dde_block)), eps);
  }

  // ----- trapezoidal cross-check -----
  // Independent, in-repo verification of third_order_se, computed by direct trapezoidal quadrature
  // (third_order_tpz, c++/triqs_xca/block_sparse_manual.hpp) of the same topology {{0,3},{1,4},{2,5}}.
  //
  auto dense_model = one_fermion_model_dense_helper(beta, Lambda, eps, 0.0);
  auto Gt_dense    = dense_model.G_ppsc_dense[0].data();
  auto &Fs_dense   = dense_model.Fset_dense.Fs;
  // itops-based overload: the beta/Lambda/eps convenience overload hardcodes a symmetrized grid internally
  auto hyb_dense = triqs_xca::hyb::coefs2vals(beta, itops, hyb_coeffs, hyb_poles);

  int n_quad              = 20;
  auto third_order_se_tpz = third_order_tpz(hyb_dense, itops, beta, Gt_dense, Fs_dense, n_quad);
  auto third_order_se0_eq = eval_eq(itops, third_order_se[0].data(), n_quad);
  auto third_order_se1_eq = eval_eq(itops, third_order_se[1].data(), n_quad);

  double tpz_tol = 2.0e-3; // ~2x the empirically observed n_quad=20 error (~9.2e-4)
  ASSERT_LE(nda::max_element(nda::abs(third_order_se0_eq(_, 0, 0) - third_order_se_tpz(_, 0, 0))), tpz_tol);
  ASSERT_LE(nda::max_element(nda::abs(third_order_se1_eq(_, 0, 0) - third_order_se_tpz(_, 1, 1))), tpz_tol);
  ASSERT_LE(max_offdiag(third_order_se_tpz), tpz_tol);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a constant hybridization
 *
 * @details This tests the evaluation of the first-, second-, and third-order single-particle Green's functions diagrams.
 */
TEST(one_fermion, const_hyb_spgf) {
  // Generate DLR imaginary-time object
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;
  auto dlr_rf   = build_dlr_rf(Lambda, eps);
  auto itops    = imtime_ops(Lambda, dlr_rf);
  int r         = itops.rank();

  auto one_fermion_model = one_fermion_model_helper(beta, Lambda, eps);
  auto &hyb_coeffs       = one_fermion_model.hyb_coeffs;
  auto &hyb_poles        = one_fermion_model.hyb_poles;
  auto &ad               = one_fermion_model.ad;
  auto &G0_ppsc          = one_fermion_model.G_ppsc;
  auto &G0_bdof          = one_fermion_model.G_bdof;
  auto dlr_it            = itops.get_itnodes();

  // Set up the dense diagram evaluator, which sums the same backbones as the block-sparse one but over
  // the full Hilbert space rather than block by block
  auto dlr_it_abs               = cppdlr::rel2abs(dlr_it);
  auto H0_dense                 = get_full_h_atomic(ad);
  auto Gt0_dense                = Hmat_to_Gtmat(H0_dense, beta, dlr_it_abs);
  auto Gt0_dense_refl           = itops.reflect(Gt0_dense);
  auto [Fs_dense, F_dags_dense] = get_operators_dense(ad);
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G0_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G0_ppsc[0].mesh(), Gt0_dense)};
  auto G0_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G0_dense_blocks);
  DenseDiagramEvaluator D_dense(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);

  // Set up the block-sparse diagram evaluator and its field operators
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);

  // ----- NCA test -----
  nda::array<int, 2> topology1 = {{0, 1}};
  // dense diagram evaluator
  auto nca_gf_dde = D_dense.compute_single_ptcle_gf(G0_ppsc_dense, topology1);
  // block-sparse diagram evaluator
  auto nca_gf     = D.compute_single_ptcle_gf(G0_ppsc, topology1);
  auto nca_gf_ana = nda::ones<dcomplex>(r);
  nca_gf_ana      = nca_gf_ana / 2;
  // Compare computed and expected NCA
  ASSERT_LE(nda::max_element(nda::abs(nca_gf(_, 0, 0) - nca_gf_ana)), eps);
  // compare to the dense diagram evaluator
  ASSERT_LE(nda::max_element(nda::abs(nca_gf - nca_gf_dde)), eps);
  // compare to manual dense NCA Green's function evaluator
  auto nca_gf_dense = NCA_gf_dense(Gt0_dense, Gt0_dense_refl, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(nca_gf - nca_gf_dense)), eps);
  // compare to manual block-sparse NCA Green's function evaluator
  std::vector<nda::array<dcomplex, 3>> G0_refl_blocks;
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) { G0_refl_blocks.push_back(nda::make_regular(itops.reflect(G0_bdof.get_block(b)))); }
  nda::vector<int> G0_zero_block_indices(G0_bdof.get_num_block_cols());
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) { G0_zero_block_indices(b) = G0_bdof.get_zero_block_index(b); }
  BlockDiagOpFun G0_bdof_refl(G0_refl_blocks, G0_zero_block_indices);
  auto nca_gf_manual = NCA_gf_bs(G0_bdof, G0_bdof_refl, Fq);
  ASSERT_LE(nda::max_element(nda::abs(nca_gf - nca_gf_manual)), eps);

  // ----- OCA test -----
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  // dense diagram evaluator
  auto oca_gf_dde = D_dense.compute_single_ptcle_gf(G0_ppsc_dense, topology2);
  // block-sparse diagram evaluator
  auto oca_gf = D.compute_single_ptcle_gf(G0_ppsc, topology2);
  // OCA contribution should be identically zero
  ASSERT_LE(nda::max_element(nda::abs(oca_gf)), eps);
  // compare to the dense diagram evaluator
  ASSERT_LE(nda::max_element(nda::abs(oca_gf - oca_gf_dde)), eps);
  // compare to manual dense OCA Green's function evaluator
  auto oca_gf_dense = OCA_gf_dense(D.hyb.coeffs, D.hyb.coeffs, D.hyb.poles, itops, beta, Gt0_dense, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(oca_gf - oca_gf_dense)), eps);
  // compare to manual block-sparse OCA Green's function evaluator
  auto oca_gf_manual = OCA_gf_bs(D.hyb.poles, itops, beta, G0_bdof, Fq);
  ASSERT_LE(nda::max_element(nda::abs(oca_gf - oca_gf_manual)), eps);

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the two diagram evaluators are compared here.
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  // dense diagram evaluator
  auto third_order_gf_dde = D_dense.compute_single_ptcle_gf(G0_ppsc_dense, topology3);
  // block-sparse diagram evaluator
  auto third_order_gf     = D.compute_single_ptcle_gf(G0_ppsc, topology3);
  auto third_order_gf_ana = nda::zeros<double>(r);
  double halfbeta         = beta / 2.0;
  double halfbetasq       = halfbeta * halfbeta;
  double halfbeta4        = halfbetasq * halfbetasq;
  for (int i = 0; i < r; ++i) {
    double t              = rel2abs(dlr_it(i)); // t = tau / beta
    third_order_gf_ana(i) = halfbeta4 * (1.0 - t) * (1.0 - t) * t * t / 2.0;
  }
  ASSERT_LE(nda::max_element(nda::abs(third_order_gf(_, 0, 0) - third_order_gf_ana)), eps);
  // compare to the dense diagram evaluator
  ASSERT_LE(nda::max_element(nda::abs(third_order_gf - third_order_gf_dde)), eps);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from a single pole
 *
 * @details This tests the evaluation of the first-, second-, and third-order single-particle Green's functions diagrams.
 */
TEST(one_fermion, one_hyb_pole_spgf) {
  // Generate DLR imaginary-time object
  double beta   = 1.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;
  auto dlr_rf   = build_dlr_rf(Lambda, eps);
  auto itops    = imtime_ops(Lambda, dlr_rf);
  int r         = itops.rank();

  auto one_fermion_model = one_fermion_model_helper(beta, Lambda, eps, 0.8);
  auto &hyb_coeffs       = one_fermion_model.hyb_coeffs;
  auto &hyb_poles        = one_fermion_model.hyb_poles;
  auto &ad               = one_fermion_model.ad;
  auto &G0_ppsc          = one_fermion_model.G_ppsc;
  auto &G0_bdof          = one_fermion_model.G_bdof;
  auto dlr_it            = itops.get_itnodes();

  // Set up the dense diagram evaluator, which sums the same backbones as the block-sparse one but over
  // the full Hilbert space rather than block by block
  auto dlr_it_abs               = cppdlr::rel2abs(dlr_it);
  auto H0_dense                 = get_full_h_atomic(ad);
  auto Gt0_dense                = Hmat_to_Gtmat(H0_dense, beta, dlr_it_abs);
  auto Gt0_dense_refl           = itops.reflect(Gt0_dense);
  auto [Fs_dense, F_dags_dense] = get_operators_dense(ad);
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G0_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G0_ppsc[0].mesh(), Gt0_dense)};
  auto G0_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G0_dense_blocks);
  DenseDiagramEvaluator D_dense(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);

  // Set up the block-sparse diagram evaluator and its field operators
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);

  // ----- NCA test -----
  nda::array<int, 2> topology1 = {{0, 1}};
  // dense diagram evaluator
  auto nca_gf_dde = D_dense.compute_single_ptcle_gf(G0_ppsc_dense, topology1);
  // block-sparse diagram evaluator
  auto nca_gf     = D.compute_single_ptcle_gf(G0_ppsc, topology1);
  auto nca_gf_ana = nda::ones<dcomplex>(r);
  nca_gf_ana      = nca_gf_ana / 2;
  // Compare computed and expected NCA
  ASSERT_LE(nda::max_element(nda::abs(nca_gf(_, 0, 0) - nca_gf_ana)), eps);
  // compare to the dense diagram evaluator
  ASSERT_LE(nda::max_element(nda::abs(nca_gf - nca_gf_dde)), eps);
  // compare to manual dense NCA Green's function evaluator
  auto nca_gf_dense = NCA_gf_dense(Gt0_dense, Gt0_dense_refl, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(nca_gf - nca_gf_dense)), eps);
  // compare to manual block-sparse NCA Green's function evaluator
  std::vector<nda::array<dcomplex, 3>> G0_refl_blocks;
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) { G0_refl_blocks.push_back(nda::make_regular(itops.reflect(G0_bdof.get_block(b)))); }
  nda::vector<int> G0_zero_block_indices(G0_bdof.get_num_block_cols());
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) { G0_zero_block_indices(b) = G0_bdof.get_zero_block_index(b); }
  BlockDiagOpFun G0_bdof_refl(G0_refl_blocks, G0_zero_block_indices);
  auto nca_gf_manual = NCA_gf_bs(G0_bdof, G0_bdof_refl, Fq);
  ASSERT_LE(nda::max_element(nda::abs(nca_gf - nca_gf_manual)), eps);

  // ----- OCA test -----
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  // dense diagram evaluator
  auto oca_gf_dde = D_dense.compute_single_ptcle_gf(G0_ppsc_dense, topology2);
  // block-sparse diagram evaluator
  auto oca_gf = D.compute_single_ptcle_gf(G0_ppsc, topology2);
  // OCA contribution should be identically zero
  ASSERT_LE(nda::max_element(nda::abs(oca_gf)), eps);
  // compare to the dense diagram evaluator
  ASSERT_LE(nda::max_element(nda::abs(oca_gf - oca_gf_dde)), eps);
  // compare to manual dense OCA Green's function evaluator
  auto oca_gf_dense = OCA_gf_dense(D.hyb.coeffs, D.hyb.coeffs, D.hyb.poles, itops, beta, Gt0_dense, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(oca_gf - oca_gf_dense)), eps);
  // compare to manual block-sparse OCA Green's function evaluator
  auto oca_gf_manual = OCA_gf_bs(D.hyb.poles, itops, beta, G0_bdof, Fq);
  ASSERT_LE(nda::max_element(nda::abs(oca_gf - oca_gf_manual)), eps);

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the two diagram evaluators are compared here.
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  // dense diagram evaluator
  auto third_order_gf_dde = D_dense.compute_single_ptcle_gf(G0_ppsc_dense, topology3);
  // block-sparse diagram evaluator
  auto third_order_gf     = D.compute_single_ptcle_gf(G0_ppsc, topology3);
  auto third_order_gf_ana = nda::zeros<double>(r);
  double om               = hyb_poles(0);
  for (int i = 0; i < r; ++i) {
    double t              = rel2abs(dlr_it(i)); // t = tau / beta
    third_order_gf_ana(i) = (t + (exp(-om * t) - 1.0) / om) * (t - beta + (exp(om * (beta - t)) - 1.0) / om)
       / (2 * om * om * (1 + exp(-beta * om)) * (exp(beta * om) + 1));
  }

  ASSERT_LE(nda::max_element(nda::abs(third_order_gf(_, 0, 0) - third_order_gf_ana)), eps);
  // compare to the dense diagram evaluator
  ASSERT_LE(nda::max_element(nda::abs(third_order_gf - third_order_gf_dde)), eps);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from two poles
 *
 * @details This tests the evaluation of the first-, second-, and third-order single-particle Green's function
 * diagrams for a two-pole hybridization Delta(tau) = K(tau, omega_1) + 2*K(tau, omega_2), mirroring the
 * two-pole self-energy test `one_fermion.two_hyb_poles_se` above.
 */
TEST(one_fermion, two_hyb_poles_spgf) {
  double beta   = 1.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;

  // Generate DLR imaginary-time object (unsymmetrized, as in the one-pole test above)
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();

  // Two-pole hybridization: weight 1 at omega_1 = 0.6, weight 2 at omega_2 = -0.9
  int p    = 2;
  int norb = 1;
  nda::array<dcomplex, 3> hyb_coeffs(p, norb, norb);
  hyb_coeffs(0, 0, 0) = 1.0;
  hyb_coeffs(1, 0, 0) = 2.0;
  nda::vector<double> hyb_poles(p);
  hyb_poles(0) = 0.6;
  hyb_poles(1) = -0.9;

  // Trivial atomic Hamiltonian H = 0, one orbital -- same as one_fermion_model_helper
  using triqs::operators::many_body_operator_complex;
  using triqs::operators::n;
  many_body_operator_complex H;
  double mu = 0.0;
  many_body_operator_complex N;
  N = n("0", 0);
  H = -mu * N;

  triqs::atom_diag::fundamental_operator_set fop_set;
  fop_set.insert("0", 0);
  auto ad = triqs::atom_diag::atom_diag<true>(H, fop_set);

  auto G0_ppsc = triqs_xca::atom_diag::ad_to_atom_prop(ad, beta, Lambda, eps);
  BlockDiagOpFun G0_bdof(G0_ppsc);

  // Set up the dense diagram evaluator, which sums the same backbones as the block-sparse one but over
  // the full Hilbert space rather than block by block
  auto dlr_it_abs               = cppdlr::rel2abs(itops.get_itnodes());
  auto H0_dense                 = get_full_h_atomic(ad);
  auto Gt0_dense                = Hmat_to_Gtmat(H0_dense, beta, dlr_it_abs);
  auto Gt0_dense_refl           = itops.reflect(Gt0_dense);
  auto [Fs_dense, F_dags_dense] = get_operators_dense(ad);
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G0_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G0_ppsc[0].mesh(), Gt0_dense)};
  auto G0_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G0_dense_blocks);
  DenseDiagramEvaluator D_dense(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);

  // Set up the block-sparse diagram evaluator and its field operators
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);

  // ----- NCA test -----
  // NCA is independent of the hybridization (no Delta lines at this order), so the reference is the same
  // constant 1/2 seen in the const-hybridization and one-pole tests above.
  nda::array<int, 2> topology1 = {{0, 1}};
  // dense diagram evaluator
  auto nca_gf_dde = D_dense.compute_single_ptcle_gf(G0_ppsc_dense, topology1);
  // block-sparse diagram evaluator
  auto nca_gf     = D.compute_single_ptcle_gf(G0_ppsc, topology1);
  auto nca_gf_ana = nda::ones<dcomplex>(r);
  nca_gf_ana      = nca_gf_ana / 2;
  ASSERT_LE(nda::max_element(nda::abs(nca_gf(_, 0, 0) - nca_gf_ana)), eps);
  // compare to the dense diagram evaluator
  ASSERT_LE(nda::max_element(nda::abs(nca_gf - nca_gf_dde)), eps);
  // compare to manual dense NCA Green's function evaluator
  auto nca_gf_dense = NCA_gf_dense(Gt0_dense, Gt0_dense_refl, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(nca_gf - nca_gf_dense)), eps);
  // compare to manual block-sparse NCA Green's function evaluator
  std::vector<nda::array<dcomplex, 3>> G0_refl_blocks;
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) { G0_refl_blocks.push_back(nda::make_regular(itops.reflect(G0_bdof.get_block(b)))); }
  nda::vector<int> G0_zero_block_indices(G0_bdof.get_num_block_cols());
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) { G0_zero_block_indices(b) = G0_bdof.get_zero_block_index(b); }
  BlockDiagOpFun G0_bdof_refl(G0_refl_blocks, G0_zero_block_indices);
  auto nca_gf_manual = NCA_gf_bs(G0_bdof, G0_bdof_refl, Fq);
  ASSERT_LE(nda::max_element(nda::abs(nca_gf - nca_gf_manual)), eps);

  // ----- OCA test -----
  // Still identically zero: the combinatorial argument (creation/annihilation operators must alternate
  // for a single fermion level) doesn't depend on how many poles Delta has.
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  // dense diagram evaluator
  auto oca_gf_dde = D_dense.compute_single_ptcle_gf(G0_ppsc_dense, topology2);
  // block-sparse diagram evaluator
  auto oca_gf = D.compute_single_ptcle_gf(G0_ppsc, topology2);
  ASSERT_LE(nda::max_element(nda::abs(oca_gf)), eps);
  // compare to the dense diagram evaluator
  ASSERT_LE(nda::max_element(nda::abs(oca_gf - oca_gf_dde)), eps);
  // compare to manual dense OCA Green's function evaluator
  auto oca_gf_dense = OCA_gf_dense(D.hyb.coeffs, D.hyb.coeffs, D.hyb.poles, itops, beta, Gt0_dense, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(oca_gf - oca_gf_dense)), eps);
  // compare to manual block-sparse OCA Green's function evaluator
  auto oca_gf_manual = OCA_gf_bs(D.hyb.poles, itops, beta, G0_bdof, Fq);
  ASSERT_LE(nda::max_element(nda::abs(oca_gf - oca_gf_manual)), eps);

  // ----- third-order test -----
  // There are no manual third-order evaluators, so only the two diagram evaluators are compared here.
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  // dense diagram evaluator
  auto third_order_gf_dde = D_dense.compute_single_ptcle_gf(G0_ppsc_dense, topology3);
  // block-sparse diagram evaluator
  auto third_order_gf        = D.compute_single_ptcle_gf(G0_ppsc, topology3);
  auto third_order_gf_coeffs = itops.vals2coefs(third_order_gf(_, 0, 0));

  // Reference values for g_3(tau) at beta=1, omega_1=0.6, omega_2=-0.9, computed from the closed form
  // in one_fermion_two_poles_analytical_solutions.ipynb and cross-validated there against independent
  // brute-force nested quadrature of the undecomposed diagram integral.
  std::vector<double> tau_pts = {0.1, 0.3, 0.5, 0.7, 0.9};
  std::vector<double> gf_ref  = {0.001818250732044, 0.010263216835087, 0.015229427191986, 0.011360586534472, 0.002226904649606};

  for (size_t k = 0; k < tau_pts.size(); ++k) {
    dcomplex gf_val = itops.coefs2eval(third_order_gf_coeffs, tau_pts[k]);
    ASSERT_LE(std::abs(gf_val - gf_ref[k]), eps);
  }

  // compare to the dense diagram evaluator
  ASSERT_LE(nda::max_element(nda::abs(third_order_gf - third_order_gf_dde)), eps);
}
