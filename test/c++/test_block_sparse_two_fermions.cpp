#include <gtest/gtest.h>

#include <nda/basic_functions.hpp>
#include <triqs_xca/atom_diag_utils.hpp>

#include <triqs_xca/hyb.hpp>
#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>

#include <triqs_xca/strong_cpl.hpp>

#include "block_sparse_utils.hpp"

using triqs::operators::n;

using nda::range;

using cppdlr::_;
using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;

using triqs_xca::dense::DenseDiagramEvaluator;

using triqs_xca::block_sparse::DiagramEvaluator;

using triqs_xca::atom_diag::ad_to_atom_prop;
using triqs_xca::atom_diag::get_operators_dense;

/**
 * @brief Test evaluation of the self-energy for a two-fermion system with a constant hybridization function
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams.
 */
TEST(two_fermions, const_hyb_se) {
  // Generate DLR imaginary-time object
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-12;
  auto dlr_rf   = build_dlr_rf(Lambda, eps);
  auto itops    = imtime_ops(Lambda, dlr_rf);
  int r         = itops.rank();

  // Two-fermion setup with one pole at zero gives the constant-hybridization test case
  auto two_fermion_model = two_fermion_model_helper(beta, Lambda, eps, 0.0, 0.0, 0.0);
  auto &hyb_coeffs       = two_fermion_model.hyb_coeffs;
  auto &hyb_poles        = two_fermion_model.hyb_poles;
  auto &ad               = two_fermion_model.ad;
  auto &G_ppsc           = two_fermion_model.G_ppsc;
  auto &G_bdof           = two_fermion_model.G_bdof;

  // Check that G_ppsc is correct by comparing to analytical expression
  auto dlr_it = itops.get_itnodes();
  auto G0_ana = nda::zeros<double>(r);
  double ln4  = std::numbers::ln2 * 2;
  for (int i = 0; i < r; ++i) {
    double t  = rel2abs(dlr_it(i));
    G0_ana(i) = -exp(-t * ln4);
  }
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) { ASSERT_LE(nda::max_element(nda::abs(G_bdof.get_block(b)(_, 0, 0) - G0_ana)), eps); }

  // Set up diagram evaluator for self-energy evaluation
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G_ppsc[0].mesh(), ad);

  // ----- NCA test -----
  nda::array<int, 2> topology1 = {{0, 1}};
  auto nca_se                  = D.compute_self_energy(G_ppsc, topology1);
  auto nca_se_ana              = nda::zeros<double>(r);
  nca_se_ana                   = -G0_ana;
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) { ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data()(_, 0, 0) - nca_se_ana)), eps); }

  // ----- OCA test -----
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  auto oca_se                  = D.compute_self_energy(G_ppsc, topology2);
  auto oca_se_ana              = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double t      = rel2abs(dlr_it(i)); // t = tau / beta
    oca_se_ana(i) = 0.25 * exp(-t * ln4) * t * t * beta * beta;
  }
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) { ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data()(_, 0, 0) - oca_se_ana)), eps); }

  // ----- third-order test -----
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  auto third_order_se          = D.compute_self_energy(G_ppsc, topology3);
  auto third_order_se_ana      = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double t              = rel2abs(dlr_it(i)); // t = tau / beta
    third_order_se_ana(i) = -1.0 / 96 * exp(-t * ln4) * pow(t, 4) * pow(beta, 4);
  }
  // third_order_se has three blocks: 1x1, 2x2, and 1x1. Check that all diagonal entries equal the analytical expression
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    ASSERT_LE(nda::max_element(nda::abs(third_order_se[b].data()(_, 0, 0) - third_order_se_ana)), eps);
  }
  ASSERT_LE(nda::max_element(nda::abs(third_order_se[1].data()(_, 1, 1) - third_order_se_ana)), eps);
}

/**
 * @brief Test evaluation of the single-particle Green's function for a two-fermion system with a constant hybridization function
 *
 * @details This tests the evaluation of the first-, second-, and third-order diagrams contributing to the single-particle Green's function.
 */
TEST(two_fermions, const_hyb_spgf) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-12;
  auto dlr_rf   = build_dlr_rf(Lambda, eps);
  auto itops    = imtime_ops(Lambda, dlr_rf);
  int r         = itops.rank();

  auto two_fermion_model = two_fermion_model_helper(beta, Lambda, eps, 0.0, 0.0, 0.0);
  auto &hyb_coeffs       = two_fermion_model.hyb_coeffs;
  auto &hyb_poles        = two_fermion_model.hyb_poles;
  auto &ad               = two_fermion_model.ad;
  auto &G_ppsc           = two_fermion_model.G_ppsc;

  auto dlr_it = itops.get_itnodes();
  auto G0_ana = nda::zeros<double>(r);
  double ln4  = std::numbers::ln2 * 2;
  for (int i = 0; i < r; ++i) {
    double t  = rel2abs(dlr_it(i));
    G0_ana(i) = -exp(-t * ln4);
  }

  // Set up diagram evaluator for single-particle Green's function evaluation
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G_ppsc[0].mesh(), ad);

  // ----- NCA test -----
  nda::array<int, 2> topology1 = {{0, 1}};
  auto nca_spgf                = D.compute_single_ptcle_gf(G_ppsc, topology1);
  auto nca_spgf_ana            = nda::zeros<double>(r, 2, 2);
  nca_spgf_ana(_, 0, 0)        = 0.5;
  nca_spgf_ana(_, 1, 1)        = 0.5;
  ASSERT_LE(nda::max_element(nda::abs(nca_spgf - nca_spgf_ana)), eps);

  // ----- OCA test -----
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  auto oca_spgf                = D.compute_single_ptcle_gf(G_ppsc, topology2);
  auto oca_spgf_ana            = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double t        = rel2abs(dlr_it(i)); // t = tau / beta
    oca_spgf_ana(i) = -0.25 * (beta * beta * t - beta * beta * t * t);
  }
  ASSERT_LE(nda::max_element(nda::abs(oca_spgf(_, 0, 0) - oca_spgf_ana)), eps);
  ASSERT_LE(nda::max_element(nda::abs(oca_spgf(_, 1, 1) - oca_spgf_ana)), eps);

  // ----- third-order test -----
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  auto third_order_spgf        = D.compute_single_ptcle_gf(G_ppsc, topology3);
  auto third_order_spgf_ana    = nda::zeros<double>(r);
  double beta4                 = beta * beta;
  beta4                        = beta4 * beta4;
  for (int i = 0; i < r; ++i) {
    double t                = rel2abs(dlr_it(i)); // t = tau / beta
    third_order_spgf_ana(i) = -0.5 * beta4 * t * t * (-t + 0.5 * (1 + t * t));
  }
  third_order_spgf_ana = third_order_spgf_ana / 16.0;
  ASSERT_LE(nda::max_element(nda::abs(third_order_spgf(_, 0, 0) - 2 * third_order_spgf_ana)), eps);
  ASSERT_LE(nda::max_element(nda::abs(third_order_spgf(_, 1, 1) - 2 * third_order_spgf_ana)), eps);
}

TEST(two_fermions, one_hyb_pole) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-12;

  auto two_fermion_model = two_fermion_model_helper(beta, Lambda, eps);
  auto &hyb_coeffs       = two_fermion_model.hyb_coeffs;
  auto &hyb_poles        = two_fermion_model.hyb_poles;
  auto &ad               = two_fermion_model.ad;
  auto &G_ppsc           = two_fermion_model.G_ppsc;

  int norb = 2;

  // compute single-particle Green's function for the two-fermion system with one hybridization pole
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G_ppsc[0].mesh(), ad);
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto spgf                   = D.compute_single_ptcle_gf(G_ppsc, topology);

  // compare to call to dense code
  auto hyb          = triqs_xca::hyb::coefs2vals(beta, Lambda, eps, hyb_coeffs, hyb_poles);
  auto dlr_rf       = build_dlr_rf(Lambda, eps);
  auto itops        = imtime_ops(Lambda, dlr_rf);
  auto hyb_refl     = itops.reflect(hyb);
  auto G_ppsc_dense = nda::zeros<dcomplex>(itops.rank(), ad.get_full_hilbert_space_dim(), ad.get_full_hilbert_space_dim());
  int s0            = 0;
  int s1            = 0;
  for (int s = 0; s < ad.n_subspaces(); ++s) {
    s1 += ad.get_fock_states(s).size();
    G_ppsc_dense(_, range(s0, s1), range(s0, s1)) = G_ppsc[s].data();
    s0                                            = s1;
  }
  auto Fset = get_operators_dense(ad, hyb_coeffs);
  hyb_poles = nda::make_regular(beta * hyb_poles);
  DenseDiagramEvaluator D_dense(beta, eps, itops, hyb_poles, hyb_coeffs, Fset);
  auto mu_ops  = Fset.Fs;
  auto kap_ops = Fset.F_dags;
  CorrelatorBackbone B(topology, norb);
  auto spgf_dense = D_dense.eval_correlator(G_ppsc_dense, B, mu_ops, kap_ops);
  ASSERT_LE(nda::max_element(nda::abs(spgf - spgf_dense)), 1.0e-15);
}
