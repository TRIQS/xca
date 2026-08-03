#include <gtest/gtest.h>

#include <nda/basic_functions.hpp>
#include <triqs_xca/atom_diag_utils.hpp>

#include <triqs_xca/hyb.hpp>
#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs_xca/block_sparse_manual.hpp>
#include <triqs_xca/block_sparse_manual_gf.hpp>

#include <triqs_xca/strong_cpl.hpp>

#include "block_sparse_utils.hpp"

using triqs::operators::n;

using nda::range;

using cppdlr::_;
using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;

using triqs_xca::dense::DenseDiagramEvaluator;

using triqs_xca::block_sparse::DiagramEvaluator;
using triqs_xca::block_sparse::NCA_dense;
using triqs_xca::block_sparse::NCA_gf_dense;
using triqs_xca::block_sparse::OCA_dense;
using triqs_xca::block_sparse::OCA_gf_dense;

using triqs_xca::atom_diag::ad_to_atom_prop;
using triqs_xca::atom_diag::get_full_h_atomic;
using triqs_xca::atom_diag::get_operators;
using triqs_xca::atom_diag::get_operators_dense;
using triqs_xca::atom_diag::get_tensor_in_atom_diag_subspace;

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
  // The self-energy is diagonal with the same entry in every occupation sector, so check every diagonal
  // entry of every block against the closed form and every off-diagonal entry against zero.
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto nca_se_block = nca_se[b].data();
    for (int d = 0; d < nca_se_block.extent(1); ++d) {
      ASSERT_LE(nda::max_element(nda::abs(nca_se_block(_, d, d) - nca_se_ana)), eps);
    }
    ASSERT_LE(max_offdiag(nca_se_block), eps);
  }
  // compare to manual block-sparse NCA
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);
  auto nca_se_manual        = NCA_bs(D.hyb.values, D.hyb.values_reflect, G_bdof, Fq);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    // NCA_bs and DiagramEvaluator::compute_self_energy use opposite overall sign conventions for the
    // self-energy (cf. the "-Sigma_Diagram_calc" negation in test_block_sparse_NCA_manual.cpp), hence "+".
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() + nca_se_manual.get_block(b))), eps);
  }
  // compare to manual dense NCA
  auto dlr_it_abs               = cppdlr::rel2abs(dlr_it);
  auto H_dense                  = get_full_h_atomic(ad);
  auto Gt_dense                 = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
  auto [Fs_dense, F_dags_dense] = get_operators_dense(ad);
  auto nca_se_dense             = NCA_dense(D.hyb.values, D.hyb.values_reflect, Gt_dense, Fs_dense, F_dags_dense);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto nca_se_dense_block = get_tensor_in_atom_diag_subspace(nca_se_dense, b, ad);
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() + nca_se_dense_block)), eps);
  }
  ASSERT_LE(max_offdiag(nca_se_dense), eps);
  // compare to the dense diagram evaluator, which sums the same backbones as the block-sparse one but
  // over the full Hilbert space, so its self-energy carries the same overall sign convention
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G_ppsc[0].mesh(), Gt_dense)};
  auto G_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G_dense_blocks);
  DenseDiagramEvaluator D_dense(hyb_poles, hyb_coeffs, G_ppsc[0].mesh(), ad);
  auto nca_se_dde = D_dense.compute_self_energy(G_ppsc_dense, topology1);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto nca_se_dde_block = get_tensor_in_atom_diag_subspace(nca_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() - nca_se_dde_block)), eps);
  }
  ASSERT_LE(max_offdiag(nca_se_dde[0].data()), eps);

  // ----- OCA test -----
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  auto oca_se                  = D.compute_self_energy(G_ppsc, topology2);
  auto oca_se_ana              = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double t      = rel2abs(dlr_it(i)); // t = tau / beta
    oca_se_ana(i) = 0.25 * exp(-t * ln4) * t * t * beta * beta;
  }
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto oca_se_block = oca_se[b].data();
    for (int d = 0; d < oca_se_block.extent(1); ++d) {
      ASSERT_LE(nda::max_element(nda::abs(oca_se_block(_, d, d) - oca_se_ana)), eps);
    }
    ASSERT_LE(max_offdiag(oca_se_block), eps);
  }
  // compare to manual block-sparse OCA
  auto oca_se_manual = OCA_bs(D.hyb.values, D.hyb.poles, itops, beta, G_bdof, Fq);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    // Unlike NCA (odd order), OCA (even order) does not need the sign flip; the (-1) per hybridization
    // line in NCA_bs/OCA_bs cancels against the extra line at second order.
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_manual.get_block(b))), eps);
  }
  // compare to manual dense OCA
  auto oca_se_dense =
     OCA_dense(D.hyb.values, D.hyb.coeffs, D.hyb.values_reflect, D.hyb.coeffs, D.hyb.poles, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto oca_se_dense_block = get_tensor_in_atom_diag_subspace(oca_se_dense, b, ad);
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_dense_block)), eps);
  }
  ASSERT_LE(max_offdiag(oca_se_dense), eps);
  // compare to the dense diagram evaluator
  auto oca_se_dde = D_dense.compute_self_energy(G_ppsc_dense, topology2);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto oca_se_dde_block = get_tensor_in_atom_diag_subspace(oca_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_dde_block)), eps);
  }
  ASSERT_LE(max_offdiag(oca_se_dde[0].data()), eps);

  // ----- third-order test -----
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  auto third_order_se          = D.compute_self_energy(G_ppsc, topology3);
  auto third_order_se_ana      = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double t              = rel2abs(dlr_it(i)); // t = tau / beta
    third_order_se_ana(i) = -1.0 / 96 * exp(-t * ln4) * pow(t, 4) * pow(beta, 4);
  }
  // third_order_se has three blocks: 1x1, 2x2, and 1x1. Check that all diagonal entries equal the
  // analytical expression, and that the 2x2 block's off-diagonal entries vanish.
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto third_order_se_block = third_order_se[b].data();
    for (int d = 0; d < third_order_se_block.extent(1); ++d) {
      ASSERT_LE(nda::max_element(nda::abs(third_order_se_block(_, d, d) - third_order_se_ana)), eps);
    }
    ASSERT_LE(max_offdiag(third_order_se_block), eps);
  }
  // Check that 2x2 block is indeed 2x2
  ASSERT_EQ(third_order_se[1].data().extent(1), 2);
  // compare to the dense diagram evaluator
  auto third_order_se_dde = D_dense.compute_self_energy(G_ppsc_dense, topology3);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto third_order_se_dde_block = get_tensor_in_atom_diag_subspace(third_order_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(third_order_se[b].data() - third_order_se_dde_block)), eps);
  }
  ASSERT_LE(max_offdiag(third_order_se_dde[0].data()), eps);
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
  // compare to manual block-sparse NCA Green's function evaluator
  BlockDiagOpFun G_bdof(G_ppsc);
  std::vector<nda::array<dcomplex, 3>> G_refl_blocks;
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) { G_refl_blocks.push_back(nda::make_regular(itops.reflect(G_bdof.get_block(b)))); }
  nda::vector<int> G_zero_block_indices(G_bdof.get_num_block_cols());
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) { G_zero_block_indices(b) = G_bdof.get_zero_block_index(b); }
  BlockDiagOpFun G_bdof_refl(G_refl_blocks, G_zero_block_indices);
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);
  auto nca_gf_manual        = NCA_gf_bs(G_bdof, G_bdof_refl, Fq);
  ASSERT_LE(nda::max_element(nda::abs(nca_spgf - nca_gf_manual)), eps);
  // compare to manual dense NCA Green's function evaluator
  auto dlr_it_abs               = cppdlr::rel2abs(dlr_it);
  auto H_dense                  = get_full_h_atomic(ad);
  auto Gt_dense                 = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
  auto Gt_dense_refl            = itops.reflect(Gt_dense);
  auto [Fs_dense, F_dags_dense] = get_operators_dense(ad);
  auto nca_gf_dense             = NCA_gf_dense(Gt_dense, Gt_dense_refl, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(nca_spgf - nca_gf_dense)), eps);
  // compare to the dense diagram evaluator, which sums the same backbones as the block-sparse one but
  // over the full Hilbert space rather than block by block
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G_ppsc[0].mesh(), Gt_dense)};
  auto G_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G_dense_blocks);
  DenseDiagramEvaluator D_dense(hyb_poles, hyb_coeffs, G_ppsc[0].mesh(), ad);
  auto nca_spgf_dde = D_dense.compute_single_ptcle_gf(G_ppsc_dense, topology1);
  ASSERT_LE(nda::max_element(nda::abs(nca_spgf - nca_spgf_dde)), eps);

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
  // There are no spin-flip terms in H or Delta, so the spin off-diagonal blocks vanish identically
  ASSERT_EQ(oca_spgf.extent(1), 2);
  ASSERT_LE(max_offdiag(oca_spgf), eps);
  // compare to manual block-sparse OCA Green's function evaluator
  auto oca_gf_manual = OCA_gf_bs(D.hyb.poles, itops, beta, G_bdof, Fq);
  ASSERT_LE(nda::max_element(nda::abs(oca_spgf - oca_gf_manual)), eps);
  // compare to manual dense OCA Green's function evaluator
  auto oca_gf_dense = OCA_gf_dense(D.hyb.coeffs, D.hyb.coeffs, D.hyb.poles, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(oca_spgf - oca_gf_dense)), eps);
  // compare to the dense diagram evaluator
  auto oca_spgf_dde = D_dense.compute_single_ptcle_gf(G_ppsc_dense, topology2);
  ASSERT_LE(nda::max_element(nda::abs(oca_spgf - oca_spgf_dde)), eps);

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
  ASSERT_EQ(third_order_spgf.extent(1), 2);
  ASSERT_LE(max_offdiag(third_order_spgf), eps);
  // compare to the dense diagram evaluator
  auto third_order_spgf_dde = D_dense.compute_single_ptcle_gf(G_ppsc_dense, topology3);
  ASSERT_LE(nda::max_element(nda::abs(third_order_spgf - third_order_spgf_dde)), eps);
}

/**
 * @brief Test evaluation of the self-energy for a two-fermion system with an off-diagonal (Hermitian) hybridization
 *
 * @details Same model as two_fermions.const_hyb_se (U = mu = 0, a single hybridization pole at omega = 0, so that
 * Delta(tau) = -M/2 is tau-independent), except that the orbital-space amplitude matrix is the Hermitian
 * M = {{1, alpha}, {alpha, 1}} instead of I_2. The closed forms are derived in the "Off-diagonal (Hermitian)
 * hybridization" section of examples/two_fermion_analytical_solutions.ipynb. For Hermitian M the self-energy stays
 * diagonal in the Fock basis at every order (the off-diagonal entries an off-diagonal hybridization generates are
 * proportional to the antisymmetric part M_ud - M_du, which vanishes here), with the same entry in every occupation
 * sector. NCA is linear in Delta and only sees M_ll = 1, so it is unchanged from the diagonal case; OCA and third
 * order pick up alpha^2 corrections. Setting alpha = 0 reproduces the two_fermions.const_hyb_se references.
 */
TEST(two_fermions, hermitian_hyb_se) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-12;
  auto dlr_rf   = build_dlr_rf(Lambda, eps);
  auto itops    = imtime_ops(Lambda, dlr_rf);
  int r         = itops.rank();

  // One pole at zero gives the constant hybridization; the amplitude matrix is overwritten below
  auto two_fermion_model = two_fermion_model_helper(beta, Lambda, eps, 0.0, 0.0, 0.0);
  auto &hyb_coeffs       = two_fermion_model.hyb_coeffs;
  auto &hyb_poles        = two_fermion_model.hyb_poles;
  auto &ad               = two_fermion_model.ad;
  auto &G_ppsc           = two_fermion_model.G_ppsc;
  auto &G_bdof           = two_fermion_model.G_bdof;

  // Replace the identity amplitude matrix by the Hermitian M = {{1, alpha}, {alpha, 1}}
  double alpha        = 0.4;
  hyb_coeffs(0, 0, 1) = alpha;
  hyb_coeffs(0, 1, 0) = alpha;

  auto dlr_it = itops.get_itnodes();
  auto G0_ana = nda::zeros<double>(r);
  double ln4  = std::numbers::ln2 * 2;
  for (int i = 0; i < r; ++i) {
    double t  = rel2abs(dlr_it(i));
    G0_ana(i) = -exp(-t * ln4);
  }
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) { ASSERT_LE(nda::max_element(nda::abs(G_bdof.get_block(b)(_, 0, 0) - G0_ana)), eps); }

  DiagramEvaluator D(hyb_poles, hyb_coeffs, G_ppsc[0].mesh(), ad);

  // ----- NCA test -----
  nda::array<int, 2> topology1 = {{0, 1}};
  auto nca_se                  = D.compute_self_energy(G_ppsc, topology1);
  auto nca_se_ana              = nda::zeros<double>(r);
  nca_se_ana                   = -G0_ana; // independent of alpha
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto nca_se_block = nca_se[b].data();
    for (int d = 0; d < nca_se_block.extent(1); ++d) { ASSERT_LE(nda::max_element(nda::abs(nca_se_block(_, d, d) - nca_se_ana)), eps); }
    ASSERT_LE(max_offdiag(nca_se_block), eps);
  }
  // compare to manual block-sparse NCA
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);
  auto nca_se_manual        = NCA_bs(D.hyb.values, D.hyb.values_reflect, G_bdof, Fq);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    // NCA_bs and DiagramEvaluator::compute_self_energy use opposite overall sign conventions for the
    // self-energy (cf. the "-Sigma_Diagram_calc" negation in test_block_sparse_NCA_manual.cpp), hence "+".
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() + nca_se_manual.get_block(b))), eps);
  }
  // compare to manual dense NCA
  auto dlr_it_abs               = cppdlr::rel2abs(dlr_it);
  auto H_dense                  = get_full_h_atomic(ad);
  auto Gt_dense                 = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
  auto [Fs_dense, F_dags_dense] = get_operators_dense(ad);
  auto nca_se_dense             = NCA_dense(D.hyb.values, D.hyb.values_reflect, Gt_dense, Fs_dense, F_dags_dense);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto nca_se_dense_block = get_tensor_in_atom_diag_subspace(nca_se_dense, b, ad);
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() + nca_se_dense_block)), eps);
  }
  ASSERT_EQ(nca_se_dense.extent(1), 4);
  ASSERT_LE(max_offdiag(nca_se_dense), eps);
  // compare to the dense diagram evaluator, which sums the same backbones as the block-sparse one but
  // over the full Hilbert space, so its self-energy carries the same overall sign convention
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G_ppsc[0].mesh(), Gt_dense)};
  auto G_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G_dense_blocks);
  DenseDiagramEvaluator D_dense(hyb_poles, hyb_coeffs, G_ppsc[0].mesh(), ad);
  auto nca_se_dde = D_dense.compute_self_energy(G_ppsc_dense, topology1);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto nca_se_dde_block = get_tensor_in_atom_diag_subspace(nca_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() - nca_se_dde_block)), eps);
  }
  ASSERT_LE(max_offdiag(nca_se_dde[0].data()), eps);

  // ----- OCA test -----
  // The alpha = 0 value carries a factor -1; the Hermitian M replaces it by alpha^2 - 1
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  auto oca_se                  = D.compute_self_energy(G_ppsc, topology2);
  auto oca_se_ana              = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double tau    = beta * rel2abs(dlr_it(i));
    oca_se_ana(i) = 0.25 * (alpha * alpha - 1) * exp(-tau / beta * ln4) * tau * tau;
  }
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto oca_se_block = oca_se[b].data();
    for (int d = 0; d < oca_se_block.extent(1); ++d) { ASSERT_LE(nda::max_element(nda::abs(oca_se_block(_, d, d) - oca_se_ana)), eps); }
    ASSERT_LE(max_offdiag(oca_se_block), eps);
  }
  // compare to manual block-sparse OCA
  auto oca_se_manual = OCA_bs(D.hyb.values, D.hyb.poles, itops, beta, G_bdof, Fq);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    // Unlike NCA (odd order), OCA (even order) does not need the sign flip; the (-1) per hybridization
    // line in NCA_bs/OCA_bs cancels against the extra line at second order.
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_manual.get_block(b))), eps);
  }
  // compare to manual dense OCA
  auto oca_se_dense =
     OCA_dense(D.hyb.values, D.hyb.coeffs, D.hyb.values_reflect, D.hyb.coeffs, D.hyb.poles, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto oca_se_dense_block = get_tensor_in_atom_diag_subspace(oca_se_dense, b, ad);
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_dense_block)), eps);
  }
  ASSERT_EQ(oca_se_dense.extent(1), 4);
  ASSERT_LE(max_offdiag(oca_se_dense), eps);
  // compare to the dense diagram evaluator
  auto oca_se_dde = D_dense.compute_self_energy(G_ppsc_dense, topology2);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto oca_se_dde_block = get_tensor_in_atom_diag_subspace(oca_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_dde_block)), eps);
  }
  ASSERT_LE(max_offdiag(oca_se_dde[0].data()), eps);

  // ----- third-order test -----
  // The alpha = 0 value carries a factor 1/3 (relative to tau^4 / 32); the Hermitian M replaces it by alpha^2 + 1/3
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  auto third_order_se          = D.compute_self_energy(G_ppsc, topology3);
  auto third_order_se_ana      = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double tau            = beta * rel2abs(dlr_it(i));
    third_order_se_ana(i) = (3 * alpha * alpha + 1) / 96.0 * exp(-tau / beta * ln4) * pow(tau, 4);
  }
  // third_order_se has three blocks: 1x1, 2x2, and 1x1. Check that all diagonal entries equal the
  // analytical expression, and that the 2x2 block's off-diagonal entries vanish.
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto third_order_se_block = third_order_se[b].data();
    for (int d = 0; d < third_order_se_block.extent(1); ++d) {
      ASSERT_LE(nda::max_element(nda::abs(third_order_se_block(_, d, d) - third_order_se_ana)), eps);
    }
    ASSERT_LE(max_offdiag(third_order_se_block), eps);
  }
  ASSERT_EQ(third_order_se[1].data().extent(1), 2);
  // compare to the dense diagram evaluator
  auto third_order_se_dde = D_dense.compute_self_energy(G_ppsc_dense, topology3);
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) {
    auto third_order_se_dde_block = get_tensor_in_atom_diag_subspace(third_order_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(third_order_se[b].data() - third_order_se_dde_block)), eps);
  }
  ASSERT_LE(max_offdiag(third_order_se_dde[0].data()), eps);
}

/**
 * @brief Test evaluation of the single-particle Green's function for a two-fermion system with an off-diagonal
 * (Hermitian) hybridization
 *
 * @details Same model as two_fermions.hermitian_hyb_se: U = mu = 0 and Delta(tau) = -M/2 with the Hermitian
 * M = {{1, alpha}, {alpha, 1}}. Unlike the self-energy, the single-particle Green's function does acquire
 * off-diagonal orbital entries here, linear in alpha; the closed forms are derived in the "Off-diagonal (Hermitian)
 * hybridization" section of examples/two_fermion_analytical_solutions.ipynb. Setting alpha = 0 reproduces the
 * two_fermions.const_hyb_spgf references.
 */
TEST(two_fermions, hermitian_hyb_spgf) {
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

  double alpha        = 0.4;
  hyb_coeffs(0, 0, 1) = alpha;
  hyb_coeffs(0, 1, 0) = alpha;

  auto dlr_it = itops.get_itnodes();

  DiagramEvaluator D(hyb_poles, hyb_coeffs, G_ppsc[0].mesh(), ad);

  // ----- NCA test -----
  // First order has no hybridization line, so it does not see M at all
  nda::array<int, 2> topology1 = {{0, 1}};
  auto nca_spgf                = D.compute_single_ptcle_gf(G_ppsc, topology1);
  auto nca_spgf_ana            = nda::zeros<double>(r, 2, 2);
  nca_spgf_ana(_, 0, 0)        = 0.5;
  nca_spgf_ana(_, 1, 1)        = 0.5;
  ASSERT_LE(nda::max_element(nda::abs(nca_spgf - nca_spgf_ana)), eps);
  // compare to manual block-sparse NCA Green's function evaluator
  BlockDiagOpFun G_bdof(G_ppsc);
  std::vector<nda::array<dcomplex, 3>> G_refl_blocks;
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) { G_refl_blocks.push_back(nda::make_regular(itops.reflect(G_bdof.get_block(b)))); }
  nda::vector<int> G_zero_block_indices(G_bdof.get_num_block_cols());
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) { G_zero_block_indices(b) = G_bdof.get_zero_block_index(b); }
  BlockDiagOpFun G_bdof_refl(G_refl_blocks, G_zero_block_indices);
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);
  auto nca_gf_manual        = NCA_gf_bs(G_bdof, G_bdof_refl, Fq);
  ASSERT_LE(nda::max_element(nda::abs(nca_spgf - nca_gf_manual)), eps);
  // compare to manual dense NCA Green's function evaluator
  auto dlr_it_abs               = cppdlr::rel2abs(dlr_it);
  auto H_dense                  = get_full_h_atomic(ad);
  auto Gt_dense                 = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
  auto Gt_dense_refl            = itops.reflect(Gt_dense);
  auto [Fs_dense, F_dags_dense] = get_operators_dense(ad);
  auto nca_gf_dense             = NCA_gf_dense(Gt_dense, Gt_dense_refl, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(nca_spgf - nca_gf_dense)), eps);
  // compare to the dense diagram evaluator, which sums the same backbones as the block-sparse one but
  // over the full Hilbert space rather than block by block
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G_ppsc[0].mesh(), Gt_dense)};
  auto G_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G_dense_blocks);
  DenseDiagramEvaluator D_dense(hyb_poles, hyb_coeffs, G_ppsc[0].mesh(), ad);
  auto nca_spgf_dde = D_dense.compute_single_ptcle_gf(G_ppsc_dense, topology1);
  ASSERT_LE(nda::max_element(nda::abs(nca_spgf - nca_spgf_dde)), eps);

  // ----- OCA test -----
  // g_aa = tau (beta - tau) / 4, g_ab = -alpha tau (beta - tau) / 4
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  auto oca_spgf                = D.compute_single_ptcle_gf(G_ppsc, topology2);
  auto oca_spgf_ana            = nda::zeros<double>(r, 2, 2);
  for (int i = 0; i < r; ++i) {
    double tau            = beta * rel2abs(dlr_it(i));
    double diag           = 0.25 * tau * (beta - tau);
    oca_spgf_ana(i, 0, 0) = diag;
    oca_spgf_ana(i, 1, 1) = diag;
    oca_spgf_ana(i, 0, 1) = -alpha * diag;
    oca_spgf_ana(i, 1, 0) = -alpha * diag;
  }
  ASSERT_EQ(oca_spgf.extent(1), 2);
  ASSERT_LE(nda::max_element(nda::abs(oca_spgf - oca_spgf_ana)), eps);
  // compare to manual block-sparse OCA Green's function evaluator
  auto oca_gf_manual = OCA_gf_bs(D.hyb.poles, itops, beta, G_bdof, Fq);
  ASSERT_LE(nda::max_element(nda::abs(oca_spgf - oca_gf_manual)), eps);
  // compare to manual dense OCA Green's function evaluator
  auto oca_gf_dense = OCA_gf_dense(D.hyb.coeffs, D.hyb.coeffs, D.hyb.poles, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(oca_spgf - oca_gf_dense)), eps);
  // compare to the dense diagram evaluator
  auto oca_spgf_dde = D_dense.compute_single_ptcle_gf(G_ppsc_dense, topology2);
  ASSERT_LE(nda::max_element(nda::abs(oca_spgf - oca_spgf_dde)), eps);

  // ----- third-order test -----
  // g_aa = (alpha^2 + 1) tau^2 (beta - tau)^2 / 32, g_ab = alpha tau^2 (beta - tau)^2 / 16
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  auto third_order_spgf        = D.compute_single_ptcle_gf(G_ppsc, topology3);
  auto third_order_spgf_ana    = nda::zeros<double>(r, 2, 2);
  for (int i = 0; i < r; ++i) {
    double tau                    = beta * rel2abs(dlr_it(i));
    double base                   = tau * tau * (beta - tau) * (beta - tau);
    third_order_spgf_ana(i, 0, 0) = (alpha * alpha + 1) * base / 32.0;
    third_order_spgf_ana(i, 1, 1) = (alpha * alpha + 1) * base / 32.0;
    third_order_spgf_ana(i, 0, 1) = alpha * base / 16.0;
    third_order_spgf_ana(i, 1, 0) = alpha * base / 16.0;
  }
  ASSERT_EQ(third_order_spgf.extent(1), 2);
  ASSERT_LE(nda::max_element(nda::abs(third_order_spgf - third_order_spgf_ana)), eps);
  // compare to the dense diagram evaluator
  auto third_order_spgf_dde = D_dense.compute_single_ptcle_gf(G_ppsc_dense, topology3);
  ASSERT_LE(nda::max_element(nda::abs(third_order_spgf - third_order_spgf_dde)), eps);
}

/**
 * @brief Test evaluation of the self-energy for a two-fermion system with a hybridization formed from a single pole
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams,
 * in analogy with Backbone.one_fermion_three_orders_hyb_one_pole. As in the const_hyb tests above,
 * U = mu = 0 so that the pseudo-particle propagator is G(tau) = -4^{-tau/beta} I_4, but the
 * hybridization is Delta(tau) = K(tau, omega) I_2 with the helper's default pole omega = -1.5.
 * The analytic references are derived in examples/two_fermion_analytical_solutions.ipynb: the
 * self-energy is diagonal, with equal entries within each occupation sector N = 0, 1, 2, so each
 * block of the computed result is compared against the closed form for its sector (blocks are
 * matched to sectors by the particle number of their Fock states rather than by block order).
 */
TEST(two_fermions, one_hyb_pole_se) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-12;
  auto dlr_rf   = build_dlr_rf(Lambda, eps);
  auto itops    = imtime_ops(Lambda, dlr_rf);
  int r         = itops.rank();

  auto two_fermion_model = two_fermion_model_helper(beta, Lambda, eps, 0.0, 0.0);
  auto &hyb_coeffs       = two_fermion_model.hyb_coeffs;
  auto &hyb_poles        = two_fermion_model.hyb_poles;
  auto &ad               = two_fermion_model.ad;
  auto &G_ppsc           = two_fermion_model.G_ppsc;
  auto &G_bdof           = two_fermion_model.G_bdof;
  double om              = hyb_poles(0);

  // Check that G_ppsc is correct by comparing to analytical expression
  auto dlr_it = itops.get_itnodes();
  auto G0_ana = nda::zeros<double>(r);
  double ln4  = std::numbers::ln2 * 2;
  for (int i = 0; i < r; ++i) {
    double t  = rel2abs(dlr_it(i));
    G0_ana(i) = -exp(-t * ln4);
  }
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) { ASSERT_LE(nda::max_element(nda::abs(G_bdof.get_block(b)(_, 0, 0) - G0_ana)), eps); }

  // Occupation sector N of each block, used to index the analytic references below
  auto block_N = nda::zeros<int>(ad.n_subspaces());
  for (int b = 0; b < ad.n_subspaces(); ++b) { block_N(b) = __builtin_popcountl(ad.get_fock_states(b)[0]); }

  // Check that the maximum block dimension is 2
  int max_block_dim = 0;
  for (int b = 0; b < ad.n_subspaces(); ++b) { max_block_dim = std::max<int>(max_block_dim, ad.get_fock_states(b).size()); }
  ASSERT_EQ(max_block_dim, 2);

  // Set up diagram evaluator for self-energy evaluation
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G_ppsc[0].mesh(), ad);

  // ----- NCA test -----
  nda::array<int, 2> topology1 = {{0, 1}};
  auto nca_se                  = D.compute_self_energy(G_ppsc, topology1);
  auto nca_se_ana              = nda::zeros<double>(r, 3); // columns are occupation sectors N = 0, 1, 2
  for (int i = 0; i < r; ++i) {
    double t         = rel2abs(dlr_it(i)); // t = tau / beta
    double tau       = beta * t;
    double g4        = exp(-t * ln4);
    nca_se_ana(i, 0) = 2 * g4 * exp(om * tau) / (exp(beta * om) + 1);
    nca_se_ana(i, 2) = 2 * g4 * exp(-om * tau) / (exp(-beta * om) + 1);
    nca_se_ana(i, 1) = 0.5 * (nca_se_ana(i, 0) + nca_se_ana(i, 2));
  }
  for (int b = 0; b < ad.n_subspaces(); ++b) {
    for (int d = 0; d < nca_se[b].data().extent(1); ++d) {
      ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data()(_, d, d) - nca_se_ana(_, block_N(b)))), eps);
      for (int d2 = 0; d2 < nca_se[b].data().extent(2); ++d2) {
        if (d2 != d) { ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data()(_, d, d2))), eps); }
      }
    }
  }
  // compare to manual block-sparse NCA
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);
  auto nca_se_manual        = NCA_bs(D.hyb.values, D.hyb.values_reflect, G_bdof, Fq);
  for (int b = 0; b < ad.n_subspaces(); ++b) {
    // NCA_bs and DiagramEvaluator::compute_self_energy use opposite overall sign conventions for the
    // self-energy (cf. the "-Sigma_Diagram_calc" negation in test_block_sparse_NCA_manual.cpp), hence "+".
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() + nca_se_manual.get_block(b))), eps);
  }
  // compare to manual dense NCA
  auto dlr_it_abs               = cppdlr::rel2abs(dlr_it);
  auto H_dense                  = get_full_h_atomic(ad);
  auto Gt_dense                 = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
  auto [Fs_dense, F_dags_dense] = get_operators_dense(ad);
  auto nca_se_dense             = NCA_dense(D.hyb.values, D.hyb.values_reflect, Gt_dense, Fs_dense, F_dags_dense);
  for (int b = 0; b < ad.n_subspaces(); ++b) {
    auto nca_se_dense_block = get_tensor_in_atom_diag_subspace(nca_se_dense, b, ad);
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() + nca_se_dense_block)), eps);
  }
  ASSERT_EQ(nca_se_dense.extent(1), 4);
  ASSERT_LE(max_offdiag(nca_se_dense), eps);
  // compare to the dense diagram evaluator, which sums the same backbones as the block-sparse one but
  // over the full Hilbert space, so its self-energy carries the same overall sign convention
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G_ppsc[0].mesh(), Gt_dense)};
  auto G_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G_dense_blocks);
  DenseDiagramEvaluator D_dense(hyb_poles, hyb_coeffs, G_ppsc[0].mesh(), ad);
  auto nca_se_dde = D_dense.compute_self_energy(G_ppsc_dense, topology1);
  for (int b = 0; b < ad.n_subspaces(); ++b) {
    auto nca_se_dde_block = get_tensor_in_atom_diag_subspace(nca_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(nca_se[b].data() - nca_se_dde_block)), eps);
  }
  ASSERT_LE(max_offdiag(nca_se_dde[0].data()), eps);

  // ----- OCA test -----
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  auto oca_se                  = D.compute_self_energy(G_ppsc, topology2);
  auto oca_se_ana              = nda::zeros<double>(r, 3);
  double denom2                = om * om * (exp(beta * om) + 1) * (exp(beta * om) + 1);
  for (int i = 0; i < r; ++i) {
    double t         = rel2abs(dlr_it(i)); // t = tau / beta
    double tau       = beta * t;
    double tom       = om * tau;
    double g4        = exp(-t * ln4);
    oca_se_ana(i, 0) = -2 * g4 * (exp(tom) - tom - 1) * exp(tom) / denom2;
    oca_se_ana(i, 1) = -g4 * (exp(tom) - 1) * (exp(tom) - 1) * exp(om * (beta - tau)) / denom2;
    oca_se_ana(i, 2) = -2 * g4 * (tom * exp(tom) - exp(tom) + 1) * exp(2 * om * (beta - tau)) / denom2;
  }
  for (int b = 0; b < ad.n_subspaces(); ++b) {
    for (int d = 0; d < oca_se[b].data().extent(1); ++d) {
      ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data()(_, d, d) - oca_se_ana(_, block_N(b)))), eps);
      for (int d2 = 0; d2 < oca_se[b].data().extent(2); ++d2) {
        if (d2 != d) { ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data()(_, d, d2))), eps); }
      }
    }
  }
  // compare to manual block-sparse OCA
  auto oca_se_manual = OCA_bs(D.hyb.values, D.hyb.poles, itops, beta, G_bdof, Fq);
  for (int b = 0; b < ad.n_subspaces(); ++b) {
    // Unlike NCA (odd order), OCA (even order) does not need the sign flip; the (-1) per hybridization
    // line in NCA_bs/OCA_bs cancels against the extra line at second order.
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_manual.get_block(b))), eps);
  }
  // compare to manual dense OCA
  auto oca_se_dense =
     OCA_dense(D.hyb.values, D.hyb.coeffs, D.hyb.values_reflect, D.hyb.coeffs, D.hyb.poles, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  for (int b = 0; b < ad.n_subspaces(); ++b) {
    auto oca_se_dense_block = get_tensor_in_atom_diag_subspace(oca_se_dense, b, ad);
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_dense_block)), eps);
  }
  ASSERT_EQ(oca_se_dense.extent(1), 4);
  ASSERT_LE(max_offdiag(oca_se_dense), eps);
  // compare to the dense diagram evaluator
  auto oca_se_dde = D_dense.compute_self_energy(G_ppsc_dense, topology2);
  for (int b = 0; b < ad.n_subspaces(); ++b) {
    auto oca_se_dde_block = get_tensor_in_atom_diag_subspace(oca_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(oca_se[b].data() - oca_se_dde_block)), eps);
  }
  ASSERT_LE(max_offdiag(oca_se_dde[0].data()), eps);

  // ----- third-order test -----
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  auto third_order_se          = D.compute_self_energy(G_ppsc, topology3);
  auto third_order_se_ana      = nda::zeros<double>(r, 3);
  double denom3                = om * om * om * om * (exp(beta * om) + 1) * (exp(beta * om) + 1) * (exp(beta * om) + 1);
  for (int i = 0; i < r; ++i) {
    double t                 = rel2abs(dlr_it(i)); // t = tau / beta
    double tau               = beta * t;
    double tom               = om * tau;
    double g4                = exp(-t * ln4);
    third_order_se_ana(i, 0) = g4 * (tom * tom * exp(tom) - 4 * tom * exp(tom) - 2 * tom + 6 * exp(tom) - 6) * exp(beta * om) / denom3;
    third_order_se_ana(i, 2) = g4 * (tom * tom + 4 * tom + 2 * (tom - 3) * exp(tom) + 6) * exp(om * (2 * beta - tau)) / denom3;
    third_order_se_ana(i, 1) = 0.5 * (third_order_se_ana(i, 0) + third_order_se_ana(i, 2));
  }
  for (int b = 0; b < ad.n_subspaces(); ++b) {
    for (int d = 0; d < third_order_se[b].data().extent(1); ++d) {
      ASSERT_LE(nda::max_element(nda::abs(third_order_se[b].data()(_, d, d) - third_order_se_ana(_, block_N(b)))), eps);
      for (int d2 = 0; d2 < third_order_se[b].data().extent(2); ++d2) {
        if (d2 != d) { ASSERT_LE(nda::max_element(nda::abs(third_order_se[b].data()(_, d, d2))), eps); }
      }
    }
  }
  // compare to the dense diagram evaluator
  auto third_order_se_dde = D_dense.compute_self_energy(G_ppsc_dense, topology3);
  for (int b = 0; b < ad.n_subspaces(); ++b) {
    auto third_order_se_dde_block = get_tensor_in_atom_diag_subspace(third_order_se_dde[0].data(), b, ad);
    ASSERT_LE(nda::max_element(nda::abs(third_order_se[b].data() - third_order_se_dde_block)), eps);
  }
  ASSERT_LE(max_offdiag(third_order_se_dde[0].data()), eps);
}

/**
 * @brief Test evaluation of the single-particle Green's function for a two-fermion system with a
 * hybridization formed from a single pole
 *
 * @details This tests the evaluation of the first-, second-, and third-order single-particle Green's
 * function diagrams. As in Backbone.one_hyb_pole_se, U = mu = 0 so that the pseudo-particle propagator
 * is G(tau) = -4^{-tau/beta} I_4, and the hybridization is Delta(tau) = K(tau, omega) I_2 with the
 * helper's default pole omega = -1.5.
 */
TEST(two_fermions, one_hyb_pole_spgf) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-12;

  auto two_fermion_model = two_fermion_model_helper(beta, Lambda, eps, 0.0, 0.0);
  auto &hyb_coeffs       = two_fermion_model.hyb_coeffs;
  auto &hyb_poles        = two_fermion_model.hyb_poles;
  auto &ad               = two_fermion_model.ad;
  auto &G_ppsc           = two_fermion_model.G_ppsc;

  int norb = 2;

  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // compute single-particle Green's function for the two-fermion system with one hybridization pole
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G_ppsc[0].mesh(), ad);

  // ----- NCA test -----
  // First order has no hybridization line at all, so the reference is the same hybridization-independent
  // constant 1/2 seen in the const-hybridization test above.
  nda::array<int, 2> topology1 = {{0, 1}};
  auto nca_spgf                = D.compute_single_ptcle_gf(G_ppsc, topology1);
  auto nca_spgf_ana            = nda::zeros<double>(itops.rank(), 2, 2);
  nca_spgf_ana(_, 0, 0)        = 0.5;
  nca_spgf_ana(_, 1, 1)        = 0.5;
  ASSERT_LE(nda::max_element(nda::abs(nca_spgf - nca_spgf_ana)), eps);
  // compare to manual block-sparse NCA Green's function evaluator
  BlockDiagOpFun G_bdof(G_ppsc);
  std::vector<nda::array<dcomplex, 3>> G_refl_blocks;
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) { G_refl_blocks.push_back(nda::make_regular(itops.reflect(G_bdof.get_block(b)))); }
  nda::vector<int> G_zero_block_indices(G_bdof.get_num_block_cols());
  for (int b = 0; b < G_bdof.get_num_block_cols(); ++b) { G_zero_block_indices(b) = G_bdof.get_zero_block_index(b); }
  BlockDiagOpFun G_bdof_refl(G_refl_blocks, G_zero_block_indices);
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);
  auto nca_gf_manual        = NCA_gf_bs(G_bdof, G_bdof_refl, Fq);
  ASSERT_LE(nda::max_element(nda::abs(nca_spgf - nca_gf_manual)), eps);
  // compare to manual dense NCA Green's function evaluator
  auto dlr_it_abs               = cppdlr::rel2abs(itops.get_itnodes());
  auto H_dense                  = get_full_h_atomic(ad);
  auto Gt_dense                 = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
  auto Gt_dense_refl            = itops.reflect(Gt_dense);
  auto [Fs_dense, F_dags_dense] = get_operators_dense(ad);
  auto nca_gf_dense             = NCA_gf_dense(Gt_dense, Gt_dense_refl, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(nca_spgf - nca_gf_dense)), eps);
  // compare to the dense diagram evaluator, which sums the same backbones as the block-sparse one but
  // over the full Hilbert space rather than block by block
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G_ppsc[0].mesh(), Gt_dense)};
  auto G_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G_dense_blocks);
  DenseDiagramEvaluator D_dense(hyb_poles, hyb_coeffs, G_ppsc[0].mesh(), ad);
  auto nca_spgf_dde = D_dense.compute_single_ptcle_gf(G_ppsc_dense, topology1);
  ASSERT_LE(nda::max_element(nda::abs(nca_spgf - nca_spgf_dde)), eps);

  // ----- OCA test -----
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  auto spgf                    = D.compute_single_ptcle_gf(G_ppsc, topology2);
  // compare to reference values computed from the closed form derived in examples/two_fermion_analytical_solutions.ipynb
  auto hyb                        = triqs_xca::hyb::coefs2vals(beta, Lambda, eps, hyb_coeffs, hyb_poles);
  auto oca_spgf_coeffs            = itops.vals2coefs(spgf(_, 0, 0));
  std::vector<double> oca_tau_pts = {0.1, 0.3, 0.5, 0.7, 0.9};
  std::vector<double> oca_gf_ref  = {0.051177221707610, 0.110236319147384, 0.127756436679493, 0.110236319147384, 0.051177221707610};
  for (size_t k = 0; k < oca_tau_pts.size(); ++k) {
    dcomplex oca_val_up   = itops.coefs2eval(oca_spgf_coeffs, oca_tau_pts[k]);
    dcomplex oca_val_down = itops.coefs2eval(itops.vals2coefs(spgf(_, 1, 1)), oca_tau_pts[k]);
    ASSERT_LE(std::abs(oca_val_up - oca_gf_ref[k]), eps);
    ASSERT_LE(std::abs(oca_val_down - oca_gf_ref[k]), eps);
  }
  ASSERT_LE(nda::max_element(nda::abs(spgf(_, 0, 1))), eps);
  ASSERT_LE(nda::max_element(nda::abs(spgf(_, 1, 0))), eps);
  // compare to manual block-sparse OCA Green's function evaluator
  auto oca_gf_manual = OCA_gf_bs(D.hyb.poles, itops, beta, G_bdof, Fq);
  ASSERT_LE(nda::max_element(nda::abs(spgf - oca_gf_manual)), eps);
  // compare to manual dense OCA Green's function evaluator
  auto oca_gf_dense = OCA_gf_dense(D.hyb.coeffs, D.hyb.coeffs, D.hyb.poles, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(spgf - oca_gf_dense)), eps);
  // compare to the dense diagram evaluator
  auto oca_spgf_dde = D_dense.compute_single_ptcle_gf(G_ppsc_dense, topology2);
  ASSERT_LE(nda::max_element(nda::abs(spgf - oca_spgf_dde)), eps);

  // ----- third-order test -----
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  auto third_order_spgf        = D.compute_single_ptcle_gf(G_ppsc, topology3);
  auto third_order_spgf_coeffs = itops.vals2coefs(third_order_spgf(_, 0, 0));

  // Reference values for g_{up,up}(tau) at beta=2, omega=-1.5, computed from the closed form derived
  // in examples/two_fermion_analytical_solutions.ipynb
  std::vector<double> tau_pts = {0.1, 0.3, 0.5, 0.7, 0.9};
  std::vector<double> gf_ref  = {0.000393140721520, 0.003052337792819, 0.006393977070682, 0.006929916603937, 0.002036125469681};

  for (size_t k = 0; k < tau_pts.size(); ++k) {
    dcomplex gf_val_up   = itops.coefs2eval(third_order_spgf_coeffs, tau_pts[k]);
    dcomplex gf_val_down = itops.coefs2eval(itops.vals2coefs(third_order_spgf(_, 1, 1)), tau_pts[k]);
    ASSERT_LE(std::abs(gf_val_up - gf_ref[k]), eps);
    ASSERT_LE(std::abs(gf_val_down - gf_ref[k]), eps);
  }
  ASSERT_LE(nda::max_element(nda::abs(third_order_spgf(_, 0, 1))), eps);
  ASSERT_LE(nda::max_element(nda::abs(third_order_spgf(_, 1, 0))), eps);
  // compare to the dense diagram evaluator
  auto third_order_spgf_dde = D_dense.compute_single_ptcle_gf(G_ppsc_dense, topology3);
  ASSERT_LE(nda::max_element(nda::abs(third_order_spgf - third_order_spgf_dde)), eps);
}
