#include <gtest/gtest.h>

#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/strong_cpl.hpp>

#include <triqs_xca/atom_diag_utils.hpp>

#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs_xca/block_sparse_manual.hpp>

#include "block_sparse_utils.hpp"

using nda::range;

using cppdlr::_;
using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;
using cppdlr::k_it;

using triqs_xca::block_sparse::BlockOp;

using triqs_xca::block_sparse::eval_eq;
using triqs_xca::block_sparse::OCA_dense;

TEST(BlockSparseOCAManual, single_exponential) {
  // DLR parameters
  double beta        = 1.0;
  double Lambda      = 100.0;
  double eps         = 1.0e-13;
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  int r              = itops.rank();

  auto dlr_it_abs = cppdlr::rel2abs(dlr_it);

  // create hybridization
  double D    = -10.0;
  auto Deltat = nda::array<dcomplex, 3>(r, 1, 1);
  // Deltat(_,0,0) = exp(-D*dlr_it_abs*beta);
  for (int t = 0; t < r; t++) Deltat(t, 0, 0) = k_it(dlr_it(t), D);

  // create Green's function
  double g                 = -13.0;
  auto Gt_block            = nda::array<dcomplex, 3>(r, 1, 1);
  auto Gt_zero_block_index = nda::ones<int>(1);
  for (int t = 0; t < r; t++) Gt_block(t, 0, 0) = k_it(dlr_it(t), g);
  std::vector<nda::array<dcomplex, 3>> Gt_blocks = {Gt_block};
  auto Gt                                        = BlockDiagOpFun(Gt_blocks, Gt_zero_block_index);

  // create annihilation operator
  auto F_block                                  = nda::ones<dcomplex>(1, 1);
  auto F_block_indices                          = nda::vector<int>(1);
  F_block_indices                               = 0;
  std::vector<nda::array<dcomplex, 2>> F_blocks = {F_block};
  auto F                                        = BlockOp(F_block_indices, F_blocks);
  std::vector<BlockOp> Fs                       = {F};

  auto OCA_result = OCA_bs(Deltat, itops, beta, Gt, Fs);
  auto OCA_ana    = nda::zeros<dcomplex>(r);
  for (int i = 0; i < r; i++) {
    auto tau = dlr_it_abs(i);
    // ff term
    OCA_ana(i) = exp(-g * (-3 + tau) - 2 * D * (-1 + tau)) * (1 + exp(D * tau) * (-1 + D * tau))
       / (D * D * (1 + exp(D)) * (1 + exp(D)) * (1 + exp(g)) * (1 + exp(g)) * (1 + exp(g)));
    // fb term
    OCA_ana(i) += exp(D + 3 * g - (D + g) * tau) * (-1 + exp(D * tau)) * (-1 + exp(D * tau))
       / (2 * D * D * (1 + exp(D)) * (1 + exp(D)) * (1 + exp(g)) * (1 + exp(g)) * (1 + exp(g)));
    // bf term
    OCA_ana(i) += exp(D + 3 * g - (D + g) * tau) * (-1 + exp(D * tau)) * (-1 + exp(D * tau))
       / (2 * D * D * (1 + exp(D)) * (1 + exp(D)) * (1 + exp(g)) * (1 + exp(g)) * (1 + exp(g)));
    // bb term
    OCA_ana(i) += -exp(-g * (-3 + tau) + D * tau) * (1 - exp(D * tau) + D * tau)
       / (D * D * (1 + exp(D)) * (1 + exp(D)) * (1 + exp(g)) * (1 + exp(g)) * (1 + exp(g)));
  }
  EXPECT_LT(nda::linalg::norm((OCA_result.get_block(0)(_, 0, 0) - OCA_ana), std::numeric_limits<double>::infinity()), 1.0e-7);
}

TEST(BlockSparseOCAManual, two_band_discrete_bath_bs) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 20 * beta;
  double eps    = 1.0e-4;

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // set up local Hamiltonian and field operators
  // model setup
  auto [Deltat, Deltat_refl]              = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);
  auto hyb_coeffs                         = itops.vals2coefs(Deltat);
  auto hyb_refl_coeffs                    = itops.vals2coefs(Deltat_refl);
  auto [Gt, Fq, sym_set_labels]           = two_band_helper(beta, Lambda, eps, hyb_coeffs);

  // block-sparse OCA computations
  auto OCA_result = OCA_bs(Deltat, dlr_rf, itops, beta, Gt, Fq);

  auto Deltadlr                            = itops.vals2coefs(Deltat); //obtain dlr coefficient of Delta(t)
  nda::vector<double> dlr_rf_reflect       = -dlr_rf;
  nda::array<dcomplex, 3> Deltadlr_reflect = Deltadlr * 1.0;
  for (int i = 0; i < Deltadlr.shape(0); ++i) Deltadlr_reflect(i, _, _) = transpose(Deltadlr(i, _, _));
  auto Delta_decomp         = hyb_decomp(Deltadlr, dlr_rf, eps);                 //decomposition of Delta(t) using DLR coefficient
  auto Delta_decomp_reflect = hyb_decomp(Deltadlr_reflect, dlr_rf_reflect, eps); // decomposition of Delta(-t) using DLR coefficient
  int dim                   = Deltat.shape(1);
  hyb_F Delta_F(16, r, dim);
  hyb_F Delta_F_reflect(16, r, dim);
  Delta_F.update_inplace(Delta_decomp, dlr_it, Fs_dense, F_dags_dense); // Compression of Delta(t) and F, F_dag matrices
  Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F_dags_dense, Fs_dense);
  nda::array<int, 2> D2 = {{0, 2}, {1, 3}}; // topology for OCA diagram evaluator
  auto fb               = nda::vector<int64_t>(2);
  fb                    = 0;
  auto OCA_forward  = Sigma_Diagram_calc(Delta_F, Delta_F_reflect, D2, Deltat, Deltat_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense, fb, true);
  fb(1)             = 1;
  auto OCA_backward = Sigma_Diagram_calc(Delta_F, Delta_F_reflect, D2, Deltat, Deltat_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense, fb, true);
  auto OCA_old      = nda::make_regular(-OCA_forward - OCA_backward);

  // check that block-sparse OCA calculation agrees with twoband.py

  auto ad = two_band_atom_diag_helper();
  for (int i = 0; i < OCA_result.get_num_block_cols(); i++) {
    auto result_dense_block = triqs_xca::atom_diag::get_tensor_in_atom_diag_subspace(OCA_old, i, ad);
    ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - result_dense_block)), 10 * eps);
  }
}

TEST(BlockSparseOCAManual, two_band_discrete_bath_bs_vs_dense) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-6;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  auto [Deltat, Deltat_refl]              = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);
  auto hyb_coeffs                         = itops.vals2coefs(Deltat);
  auto hyb_refl_coeffs                    = itops.vals2coefs(Deltat_refl);
  auto [Gt, Fq, sym_set_labels]           = two_band_helper(beta, Lambda, eps, hyb_coeffs);

  // dense-matrix OCA computation
  auto OCA_dense_result = OCA_dense(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  // block-sparse OCA computation
  auto OCA_bs_result = OCA_bs(Deltat, dlr_rf, itops, beta, Gt, Fq);

  auto ad = two_band_atom_diag_helper();
  for (int i = 0; i < OCA_bs_result.get_num_block_cols(); i++) {
    auto result_dense_block = triqs_xca::atom_diag::get_tensor_in_atom_diag_subspace(OCA_dense_result, i, ad);
    ASSERT_LE(nda::max_element(nda::abs(OCA_bs_result.get_block(i) - result_dense_block)), 10 * eps);
  }
}

TEST(BlockSparseOCAManual, two_band_semicircle_bath_aaa) {
  // DLR parameters
  double beta   = 8.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();

  // call two band helper just for Gt_dense, Fs_dense, F_dags_dense
  // model setup
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);

  // hybridization setup: precomputed values from AAA decomposition
  int p = 7;
  int n = 4;
  nda::vector<dcomplex> hyb_vals(r), hyb_coeff_vals(p);
  nda::array<dcomplex, 3> hyb(r, n, n), hyb_coeffs(p, n, n);
  hyb_vals       = {-0.4997496184487105, -0.4867352379479528, -0.4603465101833711, -0.4239204950540695, -0.3716597467714097,
                    -0.2884886574148449, -0.2479810727230272, -0.2065525284769785, -0.1635819676241178, -0.1326995066858671,
                    -0.1225444804140666, -0.1282199855712255, -0.1386184647087601, -0.1720919948804938, -0.2300400167898313,
                    -0.3000508284935615, -0.3759657450111002, -0.4545389745912252, -0.4821599768174421, -0.4997496184487105};
  hyb_coeff_vals = {0.0028042961182163, 0.088487039172428,  0.1575418229076625, 0.1953880665937937,
                    0.2145207908265103, 0.1832496441339733, 0.1580088741667851};
  for (int t = 0; t < r; t++) {
    hyb(t, 0, 0) = hyb_vals(t);
    hyb(t, 1, 1) = hyb_vals(t);
    hyb(t, 2, 2) = hyb_vals(t);
    hyb(t, 3, 3) = hyb_vals(t);
    hyb(t, 0, 1) = hyb_vals(t);
    hyb(t, 1, 0) = hyb_vals(t);
    hyb(t, 2, 3) = hyb_vals(t);
    hyb(t, 3, 2) = hyb_vals(t);
  }
  for (int l = 0; l < p; l++) {
    hyb_coeffs(l, 0, 0) = hyb_coeff_vals(l);
    hyb_coeffs(l, 1, 1) = hyb_coeff_vals(l);
    hyb_coeffs(l, 2, 2) = hyb_coeff_vals(l);
    hyb_coeffs(l, 3, 3) = hyb_coeff_vals(l);
    hyb_coeffs(l, 0, 1) = hyb_coeff_vals(l);
    hyb_coeffs(l, 1, 0) = hyb_coeff_vals(l);
    hyb_coeffs(l, 2, 3) = hyb_coeff_vals(l);
    hyb_coeffs(l, 3, 2) = hyb_coeff_vals(l);
  }
  auto hyb_refl = hyb;
  nda::array<dcomplex, 3> hyb_refl_coeffs(p, n, n);
  hyb_refl_coeffs = hyb_coeffs;

  nda::vector<double> hyb_poles(p);
  hyb_poles = {-2.537191963500981,  1.7111725610238615, -1.514666605887425, 1.04941790134832,
               -0.7410379494142222, 0.3763525311836938, -0.1312888711963961};
  hyb_poles = hyb_poles * beta;

  auto OCA_dense_result = OCA_dense(hyb, hyb_coeffs, hyb_refl, hyb_refl_coeffs, hyb_poles, itops, beta, Gt_dense, Fs_dense, F_dags_dense);

  // old OCA computation
  nda::vector<double> hyb_poles_reflect = -hyb_poles;
  auto Delta_decomp                     = hyb_decomp(hyb_coeffs, hyb_poles, eps);              //decomposition of Delta(t) using DLR coefficient
  auto Delta_decomp_reflect             = hyb_decomp(hyb_refl_coeffs, hyb_poles_reflect, eps); // decomposition of Delta(-t) using DLR coefficient
  hyb_F Delta_F(16, p, n);
  hyb_F Delta_F_reflect(16, p, n);
  auto dlr_it = itops.get_itnodes();
  Delta_F.update_inplace(Delta_decomp, dlr_it, Fs_dense, F_dags_dense); // Compression of Delta(t) and F, F_dag matrices
  Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F_dags_dense, Fs_dense);
  nda::array<int, 2> D2 = {{0, 2}, {1, 3}}; // topology for OCA diagram evaluator

  // Get Delta(t-t1) backward Delta(t2,t0) forward
  auto fb           = nda::vector<int64_t>(2);
  fb                = 0;
  auto OCA_forward  = Sigma_Diagram_calc(Delta_F, Delta_F_reflect, D2, hyb, hyb_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense, fb, true);
  fb(1)             = 1;
  auto OCA_backward = Sigma_Diagram_calc(Delta_F, Delta_F_reflect, D2, hyb, hyb_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense, fb, true);
  auto OCA_old      = nda::make_regular(-OCA_forward - OCA_backward);

  // check that dense OCA calculation agree with old calculation
  ASSERT_LE(nda::max_element(nda::abs(OCA_dense_result - OCA_old)), eps);

  // block-sparse OCA computation
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);
  auto OCA_bs_result            = OCA_bs(hyb, hyb_poles, itops, beta, Gt, Fq);

  auto ad = two_band_atom_diag_helper();
  for (int i = 0; i < OCA_bs_result.get_num_block_cols(); i++) {
    auto result_dense_block = triqs_xca::atom_diag::get_tensor_in_atom_diag_subspace(OCA_old, i, ad);
    ASSERT_LE(nda::max_element(nda::abs(OCA_bs_result.get_block(i) - result_dense_block)), 10 * eps);
  }
}

TEST(BlockSparseOCAManual, H5_two_band_discrete_bath_tpz) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-6;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  auto [Deltat, Deltat_refl]              = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);
  auto hyb_coeffs                         = itops.vals2coefs(Deltat);
  auto hyb_refl_coeffs                    = itops.vals2coefs(Deltat_refl);
  auto [Gt, Fq, sym_set_labels]           = two_band_helper(beta, Lambda, eps, hyb_coeffs);

  // block-sparse OCA compuation
  auto OCA_result = OCA_bs(Deltat, dlr_rf, itops, beta, Gt, Fq);

  int n_quad = 100;
  // load precomputed reference data
  h5::file tpz_file("h5/tpz100.ref.h5", 'r');
  nda::array<dcomplex, 3> OCA_tpz_result(101, 16, 16);
  h5::read(tpz_file, "OCA_tpz_result", OCA_tpz_result);

  // check that trapezoidal OCA calculation agrees with block-sparse calc.
  auto OCA_result_block_0    = OCA_result.get_block(0)(_, _, _);
  auto OCA_result_block_eq_0 = eval_eq(itops, OCA_result_block_0, n_quad);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result_block_eq_0 - OCA_tpz_result(_, range(0, 4), range(0, 4)))), 4e-4);

  auto OCA_result_block_1    = OCA_result.get_block(1)(_, _, _);
  auto OCA_result_block_eq_1 = eval_eq(itops, OCA_result_block_1, n_quad);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result_block_eq_1 - OCA_tpz_result(_, range(4, 10), range(4, 10)))), 4e-4);

  auto OCA_result_block_2    = OCA_result.get_block(2)(_, _, _);
  auto OCA_result_block_eq_2 = eval_eq(itops, OCA_result_block_2, n_quad);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result_block_eq_2 - OCA_tpz_result(_, range(10, 11), range(10, 11)))), 4e-4);

  auto OCA_result_block_3    = OCA_result.get_block(3)(_, _, _);
  auto OCA_result_block_eq_3 = eval_eq(itops, OCA_result_block_3, n_quad);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result_block_eq_3 - OCA_tpz_result(_, range(11, 15), range(11, 15)))), 4e-4);

  auto OCA_result_block_4    = OCA_result.get_block(4)(_, _, _);
  auto OCA_result_block_eq_4 = eval_eq(itops, OCA_result_block_4, n_quad);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result_block_eq_4 - OCA_tpz_result(_, range(15, 16), range(15, 16)))), 4e-4);
}
