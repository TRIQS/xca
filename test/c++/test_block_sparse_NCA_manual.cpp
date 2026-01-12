#include <gtest/gtest.h>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/block_sparse_manual.hpp>

using namespace nda;

TEST(BlockSparseNCAManual, simple) {
  // set up arguments to block_sparse/NCA_bs()
  int N = 4;
  int r = 1;
  int n = 2;

  // set up hybridization
  nda::array<dcomplex, 3> hyb({r, n, n});
  nda::array<dcomplex, 3> hyb_refl({r, n, n});
  for (int t = 0; t < r; ++t) {
    hyb(t, 0, 0)      = 1;
    hyb(t, 1, 1)      = -1;
    hyb(t, 0, 1)      = -1;
    hyb(t, 1, 0)      = 4;
    hyb_refl(t, _, _) = nda::transpose(hyb(t, _, _));
  }

  // set up Green's function
  dcomplex mu = 0.2789;
  dcomplex U  = 1.01;
  dcomplex V  = 0.123;
  nda::array<dcomplex, 3> block0({r, 1, 1});
  nda::array<dcomplex, 3> block1({r, 2, 2});
  nda::array<dcomplex, 3> block2({r, 1, 1});
  for (int t = 0; t < r; ++t) {
    block0(t, 0, 0) = 0;
    block1(t, 0, 0) = mu;
    block1(t, 1, 1) = mu;
    block1(t, 0, 1) = V;
    block1(t, 1, 0) = V;
    block2(t, 0, 0) = 2 * mu + U;
  }
  std::vector<nda::array<dcomplex, 3>> Gt_blocks = {block0, block1, block2};
  nda::vector<int> zero_block_indices            = {-1, 0, 0};
  BlockDiagOpFun Gt(Gt_blocks, zero_block_indices);

  // set up annihilation operators
  nda::vector<int> block_indices_F = {-1, 0, 1};

  nda::array<dcomplex, 2> F_up_block0              = {{0}};
  nda::array<dcomplex, 2> F_up_block1              = {{1, 0}};
  nda::array<dcomplex, 2> F_up_block2              = {{0}, {1}};
  std::vector<nda::array<dcomplex, 2>> F_up_blocks = {F_up_block0, F_up_block1, F_up_block2};
  BlockOp F_up(block_indices_F, F_up_blocks);

  nda::array<dcomplex, 2> F_down_block0              = {{0}};
  nda::array<dcomplex, 2> F_down_block1              = {{0, 1}};
  nda::array<dcomplex, 2> F_down_block2              = {{-1}, {0}};
  std::vector<nda::array<dcomplex, 2>> F_down_blocks = {F_down_block0, F_down_block1, F_down_block2};
  BlockOp F_down(block_indices_F, F_down_blocks);

  std::vector<BlockOp> Fs   = {F_up, F_down};
  BlockDiagOpFun NCA_result = NCA_bs(hyb, hyb_refl, Gt, Fs);

  // compute NCA_result using dense storage

  nda::array<dcomplex, 3> Gt_dense({r, N, N});
  Gt_dense(0, 0, 0) = 0;
  Gt_dense(0, 1, 1) = mu;
  Gt_dense(0, 2, 2) = mu;
  Gt_dense(0, 1, 2) = V;
  Gt_dense(0, 2, 1) = V;
  Gt_dense(0, 3, 3) = 2 * mu + U;

  nda::array<dcomplex, 2> F_up_dense({N, N});
  F_up_dense(0, 1) = 1;
  F_up_dense(2, 3) = 1;

  nda::array<dcomplex, 2> F_down_dense({N, N});
  F_down_dense(0, 2) = 1;
  F_down_dense(1, 3) = -1;

  nda::array<dcomplex, 2> F_up_dag_dense   = nda::transpose(F_up_dense);
  nda::array<dcomplex, 2> F_down_dag_dense = nda::transpose(F_down_dense);

  auto NCA_result_dense = nda::zeros<dcomplex>(r, N, N);
  nda::array<dcomplex, 2> temp_dense({N, N});
  for (int t = 0; t < r; ++t) {
    // backward diagram
    temp_dense = matmul(F_up_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 0, 0) * matmul(temp_dense, F_up_dag_dense);
    temp_dense = matmul(F_up_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 0, 1) * matmul(temp_dense, F_down_dag_dense);
    temp_dense = matmul(F_down_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 1, 0) * matmul(temp_dense, F_up_dag_dense);
    temp_dense = matmul(F_down_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 1, 1) * matmul(temp_dense, F_down_dag_dense);

    // forward diagram
    temp_dense = matmul(F_up_dag_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 0, 0) * matmul(temp_dense, F_up_dense);
    temp_dense = matmul(F_up_dag_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 0, 1) * matmul(temp_dense, F_down_dense);
    temp_dense = matmul(F_down_dag_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 1, 0) * matmul(temp_dense, F_up_dense);
    temp_dense = matmul(F_down_dag_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 1, 1) * matmul(temp_dense, F_down_dense);
  }

  EXPECT_EQ(NCA_result.get_block(0)(_, 0, 0), NCA_result_dense(_, 0, 0));
  EXPECT_EQ(NCA_result.get_block(1), NCA_result_dense(_, range(1, 3), range(1, 3)));
  EXPECT_EQ(NCA_result.get_block(2)(_, 0, 0), NCA_result_dense(_, 3, 3));
}

TEST(BlockSparseNCAManual, single_exponential) {
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
  double D             = 0.03;
  auto Deltat          = nda::array<dcomplex, 3>(r, 1, 1);
  auto Deltat_refl     = nda::array<dcomplex, 3>(r, 1, 1);
  Deltat(_, 0, 0)      = exp(-D * dlr_it_abs * beta);
  Deltat_refl(_, 0, 0) = exp(D * dlr_it_abs * beta);

  // create Green's function
  double g                                       = -0.54;
  auto Gt_block                                  = nda::array<dcomplex, 3>(r, 1, 1);
  auto Gt_zero_block_index                       = nda::ones<int>(1);
  Gt_block(_, 0, 0)                              = exp(-g * dlr_it_abs * beta);
  std::vector<nda::array<dcomplex, 3>> Gt_blocks = {Gt_block};
  auto Gt                                        = BlockDiagOpFun(Gt_blocks, Gt_zero_block_index);

  // create annihilation operator
  auto F_block                                  = nda::ones<dcomplex>(1, 1);
  auto F_block_indices                          = nda::vector<int>(1);
  F_block_indices                               = 0;
  std::vector<nda::array<dcomplex, 2>> F_blocks = {F_block};
  auto F                                        = BlockOp(F_block_indices, F_blocks);
  std::vector<BlockOp> Fs                       = {F};

  BlockDiagOpFun NCA_result = NCA_bs(Deltat, Deltat_refl, Gt, Fs);
  auto NCA_ana              = nda::zeros<dcomplex>(r);
  for (int i = 0; i < r; i++) {
    auto tau   = dlr_it_abs(i);
    NCA_ana(i) = -exp(-(D + g) * tau) - exp((D - g) * tau);
  }

  EXPECT_LT(nda::linalg::norm((NCA_result.get_block(0)(_, 0, 0) - NCA_ana), std::numeric_limits<double>::infinity())
               / nda::linalg::norm(NCA_ana, std::numeric_limits<double>::infinity()),
            1.0e-12);
}

TEST(BlockSparseNCAManual, two_band_discrete_bath_bs_dense) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 100 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  // model setup
  auto [Deltat, Deltat_refl]              = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);
  auto hyb_coeffs                         = itops.vals2coefs(Deltat);
  auto hyb_refl_coeffs                    = itops.vals2coefs(Deltat_refl);
  auto [Gt, Fq, sym_set_labels]           = two_band_helper(beta, Lambda, eps, hyb_coeffs);

  // block-sparse NCA compuation
  auto NCA_result = NCA_bs(Deltat, Deltat_refl, Gt, Fq);

  // dense-matrix NCA computation
  auto NCA_dense_result = NCA_dense(Deltat, Deltat_refl, Gt_dense, Fs_dense, F_dags_dense);

  // old NCA computation
  nda::vector<double> dlr_rf_reflect = -dlr_rf;
  auto Delta_decomp                  = hyb_decomp(hyb_coeffs, dlr_rf, eps);              //decomposition of Delta(t) using DLR coefficient
  auto Delta_decomp_reflect          = hyb_decomp(hyb_refl_coeffs, dlr_rf_reflect, eps); // decomposition of Delta(-t) using DLR coefficient
  int dim                            = Deltat.shape(1);
  int r                              = itops.rank();
  hyb_F Delta_F(16, r, dim);
  hyb_F Delta_F_reflect(16, r, dim);
  Delta_F.update_inplace(Delta_decomp, dlr_it, Fs_dense, F_dags_dense); // Compression of Delta(t) and F, F_dag matrices
  Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F_dags_dense, Fs_dense);
  auto fb               = nda::vector<int64_t>(2);
  nda::array<int, 2> D1 = {{0, 1}}; // topology for NCA diagram evaluator
  auto NCA_old = -Sigma_Diagram_calc(Delta_F, Delta_F_reflect, D1, Deltat, Deltat_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense, fb, true);

  ASSERT_LE(nda::max_element(nda::abs(NCA_result.get_block(0) - NCA_dense_result(_, range(0, 4), range(0, 4)))), 10 * eps);
  ASSERT_LE(nda::max_element(nda::abs(NCA_result.get_block(1) - NCA_dense_result(_, range(4, 10), range(4, 10)))), 10 * eps);
  ASSERT_LE(nda::max_element(nda::abs(NCA_result.get_block(2) - NCA_dense_result(_, range(10, 11), range(10, 11)))), 10 * eps);
  ASSERT_LE(nda::max_element(nda::abs(NCA_result.get_block(3) - NCA_dense_result(_, range(11, 15), range(11, 15)))), 10 * eps);
  ASSERT_LE(nda::max_element(nda::abs(NCA_result.get_block(4) - NCA_dense_result(_, range(15, 16), range(15, 16)))), 10 * eps);

  ASSERT_LE(nda::max_element(nda::abs(NCA_dense_result - NCA_old)), eps);
}

TEST(BlockSparseNCAManual, two_band_semicircle_bath_dense_aaa) {
  // DLR parameters
  double beta   = 8.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();

  // model setup
  auto [Deltat, Deltat_refl]              = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);

  // precomputed AAA decomposition of hybridization
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
  auto NCA_result = NCA_dense(hyb, hyb_refl, Gt_dense, Fs_dense, F_dags_dense);

  // old NCA computation
  nda::vector<double> hyb_poles(p);
  hyb_poles                             = {-2.537191963500981,  1.7111725610238615, -1.514666605887425, 1.04941790134832,
                                           -0.7410379494142222, 0.3763525311836938, -0.1312888711963961};
  hyb_poles                             = hyb_poles * beta;
  nda::vector<double> hyb_poles_reflect = -hyb_poles;
  auto Delta_decomp                     = hyb_decomp(hyb_coeffs, hyb_poles, eps);              //decomposition of Delta(t) using DLR coefficient
  auto Delta_decomp_reflect             = hyb_decomp(hyb_refl_coeffs, hyb_poles_reflect, eps); // decomposition of Delta(-t) using DLR coefficient
  int dim                               = Deltat.shape(1);
  hyb_F Delta_F(16, p, dim);
  hyb_F Delta_F_reflect(16, p, dim);
  auto dlr_it = itops.get_itnodes();
  Delta_F.update_inplace(Delta_decomp, dlr_it, Fs_dense, F_dags_dense); // Compression of Delta(t) and F, F_dag matrices
  Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F_dags_dense, Fs_dense);
  auto fb               = nda::vector<int64_t>(2);
  fb(1)                 = 0;
  nda::array<int, 2> D1 = {{0, 1}}; // topology for OCA diagram evaluator
  auto NCA_old          = -Sigma_Diagram_calc(Delta_F, Delta_F_reflect, D1, hyb, hyb_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense, fb, true);

  ASSERT_LE(nda::max_element(nda::abs(NCA_result - NCA_old)), eps);
}
