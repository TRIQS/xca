#include <cppdlr/utils.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <itertools/range.hpp>
#include <nda/algorithms.hpp>
#include <nda/mapped_functions.hxx>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include "triqs_xca/strong_cpl.hpp"
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/block_sparse_manual.hpp>
#include <triqs_xca/block_sparse_manual_gf.hpp>

using namespace nda;

TEST(DenseOCAGF, single_exponential) {
  double beta        = 1.0;
  double Lambda      = 100.0;
  double eps         = 1e-8;
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  double omega = 0.1;
  nda::array<dcomplex, 3> Gt(r, 1, 1);
  for (int t = 0; t < r; ++t) { Gt(t, 0, 0) = k_it(dlr_it(t), omega); }

  // create hybridization
  double D    = -1.0; // 1.0;
  auto Deltat = nda::array<dcomplex, 3>(r, 1, 1);
  for (int t = 0; t < r; t++) Deltat(t, 0, 0) = k_it(dlr_it(t), D);
  auto hyb_coeffs      = itops.vals2coefs(Deltat);
  auto Deltat_refl     = itops.reflect(Deltat);
  auto hyb_refl_coeffs = itops.vals2coefs(Deltat_refl);

  int n = 1;
  nda::array<dcomplex, 3> Fs(n, 1, 1), F_dags(n, 1, 1);
  for (int i = 0; i < n; ++i) {
    Fs(i, 0, 0)     = 1.0;
    F_dags(i, 0, 0) = 1.0;
  }
  auto OCA_gf = OCA_gf_dense(hyb_coeffs, hyb_refl_coeffs, dlr_rf, itops, beta, Gt, Fs, F_dags);

  nda::array<dcomplex, 3> OCA_gf_ana(r, 1, 1);
  for (int t = 0; t < r; ++t) {
    double tau          = dlr_it_abs(t);
    OCA_gf_ana(t, 0, 0) = 2 * exp(3 * omega - tau * D) * (exp(tau * D) - exp(D)) * (exp(tau * D) - 1);
  }
  OCA_gf_ana = OCA_gf_ana / (D * D * (1 + exp(D)) * (1 + exp(omega)) * (1 + exp(omega)) * (1 + exp(omega)) * (1 + exp(omega)));

  // trapezoidal
  auto Gt_coeffs     = itops.vals2coefs(Gt);
  int n_quad         = 80;
  auto OCA_gf_trap   = OCA_gf_tpz(hyb_coeffs, hyb_refl_coeffs, itops, beta, Gt_coeffs, Fs, n_quad);
  auto OCA_gf_eq     = eval_eq(itops, OCA_gf, n_quad);
  auto OCA_gf_ana_eq = eval_eq(itops, OCA_gf_ana, n_quad);

  ASSERT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_ana)), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_gf_eq - OCA_gf_trap)), 1e-1 / (n_quad * n_quad));
}

TEST(DenseOCAGF, degen_matrices) {
  double beta        = 1.0;
  double Lambda      = 100.0;
  double eps         = 1e-8;
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  int n = 2, N = 4;
  double omega = 0.1;
  nda::array<dcomplex, 3> Gt(r, N, N);
  for (int t = 0; t < r; ++t) {
    for (int i = 0; i < N; ++i) { Gt(t, i, i) = k_it(dlr_it(t), omega); }
  }

  // create hybridization
  double D    = -1.0; // 1.0;
  auto Deltat = nda::array<dcomplex, 3>(r, n, n);
  for (int t = 0; t < r; ++t) {
    for (int i = 0; i < n; ++i) { Deltat(t, i, i) = k_it(dlr_it(t), D); }
  }
  auto hyb_coeffs      = itops.vals2coefs(Deltat);
  auto Deltat_refl     = itops.reflect(Deltat);
  auto hyb_refl_coeffs = itops.vals2coefs(Deltat_refl);

  nda::array<dcomplex, 3> Fs(n, N, N), F_dags(n, N, N);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < N; ++j) {
      Fs(i, j, j)     = 1.0;
      F_dags(i, j, j) = 1.0;
    }
  }
  auto OCA_gf = OCA_gf_dense(hyb_coeffs, hyb_refl_coeffs, dlr_rf, itops, beta, Gt, Fs, F_dags);

  nda::array<dcomplex, 3> OCA_gf_ana(r, n, n);
  for (int t = 0; t < r; ++t) {
    double tau = dlr_it_abs(t);
    for (int i = 0; i < n; ++i) { OCA_gf_ana(t, i, i) = 2 * n * N * exp(3 * omega - tau * D) * (exp(tau * D) - exp(D)) * (exp(tau * D) - 1); }
  }
  OCA_gf_ana = OCA_gf_ana / (D * D * (1 + exp(D)) * (1 + exp(omega)) * (1 + exp(omega)) * (1 + exp(omega)) * (1 + exp(omega)));

  // trapezoidal
  auto Gt_coeffs     = itops.vals2coefs(Gt);
  int n_quad         = 80;
  auto OCA_gf_trap   = OCA_gf_tpz(hyb_coeffs, hyb_refl_coeffs, itops, beta, Gt_coeffs, Fs, n_quad);
  auto OCA_gf_eq     = eval_eq(itops, OCA_gf, n_quad);
  auto OCA_gf_ana_eq = eval_eq(itops, OCA_gf_ana, n_quad);

  std::cout << "dense = " << nda::make_regular(nda::real(OCA_gf_eq(_, 0, 0))) << std::endl;
  std::cout << "\nanalytic = " << nda::make_regular(nda::real(OCA_gf_ana_eq(_, 0, 0))) << std::endl;
  std::cout << "\ntpz = " << nda::make_regular(nda::real(OCA_gf_trap(_, 0, 0))) << std::endl;
  std::cout << "\nanalytic / dense = " << nda::make_regular(nda::real(OCA_gf_ana_eq(_, 0, 0) / OCA_gf_eq(_, 0, 0))) << std::endl;
  std::cout << "\ntpz / dense = " << nda::make_regular(nda::real(OCA_gf_trap(_, 0, 0) / OCA_gf_eq(_, 0, 0))) << std::endl;
  ASSERT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_ana)), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_gf_eq - OCA_gf_trap)), 1e-1 / (n_quad * n_quad));
}

TEST(DenseOCAGF, identity_hyb) {
  int N = 4;

  // set up pseudoparticle Green's function
  dcomplex mu = 0.2789;
  dcomplex U  = 1.01;
  dcomplex V  = 0.123;

  nda::array<dcomplex, 2> H({N, N});
  H(0, 0) = 0;
  H(1, 1) = mu;
  H(2, 2) = mu;
  H(2, 1) = V;
  H(1, 2) = V;
  H(3, 3) = 2 * mu + U;

  double beta        = 2.0;
  double Lambda      = 10 * beta;
  double eps         = 1.0e-6;
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  auto Gt            = Hmat_to_Gtmat(H, beta, dlr_it_abs);

  // set up creation/annihilation operators
  int n = 2;
  nda::array<dcomplex, 3> Fs({n, N, N});
  Fs(0, 0, 1) = 1;
  Fs(0, 2, 3) = 1;
  Fs(1, 0, 2) = 1;
  Fs(1, 1, 3) = -1;

  nda::array<dcomplex, 3> F_dags({n, N, N});
  F_dags(0, 1, 0) = 1;
  F_dags(0, 3, 2) = 1;
  F_dags(1, 2, 0) = 1;
  F_dags(1, 3, 1) = -1;

  // hybridization generation
  int r       = itops.rank();
  auto Deltat = nda::array<dcomplex, 3>(r, n, n);
  for (int t = 0; t < r; ++t) { Deltat(t, _, _) = nda::eye<dcomplex>(n); }
  auto Deltat_refl     = itops.reflect(Deltat);
  auto hyb_coeffs      = itops.vals2coefs(Deltat);
  auto hyb_refl_coeffs = itops.vals2coefs(Deltat_refl);

  auto OCA_gf = OCA_gf_dense(hyb_coeffs, hyb_refl_coeffs, dlr_rf, itops, beta, Gt, Fs, F_dags);

  // trapezoidal
  auto Gt_coeffs   = itops.vals2coefs(Gt);
  int n_quad       = 40;
  auto OCA_gf_trap = OCA_gf_tpz(hyb_coeffs, hyb_refl_coeffs, itops, beta, Gt_coeffs, Fs, n_quad);
  auto OCA_gf_eq   = eval_eq(itops, OCA_gf, n_quad);

  ASSERT_LE(nda::max_element(nda::abs(OCA_gf_eq - OCA_gf_trap)), 15.0 / (n_quad * n_quad));
}

TEST(DenseOCAGF, two_band_discrete_bath_dense) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);
  int r       = itops.rank();
  Deltat_refl = itops.reflect(Deltat);

  auto hyb_coeffs      = itops.vals2coefs(Deltat);
  auto hyb_refl_coeffs = itops.vals2coefs(Deltat_refl);
  // hyb_refl_coeffs = hyb_coeffs;

  auto OCA_gf_result = OCA_gf_dense(hyb_coeffs, hyb_refl_coeffs, dlr_rf, itops, beta, Gt_dense, Fs_dense, F_dags_dense);

  // trapezoidal
  auto Gt_coeffs   = itops.vals2coefs(Gt_dense);
  int n_quad       = 10;
  auto OCA_gf_trap = OCA_gf_tpz(hyb_coeffs, hyb_refl_coeffs, itops, beta, Gt_coeffs, Fs_dense, n_quad);
  auto OCA_gf_eq   = eval_eq(itops, OCA_gf_result, n_quad);

  // Zhen
  auto Deltadlr                            = itops.vals2coefs(Deltat); //obtain dlr coefficient of Delta(t)
  nda::vector<double> dlr_rf_reflect       = -dlr_rf;
  nda::array<dcomplex, 3> Deltadlr_reflect = Deltadlr * 1.0;
  for (int i = 0; i < Deltadlr.shape(0); ++i) Deltadlr_reflect(i, _, _) = transpose(Deltadlr(i, _, _));
  auto Delta_decomp         = hyb_decomp(Deltadlr, dlr_rf, eps);                 //decomposition of Delta(t) using DLR coefficient
  auto Delta_decomp_reflect = hyb_decomp(Deltadlr_reflect, dlr_rf_reflect, eps); // decomposition of Delta(-t) using DLR coefficient
  int dim                   = static_cast<int>(Deltat.extent(1));
  // int r                     = itops.rank();
  hyb_F Delta_F(16, r, dim), Delta_F_reflect(16, r, dim);
  Delta_F.update_inplace(Delta_decomp, dlr_it, Fs_dense, F_dags_dense); // Compression of Delta(t) and F, F_dag matrices
  Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F_dags_dense, Fs_dense);
  nda::vector<int> fb      = {1, 0};
  nda::array<int, 2> D_OCA = {{0, 2}, {1, 3}};
  auto OCA_gf_Zhen         = G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D_OCA, Gt_dense, itops, beta, Fs_dense, F_dags_dense);
  auto OCA_gf_Zhen_eq = eval_eq(itops, OCA_gf_Zhen, n_quad);

  ASSERT_LE(nda::max_element(nda::abs(OCA_gf_eq - OCA_gf_trap)), 3.0 / (n_quad * n_quad));
  ASSERT_LE(nda::max_element(nda::abs(OCA_gf_eq - OCA_gf_Zhen_eq)), eps);
}

TEST(DenseOCAGF, two_band_semic_bath_dense) {
  double beta   = 8.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();

  // call two band helper just for Gt_dense, Fs_dense, F_dags_dense
  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  int p = 7;
  int n = 4;
  nda::array<dcomplex, 3> hyb(r, n, n), hyb_coeffs(p, n, n);
  hyb           = {{{-0.4997496184487105, -0.4997496184487105, -0., -0.},
                    {-0.4997496184487105, -0.4997496184487105, -0., -0.},
                    {-0., -0., -0.4997496184487105, -0.4997496184487105},
                    {-0., -0., -0.4997496184487105, -0.4997496184487105}},
                   {{-0.4867352379479528, -0.4867352379479528, -0., -0.},
                    {-0.4867352379479528, -0.4867352379479528, -0., -0.},
                    {-0., -0., -0.4867352379479528, -0.4867352379479528},
                    {-0., -0., -0.4867352379479528, -0.4867352379479528}},
                   {{-0.4603465101833711, -0.4603465101833711, -0., -0.},
                    {-0.4603465101833711, -0.4603465101833711, -0., -0.},
                    {-0., -0., -0.4603465101833711, -0.4603465101833711},
                    {-0., -0., -0.4603465101833711, -0.4603465101833711}},
                   {{-0.4239204950540695, -0.4239204950540695, -0., -0.},
                    {-0.4239204950540695, -0.4239204950540695, -0., -0.},
                    {-0., -0., -0.4239204950540695, -0.4239204950540695},
                    {-0., -0., -0.4239204950540695, -0.4239204950540695}},
                   {{-0.3716597467714097, -0.3716597467714097, -0., -0.},
                    {-0.3716597467714097, -0.3716597467714097, -0., -0.},
                    {-0., -0., -0.3716597467714097, -0.3716597467714097},
                    {-0., -0., -0.3716597467714097, -0.3716597467714097}},
                   {{-0.2884886574148449, -0.2884886574148449, -0., -0.},
                    {-0.2884886574148449, -0.2884886574148449, -0., -0.},
                    {-0., -0., -0.2884886574148449, -0.2884886574148449},
                    {-0., -0., -0.2884886574148449, -0.2884886574148449}},
                   {{-0.2479810727230272, -0.2479810727230272, -0., -0.},
                    {-0.2479810727230272, -0.2479810727230272, -0., -0.},
                    {-0., -0., -0.2479810727230272, -0.2479810727230272},
                    {-0., -0., -0.2479810727230272, -0.2479810727230272}},
                   {{-0.2065525284769785, -0.2065525284769785, -0., -0.},
                    {-0.2065525284769785, -0.2065525284769785, -0., -0.},
                    {-0., -0., -0.2065525284769785, -0.2065525284769785},
                    {-0., -0., -0.2065525284769785, -0.2065525284769785}},
                   {{-0.1635819676241178, -0.1635819676241178, -0., -0.},
                    {-0.1635819676241178, -0.1635819676241178, -0., -0.},
                    {-0., -0., -0.1635819676241178, -0.1635819676241178},
                    {-0., -0., -0.1635819676241178, -0.1635819676241178}},
                   {{-0.1326995066858671, -0.1326995066858671, -0., -0.},
                    {-0.1326995066858671, -0.1326995066858671, -0., -0.},
                    {-0., -0., -0.1326995066858671, -0.1326995066858671},
                    {-0., -0., -0.1326995066858671, -0.1326995066858671}},
                   {{-0.1225444804140666, -0.1225444804140666, -0., -0.},
                    {-0.1225444804140666, -0.1225444804140666, -0., -0.},
                    {-0., -0., -0.1225444804140666, -0.1225444804140666},
                    {-0., -0., -0.1225444804140666, -0.1225444804140666}},
                   {{-0.1282199855712255, -0.1282199855712255, -0., -0.},
                    {-0.1282199855712255, -0.1282199855712255, -0., -0.},
                    {-0., -0., -0.1282199855712255, -0.1282199855712255},
                    {-0., -0., -0.1282199855712255, -0.1282199855712255}},
                   {{-0.1386184647087601, -0.1386184647087601, -0., -0.},
                    {-0.1386184647087601, -0.1386184647087601, -0., -0.},
                    {-0., -0., -0.1386184647087601, -0.1386184647087601},
                    {-0., -0., -0.1386184647087601, -0.1386184647087601}},
                   {{-0.1720919948804938, -0.1720919948804938, -0., -0.},
                    {-0.1720919948804938, -0.1720919948804938, -0., -0.},
                    {-0., -0., -0.1720919948804938, -0.1720919948804938},
                    {-0., -0., -0.1720919948804938, -0.1720919948804938}},
                   {{-0.2300400167898313, -0.2300400167898313, -0., -0.},
                    {-0.2300400167898313, -0.2300400167898313, -0., -0.},
                    {-0., -0., -0.2300400167898313, -0.2300400167898313},
                    {-0., -0., -0.2300400167898313, -0.2300400167898313}},
                   {{-0.3000508284935615, -0.3000508284935615, -0., -0.},
                    {-0.3000508284935615, -0.3000508284935615, -0., -0.},
                    {-0., -0., -0.3000508284935615, -0.3000508284935615},
                    {-0., -0., -0.3000508284935615, -0.3000508284935615}},
                   {{-0.3759657450111002, -0.3759657450111002, -0., -0.},
                    {-0.3759657450111002, -0.3759657450111002, -0., -0.},
                    {-0., -0., -0.3759657450111002, -0.3759657450111002},
                    {-0., -0., -0.3759657450111002, -0.3759657450111002}},
                   {{-0.4545389745912252, -0.4545389745912252, -0., -0.},
                    {-0.4545389745912252, -0.4545389745912252, -0., -0.},
                    {-0., -0., -0.4545389745912252, -0.4545389745912252},
                    {-0., -0., -0.4545389745912252, -0.4545389745912252}},
                   {{-0.4821599768174421, -0.4821599768174421, -0., -0.},
                    {-0.4821599768174421, -0.4821599768174421, -0., -0.},
                    {-0., -0., -0.4821599768174421, -0.4821599768174421},
                    {-0., -0., -0.4821599768174421, -0.4821599768174421}},
                   {{-0.4997496184487105, -0.4997496184487105, -0., -0.},
                    {-0.4997496184487105, -0.4997496184487105, -0., -0.},
                    {-0., -0., -0.4997496184487105, -0.4997496184487105},
                    {-0., -0., -0.4997496184487105, -0.4997496184487105}}};
  hyb_coeffs    = {{{0.0028042961182163, 0.0028042961182163, 0., 0.},
                    {0.0028042961182163, 0.0028042961182163, 0., 0.},
                    {0., 0., 0.0028042961182163, 0.0028042961182163},
                    {0., 0., 0.0028042961182163, 0.0028042961182163}},
                   {{0.088487039172428, 0.088487039172428, 0., 0.},
                    {0.088487039172428, 0.088487039172428, 0., 0.},
                    {0., 0., 0.088487039172428, 0.088487039172428},
                    {0., 0., 0.088487039172428, 0.088487039172428}},
                   {{0.1575418229076625, 0.1575418229076625, 0., 0.},
                    {0.1575418229076625, 0.1575418229076625, 0., 0.},
                    {0., 0., 0.1575418229076625, 0.1575418229076625},
                    {0., 0., 0.1575418229076625, 0.1575418229076625}},
                   {{0.1953880665937937, 0.1953880665937937, 0., 0.},
                    {0.1953880665937937, 0.1953880665937937, 0., 0.},
                    {0., 0., 0.1953880665937937, 0.1953880665937937},
                    {0., 0., 0.1953880665937937, 0.1953880665937937}},
                   {{0.2145207908265103, 0.2145207908265103, 0., 0.},
                    {0.2145207908265103, 0.2145207908265103, 0., 0.},
                    {0., 0., 0.2145207908265103, 0.2145207908265103},
                    {0., 0., 0.2145207908265103, 0.2145207908265103}},
                   {{0.1832496441339733, 0.1832496441339733, 0., 0.},
                    {0.1832496441339733, 0.1832496441339733, 0., 0.},
                    {0., 0., 0.1832496441339733, 0.1832496441339733},
                    {0., 0., 0.1832496441339733, 0.1832496441339733}},
                   {{0.1580088741667851, 0.1580088741667851, 0., 0.},
                    {0.1580088741667851, 0.1580088741667851, 0., 0.},
                    {0., 0., 0.1580088741667851, 0.1580088741667851},
                    {0., 0., 0.1580088741667851, 0.1580088741667851}}};
  auto hyb_refl = nda::make_regular(hyb);
  nda::array<dcomplex, 3> hyb_refl_coeffs(p, n, n);
  hyb_refl_coeffs = nda::make_regular(hyb_coeffs);

  nda::vector<double> hyb_poles(p);
  hyb_poles = {-2.537191963500981,  1.7111725610238615, -1.514666605887425, 1.04941790134832,
               -0.7410379494142222, 0.3763525311836938, -0.1312888711963961};
  hyb_poles = hyb_poles * beta;

  auto OCA_gf_result = OCA_gf_dense(hyb_coeffs, hyb_refl_coeffs, hyb_poles, itops, beta, Gt_dense, Fs_dense, F_dags_dense);

  // trapezoidal
  auto Gt_coeffs           = itops.vals2coefs(Gt_dense);
  int n_quad               = 40;
  auto hyb_dlr_coeffs      = itops.vals2coefs(hyb);
  auto hyb_refl_dlr_coeffs = itops.vals2coefs(hyb_refl);
  auto OCA_gf_trap         = OCA_gf_tpz(hyb_dlr_coeffs, hyb_refl_dlr_coeffs, itops, beta, Gt_coeffs, Fs_dense, n_quad);
  auto OCA_gf_eq           = eval_eq(itops, OCA_gf_result, n_quad);

  ASSERT_LE(nda::max_element(nda::abs(OCA_gf_eq - OCA_gf_trap)), 20.0 / (n_quad * n_quad));
}

TEST(BSOCAGF, single_exponential) {
  double beta        = 1.0;
  double Lambda      = 100.0;
  double eps         = 1e-8;
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  double omega = 0.1;
  nda::array<dcomplex, 3> Gt_dense(r, 1, 1);
  for (int t = 0; t < r; ++t) { Gt_dense(t, 0, 0) = k_it(dlr_it(t), omega); }
  std::vector<nda::array<dcomplex, 3>> Gt_vec{Gt_dense};
  nda::vector<int> bi{0};
  BlockDiagOpFun Gt(Gt_vec, bi);

  // create hybridization
  double D    = -1.0; // 1.0;
  auto Deltat = nda::array<dcomplex, 3>(r, 1, 1);
  for (int t = 0; t < r; t++) Deltat(t, 0, 0) = k_it(dlr_it(t), D);
  auto hyb_coeffs      = itops.vals2coefs(Deltat);
  auto Deltat_refl     = itops.reflect(Deltat);
  auto hyb_refl_coeffs = itops.vals2coefs(Deltat_refl);

  int n = 1;
  nda::array<dcomplex, 3> Fs(n, 1, 1), F_dags(n, 1, 1);
  for (int i = 0; i < n; ++i) {
    Fs(i, 0, 0)     = 1.0;
    F_dags(i, 0, 0) = 1.0;
  }
  std::vector<nda::array<dcomplex, 3>> Fs_vec{Fs};
  BlockOpSymSet Fset(bi, Fs_vec);
  std::vector<nda::array<dcomplex, 3>> Fdags_vec{F_dags};
  BlockOpSymSet Fdagset(bi, Fdags_vec);
  std::vector<BlockOpSymSet> Fset_vec{Fset}, Fdagset_vec{Fdagset};
  nda::vector<long> sym_set_labels{0};
  BlockOpSymQuartet Fq(Fset_vec, Fdagset_vec, hyb_coeffs, hyb_refl_coeffs, sym_set_labels);

  auto OCA_gf = OCA_gf_bs(dlr_rf, itops, beta, Gt, Fq);

  nda::array<dcomplex, 3> OCA_gf_ana(r, 1, 1);
  for (int t = 0; t < r; ++t) {
    double tau          = dlr_it_abs(t);
    OCA_gf_ana(t, 0, 0) = 2 * exp(3 * omega - tau * D) * (exp(tau * D) - exp(D)) * (exp(tau * D) - 1);
  }
  OCA_gf_ana = OCA_gf_ana / (D * D * (1 + exp(D)) * (1 + exp(omega)) * (1 + exp(omega)) * (1 + exp(omega)) * (1 + exp(omega)));

  std::cout << "OCA_gf = " << nda::make_regular(nda::real(OCA_gf(_, 0, 0))) << std::endl;
  std::cout << "\nOCA_gf_ana = " << nda::make_regular(nda::real(OCA_gf_ana(_, 0, 0))) << std::endl;
  ASSERT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_ana)), eps);
}

TEST(BSOCAGF, two_band_discrete_bath_bs) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order, Fq] =
     two_band_discrete_bath_helper_sym(beta, Lambda, eps);
  Deltat_refl = itops.reflect(Deltat);

  auto hyb_coeffs      = itops.vals2coefs(Deltat);
  auto hyb_refl_coeffs = itops.vals2coefs(Deltat_refl);

  auto OCA_gf_result       = OCA_gf_bs(dlr_rf, itops, beta, Gt, Fq);
  auto OCA_gf_dense_result = OCA_gf_dense(hyb_coeffs, hyb_refl_coeffs, dlr_rf, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  std::cout << "bs = " << OCA_gf_result(_, 0, 0) << std::endl;
  std::cout << "\ndense = " << OCA_gf_dense_result(_, 0, 0) << std::endl;

  ASSERT_LE(nda::max_element(nda::abs(OCA_gf_result - OCA_gf_dense_result)), eps);
}
