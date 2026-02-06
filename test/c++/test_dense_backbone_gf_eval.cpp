#include <gtest/gtest.h>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/block_sparse_manual_gf.hpp>
#include <triqs_xca/backbone.hpp>
#include <triqs_xca/dense_backbone.hpp>

TEST(DenseGFBackbone, NCA) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-6;

  // load Green's functions, field operators
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);
  auto [Deltat, Deltat_refl]              = discrete_bath_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  auto Gt_refl_dense = itops.reflect(Gt_dense);

  // compute Fbars and Fdagbars and store in Fset
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl_coeffs = itops.vals2coefs(Deltat_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs);

  // initialize Backbone and DiagramEvaluator
  nda::array<int, 2> topology = {{0, 1}};
  int n                       = 4;
  auto B                      = CorrelatorBackbone(topology, 4);
  auto C                      = DenseDiagramEvaluator(beta, itops, Deltat, Deltat_refl, dlr_rf, Gt_dense, Fset);
  auto gf                     = C.eval_correlator(B, Fs_dense, F_dags_dense);

  // compare against manually-computed NCA result
  auto NCA_gf_dense_result = NCA_gf_dense(Gt_dense, Gt_refl_dense, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(gf - NCA_gf_dense_result)), eps);

  // compare against old code
  nda::vector<double> dlr_rf_reflect = -dlr_rf;
  auto decomp                        = hyb_decomp(hyb_coeffs, dlr_rf, eps);
  auto decomp_reflect                = hyb_decomp(hyb_refl_coeffs, dlr_rf_reflect, eps);
  hyb_F Delta_F(16, itops.rank(), n), Delta_F_reflect(16, itops.rank(), n);
  Delta_F.update_inplace(decomp, dlr_it, Fs_dense, F_dags_dense);
  Delta_F_reflect.update_inplace(decomp_reflect, dlr_it, F_dags_dense, Fs_dense);
  nda::vector<int> fb = {1};
  auto NCA_gf_old     = G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, topology, Gt_dense, itops, beta, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(gf - NCA_gf_old)), eps);
}

TEST(DenseGFBackbone, OCA) {
  int n         = 4;
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  // load hybridization, Green's functions, field operators
  auto [Deltat, Deltat_refl]              = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  // compute Fbars and Fdagbars and store in Fset
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl        = Deltat;
  auto hyb_refl_coeffs = hyb_coeffs;
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs);

  // initialize Backbone and DiagramEvaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = CorrelatorBackbone(topology, n);
  auto C                      = DenseDiagramEvaluator(beta, itops, Deltat, hyb_refl, dlr_rf, Gt_dense, Fset);
  auto gf                     = C.eval_correlator(B, Fs_dense, F_dags_dense);

  // compare against manually-computed OCA result
  auto OCA_gf_dense_result = OCA_gf_dense(hyb_coeffs, hyb_refl_coeffs, dlr_rf, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(gf - OCA_gf_dense_result)), eps);

  // compare against old code
  nda::vector<double> dlr_rf_reflect = -dlr_rf;
  auto decomp                        = hyb_decomp(hyb_coeffs, dlr_rf, eps);
  auto decomp_reflect                = hyb_decomp(hyb_refl_coeffs, dlr_rf_reflect, eps);
  hyb_F Delta_F(16, itops.rank(), n), Delta_F_reflect(16, itops.rank(), n);
  Delta_F.update_inplace(decomp, dlr_it, Fs_dense, F_dags_dense);
  Delta_F_reflect.update_inplace(decomp_reflect, dlr_it, F_dags_dense, Fs_dense);
  nda::vector<int> fb = {1, 0};
  auto OCA_gf_old     = G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, topology, Gt_dense, itops, beta, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(gf - OCA_gf_old)), eps);
}

TEST(DenseBAckbone, OCA_semicircle_bath_aaa) {
  // DLR parameters
  double beta   = 8.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();
  auto dlr_it = itops.get_itnodes();

  // load Green's functions, field operators
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);

  // hybridization from AAA fitting
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

  // compute Fbars and Fdagbars and store in Fset
  auto Fset = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs);

  // initialize Backbone and DiagramEvaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = CorrelatorBackbone(topology, n);
  auto C                      = DenseDiagramEvaluator(beta, itops, hyb, hyb_refl, hyb_poles, Gt_dense, Fset);
  auto gf                     = C.eval_correlator(B, Fs_dense, F_dags_dense);

  // compare against manually-computed OCA result
  auto OCA_dense_gf_result = OCA_gf_dense(hyb_coeffs, hyb_refl_coeffs, hyb_poles, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(gf - OCA_dense_gf_result)), eps);

  // compare against old code
  nda::vector<double> hyb_poles_reflect = -hyb_poles;
  auto decomp                           = hyb_decomp(hyb_coeffs, hyb_poles, eps);
  auto decomp_reflect                   = hyb_decomp(hyb_refl_coeffs, hyb_poles_reflect, eps);
  hyb_F Delta_F(16, p, n), Delta_F_reflect(16, p, n);
  Delta_F.update_inplace(decomp, dlr_it, Fs_dense, F_dags_dense);
  Delta_F_reflect.update_inplace(decomp_reflect, dlr_it, F_dags_dense, Fs_dense);
  nda::vector<int> fb = {1, 0};
  auto OCA_gf_old     = G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, topology, Gt_dense, itops, beta, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(gf - OCA_gf_old)), eps);
}