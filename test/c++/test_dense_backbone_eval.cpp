#include <gtest/gtest.h>
#include <iomanip>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/block_sparse_manual.hpp>
#include <triqs_xca/backbone.hpp>
#include <triqs_xca/dense_backbone.hpp>

TEST(DenseBackbone, one_vertex_and_edge) {
  nda::array<int, 2> topology = {{0, 2}, {1, 4}, {3, 5}};
  int n = 4, N = 16;
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-6;

  auto [Deltat, Deltat_refl]              = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // create cre/ann operators
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(Deltat));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  auto D = DiagramEvaluator(beta, itops, Deltat, Deltat_refl, dlr_rf, Gt_dense, Fset);
  for (int fb1 = 0; fb1 <= 1; fb1++) {
    // initialize backbone
    auto B = Backbone(topology, n);

    // set line directions
    nda::vector<int> fb = {1, fb1, 0};
    B.set_directions(fb);

    // set pole indices
    nda::vector<int> pole_inds = {0, r - 1};
    B.set_pole_inds(pole_inds, dlr_rf);

    // set orbital indices
    nda::vector<int> orb_inds = {1, 0, 1, 2, 0, 2};
    B.set_orb_inds(orb_inds);

    // multiply T by vertex 1
    for (int t = 0; t < r; t++) D.T(t, _, _) = nda::eye<dcomplex>(N);
    D.multiply_vertex_dense(B, 1);

    // do the same multiplication manually
    nda::array<dcomplex, 3> Tact(r, N, N);
    if (fb1 == 1) {
      for (int t = 0; t < r; t++) Tact(t, _, _) = k_it(dlr_it(t), -dlr_rf(pole_inds(0))) * Fs_dense(0, _, _);
    } else {
      for (int t = 0; t < r; t++) Tact(t, _, _) = F_dags_dense(0, _, _);
    }
    ASSERT_LE(nda::max_element(nda::abs(D.T - Tact)), 1e-12);

    // check that convolution with function on first edge is correct
    D.compose_with_edge_dense(B, 1);
    if (fb1 == 1) {
      Tact = itops.convolve(beta, itops.vals2coefs(Gt_dense), itops.vals2coefs(Tact), TIME_ORDERED);
    } else {
      nda::array<dcomplex, 3> GKt_act(r, N, N);
      for (int t = 0; t < r; t++) GKt_act(t, _, _) = k_it(dlr_it(t), -dlr_rf(pole_inds(0))) * Gt_dense(t, _, _);
      Tact = itops.convolve(beta, itops.vals2coefs(GKt_act), itops.vals2coefs(Tact), TIME_ORDERED);
    }
    ASSERT_LE(nda::max_element(nda::abs(D.T - Tact)), 1e-12);
  }
}

TEST(DenseBackbone, OCA) {
  int n         = 4;
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  // load in functions from two_band.py
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
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  // initialize Backbone and DiagramEvaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, n);
  auto D                      = DiagramEvaluator(beta, itops, Deltat, hyb_refl, dlr_rf, Gt_dense, Fset);

  // evaluate OCA self-energy contribution
  D.eval_diagram_dense(B);
  auto OCA_result = D.Sigma;

  // compare against manually-computed OCA result
  auto OCA_dense_result = OCA_dense(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result - OCA_dense_result)), eps);

  // compute OCA using old code
  auto Deltadlr                            = itops.vals2coefs(Deltat); //obtain dlr coefficient of Delta(t)
  nda::vector<double> dlr_rf_reflect       = -dlr_rf;
  nda::array<dcomplex, 3> Deltadlr_reflect = Deltadlr * 1.0;
  for (int i = 0; i < Deltadlr.shape(0); ++i) Deltadlr_reflect(i, _, _) = transpose(Deltadlr(i, _, _));
  auto Delta_decomp         = hyb_decomp(Deltadlr, dlr_rf, eps);                 //decomposition of Delta(t) using DLR coefficient
  auto Delta_decomp_reflect = hyb_decomp(Deltadlr_reflect, dlr_rf_reflect, eps); // decomposition of Delta(-t) using DLR coefficient
  int dim                   = Deltat.shape(1);
  int r                     = itops.rank();
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

  ASSERT_LE(nda::max_element(nda::abs(OCA_result - OCA_old)), eps);
}

TEST(DenseBackbone, third_order_manual) {
  int n = 4, N = 16;
  double beta   = 2.0;
  double Lambda = 10.0 * beta; // 1000.0*beta;
  double eps    = 1.0e-10;

  auto [Deltat, Deltat_refl]              = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // compute hybridization function
  auto hyb             = Deltat;
  auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(hyb));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
  auto Fset            = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  // compute self-energy contribution of one third-order diagram topology,
  // with all forward hybridization lines and particular poles
  auto Sigma_manual           = third_order_dense_partial(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense);
  nda::array<int, 2> topology = {{0, 2}, {1, 4}, {3, 5}};
  auto B                      = Backbone(topology, n);
  nda::vector<int> fb{1, 1, 1}, pole_inds{7, 9};
  B.set_directions(fb);
  B.set_pole_inds(pole_inds, dlr_rf);
  auto D = DiagramEvaluator(beta, itops, Deltat, Deltat_refl, dlr_rf, Gt_dense, Fset);

  // perform the same calculation using the a routine called by eval_diagram_dense()
  nda::array<dcomplex, 3> T(r, N, N), GKt(r, N, N), Tmu(r, N, N), Sigma_generic(r, N, N);
  nda::array<dcomplex, 4> Tkaps(n, r, N, N);
  nda::vector<int> states(6);
  // f_ix = o_ix + n^(m-1) * p_ix + (n * r)^(m-1) * fb_ix
  int pow_n_mm1  = static_cast<int>(std::pow(n, B.m - 1));
  int pow_nr_mm1 = static_cast<int>(std::pow(n * r, B.m - 1));
  int f_ix_start = pow_n_mm1 * (pole_inds(0) + r * pole_inds(1)) + pow_nr_mm1 * (fb(0) + 2 * fb(1) + 4 * fb(2));
  for (int f_ix_off = 0; f_ix_off < pow_n_mm1; f_ix_off++) {
    B.set_flat_index(f_ix_start + f_ix_off, dlr_rf);
    D.eval_backbone_fixed_indices_dense(B);
    B.reset_all_inds();
  }

  ASSERT_LE(nda::max_element(nda::abs(Sigma_manual(10, _, _) - D.Sigma(10, _, _))), eps);
}

TEST(DenseBackbone, OCA_semicircle_bath_aaa) {
  // DLR parameters
  double beta   = 8.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();

  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);

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

  nda::vector<double> hyb_poles_reflect = -hyb_poles;
  auto Delta_decomp                     = hyb_decomp(hyb_coeffs, hyb_poles, eps);              //decomposition of Delta(t) using DLR coefficient
  auto Delta_decomp_reflect             = hyb_decomp(hyb_refl_coeffs, hyb_poles_reflect, eps); // decomposition of Delta(-t) using DLR coefficient
  hyb_F Delta_F(16, p, n);
  hyb_F Delta_F_reflect(16, p, n);
  auto dlr_it = itops.get_itnodes();
  Delta_F.update_inplace(Delta_decomp, dlr_it, Fs_dense, F_dags_dense); // Compression of Delta(t) and F, F_dag matrices
  Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F_dags_dense, Fs_dense);
  nda::array<int, 2> D2 = {{0, 2}, {1, 3}}; // topology for OCA diagram evaluator
  auto fb               = nda::vector<int64_t>(2);
  fb                    = 0;
  auto OCA_forward      = Sigma_Diagram_calc(Delta_F, Delta_F_reflect, D2, hyb, hyb_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense, fb, true);
  fb(1)                 = 1;
  auto OCA_backward     = Sigma_Diagram_calc(Delta_F, Delta_F_reflect, D2, hyb, hyb_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense, fb, true);
  auto OCA_old          = nda::make_regular(-OCA_forward - OCA_backward);

  // check that dense OCA calculation agree with old
  ASSERT_LE(nda::max_element(nda::abs(OCA_dense_result - OCA_old)), eps);

  // generic diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, n);
  auto Fset                   = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);
  auto D                      = DiagramEvaluator(beta, itops, hyb, hyb_refl, hyb_poles, Gt_dense, Fset);
  D.eval_diagram_dense(B);   // evaluate OCA diagram
  auto OCA_result = D.Sigma; // get the result from the DiagramEvaluator

  // compare with the dense result
  ASSERT_LE(nda::max_element(nda::abs(OCA_result - OCA_dense_result)), eps);
}
