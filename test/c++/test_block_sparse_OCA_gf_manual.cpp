#include <cppdlr/utils.hpp>
#include <gtest/gtest.h>
#include <itertools/range.hpp>
#include <nda/algorithms.hpp>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include <triqs_soehyb/block_sparse.hpp>
#include <triqs_soehyb/block_sparse_manual.hpp>
#include <triqs_soehyb/block_sparse_manual_gf.hpp>

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
  double D    = 1.0;
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

  std::cout << "dense = " << OCA_gf_eq << std::endl;
  std::cout << "analytic = " << OCA_gf_ana_eq << std::endl;
  std::cout << "tpz = " << OCA_gf_trap << std::endl;
  ASSERT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_ana)), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_gf_eq - OCA_gf_trap)), 1e-1 / (n_quad * n_quad));
}

TEST(DenseOCAGF, matrices) {
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
  auto Gt_refl       = Hmat_to_Gtmat(H, beta, -dlr_it_abs);

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

  // hybridization parameters
  double s = 0.5;
  double t = 1.0;
  nda::array<double, 1> e{-2.3 * t, 2.3 * t};

  // hybridization generation
  int r   = itops.rank();
  auto Jt = nda::array<dcomplex, 3>(r, 1, 1);
  for (int i = 0; i <= 1; i++) {
    for (int u = 0; u < r; u++) { Jt(u, 0, 0) += k_it(dlr_it_abs(u), e(i), beta); }
  }

  // orbital index order: do 0, do 1, up 0, up 1. same level <-> same parity index
  auto Deltat = nda::array<dcomplex, 3>(r, 4, 4);

  for (int i = 0; i < Deltat.extent(1); i++) {
    for (int j = i; j < Deltat.extent(2); j++) {
      if (i == j) {
        Deltat(_, i, j) = Jt(_, 0, 0);
      } else if ((i == 0 && j == 1) || (i == 1 && j == 0) || (i == 2 && j == 3) || (i == 3 && j == 2)) {
        Deltat(_, i, j) = s * Jt(_, 0, 0);
      }
    }
  }
  Deltat               = t * t * Deltat;
  auto Deltat_refl     = itops.reflect(Deltat);
  auto hyb_coeffs      = itops.vals2coefs(Deltat);
  auto hyb_refl_coeffs = itops.vals2coefs(Deltat_refl);

  auto OCA_gf = OCA_gf_dense(hyb_coeffs, hyb_refl_coeffs, dlr_rf, itops, beta, Gt, Fs, F_dags);

  // trapezoidal
  auto Gt_coeffs   = itops.vals2coefs(Gt);
  int n_quad       = 40;
  auto OCA_gf_trap = OCA_gf_tpz(hyb_coeffs, hyb_refl_coeffs, itops, beta, Gt_coeffs, Fs, n_quad);
  auto OCA_gf_eq   = eval_eq(itops, OCA_gf, n_quad);

  std::cout << "dense = " << OCA_gf_eq(_, 0, 0) << std::endl;
  std::cout << "tpz = " << OCA_gf_trap(_, 0, 0) << std::endl;
  std::cout << "ratio = " << nda::make_regular(OCA_gf_trap(_, 0, 0) / OCA_gf_eq(_, 0, 0)) << std::endl;
  ASSERT_LE(nda::max_element(nda::abs(OCA_gf_eq - OCA_gf_trap)), 15.0 / (n_quad * n_quad));
}