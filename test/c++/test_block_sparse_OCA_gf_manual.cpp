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
  for (int t = 0; t < r; ++t) {
    Gt(t, 0, 0)       = k_it(dlr_it(t), omega);
  }
  // std::cout << "Gt = " << Gt(_, 0, 0) << std::endl;

  // create hybridization
  double D    = 1.0;
  auto Deltat = nda::array<dcomplex, 3>(r, 1, 1);
  auto Deltat_test = nda::array<dcomplex, 3>(r, 1, 1);
  for (int t = 0; t < r; t++) Deltat(t, 0, 0) = k_it(dlr_it(t), D);
  for (int t = 0; t < r; t++) {
    if (dlr_it(t) >= 0) {
      Deltat_test(t, 0, 0) = k_it(1 - dlr_it(t), -D);
    } else {
      Deltat_test(t, 0, 0) = k_it(-dlr_it(t), -D);
    }
  }
  auto hyb_coeffs = itops.vals2coefs(Deltat);
  // std::cout << "Deltat = " << Deltat << "\n";
  // std::cout << "Deltat_test = " << Deltat_test << std::endl;
  // std::cout << "hyb_coeffs = " << hyb_coeffs << std::endl;

  int n = 1;
  nda::array<dcomplex, 3> Fs(n, 1, 1), F_dags(n, 1, 1);
  for (int i = 0; i < n; ++i) {
    Fs(i, 0, 0)     = 1.0;
    F_dags(i, 0, 0) = 1.0;
  }
  auto OCA_gf = OCA_gf_dense(hyb_coeffs, hyb_coeffs, dlr_rf, itops, beta, Gt, Fs, F_dags);

  nda::array<dcomplex, 3> OCA_gf_ana(r, 1, 1);
  for (int t = 0; t < r; ++t) {
    double tau = dlr_it_abs(t);
    // OCA_gf_ana(t, 0, 0) = (1 - exp(D)) * exp(3 * omega - tau * D) * (exp(tau * D) - exp(D)) * (exp(tau * D) - 1);

    // forward only
    if (abs(D) < 1e-16) {
      OCA_gf_ana(t, 0, 0) = -exp(3 * omega) * (1 - tau) * tau;
    } else {
      OCA_gf_ana(t, 0, 0) = exp(3 * omega - tau * D) * (exp(tau * D) - exp(D)) * (exp(tau * D) - 1);
    }
  }
  if (abs(D) < 1e-16)
    OCA_gf_ana = OCA_gf_ana / ((1 + exp(D)) * (1 + exp(omega)) * (1 + exp(omega)) * (1 + exp(omega)) * (1 + exp(omega)));
  else {
    OCA_gf_ana = OCA_gf_ana / (D * D * (1 + exp(D)) * (1 + exp(omega)) * (1 + exp(omega)) * (1 + exp(omega)) * (1 + exp(omega)));
  }

  nda::array<dcomplex, 2> OCA_gf_right(r, r);
  nda::array<dcomplex, 1> GKt(r);
  for (int l = 0; l < r; ++l) {
    if (dlr_rf(l) <= 0) {
      for (int t = 0; t < r; ++t) {
        GKt(t)     = k_it(dlr_it(t), omega) * k_it(dlr_it(t), -dlr_rf(l));
      }

      OCA_gf_right(l, _) = itops.convolve(beta, Fermion, itops.vals2coefs(Gt(_, 0, 0)), itops.vals2coefs(GKt), TIME_ORDERED);
    } else {
      for (int t = 0; t < r; ++t) {
        GKt(t)     = k_it(dlr_it(t), omega) * k_it(dlr_it(t), dlr_rf(l));
      }
      OCA_gf_right(l, _) = itops.convolve(beta, Fermion, itops.vals2coefs(GKt), itops.vals2coefs(Gt(_, 0, 0)), TIME_ORDERED);
    }
  }
  nda::array<dcomplex, 2> OCA_gf_left(r, r);
  nda::array<dcomplex, 3> left_temp(r, 1, 1);
  for (int l = 0; l < r; ++l) {
    
    if (dlr_rf(l) <= 0) {
      for (int t = 0; t < r; ++t) { GKt(t) = Gt(t, 0, 0) * k_it(dlr_it(t), -dlr_rf(l)); }
    } else {
      for (int t = 0; t < r; ++t) { GKt(t) = Gt(t, 0, 0) * k_it(dlr_it(t), dlr_rf(l)); }
    }
    
    // std::cout << "Gt in test = " << Gt(_, 0, 0) << std::endl;
    // std::cout << "GKt in test = " << GKt << std::endl;
    left_temp(_, 0, 0) = itops.convolve(beta, Fermion, itops.vals2coefs(Gt(_, 0, 0)), itops.vals2coefs(GKt), TIME_ORDERED);
    OCA_gf_left(l, _)  = itops.reflect(left_temp)(_, 0, 0);
  }

  // trapezoidal
  auto Gt_coeffs = itops.vals2coefs(Gt);
  int n_quad = 20;
  auto OCA_gf_trap = OCA_gf_tpz(hyb_coeffs, nda::make_regular(-hyb_coeffs), itops, beta, Gt_coeffs, Fs, n_quad);
  auto OCA_gf_eq = eval_eq(itops, OCA_gf, n_quad);
  auto OCA_gf_ana_eq = eval_eq(itops, OCA_gf_ana, n_quad);

  std::cout << "dense = " << OCA_gf_eq << std::endl;
  std::cout << "analytic = " << OCA_gf_ana_eq << std::endl;
  // std::cout << OCA_gf_right << std::endl;
  // std::cout << OCA_gf_left << std::endl;
  std::cout << "tpz = " << OCA_gf_trap << std::endl;
  // std::cout << nda::max_element(nda::abs(OCA_gf_trap - OCA_gf_eq)) << std::endl;
  // std::cout << nda::max_element(nda::abs(OCA_gf_trap - OCA_gf_ana_eq)) << std::endl;
  std::cout << "ratio dense : analytic = " << nda::make_regular(OCA_gf_eq / OCA_gf_ana_eq) << std::endl;
  ASSERT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_ana)), eps);
}