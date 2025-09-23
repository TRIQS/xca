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

  // create hybridization
  double D    = 1.0;
  auto Deltat = nda::array<dcomplex, 3>(r, 1, 1);
  for (int t = 0; t < r; t++) Deltat(t, 0, 0) = k_it(dlr_it(t), D);
  auto hyb_coeffs = itops.vals2coefs(Deltat);
  auto Deltat_refl = itops.reflect(Deltat);
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
    double tau = dlr_it_abs(t);
    OCA_gf_ana(t, 0, 0) = 2 * exp(3 * omega - tau * D) * (exp(tau * D) - exp(D)) * (exp(tau * D) - 1);
  }
  OCA_gf_ana = OCA_gf_ana / (D * D * (1 + exp(D)) * (1 + exp(omega)) * (1 + exp(omega)) * (1 + exp(omega)) * (1 + exp(omega)));

  // trapezoidal
  auto Gt_coeffs = itops.vals2coefs(Gt);
  int n_quad = 80;
  auto OCA_gf_trap = OCA_gf_tpz(hyb_coeffs, hyb_refl_coeffs, itops, beta, Gt_coeffs, Fs, n_quad);
  auto OCA_gf_eq = eval_eq(itops, OCA_gf, n_quad);
  auto OCA_gf_ana_eq = eval_eq(itops, OCA_gf_ana, n_quad);

  std::cout << "dense = " << OCA_gf_eq << std::endl;
  std::cout << "analytic = " << OCA_gf_ana_eq << std::endl;
  std::cout << "tpz = " << OCA_gf_trap << std::endl;
  ASSERT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_ana)), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_gf_eq - OCA_gf_trap)), 1e-1 / (n_quad * n_quad));
}