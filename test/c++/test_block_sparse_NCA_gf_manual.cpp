#include <gtest/gtest.h>
#include <itertools/range.hpp>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/block_sparse_manual_gf.hpp>

using namespace nda;

TEST(DenseNCAGF, single_exponential) {
  double beta        = 1.0;
  double Lambda      = 10.0;
  double eps         = 1e-8;
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  int r              = itops.rank();

  double omega = 0.123;
  nda::array<dcomplex, 3> Gt(r, 1, 1), Gt_refl(r, 1, 1), Gt_refl2(r, 1, 1);
  for (int t = 0; t < r; ++t) {
    Gt(t, 0, 0)       = k_it(dlr_it(t), omega);
    Gt_refl(t, 0, 0)  = k_it(-dlr_it(t), omega);
    Gt_refl2(t, 0, 0) = k_it(dlr_it(t), -1 * omega);
  }
  ASSERT_LE(nda::max_element(nda::abs(Gt_refl - Gt_refl2)), 1e-12);

  int n = 1;
  nda::array<dcomplex, 3> Fs(n, 1, 1), F_dags(n, 1, 1);
  for (int i = 0; i < n; ++i) {
    Fs(i, 0, 0)     = 1.0;
    F_dags(i, 0, 0) = 1.0;
  }
  auto NCA_gf = NCA_gf_dense(Gt, Gt_refl, Fs, F_dags);

  nda::array<dcomplex, 3> NCA_gf_ana(r, 1, 1);
  for (int t = 0; t < r; ++t) { NCA_gf_ana(t, 0, 0) = 1 / (2 + exp(-beta * omega) + exp(beta * omega)); }

  ASSERT_LE(nda::max_element(nda::abs(NCA_gf - NCA_gf_ana)), eps);
}

TEST(DenseNCAGF, matrices) {
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

  auto NCA_gf = NCA_gf_dense(Gt, Gt_refl, Fs, F_dags);

  int r = itops.rank();
  nda::array<dcomplex, 3> NCA_gf_man(r, n, n);
  nda::array<dcomplex, 2> gf_temp(N, N);
  for (int lam = 0; lam < n; lam++) {
    for (int kap = 0; kap < n; kap++) {
      for (int t = 0; t < r; t++) {
        gf_temp = nda::matmul(Gt_refl(t, _, _), Fs(lam, _, _));
        gf_temp = nda::matmul(gf_temp, Gt(t, _, _));
        gf_temp                 = nda::matmul(gf_temp, F_dags(kap, _, _));
        NCA_gf_man(t, lam, kap) = nda::trace(gf_temp);
      }
    }
  }

  ASSERT_LE(nda::max_element(nda::abs(NCA_gf_man - NCA_gf)), 1e-12);
}

TEST(BSNCAGF, single_exponential) {
  double beta        = 1.0;
  double Lambda      = 10.0;
  double eps         = 1e-8;
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  int r              = itops.rank();

  double omega = 0.123;
  nda::array<dcomplex, 3> Gt_arr(r, 1, 1), Gt_refl_arr(r, 1, 1);
  for (int t = 0; t < r; ++t) {
    Gt_arr(t, 0, 0)      = k_it(dlr_it(t), omega);
    Gt_refl_arr(t, 0, 0) = k_it(-dlr_it(t), omega);
  }
  std::vector Gt_vec{Gt_arr}, Gt_refl_vec{Gt_refl_arr};
  nda::vector Gt_0bi{0}, Gt_refl_0bi{0};
  BlockDiagOpFun Gt(Gt_vec, Gt_0bi), Gt_refl(Gt_refl_vec, Gt_refl_0bi);

  int n = 1;
  nda::array<dcomplex, 3> Fs_block(n, 1, 1), F_dags_block(n, 1, 1);
  for (int i = 0; i < n; ++i) {
    Fs_block(i, 0, 0)     = 1.0;
    F_dags_block(i, 0, 0) = 1.0;
  }
  std::vector<nda::array<dcomplex, 3>> Fs_vec{Fs_block}, F_dags_vec{F_dags_block};
  nda::vector<int> bi{0};
  BlockOpSymSet Fs(bi, Fs_vec), F_dags(bi, F_dags_vec);
  std::vector<BlockOpSymSet> Fs_sets{Fs}, F_dags_sets{F_dags};
  nda::array<dcomplex, 3> hyb_coeffs(r, 1, 1), hyb_refl_coeffs(r, 1, 1);
  nda::vector<long> sym_set_labels{0};
  BlockOpSymQuartet Fq(Fs_sets, F_dags_sets, hyb_coeffs, hyb_refl_coeffs, sym_set_labels);

  auto NCA_gf = NCA_gf_bs(Gt, Gt_refl, Fq);

  nda::array<dcomplex, 3> NCA_gf_ana(r, 1, 1);
  for (int t = 0; t < r; ++t) { NCA_gf_ana(t, 0, 0) = 1 / (2 + exp(-beta * omega) + exp(beta * omega)); }

  ASSERT_LE(nda::max_element(nda::abs(NCA_gf - NCA_gf_ana)), eps);
}

TEST(BSNCAGF, matrices) {
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
  auto Gt_arr        = Hmat_to_Gtmat(H, beta, dlr_it_abs);
  auto Gt_refl_arr   = Hmat_to_Gtmat(H, beta, -dlr_it_abs);
  std::vector<nda::array<dcomplex, 3>> Gt_vec{Gt_arr(_, range(0, 1), range(0, 1)), Gt_arr(_, range(1, 3), range(1, 3)),
                                              Gt_arr(_, range(3, 4), range(3, 4))};
  std::vector<nda::array<dcomplex, 3>> Gt_refl_vec{Gt_refl_arr(_, range(0, 1), range(0, 1)), Gt_refl_arr(_, range(1, 3), range(1, 3)),
                                                   Gt_refl_arr(_, range(3, 4), range(3, 4))};
  nda::vector Gt_0bi{-1, 0, 0}, Gt_refl_0bi{-1, 0, 0};
  BlockDiagOpFun Gt(Gt_vec, Gt_0bi), Gt_refl(Gt_refl_vec, Gt_refl_0bi);

  // set up creation/annihilation operators
  int n = 2;
  nda::array<dcomplex, 3> Fs_block_1({n, 1, 2}), Fs_block_2({n, 2, 1}), F_dags_block_0({n, 2, 1}), F_dags_block_1({n, 1, 2});
  Fs_block_1(0, 0, 0) = 1;
  Fs_block_2(0, 1, 0) = 1;
  Fs_block_1(1, 0, 1) = 1;
  Fs_block_2(1, 0, 0) = -1;

  F_dags_block_0(0, 0, 0) = 1;
  F_dags_block_1(0, 1, 0) = 1;
  F_dags_block_0(1, 0, 1) = 1;
  F_dags_block_1(1, 0, 0) = -1;

  nda::array<dcomplex, 3> zero_block({1, 1, 1});
  zero_block(0, 0, 0) = 0.0;

  nda::vector<int> F_bi{-1, 0, 1}, F_dag_bi{1, 2, -1};
  std::vector<nda::array<dcomplex, 3>> Fs_vec{zero_block, Fs_block_1, Fs_block_2}, F_dags_vec{F_dags_block_0, F_dags_block_1, zero_block};
  BlockOpSymSet Fs{F_bi, Fs_vec}, F_dags{F_dag_bi, F_dags_vec};
  std::vector<BlockOpSymSet> Fs_sets{Fs}, F_dags_sets{F_dags};
  int r = itops.rank();
  nda::array<dcomplex, 3> hyb_coeffs(r, 1, 1), hyb_refl_coeffs(r, 1, 1);
  nda::vector<long> sym_set_labels{0, 0};
  BlockOpSymQuartet Fq(Fs_sets, F_dags_sets, hyb_coeffs, hyb_refl_coeffs, sym_set_labels);

  auto NCA_gf = NCA_gf_bs(Gt, Gt_refl, Fq);

  nda::array<dcomplex, 3> NCA_gf_man(r, n, n);
  nda::array<dcomplex, 2> gf_temp(N, N);
  for (int lam = 0; lam < n; lam++) {
    for (int kap = 0; kap < n; kap++) {
      for (int t = 0; t < r; t++) {
        gf_temp                 = nda::matmul(Gt_refl.get_block(1)(t, _, _), Fs.get_block(2)(lam, _, _));
        gf_temp                 = nda::matmul(gf_temp, Gt.get_block(2)(t, _, _));
        gf_temp                 = nda::matmul(gf_temp, F_dags.get_block(1)(kap, _, _));
        NCA_gf_man(t, lam, kap) = nda::trace(gf_temp);
      }
    }
  }
  ASSERT_LE(nda::max_element(nda::abs(NCA_gf_man - NCA_gf)), eps);
}