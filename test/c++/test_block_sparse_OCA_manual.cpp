#include <cppdlr/dlr_kernels.hpp>
#include <gtest/gtest.h>
#include <nda/basic_functions.hpp>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include <set>
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/block_sparse_manual.hpp>
#include <triqs_xca/backbone.hpp>
#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <iomanip>

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
  EXPECT_LT(nda::norm((OCA_result.get_block(0)(_, 0, 0) - OCA_ana), std::numeric_limits<double>::infinity()), 1.0e-7);
}

TEST(BlockSparseOCAManual, H5_two_band_discrete_bath_bs) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 100 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // block-sparse NCA and OCA computations
  auto OCA_result = OCA_bs(Deltat, itops, beta, Gt, Fs);
  // load NCA and OCA results from twoband.py
  h5::file Gtfile("h5/two_band_py.ref.h5", 'r');
  h5::group Gtgroup(Gtfile);
  auto NCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "NCA", NCA_py);
  auto OCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "OCA", OCA_py);

  // permute twoband.py results to match block structure from atom_diag
  auto NCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  auto OCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
        NCA_py_perm(t, i, j) = NCA_py(t, fock_state_order[i], fock_state_order[j]);
        OCA_py_perm(t, i, j) = OCA_py(t, fock_state_order[i], fock_state_order[j]);
      }
    }
  }

  // check that block-sparse OCA calculation agrees with twoband.py
  int s0 = 0;
  int s1 = subspaces[0].size();
  for (int i = 0; i < num_blocks; i++) { // compare each block
    ASSERT_LE(nda::max_element(
                 nda::abs(OCA_result.get_block(i) - OCA_py_perm(_, range(s0, s1), range(s0, s1)) + NCA_py_perm(_, range(s0, s1), range(s0, s1)))),
              eps);
    s0 = s1;
    if (i < num_blocks - 1) s1 += subspaces[i + 1].size();
  }
}

TEST(BlockSparseOCAManual, H5_two_band_discrete_bath_dense) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // dense-matrix OCA computation
  auto OCA_dense_result = OCA_dense(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense);

  // load NCA and OCA results from twoband.py
  h5::file Gtfile("h5/two_band_py.ref.h5", 'r');
  h5::group Gtgroup(Gtfile);
  auto NCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "NCA", NCA_py);
  auto OCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "OCA", OCA_py);

  // permute twoband.py results to match block structure from atom_diag
  auto NCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  auto OCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
        NCA_py_perm(t, i, j) = NCA_py(t, fock_state_order[i], fock_state_order[j]);
        OCA_py_perm(t, i, j) = OCA_py(t, fock_state_order[i], fock_state_order[j]);
      }
    }
  }

  // check that dense OCA calculation agree with twoband.py
  ASSERT_LE(nda::max_element(nda::abs(OCA_dense_result - OCA_py_perm + NCA_py_perm)), eps);
}

TEST(BlockSparseOCAManual, H5_two_band_semicircle_bath_aaa) {
  std::cout << std::setprecision(16);
  // DLR parameters
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
  auto hyb_refl = nda::make_regular(-hyb);
  nda::array<dcomplex, 3> hyb_refl_coeffs(p, n, n);
  hyb_refl_coeffs = nda::make_regular(hyb_coeffs);

  nda::vector<double> hyb_poles(p);
  hyb_poles = {-2.537191963500981,  1.7111725610238615, -1.514666605887425, 1.04941790134832,
               -0.7410379494142222, 0.3763525311836938, -0.1312888711963961};
  hyb_poles = hyb_poles * beta;

  auto OCA_dense_result = OCA_dense(hyb, hyb_coeffs, hyb_refl, hyb_refl_coeffs, hyb_poles, itops, beta, Gt_dense, Fs_dense, F_dags_dense);

  // load NCA and OCA results from twoband.py
  h5::file Gtfile("h5/two_band_py_semic.ref.h5", 'r');
  h5::group Gtgroup(Gtfile);
  auto NCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "NCA", NCA_py);
  auto OCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "OCA", OCA_py);

  // permute twoband.py results to match block structure from atom_diag
  auto NCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  auto OCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) {
        NCA_py_perm(t, i, j) = NCA_py(t, fock_state_order[i], fock_state_order[j]);
        OCA_py_perm(t, i, j) = OCA_py(t, fock_state_order[i], fock_state_order[j]);
      }
    }
  }

  // check that dense OCA calculation agree with twoband.py
  ASSERT_LE(nda::max_element(nda::abs(OCA_dense_result - OCA_py_perm + NCA_py_perm)), eps);

  // block-sparse NCA and OCA computations
  auto OCA_bs_result = OCA_bs(hyb, hyb_coeffs, hyb_refl, hyb_refl_coeffs, hyb_poles, itops, beta, Gt, Fs);

  // check that block-sparse OCA calculation agrees with twoband.py
  int s0 = 0;
  int s1 = subspaces[0].size();
  for (int i = 0; i < num_blocks; i++) { // compare each block
    ASSERT_LE(nda::max_element(
                 nda::abs(OCA_bs_result.get_block(i) - OCA_py_perm(_, range(s0, s1), range(s0, s1)) + NCA_py_perm(_, range(s0, s1), range(s0, s1)))),
              eps);
    s0 = s1;
    if (i < num_blocks - 1) s1 += subspaces[i + 1].size();
  }
}

TEST(BlockSparseOCAManual, two_band_discrete_bath_bs_vs_dense) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 100 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // block-sparse OCA computation
  auto OCA_result = OCA_bs(Deltat, itops, beta, Gt, Fs);

  // dense-matrix OCA computation
  auto OCA_dense_result = OCA_dense(Deltat, itops, beta, Gt_dense, Fs_dense, F_dags_dense);

  // check that block-sparse OCA calculation agrees with dense OCA calculation
  int s0 = 0;
  int s1 = subspaces[0].size();
  for (int i = 0; i < num_blocks; i++) { // compare each block
    ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - OCA_dense_result(_, range(s0, s1), range(s0, s1)))), eps);
    s0 = s1;
    if (i < num_blocks - 1) s1 += subspaces[i + 1].size();
  }
}

TEST(BlockSparseOCAManual, H5_two_band_discrete_bath_tpz) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 1000 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  // block-sparse OCA compuation
  auto OCA_result = OCA_bs(Deltat, itops, beta, Gt, Fs);

  int n_quad = 100;
  // compute OCA using trapezoidal rule using 100 quadrature nodes0
  // load precomputed values from the following 3 lines:
  // auto OCA_tpz_result = OCA_tpz(Deltat, itops, beta, Gt_dense, Fs_dense, n_quad);
  // h5::file tpz_file("h5/tpz100.ref.h5", 'w');
  // h5::write(tpz_file, "OCA_tpz_result", OCA_tpz_result);
  h5::file tpz_file("h5/tpz100.ref.h5", 'r');
  nda::array<dcomplex, 3> OCA_tpz_result(101, 16, 16);
  h5::read(tpz_file, "OCA_tpz_result", OCA_tpz_result);

  // check that trapezoidal OCA calculation agrees with block-sparse calc.
  int s0 = 0;
  int s1 = subspaces[0].size();
  for (int i = 0; i < num_blocks; i++) { // compare each block
    auto OCA_result_block    = OCA_result.get_block(i)(_, _, _);
    auto OCA_result_block_eq = eval_eq(itops, OCA_result_block, n_quad);
    ASSERT_LE(nda::max_element(nda::abs(OCA_result_block_eq - OCA_tpz_result(_, range(s0, s1), range(s0, s1)))), 2e-4);
    s0 = s1;
    if (i < num_blocks - 1) s1 += subspaces[i + 1].size();
  }
}