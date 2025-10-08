#include <gtest/gtest.h>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include <triqs_soehyb/block_sparse.hpp>
#include <triqs_soehyb/block_sparse_manual.hpp>

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
    temp_dense = nda::matmul(F_up_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 0, 0) * nda::matmul(temp_dense, F_up_dag_dense);
    temp_dense = nda::matmul(F_up_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 0, 1) * nda::matmul(temp_dense, F_down_dag_dense);
    temp_dense = nda::matmul(F_down_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 1, 0) * nda::matmul(temp_dense, F_up_dag_dense);
    temp_dense = nda::matmul(F_down_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 1, 1) * nda::matmul(temp_dense, F_down_dag_dense);

    // forward diagram
    temp_dense = nda::matmul(F_up_dag_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 0, 0) * nda::matmul(temp_dense, F_up_dense);
    temp_dense = nda::matmul(F_up_dag_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 0, 1) * nda::matmul(temp_dense, F_down_dense);
    temp_dense = nda::matmul(F_down_dag_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 1, 0) * nda::matmul(temp_dense, F_up_dense);
    temp_dense = nda::matmul(F_down_dag_dense, Gt_dense(t, _, _));
    NCA_result_dense(t, _, _) -= hyb(0, 1, 1) * nda::matmul(temp_dense, F_down_dense);
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

  EXPECT_LT(nda::norm((NCA_result.get_block(0)(_, 0, 0) - NCA_ana), std::numeric_limits<double>::infinity())
               / nda::norm(NCA_ana, std::numeric_limits<double>::infinity()),
            1.0e-12);
}

TEST(BlockSparseNCAManual, two_band_discrete_bath_bs_vs_dense) {
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

  // block-sparse NCA compuation
  auto NCA_result = NCA_bs(Deltat, Deltat_refl, Gt, Fs);

  // dense-matrix NCA computation
  auto NCA_dense_result = NCA_dense(Deltat, Deltat_refl, Gt_dense, Fs_dense, F_dags_dense);

  // check that block-sparse NCA calculation agrees with dense NCA calculation
  int s0 = 0;
  int s1 = subspaces[0].size();
  for (int i = 0; i < num_blocks; i++) { // compare each block
    ASSERT_LE(nda::max_element(nda::abs(NCA_result.get_block(i) - NCA_dense_result(_, range(s0, s1), range(s0, s1)))), 10 * eps);
    s0 = s1;
    if (i < num_blocks - 1) s1 += subspaces[i + 1].size();
  }
}

TEST(BlockSparseNCAManual, H5_two_band_discrete_bath_bs_vs_py) {
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

  // block-sparse NCA compuation
  auto NCA_result = NCA_bs(Deltat, Deltat_refl, Gt, Fs);

  // load NCA twoband.py
  h5::file Gtfile("../test/c++/h5/two_band_py.h5", 'r');
  h5::group Gtgroup(Gtfile);
  auto NCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "NCA", NCA_py);

  // permute twoband.py results to match block structure from atom_diag
  auto NCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) { NCA_py_perm(t, i, j) = NCA_py(t, fock_state_order[i], fock_state_order[j]); }
    }
  }

  // check that NCA calculation agrees with python NCA calculation
  int s0 = 0;
  int s1 = subspaces[0].size();
  for (int i = 0; i < num_blocks; i++) { // compare each block
    ASSERT_LE(nda::max_element(nda::abs(NCA_result.get_block(i) - NCA_py_perm(_, range(s0, s1), range(s0, s1)))), 10 * eps);
    s0 = s1;
    if (i < num_blocks - 1) s1 += subspaces[i + 1].size();
  }
}

TEST(BlockSparseNCAManual, H5_two_band_semicircle_bath_dense_aaa) {
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
  hyb = {{{-0.4997496184487105, -0.4997496184487105, -0., -0.},
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

  auto hyb_refl = nda::make_regular(itops.reflect(hyb));
  for (int i = 0; i < r; i++) { std::cout << "Gt_dense top block = " << Gt_dense(i, range(0, 4), range(0, 4)) << std::endl; }
  auto NCA_result = NCA_dense(hyb, hyb_refl, Gt_dense, Fs_dense, F_dags_dense);

  // load NCA result from twoband.py
  h5::file Gtfile("../test/c++/h5/two_band_py_semic.h5", 'r');
  h5::group Gtgroup(Gtfile);
  auto NCA_py = nda::zeros<dcomplex>(r, 16, 16);
  h5::read(Gtgroup, "NCA", NCA_py);

  // permute twoband.py results to match block structure from atom_diag
  auto NCA_py_perm = nda::zeros<dcomplex>(r, 16, 16);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 16; j++) { NCA_py_perm(t, i, j) = NCA_py(t, fock_state_order[i], fock_state_order[j]); }
    }
  }

  std::cout << "NCA result = " << NCA_result(10, _, _) << std::endl;
  std::cout << "NCA py perm = " << NCA_py_perm(10, _, _) << std::endl;
  std::cout << "diff = " << nda::make_regular(NCA_result(10, _, _) - NCA_py_perm(10, _, _)) << std::endl;
  ASSERT_LE(nda::max_element(nda::abs(NCA_result - NCA_py_perm)), eps);
}