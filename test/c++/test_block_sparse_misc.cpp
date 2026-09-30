#include <gtest/gtest.h>

#include <triqs_xca/hyb.hpp>

#include "block_sparse_utils.hpp"

using nda::range;
using nda::linalg::matmul;

using cppdlr::_;
using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;

using triqs_xca::block_sparse::nonint_gf_BDOF;

TEST(BlockSparseMisc, compute_nonint_gf) {
  // DLR parameters
  double beta   = 2.0;
  double Lambda = 1000 * beta;
  double eps    = 1.0e-10;
  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // the following variables can be read from the output of benchmarks/atom_diag_to_text.py
  int num_blocks = 5;                                        // number of blocks in Hamiltonian
  std::vector<nda::array<dcomplex, 2>> H_blocks(num_blocks); // Hamiltonian in sparse storage
  H_blocks[0]                           = nda::make_regular(-1 * nda::eye<dcomplex>(4));
  H_blocks[1]                           = {{-0.6, 0, 0, 0, 0, 0},   {0, 8.27955e-19, 0, 0, 0.2, 0}, {0, 0, -0.4, 0.2, 0, 0},
                                           {0, 0, 0.2, -0.4, 0, 0}, {0, 0.2, 0, 0, 8.27955e-19, 0}, {0, 0, 0, 0, 0, -0.6}};
  H_blocks[2]                           = {{0}};
  H_blocks[3]                           = nda::make_regular(2 * nda::eye<dcomplex>(4));
  H_blocks[4]                           = {{6}};
  nda::vector<int> H_block_inds         = {0, 0, -1, 0, 0};
  auto H_dense                          = nda::zeros<dcomplex>(16, 16); // Hamiltonian in dense storage
  H_dense(range(0, 4), range(0, 4))     = H_blocks[0];
  H_dense(range(4, 10), range(4, 10))   = H_blocks[1];
  H_dense(range(11, 15), range(11, 15)) = H_blocks[3];
  H_dense(15, 15)                       = 6;

  // compute noninteracting Green's function from dense Hamiltonian
  auto [H_loc_eval, H_loc_evec] = nda::linalg::eigh(H_dense);
  auto E0                       = nda::min_element(H_loc_eval);
  H_loc_eval -= E0;
  auto tr_exp_minusbetaH = nda::sum(exp(-beta * H_loc_eval));
  auto eta_0             = nda::log(tr_exp_minusbetaH) / beta;
  H_loc_eval += eta_0;
  auto Gt_evals_t = nda::zeros<dcomplex>(16, 16);
  auto Gt_mat     = nda::zeros<dcomplex>(r, 16, 16);
  auto Gbeta      = nda::zeros<dcomplex>(16, 16);
  Gt_mat          = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
  for (int i = 0; i < 16; i++) { Gbeta(i, i) = -exp(-beta * H_loc_eval(i)); }
  Gbeta = matmul(Gbeta, nda::transpose(H_loc_evec));
  Gbeta = matmul(H_loc_evec, Gbeta);
  // check that trace of noninteracting Green's function from dense
  // Hamiltonian at tau = beta has trace 1
  ASSERT_LE(nda::abs(nda::trace(Gbeta) + 1), 1e-13);

  auto Gt = nonint_gf_BDOF(H_blocks, H_block_inds, beta, dlr_it_abs);
  // check that the noninteracting Green's function, computing from the
  // sparse- and dense-storage Hamiltonians are the same
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(0, 4), range(0, 4)) - Gt.get_block(0))), 1e-13);
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(4, 10), range(4, 10)) - Gt.get_block(1))), 1e-13);
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(10, 11), range(10, 11)) - Gt.get_block(2))), 1e-13);
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(11, 15), range(11, 15)) - Gt.get_block(3))), 1e-13);
  ASSERT_LE(nda::max_element(nda::abs(Gt_mat(_, range(15, 16), range(15, 16)) - Gt.get_block(4))), 1e-13);
}

TEST(BlockSparseMisc, block_gf_to_BDOF) {
  double beta   = 1;
  double Lambda = 10 * beta;
  double eps    = 1.0e-6;
  // generate a DLR imaginary time mesh
  auto iw_dlr_mesh  = triqs::mesh::dlr_imfreq(beta, triqs::mesh::Fermion, Lambda, eps);
  auto tau_dlr_mesh = triqs::mesh::dlr_imtime(iw_dlr_mesh);

  auto g = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>{
     {"bl0", "bl1"},
     {triqs::gfs::gf<triqs::mesh::dlr_imtime>{{tau_dlr_mesh}, {2, 2}}, triqs::gfs::gf<triqs::mesh::dlr_imtime>{{tau_dlr_mesh}, {3, 3}}}};

  // Initialize bl0 (2x2 block)
  for (auto tau : g[0].mesh()) { g[0][tau] = nda::matrix<dcomplex>{{1e-17, 0}, {0, 1e-17}}; }

  // Initialize bl1 (3x3 block)
  for (auto tau : g[1].mesh()) {
    g[1][tau] = nda::matrix<dcomplex>{{2.0 + 0.1 * tau, 0.3, 0.0}, {0.3, 2.5 + 0.1 * tau, 0.4}, {0.0, 0.4, 3.0 + 0.1 * tau}};
  }
  auto BDOF = BlockDiagOpFun(g);
  ASSERT_EQ(BDOF.get_num_block_cols(), 2);
  ASSERT_EQ(BDOF.get_block_size(0), 2);
  ASSERT_EQ(BDOF.get_block_size(1), 3);
  ASSERT_EQ(BDOF.get_zero_block_index(0), -1); // first block is zero
  ASSERT_EQ(BDOF.get_zero_block_index(1), 0);
}

/**
 * @brief Check that add_block rejects a contribution whose shape does not match the target block
 *
 * @details The assign branch for a block marked zero would silently resize it, and the accumulate branch reads past the contribution's extent
 * in a build without bounds checks, which turned a 1e-14 cross-symmetry-set contribution into an O(1) change of the self-energy.
 */
TEST(BlockDiagOpFun, add_block_rejects_a_wrongly_shaped_contribution) {
  int r                        = 3;
  nda::vector<int> block_sizes = {2, 1};
  auto A                       = triqs_xca::block_sparse::BlockDiagOpFun(r, block_sizes);
  ASSERT_EQ(A.get_zero_block_index(0), -1) << "premise: this constructor marks every block zero, so the first "
                                              "add_block below takes the assign branch and the second the accumulate one";

  // the right shape is accepted on both branches
  ASSERT_NO_THROW(A.add_block(0, nda::zeros<dcomplex>(r, 2, 2))); // assign branch: block 0 starts marked zero
  ASSERT_NO_THROW(A.add_block(0, nda::zeros<dcomplex>(r, 2, 2))); // accumulate branch: now marked non-zero

  EXPECT_THROW(A.add_block(0, nda::zeros<dcomplex>(r, 1, 2)), std::invalid_argument) << "accumulate branch, wrong rows";
  EXPECT_THROW(A.add_block(0, nda::zeros<dcomplex>(r, 2, 1)), std::invalid_argument) << "accumulate branch, wrong cols";
  EXPECT_THROW(A.add_block(0, nda::zeros<dcomplex>(r + 1, 2, 2)), std::invalid_argument) << "accumulate branch, wrong DLR rank";
  EXPECT_THROW(A.add_block(1, nda::zeros<dcomplex>(r, 2, 2)), std::invalid_argument) << "assign branch, wrong shape";

  // the message must name the block
  try {
    A.add_block(1, nda::zeros<dcomplex>(r, 2, 2));
    FAIL() << "expected a throw";
  } catch (std::invalid_argument const &e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find("add_block"), std::string::npos) << msg;
    EXPECT_NE(msg.find('1'), std::string::npos) << "the message should name the block index: " << msg;
  }
}
