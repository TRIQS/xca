#include <gtest/gtest.h>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include <triqs_xca/block_sparse.hpp>

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
  int num_blocks = 5;                                      // number of blocks in Hamiltonian
  std::vector<nda::array<double, 2>> H_blocks(num_blocks); // Hamiltonian in sparse storage
  H_blocks[0]                           = nda::make_regular(-1 * nda::eye<double>(4));
  H_blocks[1]                           = {{-0.6, 0, 0, 0, 0, 0},   {0, 8.27955e-19, 0, 0, 0.2, 0}, {0, 0, -0.4, 0.2, 0, 0},
                                           {0, 0, 0.2, -0.4, 0, 0}, {0, 0.2, 0, 0, 8.27955e-19, 0}, {0, 0, 0, 0, 0, -0.6}};
  H_blocks[2]                           = {{0}};
  H_blocks[3]                           = nda::make_regular(2 * nda::eye<double>(4));
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
