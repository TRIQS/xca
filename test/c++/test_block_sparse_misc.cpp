#include <gtest/gtest.h>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/self_energy.hpp>

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

TEST(BlockSparseMisc, aaa_coefs2vals) {
  // example hybridization function
  nda::vector<dcomplex> hyb00{-0.4997496184487105, -0.4867352379479528, -0.4603465101833711, -0.4239204950540695, -0.3716597467714097,
                              -0.2884886574148449, -0.2479810727230272, -0.2065525284769785, -0.1635819676241178, -0.1326995066858671,
                              -0.1225444804140666, -0.1282199855712255, -0.1386184647087601, -0.1720919948804938, -0.2300400167898313,
                              -0.3000508284935615, -0.3759657450111002, -0.4545389745912252, -0.4821599768174421, -0.4997496184487105};
  // hybridizations coefficients computed using a Python AAA routine
  nda::vector<dcomplex> coefs00{0.0028042961182163, 0.088487039172428,  0.1575418229076625, 0.1953880665937937,
                                0.2145207908265103, 0.1832496441339733, 0.1580088741667851};
  // filling hybridization and coefficient arrays
  nda::array<dcomplex, 3> hyb_py(20, 4, 4), coefs(7, 4, 4);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i / 2 == j / 2) {
        hyb_py(_, i, j) = hyb00;
        coefs(_, i, j)  = coefs00;
      } else {
        hyb_py(_, i, j) = 0.0;
        coefs(_, i, j)  = 0.0;
      }
    }
  }
  // hybridization poles computed using a Python AAA routine
  nda::vector<double> poles{-2.537191963500981,  1.7111725610238615, -1.514666605887425, 1.04941790134832,
                            -0.7410379494142222, 0.3763525311836938, -0.1312888711963961};
  double beta   = 8.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-6;
  auto hyb      = aaa_coefs2vals(beta, Lambda, eps, coefs, poles);
  ASSERT_LE(nda::max_element(nda::abs(hyb - hyb_py)), eps);
}
