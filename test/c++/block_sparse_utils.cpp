#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include <triqs_soehyb/block_sparse.hpp>
#include "block_sparse_utils.hpp"

using namespace nda;
using namespace cppdlr;

nda::array<dcomplex, 3> Hmat_to_Gtmat(nda::array<dcomplex, 2> Hmat, double beta, nda::array<double, 1> dlr_it_abs) {
  // Helper function for computing the non-interacting Green's function from the Hamiltonian, both in dense storage

  int N                         = Hmat.extent(0);
  auto [H_loc_eval, H_loc_evec] = nda::linalg::eigenelements(Hmat);
  auto E0                       = nda::min_element(H_loc_eval);
  H_loc_eval -= E0;
  auto tr_exp_minusbetaH = nda::sum(exp(-beta * H_loc_eval));
  auto eta_0             = nda::log(tr_exp_minusbetaH) / beta;
  H_loc_eval += eta_0;
  auto Gt_evals_t = nda::zeros<dcomplex>(N, N);
  int r           = dlr_it_abs.extent(0);
  auto Gt_mat     = nda::zeros<dcomplex>(r, N, N);
  auto Gbeta      = nda::zeros<dcomplex>(N, N);
  for (int t = 0; t < r; t++) {
    for (int i = 0; i < N; i++) { Gt_evals_t(i, i) = -exp(-beta * dlr_it_abs(t) * H_loc_eval(i)); }
    Gt_mat(t, _, _) = nda::matmul(H_loc_evec, nda::matmul(Gt_evals_t, nda::transpose(H_loc_evec)));
  }
  return Gt_mat;
}

std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> discrete_bath_helper(double beta, double Lambda, double eps) {
  // Helper function for setting up the discrete bath model

  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = rel2abs(dlr_it);
  int r              = itops.rank();

  // hybridization parameters
  double s = 0.5;
  double t = 1.0;
  nda::array<double, 1> e{-2.3 * t, 2.3 * t};

  // hybridization generation
  auto Jt      = nda::array<dcomplex, 3>(r, 1, 1);
  auto Jt_refl = nda::array<dcomplex, 3>(r, 1, 1);
  for (int i = 0; i <= 1; i++) {
    for (int u = 0; u < r; u++) {
      Jt(u, 0, 0) += k_it(dlr_it_abs(u), e(i), beta);
      Jt_refl(u, 0, 0) += k_it(-dlr_it_abs(u), e(i), beta);
    }
  }

  // orbital index order: do 0, do 1, up 0, up 1. same level <-> same parity index
  auto Deltat      = nda::array<dcomplex, 3>(r, 4, 4);
  auto Deltat_refl = nda::array<dcomplex, 3>(r, 4, 4);

  for (int i = 0; i < Deltat.extent(1); i++) {
    for (int j = i; j < Deltat.extent(2); j++) {
      if (i == j) {
        Deltat(_, i, j)      = Jt(_, 0, 0);
        Deltat_refl(_, i, j) = Jt_refl(_, 0, 0);
      } else if ((i == 0 && j == 1) || (i == 1 && j == 0) || (i == 2 && j == 3) || (i == 3 && j == 2)) {
        Deltat(_, i, j)      = s * Jt(_, 0, 0);
        Deltat_refl(_, i, j) = s * Jt_refl(_, 0, 0);
      }
    }
  }
  Deltat      = t * t * Deltat;
  Deltat_refl = t * t * Deltat_refl;

  return std::make_tuple(Deltat, Deltat_refl);
}

std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> discrete_bath_spin_flip_helper(double beta, double Lambda, double eps, int n) {
  // Helper function for setting up the discrete bath model

  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // hybridization parameters
  double s = 0.5;
  double t = 1.0;
  nda::array<double, 1> e{-2.3 * t, 2.3 * t};

  // hybridization generation
  auto Jt      = nda::array<dcomplex, 3>(r, 1, 1);
  auto Jt_refl = nda::array<dcomplex, 3>(r, 1, 1);
  for (int i = 0; i <= 1; i++) {
    for (int u = 0; u < r; u++) {
      Jt(u, 0, 0) += k_it(dlr_it_abs(u), e(i), beta);
      Jt_refl(u, 0, 0) += k_it(-dlr_it_abs(u), e(i), beta);
    }
  }

  // orbital index order: do 0, up 0, do 1, up 1
  auto Deltat      = nda::array<dcomplex, 3>(r, n, n);
  auto Deltat_refl = nda::array<dcomplex, 3>(r, n, n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        Deltat(_, i, j)      = Jt(_, 0, 0);
        Deltat_refl(_, i, j) = Jt_refl(_, 0, 0);
      } else if ((j - i) % (n / 2) == 0) { // i and j have the same parity
        Deltat(_, i, j)      = s * Jt(_, 0, 0);
        Deltat_refl(_, i, j) = s * Jt_refl(_, 0, 0);
      }
    }
  }
  Deltat      = t * t * Deltat;
  Deltat_refl = t * t * Deltat_refl;

  return std::make_tuple(Deltat, Deltat_refl);
}

std::tuple<BlockDiagOpFun, BlockOpSymQuartet, nda::vector<int>> two_band_helper(double beta, double Lambda, double eps,
                                                                                nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                                                                nda::array_const_view<dcomplex, 3> hyb_refl_coeffs) {
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  // get Hamiltonian, creation/annihilation operators in block-sparse storage
  int num_blocks = 5; // number of blocks of Hamiltonian

  // Hamiltonian
  std::vector<nda::array<double, 2>> H_blocks(num_blocks); // Hamiltonian in sparse storage
  H_blocks[0]                   = nda::make_regular(-1 * nda::eye<double>(4));
  H_blocks[1]                   = {{-0.6, 0, 0, 0, 0, 0},   {0, 8.27955e-19, 0, 0, 0.2, 0}, {0, 0, -0.4, 0.2, 0, 0},
                                   {0, 0, 0.2, -0.4, 0, 0}, {0, 0.2, 0, 0, 8.27955e-19, 0}, {0, 0, 0, 0, 0, -0.6}};
  H_blocks[2]                   = {{0}};
  H_blocks[3]                   = nda::make_regular(2 * nda::eye<double>(4));
  H_blocks[4]                   = {{6}};
  nda::vector<int> H_block_inds = {0, 0, -1, 0, 0};

  // Green's function
  auto Gt = nonint_gf_BDOF(H_blocks, H_block_inds, beta, dlr_it_abs);

  // creation/annihilation operators
  nda::vector<int> ann_conn = {2, 0, -1, 1, 3}; // block column indices of F operators
  nda::vector<int> cre_conn = {1, 3, 0, 4, -1}; // block column indices of F^dag operators
  std::vector<nda::array<dcomplex, 3>> F_blocks(num_blocks), Fdag_blocks(num_blocks);

  F_blocks[0] = {{{1, 0, 0, 0}}, {{0, 1, 0, 0}}, {{0, 0, 1, 0}}, {{0, 0, 0, 1}}};
  F_blocks[1] = {{{0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0}},
                 {{-1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0}},
                 {{0, -1, 0, 0, 0, 0}, {0, 0, -1, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}},
                 {{0, 0, 0, -1, 0, 0}, {0, 0, 0, 0, -1, 0}, {0, 0, 0, 0, 0, -1}, {0, 0, 0, 0, 0, 0}}};
  F_blocks[2] = {{{0}}};
  F_blocks[3] = {{{0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}},
                 {{0, 0, 0, 0}, {-1, 0, 0, 0}, {0, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}},
                 {{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, -1}, {0, 0, 0, 0}},
                 {{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}};
  F_blocks[4] = {{{0}, {0}, {0}, {1}}, {{0}, {0}, {-1}, {0}}, {{0}, {1}, {0}, {0}}, {{-1}, {0}, {0}, {0}}};

  Fdag_blocks[0] = {{{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}},
                    {{-1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}},
                    {{0, 0, 0, 0}, {-1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}},
                    {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {-1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}}};
  Fdag_blocks[1] = {{{0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0}},
                    {{0, -1, 0, 0, 0, 0}, {0, 0, 0, -1, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}},
                    {{1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, -1, 0, 0}, {0, 0, 0, 0, -1, 0}},
                    {{0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0}}};
  Fdag_blocks[2] = {{{1}, {0}, {0}, {0}}, {{0}, {1}, {0}, {0}}, {{0}, {0}, {1}, {0}}, {{0}, {0}, {0}, {1}}};
  Fdag_blocks[3] = {{{0, 0, 0, 1}}, {{0, 0, -1, 0}}, {{0, 1, 0, 0}}, {{-1, 0, 0, 0}}};
  Fdag_blocks[4] = {{{0}}};

  BlockOpSymSet F_BOSS(ann_conn, F_blocks), Fdag_BOSS(cre_conn, Fdag_blocks);
  std::vector<BlockOpSymSet> F_sym_vec{F_BOSS}, F_dag_sym_vec{Fdag_BOSS};
  nda::vector<long> sym_set_labels(4);
  sym_set_labels = 0; // all operators belong to the same symmetry set
  BlockOpSymQuartet Fq(F_sym_vec, F_dag_sym_vec, hyb_coeffs, hyb_refl_coeffs, sym_set_labels);

  return std::make_tuple(Gt, Fq, sym_set_labels);
}

std::tuple<int, nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, BlockDiagOpFun, std::vector<BlockOp>, std::vector<BlockOp>, nda::array<dcomplex, 3>,
           nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, std::vector<std::vector<unsigned long>>, std::vector<long>>
two_band_discrete_bath_helper(double beta, double Lambda, double eps) {
  // Helper function for setting up the two-band discrete bath model

  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // hybridization parameters
  double s = 0.5;
  double t = 1.0;
  nda::array<double, 1> e{-2.3 * t, 2.3 * t};

  // hybridization generation
  auto Jt      = nda::array<dcomplex, 3>(r, 1, 1);
  auto Jt_refl = nda::array<dcomplex, 3>(r, 1, 1);
  for (int i = 0; i <= 1; i++) {
    for (int u = 0; u < r; u++) {
      Jt(u, 0, 0) += k_it(dlr_it_abs(u), e(i), beta);
      Jt_refl(u, 0, 0) += k_it(-dlr_it_abs(u), e(i), beta);
    }
  }
  // orbital index order: do 0, do 1, up 0, up 1. same level <-> same parity index
  auto Deltat      = nda::array<dcomplex, 3>(r, 4, 4);
  auto Deltat_refl = nda::array<dcomplex, 3>(r, 4, 4);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      if (i == j) {
        Deltat(_, i, j)      = Jt(_, 0, 0);
        Deltat_refl(_, i, j) = Jt_refl(_, 0, 0);
      } else if ((i == 0 && j == 1) || (i == 1 && j == 0) || (i == 2 && j == 3) || (i == 3 && j == 2)) {
        Deltat(_, i, j)      = s * Jt(_, 0, 0);
        Deltat_refl(_, i, j) = s * Jt_refl(_, 0, 0);
      }
    }
  }
  Deltat      = t * t * Deltat;
  Deltat_refl = t * t * Deltat_refl;

  // get Hamiltonian, creation/annihilation operators in block-sparse storage
  int num_blocks = 5; // number of blocks of Hamiltonian

  // Hamiltonian
  std::vector<nda::array<double, 2>> H_blocks(num_blocks); // Hamiltonian in sparse storage
  H_blocks[0]                   = nda::make_regular(-1 * nda::eye<double>(4));
  H_blocks[1]                   = {{-0.6, 0, 0, 0, 0, 0},   {0, 8.27955e-19, 0, 0, 0.2, 0}, {0, 0, -0.4, 0.2, 0, 0},
                                   {0, 0, 0.2, -0.4, 0, 0}, {0, 0.2, 0, 0, 8.27955e-19, 0}, {0, 0, 0, 0, 0, -0.6}};
  H_blocks[2]                   = {{0}};
  H_blocks[3]                   = nda::make_regular(2 * nda::eye<double>(4));
  H_blocks[4]                   = {{6}};
  nda::vector<int> H_block_inds = {0, 0, -1, 0, 0};

  // Green's function
  auto Gt = nonint_gf_BDOF(H_blocks, H_block_inds, beta, dlr_it_abs);

  // creation/annihilation operators
  nda::vector<int> ann_conn = {2, 0, -1, 1, 3}; // block column indices of F operators
  nda::vector<int> cre_conn = {1, 3, 0, 4, -1}; // block column indices of F^dag operators
  std::vector<BlockOp> Fs;
  std::vector<BlockOp> Fdags;
  std::vector<nda::array<dcomplex, 2>> dummy(num_blocks);
  std::vector<std::vector<nda::array<dcomplex, 2>>> F_blocks(4, dummy);
  std::vector<std::vector<nda::array<dcomplex, 2>>> Fdag_blocks(4, dummy);

  F_blocks[0][0] = {{1, 0, 0, 0}};
  F_blocks[0][1] = {{0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0}};
  F_blocks[0][2] = {{0}};
  F_blocks[0][3] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
  F_blocks[0][4] = {{0}, {0}, {0}, {1}};

  F_blocks[1][0] = {{0, 1, 0, 0}};
  F_blocks[1][1] = {{-1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0}};
  F_blocks[1][2] = {{0}};
  F_blocks[1][3] = {{0, 0, 0, 0}, {-1, 0, 0, 0}, {0, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}};
  F_blocks[1][4] = {{0}, {0}, {-1}, {0}};

  F_blocks[2][0] = {{0, 0, 1, 0}};
  F_blocks[2][1] = {{0, -1, 0, 0, 0, 0}, {0, 0, -1, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}};
  F_blocks[2][2] = {{0}};
  F_blocks[2][3] = {{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, -1}, {0, 0, 0, 0}};
  F_blocks[2][4] = {{0}, {1}, {0}, {0}};

  F_blocks[3][0] = {{0, 0, 0, 1}};
  F_blocks[3][1] = {{0, 0, 0, -1, 0, 0}, {0, 0, 0, 0, -1, 0}, {0, 0, 0, 0, 0, -1}, {0, 0, 0, 0, 0, 0}};
  F_blocks[3][2] = {{0}};
  F_blocks[3][3] = {{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  F_blocks[3][4] = {{-1}, {0}, {0}, {0}};

  Fdag_blocks[0][0] = {{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  Fdag_blocks[0][1] = {{0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0}};
  Fdag_blocks[0][2] = {{1}, {0}, {0}, {0}};
  Fdag_blocks[0][3] = {{0, 0, 0, 1}};
  Fdag_blocks[0][4] = {{0}};

  Fdag_blocks[1][0] = {{-1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 0}};
  Fdag_blocks[1][1] = {{0, -1, 0, 0, 0, 0}, {0, 0, 0, -1, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}};
  Fdag_blocks[1][2] = {{0}, {1}, {0}, {0}};
  Fdag_blocks[1][3] = {{0, 0, -1, 0}};
  Fdag_blocks[1][4] = {{0}};

  Fdag_blocks[2][0] = {{0, 0, 0, 0}, {-1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 1}};
  Fdag_blocks[2][1] = {{1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, -1, 0, 0}, {0, 0, 0, 0, -1, 0}};
  Fdag_blocks[2][2] = {{0}, {0}, {1}, {0}};
  Fdag_blocks[2][3] = {{0, 1, 0, 0}};
  Fdag_blocks[2][4] = {{0}};

  Fdag_blocks[3][0] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {-1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}};
  Fdag_blocks[3][1] = {{0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}, {0, 0, 1, 0, 0, 0}};
  Fdag_blocks[3][2] = {{0}, {0}, {0}, {1}};
  Fdag_blocks[3][3] = {{-1, 0, 0, 0}};
  Fdag_blocks[3][4] = {{0}};

  for (int i = 0; i < 4; i++) {
    Fs.emplace_back(ann_conn, F_blocks[i]);
    Fdags.emplace_back(cre_conn, Fdag_blocks[i]);
  }

  // subspace indices
  std::vector<unsigned long> dummy2;
  std::vector<std::vector<unsigned long>> subspaces(num_blocks, dummy2);
  subspaces[0] = {1, 2, 4, 8};
  subspaces[1] = {3, 5, 6, 9, 10, 12};
  subspaces[2] = {0};
  subspaces[3] = {7, 11, 13, 14};
  subspaces[4] = {15};
  std::vector<long> fock_state_order(begin(subspaces[0]), end(subspaces[0]));
  for (int i = 1; i < num_blocks; i++) { fock_state_order.insert(end(fock_state_order), begin(subspaces[i]), end(subspaces[i])); }

  // Hamiltonian in dense storage
  auto H_dense    = nda::zeros<dcomplex>(16, 16);
  H_dense(0, 0)   = -1;
  H_dense(1, 1)   = -1;
  H_dense(2, 2)   = -1;
  H_dense(3, 3)   = -1;
  H_dense(4, 4)   = -0.6;
  H_dense(5, 8)   = 0.2;
  H_dense(6, 6)   = -0.4;
  H_dense(6, 7)   = 0.2;
  H_dense(7, 6)   = 0.2;
  H_dense(7, 7)   = -0.4;
  H_dense(8, 5)   = 0.2;
  H_dense(9, 9)   = -0.6;
  H_dense(11, 11) = 2;
  H_dense(12, 12) = 2;
  H_dense(13, 13) = 2;
  H_dense(14, 14) = 2;
  H_dense(15, 15) = 6;

  // Green's function in dense storage
  auto Gt_dense = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);

  // creation/annihilation operators in dense storage
  auto Fs_dense = nda::zeros<dcomplex>(4, 16, 16);
  // h5::read(hgroup, "c_dense", Fs_dense);
  // copied from a text dump of an h5 file output from atom_diag
  Fs_dense          = {{{0, 0, 0, 0, 0, -1, 0, 0, -2.23711e-17, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, -1, -2.23711e-17, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.23711e-17, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.23711e-17, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                       {{0, 0, 0, 0, 0, 0, -2.23711e-17, -1, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, -2.23711e-17, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.23711e-17, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.23711e-17, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                       {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 1, 0, 0, 2.23711e-17, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 2.23711e-17, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.23711e-17, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.23711e-17, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                       {{0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 1, 2.23711e-17, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 2.23711e-17, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.23711e-17, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.23711e-17, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                        {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}};
  auto F_dags_dense = nda::zeros<dcomplex>(4, 16, 16);
  for (int i = 0; i < 4; i++) { F_dags_dense(i, _, _) = nda::transpose(nda::conj(Fs_dense(i, _, _))); }

  return std::make_tuple(num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order);
}

std::tuple<int, nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, BlockDiagOpFun, std::vector<BlockOp>, std::vector<BlockOp>, nda::array<dcomplex, 3>,
           nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, std::vector<std::vector<unsigned long>>, std::vector<long>, BlockOpSymQuartet>
two_band_discrete_bath_helper_sym(double beta, double Lambda, double eps) {

  auto [num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order] =
     two_band_discrete_bath_helper(beta, Lambda, eps);

  int n = 4, k = num_blocks;
  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // create cre/ann operators
  auto hyb_coeffs      = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(Deltat));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);

  // TODO replace this with read from atom_diag
  // compute creation and annihilation operators in block-sparse storage
  auto F_sym = BlockOpSymSet(n, Fs[0].get_block_indices(), Fs[0].get_block_sizes());
  for (int i = 0; i < k; i++) {
    if (Fs[0].get_block_index(i) == -1) continue; // skip empty blocks
    auto Fs_sym_block = nda::zeros<dcomplex>(n, Fs[0].get_block_size(i, 0), Fs[0].get_block_size(i, 1));
    for (int j = 0; j < n; j++) { Fs_sym_block(j, _, _) = Fs[j].get_block(i); }
    F_sym.set_block(i, Fs_sym_block);
  }
  std::vector<BlockOpSymSet> F_sym_vec{F_sym};

  auto F_dag_sym = BlockOpSymSet(n, Fdags[0].get_block_indices(), Fdags[0].get_block_sizes());
  for (int i = 0; i < k; i++) {
    if (Fdags[0].get_block_index(i) == -1) continue; // skip empty blocks
    auto F_dags_sym_block = nda::zeros<dcomplex>(n, Fdags[0].get_block_size(i, 0), Fdags[0].get_block_size(i, 1));
    for (int j = 0; j < n; j++) { F_dags_sym_block(j, _, _) = Fdags[j].get_block(i); }
    F_dag_sym.set_block(i, F_dags_sym_block);
  }
  std::vector<BlockOpSymSet> F_dag_sym_vec{F_dag_sym};
  nda::vector<long> sym_set_labels_triv(n);
  sym_set_labels_triv = 0;

  auto Fq = BlockOpSymQuartet(F_sym_vec, F_dag_sym_vec, hyb_coeffs, hyb_refl_coeffs, sym_set_labels_triv);

  return std::make_tuple(num_blocks, Deltat, Deltat_refl, Gt, Fs, Fdags, Gt_dense, Fs_dense, F_dags_dense, subspaces, fock_state_order, Fq);
}

std::tuple<nda::array<dcomplex, 3>, DenseFSet, std::vector<nda::vector<unsigned long>>>
spin_flip_fermion_dense_helper(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                               nda::array_const_view<dcomplex, 3> hyb_refl_coeffs, std::string filename) {
  // DLR generation
  auto dlr_rf     = build_dlr_rf(Lambda, eps);
  auto itops      = imtime_ops(Lambda, dlr_rf);
  auto dlr_it     = itops.get_itnodes();
  auto dlr_it_abs = rel2abs(dlr_it);

  h5::file f(filename, 'r');
  h5::group g(f);

  long n = 0;
  h5::read(g, "norb", n);
  int N                           = static_cast<int>(pow(4, static_cast<double>(n))); // number of Fock states
  nda::array<dcomplex, 2> H_dense = nda::zeros<dcomplex>(N, N);
  h5::read(g, "H_mat_dense", H_dense);
  auto Gt_dense                    = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
  nda::array<dcomplex, 3> Fs_dense = nda::zeros<dcomplex>(2 * n, N, N);
  h5::read(g, "c_dense", Fs_dense);
  nda::array<dcomplex, 3> F_dags_dense = nda::zeros<dcomplex>(2 * n, N, N);
  h5::read(g, "cdag_dense", F_dags_dense);

  DenseFSet Fset(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);

  long k = 0;
  h5::read(g, "num_blocks", k);
  std::vector<nda::vector<unsigned long>> subspaces(k, nda::vector<unsigned long>());
  for (int i = 0; i < k; ++i) {
    nda::vector<unsigned long> subspace;
    h5::read(g, "ad/sub_hilbert_spaces/" + std::to_string(i) + "/fock_states", subspace);
    subspaces[i] = subspace;
  }

  return std::make_tuple(Gt_dense, Fset, subspaces);
}