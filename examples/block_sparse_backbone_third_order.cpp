#include <chrono>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/block_sparse_manual.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>

using namespace nda;
using namespace cppdlr;

nda::array<dcomplex, 3> Hmat_to_Gtmat(nda::array<dcomplex, 2> Hmat, double beta, nda::array<double, 1> dlr_it_abs) {
  // Helper function for computing the non-interacting Green's function from the Hamiltonian, both in dense storage

  int N                         = Hmat.extent(0);
  auto [H_loc_eval, H_loc_evec] = nda::linalg::eigh(Hmat);
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
    Gt_mat(t, _, _) = matmul(H_loc_evec, matmul(Gt_evals_t, nda::transpose(H_loc_evec)));
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
      Jt(u, 0, 0) += k_it(dlr_it(u), e(i), beta);
      Jt_refl(u, 0, 0) += k_it(-dlr_it(u), e(i), beta);
    }
  }

  // orbital index order: do 0, do 1, up 0, up 1. same level <-> same parity index
  auto Deltat      = nda::array<dcomplex, 3>(r, 4, 4);
  auto Deltat_refl = nda::array<dcomplex, 3>(r, 4, 4);

  for (int i = 0; i < Deltat.extent(1); i++) {
    for (int j = 0; j < Deltat.extent(2); j++) {
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

std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> two_band_dense_helper(double beta, double Lambda, double eps) {

  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

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
  // copied from a text dump of an h5 file output from atom_diag
  Fs_dense          = {{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
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
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                       {{0, 0, 0, 0, 0, -1, 0, 0, -2.23711e-17, 0, 0, 0, 0, 0, 0, 0},
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
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}};
  auto F_dags_dense = nda::zeros<dcomplex>(4, 16, 16);
  for (int i = 0; i < 4; i++) { F_dags_dense(i, _, _) = nda::transpose(nda::conj(Fs_dense(i, _, _))); }

  return std::make_tuple(Gt_dense, Fs_dense, F_dags_dense);
}

std::tuple<BlockDiagOpFun, BlockOpSymQuartet, nda::vector<int>> two_band_helper(double beta, double Lambda, double eps,
                                                                                nda::array_const_view<dcomplex, 3> hyb_coeffs) {
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
  BlockOpSymQuartet Fq(F_sym_vec, F_dag_sym_vec, hyb_coeffs, sym_set_labels);

  return std::make_tuple(Gt, Fq, sym_set_labels);
}

int main() {
  nda::array<int, 3> topologies = {{{0, 2}, {1, 4}, {3, 5}}, {{0, 3}, {1, 5}, {2, 4}}, {{0, 4}, {1, 3}, {2, 5}}, {{0, 3}, {1, 4}, {2, 5}}};
  nda::vector<int> topo_sign{1, 1, 1, -1}; // topo_sign(i) = (-1)^{# of line crossings in topology i}

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  int n = 4, N = 16;
  double beta   = 2.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-4;

  // generate hybridization, noninteracting Green's function, creation/annihilation operators
  auto [Deltat, Deltat_refl]              = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);

  // DLR generation
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);
  int r              = itops.rank();

  // hybridization
  auto hyb_coeffs               = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl                 = Deltat;
  auto hyb_refl_coeffs          = hyb_coeffs;
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);

  auto D = DiagramEvaluator(beta, itops, Deltat, hyb_refl, dlr_rf, Gt, Fq); // create DiagramEvaluator object

  auto Deltadlr                            = itops.vals2coefs(Deltat); //obtain dlr coefficient of Delta(t)
  nda::vector<double> dlr_rf_reflect       = -dlr_rf;
  nda::array<dcomplex, 3> Deltadlr_reflect = Deltadlr * 1.0;
  for (int i = 0; i < Deltadlr.shape(0); ++i) Deltadlr_reflect(i, _, _) = transpose(Deltadlr(i, _, _));
  auto Delta_decomp         = hyb_decomp(Deltadlr, dlr_rf, eps);
  auto Delta_decomp_reflect = hyb_decomp(Deltadlr_reflect, dlr_rf_reflect, eps);
  hyb_F Delta_F(N, r, n);
  hyb_F Delta_F_reflect(N, r, n);
  Delta_F.update_inplace(Delta_decomp, dlr_it, Fs_dense, F_dags_dense);
  Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F_dags_dense, Fs_dense);
  auto fb = nda::vector<int64_t>(3);
  fb(0)   = 0;

  BlockDiagOpFun third_order_result(r, Gt.get_block_sizes());
  for (int i = 0; i < 4; ++i) {
    std::cout << "Evaluating topology " << i << std::endl;

    // Compute third-order contribution using DiagramEvaluator
    auto B = Backbone(topologies(i, _, _), n);
    D.eval_self_energy(B);
    third_order_result = D.get_self_energy();
    D.reset(); // reset the DiagramEvaluator for the next topology

    // Compute third-order contribution using old code
    nda::array<dcomplex, 3> TCA_old(r, N, N);
    TCA_old = 0;
    fb(1)   = 0;
    fb(2)   = 0;
    TCA_old += Sigma_Diagram_calc(Delta_F, Delta_F_reflect, topologies(i, _, _), Deltat, Deltat_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense,
                                  fb, true);
    fb(1) = 1;
    TCA_old += Sigma_Diagram_calc(Delta_F, Delta_F_reflect, topologies(i, _, _), Deltat, Deltat_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense,
                                  fb, true);
    fb(2) = 1;
    TCA_old += Sigma_Diagram_calc(Delta_F, Delta_F_reflect, topologies(i, _, _), Deltat, Deltat_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense,
                                  fb, true);
    fb(1) = 0;
    TCA_old += Sigma_Diagram_calc(Delta_F, Delta_F_reflect, topologies(i, _, _), Deltat, Deltat_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense,
                                  fb, true);

    std::cout << "max error in block 0: " << nda::max_element(nda::abs(TCA_old(_, range(0, 4), range(0, 4)) - third_order_result.get_block(0)))
              << std::endl;
    std::cout << "max error in block 1: " << nda::max_element(nda::abs(TCA_old(_, range(4, 10), range(4, 10)) - third_order_result.get_block(1)))
              << std::endl;
    std::cout << "max error in block 2: " << nda::max_element(nda::abs(TCA_old(_, range(10, 11), range(10, 11)) - third_order_result.get_block(2)))
              << std::endl;
    std::cout << "max error in block 3: " << nda::max_element(nda::abs(TCA_old(_, range(11, 15), range(11, 15)) - third_order_result.get_block(3)))
              << std::endl;
    std::cout << "max error in block 4: " << nda::max_element(nda::abs(TCA_old(_, range(15, 16), range(15, 16)) - third_order_result.get_block(4)))
              << std::endl;
    std::cout << std::endl;
  }
}
