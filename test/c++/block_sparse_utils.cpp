#include <triqs/operators/many_body_operator.hpp>

#include "block_sparse_utils.hpp"

using nda::dcomplex;
using nda::linalg::matmul;

using cppdlr::_;
using cppdlr::build_dlr_rf;
using cppdlr::imtime_ops;
using cppdlr::k_it;
using cppdlr::rel2abs;

using triqs_xca::block_sparse::BlockOpSymSet;
using triqs_xca::block_sparse::nonint_gf_BDOF;

FermionModelData one_fermion_model_helper(double beta, double Lambda, double eps, double hyb_pole) {
  // Helper function for setting up one-fermion tests with H = 0 and a one-pole hybridization decomposition.
  int p    = 1;
  int norb = 1;
  nda::array<dcomplex, 3> hyb_coeffs(p, norb, norb);
  hyb_coeffs = 1.0;

  nda::vector<double> hyb_poles(p);
  hyb_poles = hyb_pole;

  using triqs::operators::many_body_operator_complex;
  using triqs::operators::n;
  many_body_operator_complex H;
  double mu = 0.0;
  many_body_operator_complex N;
  N = n("0", 0);
  H = -mu * N;

  triqs::atom_diag::fundamental_operator_set fop_set;
  fop_set.insert("0", 0);
  auto ad = triqs::atom_diag::atom_diag<true>(H, fop_set);

  auto G_ppsc = triqs_xca::atom_diag::ad_to_atom_prop(ad, beta, Lambda, eps);
  auto G_bdof = BlockDiagOpFun(G_ppsc);

  return {.hyb_coeffs = hyb_coeffs, .hyb_poles = hyb_poles, .ad = ad, .G_ppsc = G_ppsc, .G_bdof = G_bdof};
}

FermionModelData two_fermion_model_helper(double beta, double Lambda, double eps, double U, double mu, double hyb_pole) {
  // Helper function for setting up two-fermion tests with H = -mu * (n0 + n1) + U * n0 * n1 and one-pole hybridization.
  using triqs::operators::many_body_operator_complex;
  using triqs::operators::n;

  many_body_operator_complex H;
  auto N0  = n("0", 0);
  auto N1  = n("1", 0);
  auto Nop = N0 + N1;
  H        = -mu * Nop + U * N0 * N1;

  triqs::atom_diag::fundamental_operator_set fop_set;
  fop_set.insert("0", 0);
  fop_set.insert("1", 0);

  std::vector<many_body_operator_complex> sym_ops = {Nop};
  auto ad                                         = triqs::atom_diag::atom_diag<true>(H, fop_set, sym_ops);

  int p    = 1;
  int norb = 2;
  nda::array<dcomplex, 3> hyb_coeffs(p, norb, norb);
  hyb_coeffs(0, _, _) = nda::eye(norb);

  nda::vector<double> hyb_poles(p);
  hyb_poles = hyb_pole;

  auto G_ppsc = triqs_xca::atom_diag::ad_to_atom_prop(ad, beta, Lambda, eps);

  auto G_bdof = BlockDiagOpFun(G_ppsc);

  return {.hyb_coeffs = hyb_coeffs, .hyb_poles = hyb_poles, .ad = ad, .G_ppsc = G_ppsc, .G_bdof = G_bdof};
}

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
      Jt(u, 0, 0) += k_it(dlr_it(u), e(i), beta);
      Jt_refl(u, 0, 0) += k_it(-dlr_it(u), e(i), beta);
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

triqs::operators::many_body_operator_complex make_kanamori_interaction(int n_orb, double U, double J, double mu) {
  using triqs::operators::c;
  using triqs::operators::c_dag;
  using triqs::operators::n;
  triqs::operators::many_body_operator_complex H;
  double U_prime = U - 2 * J;
  for (int o = 0; o < n_orb; o++) H -= mu * (n("up", o) + n("do", o));
  for (int o = 0; o < n_orb; o++) H += U * n("up", o) * n("do", o);
  for (int o1 = 0; o1 < n_orb; o1++) {
    for (int o2 = 0; o2 < n_orb; o2++) {
      if (o1 != o2) {
        H += 0.5 * U_prime * (n("up", o1) * n("do", o2) + n("do", o1) * n("up", o2));       // opposite spin
        H += 0.5 * (U_prime - J) * (n("up", o1) * n("up", o2) + n("do", o1) * n("do", o2)); // equal spin
        H += -J * c_dag("up", o1) * c("do", o1) * c_dag("do", o2) * c("up", o2);            // spin-flip
        H += -J * c_dag("up", o1) * c_dag("do", o1) * c("up", o2) * c("do", o2);            // pair-hopping
      }
    }
  }
  return H;
}

triqs::operators::many_body_operator_complex make_total_density_operator(int n_orb) {
  using triqs::operators::n;
  triqs::operators::many_body_operator_complex N_tot;
  for (int o = 0; o < n_orb; o++) N_tot += n("up", o) + n("do", o);
  return N_tot;
}

triqs::atom_diag::fundamental_operator_set get_fundamental_operator_set(int n_orb) {
  triqs::atom_diag::fundamental_operator_set fop_set;
  for (int o = 0; o < n_orb; o++) fop_set.insert("up", o);
  for (int o = 0; o < n_orb; o++) fop_set.insert("do", o);
  return fop_set;
}

triqs::atom_diag::atom_diag<true> two_band_atom_diag_helper() {
  // Kanamori interaction with n_orb = 2, U = 2.0, J = 0.2, and mu = (3*U - 5*J)/2 - 1.5
  auto H_kana  = make_kanamori_interaction(2, 2.0, 0.2, (3 * 2.0 - 5 * 0.2) / 2 - 1.5);
  auto fop_set = get_fundamental_operator_set(2);
  auto N_tot   = make_total_density_operator(2);
  auto ad      = triqs::atom_diag::atom_diag<true>(H_kana, fop_set, {N_tot});
  return ad;
}

std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> two_band_dense_helper(double beta, double Lambda, double eps) {

  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  auto ad                       = two_band_atom_diag_helper();
  auto H_dense                  = triqs_xca::atom_diag::get_full_h_atomic(ad);
  auto Gt_dense                 = Hmat_to_Gtmat(H_dense, beta, dlr_it_abs);
  auto [Fs_dense, F_dags_dense] = triqs_xca::atom_diag::get_operators_dense(ad);

  return std::make_tuple(Gt_dense, Fs_dense, F_dags_dense);
}

std::tuple<BlockDiagOpFun, BlockOpSymQuartet, nda::vector<int>> two_band_helper(double beta, double Lambda, double eps,
                                                                                nda::array_const_view<dcomplex, 3> hyb_coeffs) {
  auto dlr_rf        = build_dlr_rf(Lambda, eps);
  auto itops         = imtime_ops(Lambda, dlr_rf);
  auto const &dlr_it = itops.get_itnodes();
  auto dlr_it_abs    = cppdlr::rel2abs(dlr_it);

  auto ad                       = two_band_atom_diag_helper();
  auto [H_blocks, H_block_inds] = triqs_xca::atom_diag::get_hamiltonian_blocks(ad);
  auto Gt                       = nonint_gf_BDOF(H_blocks, H_block_inds, beta, dlr_it_abs); // pseudo-particle Green's function
  auto [Fq, sym_set_labels]     = triqs_xca::atom_diag::get_operators(ad, hyb_coeffs);

  return std::make_tuple(Gt, Fq, sym_set_labels);
}

// ---- test-only reference evaluator, moved out of triqs_xca::block_sparse ----
using cppdlr::_;
using cppdlr::imtime_ops;
using nda::linalg::matmul;

nda::array<dcomplex, 3> third_order_dense_partial(nda::array_const_view<dcomplex, 3> hyb, imtime_ops &itops, double beta,
                                                  nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> Fs,
                                                  nda::array_const_view<dcomplex, 3> F_dags) {
  nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
  nda::vector_const_view<double> dlr_it = itops.get_itnodes();
  // number of imaginary time nodes
  int r = dlr_it.extent(0);
  int N = Gt.extent(1);

  auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs
  auto hyb_refl        = nda::make_regular(-itops.reflect(hyb));
  auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);
  int n                = Fs.extent(0);

  // compute Fbars and Fdagbars
  auto Fdagbars  = nda::array<dcomplex, 4>(n, r, N, N);
  auto Fbarsrefl = nda::array<dcomplex, 4>(n, r, N, N);
  for (int lam = 0; lam < n; lam++) {
    for (int l = 0; l < r; l++) {
      for (int nu = 0; nu < n; nu++) {
        Fdagbars(lam, l, _, _) += hyb_coeffs(l, nu, lam) * F_dags(nu, _, _);
        Fbarsrefl(nu, l, _, _) += hyb_refl_coeffs(l, nu, lam) * Fs(lam, _, _);
      }
    }
  }

  // initialize self-energy
  nda::array<dcomplex, 3> Sigma(r, N, N), T(r, N, N), GKt(r, N, N), Tmu(r, N, N);
  nda::array<dcomplex, 4> Tkaps(n, r, N, N);

  // just {{0, 2}, {1, 4}, {3, 5}}, forward forward forward, omega_l,l` > 0 for now
  nda::vector<double> l{9, 7}, poles(2);
  for (int i = 0; i < 2; i++) poles(i) = dlr_rf(l(i));
  nda::array<int, 2> topology{{0, 2}, {1, 4}, {3, 5}};
  int m = 3;
  nda::vector<int> states(2 * m);
  states = 0;
  for (int s = 0; s < pow(n, m - 1); s++) {
    int s0 = s;
    for (int i = 1; i < m; i++) {
      states(topology(i, 0)) = s0 % n;
      states(topology(i, 1)) = s0 % n;
      s0                     = s0 / n;
    }

    T = Gt;

    if (poles(0) <= 0 && poles(1) <= 0) {
      int v = 1;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = cppdlr::k_it(dlr_it(t), -1 * poles(1)) * T(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 2;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) Tkaps(kap, t, _, _) = matmul(T(t, _, _), Fs(kap, _, _));
      }
      T = 0;
      for (int mu = 0; mu < n; mu++) {
        Tmu = 0;
        for (int kap = 0; kap < n; kap++) {
          for (int t = 0; t < r; t++) Tmu(t, _, _) += hyb(t, mu, kap) * Tkaps(kap, t, _, _);
        }
        for (int t = 0; t < r; t++) T(t, _, _) += matmul(F_dags(mu, _, _), Tmu(t, _, _));
      }
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 3;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = cppdlr::k_it(dlr_it(t), -1 * poles(0)) * T(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 4;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fdagbars(states(v), l(1), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = cppdlr::k_it(dlr_it(t), poles(1)) * T(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 5;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fdagbars(states(v), l(0), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = cppdlr::k_it(dlr_it(t), poles(0)) * T(t, _, _);

      T = T / (cppdlr::k_it(0, -1 * poles(0)) * cppdlr::k_it(0, -1 * poles(1)));
      Sigma += T;
    } else if (poles(0) <= 0 && poles(1) > 0) {
      int v = 1;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) GKt(t, _, _) = cppdlr::k_it(dlr_it(t), poles(1)) * Gt(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 2;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) Tkaps(kap, t, _, _) = matmul(T(t, _, _), Fs(kap, _, _));
      }
      T = 0;
      for (int mu = 0; mu < n; mu++) {
        Tmu = 0;
        for (int kap = 0; kap < n; kap++) {
          for (int t = 0; t < r; t++) Tmu(t, _, _) += hyb(t, mu, kap) * Tkaps(kap, t, _, _);
        }
        for (int t = 0; t < r; t++) T(t, _, _) += matmul(F_dags(mu, _, _), Tmu(t, _, _));
      }
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 3;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = cppdlr::k_it(dlr_it(t), -poles(0)) * T(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 4;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fdagbars(states(v), l(1), _, _), T(t, _, _));
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 5;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fdagbars(states(v), l(0), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = cppdlr::k_it(dlr_it(t), poles(0)) * T(t, _, _);

      T = T / (cppdlr::k_it(0, -poles(0)) * cppdlr::k_it(0, poles(1)) * cppdlr::k_it(0, poles(1)));
      Sigma += T;
    } else if (poles(0) > 0 && poles(1) <= 0) {
      int v = 1;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = cppdlr::k_it(dlr_it(t), -1 * poles(1)) * T(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 2;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) Tkaps(kap, t, _, _) = matmul(T(t, _, _), Fs(kap, _, _));
      }
      T = 0;
      for (int mu = 0; mu < n; mu++) {
        Tmu = 0;
        for (int kap = 0; kap < n; kap++) {
          for (int t = 0; t < r; t++) Tmu(t, _, _) += hyb(t, mu, kap) * Tkaps(kap, t, _, _);
        }
        for (int t = 0; t < r; t++) T(t, _, _) += matmul(F_dags(mu, _, _), Tmu(t, _, _));
      }
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 3;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) GKt(t, _, _) = cppdlr::k_it(dlr_it(t), poles(0)) * Gt(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 4;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fdagbars(states(v), l(1), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = cppdlr::k_it(dlr_it(t), poles(1)) * T(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 5;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fdagbars(states(v), l(0), _, _), T(t, _, _));

      T = T / (cppdlr::k_it(0, poles(0)) * cppdlr::k_it(0, -1 * poles(1)));
      Sigma += T;
    } else {
      int v = 1;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) GKt(t, _, _) = cppdlr::k_it(dlr_it(t), poles(1)) * Gt(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 2;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) Tkaps(kap, t, _, _) = matmul(T(t, _, _), Fs(kap, _, _));
      }
      T = 0;
      for (int mu = 0; mu < n; mu++) {
        Tmu = 0;
        for (int kap = 0; kap < n; kap++) {
          for (int t = 0; t < r; t++) Tmu(t, _, _) += hyb(t, mu, kap) * Tkaps(kap, t, _, _);
        }
        for (int t = 0; t < r; t++) T(t, _, _) += matmul(F_dags(mu, _, _), Tmu(t, _, _));
      }
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 3;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) GKt(t, _, _) = cppdlr::k_it(dlr_it(t), poles(0)) * cppdlr::k_it(dlr_it(t), poles(1)) * Gt(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 4;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fdagbars(states(v), l(1), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) GKt(t, _, _) = cppdlr::k_it(dlr_it(t), poles(0)) * Gt(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);

      v = 5;
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fdagbars(states(v), l(0), _, _), T(t, _, _));

      T = T / (cppdlr::k_it(0, poles(0)) * cppdlr::k_it(0, poles(1)) * cppdlr::k_it(0, poles(1)));
      Sigma += T;
    }
  }
  return Sigma;
}

