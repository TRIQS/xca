#include "block_sparse.hpp"
#include <cppdlr/dlr_kernels.hpp>
#include <cstddef>
#include <nda/algorithms.hpp>
#include <nda/declarations.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/linalg/matmul.hpp>

nda::array<dcomplex, 3> NCA_gf_dense(nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> Gt_refl,
                                     nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags) {

  std::size_t r = Gt.extent(0);
  std::size_t n = Fs.extent(0);
  nda::array<dcomplex, 3> gf(r, n, n); // initialize Green's function

  for (int lam = 0; lam < n; lam++) {
    for (int kap = 0; kap < n; kap++) {
      for (int t = 0; t < r; t++) {
        gf(t, lam, kap) += nda::trace(matmul(Gt_refl(t, _, _), matmul(Fs(lam, _, _), matmul(Gt(t, _, _), F_dags(kap, _, _)))));
      }
    }
  }

  return gf;
}

nda::array<dcomplex, 3> NCA_gf_bs(const BlockDiagOpFun &Gt, const BlockDiagOpFun &Gt_refl, const BlockOpSymQuartet &Fq) {

  std::size_t r = Gt.get_num_time_nodes();
  long n        = Fq.sym_set_labels.size();
  nda::array<dcomplex, 3> gf(r, n, n); // initialize Green's function
  nda::vector<int> ind_path(2), block_dims(3);

  long q = nda::max_element(Fq.sym_set_labels) + 1;
  for (int p_lam = 0; p_lam < q; p_lam++) {
    for (int p_kap = 0; p_kap < q; p_kap++) {
      for (int b = 0; b < Gt.get_num_block_cols(); b++) {
        // backward pass
        bool path_all_nonzero = true;
        int ip                = Fq.F_dags[p_kap].get_block_index(b);
        ind_path(0)           = ip;
        if (ip == -1) {
          path_all_nonzero = false;
        } else {
          block_dims(0) = Fq.F_dags[p_kap].get_block_size(b, 1);
          block_dims(1) = Fq.F_dags[p_kap].get_block_size(b, 0);
          if (Gt.get_zero_block_index(ip) == -1) {
            path_all_nonzero = false;
          } else {
            block_dims(2) = Fq.Fs[p_lam].get_block_size(ip, 0);
            ip            = Fq.Fs[p_lam].get_block_index(ip);
            ind_path(1)   = ip;
            if (ip == -1 || Gt_refl.get_zero_block_index(ip) == -1) { path_all_nonzero = false; }
          }
        }

        // matmuls
        if (path_all_nonzero) {
          for (int lam = 0; lam < Fq.sym_set_sizes(p_lam); lam++) {
            for (int kap = 0; kap < Fq.sym_set_sizes(p_kap); kap++) {
              long lam_orb = Fq.sym_set_to_orb(p_lam, lam);
              long kap_orb = Fq.sym_set_to_orb(p_kap, kap);
              for (int t = 0; t < r; t++) {
                gf(t, lam_orb, kap_orb) =
                   nda::trace(matmul(Gt_refl.get_block(ind_path(1))(t, _, _),
                                     matmul(Fq.Fs[p_lam].get_block(ind_path(0))(lam, _, _),
                                            matmul(Gt.get_block(ind_path(0))(t, _, _), Fq.F_dags[p_kap].get_block(b)(kap, _, _)))));
              }
            }
          }
        }
      }
    }
  }
  return gf;
}

nda::array<dcomplex, 3> OCA_gf_tpz(nda::array_const_view<dcomplex, 3> hyb_coeffs, nda::array_const_view<dcomplex, 3> hyb_refl_coeffs,
                                   imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> Gt_coeffs,
                                   nda::array_const_view<dcomplex, 3> Fs, int n_quad) {

  std::size_t N = Gt_coeffs.extent(1);
  std::size_t n = Fs.extent(0);
  nda::array<dcomplex, 3> F_dags(n, N, N);
  for (int i = 0; i < n; ++i) { F_dags(i, _, _) = nda::transpose(nda::conj(Fs(i, _, _))); }

  auto it_eq = cppdlr::eqptsrel(n_quad + 1);
  nda::array<dcomplex, 3> hyb_eq(n_quad + 1, n, n), hyb_refl_eq(n_quad + 1, n, n), Gt_eq(n_quad + 1, N, N), gf_eq(n_quad + 1, n, n);
  gf_eq = 0;
  for (int i = 0; i <= n_quad; i++) {
    hyb_eq(i, _, _)      = itops.coefs2eval(hyb_coeffs, it_eq(i));
    hyb_refl_eq(i, _, _) = itops.coefs2eval(hyb_refl_coeffs, it_eq(i));
    Gt_eq(i, _, _)       = itops.coefs2eval(Gt_coeffs, it_eq(i));
  }

  double dt = beta / n_quad;

  nda::array<dcomplex, 2> GFGFGFGF(N, N);
  for (int fb = 0; fb <= 1; fb++) {
    auto const &Flams = (fb) ? Fs(_, _, _) : F_dags(_, _, _);
    auto const &Fnus  = (fb) ? F_dags(_, _, _) : Fs(_, _, _);
    auto const &hyb   = (fb) ? hyb_eq : hyb_refl_eq;
    int sfM           = 1; // (fb) ? 1 : -1;

    for (int kap = 0; kap < n; kap++) {
      for (int mu = 0; mu < n; mu++) {
        for (int lam = 0; lam < n; lam++) {
          for (int nu = 0; nu < n; nu++) {
            for (int i = 1; i <= n_quad - 1; i++) {
              for (int i1 = 0; i1 <= i; i1++) {
                for (int i2 = i; i2 <= n_quad; i2++) {
                  double w = 1.0;
                  if (i1 == 0 || i1 == i) w = w / 2;
                  if (i2 == i || i2 == n_quad) w = w / 2;
                  GFGFGFGF = matmul(
                     Gt_eq(n_quad - i2, _, _),
                     matmul(Fnus(nu, _, _),
                            matmul(Gt_eq(i2 - i, _, _),
                                   matmul(Fs(mu, _, _),
                                          matmul(Gt_eq(i - i1, _, _), matmul(Flams(lam, _, _), matmul(Gt_eq(i1, _, _), F_dags(kap, _, _))))))));

                  gf_eq(i, mu, kap) += sfM * w * hyb(i2 - i1, nu, lam) * nda::trace(GFGFGFGF);
                }
              }
            }
          }
        }
      }
    }
  }

  gf_eq = dt * dt * gf_eq;

  return gf_eq;
}

void OCA_gf_dense_right(double beta, imtime_ops &itops, nda::vector_const_view<double> dlr_it, double omega_l, bool forward,
                        nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 2> Flam, nda::array_view<dcomplex, 3> T) {
  long r = Gt.extent(0);

  if (forward) {
    if (omega_l <= 0) {
      // 1. multiply F_lambda G(tau_1) K^-(tau_1)
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), -omega_l) * matmul(Flam, Gt(t, _, _)); }
      // 2. convolve by G
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);
    } else {
      // 1. multiply G(tau-tau_1) K^+(tau-tau_1) F_lambda
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), omega_l) * matmul(Gt(t, _, _), Flam); }
      // 2. convolve by G
      T = itops.convolve(beta, itops.vals2coefs(T), itops.vals2coefs(Gt), TIME_ORDERED);
    }
  } else {
    if (omega_l >= 0) {
      // 1. multiply F_lambda G(tau_1) K^+(tau_1)
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), omega_l) * matmul(Flam, Gt(t, _, _)); }
      // 2. convolve by G
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);
    } else {
      // 1. multiply G(tau-tau_1) K^-(tau-tau_1) F_lambda
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), -omega_l) * matmul(Gt(t, _, _), Flam); }
      // 2. convolve by G
      T = itops.convolve(beta, itops.vals2coefs(T), itops.vals2coefs(Gt), TIME_ORDERED);
    }
  }
}

void OCA_gf_dense_left(double beta, imtime_ops &itops, nda::vector_const_view<double> dlr_it, double omega_l, bool forward,
                       nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 2> Fbar, nda::array_view<dcomplex, 3> T) {
  long r = Gt.extent(0);

  if (forward) {
    if (omega_l <= 0) {
      // multiply G K^- Fbar
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), -omega_l) * matmul(Gt(t, _, _), Fbar); }
      // convolve by G
      T = itops.convolve(beta, itops.vals2coefs(T), itops.vals2coefs(Gt), TIME_ORDERED);
    } else {
      // multiply Fbar G K^+
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), omega_l) * matmul(Fbar, Gt(t, _, _)); }
      // convolve
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);
    }
  } else {
    if (omega_l > 0) {
      // multiply G K^+ Fbar
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), omega_l) * matmul(Gt(t, _, _), Fbar); }
      // convolve by G
      T = itops.convolve(beta, itops.vals2coefs(T), itops.vals2coefs(Gt), TIME_ORDERED);
    } else {
      // multiply Fbar G K^-
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), -omega_l) * matmul(Fbar, Gt(t, _, _)); }
      // convolve
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);
    }
  }
  // reflect
  T = itops.reflect(T);
}

nda::array<dcomplex, 3> OCA_gf_dense(nda::array_const_view<dcomplex, 3> hyb_coeffs, nda::array_const_view<dcomplex, 3> hyb_refl_coeffs,
                                     nda::vector_const_view<double> hyb_poles, imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> Gt,
                                     nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags) {

  nda::vector_const_view<double> dlr_it = itops.get_itnodes();
  long r                                = dlr_it.extent(0);
  long p                                = hyb_poles.extent(0);
  long N                                = Gt.extent(1);
  long n                                = Fs.extent(0);

  // compute Fbars and Fdagbars
  auto Fdagbars  = nda::array<dcomplex, 4>(n, p, N, N);
  auto Fbarsrefl = nda::array<dcomplex, 4>(n, p, N, N);
  for (long lam = 0; lam < n; lam++) {
    for (long l = 0; l < p; l++) {
      for (long nu = 0; nu < n; nu++) {
        Fdagbars(lam, l, _, _) += hyb_coeffs(l, nu, lam) * F_dags(nu, _, _);
        Fbarsrefl(nu, l, _, _) += hyb_refl_coeffs(l, nu, lam) * Fs(lam, _, _);
      }
    }
  }

  // initialize Green's function
  nda::array<dcomplex, 3> gf(r, n, n);
  gf = 0;

  // initialize temporary arrays
  nda::array<dcomplex, 3> T(r, N, N), U(r, N, N), Tnu(r, N, N);

  // loop over hybridization line directions
  for (int fb = 0; fb <= 1; fb++) {
    // fb = 1 for forward line, else = 0
    auto const &F1   = (fb) ? Fs : F_dags;
    auto const &Fbar = (fb) ? Fdagbars : Fbarsrefl;

    for (int nu = 0; nu < n; nu++) {
      Tnu = 0;
      for (int lam = 0; lam < n; lam++) {
        for (int l = 0; l < p; l++) {
          T = 0;
          U = 0;
          OCA_gf_dense_right(beta, itops, dlr_it, hyb_poles(l), fb, Gt, F1(lam, _, _), T);
          OCA_gf_dense_left(beta, itops, dlr_it, hyb_poles(l), fb, Gt, Fbar(lam, l, _, _), U);
          if (hyb_poles(l) <= 0) {
            for (int t = 0; t < r; t++) { Tnu(t, _, _) += matmul(U(t, _, _), matmul(Fs(nu, _, _), T(t, _, _))) / k_it(0, -hyb_poles(l)); }
          } else {
            for (int t = 0; t < r; t++) { Tnu(t, _, _) += matmul(U(t, _, _), matmul(Fs(nu, _, _), T(t, _, _))) / k_it(0, hyb_poles(l)); }
          }
        }
      }
      // if (fb == 0) Tnu = -1 * Tnu; // minus sign for backward line
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) { gf(t, nu, kap) += nda::trace(matmul(Tnu(t, _, _), F_dags(kap, _, _))); }
      }
    }
  }
  return gf;
}

void OCA_gf_bs_right(double beta, imtime_ops &itops, nda::vector_const_view<double> dlr_it, double omega_l, bool forward, const BlockDiagOpFun &Gt,
                     const BlockOpSymSet &Flam, long lam, nda::array_view<dcomplex, 3> T, nda::vector_const_view<int> ind_path,
                     nda::vector_const_view<int> block_dims) {
  long r = Gt.get_num_time_nodes();

  if (forward) {
    if (omega_l <= 0) {
      // multiply F_lambda G(tau_1) K^-(tau_1)
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(2)), range(0, block_dims(1))) =
           k_it(dlr_it(t), -omega_l) * matmul(Flam.get_block(ind_path(0))(lam, _, _), Gt.get_block(ind_path(0))(t, _, _));
      }
      // convolve by G
      T(_, range(0, block_dims(2)), range(0, block_dims(1))) =
         itops.convolve(beta, itops.vals2coefs(Gt.get_block(ind_path(1))),
                        itops.vals2coefs(T(_, range(0, block_dims(2)), range(0, block_dims(1)))), TIME_ORDERED);
    } else {
      // multiply G(tau-tau_1) K^+(tau-tau_1) F_lambda
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(2)), range(0, block_dims(1))) =
           k_it(dlr_it(t), omega_l) * matmul(Gt.get_block(ind_path(1))(t, _, _), Flam.get_block(ind_path(0))(lam, _, _));
      }
      // convolve by G
      T(_, range(0, block_dims(2)), range(0, block_dims(1))) =
         itops.convolve(beta, itops.vals2coefs(T(_, range(0, block_dims(2)), range(0, block_dims(1)))),
                        itops.vals2coefs(Gt.get_block(ind_path(0))), TIME_ORDERED);
    }
  } else {
    if (omega_l >= 0) {
      // multiply F_lambda G(tau_1) K^+(tau_1)
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(2)), range(0, block_dims(1))) =
           k_it(dlr_it(t), omega_l) * matmul(Flam.get_block(ind_path(0))(lam, _, _), Gt.get_block(ind_path(0))(t, _, _));
      }
      // convolve by G
      T(_, range(0, block_dims(2)), range(0, block_dims(1))) =
         itops.convolve(beta, itops.vals2coefs(Gt.get_block(ind_path(1))),
                        itops.vals2coefs(T(_, range(0, block_dims(2)), range(0, block_dims(1)))), TIME_ORDERED);
    } else {
      // multiply G(tau-tau_1) K^-(tau-tau_1) F_lambda
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(2)), range(0, block_dims(1))) =
           k_it(dlr_it(t), -omega_l) * matmul(Gt.get_block(ind_path(1))(t, _, _), Flam.get_block(ind_path(0))(lam, _, _));
      }
      // convolve by G
      T(_, range(0, block_dims(2)), range(0, block_dims(1))) =
         itops.convolve(beta, itops.vals2coefs(T(_, range(0, block_dims(2)), range(0, block_dims(1)))),
                        itops.vals2coefs(Gt.get_block(ind_path(0))), TIME_ORDERED);
    }
  }
}

void OCA_gf_bs_left(double beta, imtime_ops &itops, nda::vector_const_view<double> dlr_it, double omega_l, bool forward, const BlockDiagOpFun &Gt,
                    const BlockOpSymSetBar &Fbar, long lam, long pole_ind, nda::array_view<dcomplex, 3> T, nda::vector_const_view<int> ind_path,
                    nda::vector_const_view<int> block_dims) {
  long r = Gt.get_num_time_nodes();

  if (forward) {
    if (omega_l <= 0) {
      // multiply G K^- Fbar
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(4)), range(0, block_dims(3))) =
           k_it(dlr_it(t), -omega_l) * matmul(Gt.get_block(ind_path(3))(t, _, _), Fbar.get_block(ind_path(2))(lam, pole_ind, _, _));
      }
      // convolve by G
      T(_, range(0, block_dims(4)), range(0, block_dims(3))) =
         itops.convolve(beta, itops.vals2coefs(T(_, range(0, block_dims(4)), range(0, block_dims(3)))),
                        itops.vals2coefs(Gt.get_block(ind_path(2))), TIME_ORDERED);
    } else {
      // multiply Fbar G K^+
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(4)), range(0, block_dims(3))) =
           k_it(dlr_it(t), omega_l) * matmul(Fbar.get_block(ind_path(2))(lam, pole_ind, _, _), Gt.get_block(ind_path(2))(t, _, _));
      }
      // convolve
      T(_, range(0, block_dims(4)), range(0, block_dims(3))) =
         itops.convolve(beta, itops.vals2coefs(Gt.get_block(ind_path(3))),
                        itops.vals2coefs(T(_, range(0, block_dims(4)), range(0, block_dims(3)))), TIME_ORDERED);
    }
  } else {
    if (omega_l > 0) {
      // multiply G K^+ Fbar
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(4)), range(0, block_dims(3))) =
           k_it(dlr_it(t), omega_l) * matmul(Gt.get_block(ind_path(3))(t, _, _), Fbar.get_block(ind_path(2))(lam, pole_ind, _, _));
      }
      // convolve by G
      T(_, range(0, block_dims(4)), range(0, block_dims(3))) =
         itops.convolve(beta, itops.vals2coefs(T(_, range(0, block_dims(4)), range(0, block_dims(3)))),
                        itops.vals2coefs(Gt.get_block(ind_path(2))), TIME_ORDERED);
    } else {
      // multiply Fbar G K^-
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(4)), range(0, block_dims(3))) =
           k_it(dlr_it(t), -omega_l) * matmul(Fbar.get_block(ind_path(2))(lam, pole_ind, _, _), Gt.get_block(ind_path(2))(t, _, _));
      }
      // convolve
      T(_, range(0, block_dims(4)), range(0, block_dims(3))) =
         itops.convolve(beta, itops.vals2coefs(Gt.get_block(ind_path(3))),
                        itops.vals2coefs(T(_, range(0, block_dims(4)), range(0, block_dims(3)))), TIME_ORDERED);
    }
  }
  // reflect. Why does only reflecting all of T work?
  T = itops.reflect(T);
}

nda::array<dcomplex, 3> OCA_gf_bs(nda::vector_const_view<double> hyb_poles, imtime_ops &itops, double beta, const BlockDiagOpFun &Gt,
                                  const BlockOpSymQuartet &Fq) {

  nda::vector_const_view<double> dlr_it = itops.get_itnodes();
  long r                                = dlr_it.extent(0);
  long p                                = hyb_poles.extent(0);
  long n                                = Fq.sym_set_labels.size();
  int N                                 = Gt.get_max_block_size(); // max block size of Gt for temporary arrays
  nda::array<dcomplex, 3> gf(r, n, n);
  gf = 0;
  nda::vector<int> ind_path(4), block_dims(5);
  nda::array<dcomplex, 3> T(r, N, N), U(r, N, N), Tmu(r, N, N);

  long q = nda::max_element(Fq.sym_set_labels) + 1;
  // loop over hybridization line directions
  for (int fb = 0; fb <= 1; fb++) {
    // fb = 1 for forward line, else = 0
    auto const &F1   = (fb) ? Fq.Fs : Fq.F_dags;
    auto const &Fbar = (fb) ? Fq.F_dag_bars : Fq.F_bars_refl;
    for (int p_lam = 0; p_lam < q; p_lam++) {
      for (int p_kap = 0; p_kap < q; p_kap++) {
        for (int p_mu = 0; p_mu < q; p_mu++) {
          for (int b = 0; b < Gt.get_num_block_cols(); b++) {
            // backward pass
            bool path_all_nonzero = true;
            int ip                = Fq.F_dags[p_kap].get_block_index(b);
            ind_path(0)           = ip;
            if (ip == -1) {
              path_all_nonzero = false;
            } else {
              block_dims(0) = Fq.F_dags[p_kap].get_block_size(b, 1);
              block_dims(1) = Fq.F_dags[p_kap].get_block_size(b, 0);
              if (Gt.get_zero_block_index(ip) == -1) {
                path_all_nonzero = false;
              } else {
                block_dims(2) = F1[p_lam].get_block_size(ip, 0);
                ip            = F1[p_lam].get_block_index(ip);
                ind_path(1)   = ip;
                if (ip == -1 || Gt.get_zero_block_index(ip) == -1) {
                  path_all_nonzero = false;
                } else {
                  block_dims(3) = Fq.Fs[p_mu].get_block_size(ip, 0);
                  ip            = Fq.Fs[p_mu].get_block_index(ip);
                  ind_path(2)   = ip;
                  if (ip == -1 || Gt.get_zero_block_index(ip) == -1) {
                    path_all_nonzero = false;
                  } else {
                    block_dims(4) = Fbar[p_lam].get_block_size(ip, 0);
                    ip            = Fbar[p_lam].get_block_index(ip);
                    ind_path(3)   = ip;
                    if (ip == -1 || Gt.get_zero_block_index(ip) == -1) { path_all_nonzero = false; }
                  }
                }
              }
            }

            // matmuls
            if (path_all_nonzero) {
              for (int mu = 0; mu < Fq.sym_set_sizes(p_mu); mu++) {
                Tmu         = 0;
                long mu_orb = Fq.sym_set_to_orb(p_mu, mu);
                for (int lam = 0; lam < Fq.sym_set_sizes(p_lam); lam++) {
                  for (int l = 0; l < p; l++) {
                    T = 0;
                    U = 0;
                    OCA_gf_bs_right(beta, itops, dlr_it, hyb_poles(l), fb, Gt, F1[p_lam], lam, T, ind_path, block_dims);
                    OCA_gf_bs_left(beta, itops, dlr_it, hyb_poles(l), fb, Gt, Fbar[p_lam], lam, l, U, ind_path, block_dims);
                    if (hyb_poles(l) <= 0) {
                      for (int t = 0; t < r; t++) {
                        Tmu(t, range(0, block_dims(4)), range(0, block_dims(1))) +=
                           matmul(U(t, range(0, block_dims(4)), range(0, block_dims(3))),
                                  matmul(Fq.Fs[p_mu].get_block(ind_path(1))(mu, _, _), T(t, range(0, block_dims(2)), range(0, block_dims(1)))))
                           / k_it(0, -hyb_poles(l));
                      }
                    } else {
                      for (int t = 0; t < r; t++) {
                        Tmu(t, range(0, block_dims(4)), range(0, block_dims(1))) +=
                           matmul(U(t, range(0, block_dims(4)), range(0, block_dims(3))),
                                  matmul(Fq.Fs[p_mu].get_block(ind_path(1))(mu, _, _), T(t, range(0, block_dims(2)), range(0, block_dims(1)))))
                           / k_it(0, hyb_poles(l));
                      }
                    }
                  }
                }
                for (int kap = 0; kap < Fq.sym_set_sizes(p_kap); kap++) {
                  long kap_orb = Fq.sym_set_to_orb(p_kap, kap);
                  for (int t = 0; t < r; t++) {
                    gf(t, mu_orb, kap_orb) +=
                       nda::trace(matmul(Tmu(t, range(0, block_dims(4)), range(0, block_dims(1))), Fq.F_dags[p_kap].get_block(b)(kap, _, _)));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  return gf;
}