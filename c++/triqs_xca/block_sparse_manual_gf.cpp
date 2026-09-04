#include "triqs_xca/block_sparse_manual_gf.hpp"

namespace triqs_xca::block_sparse {

  using cppdlr::_;

  using nda::range;
  using nda::trace;
  using nda::linalg::matmul;

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
    gf = 0;
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
                  gf(t, lam_orb, kap_orb) +=
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
      // hyb_refl_coeffs is the plain reflection of the hybridization and carries no sign of its own,
      // so a backward line needs no extra factor here.

      for (int kap = 0; kap < n; kap++) {
        for (int mu = 0; mu < n; mu++) {
          for (int lam = 0; lam < n; lam++) {
            for (int nu = 0; nu < n; nu++) {
              // the whole output grid is covered, but both endpoints are degenerate: at i = 0 the tau_1
              // integral runs over the empty range [0, 0], and at i = n_quad the tau_2 integral runs over
              // [beta, beta], so each contributes nothing and g stays at its exact endpoint value of zero.
              // They are skipped rather than summed because a one-point trapezoid rule would not give zero.
              for (int i = 0; i <= n_quad; i++) {
                if (i == 0 || i == n_quad) continue;
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

                    gf_eq(i, mu, kap) += w * hyb(i2 - i1, nu, lam) * nda::trace(GFGFGFGF);
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

  nda::array<dcomplex, 3> third_order_gf_tpz(nda::array_const_view<dcomplex, 3> hyb_coeffs, nda::array_const_view<dcomplex, 3> hyb_refl_coeffs,
                                             imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> Gt_coeffs,
                                             nda::array_const_view<dcomplex, 3> Fs, int n_quad) {
    // Third-order single-particle Green's function diagram for topology {{0,3},{1,4},{2,5}} (the fully-crossing
    // topology), evaluated by direct trapezoidal quadrature.
    int N = Gt_coeffs.extent(1);
    int n = Fs.extent(0);

    // get F^dagger operators
    nda::array<dcomplex, 3> F_dags(n, N, N);
    for (int i = 0; i < n; ++i) { F_dags(i, _, _) = nda::transpose(nda::conj(Fs(i, _, _))); }

    // get equispaced grid and evaluate functions on grid
    auto it_eq = cppdlr::eqptsrel(n_quad + 1);
    nda::array<dcomplex, 3> hyb_eq(n_quad + 1, n, n), hyb_refl_eq(n_quad + 1, n, n), Gt_eq(n_quad + 1, N, N), gf_eq(n_quad + 1, n, n);
    gf_eq = 0;
    for (int i = 0; i <= n_quad; i++) {
      hyb_eq(i, _, _)      = itops.coefs2eval(hyb_coeffs, it_eq(i));
      hyb_refl_eq(i, _, _) = itops.coefs2eval(hyb_refl_coeffs, it_eq(i));
      Gt_eq(i, _, _)       = itops.coefs2eval(Gt_coeffs, it_eq(i));
    }

    double dt = beta / n_quad;

    nda::array<dcomplex, 2> chain(N, N);
    // loop over the directions of the two hybridization lines
    for (int fb1 = 0; fb1 <= 1; fb1++) {
      for (int fb2 = 0; fb2 <= 1; fb2++) {
        // fb = 1 for forward line, else = 0. fb1 is the direction of line {1,4}, fb2 that of line {2,5}.
        auto const &F1list = (fb1) ? Fs(_, _, _) : F_dags(_, _, _);
        auto const &F4list = (fb1) ? F_dags(_, _, _) : Fs(_, _, _);
        auto const &F2list = (fb2) ? Fs(_, _, _) : F_dags(_, _, _);
        auto const &F5list = (fb2) ? F_dags(_, _, _) : Fs(_, _, _);
        auto const &hyb1   = (fb1) ? hyb_eq : hyb_refl_eq;
        auto const &hyb2   = (fb2) ? hyb_eq : hyb_refl_eq;
        // hyb_refl_coeffs is the plain reflection of the hybridization and carries no sign of its own,
        // so a backward line needs no extra factor here.

        for (int kap = 0; kap < n; kap++) {
          for (int mu = 0; mu < n; mu++) {
            for (int o1 = 0; o1 < n; o1++) {
              for (int o2 = 0; o2 < n; o2++) {
                for (int o4 = 0; o4 < n; o4++) {
                  for (int o5 = 0; o5 < n; o5++) {
                    // as in OCA_gf_tpz, both endpoints are degenerate: at i = 0 the vertices on [0, tau]
                    // collapse to a point and at i = n_quad those on [tau, beta] do, so each contributes
                    // nothing and g keeps its exact endpoint value of zero
                    for (int i = 0; i <= n_quad; i++) {
                      if (i == 0 || i == n_quad) continue;
                      // vertices 1 and 2, both on [0, tau]
                      for (int j2 = 0; j2 <= i; j2++) {
                        double w2 = (j2 == 0 || j2 == i) ? 0.5 : 1.0;
                        for (int j1 = 0; j1 <= j2; j1++) {
                          double w1 = w2 * ((j1 == 0 || j1 == j2) ? 0.5 : 1.0);
                          // the [0, tau] part of the operator product: G(tau - tau_3) ... G(tau_4 - 0) F^dag
                          auto right =
                             matmul(Gt_eq(i - j2, _, _),
                                    matmul(F2list(o2, _, _),
                                           matmul(Gt_eq(j2 - j1, _, _), matmul(F1list(o1, _, _), matmul(Gt_eq(j1, _, _), F_dags(kap, _, _))))));
                          // vertices 4 and 5, both on [tau, beta]
                          for (int j4 = i; j4 <= n_quad; j4++) {
                            double w4 = w1 * ((j4 == i || j4 == n_quad) ? 0.5 : 1.0);
                            for (int j5 = j4; j5 <= n_quad; j5++) {
                              double w = w4 * ((j5 == j4 || j5 == n_quad) ? 0.5 : 1.0);
                              // the [tau, beta] part: G(beta - tau_1) ... G(tau_2 - tau)
                              auto left =
                                 matmul(Gt_eq(n_quad - j5, _, _),
                                        matmul(F5list(o5, _, _), matmul(Gt_eq(j5 - j4, _, _), matmul(F4list(o4, _, _), Gt_eq(j4 - i, _, _)))));
                              chain = matmul(left, matmul(Fs(mu, _, _), right));

                              gf_eq(i, mu, kap) += w * hyb1(j4 - j1, o4, o1) * hyb2(j5 - j2, o5, o2) * nda::trace(chain);
                            } // sum over j5
                          } // sum over j4
                        } // sum over j1
                      } // sum over j2
                    } // sum over i
                  } // sum over o5
                } // sum over o4
              } // sum over o2
            } // sum over o1
          } // sum over mu
        } // sum over kappa

      } // sum over fb2
    } // sum over fb1

    gf_eq = dt * dt * dt * dt * gf_eq;

    return gf_eq;
  }

  void OCA_gf_dense_right(double beta, imtime_ops &itops, nda::vector_const_view<double> dlr_it, double omega_l, bool forward,
                          nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 2> Flam, nda::array_view<dcomplex, 3> T) {
    long r = Gt.extent(0);

    if (forward) {
      if (omega_l <= 0) {
        // 1. multiply F_lambda G(tau_1) K^-(tau_1)
        for (int t = 0; t < r; t++) { T(t, _, _) = cppdlr::k_it(dlr_it(t), -omega_l) * matmul(Flam, Gt(t, _, _)); }
        // 2. convolve by G
        T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);
      } else {
        // 1. multiply G(tau-tau_1) K^+(tau-tau_1) F_lambda
        for (int t = 0; t < r; t++) { T(t, _, _) = cppdlr::k_it(dlr_it(t), omega_l) * matmul(Gt(t, _, _), Flam); }
        // 2. convolve by G
        T = itops.convolve(beta, itops.vals2coefs(T), itops.vals2coefs(Gt), cppdlr::TIME_ORDERED);
      }
    } else {
      if (omega_l >= 0) {
        // 1. multiply F_lambda G(tau_1) K^+(tau_1)
        for (int t = 0; t < r; t++) { T(t, _, _) = cppdlr::k_it(dlr_it(t), omega_l) * matmul(Flam, Gt(t, _, _)); }
        // 2. convolve by G
        T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);
      } else {
        // 1. multiply G(tau-tau_1) K^-(tau-tau_1) F_lambda
        for (int t = 0; t < r; t++) { T(t, _, _) = cppdlr::k_it(dlr_it(t), -omega_l) * matmul(Gt(t, _, _), Flam); }
        // 2. convolve by G
        T = itops.convolve(beta, itops.vals2coefs(T), itops.vals2coefs(Gt), cppdlr::TIME_ORDERED);
      }
    }
  }

  void OCA_gf_dense_left(double beta, imtime_ops &itops, nda::vector_const_view<double> dlr_it, double omega_l, bool forward,
                         nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 2> Fbar, nda::array_view<dcomplex, 3> T) {
    long r = Gt.extent(0);

    if (forward) {
      if (omega_l <= 0) {
        // multiply G K^- Fbar
        for (int t = 0; t < r; t++) { T(t, _, _) = cppdlr::k_it(dlr_it(t), -omega_l) * matmul(Gt(t, _, _), Fbar); }
        // convolve by G
        T = itops.convolve(beta, itops.vals2coefs(T), itops.vals2coefs(Gt), cppdlr::TIME_ORDERED);
      } else {
        // multiply Fbar G K^+
        for (int t = 0; t < r; t++) { T(t, _, _) = cppdlr::k_it(dlr_it(t), omega_l) * matmul(Fbar, Gt(t, _, _)); }
        // convolve
        T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);
      }
    } else {
      if (omega_l > 0) {
        // multiply G K^+ Fbar
        for (int t = 0; t < r; t++) { T(t, _, _) = cppdlr::k_it(dlr_it(t), omega_l) * matmul(Gt(t, _, _), Fbar); }
        // convolve by G
        T = itops.convolve(beta, itops.vals2coefs(T), itops.vals2coefs(Gt), cppdlr::TIME_ORDERED);
      } else {
        // multiply Fbar G K^-
        for (int t = 0; t < r; t++) { T(t, _, _) = cppdlr::k_it(dlr_it(t), -omega_l) * matmul(Fbar, Gt(t, _, _)); }
        // convolve
        T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);
      }
    }
    // reflect
    T = itops.reflect(T);
  }

  nda::array<dcomplex, 3> OCA_gf_dense(nda::array_const_view<dcomplex, 3> hyb_coeffs, nda::array_const_view<dcomplex, 3> hyb_refl_coeffs,
                                       nda::vector_const_view<double> hyb_poles, imtime_ops &itops, double beta,
                                       nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> Fs,
                                       nda::array_const_view<dcomplex, 3> F_dags) {

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
              for (int t = 0; t < r; t++) { Tnu(t, _, _) += matmul(U(t, _, _), matmul(Fs(nu, _, _), T(t, _, _))) / cppdlr::k_it(0, -hyb_poles(l)); }
            } else {
              for (int t = 0; t < r; t++) { Tnu(t, _, _) += matmul(U(t, _, _), matmul(Fs(nu, _, _), T(t, _, _))) / cppdlr::k_it(0, hyb_poles(l)); }
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
             cppdlr::k_it(dlr_it(t), -omega_l) * matmul(Flam.get_block(ind_path(0))(lam, _, _), Gt.get_block(ind_path(0))(t, _, _));
        }
        // convolve by G
        T(_, range(0, block_dims(2)), range(0, block_dims(1))) =
           itops.convolve(beta, itops.vals2coefs(Gt.get_block(ind_path(1))), itops.vals2coefs(T(_, range(0, block_dims(2)), range(0, block_dims(1)))),
                          cppdlr::TIME_ORDERED);
      } else {
        // multiply G(tau-tau_1) K^+(tau-tau_1) F_lambda
        for (int t = 0; t < r; t++) {
          T(t, range(0, block_dims(2)), range(0, block_dims(1))) =
             cppdlr::k_it(dlr_it(t), omega_l) * matmul(Gt.get_block(ind_path(1))(t, _, _), Flam.get_block(ind_path(0))(lam, _, _));
        }
        // convolve by G
        T(_, range(0, block_dims(2)), range(0, block_dims(1))) =
           itops.convolve(beta, itops.vals2coefs(T(_, range(0, block_dims(2)), range(0, block_dims(1)))), itops.vals2coefs(Gt.get_block(ind_path(0))),
                          cppdlr::TIME_ORDERED);
      }
    } else {
      if (omega_l >= 0) {
        // multiply F_lambda G(tau_1) K^+(tau_1)
        for (int t = 0; t < r; t++) {
          T(t, range(0, block_dims(2)), range(0, block_dims(1))) =
             cppdlr::k_it(dlr_it(t), omega_l) * matmul(Flam.get_block(ind_path(0))(lam, _, _), Gt.get_block(ind_path(0))(t, _, _));
        }
        // convolve by G
        T(_, range(0, block_dims(2)), range(0, block_dims(1))) =
           itops.convolve(beta, itops.vals2coefs(Gt.get_block(ind_path(1))), itops.vals2coefs(T(_, range(0, block_dims(2)), range(0, block_dims(1)))),
                          cppdlr::TIME_ORDERED);
      } else {
        // multiply G(tau-tau_1) K^-(tau-tau_1) F_lambda
        for (int t = 0; t < r; t++) {
          T(t, range(0, block_dims(2)), range(0, block_dims(1))) =
             cppdlr::k_it(dlr_it(t), -omega_l) * matmul(Gt.get_block(ind_path(1))(t, _, _), Flam.get_block(ind_path(0))(lam, _, _));
        }
        // convolve by G
        T(_, range(0, block_dims(2)), range(0, block_dims(1))) =
           itops.convolve(beta, itops.vals2coefs(T(_, range(0, block_dims(2)), range(0, block_dims(1)))), itops.vals2coefs(Gt.get_block(ind_path(0))),
                          cppdlr::TIME_ORDERED);
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
             cppdlr::k_it(dlr_it(t), -omega_l) * matmul(Gt.get_block(ind_path(3))(t, _, _), Fbar.get_block(ind_path(2))(lam, pole_ind, _, _));
        }
        // convolve by G
        T(_, range(0, block_dims(4)), range(0, block_dims(3))) =
           itops.convolve(beta, itops.vals2coefs(T(_, range(0, block_dims(4)), range(0, block_dims(3)))), itops.vals2coefs(Gt.get_block(ind_path(2))),
                          cppdlr::TIME_ORDERED);
      } else {
        // multiply Fbar G K^+
        for (int t = 0; t < r; t++) {
          T(t, range(0, block_dims(4)), range(0, block_dims(3))) =
             cppdlr::k_it(dlr_it(t), omega_l) * matmul(Fbar.get_block(ind_path(2))(lam, pole_ind, _, _), Gt.get_block(ind_path(2))(t, _, _));
        }
        // convolve
        T(_, range(0, block_dims(4)), range(0, block_dims(3))) =
           itops.convolve(beta, itops.vals2coefs(Gt.get_block(ind_path(3))), itops.vals2coefs(T(_, range(0, block_dims(4)), range(0, block_dims(3)))),
                          cppdlr::TIME_ORDERED);
      }
    } else {
      if (omega_l > 0) {
        // multiply G K^+ Fbar
        for (int t = 0; t < r; t++) {
          T(t, range(0, block_dims(4)), range(0, block_dims(3))) = cppdlr::k_it(dlr_it(t), omega_l)
             * matmul(Gt.get_block(ind_path(3))(t, _, _), -Fbar.get_block(ind_path(2))(lam, pole_ind, _, _)); // Add -1 sign for reflected F_bar
        }
        // convolve by G
        T(_, range(0, block_dims(4)), range(0, block_dims(3))) =
           itops.convolve(beta, itops.vals2coefs(T(_, range(0, block_dims(4)), range(0, block_dims(3)))), itops.vals2coefs(Gt.get_block(ind_path(2))),
                          cppdlr::TIME_ORDERED);
      } else {
        // multiply Fbar G K^-
        for (int t = 0; t < r; t++) {
          T(t, range(0, block_dims(4)), range(0, block_dims(3))) = cppdlr::k_it(dlr_it(t), -omega_l)
             * matmul(-Fbar.get_block(ind_path(2))(lam, pole_ind, _, _), Gt.get_block(ind_path(2))(t, _, _)); // Add -1 sign for reflected F_bar
        }
        // convolve
        T(_, range(0, block_dims(4)), range(0, block_dims(3))) =
           itops.convolve(beta, itops.vals2coefs(Gt.get_block(ind_path(3))), itops.vals2coefs(T(_, range(0, block_dims(4)), range(0, block_dims(3)))),
                          cppdlr::TIME_ORDERED);
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
                             / cppdlr::k_it(0, -hyb_poles(l));
                        }
                      } else {
                        for (int t = 0; t < r; t++) {
                          Tmu(t, range(0, block_dims(4)), range(0, block_dims(1))) +=
                             matmul(U(t, range(0, block_dims(4)), range(0, block_dims(3))),
                                    matmul(Fq.Fs[p_mu].get_block(ind_path(1))(mu, _, _), T(t, range(0, block_dims(2)), range(0, block_dims(1)))))
                             / cppdlr::k_it(0, hyb_poles(l));
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

} // namespace triqs_xca::block_sparse