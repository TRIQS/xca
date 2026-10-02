#include "triqs_xca/dense/manual.hpp"

namespace triqs_xca::dense {

  using cppdlr::_;

  using nda::range;
  using nda::trace;
  using nda::linalg::matmul;

  nda::array<dcomplex, 3> NCA_dense(nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl,
                                    nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> Fs,
                                    nda::array_const_view<dcomplex, 3> F_dags) {

    // initialize self-energy, with same shape as Gt
    int r = Gt.extent(0);
    int N = Gt.extent(1);
    nda::array<dcomplex, 3> Sigma(r, N, N);
    int n = Fs.extent(0);

    for (int fb = 0; fb <= 1; fb++) {
      // fb = 1 for forward line, 0 for backward line
      auto const &F1list = (fb) ? Fs : F_dags;
      auto const &F2list = (fb) ? F_dags : Fs;

      for (int lam = 0; lam < n; lam++) {
        for (int kap = 0; kap < n; kap++) {
          auto F1 = F1list(kap, _, _);
          auto F2 = F2list(lam, _, _);

          for (int t = 0; t < r; t++) {
            if (fb == 1) {
              Sigma(t, _, _) += hyb(t, lam, kap) * matmul(F2, matmul(Gt(t, _, _), F1));
            } else {
              // backward: F1 = F_dag(kap) and F2 = F(lam), so the daggered index kap comes first
              Sigma(t, _, _) += hyb_refl(t, kap, lam) * matmul(F2, matmul(Gt(t, _, _), F1));
            }
          }
        }
      }
    }

    return Sigma;
  }

  void OCA_dense_right_in_place(double beta, imtime_ops &itops, nda::vector_const_view<double> dlr_it, double omega_l, bool forward,
                                nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 2> Flam, nda::array_view<dcomplex, 3> T) {

    int r = Gt.extent(0);

    if (forward) {
      if (omega_l <= 0) {
        // 1. multiply F_lambda G(tau_1) K^-(tau_1)
        for (int t = 0; t < r; t++) { T(t, _, _) = cppdlr::k_it(dlr_it(t), -omega_l) * matmul(Flam, Gt(t, _, _)); }
        // 2. convolve by G
        T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);
      } else {
        // 1. multiply G(tau_2-tau_1) K^+(tau_2-tau_1) F_lambda
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
        // 1. multiply G(tau_2-tau_1) K^-(tau_2-tau_1) F_lambda
        for (int t = 0; t < r; t++) { T(t, _, _) = cppdlr::k_it(dlr_it(t), -omega_l) * matmul(Gt(t, _, _), Flam); }
        // 2. convolve by G
        T = itops.convolve(beta, itops.vals2coefs(T), itops.vals2coefs(Gt), cppdlr::TIME_ORDERED);
      }
    }
  }

  void OCA_dense_middle_in_place(bool forward, nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl,
                                 nda::array_const_view<dcomplex, 3> Fkaps, nda::array_const_view<dcomplex, 3> Fmus, nda::array_view<dcomplex, 3> T,
                                 nda::array_view<dcomplex, 4> Tkaps, nda::array_view<dcomplex, 3> Tmu) {
    int num_Fs = Fkaps.extent(0);
    int r      = Tmu.extent(0);
    // 3. for each kappa, multiply by F_kappa from right
    for (int kap = 0; kap < num_Fs; kap++) {
      for (int t = 0; t < r; t++) { Tkaps(kap, t, _, _) = matmul(T(t, _, _), Fkaps(kap, _, _)); }
    }

    T = 0;
    // 4. for each mu, kap, mult by Delta_mu_kap and sum kap
    for (int mu = 0; mu < num_Fs; mu++) {
      Tmu = 0;
      for (int kap = 0; kap < num_Fs; kap++) {
        for (int t = 0; t < r; t++) {
          if (forward) {
            Tmu(t, _, _) += hyb(t, mu, kap) * Tkaps(kap, t, _, _);
          } else {
            // backward: the daggered index kap comes first
            Tmu(t, _, _) += hyb_refl(t, kap, mu) * Tkaps(kap, t, _, _);
          }
        }
      }
      // 5. multiply by F^dag_mu and sum over mu
      for (int t = 0; t < r; t++) {
        T(t, _, _) += matmul(Fmus(mu, _, _), Tmu(t, _, _)); // TODO ??? +=
      }
    }
  }

  void OCA_dense_left_in_place(double beta, imtime_ops &itops, nda::vector_const_view<double> dlr_it, double omega_l, bool forward,
                               nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 2> Fbar, nda::array_view<dcomplex, 3> T,
                               nda::array_view<dcomplex, 3> GKt) {

    int r = Gt.extent(0);

    if (forward) {
      if (omega_l <= 0) {
        // 6. convolve by G
        T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);
        // 7. multiply by Fbar
        for (int t = 0; t < r; t++) { T(t, _, _) = matmul(Fbar, T(t, _, _)); }
      } else {
        // 6. convolve by G K^+
        for (int t = 0; t < r; t++) { GKt(t, _, _) = cppdlr::k_it(dlr_it(t), omega_l) * Gt(t, _, _); }
        T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);
        // 7. multiply by Fbar
        for (int t = 0; t < r; t++) { T(t, _, _) = matmul(Fbar, T(t, _, _)); }
      }
    } else {
      if (omega_l >= 0) {
        // 6. convolve by G
        T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);
        // 7. multiply by Fbar
        for (int t = 0; t < r; t++) { T(t, _, _) = matmul(Fbar, T(t, _, _)); }
      } else {
        // 6. convolve by G K^-
        for (int t = 0; t < r; t++) { GKt(t, _, _) = cppdlr::k_it(dlr_it(t), -omega_l) * Gt(t, _, _); }
        T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), cppdlr::TIME_ORDERED);
        // 7. multiply by Fbar
        for (int t = 0; t < r; t++) { T(t, _, _) = matmul(Fbar, T(t, _, _)); }
      }
    }
  }

  nda::array<dcomplex, 3> eval_eq(imtime_ops &itops, nda::array_const_view<dcomplex, 3> f, int n_quad) {
    auto fc    = itops.vals2coefs(f);
    auto it_eq = cppdlr::eqptsrel(n_quad + 1);
    auto f_eq  = nda::array<dcomplex, 3>(n_quad + 1, f.extent(1), f.extent(2));
    for (int i = 0; i <= n_quad; i++) { f_eq(i, _, _) = itops.coefs2eval(fc, it_eq(i)); }
    return f_eq;
  }

  nda::array<dcomplex, 3> OCA_dense(nda::array_const_view<dcomplex, 3> hyb, imtime_ops itops, double beta, nda::array_const_view<dcomplex, 3> Gt,
                                    nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags) {

    // index orders:
    // Gt (time, N, N), where N = 2^n, n = number of orbital indices
    // Fs (num_Fs, N, N)
    // Fbars (num_Fs, r, N, N)

    nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
    nda::vector_const_view<double> dlr_it = itops.get_itnodes();
    // number of imaginary time nodes
    int r = dlr_it.extent(0);
    int N = Gt.extent(1);

    auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs
    auto hyb_refl        = itops.reflect(hyb);
    auto hyb_refl_coeffs = hyb_coeffs;
    int num_Fs           = Fs.extent(0);

    // compute Fbars and Fdagbars
    auto Fdagbars  = nda::array<dcomplex, 4>(num_Fs, r, N, N);
    auto Fbarsrefl = nda::array<dcomplex, 4>(num_Fs, r, N, N);
    for (int lam = 0; lam < num_Fs; lam++) {
      for (int l = 0; l < r; l++) {
        for (int nu = 0; nu < num_Fs; nu++) {
          Fdagbars(lam, l, _, _) += hyb_coeffs(l, nu, lam) * F_dags(nu, _, _);
          Fbarsrefl(nu, l, _, _) += hyb_refl_coeffs(l, nu, lam) * Fs(lam, _, _);
        }
      }
    }

    // initialize self-energy
    nda::array<dcomplex, 3> Sigma(r, N, N);

    // preallocate intermediate arrays
    nda::array<dcomplex, 3> Sigma_l(r, N, N), T(r, N, N), Tmu(r, N, N), GKt(r, N, N);
    nda::array<dcomplex, 4> Tkaps(num_Fs, r, N, N);
    // Sigma_l = term of self-energy assoc'd with pole l, rest are placeholders
    // loop over hybridization lines
    for (int fb1 = 0; fb1 <= 1; fb1++) {
      for (int fb2 = 0; fb2 <= 1; fb2++) {
        // fb = 1 for forward line, else = 0
        // fb1 corresponds with line from 0 to tau_2
        auto const &F1list     = (fb1 == 1) ? Fs(_, _, _) : F_dags(_, _, _);
        auto const &F2list     = (fb2 == 1) ? Fs(_, _, _) : F_dags(_, _, _);
        auto const &F3list     = (fb1 == 1) ? F_dags(_, _, _) : Fs(_, _, _);
        auto const &Fbar_array = (fb2 == 1) ? Fdagbars(_, _, _, _) : Fbarsrefl(_, _, _, _);
        int sfM                = -1; // (fb1 ^ fb2) ? 1 : -1; // sign

        for (int l = 0; l < r; l++) {
          Sigma_l = 0;
          // initialize summand assoc'd with index l
          for (int lam = 0; lam < num_Fs; lam++) {
            OCA_dense_right_in_place(beta, itops, dlr_it, dlr_rf(l), (fb2 == 1), Gt, F2list(lam, _, _), T);
            OCA_dense_middle_in_place((fb1 == 1), hyb, hyb_refl, F1list, F3list, T, Tkaps, Tmu);
            OCA_dense_left_in_place(beta, itops, dlr_it, dlr_rf(l), (fb2 == 1), Gt, Fbar_array(lam, l, _, _), T, GKt);
            Sigma_l += T;
          } // sum over lambda

          // prefactor with Ks
          if (fb2 == 1) {
            if (dlr_rf(l) <= 0) {
              for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = cppdlr::k_it(dlr_it(t), dlr_rf(l)) * Sigma_l(t, _, _); }
              Sigma_l = Sigma_l / cppdlr::k_it(0, -dlr_rf(l));
            } else {
              Sigma_l = Sigma_l / cppdlr::k_it(0, dlr_rf(l));
            }
          } else {
            if (dlr_rf(l) >= 0) {
              for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = cppdlr::k_it(dlr_it(t), -dlr_rf(l)) * Sigma_l(t, _, _); }
              Sigma_l = Sigma_l / cppdlr::k_it(0, dlr_rf(l));
            } else {
              Sigma_l = Sigma_l / cppdlr::k_it(0, -dlr_rf(l));
            }
          }
          Sigma += sfM * Sigma_l;
        } // sum over l
      } // sum over fb2
    } // sum over fb1
    return Sigma;
  }

  nda::array<dcomplex, 3> OCA_dense(nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                    nda::array_const_view<dcomplex, 3> hyb_refl, nda::array_const_view<dcomplex, 3> hyb_refl_coeffs,
                                    nda::vector_const_view<double> hyb_poles, imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> Gt,
                                    nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags) {

    nda::vector_const_view<double> dlr_it = itops.get_itnodes();
    int r                                 = dlr_it.extent(0);
    int p                                 = hyb_poles.extent(0);
    int N                                 = Gt.extent(1);
    int n                                 = Fs.extent(0);

    // compute Fbars and Fdagbars
    auto Fdagbars  = nda::array<dcomplex, 4>(n, p, N, N);
    auto Fbarsrefl = nda::array<dcomplex, 4>(n, p, N, N);
    for (int lam = 0; lam < n; lam++) {
      for (int l = 0; l < p; l++) {
        for (int nu = 0; nu < n; nu++) {
          Fdagbars(lam, l, _, _) += hyb_coeffs(l, nu, lam) * F_dags(nu, _, _);
          Fbarsrefl(nu, l, _, _) += hyb_refl_coeffs(l, nu, lam) * Fs(lam, _, _);
        }
      }
    }

    // initialize self-energy
    nda::array<dcomplex, 3> Sigma(r, N, N);

    // preallocate intermediate arrays
    nda::array<dcomplex, 3> Sigma_l(r, N, N), T(r, N, N), Tmu(r, N, N), GKt(r, N, N);
    nda::array<dcomplex, 4> Tkaps(n, r, N, N);
    // Sigma_l = term of self-energy assoc'd with pole l, rest are placeholders
    // loop over hybridization lines
    for (int fb1 = 0; fb1 <= 1; fb1++) {
      for (int fb2 = 0; fb2 <= 1; fb2++) {
        // fb = 1 for forward line, else = 0
        // fb1 corresponds with line from 0 to tau_2
        auto const &F1list     = (fb1 == 1) ? Fs(_, _, _) : F_dags(_, _, _);
        auto const &F2list     = (fb2 == 1) ? Fs(_, _, _) : F_dags(_, _, _);
        auto const &F3list     = (fb1 == 1) ? F_dags(_, _, _) : Fs(_, _, _);
        auto const &Fbar_array = (fb2 == 1) ? Fdagbars(_, _, _, _) : Fbarsrefl(_, _, _, _);
        int sfM                = -1; // (fb1 ^ fb2) ? 1 : -1; // sign

        for (int l = 0; l < p; l++) {
          Sigma_l = 0;
          // initialize summand assoc'd with index l
          for (int lam = 0; lam < n; lam++) {
            OCA_dense_right_in_place(beta, itops, dlr_it, hyb_poles(l), (fb2 == 1), Gt, F2list(lam, _, _), T);
            OCA_dense_middle_in_place((fb1 == 1), hyb, hyb_refl, F1list, F3list, T, Tkaps, Tmu);
            OCA_dense_left_in_place(beta, itops, dlr_it, hyb_poles(l), (fb2 == 1), Gt, Fbar_array(lam, l, _, _), T, GKt);
            Sigma_l += T;
          } // sum over lambda

          // prefactor with Ks
          if (fb2 == 1) {
            if (hyb_poles(l) <= 0) {
              for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = cppdlr::k_it(dlr_it(t), hyb_poles(l)) * Sigma_l(t, _, _); }
              Sigma_l = Sigma_l / cppdlr::k_it(0, -hyb_poles(l));
            } else {
              Sigma_l = Sigma_l / cppdlr::k_it(0, hyb_poles(l));
            }
          } else {
            if (hyb_poles(l) >= 0) {
              for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = cppdlr::k_it(dlr_it(t), -hyb_poles(l)) * Sigma_l(t, _, _); }
              Sigma_l = Sigma_l / cppdlr::k_it(0, hyb_poles(l));
            } else {
              Sigma_l = Sigma_l / cppdlr::k_it(0, -hyb_poles(l));
            }
          }
          Sigma += sfM * Sigma_l;
        } // sum over l
      } // sum over fb2
    } // sum over fb1
    return Sigma;
  }

  nda::array<dcomplex, 3> OCA_tpz(nda::array_const_view<dcomplex, 3> hyb, imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> Gt,
                                  nda::array_const_view<dcomplex, 3> Fs, int n_quad) {
    // number of imaginary time nodes
    int N = Gt.extent(1);

    auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs
    auto hyb_refl        = nda::make_regular(-itops.reflect(hyb));
    auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);

    // get F^dagger operators
    int num_Fs = Fs.extent(0);
    nda::array<dcomplex, 3> F_dags(num_Fs, N, N);
    for (int i = 0; i < num_Fs; ++i) { F_dags(i, _, _) = nda::transpose(nda::conj(Fs(i, _, _))); }

    // get equispaced grid and evaluate functions on grid
    auto it_eq = cppdlr::eqptsrel(n_quad + 1);
    nda::array<dcomplex, 3> hyb_eq(n_quad + 1, num_Fs, num_Fs);
    nda::array<dcomplex, 3> hyb_refl_eq(n_quad + 1, num_Fs, num_Fs);
    auto Gt_coeffs = itops.vals2coefs(Gt);
    nda::array<dcomplex, 3> Gt_eq(n_quad + 1, N, N);
    // auto hyb_eq = itops.coefs2eval(hyb, it_eq);
    for (int i = 0; i < n_quad + 1; i++) {
      hyb_eq(i, _, _)      = itops.coefs2eval(hyb_coeffs, it_eq(i));
      hyb_refl_eq(i, _, _) = itops.coefs2eval(hyb_refl_coeffs, it_eq(i));
      // added 29 May 2025 v
      hyb_refl_eq(i, _, _) = nda::transpose(hyb_refl_eq(i, _, _));
      Gt_eq(i, _, _)       = itops.coefs2eval(Gt_coeffs, it_eq(i));
    }
    nda::array<dcomplex, 3> Sigma_eq(n_quad + 1, N, N);

    double dt = beta / n_quad;

    for (int fb1 = 0; fb1 <= 1; fb1++) {
      for (int fb2 = 0; fb2 <= 1; fb2++) {
        // fb = 1 for forward line, else = 0
        // fb1 corresponds with line from 0 to tau_2
        auto const &F1list = (fb1 == 1) ? Fs(_, _, _) : F_dags(_, _, _);
        auto const &F2list = (fb2 == 1) ? Fs(_, _, _) : F_dags(_, _, _);
        auto const &F3list = (fb1 == 1) ? F_dags(_, _, _) : Fs(_, _, _);
        auto const &F4list = (fb2 == 1) ? F_dags(_, _, _) : Fs(_, _, _);
        auto const &hyb1   = (fb1 == 1) ? hyb_eq(_, _, _) : hyb_refl_eq(_, _, _);
        auto const &hyb2   = (fb2 == 1) ? hyb_eq(_, _, _) : hyb_refl_eq(_, _, _);
        int sfM            = (fb1 ^ fb2) ? 1 : -1; // sign

        for (int lam = 0; lam < num_Fs; lam++) {
          for (int nu = 0; nu < num_Fs; nu++) {
            for (int mu = 0; mu < num_Fs; mu++) {
              for (int kap = 0; kap < num_Fs; kap++) {
                // i = 0 is included so that the whole output grid is covered: the tau_1 loop below is then empty,
                // leaving Sigma(0) at zero, its exact value since the double integral runs over [0, tau]
                for (int i = 0; i <= n_quad; i++) {
                  for (int i1 = 1; i1 <= i; i1++) {
                    for (int i2 = 0; i2 <= i1; i2++) {
                      double w = 1.0;
                      if (i1 == i) w = w / 2;
                      if (i2 == 0 || i2 == i1) w = w / 2;
                      auto FGFGFGF =
                         matmul(F4list(nu, _, _),
                                matmul(Gt_eq(i - i1, _, _),
                                       matmul(F3list(mu, _, _),
                                              matmul(Gt_eq(i1 - i2, _, _), matmul(F2list(lam, _, _), matmul(Gt_eq(i2, _, _), F1list(kap, _, _)))))));

                      Sigma_eq(i, _, _) += sfM * w * hyb2(i - i2, lam, nu) * hyb1(i1, mu, kap) * FGFGFGF;
                    } // sum over i2
                  } // sum over i1
                } // sum over i
              } // sum over kappa
            } // sum over mu
          } // sum over nu
        } // sum over lambda

      } // sum over fb2
    } // sum over fb1

    Sigma_eq = dt * dt * Sigma_eq;

    return Sigma_eq;
  }

  nda::array<dcomplex, 3> third_order_tpz(nda::array_const_view<dcomplex, 3> hyb, imtime_ops &itops, double beta,
                                          nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> Fs, int n_quad) {
    // Third-order self-energy diagram for topology {{0,3},{1,4},{2,5}} (the fully-crossing topology),
    // evaluated by direct trapezoidal quadrature. Generalizes OCA_tpz's pattern from 2 hybridization
    // lines / 2 internal vertex times (4 vertices) to 3 lines / 4 internal vertex times (6 vertices):
    // vertex 0 is fixed at tau=0, vertex 5 is the external self-energy time, and vertices 1-4 are
    // integrated over 0 <= j1 <= j2 <= j3 <= j4 <= i. Line 0 connects vertices 0 and 3, line 1 connects
    // vertices 1 and 4, line 2 connects vertices 2 and 5.
    int N = Gt.extent(1);

    auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs
    auto hyb_refl        = nda::make_regular(-itops.reflect(hyb));
    auto hyb_refl_coeffs = itops.vals2coefs(hyb_refl);

    // get F^dagger operators
    int num_Fs = Fs.extent(0);
    nda::array<dcomplex, 3> F_dags(num_Fs, N, N);
    for (int i = 0; i < num_Fs; ++i) { F_dags(i, _, _) = nda::transpose(nda::conj(Fs(i, _, _))); }

    // get equispaced grid and evaluate functions on grid
    auto it_eq = cppdlr::eqptsrel(n_quad + 1);
    nda::array<dcomplex, 3> hyb_eq(n_quad + 1, num_Fs, num_Fs);
    nda::array<dcomplex, 3> hyb_refl_eq(n_quad + 1, num_Fs, num_Fs);
    auto Gt_coeffs = itops.vals2coefs(Gt);
    nda::array<dcomplex, 3> Gt_eq(n_quad + 1, N, N);
    for (int i = 0; i < n_quad + 1; i++) {
      hyb_eq(i, _, _)      = itops.coefs2eval(hyb_coeffs, it_eq(i));
      hyb_refl_eq(i, _, _) = itops.coefs2eval(hyb_refl_coeffs, it_eq(i));
      hyb_refl_eq(i, _, _) = nda::transpose(hyb_refl_eq(i, _, _));
      Gt_eq(i, _, _)       = itops.coefs2eval(Gt_coeffs, it_eq(i));
    }
    nda::array<dcomplex, 3> Sigma_eq(n_quad + 1, N, N);

    double dt = beta / n_quad;

    for (int d0 = 0; d0 <= 1; d0++) {
      for (int d1 = 0; d1 <= 1; d1++) {
        for (int d2 = 0; d2 <= 1; d2++) {
          // fb = 1 for forward line, else = 0
          // d0 is the direction of the line touching vertex 0 (connects vertices 0 and 3)
          // d1, d2 are the directions of the lines connecting vertices (1,4) and (2,5)
          auto const &F0list = (d0 == 1) ? Fs(_, _, _) : F_dags(_, _, _);
          auto const &F1list = (d1 == 1) ? Fs(_, _, _) : F_dags(_, _, _);
          auto const &F2list = (d2 == 1) ? Fs(_, _, _) : F_dags(_, _, _);
          auto const &F3list = (d0 == 1) ? F_dags(_, _, _) : Fs(_, _, _);
          auto const &F4list = (d1 == 1) ? F_dags(_, _, _) : Fs(_, _, _);
          auto const &F5list = (d2 == 1) ? F_dags(_, _, _) : Fs(_, _, _);

          auto const &hyb0 = (d0 == 1) ? hyb_eq(_, _, _) : hyb_refl_eq(_, _, _); // line 0-3
          auto const &hyb1 = (d1 == 1) ? hyb_eq(_, _, _) : hyb_refl_eq(_, _, _); // line 1-4
          auto const &hyb2 = (d2 == 1) ? hyb_eq(_, _, _) : hyb_refl_eq(_, _, _); // line 2-5

          int sfM = ((d0 + d1 + d2) % 2 == 0) ? -1 : 1; // sign

          for (int o0 = 0; o0 < num_Fs; o0++) {
            for (int o1 = 0; o1 < num_Fs; o1++) {
              for (int o2 = 0; o2 < num_Fs; o2++) {
                for (int o3 = 0; o3 < num_Fs; o3++) {
                  for (int o4 = 0; o4 < num_Fs; o4++) {
                    for (int o5 = 0; o5 < num_Fs; o5++) {
                      // as in OCA_tpz, i = 0 is included so that the whole output grid is covered: the j4 loop
                      // below is then empty, leaving Sigma(0) at its exact value of zero
                      for (int i = 0; i <= n_quad; i++) {
                        for (int j4 = 1; j4 <= i; j4++) {
                          double w4 = (j4 == i) ? 0.5 : 1.0;
                          for (int j3 = 0; j3 <= j4; j3++) {
                            double w3 = w4 * ((j3 == 0 || j3 == j4) ? 0.5 : 1.0);
                            for (int j2 = 0; j2 <= j3; j2++) {
                              double w2 = w3 * ((j2 == 0 || j2 == j3) ? 0.5 : 1.0);
                              for (int j1 = 0; j1 <= j2; j1++) {
                                double w = w2 * ((j1 == 0 || j1 == j2) ? 0.5 : 1.0);

                                auto chain =
                                   matmul(F5list(o5, _, _),
                                          matmul(Gt_eq(i - j4, _, _),
                                                 matmul(F4list(o4, _, _),
                                                        matmul(Gt_eq(j4 - j3, _, _),
                                                               matmul(F3list(o3, _, _),
                                                                      matmul(Gt_eq(j3 - j2, _, _),
                                                                             matmul(F2list(o2, _, _),
                                                                                    matmul(Gt_eq(j2 - j1, _, _),
                                                                                           matmul(F1list(o1, _, _),
                                                                                                  matmul(Gt_eq(j1, _, _), F0list(o0, _, _)))))))))));

                                Sigma_eq(i, _, _) += sfM * w * hyb0(j3, o3, o0) * hyb1(j4 - j1, o1, o4) * hyb2(i - j2, o2, o5) * chain;
                              } // sum over j1
                            } // sum over j2
                          } // sum over j3
                        } // sum over j4
                      } // sum over i
                    } // sum over o5
                  } // sum over o4
                } // sum over o3
              } // sum over o2
            } // sum over o1
          } // sum over o0

        } // sum over d2
      } // sum over d1
    } // sum over d0

    Sigma_eq = dt * dt * dt * dt * Sigma_eq;

    return Sigma_eq;
  }

} // namespace triqs_xca::dense
