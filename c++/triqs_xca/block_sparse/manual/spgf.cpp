#include "triqs_xca/block_sparse/manual/spgf.hpp"

namespace triqs_xca::block_sparse::manual {

  using cppdlr::_;

  using nda::range;
  using nda::trace;
  using nda::linalg::matmul;

  nda::array<dcomplex, 3> spgf_nca(const BlockDiagOpFun &Gt, const BlockDiagOpFun &Gt_refl, const BlockOpSymQuartet &Fq) {

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

  void spgf_oca_right(double beta, imtime_ops &itops, nda::vector_const_view<double> dlr_it, double omega_l, bool forward, const BlockDiagOpFun &Gt,
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

  void spgf_oca_left(double beta, imtime_ops &itops, nda::vector_const_view<double> dlr_it, double omega_l, bool forward, const BlockDiagOpFun &Gt,
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
    // reflect the whole buffer: a sub-block view is not contiguous, and reflect reshapes its argument
    T = itops.reflect(T);
  }

  nda::array<dcomplex, 3> spgf_oca(nda::vector_const_view<double> hyb_poles, imtime_ops &itops, double beta, const BlockDiagOpFun &Gt,
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
                      spgf_oca_right(beta, itops, dlr_it, hyb_poles(l), fb, Gt, F1[p_lam], lam, T, ind_path, block_dims);
                      spgf_oca_left(beta, itops, dlr_it, hyb_poles(l), fb, Gt, Fbar[p_lam], lam, l, U, ind_path, block_dims);
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

} // namespace triqs_xca::block_sparse::manual
