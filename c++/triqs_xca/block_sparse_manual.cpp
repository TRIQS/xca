#include "block_sparse.hpp"
#include "block_sparse_manual.hpp"
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/utils.hpp>
#include <h5/format.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <nda/algorithms.hpp>
#include <nda/blas/tools.hpp>
#include <nda/declarations.hpp>
#include <nda/basic_functions.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/linalg/eigenelements.hpp>
#include <nda/print.hpp>
#include <vector>

BlockDiagOpFun NCA_bs(nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl, BlockDiagOpFun const &Gt,
                      const std::vector<BlockOp> &Fs) {
  // get F^dagger operators
  int num_Fs  = Fs.size();
  auto F_dags = Fs;
  for (int i = 0; i < num_Fs; ++i) { F_dags[i] = dagger_bs(Fs[i]); }
  // initialize self-energy, with same shape as Gt
  int r = Gt.get_num_time_nodes();
  BlockDiagOpFun Sigma(r, Gt.get_block_sizes());

  for (int fb = 0; fb <= 1; fb++) {
    // fb = 1 for forward line, 0 for backward line
    auto const &F1list = (fb) ? Fs : F_dags;
    auto const &F2list = (fb) ? F_dags : Fs;
    int sfM            = -1;

    for (int lam = 0; lam < num_Fs; lam++) {
      for (int kap = 0; kap < num_Fs; kap++) {
        auto &F1              = F1list[kap];
        auto &F2              = F2list[lam];
        int ind_path          = 0;
        bool path_all_nonzero = true; // if set to false during backwards pass,
        // the i-th block of Sigma is zero, so no computation needed

        for (int i = 0; i < Gt.get_num_block_cols(); i++) {
          // "backwards pass"
          // for each self-energy block, find contributing blocks of factors
          path_all_nonzero = true;
          ind_path         = F1.get_block_index(i); // Sigma = F2 G F1
          // ind_path = block-column index of F1 corresponding with
          // block that contributes to i-th block of Sigma
          if (ind_path == -1 || Gt.get_zero_block_index(ind_path) == -1 || F2.get_block_index(ind_path) == -1) {
            path_all_nonzero = false; // one of the blocks of F1,
                                      // Gt, Ft that contribute to block i of Sigma is zero
          }

          // matmuls
          // if path involves all nonzero blocks, compute product
          // of blocks indexed by ind_path
          if (path_all_nonzero) {
            auto block = nda::zeros<dcomplex>(r, Gt.get_block_size(i), Gt.get_block_size(i));
            for (int t = 0; t < r; t++) {
              if (fb == 1) {
                block(t, _, _) =
                   hyb(t, lam, kap) * nda::matmul(F2.get_block(ind_path), nda::matmul(Gt.get_block(ind_path)(t, _, _), F1.get_block(i)));
              } else {
                block(t, _, _) =
                   hyb_refl(t, kap, lam) * nda::matmul(F2.get_block(ind_path), nda::matmul(Gt.get_block(ind_path)(t, _, _), F1.get_block(i)));
              }
            }
            block = sfM * block;
            Sigma.add_block(i, block);
          }
        }
      }
    }
  }

  return Sigma;
}

BlockDiagOpFun NCA_bs(nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl, BlockDiagOpFun const &Gt,
                      const BlockOpSymQuartet &Fq) {
  // get F^dagger operators
  int n = Fq.sym_set_labels.size();
  // initialize self-energy, with same shape as Gt
  int r = Gt.get_num_time_nodes();
  BlockDiagOpFun Sigma(r, Gt.get_block_sizes());

  for (int fb = 0; fb <= 1; fb++) {
    // fb = 1 for forward line, 0 for backward line
    auto const &F1 = (fb) ? Fq.Fs[0] : Fq.F_dags[0];
    auto const &F2 = (fb) ? Fq.F_dags[0] : Fq.Fs[0];
    int sfM        = -1;

    for (int lam = 0; lam < n; lam++) {
      for (int kap = 0; kap < n; kap++) {
        // auto &F1              = F1list(kap, _, _);
        // auto &F2              = F2list[lam];
        int ind_path          = 0;
        bool path_all_nonzero = true; // if set to false during backwards pass,
        // the i-th block of Sigma is zero, so no computation needed

        for (int i = 0; i < Gt.get_num_block_cols(); i++) {
          // "backwards pass"
          // for each self-energy block, find contributing blocks of factors
          path_all_nonzero = true;
          ind_path         = F1.get_block_index(i); // Sigma = F2 G F1
          // ind_path = block-column index of F1 corresponding with
          // block that contributes to i-th block of Sigma
          if (ind_path == -1 || Gt.get_zero_block_index(ind_path) == -1 || F2.get_block_index(ind_path) == -1) {
            path_all_nonzero = false; // one of the blocks of F1,
                                      // Gt, Ft that contribute to block i of Sigma is zero
          }

          // matmuls
          // if path involves all nonzero blocks, compute product
          // of blocks indexed by ind_path
          if (path_all_nonzero) {
            auto block = nda::zeros<dcomplex>(r, Gt.get_block_size(i), Gt.get_block_size(i));
            for (int t = 0; t < r; t++) {
              if (fb == 1) {
                block(t, _, _) = hyb(t, lam, kap)
                   * nda::matmul(F2.get_block(ind_path)(lam, _, _), nda::matmul(Gt.get_block(ind_path)(t, _, _), F1.get_block(i)(kap, _, _)));
              } else {
                block(t, _, _) = hyb_refl(t, kap, lam)
                   * nda::matmul(F2.get_block(ind_path)(lam, _, _), nda::matmul(Gt.get_block(ind_path)(t, _, _), F1.get_block(i)(kap, _, _)));
              }
            }
            block = sfM * block;
            Sigma.add_block(i, block);
          }
        }
      }
    }
  }

  return Sigma;
}

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
    int sfM            = -1;

    for (int lam = 0; lam < n; lam++) {
      for (int kap = 0; kap < n; kap++) {
        auto F1 = F1list(kap, _, _);
        auto F2 = F2list(lam, _, _);

        for (int t = 0; t < r; t++) {
          if (fb == 1) {
            Sigma(t, _, _) += sfM * hyb(t, lam, kap) * nda::matmul(F2, nda::matmul(Gt(t, _, _), F1));
          } else {
            Sigma(t, _, _) += sfM * hyb_refl(t, lam, kap) * nda::matmul(F2, nda::matmul(Gt(t, _, _), F1));
          }
        }
      }
    }
  }

  return Sigma;
}

void OCA_bs_right_in_place(double beta, imtime_ops &itops, nda::vector_const_view<double> dlr_it, double omega_l, bool forward,
                           nda::array_const_view<dcomplex, 3> Gt0, nda::array_const_view<dcomplex, 3> Gt1, nda::array_const_view<dcomplex, 2> Flam,
                           nda::array_view<dcomplex, 3> T) {

  int r = Gt0.extent(0);
  // nda::array<dcomplex,3> T(r,Flam.extent(0),Flam.extent(1));

  if (forward) {
    if (omega_l <= 0) {
      // 1. multiply F_lambda G(tau_1) K^-(tau_1)
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), -omega_l) * nda::matmul(Flam, Gt0(t, _, _)); }
      // 2. convolve by G
      T = itops.convolve(beta, itops.vals2coefs(Gt1), itops.vals2coefs(T), TIME_ORDERED);
    } else {
      // 1. multiply G(tau_2-tau_1) K^+(tau_2-tau_1) F_lambda
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), omega_l) * nda::matmul(Gt1(t, _, _), Flam); }
      // 2. convolve by G
      T = itops.convolve(beta, itops.vals2coefs(T), itops.vals2coefs(Gt0), TIME_ORDERED);
    }
  } else {
    if (omega_l >= 0) {
      // 1. multiply F_lambda G(tau_1) K^+(tau_1)
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), omega_l) * nda::matmul(Flam, Gt0(t, _, _)); }
      // 2. convolve by G
      T = itops.convolve(beta, itops.vals2coefs(Gt1), itops.vals2coefs(T), TIME_ORDERED);
    } else {
      // 1. multiply G(tau_2-tau_1) K^-(tau_2-tau_1) F_lambda
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), -omega_l) * nda::matmul(Gt1(t, _, _), Flam); }
      // 2. convolve by G
      T = itops.convolve(beta, itops.vals2coefs(T), itops.vals2coefs(Gt0), TIME_ORDERED);
    }
  }
}

void OCA_bs_middle_in_place(bool forward, nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl,
                            nda::array_const_view<dcomplex, 3> Fkaps, nda::array_const_view<dcomplex, 3> Fmus, nda::array_view<dcomplex, 3> Tin,
                            nda::array_view<dcomplex, 3> Tout, nda::array_view<dcomplex, 4> Tkaps, nda::array_view<dcomplex, 3> Tmu) {

  int num_Fs = Fkaps.extent(0);
  int r      = Tmu.extent(0);
  // 3. for each kappa, multiply by F_kappa from right
  // auto Tkap = nda::zeros<dcomplex>(num_Fs, r, T.extent(1), Fkaps.extent(2));
  for (int kap = 0; kap < num_Fs; kap++) {
    for (int t = 0; t < r; t++) { Tkaps(kap, t, _, _) = nda::matmul(Tin(t, _, _), Fkaps(kap, _, _)); }
  }

  // 4. for each mu, kap, mult by Delta_mu_kap and sum kap
  // nda::array<dcomplex,3> Tmu(r, T.extent(1), Fkaps.extent(2));
  Tout   = 0;
  auto U = nda::zeros<dcomplex>(r, Fmus.extent(1), Fkaps.extent(2));
  for (int mu = 0; mu < num_Fs; mu++) {
    Tmu = 0;
    for (int kap = 0; kap < num_Fs; kap++) {
      for (int t = 0; t < r; t++) {
        if (forward) {
          Tmu(t, _, _) += hyb(t, mu, kap) * Tkaps(kap, t, _, _);
        } else {
          Tmu(t, _, _) += hyb_refl(t, kap, mu) * Tkaps(kap, t, _, _);
        }
      }
    }
    // 5. multiply by F^dag_mu and sum over mu
    for (int t = 0; t < r; t++) {
      auto Fmu = Fmus(mu, _, _);
      Tout(t, _, _) += nda::matmul(Fmu, Tmu(t, _, _));
    }
  }
}

void OCA_bs_left_in_place(double beta, imtime_ops &itops, nda::vector_const_view<double> dlr_it, double omega_l, bool forward,
                          nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 2> Fbar, nda::array_view<dcomplex, 3> Tin,
                          nda::array_view<dcomplex, 3> Tout, nda::array_view<dcomplex, 3> GKt) {

  int r = Gt.extent(0);
  // nda::array<dcomplex,3> Sigma_l(r,Fbar.extent(0),U.extent(2));
  // auto dlr_it = itops.get_itnodes();

  if (forward) {
    if (omega_l <= 0) {
      // 6. convolve by G
      Tin = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(Tin), TIME_ORDERED);
      // 7. multiply by Fbar
      for (int t = 0; t < r; t++) { Tout(t, _, _) = nda::matmul(Fbar, Tin(t, _, _)); }
    } else {
      // 6. convolve by G K^+
      for (int t = 0; t < r; t++) { GKt(t, _, _) = k_it(dlr_it(t), omega_l) * Gt(t, _, _); }
      Tin = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(Tin), TIME_ORDERED);
      // 7. multiply by Fbar
      for (int t = 0; t < r; t++) { Tout(t, _, _) = nda::matmul(Fbar, Tin(t, _, _)); }
    }
  } else {
    if (omega_l >= 0) {
      // 6. convolve by G
      Tin = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(Tin), TIME_ORDERED);
      // 7. multiply by Fbar
      for (int t = 0; t < r; t++) { Tout(t, _, _) = nda::matmul(Fbar, Tin(t, _, _)); }
    } else {
      // 6. convolve by G K^-
      for (int t = 0; t < r; t++) { GKt(t, _, _) = k_it(dlr_it(t), -omega_l) * Gt(t, _, _); }
      Tin = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(Tin), TIME_ORDERED);
      // 7. multiply by Fbar
      for (int t = 0; t < r; t++) { Tout(t, _, _) = nda::matmul(Fbar, Tin(t, _, _)); }
    }
  }
}

BlockDiagOpFun OCA_bs(nda::array_const_view<dcomplex, 3> hyb, imtime_ops &itops, double beta, const BlockDiagOpFun &Gt,
                      const std::vector<BlockOp> &Fs) {

  nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
  nda::vector_const_view<double> dlr_it = itops.get_itnodes();
  // number of imaginary time nodes
  int r = dlr_it.shape(0);

  auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs
  auto hyb_refl        = itops.reflect(hyb);
  auto hyb_refl_coeffs = hyb_coeffs;

  // get F^dagger operators
  int num_Fs  = Fs.size();
  auto F_dags = Fs;
  for (int i = 0; i < num_Fs; ++i) { F_dags[i] = dagger_bs(Fs[i]); }

  // compute Fbars and Fdagbars
  auto Fbar_indices    = Fs[0].get_block_indices();
  auto Fbar_sizes      = Fs[0].get_block_sizes();
  auto Fdagbar_indices = F_dags[0].get_block_indices();
  auto Fdagbar_sizes   = F_dags[0].get_block_sizes();
  std::vector<std::vector<BlockOp>> Fdagbars(num_Fs, std::vector<BlockOp>(r, BlockOp(Fdagbar_indices, Fdagbar_sizes)));
  std::vector<std::vector<BlockOp>> Fbarsrefl(num_Fs, std::vector<BlockOp>(r, BlockOp(Fbar_indices, Fbar_sizes)));
  for (int lam = 0; lam < num_Fs; lam++) {
    for (int l = 0; l < r; l++) {
      for (int nu = 0; nu < num_Fs; nu++) {
        Fdagbars[lam][l] += hyb_coeffs(l, nu, lam) * F_dags[nu];
        Fbarsrefl[lam][l] += hyb_refl_coeffs(l, lam, nu) * Fs[nu];
      }
    }
  }

  // initialize self-energy
  BlockDiagOpFun Sigma = BlockDiagOpFun(r, Gt.get_block_sizes());
  int num_block_cols   = Gt.get_num_block_cols();

  // preallocation
  bool path_all_nonzero;
  nda::vector<int> ind_path(3);
  nda::vector<int> block_dims(5); // intermediate block dimensions

  // loop over hybridization lines
  for (int fb1 = 0; fb1 <= 1; fb1++) {
    for (int fb2 = 0; fb2 <= 1; fb2++) {
      // fb = 1 for forward line, else = 0
      // fb1 corresponds with line from 0 to tau_2
      auto const &F1list    = (fb1) ? Fs : F_dags;
      auto const &F2list    = (fb2) ? Fs : F_dags;
      auto const &F3list    = (fb1) ? F_dags : Fs;
      auto const Fbar_array = (fb2) ? Fdagbars : Fbarsrefl;
      int sfM               = -1; // (fb1 ^ fb2) ? 1 : -1; // sign

      for (int i = 0; i < num_block_cols; i++) {

        // "backwards pass"
        //
        // for each self-energy block, find contributing blocks of factors
        //
        // paths_all_nonzero: false if for i-th block of
        // Sigma, factors assoc'd with lambda, mu, kappa don't contribute
        //
        // ind_path: a vector of column indices of the
        // blocks of the factors that contribute. if paths_all_nonzero is
        // false at this index, values in ind_path are garbage.
        //
        // ATTN: assumes all BlockOps in F(1,2,3)list have the same structure
        // i.e, index path is independent of kappa, mu
        path_all_nonzero   = true;
        auto Sigma_block_i = nda::make_regular(0 * Sigma.get_block(i));

        ind_path(0) = F1list[0].get_block_index(i);
        if (ind_path(0) == -1 || Gt.get_zero_block_index(ind_path(0)) == -1) {
          path_all_nonzero = false;
        } else {
          block_dims(0) = F1list[0].get_block_size(i, 1);
          block_dims(1) = F1list[0].get_block_size(i, 0);
          ind_path(1)   = F2list[0].get_block_index(ind_path(0));
          if (ind_path(1) == -1 || Gt.get_zero_block_index(ind_path(1)) == -1) {
            path_all_nonzero = false;
          } else {
            block_dims(2) = F2list[0].get_block_size(ind_path(0), 0);
            ind_path(2)   = F3list[0].get_block_index(ind_path(1));
            if (ind_path(2) == -1 || Gt.get_zero_block_index(ind_path(2)) == -1 || Fbar_array[0][0].get_block_index(ind_path(2)) == -1) {
              path_all_nonzero = false;
            } else {
              block_dims(3) = F3list[0].get_block_size(ind_path(1), 0);
              block_dims(4) = Fbar_array[0][0].get_block_size(ind_path(2), 0);
            }
          }
        }

        // matmuls and convolutions
        if (path_all_nonzero) {
          // preallocate for i-th block
          auto Sigma_l = nda::make_regular(0 * Sigma.get_block(i));
          // sizes of intermediate matrices are known
          nda::array<dcomplex, 3> Tright(r, block_dims(2), block_dims(1));        // output of OCA_bs_right
          nda::array<dcomplex, 3> Tmid(r, block_dims(3), block_dims(0));          // output of OCA_bs_middle
          nda::array<dcomplex, 4> Tkaps(num_Fs, r, block_dims(2), block_dims(0)); // storage in OCA_bs_middle
          nda::array<dcomplex, 3> Tmu(r, block_dims(2), block_dims(0));           // storage in OCA_bs_middle
          nda::array<dcomplex, 3> Tleft(r, block_dims(4), block_dims(0));         // output of OCA_bs_left
          nda::array<dcomplex, 3> GKt(r, block_dims(3), block_dims(3));           // storage in OCA_bs_left

          // TODO: make Fs have blocks that are 3D nda::array?
          auto Fkaps = nda::zeros<dcomplex>(num_Fs, F1list[0].get_block_size(i, 0), F1list[0].get_block_size(i, 1));
          auto Fmus  = nda::zeros<dcomplex>(num_Fs, F3list[0].get_block_size(ind_path(1), 0), F3list[0].get_block_size(ind_path(1), 1));
          for (int j = 0; j < num_Fs; j++) {
            Fkaps(j, _, _) = F1list[j].get_block(i);
            Fmus(j, _, _)  = F3list[j].get_block(ind_path(1));
          }

          for (int l = 0; l < r; l++) {
            Sigma_l = 0;
            for (int lam = 0; lam < num_Fs; lam++) {
              OCA_bs_right_in_place(beta, itops, dlr_it, dlr_rf(l), (fb2 == 1), Gt.get_block(ind_path(0)), Gt.get_block(ind_path(1)),
                                    F2list[lam].get_block(ind_path(0)), Tright);
              OCA_bs_middle_in_place((fb1 == 1), hyb, hyb_refl, Fkaps, Fmus, Tright, Tmid, Tkaps, Tmu);
              OCA_bs_left_in_place(beta, itops, dlr_it, dlr_rf(l), (fb2 == 1), Gt.get_block(ind_path(2)), Fbar_array[lam][l].get_block(ind_path(2)),
                                   Tmid, Tleft, GKt);
              Sigma_l += Tleft;
            } // sum over lambda

            // prefactor with Ks
            if (fb2 == 1) {
              if (dlr_rf(l) <= 0) {
                for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = k_it(dlr_it(t), dlr_rf(l)) * Sigma_l(t, _, _); }
                Sigma_l = Sigma_l / k_it(0, -dlr_rf(l));
              } else {
                Sigma_l = Sigma_l / k_it(0, dlr_rf(l));
              }
            } else {
              if (dlr_rf(l) >= 0) {
                for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = k_it(dlr_it(t), -dlr_rf(l)) * Sigma_l(t, _, _); }
                Sigma_l = Sigma_l / k_it(0, dlr_rf(l));
              } else {
                Sigma_l = Sigma_l / k_it(0, -dlr_rf(l));
              }
            }
            Sigma_block_i += nda::make_regular(sfM * Sigma_l);
          } // sum over l
          Sigma.add_block(i, Sigma_block_i);
        } // end if(path_all_nonzero)
      } // end loop over i
    } // end loop over fb2
  } // end loop over fb1
  return Sigma;
}

BlockDiagOpFun OCA_bs(nda::array_const_view<dcomplex, 3> hyb, imtime_ops &itops, double beta, const BlockDiagOpFun &Gt, const BlockOpSymQuartet &Fq) {

  long n                                = Fq.sym_set_labels.size();
  nda::vector_const_view<double> dlr_rf = itops.get_rfnodes();
  nda::vector_const_view<double> dlr_it = itops.get_itnodes();
  // number of imaginary time nodes
  long r = dlr_it.shape(0);

  auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs
  auto hyb_refl        = itops.reflect(hyb);
  auto hyb_refl_coeffs = hyb_coeffs;

  // initialize self-energy
  BlockDiagOpFun Sigma = BlockDiagOpFun(r, Gt.get_block_sizes());
  int num_block_cols   = Gt.get_num_block_cols();

  // preallocation
  bool path_all_nonzero = true;
  nda::vector<int> ind_path(3);
  nda::vector<int> block_dims(5); // intermediate block dimensions

  // loop over hybridization lines
  for (int fb1 = 0; fb1 <= 1; fb1++) {
    for (int fb2 = 0; fb2 <= 1; fb2++) {
      // fb = 1 for forward line, else = 0
      // fb1 corresponds with line from 0 to tau_2
      auto const &F1   = (fb1) ? Fq.Fs : Fq.F_dags;
      auto const &F2   = (fb2) ? Fq.Fs : Fq.F_dags;
      auto const &F3   = (fb1) ? Fq.F_dags : Fq.Fs;
      auto const &Fbar = (fb2) ? Fq.F_dag_bars : Fq.F_bars_refl;
      int sfM          = -1; // (fb1 ^ fb2) ? 1 : -1; // sign

      for (int i = 0; i < num_block_cols; i++) {

        // "backwards pass"
        //
        // for each self-energy block, find contributing blocks of factors
        //
        // paths_all_nonzero: false if for i-th block of
        // Sigma, factors assoc'd with lambda, mu, kappa don't contribute
        //
        // ind_path: a vector of column indices of the
        // blocks of the factors that contribute. if paths_all_nonzero is
        // false at this index, values in ind_path are garbage.
        //
        // ATTN: assumes all BlockOps in F(1,2,3)list have the same structure
        // i.e, index path is independent of kappa, mu
        path_all_nonzero   = true;
        auto Sigma_block_i = nda::make_regular(0 * Sigma.get_block(i));

        ind_path(0) = F1[0].get_block_index(i);
        if (ind_path(0) == -1 || Gt.get_zero_block_index(ind_path(0)) == -1) {
          path_all_nonzero = false;
        } else {
          block_dims(0) = F1[0].get_block_size(i, 1);
          block_dims(1) = F1[0].get_block_size(i, 0);
          ind_path(1)   = F2[0].get_block_index(ind_path(0));
          if (ind_path(1) == -1 || Gt.get_zero_block_index(ind_path(1)) == -1) {
            path_all_nonzero = false;
          } else {
            block_dims(2) = F2[0].get_block_size(ind_path(0), 0);
            ind_path(2)   = F3[0].get_block_index(ind_path(1));
            if (ind_path(2) == -1 || Gt.get_zero_block_index(ind_path(2)) == -1 || Fbar[0].get_block_index(ind_path(2)) == -1) {
              path_all_nonzero = false;
            } else {
              block_dims(3) = F3[0].get_block_size(ind_path(1), 0);
              block_dims(4) = Fbar[0].get_block_size(ind_path(2), 0);
            }
          }
        }

        // matmuls and convolutions
        if (path_all_nonzero) {
          // preallocate for i-th block
          auto Sigma_l = nda::make_regular(0 * Sigma.get_block(i));
          // sizes of intermediate matrices are known
          nda::array<dcomplex, 3> Tright(r, block_dims(2), block_dims(1));   // output of OCA_bs_right
          nda::array<dcomplex, 3> Tmid(r, block_dims(3), block_dims(0));     // output of OCA_bs_middle
          nda::array<dcomplex, 4> Tkaps(n, r, block_dims(2), block_dims(0)); // storage in OCA_bs_middle
          nda::array<dcomplex, 3> Tmu(r, block_dims(2), block_dims(0));      // storage in OCA_bs_middle
          nda::array<dcomplex, 3> Tleft(r, block_dims(4), block_dims(0));    // output of OCA_bs_left
          nda::array<dcomplex, 3> GKt(r, block_dims(3), block_dims(3));      // storage in OCA_bs_left

          auto Fkaps = F1[0].get_block(i);
          auto Fmus  = F3[0].get_block(ind_path(1));

          for (int l = 0; l < r; l++) {
            Sigma_l = 0;
            for (int lam = 0; lam < n; lam++) {
              OCA_bs_right_in_place(beta, itops, dlr_it, dlr_rf(l), (fb2 == 1), Gt.get_block(ind_path(0)), Gt.get_block(ind_path(1)),
                                    F2[0].get_block(ind_path(0))(lam, _, _), Tright);
              OCA_bs_middle_in_place((fb1 == 1), hyb, hyb_refl, Fkaps, Fmus, Tright, Tmid, Tkaps, Tmu);
              OCA_bs_left_in_place(beta, itops, dlr_it, dlr_rf(l), (fb2 == 1), Gt.get_block(ind_path(2)),
                                   Fbar[0].get_block(ind_path(2))(lam, l, _, _), Tmid, Tleft, GKt);
              Sigma_l += Tleft;
            } // sum over lambda

            // prefactor with Ks
            if (fb2 == 1) {
              if (dlr_rf(l) <= 0) {
                for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = k_it(dlr_it(t), dlr_rf(l)) * Sigma_l(t, _, _); }
                Sigma_l = Sigma_l / k_it(0, -dlr_rf(l));
              } else {
                Sigma_l = Sigma_l / k_it(0, dlr_rf(l));
              }
            } else {
              if (dlr_rf(l) >= 0) {
                for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = k_it(dlr_it(t), -dlr_rf(l)) * Sigma_l(t, _, _); }
                Sigma_l = Sigma_l / k_it(0, dlr_rf(l));
              } else {
                Sigma_l = Sigma_l / k_it(0, -dlr_rf(l));
              }
            }
            Sigma_block_i += nda::make_regular(sfM * Sigma_l);
          } // sum over l
          Sigma.add_block(i, Sigma_block_i);
        } // end if(path_all_nonzero)
      } // end loop over i
    } // end loop over fb2
  } // end loop over fb1
  return Sigma;
}

BlockDiagOpFun OCA_bs(nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                      nda::array_const_view<dcomplex, 3> hyb_refl, nda::array_const_view<dcomplex, 3> hyb_refl_coeffs,
                      nda::vector_const_view<double> hyb_poles, imtime_ops &itops, double beta, const BlockDiagOpFun &Gt,
                      const std::vector<BlockOp> &Fs) {

  nda::vector_const_view<double> dlr_it = itops.get_itnodes();
  int r                                 = dlr_it.extent(0);
  int p                                 = hyb_poles.extent(0);
  int n                                 = Fs.size();

  auto F_dags = Fs;
  for (int i = 0; i < n; ++i) { F_dags[i] = dagger_bs(Fs[i]); }

  // compute Fbars and Fdagbars
  auto Fbar_indices    = Fs[0].get_block_indices();
  auto Fbar_sizes      = Fs[0].get_block_sizes();
  auto Fdagbar_indices = F_dags[0].get_block_indices();
  auto Fdagbar_sizes   = F_dags[0].get_block_sizes();
  std::vector<std::vector<BlockOp>> Fdagbars(n, std::vector<BlockOp>(p, BlockOp(Fdagbar_indices, Fdagbar_sizes)));
  std::vector<std::vector<BlockOp>> Fbarsrefl(n, std::vector<BlockOp>(p, BlockOp(Fbar_indices, Fbar_sizes)));
  for (int lam = 0; lam < n; lam++) {
    for (int l = 0; l < p; l++) {
      for (int nu = 0; nu < n; nu++) {
        Fdagbars[lam][l] += hyb_coeffs(l, nu, lam) * F_dags[nu];
        Fbarsrefl[lam][l] += hyb_refl_coeffs(l, lam, nu) * Fs[nu];
      }
    }
  }

  // initialize self-energy
  BlockDiagOpFun Sigma = BlockDiagOpFun(r, Gt.get_block_sizes());
  int num_block_cols   = Gt.get_num_block_cols();

  // preallocation
  bool path_all_nonzero;
  nda::vector<int> ind_path(3);
  nda::vector<int> block_dims(5); // intermediate block dimensions

  // loop over hybridization lines
  for (int fb1 = 0; fb1 <= 1; fb1++) {
    for (int fb2 = 0; fb2 <= 1; fb2++) {
      // fb = 1 for forward line, else = 0
      // fb1 corresponds with line from 0 to tau_2
      auto const &F1list    = (fb1) ? Fs : F_dags;
      auto const &F2list    = (fb2) ? Fs : F_dags;
      auto const &F3list    = (fb1) ? F_dags : Fs;
      auto const Fbar_array = (fb2) ? Fdagbars : Fbarsrefl;
      int sfM               = -1; // (fb1 ^ fb2) ? 1 : -1; // sign

      for (int i = 0; i < num_block_cols; i++) {

        // "backwards pass"
        //
        // for each self-energy block, find contributing blocks of factors
        //
        // paths_all_nonzero: false if for i-th block of
        // Sigma, factors assoc'd with lambda, mu, kappa don't contribute
        //
        // ind_path: a vector of column indices of the
        // blocks of the factors that contribute. if paths_all_nonzero is
        // false at this index, values in ind_path are garbage.
        //
        // ATTN: assumes all BlockOps in F(1,2,3)list have the same structure
        // i.e, index path is independent of kappa, mu
        path_all_nonzero   = true;
        auto Sigma_block_i = nda::make_regular(0 * Sigma.get_block(i));

        ind_path(0) = F1list[0].get_block_index(i);
        if (ind_path(0) == -1 || Gt.get_zero_block_index(ind_path(0)) == -1) {
          path_all_nonzero = false;
        } else {
          block_dims(0) = F1list[0].get_block_size(i, 1);
          block_dims(1) = F1list[0].get_block_size(i, 0);
          ind_path(1)   = F2list[0].get_block_index(ind_path(0));
          if (ind_path(1) == -1 || Gt.get_zero_block_index(ind_path(1)) == -1) {
            path_all_nonzero = false;
          } else {
            block_dims(2) = F2list[0].get_block_size(ind_path(0), 0);
            ind_path(2)   = F3list[0].get_block_index(ind_path(1));
            if (ind_path(2) == -1 || Gt.get_zero_block_index(ind_path(2)) == -1 || Fbar_array[0][0].get_block_index(ind_path(2)) == -1) {
              path_all_nonzero = false;
            } else {
              block_dims(3) = F3list[0].get_block_size(ind_path(1), 0);
              block_dims(4) = Fbar_array[0][0].get_block_size(ind_path(2), 0);
            }
          }
        }

        // matmuls and convolutions
        if (path_all_nonzero) {
          // preallocate for i-th block
          auto Sigma_l = nda::make_regular(0 * Sigma.get_block(i));
          // sizes of intermediate matrices are known
          nda::array<dcomplex, 3> Tright(r, block_dims(2), block_dims(1));   // output of OCA_bs_right
          nda::array<dcomplex, 3> Tmid(r, block_dims(3), block_dims(0));     // output of OCA_bs_middle
          nda::array<dcomplex, 4> Tkaps(n, r, block_dims(2), block_dims(0)); // storage in OCA_bs_middle
          nda::array<dcomplex, 3> Tmu(r, block_dims(2), block_dims(0));      // storage in OCA_bs_middle
          nda::array<dcomplex, 3> Tleft(r, block_dims(4), block_dims(0));    // output of OCA_bs_left
          nda::array<dcomplex, 3> GKt(r, block_dims(3), block_dims(3));      // storage in OCA_bs_left

          // TODO: make Fs have blocks that are 3D nda::array?
          auto Fkaps = nda::zeros<dcomplex>(n, F1list[0].get_block_size(i, 0), F1list[0].get_block_size(i, 1));
          auto Fmus  = nda::zeros<dcomplex>(n, F3list[0].get_block_size(ind_path(1), 0), F3list[0].get_block_size(ind_path(1), 1));
          for (int j = 0; j < n; j++) {
            Fkaps(j, _, _) = F1list[j].get_block(i);
            Fmus(j, _, _)  = F3list[j].get_block(ind_path(1));
          }

          for (int l = 0; l < p; l++) {
            Sigma_l = 0;
            for (int lam = 0; lam < n; lam++) {
              OCA_bs_right_in_place(beta, itops, dlr_it, hyb_poles(l), (fb2 == 1), Gt.get_block(ind_path(0)), Gt.get_block(ind_path(1)),
                                    F2list[lam].get_block(ind_path(0)), Tright);
              OCA_bs_middle_in_place((fb1 == 1), hyb, hyb_refl, Fkaps, Fmus, Tright, Tmid, Tkaps, Tmu);
              OCA_bs_left_in_place(beta, itops, dlr_it, hyb_poles(l), (fb2 == 1), Gt.get_block(ind_path(2)),
                                   Fbar_array[lam][l].get_block(ind_path(2)), Tmid, Tleft, GKt);
              Sigma_l += Tleft;
            } // sum over lambda

            // prefactor with Ks
            if (fb2 == 1) {
              if (hyb_poles(l) <= 0) {
                for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = k_it(dlr_it(t), hyb_poles(l)) * Sigma_l(t, _, _); }
                Sigma_l = Sigma_l / k_it(0, -hyb_poles(l));
              } else {
                Sigma_l = Sigma_l / k_it(0, hyb_poles(l));
              }
            } else {
              if (hyb_poles(l) >= 0) {
                for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = k_it(dlr_it(t), -hyb_poles(l)) * Sigma_l(t, _, _); }
                Sigma_l = Sigma_l / k_it(0, hyb_poles(l));
              } else {
                Sigma_l = Sigma_l / k_it(0, -hyb_poles(l));
              }
            }
            Sigma_block_i += nda::make_regular(sfM * Sigma_l);
          } // sum over l
          Sigma.add_block(i, Sigma_block_i);
        } // end if(path_all_nonzero)
      } // end loop over i
    } // end loop over fb2
  } // end loop over fb1
  return Sigma;
}

void OCA_dense_right_in_place(double beta, imtime_ops &itops, nda::vector_const_view<double> dlr_it, double omega_l, bool forward,
                              nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 2> Flam, nda::array_view<dcomplex, 3> T) {

  int r = Gt.extent(0);

  if (forward) {
    if (omega_l <= 0) {
      // 1. multiply F_lambda G(tau_1) K^-(tau_1)
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), -omega_l) * nda::matmul(Flam, Gt(t, _, _)); }
      // 2. convolve by G
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);
    } else {
      // 1. multiply G(tau_2-tau_1) K^+(tau_2-tau_1) F_lambda
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), omega_l) * nda::matmul(Gt(t, _, _), Flam); }
      // 2. convolve by G
      T = itops.convolve(beta, itops.vals2coefs(T), itops.vals2coefs(Gt), TIME_ORDERED);
    }
  } else {
    if (omega_l >= 0) {
      // 1. multiply F_lambda G(tau_1) K^+(tau_1)
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), omega_l) * nda::matmul(Flam, Gt(t, _, _)); }
      // 2. convolve by G
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);
    } else {
      // 1. multiply G(tau_2-tau_1) K^-(tau_2-tau_1) F_lambda
      for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), -omega_l) * nda::matmul(Gt(t, _, _), Flam); }
      // 2. convolve by G
      T = itops.convolve(beta, itops.vals2coefs(T), itops.vals2coefs(Gt), TIME_ORDERED);
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
    for (int t = 0; t < r; t++) { Tkaps(kap, t, _, _) = nda::matmul(T(t, _, _), Fkaps(kap, _, _)); }
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
          Tmu(t, _, _) += hyb_refl(t, mu, kap) * Tkaps(kap, t, _, _);
        }
      }
    }
    // 5. multiply by F^dag_mu and sum over mu
    for (int t = 0; t < r; t++) {
      T(t, _, _) += nda::matmul(Fmus(mu, _, _), Tmu(t, _, _)); // TODO ??? +=
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
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);
      // 7. multiply by Fbar
      for (int t = 0; t < r; t++) { T(t, _, _) = nda::matmul(Fbar, T(t, _, _)); }
    } else {
      // 6. convolve by G K^+
      for (int t = 0; t < r; t++) { GKt(t, _, _) = k_it(dlr_it(t), omega_l) * Gt(t, _, _); }
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED);
      // 7. multiply by Fbar
      for (int t = 0; t < r; t++) { T(t, _, _) = nda::matmul(Fbar, T(t, _, _)); }
    }
  } else {
    if (omega_l >= 0) {
      // 6. convolve by G
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);
      // 7. multiply by Fbar
      for (int t = 0; t < r; t++) { T(t, _, _) = nda::matmul(Fbar, T(t, _, _)); }
    } else {
      // 6. convolve by G K^-
      for (int t = 0; t < r; t++) { GKt(t, _, _) = k_it(dlr_it(t), -omega_l) * Gt(t, _, _); }
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED);
      // 7. multiply by Fbar
      for (int t = 0; t < r; t++) { T(t, _, _) = nda::matmul(Fbar, T(t, _, _)); }
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
            for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = k_it(dlr_it(t), dlr_rf(l)) * Sigma_l(t, _, _); }
            Sigma_l = Sigma_l / k_it(0, -dlr_rf(l));
          } else {
            Sigma_l = Sigma_l / k_it(0, dlr_rf(l));
          }
        } else {
          if (dlr_rf(l) >= 0) {
            for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = k_it(dlr_it(t), -dlr_rf(l)) * Sigma_l(t, _, _); }
            Sigma_l = Sigma_l / k_it(0, dlr_rf(l));
          } else {
            Sigma_l = Sigma_l / k_it(0, -dlr_rf(l));
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
            for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = k_it(dlr_it(t), hyb_poles(l)) * Sigma_l(t, _, _); }
            Sigma_l = Sigma_l / k_it(0, -hyb_poles(l));
          } else {
            Sigma_l = Sigma_l / k_it(0, hyb_poles(l));
          }
        } else {
          if (hyb_poles(l) >= 0) {
            for (int t = 0; t < r; t++) { Sigma_l(t, _, _) = k_it(dlr_it(t), -hyb_poles(l)) * Sigma_l(t, _, _); }
            Sigma_l = Sigma_l / k_it(0, hyb_poles(l));
          } else {
            Sigma_l = Sigma_l / k_it(0, -hyb_poles(l));
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
              for (int i = 1; i <= n_quad; i++) {
                for (int i1 = 1; i1 <= i; i1++) {
                  for (int i2 = 0; i2 <= i1; i2++) {
                    double w = 1.0;
                    if (i1 == i) w = w / 2;
                    if (i2 == 0 || i2 == i1) w = w / 2;
                    auto FGFGFGF = nda::matmul(
                       F4list(nu, _, _),
                       nda::matmul(Gt_eq(i - i1, _, _),
                                   nda::matmul(F3list(mu, _, _),
                                               nda::matmul(Gt_eq(i1 - i2, _, _),
                                                           nda::matmul(F2list(lam, _, _), nda::matmul(Gt_eq(i2, _, _), F1list(kap, _, _)))))));

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
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = k_it(dlr_it(t), -1 * poles(1)) * T(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);

      v = 2;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) Tkaps(kap, t, _, _) = nda::matmul(T(t, _, _), Fs(kap, _, _));
      }
      T = 0;
      for (int mu = 0; mu < n; mu++) {
        Tmu = 0;
        for (int kap = 0; kap < n; kap++) {
          for (int t = 0; t < r; t++) Tmu(t, _, _) += hyb(t, mu, kap) * Tkaps(kap, t, _, _);
        }
        for (int t = 0; t < r; t++) T(t, _, _) += nda::matmul(F_dags(mu, _, _), Tmu(t, _, _));
      }
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);

      v = 3;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = k_it(dlr_it(t), -1 * poles(0)) * T(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);

      v = 4;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fdagbars(states(v), l(1), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = k_it(dlr_it(t), poles(1)) * T(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);

      v = 5;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fdagbars(states(v), l(0), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = k_it(dlr_it(t), poles(0)) * T(t, _, _);

      T = T / (k_it(0, -1 * poles(0)) * k_it(0, -1 * poles(1)));
      Sigma += T;
    } else if (poles(0) <= 0 && poles(1) > 0) {
      int v = 1;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) GKt(t, _, _) = k_it(dlr_it(t), poles(1)) * Gt(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED);

      v = 2;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) Tkaps(kap, t, _, _) = nda::matmul(T(t, _, _), Fs(kap, _, _));
      }
      T = 0;
      for (int mu = 0; mu < n; mu++) {
        Tmu = 0;
        for (int kap = 0; kap < n; kap++) {
          for (int t = 0; t < r; t++) Tmu(t, _, _) += hyb(t, mu, kap) * Tkaps(kap, t, _, _);
        }
        for (int t = 0; t < r; t++) T(t, _, _) += nda::matmul(F_dags(mu, _, _), Tmu(t, _, _));
      }
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED);

      v = 3;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = k_it(dlr_it(t), -poles(0)) * T(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED);

      v = 4;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fdagbars(states(v), l(1), _, _), T(t, _, _));
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);

      v = 5;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fdagbars(states(v), l(0), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = k_it(dlr_it(t), poles(0)) * T(t, _, _);

      T = T / (k_it(0, -poles(0)) * k_it(0, poles(1)) * k_it(0, poles(1)));
      Sigma += T;
    } else if (poles(0) > 0 && poles(1) <= 0) {
      int v = 1;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = k_it(dlr_it(t), -1 * poles(1)) * T(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);

      v = 2;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) Tkaps(kap, t, _, _) = nda::matmul(T(t, _, _), Fs(kap, _, _));
      }
      T = 0;
      for (int mu = 0; mu < n; mu++) {
        Tmu = 0;
        for (int kap = 0; kap < n; kap++) {
          for (int t = 0; t < r; t++) Tmu(t, _, _) += hyb(t, mu, kap) * Tkaps(kap, t, _, _);
        }
        for (int t = 0; t < r; t++) T(t, _, _) += nda::matmul(F_dags(mu, _, _), Tmu(t, _, _));
      }
      T = itops.convolve(beta, itops.vals2coefs(Gt), itops.vals2coefs(T), TIME_ORDERED);

      v = 3;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) GKt(t, _, _) = k_it(dlr_it(t), poles(0)) * Gt(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED);

      v = 4;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fdagbars(states(v), l(1), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) T(t, _, _) = k_it(dlr_it(t), poles(1)) * T(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED);

      v = 5;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fdagbars(states(v), l(0), _, _), T(t, _, _));

      T = T / (k_it(0, poles(0)) * k_it(0, -1 * poles(1)));
      Sigma += T;
    } else {
      int v = 1;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) GKt(t, _, _) = k_it(dlr_it(t), poles(1)) * Gt(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED);

      v = 2;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) Tkaps(kap, t, _, _) = nda::matmul(T(t, _, _), Fs(kap, _, _));
      }
      T = 0;
      for (int mu = 0; mu < n; mu++) {
        Tmu = 0;
        for (int kap = 0; kap < n; kap++) {
          for (int t = 0; t < r; t++) Tmu(t, _, _) += hyb(t, mu, kap) * Tkaps(kap, t, _, _);
        }
        for (int t = 0; t < r; t++) T(t, _, _) += nda::matmul(F_dags(mu, _, _), Tmu(t, _, _));
      }
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED);

      v = 3;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fs(states(v), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) GKt(t, _, _) = k_it(dlr_it(t), poles(0)) * k_it(dlr_it(t), poles(1)) * Gt(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED);

      v = 4;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fdagbars(states(v), l(1), _, _), T(t, _, _));
      for (int t = 0; t < r; t++) GKt(t, _, _) = k_it(dlr_it(t), poles(0)) * Gt(t, _, _);
      T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED);

      v = 5;
      for (int t = 0; t < r; t++) T(t, _, _) = nda::matmul(Fdagbars(states(v), l(0), _, _), T(t, _, _));

      T = T / (k_it(0, poles(0)) * k_it(0, poles(1)) * k_it(0, poles(1)));
      Sigma += T;
    }
  }
  return Sigma;
}
