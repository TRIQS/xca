#include <cppdlr/utils.hpp>
#include <gtest/gtest.h>
#include "triqs_xca/strong_cpl.hpp"
#include <triqs_xca/block_sparse.hpp>
#include <triqs/atom_diag/gf.hpp>
#include "block_sparse_utils.hpp"
#include "triqs_xca/backbone.hpp"
#include "triqs_xca/block_sparse_backbone.hpp"
#include "triqs_xca/dense_backbone.hpp"
#include "triqs_xca/atom_diag_utils.hpp"
#include <algorithm>

using namespace triqs;
using namespace triqs::operators;
using namespace triqs::atom_diag;

TEST(BSGFBackbone, OCA_BDOF_construct) {
  int n         = 4;
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // load hybridization, Green's functions, field operators
  auto [Deltat, Deltat_refl]    = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, itops.vals2coefs(Deltat));

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = CorrelatorBackbone(topology, n);
  DiagramEvaluator D(beta, Lambda, eps, Deltat, nda::make_regular(dlr_rf / beta), Gt, Fq);
  // for now, convert Fq.Fs and Fq.F_dags to vectors of BlockOp
  std::vector<BlockOp> mu_ops, kap_ops;
  for (int i = 0; i < Fq.Fs[0].get_size_sym_set(); ++i) {
    std::vector<nda::array<dcomplex, 2>> mu_blocks, kap_blocks;
    for (int j = 0; j < Fq.Fs[0].get_num_block_cols(); ++j) {
      if (Fq.Fs[0].get_block_index(j) != -1) {
        mu_blocks.emplace_back(Fq.Fs[0].get_block(j)(i, _, _));
      } else {
        mu_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      }
      if (Fq.F_dags[0].get_block_index(j) != -1) {
        kap_blocks.emplace_back(Fq.F_dags[0].get_block(j)(i, _, _));
      } else {
        kap_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      }
    }
    nda::vector<int> block_indices = Fq.Fs[0].get_block_indices()(_);
    BlockOp mu_op(block_indices, mu_blocks);
    mu_ops.push_back(mu_op);
    block_indices = Fq.F_dags[0].get_block_indices()(_);
    BlockOp kap_op(block_indices, kap_blocks);
    kap_ops.push_back(kap_op);
  }

  auto start                            = std::chrono::high_resolution_clock::now();
  auto OCA_result_gf                    = D.eval_correlator(B, mu_ops, kap_ops);
  auto end                              = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "OCA correlator evaluation took " << elapsed.count() << " seconds.\n";

  // compare to dense backbone result
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);
  auto Fset                               = DenseFSet(Fs_dense, F_dags_dense, itops.vals2coefs(Deltat));
  auto C                                  = DenseDiagramEvaluator(beta, itops, Deltat, Deltat_refl, dlr_rf, Gt_dense, Fset);
  auto OCA_result_gf_dense                = C.eval_correlator(B, Fs_dense, F_dags_dense);

  ASSERT_LE(nda::max_element(nda::abs(OCA_result_gf - OCA_result_gf_dense)), 1.0e-15);
}

TEST(Backbone, spin_flip_fermion) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  int norb             = 2;
  int nn               = 2 * norb; // 2 * number of orbitals
  auto [hyb, hyb_refl] = discrete_bath_spin_flip_helper(beta, Lambda, eps, nn);
  auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs
  hyb_refl             = hyb;
  auto hyb_refl_coeffs = hyb_coeffs;

  // set up Hamiltonian
  triqs::operators::many_body_operator_real H;
  fundamental_operator_set fop_set;

  double mu = 0.25;
  double U  = 1.0;
  double V  = 0.1;
  for (int i = 0; i < norb; i++) {
    H += U * n("up", i) * n("do", i) + mu * (n("up", i) + n("do", i)) + V * (c_dag("up", i) * c("do", i) + c_dag("do", i) * c("up", i));
    fop_set.insert("do", i);
  }
  for (int i = 0; i < norb; i++) { fop_set.insert("up", i); }

  // Construct particle number operator
  triqs::operators::many_body_operator_real N;
  for (int kap = 0; kap < norb; ++kap) { N += n("up", kap) + n("do", kap); }
  std::vector<triqs::operators::many_body_operator_real> sym_ops = {N};

  // create atom_diag object
  triqs::atom_diag::atom_diag<false> ad(H, fop_set, sym_ops);

  // get blocks of Hamiltonian and compute noninteracting Green's function
  auto [H_blocks, H_block_inds] = get_hamiltonian_blocks(ad);
  auto dlr_it                   = itops.get_itnodes();
  auto dlr_it_abs               = cppdlr::rel2abs(dlr_it);
  auto Gt                       = ad_to_atom_prop(ad, beta, itops);
  auto Gt_block_sizes           = Gt.get_block_sizes();

  // generate creation/annihilation operators in block-sparse storage
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);
  std::vector<BlockOp> mu_ops, kap_ops;
  for (int i = 0; i < Fq.Fs[0].get_size_sym_set(); ++i) {
    std::vector<nda::array<dcomplex, 2>> mu_blocks, kap_blocks;
    for (int j = 0; j < Fq.Fs[0].get_num_block_cols(); ++j) {
      if (Fq.Fs[0].get_block_index(j) != -1) {
        mu_blocks.emplace_back(Fq.Fs[0].get_block(j)(i, _, _));
      } else {
        mu_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      }
      if (Fq.F_dags[0].get_block_index(j) != -1) {
        kap_blocks.emplace_back(Fq.F_dags[0].get_block(j)(i, _, _));
      } else {
        kap_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      }
    }
    nda::vector<int> block_indices = Fq.Fs[0].get_block_indices()(_);
    BlockOp mu_op(block_indices, mu_blocks);
    mu_ops.push_back(mu_op);
    block_indices = Fq.F_dags[0].get_block_indices()(_);
    BlockOp kap_op(block_indices, kap_blocks);
    kap_ops.push_back(kap_op);
  }

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = CorrelatorBackbone(topology, nn);
  DiagramEvaluator D(beta, Lambda, eps, hyb, nda::make_regular(dlr_rf / beta), Gt, Fq);
  auto start                            = std::chrono::high_resolution_clock::now();
  auto OCA_result_gf                    = D.eval_correlator(B, mu_ops, kap_ops);
  auto end                              = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "OCA correlator evaluation took " << elapsed.count() << " seconds.\n";

  // compare to dense backbone result
  auto H_mat               = get_full_h_atomic_perm(ad);
  auto Gt_dense            = Hmat_to_Gtmat(H_mat, beta, dlr_it_abs);
  auto Fset                = get_operators_dense(ad, norb, hyb_coeffs);
  auto C                   = DenseDiagramEvaluator(beta, itops, hyb, hyb_refl, dlr_rf, Gt_dense, Fset);
  auto OCA_result_gf_dense = C.eval_correlator(B, Fset.Fs, Fset.F_dags);

  ASSERT_LE(nda::max_element(nda::abs(OCA_result_gf - OCA_result_gf_dense)), 1.0e-15);
}

TEST(Backbone, spin_flip_fermion_sym_sets) {

  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  int norb             = 2;
  int nn               = 2 * norb; // 2 * number of orbitals
  auto [hyb, hyb_refl] = discrete_bath_spin_flip_helper(beta, Lambda, eps, nn);
  auto hyb_coeffs      = itops.vals2coefs(hyb); // hybridization DLR coeffs
  hyb_refl             = hyb;
  auto hyb_refl_coeffs = hyb_coeffs;

  // set up Hamiltonian
  triqs::operators::many_body_operator_real H;
  fundamental_operator_set fop_set;

  double mu = 0.25;
  double U  = 1.0;
  double V  = 0.1;
  for (int i = 0; i < norb; i++) {
    H += U * n("up", i) * n("do", i) + mu * (n("up", i) + n("do", i)) + V * (c_dag("up", i) * c("do", i) + c_dag("do", i) * c("up", i));
    fop_set.insert("do", i);
  }
  for (int i = 0; i < norb; i++) { fop_set.insert("up", i); }

  // Construct particle number operator
  triqs::operators::many_body_operator_real N;
  for (int kap = 0; kap < norb; ++kap) { N += n("up", kap) + n("do", kap); }

  // create atom_diag object
  triqs::atom_diag::atom_diag<false> ad(H, fop_set);

  // get blocks of Hamiltonian and compute noninteracting Green's function
  auto [H_blocks, H_block_inds] = get_hamiltonian_blocks(ad);
  auto dlr_it                   = itops.get_itnodes();
  auto dlr_it_abs               = cppdlr::rel2abs(dlr_it);
  auto Gt                       = ad_to_atom_prop(ad, beta, itops);
  auto Gt_block_sizes           = Gt.get_block_sizes();

  // generate creation/annihilation operators in block-sparse storage
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);
  std::vector<BlockOp> mu_ops, kap_ops;
  for (auto &F : Fq.Fs) {
    for (int i = 0; i < F.get_size_sym_set(); ++i) {
      std::vector<nda::array<dcomplex, 2>> mu_blocks;
      for (int j = 0; j < F.get_num_block_cols(); ++j) {
        if (F.get_block_index(j) != -1) {
          mu_blocks.emplace_back(F.get_block(j)(i, _, _));
        } else {
          mu_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
        }
      }
      nda::vector<int> block_indices = F.get_block_indices()(_);
      BlockOp mu_op(block_indices, mu_blocks);
      mu_ops.push_back(mu_op);
    }
  }
  for (auto &F_dag : Fq.F_dags) {
    for (int i = 0; i < F_dag.get_size_sym_set(); ++i) {
      std::vector<nda::array<dcomplex, 2>> kap_blocks;
      for (int j = 0; j < F_dag.get_num_block_cols(); ++j) {
        if (F_dag.get_block_index(j) != -1) {
          kap_blocks.emplace_back(F_dag.get_block(j)(i, _, _));
        } else {
          kap_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
        }
      }
      nda::vector<int> block_indices = F_dag.get_block_indices()(_);
      BlockOp kap_op(block_indices, kap_blocks);
      kap_ops.push_back(kap_op);
    }
  }
  std::swap(mu_ops[1], mu_ops[2]);
  std::swap(kap_ops[1], kap_ops[2]);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = CorrelatorBackbone(topology, nn);
  DiagramEvaluator D(beta, Lambda, eps, hyb, nda::make_regular(dlr_rf / beta), Gt, Fq);
  auto start                            = std::chrono::high_resolution_clock::now();
  auto OCA_result_gf                    = D.eval_correlator(B, mu_ops, kap_ops);
  auto end                              = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "OCA correlator evaluation took " << elapsed.count() << " seconds.\n";

  // compare to dense backbone result
  auto H_mat               = get_full_h_atomic_perm(ad);
  auto Gt_dense            = Hmat_to_Gtmat(H_mat, beta, dlr_it_abs);
  auto Fset                = get_operators_dense(ad, norb, hyb_coeffs);
  auto C                   = DenseDiagramEvaluator(beta, itops, hyb, hyb_refl, dlr_rf, Gt_dense, Fset);
  auto OCA_result_gf_dense = C.eval_correlator(B, Fset.Fs, Fset.F_dags);

  ASSERT_LE(nda::max_element(nda::abs(OCA_result_gf - OCA_result_gf_dense)), 1.0e-15);
}

TEST(Backbone, OCA_semicircle_bath_aaa) {
  // DLR parameters
  double beta   = 8.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();

  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);

  int p = 7;
  int n = 4;
  nda::vector<dcomplex> hyb_vals(r), hyb_coeff_vals(p);
  nda::array<dcomplex, 3> hyb(r, n, n), hyb_coeffs(p, n, n);
  hyb_vals       = {-0.4997496184487105, -0.4867352379479528, -0.4603465101833711, -0.4239204950540695, -0.3716597467714097,
                    -0.2884886574148449, -0.2479810727230272, -0.2065525284769785, -0.1635819676241178, -0.1326995066858671,
                    -0.1225444804140666, -0.1282199855712255, -0.1386184647087601, -0.1720919948804938, -0.2300400167898313,
                    -0.3000508284935615, -0.3759657450111002, -0.4545389745912252, -0.4821599768174421, -0.4997496184487105};
  hyb_coeff_vals = {0.0028042961182163, 0.088487039172428,  0.1575418229076625, 0.1953880665937937,
                    0.2145207908265103, 0.1832496441339733, 0.1580088741667851};
  for (int t = 0; t < r; t++) {
    hyb(t, 0, 0) = hyb_vals(t);
    hyb(t, 1, 1) = hyb_vals(t);
    hyb(t, 2, 2) = hyb_vals(t);
    hyb(t, 3, 3) = hyb_vals(t);
    hyb(t, 0, 1) = hyb_vals(t);
    hyb(t, 1, 0) = hyb_vals(t);
    hyb(t, 2, 3) = hyb_vals(t);
    hyb(t, 3, 2) = hyb_vals(t);
  }
  for (int l = 0; l < p; l++) {
    hyb_coeffs(l, 0, 0) = hyb_coeff_vals(l);
    hyb_coeffs(l, 1, 1) = hyb_coeff_vals(l);
    hyb_coeffs(l, 2, 2) = hyb_coeff_vals(l);
    hyb_coeffs(l, 3, 3) = hyb_coeff_vals(l);
    hyb_coeffs(l, 0, 1) = hyb_coeff_vals(l);
    hyb_coeffs(l, 1, 0) = hyb_coeff_vals(l);
    hyb_coeffs(l, 2, 3) = hyb_coeff_vals(l);
    hyb_coeffs(l, 3, 2) = hyb_coeff_vals(l);
  }
  auto hyb_refl = hyb;
  nda::array<dcomplex, 3> hyb_refl_coeffs(p, n, n);
  hyb_refl_coeffs = hyb_coeffs;

  nda::vector<double> hyb_poles(p);
  hyb_poles = {-2.537191963500981,  1.7111725610238615, -1.514666605887425, 1.04941790134832,
               -0.7410379494142222, 0.3763525311836938, -0.1312888711963961};
  hyb_poles = hyb_poles * beta;

  nda::vector<double> hyb_poles_reflect = -hyb_poles;
  auto Delta_decomp                     = hyb_decomp(hyb_coeffs, hyb_poles, eps);              //decomposition of Delta(t) using DLR coefficient
  auto Delta_decomp_reflect             = hyb_decomp(hyb_refl_coeffs, hyb_poles_reflect, eps); // decomposition of Delta(-t) using DLR coefficient
  hyb_F Delta_F(16, p, n), Delta_F_reflect(16, p, n);
  auto dlr_it = itops.get_itnodes();
  Delta_F.update_inplace(Delta_decomp, dlr_it, Fs_dense, F_dags_dense); // Compression of Delta(t) and F, F_dag matrices
  Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F_dags_dense, Fs_dense);
  nda::array<int, 2> D2 = {{0, 2}, {1, 3}}; // topology for OCA diagram evaluator
  auto OCA_gf_old       = G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D2, Gt_dense, itops, beta, Fs_dense, F_dags_dense);

  // generic diagram evaluator
  nda::array<int, 2> topology   = {{0, 2}, {1, 3}};
  auto B                        = CorrelatorBackbone(topology, n);
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);

  std::vector<BlockOp> mu_ops, kap_ops;
  for (int i = 0; i < Fq.Fs[0].get_size_sym_set(); ++i) {
    std::vector<nda::array<dcomplex, 2>> mu_blocks, kap_blocks;
    for (int j = 0; j < Fq.Fs[0].get_num_block_cols(); ++j) {
      if (Fq.Fs[0].get_block_index(j) != -1) {
        mu_blocks.emplace_back(Fq.Fs[0].get_block(j)(i, _, _));
      } else {
        mu_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      }
      if (Fq.F_dags[0].get_block_index(j) != -1) {
        kap_blocks.emplace_back(Fq.F_dags[0].get_block(j)(i, _, _));
      } else {
        kap_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      }
    }
    nda::vector<int> block_indices = Fq.Fs[0].get_block_indices()(_);
    BlockOp mu_op(block_indices, mu_blocks);
    mu_ops.push_back(mu_op);
    block_indices = Fq.F_dags[0].get_block_indices()(_);
    BlockOp kap_op(block_indices, kap_blocks);
    kap_ops.push_back(kap_op);
  }

  auto D      = DiagramEvaluator(beta, Lambda, eps, hyb, nda::make_regular(hyb_poles / beta), Gt, Fq);
  auto OCA_gf = D.eval_correlator(B, mu_ops, kap_ops);

  ASSERT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_old)), eps);
}