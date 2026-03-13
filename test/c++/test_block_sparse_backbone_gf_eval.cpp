#include <gtest/gtest.h>

#include <triqs_xca/atom_diag_utils.hpp>

#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>

#include <triqs_xca/strong_cpl.hpp>

#include "block_sparse_utils.hpp"

using triqs::operators::n;
using triqs::operators::c;
using triqs::operators::c_dag;

using triqs_xca::dense::DenseDiagramEvaluator;
using triqs_xca::dense::DenseFSet;

using triqs_xca::block_sparse::BlockOp;
using triqs_xca::block_sparse::DiagramEvaluator;

using triqs_xca::atom_diag::get_operators;
using triqs_xca::atom_diag::get_operators_dense;
using triqs_xca::atom_diag::get_full_h_atomic;
using triqs_xca::atom_diag::get_hamiltonian_blocks;
using triqs_xca::atom_diag::ad_to_atom_prop;

TEST(BSGFBackbone, NCA) {
  int n         = 4;
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // load hybridization, Green's functions, field operators
  auto [Deltat, Deltat_refl]    = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, itops.vals2coefs(Deltat));

  nda::array<int, 2> topology = {{0, 1}};
  auto B                      = CorrelatorBackbone(topology, n);
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), itops.vals2coefs(Deltat), Fq);
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
  auto NCA_result_gf                    = D.eval_correlator(Gt, B, mu_ops, kap_ops);
  auto end                              = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "NCA correlator evaluation took " << elapsed.count() << " seconds.\n";

  // compare to dense backbone result
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);
  auto Fset                               = DenseFSet(Fs_dense, F_dags_dense, itops.vals2coefs(Deltat));
  auto C                                  = DenseDiagramEvaluator(beta, eps, itops, dlr_rf, itops.vals2coefs(Deltat), Fset);
  auto NCA_result_gf_dense                = C.eval_correlator(Gt_dense, B, Fs_dense, F_dags_dense);

  ASSERT_LE(nda::max_element(nda::abs(NCA_result_gf - NCA_result_gf_dense)), 1.0e-15);
}

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
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), itops.vals2coefs(Deltat), Fq);
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
  auto OCA_result_gf                    = D.eval_correlator(Gt, B, mu_ops, kap_ops);
  auto end                              = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "OCA correlator evaluation took " << elapsed.count() << " seconds.\n";

  // compare to dense backbone result
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);
  auto Fset                               = DenseFSet(Fs_dense, F_dags_dense, itops.vals2coefs(Deltat));
  auto C                                  = DenseDiagramEvaluator(beta, eps, itops, dlr_rf, itops.vals2coefs(Deltat), Fset);
  auto OCA_result_gf_dense                = C.eval_correlator(Gt_dense, B, Fs_dense, F_dags_dense);

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
  triqs::atom_diag::fundamental_operator_set fop_set;

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
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto start                            = std::chrono::high_resolution_clock::now();
  auto OCA_result_gf                    = D.eval_correlator(Gt, B, mu_ops, kap_ops);
  auto end                              = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "OCA correlator evaluation took " << elapsed.count() << " seconds.\n";

  // compare to dense backbone result
  auto H_mat               = get_full_h_atomic(ad);
  auto Gt_dense            = Hmat_to_Gtmat(H_mat, beta, dlr_it_abs);
  auto Fset                = get_operators_dense(ad, hyb_coeffs);
  auto C                   = DenseDiagramEvaluator(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  auto OCA_result_gf_dense = C.eval_correlator(Gt_dense, B, Fset.Fs, Fset.F_dags);

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
  triqs::atom_diag::fundamental_operator_set fop_set;

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
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto start                            = std::chrono::high_resolution_clock::now();
  auto OCA_result_gf                    = D.eval_correlator(Gt, B, mu_ops, kap_ops);
  auto end                              = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "OCA correlator evaluation took " << elapsed.count() << " seconds.\n";

  // compare to dense backbone result
  auto H_mat               = get_full_h_atomic(ad);
  auto Gt_dense            = Hmat_to_Gtmat(H_mat, beta, dlr_it_abs);
  auto Fset                = get_operators_dense(ad, hyb_coeffs);
  auto C                   = DenseDiagramEvaluator(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  auto OCA_result_gf_dense = C.eval_correlator(Gt_dense, B, Fset.Fs, Fset.F_dags);

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
  nda::array<int, 2> D2                 = {{0, 2}, {1, 3}}; // topology for OCA diagram evaluator
  auto start                            = std::chrono::high_resolution_clock::now();
  auto OCA_gf_old                       = G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, D2, Gt_dense, itops, beta, Fs_dense, F_dags_dense);
  auto end                              = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Old dense OCA correlator evaluation took " << elapsed.count() << " seconds\n";

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

  auto D      = DiagramEvaluator(beta, Lambda, eps, nda::make_regular(hyb_poles / beta), hyb_coeffs, Fq);
  start       = std::chrono::high_resolution_clock::now();
  auto OCA_gf = D.eval_correlator(Gt, B, mu_ops, kap_ops);
  end         = std::chrono::high_resolution_clock::now();

  elapsed = end - start;
  std::cout << "OCA correlator evaluation took " << elapsed.count() << " seconds\n";

  ASSERT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_old)), eps);
}

TEST(Backbone, OCA_py_constructors) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_reflect]           = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);
  auto hyb_coeffs                         = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  // set up Kanamori Hamiltonian
  triqs::operators::many_body_operator_real H;
  triqs::atom_diag::fundamental_operator_set fop_set;
  int norb = 2;
  double U = 2.0;
  for (int i = 0; i < norb; i++) {
    H += U * n("up", i) * n("do", i);
    fop_set.insert("do", i);
  }
  double J  = 0.2;
  double Up = U - 2.0 * J;
  H += (Up - J) * (n("up", 0) * n("up", 1) + n("do", 0) * n("do", 1));
  H += Up * (n("up", 0) * n("do", 1) + n("do", 0) * n("up", 1));
  for (int k = 0; k < norb; ++k) {
    for (int l = 0; l < norb; ++l) {
      if (k != l) {
        H += J * (c_dag("up", k) * c_dag("do", k) * c("do", l) * c("up", l) + c_dag("up", k) * c_dag("do", l) * c("do", k) * c("up", l));
      }
    }
  }
  for (int i = 0; i < norb; i++) { fop_set.insert("up", i); }

  // Construct particle number operator and atom_diag object
  triqs::operators::many_body_operator_real N;
  for (int kap = 0; kap < norb; ++kap) { N += n("up", kap) + n("do", kap); }
  double mu = (3 * U - 5 * J) / 2 - 1.5;
  H -= mu * N;
  std::vector<triqs::operators::many_body_operator_real> sym_ops = {N};
  triqs::atom_diag::atom_diag<false> ad(H, fop_set, sym_ops);
  nda::vector<long> block_sizes(ad.n_subspaces());
  for (int i = 0; i < ad.n_subspaces(); ++i) { block_sizes(i) = ad.get_fock_states(i).size(); }

  // compute atomic propagator as a block_gf
  auto G0_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);
  BlockDiagOpFun G0_bdof(G0_ppsc);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  DiagramEvaluator D(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto OCA_gf = D.compute_single_ptcle_gf(G0_ppsc, topology);

  // compare against constructing Gt, Fq manually
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);
  DiagramEvaluator D2(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto OCA_gf_2 = D2.compute_single_ptcle_gf(Gt, topology);

  // compare against single index gf evaluator
  DiagramEvaluator D3(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto OCA_gf_3 = nda::make_regular(0 * OCA_gf);
  for (int f = 0; f < D.get_num_single_ptcle_gf_backbones(topology); ++f) { OCA_gf_3 += D3.compute_single_ptcle_gf(G0_ppsc, topology, f); }

  ASSERT_LE(nda::max_element(nda::abs(D.hyb.values - D2.hyb.values)), eps);
  ASSERT_EQ(D.hyb.poles, D2.hyb.poles);
  ASSERT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_2)), 1.0e-10);
  ASSERT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_3)), 1.0e-10);
}

TEST(Backbone, one_fermion_third_order_const_hyb) {
  double beta   = 4.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();

  // trivial hybridization, Delta = -1/2
  nda::array<dcomplex, 3> hyb(r, 1, 1);
  hyb      = -0.5;
  int p    = 1;
  int norb = 1;
  nda::array<dcomplex, 3> hyb_coeffs(p, norb, norb);
  hyb_coeffs = 1.0;
  nda::vector<double> hyb_poles(p);
  hyb_poles = 0.0;

  // trivial atomic Hamiltonian, H = 0
  triqs::operators::many_body_operator_real H;
  double mu = 0.0;
  triqs::operators::many_body_operator_real N;
  N = n("0", 0);
  H = -mu * N;
  triqs::atom_diag::fundamental_operator_set fop_set;
  fop_set.insert("0", 0);
  triqs::atom_diag::atom_diag<false> ad(H, fop_set);
  auto G0_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);
  BlockDiagOpFun G0_bdof(G0_ppsc);
  auto dlr_it = itops.get_itnodes();
  auto G0_ana = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double t  = rel2abs(dlr_it(i));
    G0_ana(i) = -exp(-t * std::numbers::ln2);
  }
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) { ASSERT_LE(nda::max_element(nda::abs(G0_bdof.get_block(b)(_, 0, 0) - G0_ana)), eps); }

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 3}, {1, 4}, {2, 5}};
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto third_order_gf     = D.compute_single_ptcle_gf(G0_ppsc, topology);
  auto third_order_gf_ana = nda::zeros<double>(r);
  double halfbeta         = beta / 2.0;
  double halfbetasq       = halfbeta * halfbeta;
  double halfbeta4        = halfbetasq * halfbetasq;
  for (int i = 0; i < r; ++i) {
    double t              = rel2abs(dlr_it(i)); // t = tau / beta
    third_order_gf_ana(i) = halfbeta4 * (1.0 - t) * (1.0 - t) * t * t / 2.0;
  }
  ASSERT_LE(nda::max_element(nda::abs(third_order_gf(_, 0, 0) - third_order_gf_ana)), eps);
}

TEST(Backbone, one_fermion_third_order_hyb_one_pole) {
  double beta   = 1.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();

  // hybridization with one pole
  int p    = 1;
  int norb = 1;
  nda::array<dcomplex, 3> hyb_coeffs(p, norb, norb);
  hyb_coeffs = 1.0;
  nda::vector<double> hyb_poles(p);
  hyb_poles = 0.8;

  // trivial atomic Hamiltonian, H = 0
  triqs::operators::many_body_operator_real H;
  double mu = 0.0;
  triqs::operators::many_body_operator_real N;
  N = n("0", 0);
  H = -mu * N;
  triqs::atom_diag::fundamental_operator_set fop_set;
  fop_set.insert("0", 0);
  triqs::atom_diag::atom_diag<false> ad(H, fop_set);
  auto G0_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);
  BlockDiagOpFun G0_bdof(G0_ppsc);
  auto dlr_it = itops.get_itnodes();
  auto G0_ana = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double t  = rel2abs(dlr_it(i));
    G0_ana(i) = -exp(-t * std::numbers::ln2);
  }
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) { ASSERT_LE(nda::max_element(nda::abs(G0_bdof.get_block(b)(_, 0, 0) - G0_ana)), eps); }

  // third order single particle gf
  nda::array<int, 2> topology = {{0, 3}, {1, 4}, {2, 5}};
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto third_order_gf     = D.compute_single_ptcle_gf(G0_ppsc, topology);
  auto third_order_gf_ana = nda::zeros<double>(r);
  double om               = hyb_poles(0);
  for (int i = 0; i < r; ++i) {
    double t              = rel2abs(dlr_it(i)); // t = tau / beta
    third_order_gf_ana(i) = (t + (exp(-om * t) - 1.0) / om) * (t - beta + (exp(om * (beta - t)) - 1.0) / om)
       / (2 * om * om * (1 + exp(-beta * om)) * (exp(beta * om) + 1));
  }

  ASSERT_LE(nda::max_element(nda::abs(third_order_gf(_, 0, 0) - third_order_gf_ana)), eps);
}

/*
TEST(Backbone, one_fermion_third_order_semic_hyb) {
  double beta   = 2.0;
  double Lambda = 200.0 * beta;
  double eps    = 1.0e-12;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();

  // hybridization with semicircular density
  int p    = 5;
  int norb = 1;
  nda::vector<dcomplex> weight{0.0890770688907171, 0.2501499813840898, 0.08907706889071, 0.2501499813841001, 0.3215458994474167};
  nda::array<dcomplex, 3> hyb_coeffs(p, norb, norb);
  hyb_coeffs          = 0;
  hyb_coeffs(_, 0, 0) = weight;
  nda::vector<double> hyb_poles{8.5871293257355785e-01, 4.8695002779475649e-01, -8.5871293257354586e-01, -4.8695002779478036e-01,
                                1.2681840371438871e-14};

  // trivial atomic Hamiltonian, H = 0
  triqs::operators::many_body_operator_real H;
  double mu = 0.0;
  triqs::operators::many_body_operator_real N;
  N = n("0", 0);
  H = -mu * N;
  triqs::atom_diag::fundamental_operator_set fop_set;
  fop_set.insert("0", 0);
  triqs::atom_diag::atom_diag<false> ad(H, fop_set);

  // get pseudoparticle propagator
  auto G0_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);
  BlockDiagOpFun G0_bdof(G0_ppsc);
  nda::array<dcomplex, 3> G0_dense(r, 2, 2);
  G0_dense                              = 0;
  G0_dense(_, range(0, 1), range(0, 1)) = G0_bdof.get_block(0);
  G0_dense(_, range(1, 2), range(1, 2)) = G0_bdof.get_block(1);

  auto dlr_it = itops.get_itnodes();
  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 3}, {1, 4}, {2, 5}};
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto third_order_gf = D.compute_single_ptcle_gf(G0_ppsc, topology);
  std::cout << "r = " << r << "\n";
  std::cout << std::setprecision(16) << "third_order_gf = " << third_order_gf(_, 0, 0) << std::endl;

  hyb_poles                             = hyb_poles * beta;
  nda::vector<double> hyb_poles_reflect = -hyb_poles;
  auto Delta_decomp                     = hyb_decomp(hyb_coeffs, hyb_poles, eps);
  auto Delta_decomp_reflect             = hyb_decomp(hyb_coeffs, hyb_poles_reflect, eps);
  hyb_F Delta_F(2, p, norb), Delta_F_reflect(2, p, norb);
  nda::array<dcomplex, 3> Fs(1, 2, 2), F_dags(1, 2, 2);
  Fs              = 0;
  F_dags          = 0;
  Fs(0, 0, 1)     = 1.0;
  F_dags(0, 1, 0) = 1.0;
  Delta_F.update_inplace(Delta_decomp, dlr_it, Fs, F_dags);
  Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F_dags, Fs);
  auto third_order_gf_old = G_Diagram_calc_sum_all(Delta_F, Delta_F_reflect, topology, G0_dense, itops, beta, Fs, F_dags);
  std::cout << std::setprecision(16) << "third_order_gf_old = " << third_order_gf_old(_, 0, 0) << std::endl;
}
*/