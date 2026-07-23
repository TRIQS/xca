#include <gtest/gtest.h>

#include <nda/algorithms.hpp>
#include <triqs_xca/atom_diag_utils.hpp>

#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs_xca/hyb.hpp>

#include <triqs_xca/strong_cpl.hpp>

#include "block_sparse_utils.hpp"

using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::n;

using triqs_xca::dense::DenseDiagramEvaluator;
using triqs_xca::dense::DenseFSet;

using triqs_xca::block_sparse::BlockOp;
using triqs_xca::block_sparse::DiagramEvaluator;

using triqs_xca::atom_diag::ad_to_atom_prop;
using triqs_xca::atom_diag::get_full_h_atomic;
using triqs_xca::atom_diag::get_hamiltonian_blocks;
using triqs_xca::atom_diag::get_operators;
using triqs_xca::atom_diag::get_operators_dense;

/**
 * @brief Test computation of several diagrams for a spinless fermion with a constant hybridization
 *
 * @details This tests the evaluation of the first-, second-, and third-order single-particle Green's functions diagrams.
 */
TEST(Backbone, one_fermion_three_orders_const_hyb) {
  // Generate DLR imaginary-time object
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;
  auto dlr_rf   = build_dlr_rf(Lambda, eps);
  auto itops    = imtime_ops(Lambda, dlr_rf);
  int r         = itops.rank();

  auto one_fermion_model = one_fermion_model_helper(beta, Lambda, eps);
  auto &hyb_coeffs       = one_fermion_model.hyb_coeffs;
  auto &hyb_poles        = one_fermion_model.hyb_poles;
  auto &ad               = one_fermion_model.ad;
  auto &G0_ppsc          = one_fermion_model.G_ppsc;
  auto dlr_it            = itops.get_itnodes();

  // Set up diagram evaluator for single-particle Green's function evalution
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);

  // ----- NCA test -----
  nda::array<int, 2> topology1 = {{0, 1}};
  auto nca_gf                  = D.compute_single_ptcle_gf(G0_ppsc, topology1);
  auto nca_gf_ana              = nda::ones<dcomplex>(r);
  nca_gf_ana                   = nca_gf_ana / 2;
  // Compare computed and expected NCA
  ASSERT_LE(nda::max_element(nda::abs(nca_gf(_, 0, 0) - nca_gf_ana)), eps);

  // ----- OCA test -----
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  auto oca_gf                  = D.compute_single_ptcle_gf(G0_ppsc, topology2);
  // OCA contribution should be identically zero
  ASSERT_LE(nda::max_element(nda::abs(oca_gf)), eps);

  // ----- third-order test -----
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  auto third_order_gf          = D.compute_single_ptcle_gf(G0_ppsc, topology3);
  auto third_order_gf_ana      = nda::zeros<double>(r);
  double halfbeta              = beta / 2.0;
  double halfbetasq            = halfbeta * halfbeta;
  double halfbeta4             = halfbetasq * halfbetasq;
  for (int i = 0; i < r; ++i) {
    double t              = rel2abs(dlr_it(i)); // t = tau / beta
    third_order_gf_ana(i) = -halfbeta4 * (1.0 - t) * (1.0 - t) * t * t / 2.0;
  }
  ASSERT_LE(nda::max_element(nda::abs(third_order_gf(_, 0, 0) - third_order_gf_ana)), eps);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from a single pole
 *
 * @details This tests the evaluation of the first-, second-, and third-order single-particle Green's functions diagrams.
 */
TEST(Backbone, one_fermion_three_orders_hyb_one_pole) {
  // Generate DLR imaginary-time object
  double beta   = 1.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;
  auto dlr_rf   = build_dlr_rf(Lambda, eps);
  auto itops    = imtime_ops(Lambda, dlr_rf);
  int r         = itops.rank();

  auto one_fermion_model = one_fermion_model_helper(beta, Lambda, eps, 0.8);
  auto &hyb_coeffs       = one_fermion_model.hyb_coeffs;
  auto &hyb_poles        = one_fermion_model.hyb_poles;
  auto &ad               = one_fermion_model.ad;
  auto &G0_ppsc          = one_fermion_model.G_ppsc;
  auto dlr_it            = itops.get_itnodes();

  // Set up diagram evaluator for single-particle Green's function evalution
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);

  // ----- NCA test -----
  nda::array<int, 2> topology1 = {{0, 1}};
  auto nca_gf                  = D.compute_single_ptcle_gf(G0_ppsc, topology1);
  auto nca_gf_ana              = nda::ones<dcomplex>(r);
  nca_gf_ana                   = nca_gf_ana / 2;
  // Compare computed and expected NCA
  ASSERT_LE(nda::max_element(nda::abs(nca_gf(_, 0, 0) - nca_gf_ana)), eps);

  // ----- OCA test -----
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  auto oca_gf                  = D.compute_single_ptcle_gf(G0_ppsc, topology2);
  // OCA contribution should be identically zero
  ASSERT_LE(nda::max_element(nda::abs(oca_gf)), eps);

  // ----- third-order test -----
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  auto third_order_gf          = D.compute_single_ptcle_gf(G0_ppsc, topology3);
  auto third_order_gf_ana      = nda::zeros<double>(r);
  double om                    = hyb_poles(0);
  for (int i = 0; i < r; ++i) {
    double t              = rel2abs(dlr_it(i)); // t = tau / beta
    third_order_gf_ana(i) = -(t + (exp(-om * t) - 1.0) / om) * (t - beta + (exp(om * (beta - t)) - 1.0) / om)
       / (2 * om * om * (1 + exp(-beta * om)) * (exp(beta * om) + 1));
  }

  ASSERT_LE(nda::max_element(nda::abs(third_order_gf(_, 0, 0) - third_order_gf_ana)), eps);
}

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
  triqs::operators::many_body_operator_complex H;
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
  triqs::operators::many_body_operator_complex N;
  for (int kap = 0; kap < norb; ++kap) { N += n("up", kap) + n("do", kap); }
  std::vector<triqs::operators::many_body_operator_complex> sym_ops = {N};

  // create atom_diag object
  triqs::atom_diag::atom_diag<true> ad(H, fop_set, sym_ops);

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
  triqs::operators::many_body_operator_complex H;
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
  triqs::operators::many_body_operator_complex N;
  for (int kap = 0; kap < norb; ++kap) { N += n("up", kap) + n("do", kap); }

  // create atom_diag object
  triqs::atom_diag::atom_diag<true> ad(H, fop_set);

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
  triqs::operators::many_body_operator_complex H;
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
  triqs::operators::many_body_operator_complex N;
  for (int kap = 0; kap < norb; ++kap) { N += n("up", kap) + n("do", kap); }
  double mu = (3 * U - 5 * J) / 2 - 1.5;
  H -= mu * N;
  std::vector<triqs::operators::many_body_operator_complex> sym_ops = {N};
  triqs::atom_diag::atom_diag<true> ad(H, fop_set, sym_ops);
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
