#include <gtest/gtest.h>

#include <triqs/operators/many_body_operator.hpp>

#include <triqs_xca/atom_diag_utils.hpp>

#include <triqs_xca/dense_backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs_xca/hyb.hpp>

#include <triqs_xca/strong_cpl.hpp>

#include "block_sparse_utils.hpp"

using triqs::atom_diag::atom_diag;
using triqs::atom_diag::fundamental_operator_set;

using triqs::operators::c;
using triqs::operators::c_dag;
using triqs::operators::many_body_operator_complex;
using triqs::operators::many_body_operator_real;
using triqs::operators::n;

using triqs_xca::dense::DenseDiagramEvaluator;
using triqs_xca::dense::DenseFSet;

using triqs_xca::block_sparse::BlockOpSymSet;

using triqs_xca::block_sparse::DiagramEvaluator;

using triqs_xca::atom_diag::ad_to_atom_prop;
using triqs_xca::atom_diag::get_full_h_atomic;
using triqs_xca::atom_diag::get_hamiltonian_blocks;
using triqs_xca::atom_diag::get_operators;
using triqs_xca::atom_diag::get_operators_dense;

using triqs_xca::atom_diag::get_tensor_in_atom_diag_subspace;

/**
 * @brief Test creation of a Backbone object 
 */
TEST(Backbone, indexing) {
  nda::array<int, 2> topology = {{0, 2}, {1, 4}, {3, 5}}; // a third-order diagram topology
  int n                       = 4;

  // DLR generation
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;
  auto dlr_rf   = build_dlr_rf(Lambda, eps);
  int r         = dlr_rf.size(); // r = 36

  // Indices for a particular backbone diagram
  nda::vector<int> fb{1, 1, 0};      // 0-2 line is forward, 1-4 line is forward, 3-5 line is backward
  nda::vector<int> pole_inds{3, 19}; // values of l, l'. omega_l < 0 and omega_l` > 0
  nda::vector<int> orb_inds{1, 3, 1, 2, 3, 2};

  auto B = Backbone(topology, n);
  B.set_directions(fb);
  B.set_pole_inds(pole_inds, dlr_rf);
  B.set_orb_inds(orb_inds);

  int fb_ix = 1 + 2 * 1;
  int p_ix  = 3 + r * 19;
  int o_ix  = 3 + n * 2;
  auto B2   = Backbone(topology, n);
  B2.set_directions(fb_ix);
  B2.set_pole_inds(p_ix, dlr_rf);
  B2.set_orb_inds(o_ix);

  // --- Check that the two ways of setting indices give the same result ---
  ASSERT_EQ(B.get_fb(0), B2.get_fb(0));
  ASSERT_EQ(B.get_fb(1), B2.get_fb(1));
  ASSERT_EQ(B.get_fb(2), B2.get_fb(2));
  ASSERT_EQ(B.get_pole_ind(0), B2.get_pole_ind(0));
  ASSERT_EQ(B.get_pole_ind(1), B2.get_pole_ind(1));
  ASSERT_EQ(B.get_orb_ind(1), B2.get_orb_ind(1));
  ASSERT_EQ(B.get_orb_ind(3), B2.get_orb_ind(3));
  ASSERT_EQ(B.get_orb_ind(4), B2.get_orb_ind(4));
  ASSERT_EQ(B.get_orb_ind(5), B2.get_orb_ind(5));

  int f_ix = o_ix + p_ix * n * n + fb_ix * n * n * r * r;
  auto B3  = Backbone(topology, n);
  B3.set_flat_index(f_ix, dlr_rf);

  ASSERT_EQ(B.get_fb(0), B3.get_fb(0));
  ASSERT_EQ(B.get_fb(1), B3.get_fb(1));
  ASSERT_EQ(B.get_fb(2), B3.get_fb(2));
  ASSERT_EQ(B.get_pole_ind(0), B3.get_pole_ind(0));
  ASSERT_EQ(B.get_pole_ind(1), B3.get_pole_ind(1));
  ASSERT_EQ(B.get_orb_ind(1), B3.get_orb_ind(1));
  ASSERT_EQ(B.get_orb_ind(3), B3.get_orb_ind(3));
  ASSERT_EQ(B.get_orb_ind(4), B3.get_orb_ind(4));
  ASSERT_EQ(B.get_orb_ind(5), B3.get_orb_ind(5));

  // --- Check that vertices are correct ---
  ASSERT_EQ(B.has_vertex_bar(0), false);
  ASSERT_EQ(B.has_vertex_bar(1), false);
  ASSERT_EQ(B.has_vertex_bar(2), false);
  ASSERT_EQ(B.has_vertex_bar(3), false);
  ASSERT_EQ(B.has_vertex_bar(4), true);
  ASSERT_EQ(B.has_vertex_bar(5), true);

  ASSERT_EQ(B.has_vertex_dag(0), false);
  ASSERT_EQ(B.has_vertex_dag(1), false);
  ASSERT_EQ(B.has_vertex_dag(2), true);
  ASSERT_EQ(B.has_vertex_dag(3), true);
  ASSERT_EQ(B.has_vertex_dag(4), true);
  ASSERT_EQ(B.has_vertex_dag(5), false);

  ASSERT_EQ(B.get_vertex_hyb_ind(1), 0);
  ASSERT_EQ(B.get_vertex_hyb_ind(3), 1);
  ASSERT_EQ(B.get_vertex_hyb_ind(4), 0);
  ASSERT_EQ(B.get_vertex_hyb_ind(5), 1);

  ASSERT_EQ(B.get_vertex_Ksign(1), -1);
  ASSERT_EQ(B.get_vertex_Ksign(3), 1);
  ASSERT_EQ(B.get_vertex_Ksign(4), 1);
  ASSERT_EQ(B.get_vertex_Ksign(5), -1);

  // --- Check that edges are correct ---
  nda::array<int, 2> exp_edges{2 * B.m - 1, B.m - 1};
  exp_edges = 0;
  for (int e = 0; e < 2 * B.m - 1; e++) {
    for (int p = 0; p < B.m - 1; p++) { ASSERT_EQ(B.get_edge(e, p), exp_edges(e, p)); }
  }

  // --- Check that prefactor is correct ---
  nda::vector<int> prefactor_Ksigns{-1, 1};
  nda::vector<int> prefactor_Kexps(B.m - 1);
  prefactor_Kexps = 1;
  for (int i = 0; i < B.m - 1; ++i) {
    ASSERT_EQ(B.get_prefactor_Ksign(i), prefactor_Ksigns(i));
    ASSERT_EQ(B.get_prefactor_Kexp(i), prefactor_Kexps(i));
  }
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a constant hybridization
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams.
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
  auto &G0_bdof          = one_fermion_model.G_bdof;

  // Check that G0_ppsc is correct by comparing to analytical expression
  auto dlr_it = itops.get_itnodes();
  auto G0_ana = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double t  = rel2abs(dlr_it(i));
    G0_ana(i) = -exp(-t * std::numbers::ln2);
  }
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) { ASSERT_LE(nda::max_element(nda::abs(G0_bdof.get_block(b)(_, 0, 0) - G0_ana)), eps); }

  // Set up diagram evaluator for self-energy evaluation
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);

  // ----- NCA test -----
  nda::array<int, 2> topology1 = {{0, 1}};
  auto nca_se                  = D.compute_self_energy(G0_ppsc, topology1);
  auto nca_se_ana              = nda::zeros<double>(r);
  nca_se_ana                   = -G0_ana / 2; // self-energy NCA contribution computed analytically
  // Compare computed and expected NCA
  ASSERT_LE(nda::max_element(nda::abs(nca_se[0].data()(_, 0, 0) - nca_se_ana)), eps);
  ASSERT_LE(nda::max_element(nda::abs(nca_se[1].data()(_, 0, 0) - nca_se_ana)), eps);

  // ----- OCA test -----
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  auto oca_se                  = D.compute_self_energy(G0_ppsc, topology2);
  // OCA contribution should be identically zero
  for (int i = 0; i < 2; ++i) { ASSERT_LE(nda::max_element(nda::abs(oca_se[i].data()(_, 0, 0))), eps); }

  // ----- third-order test -----
  nda::array<int, 2> topology = {{0, 3}, {1, 4}, {2, 5}};
  auto third_order_se         = D.compute_self_energy(G0_ppsc, topology);
  auto third_order_se_ana     = nda::zeros<double>(r);
  double t                    = 0;
  double bt4                  = 0;
  for (int i = 0; i < r; ++i) {
    t                     = rel2abs(dlr_it(i)); // t = tau / beta
    bt4                   = beta * t;
    bt4                   = bt4 * bt4;
    bt4                   = bt4 * bt4;
    third_order_se_ana(i) = -bt4 * exp(-t * std::numbers::ln2) / 192.0;
  }
  ASSERT_LE(nda::max_element(nda::abs(third_order_se[0].data()(_, 0, 0) - third_order_se_ana)), eps);
}

/**
 * @brief Test computation of several diagrams for a spinless fermion with a hybridization formed from a single pole
 *
 * @details This tests the evaluation of the first-, second-, and third-order self-energy diagrams.
 */
TEST(Backbone, one_fermion_three_orders_hyb_one_pole) {
  double beta   = 1.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-10;

  // Generate DLR imaginary-time object
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);
  int r       = itops.rank();

  auto one_fermion_model = one_fermion_model_helper(beta, Lambda, eps, 0.8);
  auto &hyb_coeffs       = one_fermion_model.hyb_coeffs;
  auto &hyb_poles        = one_fermion_model.hyb_poles;
  auto &ad               = one_fermion_model.ad;
  auto &G0_ppsc          = one_fermion_model.G_ppsc;
  auto &G0_bdof          = one_fermion_model.G_bdof;
  auto dlr_it = itops.get_itnodes();
  auto G0_ana = nda::zeros<double>(r);
  for (int i = 0; i < r; ++i) {
    double t  = rel2abs(dlr_it(i));
    G0_ana(i) = -exp(-t * std::numbers::ln2);
  }
  for (int b = 0; b < G0_bdof.get_num_block_cols(); ++b) { ASSERT_LE(nda::max_element(nda::abs(G0_bdof.get_block(b)(_, 0, 0) - G0_ana)), eps); }

  // NCA: set up backbone and diagram evaluator
  nda::array<int, 2> topology1 = {{0, 1}};
  DiagramEvaluator D(hyb_poles, hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto nca_se     = D.compute_self_energy(G0_ppsc, topology1);
  auto nca_se_ana = nda::zeros<double>(r, 2);
  double t        = 0;
  double om       = hyb_poles(0);
  double tom      = 0;
  for (int i = 0; i < r; ++i) {
    t                = rel2abs(dlr_it(i)); // t = tau / beta
    tom              = t * om;
    nca_se_ana(i, 0) = exp(-t * std::numbers::ln2) * exp(tom) / (exp(beta * om) + 1);
    nca_se_ana(i, 1) = exp(-t * std::numbers::ln2) * exp(-tom) / (exp(-beta * om) + 1);
  }
  ASSERT_LE(nda::max_element(nda::abs(nca_se[0].data()(_, 0, 0) - nca_se_ana(_, 0))), eps);
  ASSERT_LE(nda::max_element(nda::abs(nca_se[1].data()(_, 0, 0) - nca_se_ana(_, 1))), eps);

  // OCA
  nda::array<int, 2> topology2 = {{0, 2}, {1, 3}};
  auto oca_se                  = D.compute_self_energy(G0_ppsc, topology2);
  ASSERT_LE(nda::max_element(nda::abs(oca_se[0].data()(_, 0, 0))), eps);
  ASSERT_LE(nda::max_element(nda::abs(oca_se[1].data()(_, 0, 0))), eps);

  // third order
  nda::array<int, 2> topology3 = {{0, 3}, {1, 4}, {2, 5}};
  auto third_order_se          = D.compute_self_energy(G0_ppsc, topology3);
  auto third_order_se_ana      = nda::zeros<double>(r, 2);
  double denom                 = om * (exp(beta * om) + 1);
  denom                        = denom * denom * denom;
  denom                        = 2 * om * denom;
  for (int i = 0; i < r; ++i) {
    t                        = rel2abs(dlr_it(i)); // t = tau / beta
    tom                      = t * om;
    third_order_se_ana(i, 0) = (tom * tom * exp(tom) - 4 * tom * exp(tom) - 2 * tom + 6 * exp(tom) - 6) * exp(beta * om);
    third_order_se_ana(i, 1) += (tom * tom + 4 * tom + 2 * (tom - 3) * exp(tom) + 6) * exp(om * (2 * beta - t));
    third_order_se_ana(i, _) *= -exp(-t * std::numbers::ln2) / denom;
  }
  ASSERT_LE(nda::max_element(nda::abs(third_order_se_ana(_, 0) - third_order_se[0].data()(_, 0, 0))), eps);
  ASSERT_LE(nda::max_element(nda::abs(third_order_se_ana(_, 1) - third_order_se[1].data()(_, 0, 0))), eps);
}

TEST(Backbone, OCA_BDOF_construct) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // generate creation/annihilation operators
  auto [Deltat, Deltat_refl]              = discrete_bath_helper(beta, Lambda, eps);
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);

  // compute hybridization function reflection and coefficients
  auto hyb_coeffs               = itops.vals2coefs(Deltat); // hybridization DLR coeffs
  auto hyb_refl                 = Deltat;
  auto hyb_refl_coeffs          = hyb_coeffs;
  auto Fset                     = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs);
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);

  // set up backbone
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};

  // block-sparse diagram evaluation
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto start                            = std::chrono::high_resolution_clock::now();
  auto OCA_result_gf                    = D.compute_self_energy(Gt, topology);
  auto end                              = std::chrono::high_resolution_clock::now();
  auto OCA_result                       = BlockDiagOpFun(OCA_result_gf);
  std::chrono::duration<double> elapsed = end - start;

  // dense diagram evaluation
  DenseDiagramEvaluator D2(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  start  = std::chrono::high_resolution_clock::now();
  int n  = 4;
  auto B = Backbone(topology, n);
  D2.eval_self_energy(Gt_dense, B);
  end                   = std::chrono::high_resolution_clock::now();
  auto OCA_dense_result = D2.Sigma;
  elapsed               = end - start;

  // use block-sparse solver but with trivial sparsity
  std::vector<nda::array<dcomplex, 3>> Gt_dense_vec{Gt_dense};
  nda::vector<int> triv_bi{0};
  BlockDiagOpFun Gt_triv(Gt_dense_vec, triv_bi);

  std::vector<nda::array<dcomplex, 3>> Fs_dense_vec{Fs_dense};
  auto F_sym_triv = BlockOpSymSet(triv_bi, Fs_dense_vec);
  std::vector<nda::array<dcomplex, 3>> F_dags_dense_vec{F_dags_dense};
  auto F_dag_sym_triv      = BlockOpSymSet(triv_bi, F_dags_dense_vec);
  auto sym_set_labels_triv = nda::zeros<long>(n);
  auto Fq_triv             = BlockOpSymQuartet({F_sym_triv}, {F_dag_sym_triv}, hyb_coeffs, sym_set_labels_triv);

  DiagramEvaluator D3(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq_triv);
  start = std::chrono::high_resolution_clock::now();
  // D3.eval_self_energy(B);
  auto OCA_trivial_bs_gf = D3.compute_self_energy(Gt_triv, topology);
  end                    = std::chrono::high_resolution_clock::now();
  auto OCA_trivial_bs    = BlockDiagOpFun(OCA_trivial_bs_gf);
  elapsed                = end - start;

  auto ad = two_band_atom_diag_helper();
  for (int i = 0; i < OCA_result.get_num_block_cols(); i++) {

    auto result_dense_block = triqs_xca::atom_diag::get_tensor_in_atom_diag_subspace(OCA_dense_result, i, ad);
    ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - result_dense_block)), 10 * eps);

    auto result_trivial_bs_block = triqs_xca::atom_diag::get_tensor_in_atom_diag_subspace(OCA_trivial_bs.get_block(0), i, ad);
    ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - result_trivial_bs_block)), 10 * eps);
  }
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

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, nn);
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto start = std::chrono::high_resolution_clock::now();
  // D.eval_self_energy(B);
  auto result_gf                         = D.compute_self_energy(Gt, topology);
  auto end                               = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  auto result                            = BlockDiagOpFun(result_gf);

  // get dense Gt, field operators
  auto H_mat    = get_full_h_atomic(ad);
  auto Gt_dense = Hmat_to_Gtmat(H_mat, beta, dlr_it_abs);
  auto Fset     = get_operators_dense(ad, hyb_coeffs);

  DenseDiagramEvaluator D2(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  start = std::chrono::high_resolution_clock::now();
  D2.eval_self_energy(Gt_dense, B);
  end               = std::chrono::high_resolution_clock::now();
  duration          = end - start;
  auto result_dense = D2.Sigma;

  for (int i = 0; i < Gt_block_sizes.size(); i++) {
    auto result_dense_block = get_tensor_in_atom_diag_subspace(result_dense, i, ad);
    ASSERT_LE(nda::max_element(nda::abs(result.get_block(i) - result_dense_block)), eps);
  }
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

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, nn);
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto start = std::chrono::high_resolution_clock::now();
  // D.eval_self_energy(B);
  auto result_gf                         = D.compute_self_energy(Gt, topology);
  auto end                               = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  auto result                            = BlockDiagOpFun(result_gf);

  // get dense Gt, field operators
  auto H_mat    = get_full_h_atomic(ad);
  auto Gt_dense = Hmat_to_Gtmat(H_mat, beta, dlr_it_abs);
  auto Fset     = get_operators_dense(ad, hyb_coeffs);

  DenseDiagramEvaluator D2(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  start = std::chrono::high_resolution_clock::now();
  D2.eval_self_energy(Gt_dense, B);
  end               = std::chrono::high_resolution_clock::now();
  duration          = end - start;
  auto result_dense = D2.Sigma;

  for (int i = 0; i < Gt_block_sizes.size(); i++) {
    auto result_dense_block = get_tensor_in_atom_diag_subspace(result_dense, i, ad);
    ASSERT_LE(nda::max_element(nda::abs(result.get_block(i) - result_dense_block)), eps);
  }
}

TEST(Backbone, OCA_semicircle_bath_aaa) {
  // DLR parameters
  double beta   = 8.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);

  int p = 7;
  int n = 4;
  nda::vector<dcomplex> hyb_coeff_vals(p);
  nda::array<dcomplex, 3> hyb_coeffs(p, n, n);
  hyb_coeff_vals = {0.0028042961182163, 0.088487039172428,  0.1575418229076625, 0.1953880665937937,
                    0.2145207908265103, 0.1832496441339733, 0.1580088741667851};
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
  nda::array<dcomplex, 3> hyb_refl_coeffs(p, n, n);
  hyb_refl_coeffs = hyb_coeffs;

  nda::vector<double> hyb_poles(p);
  hyb_poles = {-2.537191963500981,  1.7111725610238615, -1.514666605887425, 1.04941790134832,
               -0.7410379494142222, 0.3763525311836938, -0.1312888711963961};
  auto hyb = triqs_xca::hyb::coefs2vals(beta, Lambda, eps, hyb_coeffs, hyb_poles);
  auto hyb_refl = hyb;
  hyb_poles = hyb_poles * beta;

  nda::vector<double> hyb_poles_reflect = -hyb_poles;
  auto Delta_decomp                     = hyb_decomp(hyb_coeffs, hyb_poles, eps);              //decomposition of Delta(t) using DLR coefficient
  auto Delta_decomp_reflect             = hyb_decomp(hyb_refl_coeffs, hyb_poles_reflect, eps); // decomposition of Delta(-t) using DLR coefficient
  hyb_F Delta_F(16, p, n);
  hyb_F Delta_F_reflect(16, p, n);
  auto dlr_it = itops.get_itnodes();
  Delta_F.update_inplace(Delta_decomp, dlr_it, Fs_dense, F_dags_dense); // Compression of Delta(t) and F, F_dag matrices
  Delta_F_reflect.update_inplace(Delta_decomp_reflect, dlr_it, F_dags_dense, Fs_dense);
  nda::array<int, 2> D2 = {{0, 2}, {1, 3}}; // topology for OCA diagram evaluator
  // Get Delta(t-t1) backward Delta(t2,t0) forward
  auto fb           = nda::vector<int64_t>(2);
  fb                = 0;
  auto OCA_forward  = Sigma_Diagram_calc(Delta_F, Delta_F_reflect, D2, hyb, hyb_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense, fb, true);
  fb(1)             = 1;
  auto OCA_backward = Sigma_Diagram_calc(Delta_F, Delta_F_reflect, D2, hyb, hyb_refl, Gt_dense, itops, beta, Fs_dense, F_dags_dense, fb, true);
  auto OCA_old      = nda::make_regular(OCA_forward + OCA_backward);

  // generic diagram evaluator
  nda::array<int, 2> topology   = {{0, 2}, {1, 3}};
  auto B                        = Backbone(topology, n);
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);
  auto D                        = DiagramEvaluator(beta, Lambda, eps, nda::make_regular(hyb_poles / beta), hyb_coeffs, Fq);
  auto OCA_result_gf            = D.compute_self_energy(Gt, topology);
  auto OCA_result               = BlockDiagOpFun(OCA_result_gf);

  auto ad = two_band_atom_diag_helper();
  for (int i = 0; i < OCA_result.get_num_block_cols(); i++) {
    auto result_dense_block = triqs_xca::atom_diag::get_tensor_in_atom_diag_subspace(OCA_old, i, ad);
    ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - result_dense_block)), 10 * eps);
  }
}

TEST(Backbone, spin_flip_fermion_aaa) {
  double beta   = 8.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  int norb = 2;
  int nn   = 2 * norb;
  int p    = 7;
  nda::array<dcomplex, 3> hyb_coeffs(p, nn, nn);
  nda::vector<dcomplex> hyb_coeffs00(p);
  hyb_coeffs00 = {0.0028042961182163, 0.088487039172428,  0.1575418229076625, 0.1953880665937937,
                  0.2145207908265103, 0.1832496441339733, 0.1580088741667851};
  for (int i = 0; i < nn; i++) {
    for (int j = i; j < nn; j++) {
      if (i / 2 == j / 2) {
        hyb_coeffs(_, i, j) = hyb_coeffs00;
      } else {
        hyb_coeffs(_, i, j) = 0.0;
      }
    }
  }
  nda::array<dcomplex, 3> hyb_refl_coeffs(p, nn, nn);
  hyb_refl_coeffs = hyb_coeffs;
  nda::vector<double> hyb_poles(p);
  hyb_poles = {-2.537191963500981,  1.7111725610238615, -1.514666605887425, 1.04941790134832,
               -0.7410379494142222, 0.3763525311836938, -0.1312888711963961};
  hyb_poles = hyb_poles * beta;

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
  auto dlr_it         = itops.get_itnodes();
  auto dlr_it_abs     = cppdlr::rel2abs(dlr_it);
  auto Gt             = ad_to_atom_prop(ad, beta, itops);
  auto Gt_block_sizes = Gt.get_block_sizes();

  // generate creation/annihilation operators in block-sparse storage
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, nn);
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(hyb_poles / beta), hyb_coeffs, Fq);
  auto start                             = std::chrono::high_resolution_clock::now();
  auto result_gf                         = D.compute_self_energy(Gt, topology);
  auto end                               = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  auto result                            = BlockDiagOpFun(result_gf);

  // get dense Gt, field operators
  auto H_mat    = get_full_h_atomic(ad);
  auto Gt_dense = Hmat_to_Gtmat(H_mat, beta, dlr_it_abs);
  auto Fset     = get_operators_dense(ad, hyb_coeffs);

  // compare to dense result
  DenseDiagramEvaluator D2(beta, eps, itops, hyb_poles, hyb_coeffs, Fset);
  start = std::chrono::high_resolution_clock::now();
  D2.eval_self_energy(Gt_dense, B);
  end               = std::chrono::high_resolution_clock::now();
  duration          = end - start;
  auto result_dense = D2.Sigma;

  for (int i = 0; i < Gt_block_sizes.size(); i++) {
    auto result_dense_block = get_tensor_in_atom_diag_subspace(result_dense, i, ad);
    ASSERT_LE(nda::max_element(nda::abs(result.get_block(i) - result_dense_block)), eps);
  }
}

TEST(Backbone, spin_flip_fermion_all_sym_aaa) {
  double beta   = 8.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  int norb = 2;
  int nn   = 2 * norb;
  int p    = 7;
  nda::array<dcomplex, 3> hyb_coeffs(p, nn, nn);
  nda::vector<dcomplex> hyb_coeffs00(p);
  hyb_coeffs00 = {0.0028042961182163, 0.088487039172428,  0.1575418229076625, 0.1953880665937937,
                  0.2145207908265103, 0.1832496441339733, 0.1580088741667851};

  for (int i = 0; i < nn; i++) {
    for (int j = i; j < nn; j++) {
      if ((i == j) || (j - i) % (nn / 2) == 0) {
        hyb_coeffs(_, i, j) = hyb_coeffs00;
      } else {
        hyb_coeffs(_, i, j) = 0.0;
      }
    }
  }
  nda::array<dcomplex, 3> hyb_refl_coeffs(p, nn, nn);
  hyb_refl_coeffs = hyb_coeffs;
  nda::vector<double> hyb_poles(p);
  hyb_poles = {-2.537191963500981,  1.7111725610238615, -1.514666605887425, 1.04941790134832,
               -0.7410379494142222, 0.3763525311836938, -0.1312888711963961};
  hyb_poles = hyb_poles * beta;

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

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, nn);
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(hyb_poles / beta), hyb_coeffs, Fq);
  auto start                             = std::chrono::high_resolution_clock::now();
  auto result_gf                         = D.compute_self_energy(Gt, topology);
  auto end                               = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  auto result                            = BlockDiagOpFun(result_gf);

  // get dense Gt, field operators
  auto H_mat    = get_full_h_atomic(ad);
  auto Gt_dense = Hmat_to_Gtmat(H_mat, beta, dlr_it_abs);
  auto Fset     = get_operators_dense(ad, hyb_coeffs);

  // compare to dense result
  DenseDiagramEvaluator D2(beta, eps, itops, hyb_poles, hyb_coeffs, Fset);
  start = std::chrono::high_resolution_clock::now();
  D2.eval_self_energy(Gt_dense, B);
  end               = std::chrono::high_resolution_clock::now();
  duration          = end - start;
  auto result_dense = D2.Sigma;

  for (int i = 0; i < Gt_block_sizes.size(); i++) {
    auto result_dense_block = get_tensor_in_atom_diag_subspace(result_dense, i, ad);
    ASSERT_LE(nda::max_element(nda::abs(result.get_block(i) - result_dense_block)), eps);
  }
}

TEST(Backbone, OCA_gf_construct) {
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
  auto OCA_result_gf = D.compute_self_energy(G0_ppsc, topology);
  BlockDiagOpFun OCA_result(OCA_result_gf);

  // compare against constructing Gt, Fq manually
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);
  DiagramEvaluator D2(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto OCA_result_2_gf = D2.compute_self_energy(Gt, topology);
  BlockDiagOpFun OCA_result_2(OCA_result_2_gf);
  ASSERT_LE(nda::max_element(nda::abs(D.hyb.values - D2.hyb.values)), eps);
  ASSERT_EQ(D.hyb.poles, D2.hyb.poles);

  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(0) - OCA_result_2.get_block(0))), 1e-10);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(1) - OCA_result_2.get_block(1))), 1e-10);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(2) - OCA_result_2.get_block(2))), 1e-10);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(3) - OCA_result_2.get_block(3))), 1e-10);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(4) - OCA_result_2.get_block(4))), 1e-10);
}

TEST(Backbone, manual_loop) {
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
  auto OCA_result = D.compute_self_energy(G0_ppsc, topology); // evaluate self-energy using built-in loop

  for (int f = 0; f < D.get_num_self_energy_backbones(topology); ++f) {
    OCA_result -= D.compute_self_energy(G0_ppsc, topology, f); // subtract off self-energy evaluation using manual loop
  }
  for (int i = 0; i < G0_bdof.get_num_block_cols(); ++i) { ASSERT_LE(nda::max_element(nda::abs(OCA_result[i].data())), 1e-10); }
}
