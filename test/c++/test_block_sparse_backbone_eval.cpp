#include <cstdint>
#include <iomanip>
#include <iostream>
#include <gtest/gtest.h>
#include <nda/basic_functions.hpp>
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include "block_sparse_utils.hpp"
#include "triqs_xca/dense_backbone.hpp"
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/block_sparse_manual.hpp>
#include <triqs_xca/backbone.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/operators/many_body_operator.hpp>
#include <triqs_xca/atom_diag_utils.hpp>
#include <triqs_xca/self_energy.hpp>

using namespace triqs;
using namespace triqs::operators;
using namespace triqs::atom_diag;

TEST(Backbone, flat_index) {
  nda::array<int, 2> topology = {{0, 2}, {1, 4}, {3, 5}};
  int n                       = 4;

  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  int r       = dlr_rf.size();

  nda::vector<int> fb{1, 1, 0};
  nda::vector<int> pole_inds{3, 10};
  nda::vector<int> orb_inds{1, 3, 1, 2, 3, 2};

  auto B = Backbone(topology, n);
  B.set_directions(fb);
  B.set_pole_inds(pole_inds, dlr_rf);
  B.set_orb_inds(orb_inds);

  int fb_ix = 1 + 2 * 1;
  int p_ix  = 3 + r * 10;
  int o_ix  = 3 + n * 2;
  auto B2   = Backbone(topology, n);
  B2.set_directions(fb_ix);
  B2.set_pole_inds(p_ix, dlr_rf);
  B2.set_orb_inds(o_ix);

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
}

TEST(Backbone, OCA) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

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
  auto Fset                     = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs, hyb_refl_coeffs);
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs, hyb_refl_coeffs);

  // set up backbone
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  int n                       = 4;
  auto B                      = Backbone(topology, n);

  // block-sparse diagram evaluation
  DiagramEvaluator D(beta, itops, Deltat, hyb_refl, dlr_rf, Gt, Fq);
  auto start = std::chrono::high_resolution_clock::now();
  D.eval_self_energy(B);
  auto end                              = std::chrono::high_resolution_clock::now();
  auto OCA_result                       = D.Sigma;
  std::chrono::duration<double> elapsed = end - start;

  // dense diagram evaluation
  DenseDiagramEvaluator D2(beta, itops, Deltat, hyb_refl, dlr_rf, Gt_dense, Fset);
  start = std::chrono::high_resolution_clock::now();
  D2.eval_self_energy(B);
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
  auto Fq_triv             = BlockOpSymQuartet({F_sym_triv}, {F_dag_sym_triv}, hyb_coeffs, hyb_refl_coeffs, sym_set_labels_triv);

  DiagramEvaluator D3(beta, itops, Deltat, hyb_refl, dlr_rf, Gt_triv, Fq_triv);
  start = std::chrono::high_resolution_clock::now();
  D3.eval_self_energy(B);
  end                 = std::chrono::high_resolution_clock::now();
  auto OCA_trivial_bs = D3.Sigma;
  elapsed             = end - start;

  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(0) - OCA_dense_result(_, range(0, 4), range(0, 4)))), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(1) - OCA_dense_result(_, range(4, 10), range(4, 10)))), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(2) - OCA_dense_result(_, range(10, 11), range(10, 11)))), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(3) - OCA_dense_result(_, range(11, 15), range(11, 15)))), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(4) - OCA_dense_result(_, range(15, 16), range(15, 16)))), eps);

  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(0) - OCA_trivial_bs.get_block(0)(_, range(0, 4), range(0, 4)))), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(1) - OCA_trivial_bs.get_block(0)(_, range(4, 10), range(4, 10)))), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(2) - OCA_trivial_bs.get_block(0)(_, range(10, 11), range(10, 11)))), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(3) - OCA_trivial_bs.get_block(0)(_, range(11, 15), range(11, 15)))), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(4) - OCA_trivial_bs.get_block(0)(_, range(15, 16), range(15, 16)))), eps);
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
  auto Gt                       = ad_to_nonint_gf(ad, beta, dlr_it_abs);
  auto Gt_block_sizes           = Gt.get_block_sizes();

  // generate creation/annihilation operators in block-sparse storage
  auto [Fq, sym_set_labels] = get_operators(ad, norb, hyb_coeffs, hyb_refl_coeffs);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, nn);
  DiagramEvaluator D(beta, itops, hyb, hyb_refl, dlr_rf, Gt, Fq);
  auto start = std::chrono::high_resolution_clock::now();
  D.eval_self_energy(B);
  auto end                               = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  auto result                            = D.Sigma;

  // get dense Gt, field operators
  auto H_mat    = get_full_h_atomic_perm(ad);
  auto Gt_dense = Hmat_to_Gtmat(H_mat, beta, dlr_it_abs);
  auto Fset     = get_operators_dense(ad, norb, hyb_coeffs, hyb_refl_coeffs);

  DenseDiagramEvaluator D2(beta, itops, hyb, hyb_refl, dlr_rf, Gt_dense, Fset);
  start = std::chrono::high_resolution_clock::now();
  D2.eval_self_energy(B);
  end               = std::chrono::high_resolution_clock::now();
  duration          = end - start;
  auto result_dense = D2.Sigma;

  int s0 = 0, s1 = 0;
  for (int i = 0; i < Gt_block_sizes.size(); i++) {
    s1 += Gt_block_sizes[i];
    ASSERT_LE(nda::max_element(nda::abs(result.get_block(i) - result_dense(_, range(s0, s1), range(s0, s1)))), 1e-10);
    s0 = s1;
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

  // create atom_diag object
  triqs::atom_diag::atom_diag<false> ad(H, fop_set);

  // get blocks of Hamiltonian and compute noninteracting Green's function
  auto [H_blocks, H_block_inds] = get_hamiltonian_blocks(ad);
  auto dlr_it                   = itops.get_itnodes();
  auto dlr_it_abs               = cppdlr::rel2abs(dlr_it);
  auto Gt                       = ad_to_nonint_gf(ad, beta, dlr_it_abs);
  auto Gt_block_sizes           = Gt.get_block_sizes();

  // generate creation/annihilation operators in block-sparse storage
  auto [Fq, sym_set_labels] = get_operators(ad, norb, hyb_coeffs, hyb_refl_coeffs);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, nn);
  DiagramEvaluator D(beta, itops, hyb, hyb_refl, dlr_rf, Gt, Fq);
  auto start = std::chrono::high_resolution_clock::now();
  D.eval_self_energy(B);
  auto end                               = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  auto result                            = D.Sigma;

  // get dense Gt, field operators
  auto H_mat    = get_full_h_atomic_perm(ad);
  auto Gt_dense = Hmat_to_Gtmat(H_mat, beta, dlr_it_abs);
  auto Fset     = get_operators_dense(ad, norb, hyb_coeffs, hyb_refl_coeffs);

  DenseDiagramEvaluator D2(beta, itops, hyb, hyb_refl, dlr_rf, Gt_dense, Fset);
  start = std::chrono::high_resolution_clock::now();
  D2.eval_self_energy(B);
  end               = std::chrono::high_resolution_clock::now();
  duration          = end - start;
  auto result_dense = D2.Sigma;

  int s0 = 0, s1 = 0;
  for (int i = 0; i < Gt_block_sizes.size(); i++) {
    s1 += Gt_block_sizes[i];
    ASSERT_LE(nda::max_element(nda::abs(result.get_block(i) - result_dense(_, range(s0, s1), range(s0, s1)))), 1e-10);
    s0 = s1;
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
  auto OCA_old      = nda::make_regular(-OCA_forward - OCA_backward);

  // generic diagram evaluator
  nda::array<int, 2> topology   = {{0, 2}, {1, 3}};
  auto B                        = Backbone(topology, n);
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs, hyb_refl_coeffs);
  auto D                        = DiagramEvaluator(beta, itops, hyb, hyb_refl, hyb_poles, Gt, Fq);
  D.eval_self_energy(B);
  auto OCA_result = D.Sigma; // get the result from the DiagramEvaluator

  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(0) - OCA_old(_, range(0, 4), range(0, 4)))), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(1) - OCA_old(_, range(4, 10), range(4, 10)))), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(2) - OCA_old(_, range(10, 11), range(10, 11)))), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(3) - OCA_old(_, range(11, 15), range(11, 15)))), eps);
  ASSERT_LE(nda::max_element(nda::abs(OCA_result.get_block(4) - OCA_old(_, range(15, 16), range(15, 16)))), eps);
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
  int r    = itops.rank();
  nda::array<dcomplex, 3> hyb(r, nn, nn), hyb_coeffs(p, nn, nn);
  nda::vector<dcomplex> hyb00(r), hyb_coeffs00(p);
  hyb00        = {-0.4997496184487105, -0.4867352379479528, -0.4603465101833711, -0.4239204950540695, -0.3716597467714097,
                  -0.2884886574148449, -0.2479810727230272, -0.2065525284769785, -0.1635819676241178, -0.1326995066858671,
                  -0.1225444804140666, -0.1282199855712255, -0.1386184647087601, -0.1720919948804938, -0.2300400167898313,
                  -0.3000508284935615, -0.3759657450111002, -0.4545389745912252, -0.4821599768174421, -0.4997496184487105};
  hyb_coeffs00 = {0.0028042961182163, 0.088487039172428,  0.1575418229076625, 0.1953880665937937,
                  0.2145207908265103, 0.1832496441339733, 0.1580088741667851};

  for (int i = 0; i < nn; i++) {
    for (int j = i; j < nn; j++) {
      if (i / 2 == j / 2) {
        hyb(_, i, j)        = hyb00;
        hyb_coeffs(_, i, j) = hyb_coeffs00;
      } else {
        hyb(_, i, j)        = 0.0;
        hyb_coeffs(_, i, j) = 0.0;
      }
    }
  }
  auto hyb_refl = hyb;
  nda::array<dcomplex, 3> hyb_refl_coeffs(p, nn, nn);
  hyb_refl_coeffs = hyb_coeffs;
  nda::vector<double> hyb_poles(p);
  hyb_poles = {-2.537191963500981,  1.7111725610238615, -1.514666605887425, 1.04941790134832,
               -0.7410379494142222, 0.3763525311836938, -0.1312888711963961};
  hyb_poles = hyb_poles * beta;

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
  auto dlr_it         = itops.get_itnodes();
  auto dlr_it_abs     = cppdlr::rel2abs(dlr_it);
  auto Gt             = ad_to_nonint_gf(ad, beta, dlr_it_abs);
  auto Gt_block_sizes = Gt.get_block_sizes();

  // generate creation/annihilation operators in block-sparse storage
  auto [Fq, sym_set_labels] = get_operators(ad, norb, hyb_coeffs, hyb_refl_coeffs);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, nn);
  DiagramEvaluator D(beta, itops, hyb, hyb_refl, hyb_poles, Gt, Fq);
  auto start = std::chrono::high_resolution_clock::now();
  D.eval_self_energy(B);
  auto end                               = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  auto result                            = D.Sigma;

  // get dense Gt, field operators
  auto H_mat    = get_full_h_atomic_perm(ad);
  auto Gt_dense = Hmat_to_Gtmat(H_mat, beta, dlr_it_abs);
  auto Fset     = get_operators_dense(ad, norb, hyb_coeffs, hyb_refl_coeffs);

  // compare to dense result
  DenseDiagramEvaluator D2(beta, itops, hyb, hyb_refl, hyb_poles, Gt_dense, Fset);
  start = std::chrono::high_resolution_clock::now();
  D2.eval_self_energy(B);
  end               = std::chrono::high_resolution_clock::now();
  duration          = end - start;
  auto result_dense = D2.Sigma;

  int s0 = 0, s1 = 0;
  for (int i = 0; i < Gt_block_sizes.size(); i++) {
    s1 += Gt_block_sizes[i];
    ASSERT_LE(nda::max_element(nda::abs(result.get_block(i) - result_dense(_, range(s0, s1), range(s0, s1)))), 1e-10);
    s0 = s1;
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
  int r    = itops.rank();
  nda::array<dcomplex, 3> hyb(r, nn, nn), hyb_coeffs(p, nn, nn);
  nda::vector<dcomplex> hyb00(r), hyb_coeffs00(p);
  hyb00        = {-0.4997496184487105, -0.4867352379479528, -0.4603465101833711, -0.4239204950540695, -0.3716597467714097,
                  -0.2884886574148449, -0.2479810727230272, -0.2065525284769785, -0.1635819676241178, -0.1326995066858671,
                  -0.1225444804140666, -0.1282199855712255, -0.1386184647087601, -0.1720919948804938, -0.2300400167898313,
                  -0.3000508284935615, -0.3759657450111002, -0.4545389745912252, -0.4821599768174421, -0.4997496184487105};
  hyb_coeffs00 = {0.0028042961182163, 0.088487039172428,  0.1575418229076625, 0.1953880665937937,
                  0.2145207908265103, 0.1832496441339733, 0.1580088741667851};

  for (int i = 0; i < nn; i++) {
    for (int j = i; j < nn; j++) {
      if ((i == j) || (j - i) % (nn / 2) == 0) {
        hyb(_, i, j)        = hyb00;
        hyb_coeffs(_, i, j) = hyb_coeffs00;
      } else {
        hyb(_, i, j)        = 0.0;
        hyb_coeffs(_, i, j) = 0.0;
      }
    }
  }
  auto hyb_refl = hyb;
  nda::array<dcomplex, 3> hyb_refl_coeffs(p, nn, nn);
  hyb_refl_coeffs = hyb_coeffs;
  nda::vector<double> hyb_poles(p);
  hyb_poles = {-2.537191963500981,  1.7111725610238615, -1.514666605887425, 1.04941790134832,
               -0.7410379494142222, 0.3763525311836938, -0.1312888711963961};
  hyb_poles = hyb_poles * beta;

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

  // create atom_diag object
  triqs::atom_diag::atom_diag<false> ad(H, fop_set);

  // get blocks of Hamiltonian and compute noninteracting Green's function
  auto [H_blocks, H_block_inds] = get_hamiltonian_blocks(ad);
  auto dlr_it                   = itops.get_itnodes();
  auto dlr_it_abs               = cppdlr::rel2abs(dlr_it);
  auto Gt                       = ad_to_nonint_gf(ad, beta, dlr_it_abs);
  auto Gt_block_sizes           = Gt.get_block_sizes();

  // generate creation/annihilation operators in block-sparse storage
  auto [Fq, sym_set_labels] = get_operators(ad, norb, hyb_coeffs, hyb_refl_coeffs);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = Backbone(topology, nn);
  DiagramEvaluator D(beta, itops, hyb, hyb_refl, hyb_poles, Gt, Fq);
  auto start = std::chrono::high_resolution_clock::now();
  D.eval_self_energy(B);
  auto end                               = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  auto result                            = D.Sigma;

  // get dense Gt, field operators
  auto H_mat    = get_full_h_atomic_perm(ad);
  auto Gt_dense = Hmat_to_Gtmat(H_mat, beta, dlr_it_abs);
  auto Fset     = get_operators_dense(ad, norb, hyb_coeffs, hyb_refl_coeffs);

  // compare to dense result
  DenseDiagramEvaluator D2(beta, itops, hyb, hyb_refl, hyb_poles, Gt_dense, Fset);
  start = std::chrono::high_resolution_clock::now();
  D2.eval_self_energy(B);
  end               = std::chrono::high_resolution_clock::now();
  duration          = end - start;
  auto result_dense = D2.Sigma;

  int s0 = 0, s1 = 0;
  for (int i = 0; i < Gt_block_sizes.size(); i++) {
    s1 += Gt_block_sizes[i];
    ASSERT_LE(nda::max_element(nda::abs(result.get_block(i) - result_dense(_, range(s0, s1), range(s0, s1)))), 1e-10);
    s0 = s1;
  }
}

TEST(Backbone, solve) {
  // Test the solve function
  double beta   = 8.0;
  double Lambda = 10.0 * beta;
  double eps    = 1.0e-6;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  int norb = 2;
  int nn   = 2 * norb;
  int p    = 7;
  int r    = itops.rank();
  nda::array<dcomplex, 3> hyb(r, nn, nn), hyb_coeffs(p, nn, nn);
  nda::vector<dcomplex> hyb00(r), hyb_coeffs00(p);
  hyb00        = {-0.4997496184487105, -0.4867352379479528, -0.4603465101833711, -0.4239204950540695, -0.3716597467714097,
                  -0.2884886574148449, -0.2479810727230272, -0.2065525284769785, -0.1635819676241178, -0.1326995066858671,
                  -0.1225444804140666, -0.1282199855712255, -0.1386184647087601, -0.1720919948804938, -0.2300400167898313,
                  -0.3000508284935615, -0.3759657450111002, -0.4545389745912252, -0.4821599768174421, -0.4997496184487105};
  hyb_coeffs00 = {0.0028042961182163, 0.088487039172428,  0.1575418229076625, 0.1953880665937937,
                  0.2145207908265103, 0.1832496441339733, 0.1580088741667851};

  for (int i = 0; i < nn; i++) {
    for (int j = i; j < nn; j++) {
      if (i / 2 == j / 2) {
        hyb(_, i, j)        = hyb00;
        hyb_coeffs(_, i, j) = hyb_coeffs00;
      } else {
        hyb(_, i, j)        = 0.0;
        hyb_coeffs(_, i, j) = 0.0;
      }
    }
  }
  auto hyb_refl = hyb;
  nda::array<dcomplex, 3> hyb_refl_coeffs(p, nn, nn);
  hyb_refl_coeffs = hyb_coeffs;
  nda::vector<double> hyb_poles(p);
  hyb_poles = {-2.537191963500981,  1.7111725610238615, -1.514666605887425, 1.04941790134832,
               -0.7410379494142222, 0.3763525311836938, -0.1312888711963961};
  hyb_poles = hyb_poles * beta;

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

  auto Sigma = compute_self_energy(beta, Lambda, eps, hyb, hyb_poles, hyb_coeffs, ad, 2);
}