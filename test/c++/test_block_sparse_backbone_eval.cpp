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
 *
 * @details For a fixed third-order topology and a fixed pair of pole indices,
 * this loops over all 2^m line-direction indices and checks, for each of them,
 * that (i) the generated vertices, edges and prefactor agree with values derived
 * by hand from the diagram rules; (ii) all three ways of specifying a backbone
 * (index vectors, per-component integer indices, and a single flat index) give
 * that same diagram; and (iii) resetting a Backbone and setting it again
 * reproduces a freshly constructed one, which is how the diagram loop in the
 * evaluators reuses a single object.
 *
 * The pole indices are chosen so that omega_l < 0 and omega_l` > 0. As the
 * directions vary, each hybridization line then takes both branches of
 * set_pole_inds: the branch that puts the K's on the two vertices of the line,
 * and the branch that puts a K on every edge the line spans.
 */
TEST(Backbone, indexing) {
  nda::array<int, 2> topology = {{0, 2}, {1, 4}, {3, 5}}; // a third-order diagram topology
  int n                       = 4;
  int m                       = topology.extent(0);

  // DLR generation
  double beta   = 2.0;
  double Lambda = 100.0 * beta;
  double eps    = 1.0e-10;
  auto dlr_rf   = build_dlr_rf(Lambda, eps);
  int r         = dlr_rf.size(); // r = 36

  nda::vector<int> pole_inds{3, 19};           // values of l, l`
  nda::vector<int> orb_inds{1, 3, 1, 2, 3, 2}; // orbital index on each of the 2m vertices
  ASSERT_LT(dlr_rf(pole_inds(0)), 0.0);        // omega_l  < 0
  ASSERT_GT(dlr_rf(pole_inds(1)), 0.0);        // omega_l` > 0

  int p_ix = pole_inds(0) + r * pole_inds(1);
  int o_ix = orb_inds(topology(1, 0)) + n * orb_inds(topology(2, 0)); // both vertices of a line share an orbital index

  for (int fb_ix = 0; fb_ix < (1 << m); ++fb_ix) {
    SCOPED_TRACE("fb_ix = " + std::to_string(fb_ix));

    nda::vector<int> fb(m);
    for (int i = 0, x = fb_ix; i < m; ++i, x /= 2) fb(i) = x % 2; // 0 for backward, 1 for forward

    // --- Expected vertices ---
    // Bars are fixed by the topology alone: the left vertex of every line other
    // than the one attached to vertex 0 carries a bar.
    std::vector<bool> exp_bar = {false, false, false, false, true, true};

    // Daggers follow the line directions: on a forward line the right (lower-numbered)
    // vertex carries the annihilation operator and the left vertex the creation operator.
    std::vector<bool> exp_dag(static_cast<size_t>(2 * m), false);
    for (int i = 0; i < m; ++i) {
      exp_dag[topology(i, 0)] = (fb(i) == 0);
      exp_dag[topology(i, 1)] = (fb(i) == 1);
    }

    // A line puts its K's on its two vertices when it is forward with a non-positive
    // pole or backward with a non-negative pole, and on the edges it spans otherwise.
    // Line 1 spans vertices 1--4 and carries l (omega_l < 0); line 2 spans vertices
    // 3--5 and carries l` (omega_l` > 0).
    bool line1_on_vertices = (fb(1) == 1);
    bool line2_on_vertices = (fb(2) == 0);

    // The left vertex of a line always records which pole that line carries. The K's,
    // when they sit on the vertices, are K^-/K^+ on the right/left vertex of a forward
    // line and K^+/K^- on the right/left vertex of a backward line.
    nda::vector<int> exp_hyb  = nda::zeros<int>(2 * m);
    nda::vector<int> exp_Ksgn = nda::zeros<int>(2 * m);
    exp_hyb(topology(1, 1))   = 0; // line 1 -> l
    exp_hyb(topology(2, 1))   = 1; // line 2 -> l`
    if (line1_on_vertices) {
      exp_hyb(topology(1, 0))  = 0;
      exp_Ksgn(topology(1, 0)) = -1;
      exp_Ksgn(topology(1, 1)) = 1;
    }
    if (line2_on_vertices) {
      exp_hyb(topology(2, 0))  = 1;
      exp_Ksgn(topology(2, 0)) = 1;
      exp_Ksgn(topology(2, 1)) = -1;
    }

    // Vertex 0 and the vertex connected to it carry no explicitly summed orbital index.
    nda::vector<int> exp_orb    = orb_inds;
    exp_orb(0)                  = 0;
    exp_orb(topology(0, 1))     = 0;

    // --- Expected edges ---
    // A line whose K's are not on its vertices puts one K on each edge it spans,
    // with the sign of the pole it carries.
    nda::array<int, 2> exp_edges = nda::zeros<int>(2 * m - 1, m - 1);
    if (not line1_on_vertices) {
      for (int e = topology(1, 0); e < topology(1, 1); ++e) exp_edges(e, 0) = -1; // K^-_l  on edges 1, 2, 3
    }
    if (not line2_on_vertices) {
      for (int e = topology(2, 0); e < topology(2, 1); ++e) exp_edges(e, 1) = 1; // K^+_l` on edges 3, 4
    }

    // --- Expected prefactor ---
    // The K(0) in the prefactor has the sign of the pole its line carries. It appears
    // once for a line whose K's are on its vertices, and once per spanned edge beyond
    // the first otherwise. Each backward line flips the overall sign.
    nda::vector<int> exp_pKsign{-1, 1};
    nda::vector<int> exp_pKexp(m - 1);
    exp_pKexp(0)  = line1_on_vertices ? 1 : topology(1, 1) - topology(1, 0) - 1;
    exp_pKexp(1)  = line2_on_vertices ? 1 : topology(2, 1) - topology(2, 0) - 1;
    int exp_psign = ((fb(1) == 0) ? -1 : 1) * ((fb(2) == 0) ? -1 : 1);

    auto expect_generated = [&](Backbone &X) {
      for (int i = 0; i < m; ++i) { ASSERT_EQ(X.get_fb(i), fb(i)); }
      for (int v = 0; v < 2 * m; ++v) {
        ASSERT_EQ(X.has_vertex_bar(v), exp_bar[v]);
        ASSERT_EQ(X.has_vertex_dag(v), exp_dag[v]);
        ASSERT_EQ(X.get_vertex_hyb_ind(v), exp_hyb(v));
        ASSERT_EQ(X.get_vertex_Ksign(v), exp_Ksgn(v));
        ASSERT_EQ(X.get_vertex_orb(v), exp_orb(v));
      }
      for (int e = 0; e < 2 * m - 1; ++e) {
        for (int p = 0; p < m - 1; ++p) { ASSERT_EQ(X.get_edge(e, p), exp_edges(e, p)); }
      }
      for (int i = 0; i < m - 1; ++i) {
        ASSERT_EQ(X.get_pole_ind(i), pole_inds(i));
        ASSERT_EQ(X.get_prefactor_Ksign(i), exp_pKsign(i));
        ASSERT_EQ(X.get_prefactor_Kexp(i), exp_pKexp(i));
      }
      ASSERT_EQ(X.prefactor_sign, exp_psign);
    };

    // --- Check the diagram generated from index vectors ---
    auto B = Backbone(topology, n);
    B.set_directions(fb);
    B.set_pole_inds(pole_inds, dlr_rf);
    B.set_orb_inds(orb_inds);
    expect_generated(B);

    // --- Check that a reset Backbone can be set again, as the diagram loop does ---
    B.reset_all_inds();
    B.set_directions(fb);
    B.set_pole_inds(pole_inds, dlr_rf);
    B.set_orb_inds(orb_inds);
    expect_generated(B);

    // --- Check that setting the indices component-wise gives the same diagram ---
    auto B2 = Backbone(topology, n);
    B2.set_directions(fb_ix);
    B2.set_pole_inds(p_ix, dlr_rf);
    B2.set_orb_inds(o_ix);
    expect_generated(B2);

    // --- Check that setting the indices from a single flat index gives the same diagram ---
    int f_ix = o_ix + p_ix * n * n + fb_ix * n * n * r * r;
    auto B3  = Backbone(topology, n);
    B3.set_flat_index(f_ix, dlr_rf);
    expect_generated(B3);
    ASSERT_EQ(B3.get_flat_index(), f_ix);
  }
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

/**
 * @brief Compare block-sparse and dense OCA self-energies for the spin-flip fermion model
 *
 * @param[in] use_particle_number_sym if true, the atom_diag subspaces are labeled by the particle
 * number N, so that all field operators share a single symmetry set; if false, the subspaces come
 * from autopartitioning alone and the field operators are spread over several symmetry sets.
 */
static void check_spin_flip_fermion(bool use_particle_number_sym) {
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

  // create atom_diag object, either from the particle number as a quantum number or by autopartitioning
  auto ad = [&]() {
    if (use_particle_number_sym) {
      triqs::operators::many_body_operator_complex N;
      for (int kap = 0; kap < norb; ++kap) { N += n("up", kap) + n("do", kap); }
      std::vector<triqs::operators::many_body_operator_complex> sym_ops = {N};
      return triqs::atom_diag::atom_diag<true>(H, fop_set, sym_ops);
    }
    return triqs::atom_diag::atom_diag<true>(H, fop_set);
  }();

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

TEST(Backbone, spin_flip_fermion) { check_spin_flip_fermion(true); }

TEST(Backbone, spin_flip_fermion_sym_sets) { check_spin_flip_fermion(false); }

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
