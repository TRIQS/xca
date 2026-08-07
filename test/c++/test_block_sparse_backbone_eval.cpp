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

using triqs_xca::block_sparse::BlockOp;
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
 * @details For a fixed third-order topology and a fixed pair of pole indices, this loops over all 2^m line-direction indices and checks, for each of
 * them, that 
 * 1. the generated vertices, edges and prefactor agree with values derived by hand from the diagram rules;
 * 2. all three ways of specifying a backbone (index vectors, per-component integer indices, and a single flat index) give that same diagram; and 
 * 3. resetting a Backbone and setting it again reproduces a freshly constructed one, which is how the diagram loop in the evaluators reuses a single
 *    object.
 *
 * The pole indices are chosen so that omega_l < 0 and omega_l` > 0. As the directions vary, each hybridization line then takes both branches of
 * set_pole_inds: the branch that puts the K's on the two vertices of the line, and the branch that puts a K on every edge the line spans.
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
    nda::vector<int> exp_orb = orb_inds;
    exp_orb(0)               = 0;
    exp_orb(topology(0, 1))  = 0;

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

/**
 * @brief Check that the block-sparse OCA self-energy is unchanged when the block structure is discarded
 *
 * @details The block-sparse evaluator is run twice on the same model: once with the block structure of the atom_diag subspaces, and once with a
 * trivial sparsity pattern, i.e. a single block spanning the whole Hilbert space and a single symmetry set holding all four flavors. The two must
 * agree, and both must agree with the dense evaluator. The model is the two-band Kanamori atom and a two-pole discrete-bath hybridization.
 */
TEST(Backbone, OCA_trivial_sparsity) {
  // DLR generation
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-6;
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_refl] = discrete_bath_helper(beta, Lambda, eps);
  auto hyb_coeffs            = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  // backbone
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  int n                       = 4; // number of flavors
  auto B                      = Backbone(topology, n);

  // --- block-sparse evaluation, using the block structure of the atom_diag subspaces ---
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto OCA_result = BlockDiagOpFun(D.compute_self_energy(Gt, topology));

  // --- dense evaluation ---
  // The old dense constructor divides the poles it is given by beta internally, so it takes dlr_rf where the block-sparse one takes dlr_rf / beta.
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);
  auto Fset                               = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs);
  DenseDiagramEvaluator DDE(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  DDE.eval_self_energy(Gt_dense, B);
  auto OCA_dense_result = DDE.Sigma;

  // --- block-sparse evaluation with trivial sparsity: one block, one symmetry set ---
  nda::vector<int> triv_bi{0};
  std::vector<nda::array<dcomplex, 3>> Gt_dense_vec{Gt_dense};
  BlockDiagOpFun Gt_triv(Gt_dense_vec, triv_bi);

  // make dense operators compatible with trivial block-sparse evaluation
  std::vector<nda::array<dcomplex, 3>> Fs_dense_vec{Fs_dense};
  auto F_sym_triv = BlockOpSymSet(triv_bi, Fs_dense_vec);
  std::vector<nda::array<dcomplex, 3>> F_dags_dense_vec{F_dags_dense};
  auto F_dag_sym_triv      = BlockOpSymSet(triv_bi, F_dags_dense_vec);
  auto sym_set_labels_triv = nda::zeros<long>(n);
  auto Fq_triv             = BlockOpSymQuartet({F_sym_triv}, {F_dag_sym_triv}, hyb_coeffs, sym_set_labels_triv);

  DiagramEvaluator D_triv(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq_triv);
  auto OCA_trivial_bs = BlockDiagOpFun(D_triv.compute_self_energy(Gt_triv, topology));

  // Both references live on the full Hilbert space, so they are projected onto each atom_diag subspace before being compared block by block.
  auto ad = two_band_atom_diag_helper();
  for (int i = 0; i < OCA_result.get_num_block_cols(); i++) {
    SCOPED_TRACE("block " + std::to_string(i));
    auto result_dense_block = get_tensor_in_atom_diag_subspace(OCA_dense_result, i, ad);
    EXPECT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - result_dense_block)), 10 * eps);

    auto result_trivial_bs_block = get_tensor_in_atom_diag_subspace(OCA_trivial_bs.get_block(0), i, ad);
    EXPECT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - result_trivial_bs_block)), 10 * eps);
  }
}

/**
 * @brief Check that the two DiagramEvaluator constructors describe the same problem
 *
 * @details The evaluator can be built either from an atom_diag object, which derives the field operators and the block structure itself and takes the
 * propagator as a block_gf, or from a BlockOpSymQuartet assembled by the caller, which takes the propagator as a BlockDiagOpFun. The second form is
 * C2PY_IGNORE and predates the first, but is still used internally, so the two must agree on the same model.
 *
 * This is a consistency check between two code paths rather than a check of the diagram: the two build the same Hybridization from the same
 * coefficients and sum the same backbones, so they should agree to round-off, hence the 1e-10 tolerance rather than one set by eps. The correctness 
 * of the atom_diag path is established independently, against closed forms, in test_one/two_fermion_se_spgf_all_evals.cpp.
 *
 * The model is the same Kanamori atom and two-pole discrete-bath hybridization as OCA_trivial_sparsity above. Both take their atom_diag from
 * two_band_atom_diag_helper, so the subspaces, and hence the block ordering, are identical.
 */
TEST(Backbone, OCA_constructor_equivalence) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_reflect] = discrete_bath_helper(beta, Lambda, eps);
  auto hyb_coeffs               = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};

  // --- atom_diag constructor, taking the atomic propagator as a block_gf ---
  auto ad      = two_band_atom_diag_helper();
  auto G0_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);
  DiagramEvaluator D(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  BlockDiagOpFun OCA_result(D.compute_self_energy(G0_ppsc, topology));

  // --- old constructor, taking field operators as a BlockOpSymQuartet and the propagator as a BlockDiagOpFun ---
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);
  DiagramEvaluator DDE(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  BlockDiagOpFun OCA_result_2(DDE.compute_self_energy(Gt, topology));

  // the two constructors must assemble the same hybridization ...
  ASSERT_LE(nda::max_element(nda::abs(D.hyb.values - DDE.hyb.values)), eps);
  ASSERT_EQ(D.hyb.poles, DDE.hyb.poles);

  // ... and hence the same self-energy, block by block
  ASSERT_EQ(OCA_result.get_num_block_cols(), OCA_result_2.get_num_block_cols());
  for (int i = 0; i < OCA_result.get_num_block_cols(); i++) {
    SCOPED_TRACE("block " + std::to_string(i));
    EXPECT_LE(nda::max_element(nda::abs(OCA_result.get_block(i) - OCA_result_2.get_block(i))), 1e-10);
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

  DenseDiagramEvaluator DDE(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  start = std::chrono::high_resolution_clock::now();
  DDE.eval_self_energy_by_pairs(Gt_dense, B);
  end               = std::chrono::high_resolution_clock::now();
  duration          = end - start;
  auto result_dense = DDE.Sigma;

  for (int i = 0; i < Gt_block_sizes.size(); i++) {
    auto result_dense_block = get_tensor_in_atom_diag_subspace(result_dense, i, ad);
    ASSERT_LE(nda::max_element(nda::abs(result.get_block(i) - result_dense_block)), eps);
  }
}

TEST(Backbone, spin_flip_fermion) { check_spin_flip_fermion(true); } // self-energies match when using just total particle number symmetry

TEST(Backbone, spin_flip_fermion_sym_sets) { check_spin_flip_fermion(false); } // self-energies match when using all available symmetries

/**
 * @brief Check that summing the backbones one at a time reproduces the built-in diagram loop
 *
 * @details compute_self_energy has an overload taking a single flat backbone index, which is how a caller drives the diagram loop itself, e.g. to
 * distribute it. Summing that overload over all get_num_self_energy_backbones indices must reproduce the overload that loops internally, so their
 * difference is checked to vanish. Both sides come from the same evaluator, so this is self-consistency of the loop bookkeeping and is independent of
 * the model; it uses the same Kanamori atom and two-pole discrete-bath hybridization as the two tests above.
 */
TEST(Backbone, manual_loop) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_reflect] = discrete_bath_helper(beta, Lambda, eps);
  auto hyb_coeffs               = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  // Kanamori atom and its atomic propagator as a block_gf
  auto ad      = two_band_atom_diag_helper();
  auto G0_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  DiagramEvaluator D(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto OCA_result = D.compute_self_energy(G0_ppsc, topology); // evaluate self-energy using built-in loop

  for (int f = 0; f < D.get_num_self_energy_backbones(topology); ++f) {
    OCA_result -= D.compute_self_energy(G0_ppsc, topology, f); // subtract off self-energy evaluation using manual loop
  }
  for (int i = 0; i < ad.n_subspaces(); ++i) {
    SCOPED_TRACE("block " + std::to_string(i));
    EXPECT_LE(nda::max_element(nda::abs(OCA_result[i].data())), 1e-10);
  }
}

// =====================================================================================================
// Correlator (e.g., single-particle Green's function) diagrams
// =====================================================================================================

/**
 * @brief Split a BlockOpSymQuartet into the per-flavor BlockOp lists that eval_correlator takes
 *
 * @details eval_correlator predates the symmetry-set storage and still wants one BlockOp per flavor, so each flavor has to be picked out of whichever
 * symmetry set holds it: Fq.sym_set_labels(oidx) says which set, and Fq.sym_set_inds(oidx) says where within that set. Block-columns in which the
 * operator has no block are filled with a 1x1 zero, which is the placeholder the block-sparse routines expect.
 */
static std::pair<std::vector<BlockOp>, std::vector<BlockOp>> make_correlator_ops(BlockOpSymQuartet &Fq, int nflav) {
  std::vector<BlockOp> mu_ops, kap_ops;
  for (int oidx = 0; oidx < nflav; ++oidx) {
    auto &F     = Fq.Fs[Fq.sym_set_labels(oidx)];
    auto &F_dag = Fq.F_dags[Fq.sym_set_labels(oidx)];
    int i       = Fq.sym_set_inds(oidx);
    std::vector<nda::array<dcomplex, 2>> mu_blocks, kap_blocks;
    for (int j = 0; j < F.get_num_block_cols(); ++j) {
      if (F.get_block_index(j) != -1) {
        mu_blocks.emplace_back(F.get_block(j)(i, _, _));
      } else {
        mu_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      }
      if (F_dag.get_block_index(j) != -1) {
        kap_blocks.emplace_back(F_dag.get_block(j)(i, _, _));
      } else {
        kap_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      }
    }
    nda::vector<int> mu_block_indices = F.get_block_indices()(_);
    mu_ops.emplace_back(mu_block_indices, mu_blocks);
    nda::vector<int> kap_block_indices = F_dag.get_block_indices()(_);
    kap_ops.emplace_back(kap_block_indices, kap_blocks);
  }
  return {mu_ops, kap_ops};
}

/**
 * @brief Check that the block-sparse OCA correlator is unchanged when the block structure is discarded
 *
 * @details The correlator analogue of OCA_trivial_sparsity above: eval_correlator is run with the block structure of the atom_diag subspaces, with a
 * trivial sparsity pattern (a single block spanning the whole Hilbert space, and a single symmetry set holding all four flavors), and with the dense
 * evaluator, on the two-band Kanamori atom with the two-pole discrete-bath hybridization. All three must agree.
 *
 * Unlike the self-energy, the correlator is already a dense object in orbital space, so the three results are compared directly rather than block by
 * block. The block-sparse-vs-dense comparison on its own is also made by spin_flip_fermion_correlator below; what is new here is the trivial-sparsity
 * leg.
 */
TEST(Backbone, OCA_correlator_trivial_sparsity) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;
  int n         = 4; // number of flavors

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_refl] = discrete_bath_helper(beta, Lambda, eps);
  auto hyb_coeffs            = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  // backbone
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = CorrelatorBackbone(topology, n);

  // --- block-sparse evaluation, using the block structure of the atom_diag subspaces ---
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto [mu_ops, kap_ops] = make_correlator_ops(Fq, n);
  auto OCA_result        = D.eval_correlator(Gt, B, mu_ops, kap_ops);

  // --- dense evaluation ---
  // The old dense constructor divides the poles it is given by beta internally, so it takes dlr_rf where the block-sparse one takes dlr_rf / beta.
  auto [Gt_dense, Fs_dense, F_dags_dense] = two_band_dense_helper(beta, Lambda, eps);
  auto Fset                               = DenseFSet(Fs_dense, F_dags_dense, hyb_coeffs);
  DenseDiagramEvaluator D_dense(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  auto OCA_dense_result = D_dense.eval_correlator(Gt_dense, B, Fs_dense, F_dags_dense);

  // --- block-sparse evaluation with trivial sparsity: one block, one symmetry set ---
  nda::vector<int> triv_bi{0};
  std::vector<nda::array<dcomplex, 3>> Gt_dense_vec{Gt_dense};
  BlockDiagOpFun Gt_triv(Gt_dense_vec, triv_bi);

  // make dense operators compatible with trivial block-sparse evaluation
  std::vector<nda::array<dcomplex, 3>> Fs_dense_vec{Fs_dense};
  auto F_sym_triv = BlockOpSymSet(triv_bi, Fs_dense_vec);
  std::vector<nda::array<dcomplex, 3>> F_dags_dense_vec{F_dags_dense};
  auto F_dag_sym_triv      = BlockOpSymSet(triv_bi, F_dags_dense_vec);
  auto sym_set_labels_triv = nda::zeros<long>(n);
  auto Fq_triv             = BlockOpSymQuartet({F_sym_triv}, {F_dag_sym_triv}, hyb_coeffs, sym_set_labels_triv);

  DiagramEvaluator D_triv(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq_triv);
  auto [mu_ops_triv, kap_ops_triv] = make_correlator_ops(Fq_triv, n);
  auto OCA_trivial_bs              = D_triv.eval_correlator(Gt_triv, B, mu_ops_triv, kap_ops_triv);

  EXPECT_LE(nda::max_element(nda::abs(OCA_result - OCA_dense_result)), 1.0e-15);
  EXPECT_LE(nda::max_element(nda::abs(OCA_result - OCA_trivial_bs)), 1.0e-15);
}

/**
 * @brief Compare block-sparse and dense OCA correlators for the spin-flip fermion model
 *
 * @param[in] use_particle_number_sym if true, the atom_diag subspaces are labeled by the particle
 * number N, so that all field operators share a single symmetry set; if false, the subspaces come
 * from autopartitioning alone and the field operators are spread over several symmetry sets.
 */
static void check_spin_flip_fermion_correlator(bool use_particle_number_sym) {
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

  // compute atomic propagator
  auto dlr_it_abs = cppdlr::rel2abs(itops.get_itnodes());
  auto Gt         = ad_to_atom_prop(ad, beta, itops);

  // generate creation/annihilation operators in block-sparse storage
  auto [Fq, sym_set_labels] = get_operators(ad, hyb_coeffs);
  auto [mu_ops, kap_ops]    = make_correlator_ops(Fq, nn);

  // set up backbone and diagram evaluator
  nda::array<int, 2> topology = {{0, 2}, {1, 3}};
  auto B                      = CorrelatorBackbone(topology, nn);
  DiagramEvaluator D(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto result = D.eval_correlator(Gt, B, mu_ops, kap_ops);

  // compare to dense backbone result
  auto Gt_dense = Hmat_to_Gtmat(get_full_h_atomic(ad), beta, dlr_it_abs);
  auto Fset     = get_operators_dense(ad, hyb_coeffs);
  DenseDiagramEvaluator D_dense(beta, eps, itops, dlr_rf, hyb_coeffs, Fset);
  auto result_dense = D_dense.eval_correlator(Gt_dense, B, Fset.Fs, Fset.F_dags);

  ASSERT_LE(nda::max_element(nda::abs(result - result_dense)), 1.0e-15);
}

TEST(Backbone, spin_flip_fermion_correlator) { check_spin_flip_fermion_correlator(true); } // spgfs match when using total particle number symmetry

TEST(Backbone, spin_flip_fermion_correlator_sym_sets) { check_spin_flip_fermion_correlator(false); } // spgfs match when using available symmetries

/**
 * @brief Check that every route to the OCA single-particle Green's function agrees
 *
 * @details The single-particle analogue of OCA_constructor_equivalence and manual_loop combined, on the same Kanamori atom and two-pole discrete-bath
 * hybridization. Four evaluations are compared:
 *
 * 1. the block-sparse evaluator built from an atom_diag, taking the propagator as a block_gf;
 * 2. the same, built from a caller-assembled BlockOpSymQuartet and taking the propagator as a BlockDiagOpFun;
 * 3. the flat-index overload, summed by the caller over all get_num_single_ptcle_gf_backbones backbones; and
 * 4. the dense evaluator, which sums the same backbones over the full Hilbert space.
 *
 * The first three differ only in bookkeeping and so should agree to round-off; the fourth is an independent implementation. All four use the same
 * atom_diag, so the block structure is identical across them.
 */
TEST(Backbone, OCA_gf_evaluator_equivalence) {
  double beta   = 2.0;
  double Lambda = 20.0 * beta;
  double eps    = 1.0e-4;

  // DLR generation
  auto dlr_rf = build_dlr_rf(Lambda, eps);
  auto itops  = imtime_ops(Lambda, dlr_rf);

  // hybridization
  auto [Deltat, Deltat_reflect] = discrete_bath_helper(beta, Lambda, eps);
  auto hyb_coeffs               = itops.vals2coefs(Deltat); // hybridization DLR coeffs

  // Kanamori atom and its atomic propagator as a block_gf
  auto ad      = two_band_atom_diag_helper();
  auto G0_ppsc = ad_to_atom_prop(ad, beta, Lambda, eps);

  nda::array<int, 2> topology = {{0, 2}, {1, 3}};

  // --- atom_diag constructor, taking the atomic propagator as a block_gf ---
  DiagramEvaluator D(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto OCA_gf = D.compute_single_ptcle_gf(G0_ppsc, topology);

  // --- old constructor, taking field operators as a BlockOpSymQuartet and the propagator as a BlockDiagOpFun ---
  auto [Gt, Fq, sym_set_labels] = two_band_helper(beta, Lambda, eps, hyb_coeffs);
  DiagramEvaluator D2(beta, Lambda, eps, nda::make_regular(dlr_rf / beta), hyb_coeffs, Fq);
  auto OCA_gf_2 = D2.compute_single_ptcle_gf(Gt, topology);

  // --- driving the backbone loop from the caller, one flat index at a time, on a fresh evaluator ---
  DiagramEvaluator D3(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto OCA_gf_3 = nda::make_regular(0 * OCA_gf);
  for (int f = 0; f < D.get_num_single_ptcle_gf_backbones(topology); ++f) { OCA_gf_3 += D3.compute_single_ptcle_gf(G0_ppsc, topology, f); }

  // the two constructors must assemble the same hybridization ...
  ASSERT_LE(nda::max_element(nda::abs(D.hyb.values - D2.hyb.values)), eps);
  ASSERT_EQ(D.hyb.poles, D2.hyb.poles);

  // ... and all three block-sparse routes must give the same Green's function
  EXPECT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_2)), 1.0e-10);
  EXPECT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_3)), 1.0e-10);

  // --- dense evaluation, over the full Hilbert space as a single block ---
  auto G0t_dense = Hmat_to_Gtmat(get_full_h_atomic(ad), beta, cppdlr::rel2abs(itops.get_itnodes()));
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> G0_dense_blocks{triqs::gfs::gf<triqs::mesh::dlr_imtime>(G0_ppsc[0].mesh(), G0t_dense)};
  auto G0_ppsc_dense = triqs::gfs::block_gf<triqs::mesh::dlr_imtime>(G0_dense_blocks);
  DenseDiagramEvaluator D_dense(nda::make_regular(dlr_rf / beta), hyb_coeffs, G0_ppsc[0].mesh(), ad);
  auto OCA_gf_dense = D_dense.compute_single_ptcle_gf(G0_ppsc_dense, topology);
  EXPECT_LE(nda::max_element(nda::abs(OCA_gf - OCA_gf_dense)), 1.0e-10);
}
