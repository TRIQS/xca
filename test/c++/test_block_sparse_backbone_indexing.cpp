#include <gtest/gtest.h>

#include <triqs_xca/block_sparse_backbone.hpp>

#include "block_sparse_utils.hpp"

using cppdlr::build_dlr_rf;

/**
 * @file test_block_sparse_backbone_indexing.cpp
 */

/**
 * @brief Test the index bookkeeping of the Backbone classes
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
