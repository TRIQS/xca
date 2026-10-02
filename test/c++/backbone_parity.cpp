#include <stdexcept>

#include <gtest/gtest.h>

#include <triqs_xca/backbone.hpp>
#include <triqs_xca/topology.hpp>

/**
 * @file backbone_parity.cpp
 *
 * @brief Tests of Backbone::get_parity(), the only place where the statistics of a line enters the diagram
 *
 * @details A vertex is fermionic iff its orbital index is smaller than n_hyb = n - n_int, with -2 and -3 as the external-operator sentinels of
 * the correlator path. The pairs containing a bosonic vertex are dropped by topology::fermionic_topology() and the permutation parity of the
 * rest is returned. Since n_int defaults to 0, a two-argument construction calls every vertex fermionic. The third-order crossing topology is
 * the cheapest one on which it matters which line is bosonic, the order-2 crossing topology has only single-pair sub-matchings.
 */

namespace {

  // Third order, crossing. Line 0 = (0,2), line 1 = (1,4), line 2 = (3,5).
  const nda::array<int, 2> crossing3 = {{0, 2}, {1, 4}, {3, 5}};

  constexpr int n_hyb = 2;             // fermionic flavours 0 and 1
  constexpr int n_int = 1;             // one interaction flavour ...
  constexpr int o_int = n_hyb;         // ... whose orbital index is 2
  constexpr int n_ext = n_hyb + n_int; // 3

  /**
   * @brief Parity of the diagram in which the two vertices of line 0 carry the orbital indices orb_kap and orb_mu, and lines 1 and 2 carry
   * orb1 and orb2
   */
  int parity_of(int n_int_arg, int orb_kap, int orb_mu, int orb1, int orb2) {
    Backbone B(crossing3, n_ext, n_int_arg);
    nda::vector<int> fb = {1, 0, 1};
    B.set_directions(fb);
    nda::vector<int> orb = nda::zeros<int>(6);
    orb(crossing3(1, 0)) = orb1;
    orb(crossing3(1, 1)) = orb1;
    orb(crossing3(2, 0)) = orb2;
    orb(crossing3(2, 1)) = orb2;
    B.set_orb_inds(orb);
    B.set_orb_inds_of_0_and_vct0(orb_kap, orb_mu);
    return B.get_parity();
  }

} // namespace

/**
 * @brief Check the parity table of the third-order crossing topology, derived by hand
 *
 * @details F = fermionic flavour, B = the interaction flavour o_int, "kept" lists the topology rows that survive fermionic_topology(), and
 * "compacted" the same rows with the surviving vertex labels renumbered.
 *
 *   line0  line1  line2 | kept    | compacted          | parity
 *   ------------------- | ------- | ------------------ | ------
 *   F F    F      F     | 0, 1, 2 | {{0,2},{1,4},{3,5}}|  +1
 *   F F    B      F     | 0, 2    | {{0,1},{2,3}}      |  +1
 *   F F    F      B     | 0, 1    | {{0,2},{1,3}}      |  -1
 *   B B    F      F     | 1, 2    | {{0,2},{1,3}}      |  -1
 *   F B    F      F     | 1, 2    | {{0,2},{1,3}}      |  -1   (a mixed pair is dropped whole)
 *   B B    B      F     | 2       | {{0,1}}            |  +1
 *   B B    F      B     | 1       | {{0,1}}            |  +1
 *   F F    B      B     | 0       | {{0,1}}            |  +1
 *   B B    B      B     | none    | empty              |  +1
 */
TEST(BackboneParity, interaction_lines_drop_out_of_the_permutation_parity) {

  // the all-fermionic reference
  ASSERT_EQ(triqs_xca::topology::topology_parity(crossing3), 1) << "the table below is written against an even all-fermionic topology";

  EXPECT_EQ(parity_of(n_int, 0, 1, 0, 1), 1) << "all fermionic";
  EXPECT_EQ(parity_of(n_int, 0, 1, o_int, 1), 1) << "line 1 bosonic: dropping (1,4) leaves {{0,1},{2,3}}";
  EXPECT_EQ(parity_of(n_int, 0, 1, 0, o_int), -1) << "line 2 bosonic: dropping (3,5) leaves {{0,2},{1,3}}";
  EXPECT_EQ(parity_of(n_int, o_int, o_int, 0, 1), -1) << "line 0 bosonic: dropping (0,2) leaves {{0,2},{1,3}}";
  EXPECT_EQ(parity_of(n_int, 0, o_int, 0, 1), -1) << "mixed line 0: a pair with one bosonic vertex is dropped whole";
  EXPECT_EQ(parity_of(n_int, o_int, o_int, o_int, 1), 1) << "lines 0 and 1 bosonic";
  EXPECT_EQ(parity_of(n_int, o_int, o_int, 0, o_int), 1) << "lines 0 and 2 bosonic";
  EXPECT_EQ(parity_of(n_int, 0, 1, o_int, o_int), 1) << "lines 1 and 2 bosonic";
  EXPECT_EQ(parity_of(n_int, o_int, o_int, o_int, o_int), 1) << "every line bosonic: the empty permutation is even";
}

/**
 * @brief Check that a two-argument construction, i.e. n_int = 0, calls every vertex fermionic
 */
TEST(BackboneParity, a_two_argument_construction_calls_every_vertex_fermionic) {

  int all_fermionic = triqs_xca::topology::topology_parity(crossing3);

  for (int orb_kap : {0, o_int}) {
    for (int orb1 : {0, o_int}) {
      for (int orb2 : {1, o_int}) {
        SCOPED_TRACE("orb_kap = " + std::to_string(orb_kap) + ", orb1 = " + std::to_string(orb1) + ", orb2 = " + std::to_string(orb2));
        EXPECT_EQ(parity_of(0, orb_kap, orb_kap, orb1, orb2), all_fermionic) << "with n_int = 0 the interaction flavour is just another orbital";
      }
    }
  }

  // check that the two constructions differ somewhere
  EXPECT_NE(parity_of(n_int, 0, 1, 0, o_int), parity_of(0, 0, 1, 0, o_int)) << "line 2 bosonic is the row that distinguishes the two constructions";
}

/**
 * @brief Check that the correlator sentinels -2 and -3 are equivalent to a fermionic resp. an interaction orbital index on both vertices of
 * line 0
 */
TEST(BackboneParity, correlator_external_flags_agree_with_explicit_orbital_indices) {

  auto correlator_parity = [](int n_int_arg, int flag, int orb1, int orb2) {
    CorrelatorBackbone B(crossing3, n_ext, n_int_arg);
    nda::vector<int> fb = {0, 1}; // m - 1 entries: no line is attached to vertex 0
    B.set_directions(fb);
    nda::vector<int> orb = nda::zeros<int>(6);
    orb(crossing3(1, 0)) = orb1;
    orb(crossing3(1, 1)) = orb1;
    orb(crossing3(2, 0)) = orb2;
    orb(crossing3(2, 1)) = orb2;
    B.set_orb_inds(orb);
    B.set_orb_inds_of_0_and_vct0(flag, flag);
    return B.get_parity();
  };

  for (int orb1 : {0, o_int}) {
    for (int orb2 : {1, o_int}) {
      SCOPED_TRACE("orb1 = " + std::to_string(orb1) + ", orb2 = " + std::to_string(orb2));
      EXPECT_EQ(correlator_parity(n_int, -2, orb1, orb2), parity_of(n_int, 0, 1, orb1, orb2)) << "-2 must behave as a fermionic orbital index";
      EXPECT_EQ(correlator_parity(n_int, -3, orb1, orb2), parity_of(n_int, o_int, o_int, orb1, orb2))
         << "-3 must behave as an interaction orbital index";
    }
  }

  // the sentinels are read before n_hyb, so with fermionic internal lines the external flag alone decides the parity
  EXPECT_EQ(correlator_parity(0, -3, 0, 1), correlator_parity(n_int, -3, 0, 1));
  EXPECT_EQ(correlator_parity(0, -2, 0, 1), correlator_parity(n_int, -2, 0, 1));
  EXPECT_NE(correlator_parity(n_int, -2, 0, 1), correlator_parity(n_int, -3, 0, 1)) << "vacuous test: the two sentinels are indistinguishable here";
}

/**
 * @brief Check that an unset orbital index, -1, is rejected by get_parity()
 */
TEST(BackboneParity, an_unset_orbital_index_is_rejected) {

  Backbone B(crossing3, n_ext, n_int);
  nda::vector<int> fb = {1, 0, 1};
  B.set_directions(fb);
  B.set_orb_inds(nda::zeros<int>(6));
  B.set_orb_inds_of_0_and_vct0(-1, 0);

  try {
    B.get_parity();
    FAIL() << "expected std::invalid_argument for an unset orbital index";
  } catch (std::invalid_argument const &e) { EXPECT_NE(std::string(e.what()).find("is not set"), std::string::npos) << e.what(); }
}
