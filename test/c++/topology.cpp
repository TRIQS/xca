#include "triqs_xca/topology.hpp"

#include <nda/nda.hpp>
#include <gtest/gtest.h>

#include <stdexcept>

using namespace triqs_xca::topology;

TEST(test_topology, permutation_parity_identity) {
  // Identity: zero transpositions -> even -> +1
  EXPECT_EQ(permutation_parity(nda::array<int, 1>{0}), 1);
  EXPECT_EQ(permutation_parity(nda::array<int, 1>{0, 1, 2, 3}), 1);
}

TEST(test_topology, permutation_parity_single_transposition) {
  // (0 1): one transposition -> odd -> -1
  EXPECT_EQ(permutation_parity(nda::array<int, 1>{1, 0}), -1);
  EXPECT_EQ(permutation_parity(nda::array<int, 1>{1, 0, 2, 3}), -1);
}

TEST(test_topology, permutation_parity_double_transposition) {
  // (0 1)(2 3): two transpositions -> even -> +1
  EXPECT_EQ(permutation_parity(nda::array<int, 1>{1, 0, 3, 2}), 1);
}

TEST(test_topology, permutation_parity_cycles) {
  // 3-cycle (0 1 2): two transpositions -> even -> +1
  EXPECT_EQ(permutation_parity(nda::array<int, 1>{1, 2, 0}), 1);
  // 4-cycle (0 1 2 3): three transpositions -> odd -> -1
  EXPECT_EQ(permutation_parity(nda::array<int, 1>{1, 2, 3, 0}), -1);
  // Full reversal of 4 elements = (0 3)(1 2): two transpositions -> even -> +1
  EXPECT_EQ(permutation_parity(nda::array<int, 1>{3, 2, 1, 0}), 1);
}

TEST(test_topology, permutation_parity_invalid_throws) {
  // Repeated entry is not a permutation.
  EXPECT_THROW(permutation_parity(nda::array<int, 1>{0, 0, 1}), std::invalid_argument);
  // Out-of-range entry is not a permutation of {0, 1, 2}.
  EXPECT_THROW(permutation_parity(nda::array<int, 1>{0, 1, 3}), std::invalid_argument);
}

TEST(test_topology, topology_parity_matches_flattened) {
  // [[0, 1], [2, 3]] flattens to {0, 1, 2, 3} -> identity -> +1
  EXPECT_EQ(topology_parity(nda::array<int, 2>{{0, 1}, {2, 3}}), 1);

  // [[0, 2], [1, 3]] flattens to {0, 2, 1, 3} = (1 2) -> odd -> -1
  EXPECT_EQ(topology_parity(nda::array<int, 2>{{0, 2}, {1, 3}}), -1);

  // [[0, 3], [1, 2]] flattens to {0, 3, 1, 2} = 3-cycle (1 3 2) -> even -> +1
  EXPECT_EQ(topology_parity(nda::array<int, 2>{{0, 3}, {1, 2}}), 1);
}

TEST(test_topology, topology_parity_consistency_with_permutation_parity) {
  // The frontend must agree with the manually flattened call.
  auto topology = nda::array<int, 2>{{2, 0}, {5, 1}, {4, 3}};
  auto flat     = nda::array<int, 1>{2, 0, 5, 1, 4, 3};
  EXPECT_EQ(topology_parity(topology), permutation_parity(flat));
}

TEST(test_topology, topology_parity_invalid_throws) {
  // Flattened array {0, 1, 1, 2} is not a permutation.
  EXPECT_THROW(topology_parity(nda::array<int, 2>{{0, 1}, {1, 2}}), std::invalid_argument);
}

TEST(test_topology, fermionic_topology_drops_bosonic_pairs) {
  // Vertices 2 and 3 are bosonic, so the pair (2, 3) is dropped; the others are
  // kept. After removing 2 and 3, the surviving vertices 4, 5 shift down to 2, 3.
  auto topology  = nda::array<int, 2>{{0, 1}, {2, 3}, {4, 5}};
  auto fermionic = nda::array<bool, 1>{true, true, false, false, true, true};

  auto expected = nda::array<int, 2>{{0, 1}, {2, 3}};
  auto result   = fermionic_topology(topology, fermionic);

  EXPECT_EQ(result.shape(), expected.shape());
  EXPECT_EQ(result, expected);
}

TEST(test_topology, fermionic_topology_drops_mixed_pairs) {
  // A pair is kept only if BOTH vertices are fermionic; (1, 3) has a bosonic
  // vertex 1, so both 1 and 3 are removed and the survivor 2 shifts down to 1.
  auto topology  = nda::array<int, 2>{{0, 2}, {1, 3}};
  auto fermionic = nda::array<bool, 1>{true, false, true, true}; // indexed by vertex value

  auto expected = nda::array<int, 2>{{0, 1}};
  auto result   = fermionic_topology(topology, fermionic);

  EXPECT_EQ(result.shape(), expected.shape());
  EXPECT_EQ(result, expected);
}

TEST(test_topology, fermionic_topology_all_fermionic) {
  // Everything fermionic -> topology returned unchanged.
  auto topology  = nda::array<int, 2>{{0, 3}, {1, 2}};
  auto fermionic = nda::array<bool, 1>{true, true, true, true};

  auto result = fermionic_topology(topology, fermionic);

  EXPECT_EQ(result.shape(), topology.shape());
  EXPECT_EQ(result, topology);
}

TEST(test_topology, fermionic_topology_none_fermionic) {
  // No fermionic vertices -> empty topology with the original column count.
  auto topology  = nda::array<int, 2>{{0, 1}, {2, 3}};
  auto fermionic = nda::array<bool, 1>{false, false, false, false};

  auto result = fermionic_topology(topology, fermionic);

  EXPECT_EQ(result.extent(0), 0);
  EXPECT_EQ(result.extent(1), 2);
}

TEST(test_topology, fermionic_topology_invalid_length_throws) {
  // The flag vector length must equal the number of topology elements (4 here).
  auto topology  = nda::array<int, 2>{{0, 1}, {2, 3}};
  auto fermionic = nda::array<bool, 1>{true, true, true};
  EXPECT_THROW(fermionic_topology(topology, fermionic), std::invalid_argument);
}

TEST(test_topology, fermionic_topology_out_of_range_throws) {
  // Length matches, but vertex 5 cannot be looked up in a length-2 flag vector.
  auto topology  = nda::array<int, 2>{{0, 5}};
  auto fermionic = nda::array<bool, 1>{true, true};
  EXPECT_THROW(fermionic_topology(topology, fermionic), std::invalid_argument);
}

TEST(test_topology, fermionic_topology_output_is_valid_permutation) {
  // Drop the bosonic pair (2, 3); the survivors {0, 1, 4, 5} are relabelled to
  // {0, 1, 2, 3}. topology_parity only succeeds on a valid permutation, so a
  // non-throwing call confirms the relabelled output is one.
  auto topology  = nda::array<int, 2>{{1, 4}, {0, 5}, {2, 3}};
  auto fermionic = nda::array<bool, 1>{true, true, false, false, true, true};

  auto result = fermionic_topology(topology, fermionic);

  // (1, 4) -> (1, 2) and (0, 5) -> (0, 3): flattened {1, 2, 0, 3} is a 3-cycle.
  auto expected = nda::array<int, 2>{{1, 2}, {0, 3}};
  EXPECT_EQ(result.shape(), expected.shape());
  EXPECT_EQ(result, expected);

  int parity = 0;
  EXPECT_NO_THROW(parity = topology_parity(result));
  EXPECT_EQ(parity, 1); // 3-cycle (0 1 2) is even
}

TEST(test_topology, fermionic_topology_output_permutation_odd) {
  // Drop the bosonic pair (2, 5); survivors {0, 1, 3, 4} relabel to {0, 1, 2, 3}
  // (3 -> 2, 4 -> 3). Row (1, 3) -> (1, 2) and (4, 0) -> (3, 0).
  auto topology  = nda::array<int, 2>{{1, 3}, {2, 5}, {4, 0}};
  auto fermionic = nda::array<bool, 1>{true, true, false, true, true, false};

  auto result = fermionic_topology(topology, fermionic);

  auto expected = nda::array<int, 2>{{1, 2}, {3, 0}};
  EXPECT_EQ(result.shape(), expected.shape());
  EXPECT_EQ(result, expected);

  // {1, 2, 3, 0} is the 4-cycle (0 1 2 3), an odd permutation.
  int parity = 0;
  EXPECT_NO_THROW(parity = topology_parity(result));
  EXPECT_EQ(parity, -1);
}

TEST(test_topology, fermionic_topology_output_permutation_transposition) {
  // Keep both pairs (all fermionic), no relabelling. Flattened {1, 0, 2, 3}
  // is a single transposition (0 1), an odd permutation.
  auto topology  = nda::array<int, 2>{{1, 0}, {2, 3}};
  auto fermionic = nda::array<bool, 1>{true, true, true, true};

  auto result = fermionic_topology(topology, fermionic);

  int parity = 0;
  EXPECT_NO_THROW(parity = topology_parity(result));
  EXPECT_EQ(parity, -1);
}
