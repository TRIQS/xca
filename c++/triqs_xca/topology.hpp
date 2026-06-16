#pragma once

#include <nda/nda.hpp>

namespace triqs_xca::topology {

  /**
   * @brief Parity (fermionic sign) of a permutation via cycle decomposition
   *
   * The permutation is given as a 1d integer array @p perm that must be a
   * rearrangement of the integers 0, ..., n-1, where n = perm.size(). It is
   * interpreted as the map i -> perm(i). The parity is computed from the cycle
   * decomposition: a cycle of length L contributes L-1 transpositions, so the
   * sign is (-1)^(sum of (L-1)) = (-1)^(n - number_of_cycles).
   *
   * @param[in] perm 1d array holding a permutation of {0, ..., n-1}
   * @return +1 if the permutation is even, -1 if it is odd
   * @throws std::invalid_argument if @p perm is not a permutation of {0,...,n-1}
   */
  int permutation_parity(nda::array_const_view<int, 1> perm);

  /**
   * @brief Parity (fermionic sign) of a 2d topology array
   *
   * The 2d @p topology array is flattened (row-major) into a 1d integer vector,
   * which is then required to be a permutation of {0, ..., n-1} with
   * n = topology.size(). The parity of that permutation is returned.
   *
   * @param[in] topology 2d array whose flattened entries form a permutation
   * @return +1 if the flattened permutation is even, -1 if it is odd
   * @throws std::invalid_argument if the flattened array is not a permutation
   */
  int topology_parity(nda::array_const_view<int, 2> topology);

  /**
   * @brief Restrict a topology to its purely fermionic hybridization lines
   *
   * Each row of @p topology is a pair of vertices connected by a hybridization
   * line. The @p fermionic flag vector has one entry per vertex (its length
   * equals the number of elements in @p topology) and is indexed by vertex
   * value: @p fermionic(v) is true if vertex v is fermionic. A pair is kept only
   * if both of its vertices are fermionic.
   *
   * The surviving vertices are relabelled to a contiguous range 0, ..., m-1,
   * where m is the number of surviving vertices, by removing the gaps left by
   * the dropped vertices (each surviving label is decremented by the number of
   * removed vertices below it). The returned topology therefore remains a valid
   * permutation of {0, ..., m-1}.
   *
   * @param[in] topology 2d array whose rows are vertex pairs
   * @param[in] fermionic per-vertex fermionic flags, length topology.size()
   * @return a 2d topology of the all-fermionic rows, relabelled to 0, ..., m-1
   * @throws std::invalid_argument if @p fermionic has the wrong length or a
   *         topology entry is out of range for @p fermionic
   */
  nda::array<int, 2> fermionic_topology(nda::array_const_view<int, 2> topology, nda::array_const_view<bool, 1> fermionic);

} // namespace triqs_xca::topology
