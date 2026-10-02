#pragma once

#include "triqs_xca/atom_diag.hpp"
#include "triqs_xca/block_sparse/block_op.hpp"

namespace triqs_xca::block_sparse::atom_diag {

  using triqs_xca::atom_diag::imtime_ops;
  using triqs_xca::atom_diag::triqs_atom_diag;
  using triqs_xca::atom_diag::triqs_atom_diag_t;

  /**
 * @brief The field operators of an AtomDiag object, grouped into symmetry sets, without the barred operators
 *
 * @details The operators are grouped by identical c_connection row, i.e. by block-sparsity pattern, labeled in order of
 * first appearance and stored in increasing orbital order within a set. This is the hybridization independent part of
 * get_operators(), used by get_operators_and_interactions() to append the interaction symmetry sets before the
 * quartet is built.
 */
  struct BlockOpSymSets {
    std::vector<BlockOpSymSet> Fs;     // annihilation operators, one entry per symmetry set
    std::vector<BlockOpSymSet> F_dags; // creation operators, same grouping
    nda::vector<long> sym_set_labels;  // symmetry set of each of the n orbitals
  };

  /**
 * @brief Group the first n field operators of an AtomDiag object into symmetry sets
 * @param[in] ad AtomDiag object
 * @param[in] n Number of orbitals to group
 * @return The annihilation and creation operator sets and the per-orbital symmetry set labels
 */
  BlockOpSymSets get_operator_sym_sets(const triqs_atom_diag_t<true> &ad, int n);
  BlockOpSymSets get_operator_sym_sets(const triqs_atom_diag_t<false> &ad, int n);

  /**
 * @brief Get creation and annihilation operators from an AtomDiag object
 * @param[in] ad AtomDiag object
 * @param[in] hyb_coeffs Hybridization SOE coefficients
 * @return Tuple of BlockOpSymSet objects
 */
  std::tuple<BlockOpSymQuartet, nda::vector<int>> get_operators(const triqs_atom_diag_t<true> &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs);
  std::tuple<BlockOpSymQuartet, nda::vector<int>> get_operators(const triqs_atom_diag_t<false> &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs);

} // namespace triqs_xca::block_sparse::atom_diag
