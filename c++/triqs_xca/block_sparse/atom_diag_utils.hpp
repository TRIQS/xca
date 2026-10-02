#pragma once

#include "triqs_xca/atom_diag_utils.hpp"
#include "triqs_xca/block_sparse/block_op.hpp"

namespace triqs_xca::atom_diag {

  using triqs_xca::block_sparse::BlockDiagOpFun;
  using triqs_xca::block_sparse::BlockOpSymQuartet;
  using triqs_xca::block_sparse::BlockOpSymSet;

  /**
 * @brief Get symmetry blocks of Hamiltonian from an AtomDiag object
 * @param[in] ad AtomDiag object
 * @return Tuple of vectors of Hamiltonian blocks and block indices
 */
  std::tuple<std::vector<nda::array<dcomplex, 2>>, nda::vector<long>> get_hamiltonian_blocks(const triqs_atom_diag &ad);

  /**
 * @brief Get atomic propagator from an AtomDiag object as a BlockDiagOpFun
 * @param[in] ad AtomDiag object
 * @param[in] beta Inverse temperature
 * @param[in] itops Imaginary time object
 * @return BlockDiagOpFun representing the atomic propagator
 */
  BlockDiagOpFun ad_to_atom_prop(const triqs_atom_diag &ad, double beta, imtime_ops &itops);

  /**
 * @brief Get atomic propagator from an AtomDiag object as a triqs::block_gf<dlr_imtime>
 * @param[in] ad AtomDiag object
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @return triqs::block_gf<dlr_imtime> representing the atomic propagator
 */
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> ad_to_atom_prop(const triqs_atom_diag &ad, double beta, double Lambda, double eps);

  /**
 * @brief The field operators of an AtomDiag object, grouped into symmetry sets, without the barred operators
 *
 * @details The operators are grouped by identical c_connection row, i.e. by block-sparsity pattern, labeled in order of
 * first appearance and stored in increasing orbital order within a set. This is the hybridization independent part of
 * get_operators(), used by dynint::get_operators_and_interactions() to append the interaction symmetry sets before the
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

  /**
 * @brief Scatter block-sparse data over the full Hilbert space, the inverse of get_tensor_in_atom_diag_subspace
 *
 * @details Block b occupies the rows and columns ad.get_fock_states(b) of the result, so that
 * get_tensor_in_atom_diag_subspace(get_tensor_in_full_hilbert_space(G, ad), b, ad) recovers block b. Entries outside
 * the blocks are left at zero, which lets a block-sparse result be compared against a dense one by subtracting the
 * two tensors directly: weight the block structure forbids shows up in the difference rather than being skipped.
 *
 * Block dimensions and positions are taken from ad, never from G, so blocks G flags as zero are placed correctly
 * even when their storage is empty.
 *
 * @param[in] G Block-diagonal operator whose blocks are ordered by atom_diag subspace
 * @param[in] ad AtomDiag object
 * @param[in] r Number of imaginary time nodes; if negative, taken from G.get_num_time_nodes(), which is only
 * available when at least one block of G is nonzero
 * @return tensor_full Tensor over the full Hilbert space
 */
  nda::array<dcomplex, 3> get_tensor_in_full_hilbert_space(BlockDiagOpFun const &G, triqs_atom_diag const &ad, int r = -1);

  /**
 * @brief Scatter block-sparse data over the full Hilbert space, the inverse of get_tensor_in_atom_diag_subspace
 * @param[in] G Block Green's function whose blocks are ordered by atom_diag subspace
 * @param[in] ad AtomDiag object
 * @return tensor_full Tensor over the full Hilbert space
 */
  nda::array<dcomplex, 3> get_tensor_in_full_hilbert_space(triqs::gfs::block_gf_const_view<triqs::mesh::dlr_imtime> G, triqs_atom_diag const &ad);

  /**
 * @brief Overloads for an owning block_gf and for a mutable view
 *
 * @details BlockDiagOpFun is implicitly constructible from a block_gf, so without an exact match for each of these
 * a call would be ambiguous between the two overloads above. Both forward to the block_gf_const_view overload.
 */
  nda::array<dcomplex, 3> get_tensor_in_full_hilbert_space(triqs::gfs::block_gf<triqs::mesh::dlr_imtime> const &G, triqs_atom_diag const &ad);
  nda::array<dcomplex, 3> get_tensor_in_full_hilbert_space(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G, triqs_atom_diag const &ad);

} // namespace triqs_xca::atom_diag
