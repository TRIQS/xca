#pragma once

#include <nda/nda.hpp>

#include <cppdlr/dlr_imtime.hpp>

#include <triqs/gfs.hpp>
#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/atom_diag/functions.hpp>
#include <triqs/utility/first_include.hpp>

#include "triqs_xca/dense.hpp"
#include "triqs_xca/block_sparse.hpp"

namespace triqs_xca::atom_diag {

  using cppdlr::imtime_ops;

  template <bool IsComplex> using triqs_atom_diag_t = triqs::atom_diag::atom_diag<IsComplex>;

  using triqs_atom_diag = triqs_atom_diag_t<true>; // Default: complex valued Hamiltonians

  using triqs_xca::dense::DenseFSet;

  using triqs_xca::block_sparse::BlockDiagOpFun;
  using triqs_xca::block_sparse::BlockOpSymQuartet;
  using triqs_xca::block_sparse::BlockOpSymSet;

  /**
 * @brief Utility function to get full Hamiltonian matrix from an AtomDiag object.
 * @param[in] ad AtomDiag object
 */
  nda::matrix<dcomplex> get_full_h_atomic(const triqs_atom_diag &ad);

  /**
 * @brief Utility function to get full operator matrix from an AtomDiag object.
 * @param[in] ad AtomDiag object
 * @param[in] oidx operator index
 * @param[in] is_creation true for creation operator, false for annihilation operator
 */
  nda::matrix<dcomplex> get_full_operator_matrix(const triqs_atom_diag_t<true> &ad, int oidx, bool is_creation);
  nda::matrix<dcomplex> get_full_operator_matrix(const triqs_atom_diag_t<false> &ad, int oidx, bool is_creation);

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
 * @brief Get creation and annihilation operators from an AtomDiag object
 * @param[in] ad AtomDiag object
 * @param[in] hyb_coeffs Hybridization SOE coefficients
 * @return Tuple of BlockOpSymSet objects
 */
  std::tuple<BlockOpSymQuartet, nda::vector<int>> get_operators(const triqs_atom_diag_t<true> &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs);
  std::tuple<BlockOpSymQuartet, nda::vector<int>> get_operators(const triqs_atom_diag_t<false> &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs);

  /**
 * @brief Get creation and annihilation operators from an AtomDiag object in dense storage
 * @param[in] ad AtomDiag object
 * @param[in] hyb_coeffs Hybridization SOE coefficients
 * @return DenseFSet object
 */
  DenseFSet get_operators_dense(const triqs_atom_diag_t<true> &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs);
  DenseFSet get_operators_dense(const triqs_atom_diag_t<false> &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs);

  /**
 * @brief Get creation and annihilation operators from an AtomDiag object in dense storage
 * @param[in] ad AtomDiag object
 * @param[in] hyb_coeffs Hybridization SOE coefficients
 * @return tuple with Fs and Fdags in dense storage
 */
  std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> get_operators_dense(const triqs_atom_diag_t<true> &ad);
  std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> get_operators_dense(const triqs_atom_diag_t<false> &ad);

  /**
 * @brief Get a dense tensor in the full Hilbert resticted to one atom_diag subspace
 * @param[in] tensor_full Full tensor in the Hilbert space
 * @param[in] subspace_index Index of the subspace
 * @param[in] ad AtomDiag object
 * @return tensor_subspace Tensor in the subspace
 */
  nda::array<dcomplex, 3> get_tensor_in_atom_diag_subspace(nda::array_const_view<dcomplex, 3> tensor_full, int subspace_index,
                                                           triqs_atom_diag const &ad);

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