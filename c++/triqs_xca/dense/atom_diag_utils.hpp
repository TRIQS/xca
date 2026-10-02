#pragma once

#include "triqs_xca/atom_diag_utils.hpp"
#include "triqs_xca/dense/fset.hpp"

namespace triqs_xca::atom_diag {

  using triqs_xca::dense::FSet;

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
 * @brief Get a dense tensor in the full Hilbert resticted to one atom_diag subspace
 * @param[in] tensor_full Full tensor in the Hilbert space
 * @param[in] subspace_index Index of the subspace
 * @param[in] ad AtomDiag object
 * @return tensor_subspace Tensor in the subspace
 */
  nda::array<dcomplex, 3> get_tensor_in_atom_diag_subspace(nda::array_const_view<dcomplex, 3> tensor_full, int subspace_index,
                                                           triqs_atom_diag const &ad);

  /**
 * @brief Get creation and annihilation operators from an AtomDiag object in dense storage
 * @param[in] ad AtomDiag object
 * @param[in] hyb_coeffs Hybridization SOE coefficients
 * @return FSet object
 */
  FSet get_operators_dense(const triqs_atom_diag_t<true> &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs);
  FSet get_operators_dense(const triqs_atom_diag_t<false> &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs);

  /**
 * @brief Get creation and annihilation operators from an AtomDiag object in dense storage
 * @param[in] ad AtomDiag object
 * @param[in] hyb_coeffs Hybridization SOE coefficients
 * @return tuple with Fs and Fdags in dense storage
 */
  std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> get_operators_dense(const triqs_atom_diag_t<true> &ad);
  std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> get_operators_dense(const triqs_atom_diag_t<false> &ad);

} // namespace triqs_xca::atom_diag
