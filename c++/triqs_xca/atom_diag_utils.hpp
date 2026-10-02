#pragma once

#include <nda/nda.hpp>
#include <cppdlr/dlr_imtime.hpp>
#include <triqs/gfs.hpp>
#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/atom_diag/functions.hpp>
#include <triqs/utility/first_include.hpp>

namespace triqs_xca::atom_diag {

  using cppdlr::imtime_ops;

  template <bool IsComplex> using triqs_atom_diag_t = triqs::atom_diag::atom_diag<IsComplex>;

  using triqs_atom_diag = triqs_atom_diag_t<true>; // Default: complex valued Hamiltonians

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

} // namespace triqs_xca::atom_diag
