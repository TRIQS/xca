#pragma once

#include <optional>

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
 * @brief Block of a fundamental operator acting on one atom_diag subspace, in the Fock basis
 *
 * @details atom_diag stores the operator between the eigenbases of two subspaces. The block returned is rotated to the Fock-state
 * bases, U_target * block * U_source^dagger, so its rows run over the Fock states of the target subspace and its columns over those of
 * subspace s. Only the memory of these two subspaces is used, never that of the full Hilbert space.
 *
 * @param[in] ad AtomDiag object
 * @param[in] oidx operator index
 * @param[in] is_creation true for the creation operator, false for the annihilation operator
 * @param[in] s source subspace
 * @return the block, or nullopt if the operator takes subspace s to zero; the target subspace is ad.c_connection(oidx, s) for the
 * annihilation operator and ad.cdag_connection(oidx, s) for the creation operator
 */
  std::optional<nda::matrix<dcomplex>> get_operator_block(const triqs_atom_diag_t<true> &ad, int oidx, bool is_creation, int s);
  std::optional<nda::matrix<dcomplex>> get_operator_block(const triqs_atom_diag_t<false> &ad, int oidx, bool is_creation, int s);

  /**
 * @brief Hamiltonian block of one atom_diag subspace in the Fock basis, U (E + E_gs) U^dagger
 *
 * @details Rows and columns run over the Fock states of subspace s, in the order of ad.get_fock_states(s). Only the memory of this
 * subspace is used, never that of the full Hilbert space.
 *
 * @param[in] ad AtomDiag object
 * @param[in] s subspace
 */
  nda::matrix<dcomplex> get_hamiltonian_block(const triqs_atom_diag &ad, int s);

} // namespace triqs_xca::atom_diag
