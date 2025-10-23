#include "triqs_xca/block_sparse.hpp"
#include <triqs/atom_diag/atom_diag.hpp>

using namespace nda;
using namespace triqs;

/**
 * @brief Utility function to get full Hamiltonian matrix from an AtomDiag object.
 * @param[in] ad AtomDiag object
 */
matrix<double> get_full_h_atomic(const atom_diag::atom_diag<false> &ad);

/**
 * @brief Get full Hamiltonian matrix from an AtomDiag object with rows and columns permuted by Fock state ordering.
 * @param[in] ad AtomDiag object
 */
matrix<double> get_full_h_atomic_perm(const atom_diag::atom_diag<false> &ad);

/**
 * @brief Utility function to get full operator matrix from an AtomDiag object.
 * @param[in] ad AtomDiag object
 * @param[in] oidx operator index
 * @param[in] is_creation true for creation operator, false for annihilation operator
 */
matrix<double> get_full_operator_matrix(const atom_diag::atom_diag<false> &ad, int oidx, bool is_creation);

/**
 * @brief Get symmetry blocks of Hamiltonian from an AtomDiag object
 * @param[in] ad AtomDiag object
 * @return Tuple of vectors of Hamiltonian blocks and block indices
 */
std::tuple<std::vector<nda::array<double, 2>>, nda::vector<long>> get_hamiltonian_blocks(const atom_diag::atom_diag<false> &ad);

/**
 * @brief Get noninteracting Green's function from an AtomDiag object
 * @param[in] ad AtomDiag object
 * @param[in] beta Inverse temperature
 * @param[in] dlr_it_abs DLR interpolation nodes (absolute)
 * @return BlockDiagOpFun representing the noninteracting Green's function
 */
BlockDiagOpFun ad_to_nonint_gf(const atom_diag::atom_diag<false> &ad, double beta, const nda::vector_const_view<double> &dlr_it_abs);

/**
 * @brief Get creation and annihilation operators from an AtomDiag object
 * @param[in] ad AtomDiag object
 * @param[in] norb Number of orbitals
 * @param[in] hyb_coeffs Hybridization SOE coefficients
 * @param[in] hyb_refl_coeffs Reflected hybridization SOE coefficients
 * @return Tuple of BlockOpSymSet objects
 */
std::tuple<BlockOpSymQuartet, nda::vector<int>> get_operators(const atom_diag::atom_diag<false> &ad, int norb,
                                                              nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                                              nda::array_const_view<dcomplex, 3> hyb_refl_coeffs);

/**
 * @brief Get creation and annihilation operators from an AtomDiag object in dense storage
 * @param[in] ad AtomDiag object
 * @param[in] norb Number of orbitals
 * @param[in] hyb_coeffs Hybridization SOE coefficients
 * @param[in] hyb_refl_coeffs Reflected hybridization SOE coefficients
 * @return DenseFSet object
 */
DenseFSet get_operators_dense(const atom_diag::atom_diag<false> &ad, int norb, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                              nda::array_const_view<dcomplex, 3> hyb_refl_coeffs);
