#include <nda/nda.hpp>

#include <cppdlr/dlr_imtime.hpp>

#include <triqs/gfs.hpp>
#include <triqs/atom_diag/atom_diag.hpp>

#include "triqs_xca/dense.hpp"
#include "triqs_xca/block_sparse.hpp"

using cppdlr::imtime_ops;

using triqs_atom_diag = triqs::atom_diag::atom_diag<false>; // true for complex valued Hamiltonians

/**
 * @brief Utility function to get full Hamiltonian matrix from an AtomDiag object.
 * @param[in] ad AtomDiag object
 */
nda::matrix<double> get_full_h_atomic(const triqs_atom_diag &ad);

/**
 * @brief Get full Hamiltonian matrix from an AtomDiag object with rows and columns permuted by Fock state ordering.
 * @param[in] ad AtomDiag object
 */
nda::matrix<double> get_full_h_atomic_perm(const triqs_atom_diag &ad);

/**
 * @brief Utility function to get full operator matrix from an AtomDiag object.
 * @param[in] ad AtomDiag object
 * @param[in] oidx operator index
 * @param[in] is_creation true for creation operator, false for annihilation operator
 */
nda::matrix<double> get_full_operator_matrix(const triqs_atom_diag &ad, int oidx, bool is_creation);

/**
 * @brief Get symmetry blocks of Hamiltonian from an AtomDiag object
 * @param[in] ad AtomDiag object
 * @return Tuple of vectors of Hamiltonian blocks and block indices
 */
std::tuple<std::vector<nda::array<double, 2>>, nda::vector<long>> get_hamiltonian_blocks(const triqs_atom_diag &ad);

/**
 * @brief Exponentiate Hamiltonian blocks to get atomic propagator blocks
 * @param[in] H_blocks Vector of Hamiltonian blocks
 * @param[in] H_block_inds Vector of Hamiltonian block indices
 * @param[in] beta Inverse temperature
 * @param[in] itops Imaginary time object
 * @return Vector of atomic propagator blocks
 */
template <typename T>
std::vector<nda::array<T, 3>> H_to_atom_prop_blocks(std::vector<nda::array<double, 2>> &H_blocks, nda::vector_const_view<long> H_block_inds,
                                                    double beta, imtime_ops &itops);

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
std::tuple<BlockOpSymQuartet, nda::vector<int>> get_operators(const triqs_atom_diag &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs);

/**
 * @brief Get creation and annihilation operators from an AtomDiag object in dense storage
 * @param[in] ad AtomDiag object
 * @param[in] norb Number of orbitals
 * @param[in] hyb_coeffs Hybridization SOE coefficients
 * @return DenseFSet object
 */
DenseFSet get_operators_dense(const triqs_atom_diag &ad, int norb, nda::array_const_view<dcomplex, 3> hyb_coeffs);
