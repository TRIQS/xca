#include <nda/nda.hpp>

#include <cppdlr/dlr_imtime.hpp>

#include <triqs/gfs.hpp>
#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/utility/first_include.hpp>

#include "triqs_xca/dense.hpp"
#include "triqs_xca/block_sparse.hpp"

namespace triqs_xca::atom_diag {

using cppdlr::imtime_ops;

template <bool IsComplex>
using triqs_atom_diag_t = triqs::atom_diag::atom_diag<IsComplex>;

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
 * @brief Exponentiate Hamiltonian blocks to get atomic propagator blocks
 * @param[in] H_blocks Vector of Hamiltonian blocks
 * @param[in] H_block_inds Vector of Hamiltonian block indices
 * @param[in] beta Inverse temperature
 * @param[in] itops Imaginary time object
 * @return Vector of atomic propagator blocks
 */
template <typename T>
std::vector<nda::array<T, 3>> H_to_atom_prop_blocks(std::vector<nda::array<dcomplex, 2>> &H_blocks, nda::vector_const_view<long> H_block_inds,
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
nda::array<dcomplex, 3> get_tensor_in_atom_diag_subspace(nda::array_const_view<dcomplex, 3> tensor_full, int subspace_index, triqs_atom_diag const &ad);


} // namespace triqs_xca::atom_diag