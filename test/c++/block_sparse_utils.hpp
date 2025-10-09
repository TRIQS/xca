#pragma once
#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include <triqs_soehyb/block_sparse.hpp>

using namespace nda;
using namespace cppdlr;

/**
 * @brief Convert a Hamiltonian matrix to a non-interacting Green's function matrix in dense storage
 * @param[in] Hmat Hamiltonian matrix
 * @param[in] beta Inverse temperature
 * @param[in] dlr_it_abs DLR imaginary time nodes in absolute format
 * @return Non-interacting Green's function matrix in dense storage
 */
nda::array<dcomplex, 3> Hmat_to_Gtmat(nda::array<dcomplex, 2> Hmat, double beta, nda::array<double, 1> dlr_it_abs);

/**
 * @brief Helper function for setting up the discrete bath hybridization function used in two-band tests
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @return Tuple of hybridization function and its reflection
 */
std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> discrete_bath_helper(double beta, double Lambda, double eps);

/**
 * @brief Helper function for setting up the discrete bath hybridization function used in spin-flip-fermion tests
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @param[in] n 2 * number of orbitals
 * @return Tuple of hybridization function and its reflection
 */
std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> discrete_bath_spin_flip_helper(double beta, double Lambda, double eps, int n);

/**
 * @brief Helper function for setting up the two-band model
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @param[in] hyb_coeffs Hybridization function coefficients
 * @param[in] hyb_refl_coeffs Reflected hybridization function coefficients
 * @return Tuple of non-interacting Green's function as a BDOF, field operators as a BlockOpSymQuartet, and vector of block symmetry labels
 */
std::tuple<BlockDiagOpFun, BlockOpSymQuartet, nda::vector<int>> two_band_helper(double beta, double Lambda, double eps,
                                                                                nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                                                                nda::array_const_view<dcomplex, 3> hyb_refl_coeffs);

/**
 * @brief Helper function for setting up the two-band model
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @return Tuple of number of blocks, hybridization function, reflected hybridization function, non-interacting Green's function as a BDOF,
 *         annihilation operators as a vector of BlockOps, creation operators as a vector of BlockOps, non-interacting Green's function in dense
 *         storage, annihilation operators in dense storage, creation operators in dense storage, vector of vector of subspaces, and flattened version
 *         of this vector
 */
std::tuple<int, nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, BlockDiagOpFun, std::vector<BlockOp>, std::vector<BlockOp>, nda::array<dcomplex, 3>,
           nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, std::vector<std::vector<unsigned long>>, std::vector<long>>
two_band_discrete_bath_helper(double beta, double Lambda, double eps);

/**
 * @brief Helper function for setting up the two-band model with symmetry-class creation/annihilation operators
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @return Tuple of number of blocks, hybridization function, reflected hybridization function, non-interacting Green's function as a BDOF,
 *         annihilation operators as a vector of BlockOps, creation operators as a vector of BlockOps, non-interacting Green's function in dense
 *         storage, annihilation operators in dense storage, creation operators in dense storage, vector of vector of subspaces, flattened version
 *         of this vector, and the creation/annihilation operators as a BlockOpSymQuartet
 */
std::tuple<int, nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, BlockDiagOpFun, std::vector<BlockOp>, std::vector<BlockOp>, nda::array<dcomplex, 3>,
           nda::array<dcomplex, 3>, nda::array<dcomplex, 3>, std::vector<std::vector<unsigned long>>, std::vector<long>, BlockOpSymQuartet>
two_band_discrete_bath_helper_sym(double beta, double Lambda, double eps);

/**
 * @brief Helper function for setting up the spin-flip-fermion model with dense storage
 * @param[in] beta Inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 * @param[in] hyb_coeffs Hybridization function coefficients
 * @param[in] hyb_refl_coeffs Reflected hybridization function coefficients
 * @param[in] filename Path to HDF5 file containing the Hamiltonian blocks and subspaces
 * @return Tuple of hybridization function, annihilation operators in dense storage, and vector of vector of subspaces
 */
std::tuple<nda::array<dcomplex, 3>, DenseFSet, std::vector<nda::vector<unsigned long>>>
spin_flip_fermion_dense_helper(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                               nda::array_const_view<dcomplex, 3> hyb_refl_coeffs, std::string filename);