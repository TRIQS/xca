#include "nda/nda.hpp"
#include "block_sparse.hpp"
#include <cppdlr/dlr_kernels.hpp>
#include <nda/declarations.hpp>

using namespace nda;

/**
 * @brief Evaluate NCA self-energy term using block-sparse storage
 * @param[in] hyb hybridization function
 * @param[in] hyb_refl hyb evaluated at (beta - tau)
 * @param[in] Gt Greens function
 * @param[in] Fs vector of annihilation operators
 * @return NCA term of self-energy
 */
BlockDiagOpFun NCA_bs(nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl, const BlockDiagOpFun &Gt,
                      const std::vector<BlockOp> &Fs);

/**
 * @brief Evaluate NCA self-energy term using dense storage
 * @param[in] hyb hybridization function
 * @param[in] hyb_refl hyb evaluated at (beta - tau)
 * @param[in] Gt Greens function
 * @param[in] Fs vector of annihilation operators
 * @param[in] F_dags vector of creation operators
 */
nda::array<dcomplex, 3> NCA_dense(nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl,
                                  nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> Fs,
                                  nda::array_const_view<dcomplex, 3> F_dags);

/**
 * @brief Build matrix of evaluations of K at imag times and real freqs
 * @param[in] dlr_it DLR imaginary time nodes
 * @param[in] dlr_rf DLR real frequencies
 * @return matrix of K evalutions
 */
nda::array<double, 2> K_mat(nda::vector_const_view<double> dlr_it, nda::vector_const_view<double> dlr_rf, double beta);

/**
 * @brief Evaluate OCA using block-sparse storage
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt Greens function
 * @param[in] Fs F operator
 * @return OCA term of self-energy
 */
BlockDiagOpFun OCA_bs(nda::array_const_view<dcomplex, 3> hyb, imtime_ops &itops, double beta, const BlockDiagOpFun &Gt,
                      const std::vector<BlockOp> &Fs);

/**
 * @brief Evaluate OCA using block-sparse storage and allow user to provide hybridization poles and coefficients
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] hyb_coeffs hybridization coefficients
 * @param[in] hyb_refl hybridization function eval'd at negative imag. times
 * @param[in] hyb_refl_coeffs hybridization coefficients at negative imag. times
 * @param[in] hyb_poles hybridization poles
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt Greens function
 * @param[in] Fs F operator
 * @return OCA term of self-energy
 */
BlockDiagOpFun OCA_bs(nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                      nda::array_const_view<dcomplex, 3> hyb_refl, nda::array_const_view<dcomplex, 3> hyb_refl_coeffs,
                      nda::vector_const_view<double> hyb_poles, imtime_ops &itops, double beta, const BlockDiagOpFun &Gt,
                      const std::vector<BlockOp> &Fs);

nda::array<dcomplex, 3> eval_eq(imtime_ops &itops, nda::array_const_view<dcomplex, 3> f, int n_quad);

/**
 * @brief Evaluate OCA using dense storage
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt Greens function
 * @param[in] Fs F operator
 * @return OCA term of self-energy
 */
nda::array<dcomplex, 3> OCA_dense(nda::array_const_view<dcomplex, 3> hyb, imtime_ops itops, double beta, nda::array_const_view<dcomplex, 3> Gt,
                                  nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags);

/**
 * @brief Evaluate OCA using dense storage and allow user to provide hybridization poles and coefficients
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] hyb_coeffs hybridization coefficients
 * @param[in] hyb_refl hybridization function eval'd at negative imag. times
 * @param[in] hyb_refl_coeffs hybridization coefficients at negative imag. times
 * @param[in] hyb_poles hybridization poles
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt Greens function
 * @param[in] Fs F operator
 * @return OCA term of self-energy
 */
nda::array<dcomplex, 3> OCA_dense(nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                  nda::array_const_view<dcomplex, 3> hyb_refl, nda::array_const_view<dcomplex, 3> hyb_refl_coeffs,
                                  nda::vector_const_view<double> hyb_poles, imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> Gt,
                                  nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags);

/**
 * @brief Evaluate OCA directly using trapezoidal quadrature
 * @param[in] hyb hybridization function at imaginary time nodes
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt Greens function
 * @param[in] Fs F operator
 * @param[in] n_quad number of quadrature nodes
 * @return OCA term of self-energy
 */
nda::array<dcomplex, 3> OCA_tpz(nda::array_const_view<dcomplex, 3> hyb, imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> Gt,
                                nda::array_const_view<dcomplex, 3> Fs, int n_quad);

nda::array<dcomplex, 3> third_order_dense_partial(nda::array_const_view<dcomplex, 3> hyb, imtime_ops &itops, double beta,
                                                  nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> Fs,
                                                  nda::array_const_view<dcomplex, 3> F_dags);
