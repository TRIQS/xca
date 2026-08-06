#include "triqs_xca/block_sparse.hpp"

namespace triqs_xca::block_sparse {

using nda::dcomplex;

using cppdlr::imtime_ops;

/**
 * @brief Evaluate NCA Green's function using dense storage
 * @param[in] Gt pseudoparticle Green's function
 * @param[in] Gt_refl pseudoparticle Green's function at (beta - tau)
 * @param[in] Fs annihilation operators
 * @param[in] F_dags creation operators
 */
nda::array<dcomplex, 3> NCA_gf_dense(nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> Gt_refl,
                                     nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags);

/**
 * @brief Evaluate NCA Green's function using block-sparse storage
 * @param[in] Gt pseudoparticle Green's function
 * @param[in] Gt_refl pseudoparticle Green's function at (beta - tau)
 * @param[in] Fs vector of annihilation operators
 * @return NCA term of self-energy
 */
nda::array<dcomplex, 3> NCA_gf_bs(const BlockDiagOpFun &Gt, const BlockDiagOpFun &Gt_refl, const BlockOpSymQuartet &Fq);

nda::array<dcomplex, 3> OCA_gf_tpz(nda::array_const_view<dcomplex, 3> hyb_coeffs, nda::array_const_view<dcomplex, 3> hyb_refl_coeffs,
                                   imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> Gt_coeffs,
                                   nda::array_const_view<dcomplex, 3> Fs, int n_quad);

/**
 * @brief Evaluate the third-order single-particle Green's function directly using trapezoidal quadrature
 *
 * @details Topology {{0,3},{1,4},{2,5}}, i.e. the integral of
 * examples/one_fermion_analytical_solutions.ipynb, "Third order single particle Green's function diagram".
 * Stands to OCA_gf_tpz as third_order_tpz (block_sparse_manual.hpp) stands to OCA_tpz. Returns the result
 * on the equispaced grid of n_quad + 1 points; both grid endpoints are left at zero.
 *
 * @param[in] hyb_coeffs DLR coefficients of the hybridization
 * @param[in] hyb_refl_coeffs DLR coefficients of the hybridization at negative imaginary times
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt_coeffs DLR coefficients of the pseudoparticle Green's function
 * @param[in] Fs annihilation operators
 * @param[in] n_quad number of quadrature intervals
 * @return third-order contribution to the single-particle Green's function
 */
nda::array<dcomplex, 3> third_order_gf_tpz(nda::array_const_view<dcomplex, 3> hyb_coeffs, nda::array_const_view<dcomplex, 3> hyb_refl_coeffs,
                                           imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> Gt_coeffs,
                                           nda::array_const_view<dcomplex, 3> Fs, int n_quad);

/**
 * @brief Evaluate OCA Green's function using dense storage
 * @param[in] hyb_coeffs hybridization coefficients
 * @param[in] hyb_refl_coeffs hybridization coefficients at negative imag. times
 * @param[in] hyb_poles hybridization poles
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt pseudoparticle Green's function
 * @param[in] Fs F operators
 * @param[in] F_dags F^dagger operators
 */
nda::array<dcomplex, 3> OCA_gf_dense(nda::array_const_view<dcomplex, 3> hyb_coeffs, nda::array_const_view<dcomplex, 3> hyb_refl_coeffs,
                                     nda::vector_const_view<double> hyb_poles, imtime_ops &itops, double beta, nda::array_const_view<dcomplex, 3> Gt,
                                     nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags);

/**
 * @brief Evaluate OCA Green's function using block-sparse storage
 * @param[in] hyb_poles hybridization poles
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt pseudoparticle Green's function as a BDOF
 * @param[in] Fq quartet of F operators
 */
nda::array<dcomplex, 3> OCA_gf_bs(nda::vector_const_view<double> hyb_poles, imtime_ops &itops, double beta, const BlockDiagOpFun &Gt,
                                  const BlockOpSymQuartet &Fq);

} // namespace triqs_xca::block_sparse