#pragma once
#include "triqs_xca/block_sparse/block_op.hpp"

namespace triqs_xca::block_sparse {

  using nda::dcomplex;

  using cppdlr::imtime_ops;

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
 * @brief Evaluate NCA self-energy term using block-sparse storage 
 * @param[in] hyb hybridization function
 * @param[in] hyb_refl hybridization function eval'd at (beta - tau)
 * @param[in] Gt pseudoparticle Green's function as a BDOF
 * @param[in] Fq quartet of F operators
 */
  BlockDiagOpFun NCA_bs(nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl, BlockDiagOpFun const &Gt,
                        const BlockOpSymQuartet &Fq);

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

  /**
 * @brief Evaluate OCA self-energy using block-sparse storage
 * @param[in] hyb hybridization function at imaginary times
 * @param[in] hyb_poles hybridization poles
 * @param[in] itops cppdlr imaginary time object
 * @param[in] beta inverse temperature
 * @param[in] Gt pseudoparticle Green's function as a BDOF
 * @param[in] Fq quartet of F operators
 */
  BlockDiagOpFun OCA_bs(nda::array_const_view<dcomplex, 3> hyb, nda::vector_const_view<double> hyb_poles, imtime_ops &itops, double beta,
                        const BlockDiagOpFun &Gt, const BlockOpSymQuartet &Fq);

} // namespace triqs_xca::block_sparse
