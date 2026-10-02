#pragma once
#include "triqs_xca/block_sparse/block_op.hpp"

namespace triqs_xca::block_sparse {

  using nda::dcomplex;

  using cppdlr::imtime_ops;

  /**
 * @brief Evaluate NCA Green's function using block-sparse storage
 * @param[in] Gt pseudoparticle Green's function
 * @param[in] Gt_refl pseudoparticle Green's function at (beta - tau)
 * @param[in] Fs vector of annihilation operators
 * @return NCA term of self-energy
 */
  nda::array<dcomplex, 3> NCA_gf_bs(const BlockDiagOpFun &Gt, const BlockDiagOpFun &Gt_refl, const BlockOpSymQuartet &Fq);

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
