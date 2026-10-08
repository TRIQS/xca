#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <nda/nda.hpp>

#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/gfs.hpp>

#include <triqs_xca/atom_diag.hpp>
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/block_sparse/block_op.hpp>

#include "test_utils/dense.hpp"

namespace triqs_xca::test_utils {

/**
 * @file dense_comparison.hpp
 *
 * @brief Comparison of a block-sparse result against the dense evaluator's, block by block
 *
 * @details The self-energy is stored per atom_diag subspace on the block-sparse side and as one dense matrix on the dense side, so the
 * comparison cuts the dense matrix down to each block with get_tensor_in_atom_diag_subspace(). The correlator and the single-particle
 * Green's function are traces over the whole Hilbert space, indexed by orbital, and compare directly. Both helpers return {err, scale},
 * so the caller picks the tolerance. Kept in a header since both depend on gtest, which the test_utils library does not link.
 */

/**
 * @brief max|Sigma_bs - Sigma_dense| over the blocks, with the dense side cut to each subspace
 *
 * @param[in] Sigma block-sparse self-energy
 * @param[in] Sigma_dense dense self-energy on the parallel sym_ops = {} atom_diag
 * @param[in] ad the partitioned atom_diag the block-sparse side was built on
 * @return {max absolute difference, max|Sigma_dense| over the same blocks}
 */
inline std::pair<double, double> compare_sigma_with_dense(triqs_xca::block_sparse::BlockDiagOpFun &Sigma,
                                                          triqs::gfs::block_gf<triqs::mesh::dlr_imtime> const &Sigma_dense,
                                                          triqs::atom_diag::atom_diag<true> const &ad) {
  double err = 0.0, scale = 0.0;
  for (int b = 0; b < Sigma.get_num_block_cols(); ++b) {
    SCOPED_TRACE("block " + std::to_string(b));
    auto ref = get_tensor_in_atom_diag_subspace(Sigma_dense[0].data(), b, ad);
    err      = std::max(err, nda::max_element(nda::abs(Sigma.get_block(b) - ref)));
    scale    = std::max(scale, nda::max_element(nda::abs(ref)));
  }
  return {err, scale};
}

/**
 * @brief max|A - B| over two single-particle Green's functions, which have shape (r, n_hyb, n_hyb) in both evaluators
 *
 * @param[in] A first array
 * @param[in] B second array, of the same shape
 * @return {max absolute difference, max|B|}; the difference is infinite, with a test failure, if the shapes differ
 */
inline std::pair<double, double> compare_spgf(nda::array_const_view<nda::dcomplex, 3> A, nda::array_const_view<nda::dcomplex, 3> B) {
  double scale = nda::max_element(nda::abs(B));
  if (A.shape() != B.shape()) {
    ADD_FAILURE() << "spgf shapes differ: (" << A.extent(0) << ", " << A.extent(1) << ", " << A.extent(2) << ") vs (" << B.extent(0) << ", "
                  << B.extent(1) << ", " << B.extent(2) << ")";
    return {std::numeric_limits<double>::infinity(), scale};
  }
  return {nda::max_element(nda::abs(A - B)), scale};
}

} // namespace triqs_xca::test_utils
