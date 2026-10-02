#pragma once

#include <algorithm>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <nda/nda.hpp>

#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs/gfs.hpp>

#include <triqs_xca/atom_diag.hpp>
#include <triqs_xca/dense/atom_diag.hpp>
#include <triqs_xca/block_sparse/block_op.hpp>

#include "dense_utils.hpp"

/**
 * @file dense_comparison.hpp
 *
 * @brief Comparison of a block-sparse result against the dense evaluator's, block by block
 *
 * @details The self-energy is stored per atom_diag subspace on the block-sparse side and as one dense matrix on the dense side, so the
 * comparison cuts the dense matrix down to each block with get_tensor_in_atom_diag_subspace(). The correlator and the single-particle
 * Green's function are traces over the whole Hilbert space, indexed by orbital, and compare directly. Both helpers return {err, scale}
 * rather than asserting, so the caller picks the tolerance. Kept in a header since compare_sigma_with_dense depends on gtest, which the
 * shared fixtures in block_sparse_utils avoid.
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
 * @brief max|A - B| over the leading n_hyb x n_hyb window of two trace quantities
 *
 * @details The block-sparse single-particle Green's function excludes the interaction flavours and has shape (r, n_hyb, n_hyb), while the
 * dense one emits all n_ext legs, so only the leading window is comparable.
 *
 * @param[in] A first array, (r, >=n_hyb, >=n_hyb)
 * @param[in] B second array, same
 * @param[in] n_hyb number of fermionic flavours
 * @return {max absolute difference, max|B| over the window}
 */
inline std::pair<double, double> compare_leading_block(nda::array_const_view<nda::dcomplex, 3> A, nda::array_const_view<nda::dcomplex, 3> B,
                                                       int n_hyb) {
  double err = 0.0, scale = 0.0;
  for (int i = 0; i < n_hyb; ++i)
    for (int j = 0; j < n_hyb; ++j) {
      err   = std::max(err, nda::max_element(nda::abs(nda::make_regular(A(nda::range::all, i, j) - B(nda::range::all, i, j)))));
      scale = std::max(scale, nda::max_element(nda::abs(B(nda::range::all, i, j))));
    }
  return {err, scale};
}
