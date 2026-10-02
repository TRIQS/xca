#pragma once

#include <nda/nda.hpp>
#include <cppdlr/cppdlr.hpp>
#include <triqs/gfs.hpp>

#include <triqs_xca/atom_diag.hpp>
#include <triqs_xca/block_sparse/block_op.hpp>

using nda::dcomplex;

using cppdlr::imtime_ops;

using triqs_xca::atom_diag::triqs_atom_diag;

using triqs_xca::block_sparse::BlockDiagOpFun;

/**
 * @brief Utility function to get full Hamiltonian matrix from an AtomDiag object.
 * @param[in] ad AtomDiag object
 */
nda::matrix<dcomplex> get_full_h_atomic(const triqs_atom_diag &ad);

/**
 * @brief Evaluate a function on the DLR imaginary time grid at n_quad + 1 equidistant points
 * @param[in] itops imaginary time DLR operations
 * @param[in] f function values on the DLR imaginary time nodes
 * @param[in] n_quad number of equidistant intervals
 * @return function values on the equidistant grid
 */
nda::array<dcomplex, 3> eval_eq(imtime_ops &itops, nda::array_const_view<dcomplex, 3> f, int n_quad);

/**
 * @brief Get a dense tensor in the full Hilbert resticted to one atom_diag subspace
 * @param[in] tensor_full Full tensor in the Hilbert space
 * @param[in] subspace_index Index of the subspace
 * @param[in] ad AtomDiag object
 * @return tensor_subspace Tensor in the subspace
 */
nda::array<dcomplex, 3> get_tensor_in_atom_diag_subspace(nda::array_const_view<dcomplex, 3> tensor_full, int subspace_index,
                                                         triqs_atom_diag const &ad);

/**
 * @brief Scatter block-sparse data over the full Hilbert space, the inverse of get_tensor_in_atom_diag_subspace
 *
 * @details Block b occupies the rows and columns ad.get_fock_states(b) of the result, so that
 * get_tensor_in_atom_diag_subspace(get_tensor_in_full_hilbert_space(G, ad), b, ad) recovers block b. Entries outside
 * the blocks are left at zero, which lets a block-sparse result be compared against a dense one by subtracting the
 * two tensors directly: weight the block structure forbids shows up in the difference rather than being skipped.
 *
 * Block dimensions and positions are taken from ad, never from G, so blocks G flags as zero are placed correctly
 * even when their storage is empty.
 *
 * @param[in] G Block-diagonal operator whose blocks are ordered by atom_diag subspace
 * @param[in] ad AtomDiag object
 * @param[in] r Number of imaginary time nodes; if negative, taken from G.get_num_time_nodes(), which is only
 * available when at least one block of G is nonzero
 * @return tensor_full Tensor over the full Hilbert space
 */
nda::array<dcomplex, 3> get_tensor_in_full_hilbert_space(BlockDiagOpFun const &G, triqs_atom_diag const &ad, int r = -1);

/**
 * @brief Scatter block-sparse data over the full Hilbert space, the inverse of get_tensor_in_atom_diag_subspace
 * @param[in] G Block Green's function whose blocks are ordered by atom_diag subspace
 * @param[in] ad AtomDiag object
 * @return tensor_full Tensor over the full Hilbert space
 */
nda::array<dcomplex, 3> get_tensor_in_full_hilbert_space(triqs::gfs::block_gf_const_view<triqs::mesh::dlr_imtime> G, triqs_atom_diag const &ad);

/**
 * @brief Overloads for an owning block_gf and for a mutable view
 *
 * @details BlockDiagOpFun is implicitly constructible from a block_gf, so without an exact match for each of these
 * a call would be ambiguous between the two overloads above. Both forward to the block_gf_const_view overload.
 */
nda::array<dcomplex, 3> get_tensor_in_full_hilbert_space(triqs::gfs::block_gf<triqs::mesh::dlr_imtime> const &G, triqs_atom_diag const &ad);
nda::array<dcomplex, 3> get_tensor_in_full_hilbert_space(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G, triqs_atom_diag const &ad);
