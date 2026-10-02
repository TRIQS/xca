#pragma once

#include <cstddef>

#include <gtest/gtest.h>

#include <triqs/atom_diag/atom_diag.hpp>

#include <triqs_xca/atom_diag.hpp>
#include <triqs_xca/dense/atom_diag.hpp>
#include "test_utils/dense.hpp"

namespace triqs_xca::test_utils {

/**
 * @file parallel_atom_diag_check.hpp
 *
 * @brief Preconditions for comparing a block-sparse result against a dense one built on a different atom_diag
 *
 * @details The dense dynamical interaction path requires a single atom_diag subspace, so the dense reference for a partitioned block-sparse
 * model is built on a second atom_diag of the same Hamiltonian with sym_ops = {}. Both dense routes place the operators in the global Fock
 * basis, which does not depend on the partition, and the checks below assert that the two atom_diags represent the same model there.
 * Kept separate from block_sparse_utils so that the shared fixtures stay free of gtest.
 */
inline void assert_parallel_atom_diags(triqs::atom_diag::atom_diag<true> const &ad_bs, triqs::atom_diag::atom_diag<true> const &ad_flat,
                                       double tol = 1.0e-13) {
  ASSERT_EQ(ad_flat.n_subspaces(), 1) << "the dense reference needs a single-subspace atom_diag (build it with sym_ops = {})";
  ASSERT_GT(ad_bs.n_subspaces(), 1) << "the block-sparse side must be partitioned, or the test is vacuous";
  ASSERT_EQ(ad_bs.get_full_hilbert_space_dim(), ad_flat.get_full_hilbert_space_dim());
  ASSERT_EQ(ad_bs.get_fops().size(), ad_flat.get_fops().size());

  auto const &fock = ad_flat.get_fock_states(0);
  for (std::size_t i = 0; i < fock.size(); ++i)
    ASSERT_EQ(fock[i], i) << "the single-subspace Fock ordering is not the identity, so the dense dynamical-interaction path would place the "
                             "interaction operators in a different basis from the field operators";

  EXPECT_LE(nda::max_element(nda::abs(get_full_h_atomic(ad_bs) - get_full_h_atomic(ad_flat))), tol)
     << "the two atom_diags do not represent the same Hamiltonian";

  auto [Fs_bs, Fdags_bs]     = triqs_xca::dense::atom_diag::get_operators(ad_bs);
  auto [Fs_flat, Fdags_flat] = triqs_xca::dense::atom_diag::get_operators(ad_flat);
  EXPECT_LE(nda::max_element(nda::abs(Fs_bs - Fs_flat)), tol) << "the two atom_diags give different c operators in the Fock basis";
  EXPECT_LE(nda::max_element(nda::abs(Fdags_bs - Fdags_flat)), tol) << "the two atom_diags give different c^dag operators in the Fock basis";
  ASSERT_GT(nda::max_element(nda::abs(Fs_bs)), 0.5) << "non-vacuity: the comparison above must not be between two zero tensors";
}

} // namespace triqs_xca::test_utils
