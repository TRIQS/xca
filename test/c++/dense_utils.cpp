#include <optional>
#include <stdexcept>

#include "dense_utils.hpp"

using cppdlr::_;

nda::matrix<dcomplex> get_full_h_atomic(const triqs_atom_diag &ad) {
  int dim    = ad.get_full_hilbert_space_dim();
  auto H_mat = nda::zeros<dcomplex>(dim, dim);

  for (int sidx = 0; sidx < ad.n_subspaces(); ++sidx) {
    auto H_block = triqs_xca::atom_diag::get_hamiltonian_block(ad, sidx);

    // Get Fock states for this subspace
    auto fock_states = ad.get_fock_states(sidx);

    // Copy block into full matrix
    for (int i = 0; i < fock_states.size(); ++i) {
      for (int j = 0; j < fock_states.size(); ++j) { H_mat(fock_states[i], fock_states[j]) = H_block(i, j); }
    }
  }

  return H_mat;
}

nda::array<dcomplex, 3> eval_eq(imtime_ops &itops, nda::array_const_view<dcomplex, 3> f, int n_quad) {
  auto fc    = itops.vals2coefs(f);
  auto it_eq = cppdlr::eqptsrel(n_quad + 1);
  auto f_eq  = nda::array<dcomplex, 3>(n_quad + 1, f.extent(1), f.extent(2));
  for (int i = 0; i <= n_quad; i++) { f_eq(i, _, _) = itops.coefs2eval(fc, it_eq(i)); }
  return f_eq;
}

nda::array<dcomplex, 3> get_tensor_in_atom_diag_subspace(nda::array_const_view<dcomplex, 3> tensor_full, int subspace_index,
                                                         triqs_atom_diag const &ad) {
  // Permute a tensor from the full Hilbert space to the Fock state ordered basis

  int r = tensor_full.extent(0);

  std::vector<unsigned long> H_perm;
  auto fock_states = ad.get_fock_states(subspace_index);
  for (auto state : fock_states) H_perm.push_back(state);

  int N_sub                               = fock_states.size();
  nda::array<dcomplex, 3> tensor_subspace = nda::zeros<dcomplex>(r, N_sub, N_sub);

  for (int t = 0; t < r; ++t) {
    for (int i = 0; i < N_sub; ++i) {
      for (int j = 0; j < N_sub; ++j) { tensor_subspace(t, i, j) = tensor_full(t, H_perm[i], H_perm[j]); }
    }
  }

  return tensor_subspace;
}

namespace {

  /**
   * @brief Shared scatter behind both get_tensor_in_full_hilbert_space overloads
   *
   * @param[in] n_block_cols Number of blocks, which must agree with ad.n_subspaces()
   * @param[in] r Number of imaginary time nodes
   * @param[in] ad AtomDiag object supplying the block dimensions and their positions
   * @param[in] block_of Returns the data of block b, or nullopt if that block is zero
   */
  template <typename BlockFn>
  nda::array<dcomplex, 3> get_tensor_in_full_hilbert_space_impl(int n_block_cols, int r, triqs_atom_diag const &ad, BlockFn block_of) {
    if (n_block_cols != ad.n_subspaces()) {
      throw std::invalid_argument("get_tensor_in_full_hilbert_space: got " + std::to_string(n_block_cols) + " blocks, but ad has "
                                  + std::to_string(ad.n_subspaces()) + " subspaces");
    }

    int dim                             = ad.get_full_hilbert_space_dim();
    nda::array<dcomplex, 3> tensor_full = nda::zeros<dcomplex>(r, dim, dim);

    for (int b = 0; b < n_block_cols; ++b) {
      auto block = block_of(b);
      // A zero block leaves its rows and columns at zero. Its position is known from ad, so nothing
      // is read from the block-sparse object, whose storage for a zero block may be empty.
      if (!block.has_value()) continue;

      auto fock_states = ad.get_fock_states(b);
      int N_sub        = fock_states.size();

      if (block->extent(1) != N_sub || block->extent(2) != N_sub) {
        throw std::invalid_argument("get_tensor_in_full_hilbert_space: block " + std::to_string(b) + " is " + std::to_string(block->extent(1)) + "x"
                                    + std::to_string(block->extent(2)) + ", but subspace " + std::to_string(b) + " of ad has dimension "
                                    + std::to_string(N_sub));
      }
      if (block->extent(0) != r) {
        throw std::invalid_argument("get_tensor_in_full_hilbert_space: block " + std::to_string(b) + " has " + std::to_string(block->extent(0))
                                    + " time nodes, expected " + std::to_string(r));
      }

      for (int t = 0; t < r; ++t) {
        for (int i = 0; i < N_sub; ++i) {
          for (int j = 0; j < N_sub; ++j) { tensor_full(t, fock_states[i], fock_states[j]) = (*block)(t, i, j); }
        }
      }
    }

    return tensor_full;
  }

} // namespace

nda::array<dcomplex, 3> get_tensor_in_full_hilbert_space(BlockDiagOpFun const &G, triqs_atom_diag const &ad, int r) {
  if (r < 0) {
    r = G.get_num_time_nodes();
    if (r == 0) {
      throw std::invalid_argument(
         "get_tensor_in_full_hilbert_space: every block of G is zero, so the number of time nodes cannot be inferred; "
         "pass it as the r argument");
    }
  }

  return get_tensor_in_full_hilbert_space_impl(G.get_num_block_cols(), r, ad, [&G](int b) -> std::optional<nda::array_const_view<dcomplex, 3>> {
    if (G.get_zero_block_index(b) == -1) return std::nullopt;
    return G.get_block(b);
  });
}

nda::array<dcomplex, 3> get_tensor_in_full_hilbert_space(triqs::gfs::block_gf_const_view<triqs::mesh::dlr_imtime> G, triqs_atom_diag const &ad) {
  if (G.size() == 0) throw std::invalid_argument("get_tensor_in_full_hilbert_space: G has no blocks");

  int r = G[0].data().extent(0);
  return get_tensor_in_full_hilbert_space_impl(G.size(), r, ad,
                                               [&G](int b) -> std::optional<nda::array_const_view<dcomplex, 3>> { return G[b].data(); });
}

nda::array<dcomplex, 3> get_tensor_in_full_hilbert_space(triqs::gfs::block_gf<triqs::mesh::dlr_imtime> const &G, triqs_atom_diag const &ad) {
  return get_tensor_in_full_hilbert_space(triqs::gfs::block_gf_const_view<triqs::mesh::dlr_imtime>{G}, ad);
}

nda::array<dcomplex, 3> get_tensor_in_full_hilbert_space(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G, triqs_atom_diag const &ad) {
  return get_tensor_in_full_hilbert_space(triqs::gfs::block_gf_const_view<triqs::mesh::dlr_imtime>{G}, ad);
}
