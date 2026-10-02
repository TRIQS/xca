#include <optional>
#include <stdexcept>

#include "triqs_xca/block_sparse/atom_diag.hpp"

namespace triqs_xca::block_sparse::atom_diag {

  using triqs_xca::atom_diag::get_hamiltonian_block;
  using triqs_xca::atom_diag::get_operator_block;

  namespace {

    template <bool IsComplex> BlockOpSymSets get_operator_sym_sets_impl(const triqs_atom_diag_t<IsComplex> &ad, int n) {

      // Find like rows of c_connection (resp. cdag_connection), which correspond with annihilation (resp. creation) operators that have the same
      // sparsity pattern
      nda::vector<long> sym_set_labels(n);
      sym_set_labels = 0;
      int counter    = 0;
      for (int oidx = 0; oidx < n; ++oidx) { // fill each entry of sym_set_labels
        bool found_match = false;
        for (int oidx2 = 0; oidx2 < oidx; ++oidx2) { // compare full c_connection row against all previous operators
          bool same = true;
          for (int s = 0; s < ad.n_subspaces(); ++s) {
            if (ad.c_connection(oidx, s) != ad.c_connection(oidx2, s)) {
              same = false;
              break;
            }
          }
          if (same) {
            sym_set_labels(oidx) = sym_set_labels(oidx2);
            found_match          = true;
            break;
          }
        }
        if (not found_match) { // no matching operator found, so create new group
          sym_set_labels(oidx) = counter;
          counter              = counter + 1;
        }
      }
      std::set<int> unique_groups(sym_set_labels.begin(), sym_set_labels.end());
      std::size_t num_sym_sets = unique_groups.size();

      // First pass: count how many operators belong to each symmetry group
      std::vector<int> ops_per_group(num_sym_sets, 0);
      for (int oidx = 0; oidx < n; ++oidx) { ops_per_group[sym_set_labels[oidx]]++; }

      // Initialize operator blocks grouped by symmetry with proper dimensions
      std::vector<std::vector<nda::array<dcomplex, 3>>> c_blocks(num_sym_sets);
      std::vector<std::vector<nda::array<dcomplex, 3>>> cdag_blocks(num_sym_sets);

      // Initialize arrays for each symmetry group and subspace
      for (int gidx = 0; gidx < num_sym_sets; ++gidx) {
        c_blocks[gidx].resize(ad.n_subspaces());
        cdag_blocks[gidx].resize(ad.n_subspaces());

        for (int sidx = 0; sidx < ad.n_subspaces(); ++sidx) {
          // Determine dimensions for this subspace
          long cidx = -1, didx = -1;
          int dim_c_final = 0, dim_c_initial = 0;
          int dim_cdag_final = 0, dim_cdag_initial = 0;

          // Find a representative operator from this symmetry group to get dimensions
          for (int oidx = 0; oidx < n; ++oidx) {
            if (sym_set_labels[oidx] == gidx) {
              if (cidx == -1) {
                cidx = ad.c_connection(oidx, sidx);
                if (cidx >= 0) {
                  dim_c_final   = ad.get_fock_states(cidx).size();
                  dim_c_initial = ad.get_fock_states(sidx).size();
                }
              }
              if (didx == -1) {
                didx = ad.cdag_connection(oidx, sidx);
                if (didx >= 0) {
                  dim_cdag_final   = ad.get_fock_states(didx).size();
                  dim_cdag_initial = ad.get_fock_states(sidx).size();
                }
              }
              if (cidx >= 0 && didx >= 0) break;
            }
          }

          // Initialize arrays with proper dimensions
          if (cidx >= 0) { c_blocks[gidx][sidx] = nda::zeros<dcomplex>(ops_per_group[gidx], dim_c_final, dim_c_initial); }
          if (didx >= 0) { cdag_blocks[gidx][sidx] = nda::zeros<dcomplex>(ops_per_group[gidx], dim_cdag_final, dim_cdag_initial); }
        }
      }

      // Second pass: fill the arrays
      std::vector<int> op_count_per_group(num_sym_sets, 0);

      for (int oidx = 0; oidx < n; ++oidx) {
        int gidx            = sym_set_labels[oidx];
        int op_idx_in_group = op_count_per_group[gidx];

        for (int sidx = 0; sidx < ad.n_subspaces(); ++sidx) {
          // Handle annihilation operator
          long cidx = ad.c_connection(oidx, sidx);
          if (cidx >= 0) {
            // Get the operator block between the two subspaces
            auto c_block = *get_operator_block(ad, oidx, false, sidx);
            for (int i = 0; i < c_block.extent(0); ++i) {
              for (int j = 0; j < c_block.extent(1); ++j) { c_blocks[gidx][sidx](op_idx_in_group, i, j) = c_block(i, j); }
            }
          }

          // Handle creation operator
          long didx = ad.cdag_connection(oidx, sidx);
          if (didx >= 0) {
            // Get the operator block between the two subspaces
            auto cdag_block = *get_operator_block(ad, oidx, true, sidx);
            for (int i = 0; i < cdag_block.extent(0); ++i) {
              for (int j = 0; j < cdag_block.extent(1); ++j) { cdag_blocks[gidx][sidx](op_idx_in_group, i, j) = cdag_block(i, j); }
            }
          }
        }

        op_count_per_group[gidx]++;
      }

      // Fill in BlockOpSymSet objects
      nda::array<int, 2> F_block_inds     = nda::zeros<int>(num_sym_sets, ad.n_subspaces()),
                         F_dag_block_inds = nda::zeros<int>(num_sym_sets, ad.n_subspaces());
      auto filled_F_block_inds            = nda::zeros<int>(n);
      for (int i = 0; i < n; i++) {
        long label = sym_set_labels(i);
        if (filled_F_block_inds(label) == 0) {
          for (int j = 0; j < ad.n_subspaces(); ++j) { F_block_inds(label, j) = ad.c_connection(i, j); }
          filled_F_block_inds(label) = 1;
        }
      }
      auto filled_F_dag_block_inds = nda::zeros<int>(n);
      for (int i = 0; i < n; i++) {
        long label = sym_set_labels(i);
        if (filled_F_dag_block_inds(label) == 0) {
          for (int j = 0; j < ad.n_subspaces(); ++j) { F_dag_block_inds(label, j) = ad.cdag_connection(i, j); }
          filled_F_dag_block_inds(label) = 1;
        }
      }

      std::vector<BlockOpSymSet> F_sym_vec, F_dag_sym_vec;
      for (int i = 0; i < num_sym_sets; i++) {
        F_sym_vec.emplace_back(F_block_inds(i, cppdlr::_), c_blocks[i]);
        F_dag_sym_vec.emplace_back(F_dag_block_inds(i, cppdlr::_), cdag_blocks[i]);
      }
      return BlockOpSymSets{std::move(F_sym_vec), std::move(F_dag_sym_vec), std::move(sym_set_labels)};
    }

    template <bool IsComplex>
    std::tuple<BlockOpSymQuartet, nda::vector<int>> get_operators_impl(const triqs_atom_diag_t<IsComplex> &ad,
                                                                       nda::array_const_view<dcomplex, 3> hyb_coeffs) {
      // the orbital count is taken from the coefficients, which may cover only the first n fundamental operators
      auto sets = get_operator_sym_sets_impl(ad, static_cast<int>(hyb_coeffs.extent(1)));
      BlockOpSymQuartet Fq(sets.Fs, sets.F_dags, hyb_coeffs, sets.sym_set_labels);
      return std::make_tuple(Fq, sets.sym_set_labels);
    }

  } // namespace

  using nda::linalg::matmul;

  using cppdlr::_;

  namespace {
    // n indexes ad.c_connection, so an out-of-range value has to be caught here
    void check_sym_set_orbital_count(int n, long n_fops) {
      if (n < 0 || n > n_fops)
        throw std::invalid_argument("get_operator_sym_sets: asked for " + std::to_string(n) + " orbitals, but the atom_diag has "
                                    + std::to_string(n_fops) + " fundamental operators");
    }
  } // namespace

  BlockOpSymSets get_operator_sym_sets(const triqs_atom_diag_t<true> &ad, int n) {
    check_sym_set_orbital_count(n, ad.get_fops().size());
    return get_operator_sym_sets_impl(ad, n);
  }

  BlockOpSymSets get_operator_sym_sets(const triqs_atom_diag_t<false> &ad, int n) {
    check_sym_set_orbital_count(n, ad.get_fops().size());
    return get_operator_sym_sets_impl(ad, n);
  }

  std::tuple<BlockOpSymQuartet, nda::vector<int>> get_operators(const triqs_atom_diag_t<true> &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs) {
    return get_operators_impl(ad, hyb_coeffs);
  }

  std::tuple<BlockOpSymQuartet, nda::vector<int>> get_operators(const triqs_atom_diag_t<false> &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs) {
    return get_operators_impl(ad, hyb_coeffs);
  }

} // namespace triqs_xca::block_sparse::atom_diag
