#include "triqs_xca/atom_diag_utils.hpp"

namespace triqs_xca::atom_diag {

  using nda::linalg::matmul;

  using cppdlr::_;

  nda::matrix<double> get_full_h_atomic(const triqs_atom_diag &ad) {
    int dim    = ad.get_full_hilbert_space_dim();
    auto H_mat = nda::zeros<double>(dim, dim);

    for (int sidx = 0; sidx < ad.n_subspaces(); ++sidx) {
      auto U        = ad.get_unitary_matrix(sidx);
      auto energies = ad.get_energies()[sidx];

      // Create diagonal energy matrix
      nda::matrix<double> H_block_diag = nda::zeros<double>(energies.size(), energies.size());
      for (int i = 0; i < energies.size(); ++i) { H_block_diag(i, i) = energies[i] + ad.get_gs_energy(); }

      // Transform to Fock basis: H_block = U @ H_block_diag @ U^T
      nda::matrix<double> H_block = U * H_block_diag * nda::transpose(U);

      // Get Fock states for this subspace
      auto fock_states = ad.get_fock_states(sidx);

      // Copy block into full matrix
      for (int i = 0; i < fock_states.size(); ++i) {
        for (int j = 0; j < fock_states.size(); ++j) { H_mat(fock_states[i], fock_states[j]) = H_block(i, j); }
      }
    }

    return H_mat;
  }

  nda::matrix<double> get_full_operator_matrix(const triqs_atom_diag &ad, int oidx, bool is_creation) {
    int dim                    = ad.get_full_hilbert_space_dim();
    nda::matrix<double> op_mat = nda::zeros<double>(dim, dim);

    for (int s1 = 0; s1 < ad.n_subspaces(); ++s1) {
      // Get connection to target subspace
      long s2 = is_creation ? ad.cdag_connection(oidx, s1) : ad.c_connection(oidx, s1);
      if (s2 < 0) continue;

      // Get matrix block in eigenbasis
      auto block_mat = is_creation ? ad.cdag_matrix(oidx, s1) : ad.c_matrix(oidx, s1);

      // Transform to Fock basis
      auto U1                            = ad.get_unitary_matrix(s1);
      auto U2                            = ad.get_unitary_matrix(s2);
      nda::matrix<double> block_mat_fock = U2 * block_mat * nda::transpose(U1);

      // Get Fock states
      auto f1 = ad.get_fock_states(s1);
      auto f2 = ad.get_fock_states(s2);

      // Copy block into full matrix
      for (int i = 0; i < f2.size(); ++i) {
        for (int j = 0; j < f1.size(); ++j) { op_mat(f2[i], f1[j]) = block_mat_fock(i, j); }
      }
    }

    return op_mat;
  }

  std::tuple<std::vector<nda::array<double, 2>>, nda::vector<long>> get_hamiltonian_blocks(const triqs_atom_diag &ad) {
    // Get full Hamiltonian matrix
    auto H_mat = get_full_h_atomic(ad);

    // Create permutation based on Fock state ordering
    std::vector<unsigned long> H_perm;
    std::vector<nda::array<double, 2>> H_blocks;
    nda::vector<long> H_block_inds(ad.n_subspaces());

    for (int s = 0; s < ad.n_subspaces(); ++s) {
      auto fock_states = ad.get_fock_states(s);
      for (auto state : fock_states) { H_perm.push_back(state); }

      // Extract block from full matrix
      nda::array<double, 2> H_block = nda::zeros<double>(fock_states.size(), fock_states.size());
      for (int i = 0; i < fock_states.size(); ++i) {
        for (int j = 0; j < fock_states.size(); ++j) { H_block(i, j) = H_mat(fock_states[i], fock_states[j]); }
      }
      H_blocks.push_back(H_block);

      // Check if block is zero
      double max_elem = 0.0;
      for (int i = 0; i < H_block.extent(0); ++i) {
        for (int j = 0; j < H_block.extent(1); ++j) { max_elem = std::max(max_elem, std::abs(H_block(i, j))); }
      }
      H_block_inds(s) = (max_elem < 1e-16) ? -1 : 0;
    }
    return std::make_tuple(H_blocks, H_block_inds);
  }

  template <typename T>
  std::vector<nda::array<T, 3>> H_to_atom_prop_blocks(std::vector<nda::array<double, 2>> &H_blocks, nda::vector_const_view<long> H_block_inds,
                                                      double beta, imtime_ops &itops) {
    int r                    = itops.rank();
    double tr_exp_minusbetaH = 0;
    std::vector<nda::array<double, 1>> H_evals(H_blocks.size());
    std::vector<nda::array<double, 2>> H_evecs(H_blocks.size());
    for (int i = 0; i < H_block_inds.size(); ++i) {
      if (H_block_inds(i) != -1) {
        if (H_blocks[i].extent(0) == 1) {
          H_evals[i] = nda::array<double, 1>{H_blocks[i](0, 0)};
          H_evecs[i] = nda::array<double, 2>{{1}};
        } else {
          auto H_block_eig = nda::linalg::eigh(H_blocks[i]);
          H_evals[i]       = std::get<0>(H_block_eig);
          H_evecs[i]       = std::get<1>(H_block_eig);
        }
        tr_exp_minusbetaH += nda::sum(exp(-beta * H_evals[i]));
      } else {
        H_evals[i] = nda::zeros<double>(H_blocks[i].extent(0));
        H_evecs[i] = nda::eye<double>(H_blocks[i].extent(0));
        tr_exp_minusbetaH += 1.0 * H_blocks[i].extent(0); // 0 entry in the diagonal
      }
    }

    auto eta_0      = nda::log(tr_exp_minusbetaH) / beta;
    auto dlr_it     = itops.get_itnodes();
    auto dlr_it_abs = cppdlr::rel2abs(dlr_it);
    std::vector<nda::array<T, 3>> ap_blocks(H_block_inds.size());
    for (int i = 0; i < H_block_inds.size(); ++i) {
      ap_blocks[i] = nda::array<T, 2>(r, H_blocks[i].extent(0), H_blocks[i].extent(1));
      auto Gt_temp = nda::make_regular(0 * H_blocks[i]);
      for (int t = 0; t < r; t++) {
        for (int j = 0; j < H_blocks[i].extent(0); j++) { Gt_temp(j, j) = -exp(-beta * dlr_it_abs(t) * (H_evals[i](j) + eta_0)); }
        ap_blocks[i](t, _, _) = matmul(H_evecs[i], matmul(Gt_temp, nda::transpose(H_evecs[i])));
      }
    }

    return ap_blocks;
  }

  BlockDiagOpFun ad_to_atom_prop(const triqs_atom_diag &ad, double beta, imtime_ops &itops) {
    // Get Hamiltonian blocks and block indices
    auto [H_blocks, H_block_inds] = get_hamiltonian_blocks(ad);

    // Compute atomic propagator blocks
    std::vector<nda::array<dcomplex, 3>> ap_blocks = H_to_atom_prop_blocks<dcomplex>(H_blocks, H_block_inds, beta, itops);

    // Create BlockDiagOpFun
    auto zero_block_indices = nda::ones<int>(H_block_inds.size());
    return {ap_blocks, zero_block_indices};
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> ad_to_atom_prop(const triqs_atom_diag &ad, double beta, double Lambda, double eps) {
    // Get Hamiltonian blocks and block indices
    auto [H_blocks, H_block_inds] = get_hamiltonian_blocks(ad);

    // Compute atomic propagator blocks
    auto dlr_rf                                    = cppdlr::build_dlr_rf(Lambda, eps);
    auto itops                                     = imtime_ops(Lambda, dlr_rf);
    std::vector<nda::array<dcomplex, 3>> ap_blocks = H_to_atom_prop_blocks<dcomplex>(H_blocks, H_block_inds, beta, itops);

    // Create vector of gf<dlr_imtime>
    std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> gf_blocks(H_block_inds.size());
    triqs::mesh::dlr_imtime tau_mesh(beta, triqs::mesh::Fermion, Lambda / beta, eps);
    for (int i = 0; i < H_block_inds.size(); ++i) { gf_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, ap_blocks[i]); }
    return {gf_blocks};
  }

  std::tuple<BlockOpSymQuartet, nda::vector<int>> get_operators(const triqs_atom_diag &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs) {

    // Find like rows of c_connection (resp. cdag_connection), which correspond with annihilation (resp. creation) operators that have the same
    // sparsity pattern
    int n = hyb_coeffs.extent(1);
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
          auto fock_final   = ad.get_fock_states(cidx);
          auto fock_initial = ad.get_fock_states(sidx);

          // Get full operator matrix and extract block
          auto c_full = get_full_operator_matrix(ad, oidx, false);
          for (int i = 0; i < fock_final.size(); ++i) {
            for (int j = 0; j < fock_initial.size(); ++j) { c_blocks[gidx][sidx](op_idx_in_group, i, j) = c_full(fock_final[i], fock_initial[j]); }
          }
        }

        // Handle creation operator
        long didx = ad.cdag_connection(oidx, sidx);
        if (didx >= 0) {
          auto fock_final   = ad.get_fock_states(didx);
          auto fock_initial = ad.get_fock_states(sidx);

          // Get full operator matrix and extract block
          auto cdag_full = get_full_operator_matrix(ad, oidx, true);
          for (int i = 0; i < fock_final.size(); ++i) {
            for (int j = 0; j < fock_initial.size(); ++j) {
              cdag_blocks[gidx][sidx](op_idx_in_group, i, j) = cdag_full(fock_final[i], fock_initial[j]);
            }
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
      F_sym_vec.emplace_back(F_block_inds(i, _), c_blocks[i]);
      F_dag_sym_vec.emplace_back(F_dag_block_inds(i, _), cdag_blocks[i]);
    }
    BlockOpSymQuartet Fq(F_sym_vec, F_dag_sym_vec, hyb_coeffs, sym_set_labels);
    return std::make_tuple(Fq, sym_set_labels);
  }

  std::tuple<nda::array<dcomplex, 3>, nda::array<dcomplex, 3>> get_operators_dense(const triqs_atom_diag &ad) {

    int norb = ad.get_fops().size();
    int N    = ad.get_full_hilbert_space_dim();

    // Get full operator matrices
    nda::array<dcomplex, 3> Fs{norb, N, N};
    nda::array<dcomplex, 3> Fdags{norb, N, N};

    for (int oidx = 0; oidx < norb; ++oidx) {
      Fs(oidx, _, _)    = get_full_operator_matrix(ad, oidx, false);
      Fdags(oidx, _, _) = get_full_operator_matrix(ad, oidx, true);
    }
    return {Fs, Fdags};
  }

  DenseFSet get_operators_dense(const triqs_atom_diag &ad, nda::array_const_view<dcomplex, 3> hyb_coeffs) {
    auto [Fs, Fdags] = get_operators_dense(ad);
    return {Fs, Fdags, hyb_coeffs};
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

} // namespace triqs_xca::atom_diag
