#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <map>
#include <set>

#include <triqs/arrays.hpp>
#include <triqs/operators.hpp>
#include <triqs/atom_diag.hpp>
#include <h5/h5.hpp>

using namespace triqs;
using namespace triqs::arrays;
using namespace triqs::operators;
using namespace triqs::atom_diag;
using namespace triqs::hilbert_space;

// Utility function to get full Hamiltonian matrix
matrix<double> get_full_h_atomic(const triqs::atom_diag::atom_diag<false> &ad) {
  int dim              = ad.get_full_hilbert_space_dim();
  matrix<double> H_mat = zeros<double>(dim, dim);

  for (int sidx = 0; sidx < ad.n_subspaces(); ++sidx) {
    auto U        = ad.get_unitary_matrix(sidx);
    auto energies = ad.get_energies()[sidx];

    // Create diagonal energy matrix
    matrix<double> H_block_diag = zeros<double>(energies.size(), energies.size());
    for (int i = 0; i < energies.size(); ++i) { H_block_diag(i, i) = energies[i] + ad.get_gs_energy(); }

    // Transform to Fock basis: H_block = U @ H_block_diag @ U^T
    matrix<double> H_block = U * H_block_diag * transpose(U);

    // Get Fock states for this subspace
    auto fock_states = ad.get_fock_states(sidx);

    // Copy block into full matrix
    for (int i = 0; i < fock_states.size(); ++i) {
      for (int j = 0; j < fock_states.size(); ++j) { H_mat(fock_states[i], fock_states[j]) = H_block(i, j); }
    }
  }

  return H_mat;
}

// Utility function to get full operator matrix
matrix<double> get_full_operator_matrix(const triqs::atom_diag::atom_diag<false> &ad, int oidx, bool is_creation) {
  int dim               = ad.get_full_hilbert_space_dim();
  matrix<double> op_mat = zeros<double>(dim, dim);

  for (int s1 = 0; s1 < ad.n_subspaces(); ++s1) {
    // Get connection to target subspace
    long s2 = is_creation ? ad.cdag_connection(oidx, s1) : ad.c_connection(oidx, s1);
    if (s2 < 0) continue;

    // Get matrix block in eigenbasis
    auto block_mat = is_creation ? ad.cdag_matrix(oidx, s1) : ad.c_matrix(oidx, s1);

    // Transform to Fock basis
    auto U1                       = ad.get_unitary_matrix(s1);
    auto U2                       = ad.get_unitary_matrix(s2);
    matrix<double> block_mat_fock = U2 * block_mat * transpose(U1);

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

int main() {
  try {
    // Parameters to tune
    int norb             = 5;
    bool all_sym         = true; // True for all symmetries, False for just particle number symmetry
    std::string h5_fname = all_sym ? "spin_flip_fermion_all_sym.h5" : "spin_flip_fermion.h5";

    // Construct particle number operator
    triqs::operators::many_body_operator_real N;
    for (int kap = 0; kap < norb; ++kap) { N += n("up", kap) + n("do", kap); }

    std::vector<triqs::operators::many_body_operator_real> sym_ops = {N};

    // Fixed parameters
    double beta = 2.0;
    double mu   = 0.25;
    double U    = 1.0;
    double V    = 0.1;

    // Construct Hamiltonian
    triqs::operators::many_body_operator_real H;
    fundamental_operator_set fop_set;

    for (int i = 0; i < norb; ++i) {
      H += U * n("up", i) * n("do", i) + mu * (n("up", i) + n("do", i)) + V * (c_dag("up", i) * c("do", i) + c_dag("do", i) * c("up", i));
      fop_set.insert("do", i);
    }
    for (int i = 0; i < norb; ++i) { fop_set.insert("up", i); }

    // Create AtomDiag object
    triqs::atom_diag::atom_diag<false> ad;
    if (all_sym) {
      ad = triqs::atom_diag::atom_diag<false>(H, fop_set);
    } else {
      ad = triqs::atom_diag::atom_diag<false>(H, fop_set, sym_ops);
    }

    // Find symmetry groups for operators
    std::vector<int> row_groups(2 * norb, 0);
    std::iota(row_groups.begin(), row_groups.end(), 0); // Initialize with unique values

    int counter = 1;
    for (int s = 0; s < ad.n_subspaces(); ++s) {
      int oidx0        = 1;
      bool found_oidx0 = false;

      // Find first ungrouped operator
      for (int i = 0; i < 2 * norb; ++i) {
        if (!found_oidx0 && row_groups[i] == i) {
          oidx0       = i;
          found_oidx0 = true;
          break;
        }
      }

      for (int oidx = oidx0; oidx < 2 * norb; ++oidx) {
        bool found_pair = false;

        for (int oidx2 = 0; oidx2 < oidx; ++oidx2) {
          try {
            long x = ad.c_connection(oidx, s);
            long y = ad.c_connection(oidx2, s);

            if (!found_pair && x == y) {
              row_groups[oidx] = row_groups[oidx2];
              found_pair       = true;
            }
          } catch (const std::exception &e) { std::cout << "Failed at oidx=" << oidx << ", s=" << s << ": " << e.what() << std::endl; }
        }

        if (!found_pair) { row_groups[oidx] = counter++; }
      }
    }

    // Count number of symmetry sets
    std::set<int> unique_groups(row_groups.begin(), row_groups.end());
    int num_sym_sets = unique_groups.size();

    std::cout << "Number of symmetry sets: " << num_sym_sets << std::endl;

    // Get full Hamiltonian matrix
    auto H_mat = get_full_h_atomic(ad);

    // Create permutation based on Fock state ordering
    std::vector<long> H_perm;
    std::vector<matrix<double>> H_mat_blocks;
    std::vector<long> H_mat_block_inds(ad.n_subspaces());

    for (int s = 0; s < ad.n_subspaces(); ++s) {
      auto fock_states = ad.get_fock_states(s);
      for (auto state : fock_states) { H_perm.push_back(state); }

      // Extract block from full matrix
      matrix<double> H_block = zeros<double>(fock_states.size(), fock_states.size());
      for (int i = 0; i < fock_states.size(); ++i) {
        for (int j = 0; j < fock_states.size(); ++j) { H_block(i, j) = H_mat(fock_states[i], fock_states[j]); }
      }
      H_mat_blocks.push_back(H_block);

      // Check if block is zero
      double max_elem = 0.0;
      for (int i = 0; i < H_block.extent(0); ++i) {
        for (int j = 0; j < H_block.extent(1); ++j) { max_elem = std::max(max_elem, std::abs(H_block(i, j))); }
      }
      H_mat_block_inds[s] = (max_elem < 1e-16) ? -1 : s;
    }

    // Create permuted Hamiltonian matrix
    matrix<double> H_mat_perm = zeros<double>(H_perm.size(), H_perm.size());
    for (int i = 0; i < H_perm.size(); ++i) {
      for (int j = 0; j < H_perm.size(); ++j) { H_mat_perm(i, j) = H_mat(H_perm[i], H_perm[j]); }
    }

    // Initialize operator blocks grouped by symmetry
    std::vector<std::vector<std::vector<matrix<double>>>> c_blocks(num_sym_sets);
    std::vector<std::vector<std::vector<matrix<double>>>> cdag_blocks(num_sym_sets);

    for (int gidx = 0; gidx < num_sym_sets; ++gidx) {
      c_blocks[gidx].resize(ad.n_subspaces());
      cdag_blocks[gidx].resize(ad.n_subspaces());
    }

    // Save blocks of creation and annihilation operators
    for (int oidx = 0; oidx < 2 * norb; ++oidx) {
      for (int sidx = 0; sidx < ad.n_subspaces(); ++sidx) {
        long cidx = ad.c_connection(oidx, sidx);
        if (cidx >= 0) {
          auto fock_final   = ad.get_fock_states(cidx);
          auto fock_initial = ad.get_fock_states(sidx);

          // Get full operator matrix and extract block
          auto c_full            = get_full_operator_matrix(ad, oidx, false);
          matrix<double> c_block = zeros<double>(fock_final.size(), fock_initial.size());
          for (int i = 0; i < fock_final.size(); ++i) {
            for (int j = 0; j < fock_initial.size(); ++j) { c_block(i, j) = c_full(fock_final[i], fock_initial[j]); }
          }
          c_blocks[row_groups[oidx]][sidx].push_back(c_block);
        }

        long didx = ad.cdag_connection(oidx, sidx);
        if (didx >= 0) {
          auto fock_final   = ad.get_fock_states(didx);
          auto fock_initial = ad.get_fock_states(sidx);

          // Get full operator matrix and extract block
          auto cdag_full            = get_full_operator_matrix(ad, oidx, true);
          matrix<double> cdag_block = zeros<double>(fock_final.size(), fock_initial.size());
          for (int i = 0; i < fock_final.size(); ++i) {
            for (int j = 0; j < fock_initial.size(); ++j) { cdag_block(i, j) = cdag_full(fock_final[i], fock_initial[j]); }
          }
          cdag_blocks[row_groups[oidx]][sidx].push_back(cdag_block);
        }
      }
    }

    // Save dense creation and annihilation operators
    std::vector<matrix<double>> c_dense(2 * norb);
    std::vector<matrix<double>> cdag_dense(2 * norb);

    for (int oidx = 0; oidx < 2 * norb; ++oidx) {
      auto c_full    = get_full_operator_matrix(ad, oidx, false);
      auto cdag_full = get_full_operator_matrix(ad, oidx, true);

      // Apply permutation
      c_dense[oidx]    = zeros<double>(H_perm.size(), H_perm.size());
      cdag_dense[oidx] = zeros<double>(H_perm.size(), H_perm.size());

      for (int i = 0; i < H_perm.size(); ++i) {
        for (int j = 0; j < H_perm.size(); ++j) {
          c_dense[oidx](i, j)    = c_full(H_perm[i], H_perm[j]);
          cdag_dense[oidx](i, j) = cdag_full(H_perm[i], H_perm[j]);
        }
      }
    }

    std::cout << "Number of blocks: " << ad.n_subspaces() << std::endl;

    // Print block sizes
    std::map<int, int> block_size_counts;
    for (int i = 0; i < ad.n_subspaces(); ++i) {
      int size = ad.get_subspace_dim(i);
      block_size_counts[size]++;
    }
    std::cout << "Block sizes: ";
    for (const auto &pair : block_size_counts) { std::cout << pair.first << ":" << pair.second << " "; }
    std::cout << std::endl;

    // Save to HDF5
    std::string full_path = "/home/paco/feynman/soehyb/test/c++/h5/" + h5_fname;
    {
      h5::file file(full_path, 'w');
      h5::group gr = file;

      h5_write(gr, "norb", norb);
      h5_write(gr, "num_blocks", ad.n_subspaces());
      h5_write(gr, "ad", ad);
      h5_write(gr, "H_mat_blocks", H_mat_blocks);
      h5_write(gr, "H_mat_block_inds", H_mat_block_inds);
      h5_write(gr, "c_blocks", c_blocks);
      h5_write(gr, "cdag_blocks", cdag_blocks);
      h5_write(gr, "H_mat_dense", H_mat_perm);
      h5_write(gr, "c_dense", c_dense);
      h5_write(gr, "cdag_dense", cdag_dense);
      h5_write(gr, "num_sym_sets", num_sym_sets);
      h5_write(gr, "sym_set_labels", row_groups);
    }

    std::cout << "Successfully wrote data to " << full_path << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
