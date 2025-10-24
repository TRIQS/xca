#include "block_sparse.hpp"
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/utils.hpp>
#include <h5/format.hpp>
#include <h5/group.hpp>
#include <iostream>
#include <cppdlr/dlr_kernels.hpp>
#include <nda/algorithms.hpp>
#include <nda/declarations.hpp>
#include <nda/basic_functions.hpp>
#include <nda/layout_transforms.hpp>
#include <nda/linalg/eigenelements.hpp>
#include <nda/print.hpp>
#include <ostream>
#include <string>
#include <vector>
#include <stdexcept>

using namespace nda;

/////////////// BlockDiagOpFun (BDOF) class ///////////////
BlockDiagOpFun::BlockDiagOpFun(std::vector<nda::array<dcomplex, 3>> &blocks, nda::vector_const_view<int> zero_block_indices)
   : blocks(blocks), num_block_cols(blocks.size()), zero_block_indices(zero_block_indices) {}

BlockDiagOpFun::BlockDiagOpFun(int r, nda::vector_const_view<int> block_sizes) : num_block_cols(block_sizes.size()) {

  std::vector<nda::array<dcomplex, 3>> blocks(num_block_cols);
  zero_block_indices = nda::make_regular(-1 * nda::ones<int>(num_block_cols));
  for (int i = 0; i < num_block_cols; i++) { blocks[i] = nda::zeros<dcomplex>(r, block_sizes[i], block_sizes[i]); }
  this->blocks = blocks;
}

BlockDiagOpFun &BlockDiagOpFun::operator+=(const BlockDiagOpFun &G) {
  // BlockDiagOpFun addition-assignment operator

  for (int i = 0; i < this->num_block_cols; i++) {
    if (zero_block_indices(i) == -1) {
      if (G.get_zero_block_index(i) != -1) {
        this->blocks[i]       = G.blocks[i];
        zero_block_indices(i) = 0;
      }
    } else {
      if (G.get_zero_block_index(i) != -1) { this->blocks[i] += G.blocks[i]; }
    }
  }
  return *this;
}

void BlockDiagOpFun::set_blocks(std::vector<nda::array<dcomplex, 3>> &blocks) {

  this->blocks       = blocks;
  num_block_cols     = blocks.size();
  zero_block_indices = nda::zeros<int>(num_block_cols);
}

void BlockDiagOpFun::set_block(int i, nda::array_const_view<dcomplex, 3> block) {
  blocks[i]             = block;
  zero_block_indices(i) = 0;
}

void BlockDiagOpFun::set_zero_block_indices() {
  // Set zero_block_indices according to current blocks
  for (int i = 0; i < num_block_cols; i++) {
    if (nda::max_element(nda::abs(blocks[i])) < 1e-16) {
      zero_block_indices(i) = -1; // mark block as zero
    } else {
      zero_block_indices(i) = 0; // mark block as non-zero
    }
  }
}

const std::vector<nda::array<dcomplex, 3>> &BlockDiagOpFun::get_blocks() const { return blocks; }

nda::array_const_view<dcomplex, 3> BlockDiagOpFun::get_block(int i) const { return blocks[i]; }

nda::vector<int> BlockDiagOpFun::get_block_sizes() const {
  nda::vector<int> block_sizes(num_block_cols);
  for (int i = 0; i < num_block_cols; i++) { block_sizes(i) = blocks[i].shape(1); }
  return block_sizes;
}

int BlockDiagOpFun::get_block_size(int i) const { return blocks[i].shape(1); }

int BlockDiagOpFun::get_max_block_size() const {
  int max_block_size = 0;
  for (int i = 0; i < num_block_cols; i++) { max_block_size = std::max(max_block_size, (int)blocks[i].extent(1)); }
  return max_block_size;
}

int BlockDiagOpFun::get_num_block_cols() const { return num_block_cols; }

int BlockDiagOpFun::get_zero_block_index(int i) const { return zero_block_indices(i); }

int BlockDiagOpFun::get_num_time_nodes() const {
  for (int i = 0; i < num_block_cols; i++) {
    if (zero_block_indices(i) != -1) { return blocks[i].shape(0); }
  }
  return 0; // BlockDiagOpFun is all zeros anyways
}

void BlockDiagOpFun::add_block(int i, nda::array_const_view<dcomplex, 3> block) {
  if (zero_block_indices(i) == -1) {
    blocks[i] = block;
  } else {
    blocks[i] = nda::make_regular(blocks[i] + block);
  }
  zero_block_indices(i) = 0; // mark block as non-zero
}

std::string BlockDiagOpFun::hdf5_format() { return "BlockDiagOpFun"; }

void h5_write(h5::group g, const std::string &subgroup_name, const BlockDiagOpFun &BDOF) {
  auto sg = g.create_group(subgroup_name);
  h5::write_hdf5_format(sg, BDOF);
  for (int i = 0; i < BDOF.num_block_cols; i++) { h5::write(sg, "block_" + std::to_string(i), BDOF.blocks[i]); }
  h5::write(sg, "zero_block_indices", BDOF.zero_block_indices);
}

/////////////// BlockOp (BO) class ///////////////

BlockOp::BlockOp(nda::vector<int> &block_indices, std::vector<nda::array<dcomplex, 2>> &blocks)
   : block_indices(block_indices), blocks(blocks), num_block_cols(block_indices.size()) {}

BlockOp::BlockOp(nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes)
   : block_indices(block_indices), num_block_cols(block_indices.size()) {

  std::vector<nda::array<dcomplex, 2>> blocks(num_block_cols);
  for (int i = 0; i < num_block_cols; i++) {
    if (block_indices(i) != -1) {
      blocks[i] = nda::zeros<dcomplex>(block_sizes(i, 0), block_sizes(i, 1));
    } else {
      blocks[i] = nda::zeros<dcomplex>(1, 1);
    }
  }
  this->blocks = blocks;
}

BlockOp &BlockOp::operator+=(const BlockOp &F) {
  // BlockOp addition-assignment operator
  // @param[in] F BlockOp
  // TODO: exception handling
  for (int i = 0; i < this->num_block_cols; i++) {
    if (F.get_block_index(i) != -1) { this->blocks[i] += F.blocks[i]; }
  }
  return *this;
}

void BlockOp::set_block_indices(nda::vector<int> &block_indices) {

  this->block_indices = block_indices;
  num_block_cols      = block_indices.size();
}

void BlockOp::set_block_index(int i, int block_index) { block_indices(i) = block_index; }

void BlockOp::set_blocks(std::vector<nda::array<dcomplex, 2>> &blocks) {

  this->blocks   = blocks;
  num_block_cols = blocks.size();
}

void BlockOp::set_block(int i, nda::array_const_view<dcomplex, 2> block) { blocks[i] = block; }

nda::vector_const_view<int> BlockOp::get_block_indices() const { return block_indices; }

int BlockOp::get_block_index(int i) const { return block_indices(i); }

const std::vector<nda::array<dcomplex, 2>> &BlockOp::get_blocks() const { return blocks; }

nda::array_const_view<dcomplex, 2> BlockOp::get_block(int i) const {
  if (block_indices(i) == -1) {
    auto arr = nda::zeros<dcomplex>(1, 1);
    return arr;
  } else {
    return blocks[i];
  }
}

int BlockOp::get_num_block_cols() const { return num_block_cols; }

nda::array<int, 2> BlockOp::get_block_sizes() const {
  auto block_sizes = nda::zeros<int>(num_block_cols, 2);
  for (int i = 0; i < num_block_cols; i++) {
    if (block_indices(i) != -1) {
      block_sizes(i, 0) = blocks[i].shape(0);
      block_sizes(i, 1) = blocks[i].shape(1);
    } else {
      block_sizes(i, 0) = -1;
      block_sizes(i, 1) = -1;
    }
  }
  return block_sizes;
};

nda::vector<int> BlockOp::get_block_size(int i) const {
  auto block_size = nda::zeros<int>(2);
  if (block_indices(i) != -1) {
    block_size(0) = blocks[i].shape(0);
    block_size(1) = blocks[i].shape(1);
  } else {
    block_size() = -1;
  }
  return block_size;
};

int BlockOp::get_block_size(int block_ind, int dim) const {
  if (block_indices(block_ind) != -1) {
    return blocks[block_ind].shape(dim);
  } else {
    return -1;
  }
}

/////////////// BlockOp3D class ///////////////

BlockOp3D::BlockOp3D(nda::vector_const_view<int> block_indices, std::vector<nda::array<dcomplex, 3>> &blocks)
   : block_indices(block_indices), blocks(blocks), num_block_cols(block_indices.size()), zero(nda::zeros<dcomplex>(1, 1, 1)) {}

BlockOp3D::BlockOp3D(int r, nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes)
   : block_indices(block_indices), num_block_cols(block_indices.size()) {

  std::vector<nda::array<dcomplex, 3>> blocks(num_block_cols);
  for (int i = 0; i < num_block_cols; i++) {
    if (block_indices(i) != -1) {
      blocks[i] = nda::zeros<dcomplex>(r, block_sizes(i, 0), block_sizes(i, 1));
    } else {
      blocks[i] = nda::zeros<dcomplex>(1, 1, 1);
    }
  }
  this->blocks = blocks;
}

void BlockOp3D::set_block_indices(nda::vector<int> &block_indices) {

  this->block_indices = block_indices;
  num_block_cols      = block_indices.size();
}

void BlockOp3D::set_block_index(int i, int block_index) { block_indices(i) = block_index; }

void BlockOp3D::set_blocks(std::vector<nda::array<dcomplex, 3>> &blocks) {

  this->blocks   = blocks;
  num_block_cols = blocks.size();
}

void BlockOp3D::set_block(int i, nda::array_const_view<dcomplex, 3> block) { blocks[i] = block; }

nda::vector_const_view<int> BlockOp3D::get_block_indices() const { return block_indices; }

int BlockOp3D::get_block_index(int i) const { return block_indices(i); }

const std::vector<nda::array<dcomplex, 3>> &BlockOp3D::get_blocks() const { return blocks; }

nda::array_const_view<dcomplex, 3> BlockOp3D::get_block(int i) const {
  if (block_indices(i) == -1) {
    return zero;
  } else {
    return blocks[i];
  }
}

int BlockOp3D::get_num_block_cols() const { return num_block_cols; }

nda::array<int, 2> BlockOp3D::get_block_sizes() const {
  auto block_sizes = nda::zeros<int>(num_block_cols, 2);
  for (int i = 0; i < num_block_cols; i++) {
    if (block_indices(i) != -1) {
      block_sizes(i, 0) = blocks[i].shape(1);
      block_sizes(i, 1) = blocks[i].shape(2);
    } else {
      block_sizes(i, 0) = -1;
      block_sizes(i, 1) = -1;
    }
  }
  return block_sizes;
}

nda::vector<int> BlockOp3D::get_block_size(int i) const {
  auto block_size = nda::zeros<int>(2);
  if (block_indices(i) != -1) {
    block_size(0) = blocks[i].shape(1);
    block_size(1) = blocks[i].shape(2);
  } else {
    block_size() = -1;
  }
  return block_size;
}

int BlockOp3D::get_block_size(int block_ind, int dim) const {
  if (block_indices(block_ind) != -1) {
    return blocks[block_ind].shape(dim + 1); // dim = 0 for time, 1 for row, 2 for col
  } else {
    return -1;
  }
}

/////////////// BlockOpFun (BOF) class ///////////////

BlockOpFun::BlockOpFun(nda::vector_const_view<int> block_indices, std::vector<nda::array<dcomplex, 3>> &blocks) : BlockOp3D{block_indices, blocks} {}

BlockOpFun::BlockOpFun(int r, nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes)
   : BlockOp3D{r, block_indices, block_sizes} {}

int BlockOpFun::get_num_time_nodes() const {
  for (int i = 0; i < num_block_cols; i++) {
    if (block_indices(i) != -1) { return blocks[i].shape(0); }
  }
  return 0; // BlockOpFun is all zeros anyways
}

/////////////// DenseFSet class ///////////////

DenseFSet::DenseFSet(nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                     nda::array_const_view<dcomplex, 3> hyb_refl_coeffs)
   : Fs(Fs), F_dags(F_dags) {

  int n       = Fs.extent(0);
  int N       = Fs.extent(1);
  int p       = hyb_coeffs.extent(0);
  F_dag_bars  = nda::array<dcomplex, 4>(n, p, N, N);
  F_bars_refl = nda::array<dcomplex, 4>(n, p, N, N);
  for (int lam = 0; lam < n; lam++) {
    for (int l = 0; l < p; l++) {
      for (int nu = 0; nu < n; nu++) {
        F_dag_bars(lam, l, _, _) += hyb_coeffs(l, nu, lam) * F_dags(nu, _, _);
        F_bars_refl(nu, l, _, _) += hyb_refl_coeffs(l, nu, lam) * Fs(lam, _, _);
      }
    }
  }
}

std::size_t DenseFSet::get_num_orb_inds() const { return Fs.extent(0); }

/////////////// BlockOpSymSet class ///////////////

BlockOpSymSet::BlockOpSymSet(nda::vector_const_view<int> block_indices, std::vector<nda::array<dcomplex, 3>> &blocks)
   : BlockOp3D{block_indices, blocks} {}

BlockOpSymSet::BlockOpSymSet(int q, nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes)
   : BlockOp3D{q, block_indices, block_sizes} {}

int BlockOpSymSet::get_size_sym_set() const {
  for (int i = 0; i < num_block_cols; i++) {
    if (block_indices(i) != -1) { return blocks[i].shape(0); }
  }
  return 0; // BlockOpSymSet is all zeros anyways
}

////////////// BlockOpSymSetBar class ///////////////

BlockOpSymSetBar::BlockOpSymSetBar(nda::vector_const_view<int> block_indices, std::vector<nda::array<dcomplex, 4>> &blocks)
   : block_indices(block_indices), blocks(blocks), num_block_cols(block_indices.size()) {}

BlockOpSymSetBar::BlockOpSymSetBar(int q, int r, nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes)
   : block_indices(block_indices), num_block_cols(block_indices.size()) {

  std::vector<nda::array<dcomplex, 4>> blocks(num_block_cols);
  for (int i = 0; i < num_block_cols; i++) {
    if (block_indices(i) != -1) {
      blocks[i] = nda::zeros<dcomplex>(q, r, block_sizes(i, 0), block_sizes(i, 1));
    } else {
      blocks[i] = nda::zeros<dcomplex>(q, r, 1, 1);
    }
  }
  this->blocks = blocks;
}

void BlockOpSymSetBar::set_block_indices(nda::vector<int> &block_indices) {

  this->block_indices = block_indices;
  num_block_cols      = block_indices.size();
}

void BlockOpSymSetBar::set_block_index(int i, int block_index) { block_indices(i) = block_index; }

void BlockOpSymSetBar::set_blocks(std::vector<nda::array<dcomplex, 4>> &blocks) { this->blocks = blocks; }

void BlockOpSymSetBar::set_block(int i, nda::array_const_view<dcomplex, 4> block) { blocks[i] = block; }

nda::vector_const_view<int> BlockOpSymSetBar::get_block_indices() const { return block_indices; }

int BlockOpSymSetBar::get_block_index(int i) const { return block_indices(i); }

const std::vector<nda::array<dcomplex, 4>> &BlockOpSymSetBar::get_blocks() const { return blocks; }

nda::array_const_view<dcomplex, 4> BlockOpSymSetBar::get_block(int i) const {
  if (block_indices(i) == -1) {
    auto arr = nda::zeros<dcomplex>(1, 1, 1, 1);
    return arr;
  } else {
    return blocks[i];
  }
}

int BlockOpSymSetBar::get_block_size(int block_ind, int dim) const {
  if (block_indices(block_ind) != -1) {
    return blocks[block_ind].shape(dim + 2); // index 0 for sym set index, 1 for DLR indices, 2 for row, 3 for col
  } else {
    return -1;
  }
}

int BlockOpSymSetBar::get_num_block_cols() const { return num_block_cols; }

int BlockOpSymSetBar::get_size_sym_set() const {
  for (int i = 0; i < num_block_cols; i++) {
    if (block_indices(i) != -1) { return blocks[i].shape(0); }
  }
  return 0; // BlockOpSymSetBar is all zeros anyways
}

int BlockOpSymSetBar::get_num_time_nodes() const {
  for (int i = 0; i < num_block_cols; i++) {
    if (block_indices(i) != -1) { return blocks[i].shape(1); }
  }
  return 0; // BlockOpSymSetBar is all zeros anyways
}

void BlockOpSymSetBar::add_block(int i, int s, int t, nda::array_const_view<dcomplex, 2> block) {
  if (block_indices(i) != -1) { blocks[i](s, t, _, _) += block; }
}

////////////// BlockOpSymQuartet class ///////////////

BlockOpSymQuartet::BlockOpSymQuartet(std::vector<BlockOpSymSet> Fs, std::vector<BlockOpSymSet> F_dags, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                     nda::array_const_view<dcomplex, 3> hyb_refl_coeffs, nda::vector_const_view<long> sym_set_labels)
   : Fs(Fs), F_dags(F_dags), sym_set_labels(sym_set_labels) {

  // Fs and F_dags are vectors of BOSS
  // Each entry corresponds to a set of operators with the same block-sparse structure
  // Fs and F_dags have the same number of entries
  // if k = # of entries of Fs, and each entry f_i has q_i operators, then n = sum(q_i) = number of orbital indices
  int k = Fs.size();
  if (k != F_dags.size()) { throw std::invalid_argument("Fs and F_dags must have the same number of entries"); }

  // initialize F_dag_bars and F_bars_refl
  int p = hyb_coeffs.extent(0);
  std::vector<BlockOpSymSetBar> F_dag_bars, F_bars_refl;
  nda::vector<int> block_indices_dag = F_dags[0].get_block_indices();
  nda::vector<int> block_indices_f   = Fs[0].get_block_indices();
  for (int i = 0; i < k; i++) {
    block_indices_dag = F_dags[i].get_block_indices();
    block_indices_f   = Fs[i].get_block_indices();
    std::cout << "Block indices dag: " << block_indices_dag << "\n";
    F_dag_bars.emplace_back(F_dags[i].get_size_sym_set(), p, block_indices_dag, F_dags[i].get_block_sizes());
    std::cout << "Block indices f: " << block_indices_f << "\n";
    F_bars_refl.emplace_back(Fs[i].get_size_sym_set(), p, block_indices_f, Fs[i].get_block_sizes());
  }

  // calculate symmetry set indices
  long n         = sym_set_labels.size();                // number of orbital indices
  long q         = nda::max_element(sym_set_labels) + 1; // number of symmetry sets
  sym_set_inds   = nda::zeros<long>(n);                  // indices of orbital indices in symmetry sets
  sym_set_sizes  = nda::zeros<long>(q);                  // sizes of symmetry sets
  sym_set_to_orb = nda::ones<long>(q, n);                // map label and index to orbital index
  sym_set_to_orb *= -1;                                  // initialize with -1
  for (int i = 0; i < n; i++) {
    sym_set_inds(i) = sym_set_sizes(sym_set_labels(i));
    sym_set_sizes(sym_set_labels(i))++;
    sym_set_to_orb(sym_set_labels(i), sym_set_inds(i)) = i; // map symmetry set index to backbone orbital index
  }

  // compute F_dag_bars and F_bars_refl
  for (int l = 0; l < p; l++) {
    for (int p_lam = 0; p_lam < q; p_lam++) {
      for (int p_nu = 0; p_nu < q; p_nu++) {
        for (int lam = 0; lam < sym_set_sizes(p_lam); lam++) {
          for (int nu = 0; nu < sym_set_sizes(p_nu); nu++) {
            long lam_orb = sym_set_to_orb(p_lam, lam);
            long nu_orb  = sym_set_to_orb(p_nu, nu);
            for (int b = 0; b < F_dags[p_lam].get_num_block_cols(); b++) {
              if (F_dags[p_lam].get_block_index(b) != -1) {
                F_dag_bars[p_lam].add_block(b, lam, l, nda::make_regular(hyb_coeffs(l, nu_orb, lam_orb) * F_dags[p_lam].get_block(b)(nu, _, _)));
              }
            }
            for (int b = 0; b < Fs[p_nu].get_num_block_cols(); b++) {
              if (Fs[p_nu].get_block_index(b) != -1) {
                F_bars_refl[p_nu].add_block(b, nu, l, nda::make_regular(hyb_refl_coeffs(l, nu_orb, lam_orb) * Fs[p_nu].get_block(b)(lam, _, _)));
              }
            }
          }
        }
      }
    }
  }
  this->F_dag_bars  = F_dag_bars;
  this->F_bars_refl = F_bars_refl;
}

/////////////// Utilities and operator overrides ///////////////

std::ostream &operator<<(std::ostream &os, BlockDiagOpFun &D) {
  // Print BlockDiagOpFun
  // @param[in] os output stream
  // @param[in] D BlockDiagOpFun
  // @return output stream

  for (int i = 0; i < D.get_num_block_cols(); i++) { os << "Block " << i << ":\n" << D.get_block(i) << "\n"; }
  return os;
};

std::ostream &operator<<(std::ostream &os, BlockOp &F) {
  // Print BlockOp
  // @param[in] os output stream
  // @param[in] F BlockOp
  // @return output stream

  os << "Block indices: " << F.get_block_indices() << "\n";
  for (int i = 0; i < F.get_num_block_cols(); i++) {
    if (F.get_block_indices()[i] == -1) {
      os << "Block " << i << ": 0\n";
    } else {
      os << "Block " << i << ":\n" << F.get_block(i) << "\n";
    }
  }
  return os;
};

std::ostream &operator<<(std::ostream &os, BlockOp3D &F) {
  // Print BlockOp3D
  // @param[in] os output stream
  // @param[in] F BlockOp3D
  // @return output stream

  os << "Block indices: " << F.get_block_indices() << "\n";
  for (int i = 0; i < F.get_num_block_cols(); i++) {
    if (F.get_block_indices()[i] == -1) {
      os << "Block " << i << ": 0\n";
    } else {
      os << "Block " << i << ":\n" << F.get_block(i) << "\n";
    }
  }
  return os;
};

std::ostream &operator<<(std::ostream &os, BlockOpSymSetBar &F) {
  // Print BlockOpSymSetBar
  // @param[in] os output stream
  // @param[in] F BlockOpSymSetBar
  // @return output stream

  os << "Block indices: " << F.get_block_indices() << "\n";
  for (int i = 0; i < F.get_num_block_cols(); i++) {
    if (F.get_block_indices()[i] == -1) {
      os << "Block " << i << ": 0\n";
    } else {
      os << "Block " << i << ":\n" << F.get_block(i) << "\n";
    }
  }
  return os;
}

BlockOp dagger_bs(BlockOp const &F) {
  // Evaluate F^dagger in block-sparse storage
  // @param[in] F F operator
  // @return F^dagger operator

  int num_block_cols = F.get_num_block_cols();

  // find block indices for F^dagger
  nda::vector<int> block_indices_dag(num_block_cols);
  // initialize indices with -1
  block_indices_dag = -1;
  std::vector<nda::array<dcomplex, 2>> blocks_dag(num_block_cols);
  for (int i = 0; i < num_block_cols; ++i) {
    int j = F.get_block_indices()[i];
    if (j != -1) {
      block_indices_dag[j] = i;
      blocks_dag[j]        = nda::transpose(F.get_blocks()[i]);
    }
  }
  BlockOp F_dag(block_indices_dag, blocks_dag);
  return F_dag;
}

BlockDiagOpFun operator*(int i, BlockDiagOpFun const &D) {
  // Compute a product between an integer and a BlockDiagOpFun
  // @param[in] i integer
  // @param[in] D BlockDiagOpFun

  auto product = D;
  for (int j = 0; j < D.get_num_block_cols(); j++) {
    if (D.get_zero_block_index(j) != -1) {
      auto prod_block = nda::make_regular(i * D.get_block(j));
      product.set_block(j, prod_block);
    }
  }
  return product;
}

BlockOp operator*(const dcomplex c, const BlockOp &F) {
  // Compute a product between a scalar and an BlockOp
  // @param[in] c dcomplex
  // @param[in] F BlockOp

  auto product = F;
  for (int i = 0; i < F.get_num_block_cols(); i++) {
    if (F.get_block_index(i) != -1) {
      auto prod_block = nda::make_regular(c * F.get_block(i));
      product.set_block(i, prod_block);
    }
  }
  return product;
}

BlockOp3D operator*(const dcomplex c, const BlockOp3D &F) {
  // Compute a product between a scalar and an BlockOp3D
  // @param[in] c dcomplex
  // @param[in] F BlockOp3D

  auto product = F;
  for (int i = 0; i < F.get_num_block_cols(); i++) {
    if (F.get_block_index(i) != -1) {
      auto prod_block = nda::make_regular(c * F.get_block(i));
      product.set_block(i, prod_block);
    }
  }
  return product;
}

BlockDiagOpFun BOFtoBDOF(BlockOpFun const &A) {
  // Convert a BlockOpFun with diagonal structure to a BlockDiagOpFun
  // @param[in] A BlockOpFun
  // @return BlockDiagOpFun

  int num_block_cols      = A.get_num_block_cols();
  auto diag_blocks        = A.get_blocks();
  auto zero_block_indices = nda::zeros<int>(num_block_cols);
  for (int i = 0; i < num_block_cols; i++) {
    int block_index = A.get_block_index(i);
    if (block_index == -1) {
      diag_blocks[i]        = nda::zeros<dcomplex>(1, 1, 1);
      zero_block_indices(i) = -1;
    } else if (block_index != i) {
      throw std::invalid_argument("BOF is not diagonal");
    }
  }

  return BlockDiagOpFun(diag_blocks, zero_block_indices);
}

BlockDiagOpFun nonint_gf_BDOF(std::vector<nda::array<double, 2>> H_blocks, nda::vector<int> H_block_inds, double beta,
                              nda::vector_const_view<double> dlr_it_abs) {

  int num_block_cols = H_block_inds.size();
  nda::vector<int> H_block_sizes(num_block_cols);
  for (int i = 0; i < num_block_cols; i++) { H_block_sizes(i) = H_blocks[i].extent(0); }

  int r = dlr_it_abs.size();

  double tr_exp_minusbetaH = 0;
  std::vector<nda::array<double, 1>> H_evals(num_block_cols);
  std::vector<nda::array<double, 2>> H_evecs(num_block_cols);
  for (int i = 0; i < num_block_cols; i++) {
    if (H_block_inds(i) != -1) {
      if (H_block_sizes(i) == 1) {
        H_evals[i] = nda::array<double, 1>{H_blocks[i](0, 0)};
        H_evecs[i] = nda::array<double, 2>{{1}};
      } else {
        auto H_block_eig = nda::linalg::eigenelements(H_blocks[i]);
        H_evals[i]       = std::get<0>(H_block_eig);
        H_evecs[i]       = std::get<1>(H_block_eig);
      }
      tr_exp_minusbetaH += nda::sum(exp(-beta * H_evals[i]));
    } else {
      H_evals[i] = nda::zeros<double>(H_block_sizes(i));
      H_evecs[i] = nda::eye<double>(H_block_sizes(i));
      tr_exp_minusbetaH += 1.0 * H_block_sizes(i); // 0 entry in the diagonal
    }
  }

  auto eta_0 = nda::log(tr_exp_minusbetaH) / beta;
  auto Gt    = BlockDiagOpFun(r, H_block_sizes);
  for (int i = 0; i < num_block_cols; i++) {
    auto Gt_block = nda::array<dcomplex, 3>(r, H_block_sizes(i), H_block_sizes(i));
    auto Gt_temp  = nda::make_regular(0 * H_blocks[i]);
    for (int t = 0; t < r; t++) {
      for (int j = 0; j < H_block_sizes(i); j++) { Gt_temp(j, j) = -exp(-beta * dlr_it_abs(t) * (H_evals[i](j) + eta_0)); }
      Gt_block(t, _, _) = nda::matmul(H_evecs[i], nda::matmul(Gt_temp, nda::transpose(H_evecs[i])));
    }
    Gt.set_block(i, Gt_block);
  }

  return Gt;
}
