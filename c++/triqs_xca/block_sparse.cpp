#include <iostream>

#include <itertools/itertools.hpp>

#include <cppdlr/dlr_imtime.hpp>

#include "triqs_xca/block_sparse.hpp"

namespace triqs_xca::block_sparse {

  using cppdlr::_;

  using nda::range;
  using nda::linalg::matmul;

  /////////////// BlockDiagOpFun (BDOF) class ///////////////
  BlockDiagOpFun::BlockDiagOpFun(std::vector<nda::array<dcomplex, 3>> &blocks, nda::vector_const_view<int> zero_block_indices)
     : blocks(blocks), num_block_cols(blocks.size()), zero_block_indices(zero_block_indices) {}

  BlockDiagOpFun::BlockDiagOpFun(int r, nda::vector_const_view<int> block_sizes) : num_block_cols(block_sizes.size()) {
    std::vector<nda::array<dcomplex, 3>> temp_blocks(num_block_cols);
    zero_block_indices = nda::make_regular(-1 * nda::ones<int>(num_block_cols));
    for (int i = 0; i < num_block_cols; i++) { temp_blocks[i] = nda::zeros<dcomplex>(r, block_sizes[i], block_sizes[i]); }
    this->blocks = temp_blocks;
  }

  BlockDiagOpFun::BlockDiagOpFun(const triqs::gfs::block_gf<triqs::mesh::dlr_imtime> &bgf) : num_block_cols(bgf.size()) {
    // TODO set block to just a single zero if the block gf is numerically zero
    blocks.resize(num_block_cols);
    zero_block_indices = nda::zeros<int>(num_block_cols);
    int r              = bgf[0].mesh().size();

    for (int i = 0; i < num_block_cols; i++) {
      auto const &gf = bgf[i];
      int block_size = gf.target_shape()[0];

      blocks[i] = nda::array<dcomplex, 3>(r, block_size, block_size);

      for (int t = 0; t < r; t++) { blocks[i](t, _, _) = gf(t); }

      if (nda::max_element(nda::abs(blocks[i])) < 1e-16) {
        zero_block_indices(i) = -1;
      } else {
        zero_block_indices(i) = 0;
      }
    }
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

  BlockDiagOpFun &BlockDiagOpFun::operator*=(const dcomplex scalar) {
    for (int i = 0; i < this->num_block_cols; i++) {
      if (zero_block_indices(i) == -1) {
        // zero block, do nothing
      } else {
        this->blocks[i] *= scalar;
      }
    }
    return *this;
  }

  void BlockDiagOpFun::set_blocks(std::vector<nda::array<dcomplex, 3>> &new_blocks) {

    this->blocks       = new_blocks;
    num_block_cols     = new_blocks.size();
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

  /////////////// BlockOp (BO) class ///////////////

  BlockOp::BlockOp(nda::vector<int> &block_indices, std::vector<nda::array<dcomplex, 2>> &blocks)
     : block_indices(block_indices), blocks(blocks), num_block_cols(block_indices.size()) {}

  BlockOp::BlockOp(nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes)
     : block_indices(block_indices), num_block_cols(block_indices.size()) {

    std::vector<nda::array<dcomplex, 2>> temp_blocks(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
      if (block_indices(i) != -1) {
        temp_blocks[i] = nda::zeros<dcomplex>(block_sizes(i, 0), block_sizes(i, 1));
      } else {
        temp_blocks[i] = nda::zeros<dcomplex>(1, 1);
      }
    }
    this->blocks = temp_blocks;
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

  void BlockOp::set_block_indices(nda::vector<int> &new_block_indices) {

    this->block_indices = new_block_indices;
    num_block_cols      = new_block_indices.size();
  }

  void BlockOp::set_block_index(int i, int block_index) { block_indices(i) = block_index; }

  void BlockOp::set_blocks(std::vector<nda::array<dcomplex, 2>> &new_blocks) {

    this->blocks   = new_blocks;
    num_block_cols = new_blocks.size();
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

    std::vector<nda::array<dcomplex, 3>> temp_blocks(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
      if (block_indices(i) != -1) {
        temp_blocks[i] = nda::zeros<dcomplex>(r, block_sizes(i, 0), block_sizes(i, 1));
      } else {
        temp_blocks[i] = nda::zeros<dcomplex>(1, 1, 1);
      }
    }
    this->blocks = temp_blocks;
  }

  void BlockOp3D::set_block_indices(nda::vector<int> &new_block_indices) {

    this->block_indices = new_block_indices;
    num_block_cols      = new_block_indices.size();
  }

  void BlockOp3D::set_block_index(int i, int block_index) { block_indices(i) = block_index; }

  void BlockOp3D::set_blocks(std::vector<nda::array<dcomplex, 3>> &new_blocks) {

    this->blocks   = new_blocks;
    num_block_cols = new_blocks.size();
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

  void BlockOp3D::print_slice(int t) const {
    for (int i = 0; i < num_block_cols; i++) {
      std::cout << "Block " << i << " at slice " << t << ":\n";
      if (block_indices(i) != -1) {
        std::cout << blocks[i](t, _, _) << "\n";
      } else {
        std::cout << "Zero block\n";
      }
    }
  }

  /////////////// BlockOpFun (BOF) class ///////////////

  BlockOpFun::BlockOpFun(nda::vector_const_view<int> block_indices, std::vector<nda::array<dcomplex, 3>> &blocks)
     : BlockOp3D{block_indices, blocks} {}

  BlockOpFun::BlockOpFun(int r, nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes)
     : BlockOp3D{r, block_indices, block_sizes} {}

  int BlockOpFun::get_num_time_nodes() const {
    for (int i = 0; i < num_block_cols; i++) {
      if (block_indices(i) != -1) { return blocks[i].shape(0); }
    }
    return 0; // BlockOpFun is all zeros anyways
  }

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

    std::vector<nda::array<dcomplex, 4>> temp_blocks(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
      if (block_indices(i) != -1) {
        temp_blocks[i] = nda::zeros<dcomplex>(q, r, block_sizes(i, 0), block_sizes(i, 1));
      } else {
        temp_blocks[i] = nda::zeros<dcomplex>(q, r, 1, 1);
      }
    }
    this->blocks = temp_blocks;
  }

  void BlockOpSymSetBar::set_block_indices(nda::vector<int> &new_block_indices) {

    this->block_indices = new_block_indices;
    num_block_cols      = new_block_indices.size();
  }

  void BlockOpSymSetBar::set_block_index(int i, int block_index) { block_indices(i) = block_index; }

  void BlockOpSymSetBar::set_blocks(std::vector<nda::array<dcomplex, 4>> &new_blocks) { this->blocks = new_blocks; }

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

  BlockOpSymQuartet::BlockOpSymQuartet(std::vector<BlockOpSymSet> Fs, std::vector<BlockOpSymSet> F_dags,
                                       nda::array_const_view<dcomplex, 3> hyb_coeffs, nda::vector_const_view<long> sym_set_labels)
     : Fs(Fs), F_dags(F_dags), sym_set_labels(sym_set_labels), p(hyb_coeffs.extent(0)) {

    // Fs and F_dags are vectors of BOSS
    // Each entry corresponds to a set of operators with the same block-sparse structure
    // Fs and F_dags have the same number of entries
    // if k = # of entries of Fs, and each entry f_i has q_i operators, then n = sum(q_i) = number of orbital indices
    int k = Fs.size();
    if (k != F_dags.size()) { throw std::invalid_argument("Fs and F_dags must have the same number of entries"); }

    // initialize F_dag_bars and F_bars_refl
    p                                  = hyb_coeffs.extent(0);
    nda::vector<int> block_indices_dag = F_dags[0].get_block_indices();
    nda::vector<int> block_indices_f   = Fs[0].get_block_indices();
    for (int i = 0; i < k; i++) {
      block_indices_dag = F_dags[i].get_block_indices();
      block_indices_f   = Fs[i].get_block_indices();
      this->F_dag_bars.emplace_back(F_dags[i].get_size_sym_set(), p, block_indices_dag, F_dags[i].get_block_sizes());
      this->F_bars_refl.emplace_back(Fs[i].get_size_sym_set(), p, block_indices_f, Fs[i].get_block_sizes());
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
                  F_bars_refl[p_nu].add_block(
                     b, nu, l, nda::make_regular(-hyb_coeffs(l, nu_orb, lam_orb) * Fs[p_nu].get_block(b)(lam, _, _))); // Add -1 sign for reflected F
                }
              }
            }
          }
        }
      }
    }
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

  BlockDiagOpFun atom_prop_from_eigensystem(std::vector<nda::array<double, 1>> const &evals, std::vector<nda::array<dcomplex, 2>> const &evecs,
                                            double Z, double beta, nda::vector_const_view<double> dlr_it_abs) {

    int num_block_cols = evals.size();
    int r              = dlr_it_abs.size();
    double eta_0       = std::log(Z) / beta;

    std::vector<nda::array<dcomplex, 3>> blocks(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
      int n     = evals[i].size();
      blocks[i] = nda::array<dcomplex, 3>(r, n, n);

      auto Udag    = nda::make_regular(nda::dagger(evecs[i]));
      auto Gt_temp = nda::zeros<dcomplex>(n, n);
      for (int t = 0; t < r; t++) {
        for (int j = 0; j < n; j++) { Gt_temp(j, j) = -std::exp(-beta * dlr_it_abs(t) * (evals[i](j) + eta_0)); }
        blocks[i](t, _, _) = matmul(evecs[i], matmul(Gt_temp, Udag));
      }
    }

    // no block of the propagator is ever zero: a zero block of H still gives G_B(\tau) = -exp(-\tau \eta_0) I
    auto zero_block_indices = nda::zeros<int>(num_block_cols);
    return {blocks, zero_block_indices};
  }

  BlockDiagOpFun nonint_gf_BDOF(std::vector<nda::array<dcomplex, 2>> H_blocks, nda::vector<int> H_block_inds, double beta,
                                nda::vector_const_view<double> dlr_it_abs) {

    int num_block_cols = H_block_inds.size();

    std::vector<nda::array<double, 1>> H_evals(num_block_cols);
    std::vector<nda::array<dcomplex, 2>> H_evecs(num_block_cols);
    for (int i = 0; i < num_block_cols; i++) {
      int n = H_blocks[i].extent(0);
      if (H_block_inds(i) == -1) { // zero block, eigensystem is trivial
        H_evals[i] = nda::zeros<double>(n);
        H_evecs[i] = nda::eye<dcomplex>(n);
      } else if (n == 1) {
        H_evals[i] = nda::array<double, 1>{H_blocks[i](0, 0).real()};
        H_evecs[i] = nda::array<dcomplex, 2>{{1}};
      } else {
        auto H_block_eig = nda::linalg::eigh(H_blocks[i]);
        H_evals[i]       = std::get<0>(H_block_eig);
        H_evecs[i]       = std::get<1>(H_block_eig);
      }
    }

    // shift by the ground state energy before exponentiating, mirroring triqs::atom_diag; the shift
    // cancels against eta_0 inside atom_prop_from_eigensystem and keeps exp(-beta E) in range
    double E0 = nda::min_element(H_evals[0]);
    for (int i = 1; i < num_block_cols; i++) { E0 = std::min(E0, nda::min_element(H_evals[i])); }

    double Z = 0;
    for (int i = 0; i < num_block_cols; i++) {
      H_evals[i] -= E0;
      Z += nda::sum(nda::exp(-beta * H_evals[i]));
    }

    return atom_prop_from_eigensystem(H_evals, H_evecs, Z, beta, dlr_it_abs);
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> BDOF_to_block_gf(BlockDiagOpFun const &BDOF, double beta, double Lambda, double eps) {
    auto dlr_rf = cppdlr::build_dlr_rf(Lambda, eps);
    auto itops  = cppdlr::imtime_ops(Lambda, dlr_rf);

    // triqs gf mesh. dlr_imtime takes the energy cutoff w_max = Lambda / beta and rebuilds the DLR grid
    // from w_max * beta, so passing Lambda directly would attach a mesh built on Lambda * beta to data
    // sampled on the Lambda grid above.
    auto t_mesh = triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps, false);
    // create vector of gf
    std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> gf_vec(BDOF.get_num_block_cols());

    for (int i = 0; i < BDOF.get_num_block_cols(); ++i) {
      if (BDOF.get_zero_block_index(i) == 0) {
        gf_vec[i]        = triqs::gfs::gf<triqs::mesh::dlr_imtime>{t_mesh, {BDOF.get_block(i).extent(1), BDOF.get_block(i).extent(2)}};
        gf_vec[i].data() = BDOF.get_block(i);
      } else {
        // empty block
        gf_vec[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>{t_mesh, {0, 0}};
      }
    }
    return {gf_vec};
  }

  template <bool isComplex>
  dcomplex expectation_value(triqs::operators::many_body_operator_real const &op, triqs::atom_diag::atom_diag<isComplex> const &ad,
                             triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc) {

    auto op_blocks = ad.get_op_mat(op);
    auto beta      = G_ppsc[0].mesh().beta();

    dcomplex sum = 0;

    for (auto bidx : range(op_blocks.block_mat.size())) {
      assert(op_blocks.connection[bidx] == bidx);
      if (op_blocks.block_mat[bidx].shape(0) == 0) continue; // skip empty blocks
      auto g_dlr         = make_gf_dlr(G_ppsc[bidx]);
      auto U             = ad.get_unitary_matrix(bidx);
      auto op_mat_transf = U * op_blocks.block_mat[bidx] * nda::conj(nda::transpose(U));
      sum += -trace(matmul(op_mat_transf, g_dlr(beta)));
    }

    return sum;
  }

  template dcomplex expectation_value(triqs::operators::many_body_operator_real const &op, triqs::atom_diag::atom_diag<false> const &ad,
                                      triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc);

  template dcomplex expectation_value(triqs::operators::many_body_operator_real const &op, triqs::atom_diag::atom_diag<true> const &ad,
                                      triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc);

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> convolve_ppsc(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G1,
                                                              triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G2) {

    assert(G1[0].mesh() == G2[0].mesh());

    auto mesh  = G1[0].mesh();
    auto beta  = mesh.beta();
    auto itops = mesh.dlr_it();

    std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> gg_vec;

    for (auto [g1, g2] : itertools::zip(G1, G2)) {
      auto gg = itops.convolve(beta, itops.vals2coefs(g1.data()), itops.vals2coefs(g2.data()), cppdlr::TIME_ORDERED);
      gg_vec.emplace_back(triqs::gfs::gf<triqs::mesh::dlr_imtime>(mesh, gg));
    }

    return {gg_vec};
  }

  dcomplex trace(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G) {

    dcomplex trace = 0;
    for (auto g : G) {
      double beta = g.mesh().beta();
      auto g_dlr  = make_gf_dlr(g);
      trace += nda::trace(g_dlr(beta));
    }
    return trace;
  }

} // namespace triqs_xca::block_sparse
