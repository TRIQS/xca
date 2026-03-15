#pragma once
#include <vector>

#include <nda/nda.hpp>

#include <triqs/gfs.hpp>
#include <triqs/operators.hpp>
#include <triqs/atom_diag/atom_diag.hpp>

#include <cppdlr/dlr_imtime.hpp>

namespace triqs_xca::block_sparse {

  using nda::dcomplex;

  // TODO templates for double/dcomplex

  /**
 * @class BlockDiagOpFun (BDOF)
 * @brief Block-sparse storage of time-dependent block-diagonal operator (e.g. Green's f'n)
 */
  class BlockDiagOpFun {
    private:
    std::vector<nda::array<dcomplex, 3>> blocks;
    int num_block_cols;
    nda::vector<int> zero_block_indices;

    public:
    BlockDiagOpFun &operator+=(const BlockDiagOpFun &G);
    BlockDiagOpFun &operator*=(const dcomplex scalar);
    void set_blocks(std::vector<nda::array<dcomplex, 3>> &new_blocks);
    void set_block(int i, nda::array_const_view<dcomplex, 3> block);
    void set_zero_block_indices(); // set zero_block_indices according to current blocks
    const std::vector<nda::array<dcomplex, 3>> &get_blocks() const;
    nda::array_const_view<dcomplex, 3> get_block(int i) const;
    nda::vector<int> get_block_sizes() const;
    int get_block_size(int i) const;
    int get_max_block_size() const;
    int get_num_block_cols() const;
    int get_zero_block_index(int i) const;
    int get_num_time_nodes() const;
    void add_block(int i, nda::array_const_view<dcomplex, 3> block);
    static std::string hdf5_format();

    /**
   * @brief Constructor for BlockDiagOpFun
   * @param[in] blocks vector of diagonal blocks
   * @param[in] zero_block_indices if i-th entry is -1, then blocks(i) = 0
   */
    BlockDiagOpFun(std::vector<nda::array<dcomplex, 3>> &blocks, nda::vector_const_view<int> zero_block_indices);

    /**
   * @brief Constructor for BlockDiagOpFun with blocks of zeros
   * @param[in] r number of imaginary time nodes
   * @param[in] block_sizes vector of sizes of diagonal blocks
   */
    BlockDiagOpFun(int r, nda::vector_const_view<int> block_sizes);

    /**
   * @brief Constructor for BlockDiagOpFun from a triqs block_gf<dlr_imtime>
   * @param[in] bgf block_gf<dlr_imtime>
   */
    BlockDiagOpFun(const triqs::gfs::block_gf<triqs::mesh::dlr_imtime> &bgf);
  };

  /**
 * @class BlockOp (BO)
 * @brief Block-sparse storage of F or F^dagger
 */
  class BlockOp {
    private:
    nda::vector<int> block_indices;
    std::vector<nda::array<dcomplex, 2>> blocks;
    int num_block_cols;

    public:
    BlockOp &operator+=(const BlockOp &F);
    void set_block_indices(nda::vector<int> &new_block_indices);
    void set_block_index(int i, int block_index);
    void set_blocks(std::vector<nda::array<dcomplex, 2>> &new_blocks);
    void set_block(int i, nda::array_const_view<dcomplex, 2> block);
    nda::vector_const_view<int> get_block_indices() const;
    int get_block_index(int i) const;
    const std::vector<nda::array<dcomplex, 2>> &get_blocks() const;
    nda::array_const_view<dcomplex, 2> get_block(int i) const;
    int get_num_block_cols() const;
    nda::array<int, 2> get_block_sizes() const;
    nda::vector<int> get_block_size(int i) const;
    int get_block_size(int block_ind, int dim) const;

    /**
     * @brief Constructor for BlockOp
     * @param[in] block_indices vector of block-column indices in each block-col
     * @param[in] blocks vector of blocks
     * @note block_indices[i] = -1 if F does not have a block in col i
     */
    BlockOp(nda::vector<int> &block_indices, std::vector<nda::array<dcomplex, 2>> &blocks);

    /**
     * @brief Constructor for BlockOp with blocks of zeros
     * @param[in] block_indices vector of block-column indices in each block-col
     * @param[in] block_sizes array of block sizes
     * @note block_indices[i] = -1 if F does not have a block in col i
     */
    BlockOp(nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes);
  };

  /**
 * @class BlockOp3D 
 * @brief Abstract superclass for block-sparse storage of sequences of matrices with the same sparsity pattern
 */
  class BlockOp3D {
    protected:
    nda::vector<int> block_indices;
    std::vector<nda::array<dcomplex, 3>> blocks;
    int num_block_cols;
    nda::array<dcomplex, 3> zero;

    public:
    void set_block_indices(nda::vector<int> &new_block_indices);
    void set_block_index(int i, int block_index);
    void set_blocks(std::vector<nda::array<dcomplex, 3>> &new_blocks);
    void set_block(int i, nda::array_const_view<dcomplex, 3> block);
    nda::vector_const_view<int> get_block_indices() const;
    int get_block_index(int i) const;
    const std::vector<nda::array<dcomplex, 3>> &get_blocks() const;
    nda::array_const_view<dcomplex, 3> get_block(int i) const;
    int get_num_block_cols() const;
    nda::array<int, 2> get_block_sizes() const;
    nda::vector<int> get_block_size(int i) const;
    int get_block_size(int block_ind, int dim) const;
    void print_slice(int t) const;

    /**
   * @brief Constructor for BlockOpFun 
   * @param[in] block_indices vector of block-col-indices
   * @param[in] blocks vector of blocks
   * @note block_indices[i] = -1 if F does not have a block in col i
   */
    BlockOp3D(nda::vector_const_view<int> block_indices, std::vector<nda::array<dcomplex, 3>> &blocks);

    /**
   * @brief Constructor for BlockOpFun with blocks of zeros
   * @param[in] r number of imaginary time nodes
   * @param[in] block_indices vector of block-col-indices
   * @param[in] block_sizes vector of sizes of blocks
   */
    BlockOp3D(int r, nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes);
  };

  /**
 * @class BlockOpFun (BOF)
 * @brief Block-sparse storage of an arbitrary time-dependent operator
 */
  class BlockOpFun : public BlockOp3D {
    public:
    BlockOpFun &operator+=(const BlockOpFun &A);
    int get_num_time_nodes() const;

    /**
     * @brief Constructor for BlockOpFun 
     * @param[in] block_indices vector of block-col-indices
     * @param[in] blocks vector of blocks
     * @note block_indices[i] = -1 if F does not have a block in col i
     */
    BlockOpFun(nda::vector_const_view<int> block_indices, std::vector<nda::array<dcomplex, 3>> &blocks);

    /**
     * @brief Constructor for BlockOpFun with blocks of zeros
     * @param[in] r number of imaginary time nodes
     * @param[in] block_indices vector of block-col-indices
     * @param[in] block_sizes vector of sizes of blocks
     */
    BlockOpFun(int r, nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes);
  };

  /**
 * @class BlockOpSymSet (BOSS)
 * @brief Container for (linear combinations of) creation/annihilation operators with the same block-sparse structure
 */
  class BlockOpSymSet : public BlockOp3D {
    public:
    int get_size_sym_set() const;

    /**
   * @brief Constructor for BlockOpSymSet
   * @param[in] block_indices vector of block-col-indices
   * @param[in] blocks vector of blocks
   * @note block_indices[i] = -1 if F does not have a block in col i
   */
    BlockOpSymSet(nda::vector_const_view<int> block_indices, std::vector<nda::array<dcomplex, 3>> &blocks);

    /**
   * @brief Constructor for BlockOpSymSet with blocks of zeros
   * @param[in] q number of operators in this set
   * @param[in] block_indices vector of block-col-indices
   * @param[in] block_sizes vector of sizes of blocks
   */
    BlockOpSymSet(int q, nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes);
  };

  /**
 * @class BlockOpSymSetBar
 * @brief Class for block-sparse storage of a set of F^bar operators with the same sparsity pattern
 */
  class BlockOpSymSetBar {
    protected:
    nda::vector<int> block_indices;
    std::vector<nda::array<dcomplex, 4>> blocks;
    int num_block_cols;

    public:
    void set_block_indices(nda::vector<int> &new_block_indices);
    void set_block_index(int i, int block_index);
    void set_blocks(std::vector<nda::array<dcomplex, 4>> &new_blocks);
    void set_block(int i, nda::array_const_view<dcomplex, 4> block);
    nda::vector_const_view<int> get_block_indices() const;
    int get_block_index(int i) const;
    const std::vector<nda::array<dcomplex, 4>> &get_blocks() const;
    nda::array_const_view<dcomplex, 4> get_block(int i) const;
    int get_block_size(int block_ind, int dim) const;
    int get_num_block_cols() const;
    int get_size_sym_set() const;
    int get_num_time_nodes() const;
    void add_block(int i, int s, int t, nda::array_const_view<dcomplex, 2> block); // add to block i, symmetry index s, time index t

    /**
   * @brief Constructor for BlockOpSymSetBar
   * @param[in] block_indices vector of block-col-indices
   * @param[in] blocks vector of blocks
   * @note block_indices[i] = -1 if F does not have a block in col i
   */
    BlockOpSymSetBar(nda::vector_const_view<int> block_indices, std::vector<nda::array<dcomplex, 4>> &blocks);

    /**
   * @brief Constructor for BlockOpSymSetBar with blocks of zeros
   * @param[in] q number of orbital indices associated with the block-sparse structure
   * @param[in] r rank of the DLR imaginary time object
   * @param[in] block_indices vector of block-col-indices
   * @param[in] block_sizes vector of sizes of blocks
   */
    BlockOpSymSetBar(int q, int r, nda::vector_const_view<int> block_indices, nda::array_const_view<int, 2> block_sizes);
  };

  /**
 * @class BlockOpSymQuartet (BOSQ)
 * @brief Container for multiple symmetry sets of BOSS 
 */
  class BlockOpSymQuartet {
    public:
    std::vector<BlockOpSymSet> Fs;
    std::vector<BlockOpSymSet> F_dags;
    std::vector<BlockOpSymSetBar> F_dag_bars;
    std::vector<BlockOpSymSetBar> F_bars_refl;
    nda::vector<long> sym_set_labels;   // TODO comment
    nda::vector<long> sym_set_inds;     // mapping from backbone orbital index to index within the symmetry set
    nda::vector<long> sym_set_sizes;    // sizes of the symmetry sets
    nda::array<long, 2> sym_set_to_orb; // mapping from symmetry set index to backbone orbital index
    long p;                             // number of hybridization poles

    /** 
   * @brief Constructor for BlockOpSymQuartet
   * @param[in] Fs vector of annihilation operator BOSS
   * @param[in] F_dags vector of creation operator BOSS
   * @param[in] hyb_coeffs DLR coefficients of hybridization
   * @param[in] sym_set_labels vector of symmetry set labels for each orbital index
   */
    BlockOpSymQuartet(std::vector<BlockOpSymSet> Fs, std::vector<BlockOpSymSet> F_dags, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                      nda::vector_const_view<long> sym_set_labels);
  };

  /**
 * @brief Print BlockDiagOpFun to output stream
 * @param[in] os output stream
 * @param[in] D BlockDiagOpFun
 */
  std::ostream &operator<<(std::ostream &os, BlockDiagOpFun &D);

  /**
 * @brief Print BlockOp to output stream
 * @param[in] os output stream
 * @param[in] F BlockOp
 */
  std::ostream &operator<<(std::ostream &os, BlockOp &F);

  /**
 * @brief Print BlockOp3D to output stream
 * @param[in] os output stream
 * @param[in] F BlockOp3D
 */
  std::ostream &operator<<(std::ostream &os, BlockOp3D &F);

  /**
 * @brief Print BlockOpSymSetBar to output stream
 * @param[in] os output stream
 * @param[in] F BlockOpSymSetBar
 */
  std::ostream &operator<<(std::ostream &os, BlockOpSymSetBar &F);

  /**
 * @brief Compute the adjoint of a BlockOp
 * @param[in] F BlockOp
 * @return F^dagger operator
 */
  BlockOp dagger_bs(BlockOp const &F);

  /**
  * @brief Compute a product between an integer and a BlockDiagOpFun
  * @param[in] i integer
  * @param[in] D BlockDiagOpFun
  * @return i * D
  */
  BlockDiagOpFun operator*(int i, BlockDiagOpFun const &D);

  /**
 * @brief Compute a product between a scalar and a BlockOp
 * @param[in] c dcomplex
 * @param[in] F BlockOp
 */
  BlockOp operator*(const dcomplex c, const BlockOp &F);

  /**
 * @brief Compute a product between a scalar and a BlockOp3D
 * @param[in] c dcomplex
 * @param[in] F BlockOp3D
 */
  BlockOp3D operator*(const dcomplex c, const BlockOp3D &F);

  /**
 * @brief Convert a BlockOpFun with diagonal structure to a BlockDiagOpFun
 * @param[in] A BlockOpFun
 * @return BlockDiagOpFun
 */
  BlockDiagOpFun BOFtoBDOF(BlockOpFun const &A);

  /**
 * @brief Compute noninteracting Green's function from Hamiltonian as a BDOF
 * @param[in] H_blocks vector of blocks of Hamiltonian
 * @param[in] H_block_inds vector, -1 if Hamiltonian has zero block in corresponding block column
 * @param[in] beta inverse temperature
 * @param[in] dlr_it_abs imaginary time nodes in absolute format
 */
  BlockDiagOpFun nonint_gf_BDOF(std::vector<nda::array<double, 2>> H_blocks, nda::vector<int> H_block_inds, double beta,
                                nda::vector_const_view<double> dlr_it_abs);

  /**
 * @brief Convert a BlockDiagOpFun to a triqs::gfs::block_gf<triqs::mesh::dlr_imtime>
 * @param[in] BDOF BlockDiagOpFun
 * @param[in] beta inverse temperature
 * @param[in] Lambda DLR cutoff parameter
 * @param[in] eps DLR epsilon parameter
 */
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> BDOF_to_block_gf(BlockDiagOpFun const &BDOF, double beta, double Lambda, double eps);

  /**
   * @brief Compute the expectation value of the 2nd quantized operator op, <O> = -Tr[O G(\beta)]
   * using the AtomDiag instance ad to generate a block representation
   * and tracing with the many-body density matrix of the pseudo particle Green's function G_ppsc
   * @param[in] op 2nd quantization operator
   * @param[in] ad AtomDiag object
   * @param[in] G_ppsc Pseudo-particle Green's function
   * @return Expectation value -Tr[G(\beta) O]
   */
  dcomplex expectation_value(
    triqs::operators::many_body_operator_real const &op, 
    triqs::atom_diag::atom_diag<false> const &ad, 
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc); 

  /**
   * @brief Compute the Volterra "convolution" of two pseudo-particle Green's functions
   * @param[in] G1 Left Green's function
   * @param[in] G2 Right Green's function
   * @return Convolution (G1 * G2)(\tau)
   */
  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> convolve_ppsc(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G1, 
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G2);

  /**
   * @brief Take the trace of a pseudo-particle Green's function
   * @param[in] G Pseudo-particle Green's function
   * @returns Trace Tr[G(\beta)]
   */
  dcomplex trace(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G);


} // namespace triqs_xca::block_sparse