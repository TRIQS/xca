#pragma once
#include <cppdlr/dlr_kernels.hpp>
#include <nda/blas/tools.hpp>
#include <nda/declarations.hpp>
#include <triqs_xca/block_sparse.hpp>
#include <nda/nda.hpp>
#include <triqs_xca/backbone.hpp>

using namespace nda;

/**
 * @class DiagramEvaluator
 * @brief Class for evaluating a diagram of a given order and topology in block-sparse storage
 * This class is used to evaluate all the backbone decompositions of a given order and topology. It reads the information from a Backbone object
 * and contains the Green's functions and creation/annihilation operators needed to actually compute the diagram. It also contains temporary 
 * data structures required for computation.
 */
class DiagramEvaluator {
  public:
  double beta;                      // inverse temperature
  int r;                            // rank of the DLR imaginary time object
  int n;                            // number of orbitals
  int q;                            // number of symmetry sets
  int Nmax;                         // maximum block size in the Green's function
  imtime_ops itops;                 // DLR imaginary time object
  nda::array<dcomplex, 3> hyb;      // hybridization function at imaginary time nodes
  nda::array<dcomplex, 3> hyb_refl; // hybridization function at (beta - tau) nodes
  BlockDiagOpFun Gt;                // Green's function at imaginary time nodes
  BlockOpSymQuartet Fq;             // BlockOpSymQuartet (field operators with and without bars)
  nda::vector<double> dlr_it;       // DLR imaginary time nodes in relative ordering
  nda::vector<double> hyb_poles;    // hybridization poles
  BlockDiagOpFun Sigma;             // array for storing self-energy contribution (final result)
  nda::array<dcomplex, 3> T;        // array for storing intermediate result
  nda::array<dcomplex, 3> U;        // array for storing intermediate result (tau-beta side of correlator diagram)
  nda::array<dcomplex, 3> GKt;      // array for storing result of edge computation
  nda::array<dcomplex, 4> Tkaps;    // intermediate storage array
  nda::array<dcomplex, 3> Tmu;      // intermediate storage array

  // routines for any diagram
  void reset(); // reset all arrays to zero
  void multiply_vertex_block(Backbone &backbone, int v_ix, nda::vector_const_view<int> ind_path, nda::vector_const_view<int> block_dims);
  // for block b_ix, multiply by a single vertex, v_ix, in a backbone diagram
  void compose_with_edge_block(Backbone &backbone, int e_ix, nda::vector_const_view<int> ind_path, nda::vector_const_view<int> block_dims);
  // for block b_ix, convolve with a single edge, e_ix, in a backbone diagram
  void multiply_prefactor(Backbone &backbone, nda::vector_const_view<int> block_dims); // multiply by the prefactor associated with the backbone

  // routines for self-energy diagrams
  void multiply_zero_vertex_block(Backbone &backbone, bool is_forward, int b_ix_0, int p_kap, int p_mu, nda::vector_const_view<int> ind_path,
                                  nda::vector_const_view<int> block_dims); // multiply by the zero vertex and the vertex connected to zero
  void eval_self_energy(Backbone &backbone);                               // evaluate a diagram of a given order and topology in block-sparse storage
  void eval_self_energy_fixed_indices(
     Backbone &backbone, int b_ix, int p_kap, int p_mu, nda::vector_const_view<int> ind_path,
     nda::vector_const_view<int>
        block_dims); // evaluate a diagram with fixed orbital indices, poles, and line directions in dense storage, including prefactor

  // routines for correlator diagrams
  void multiply_vertex_corr_block(Backbone &backbone, int v_ix, nda::vector_const_view<int> ind_path, nda::vector_const_view<int> block_dims);
  // multiply_vertex_block on the left side of a correlator diagram
  void compose_with_edge_corr_block(Backbone &backbone, int e_ix, nda::vector_const_view<int> ind_path, nda::vector_const_view<int> block_dims);
  // compose_with_edge_block on the left side of a correlator diagram
  nda::array<dcomplex, 3> eval_correlator(Backbone &backbone, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops);
  // evaluate a diagram of a given order and topology in block-sparse storage
  void eval_correlator_fixed_indices(Backbone &backbone, int b_ix, int p_kap, int p_mu, nda::vector_const_view<int> ind_path,
                                     nda::vector_const_view<int> block_dims);
  // evaluate a diagram with fixed orbital indices, poles, and line directions in dense storage, including prefactor

  /**
   * @brief Constructor for DiagramEvaluator
   * @param[in] beta inverse temperature
   * @param[in] itops DLR imaginary time object
   * @param[in] hyb hybridization function at imaginary time nodes
   * @param[in] hyb_refl hybridization function at (beta - tau) nodes
   * @param[in] Gt Green's function at imaginary time nodes
   * @param[in] Fset BlockOpSymQuartet (cre/ann operators with and without bars)
   */
  DiagramEvaluator(double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl,
                   nda::vector_const_view<double> hyb_poles, BlockDiagOpFun &Gt, BlockOpSymQuartet &Fq);

  virtual ~DiagramEvaluator() = default;
};