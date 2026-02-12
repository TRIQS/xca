#pragma once
#include <cppdlr/dlr_kernels.hpp>
#include <nda/blas/tools.hpp>
#include <nda/declarations.hpp>
#include <triqs/atom_diag/atom_diag.hpp>
#include <triqs_xca/block_sparse.hpp>
#include <nda/nda.hpp>
#include <triqs_xca/backbone.hpp>

#ifdef __clang__
#define C2PY_IGNORE __attribute__((annotate("c2py_ignore")))
#else
#define C2PY_IGNORE
#endif

using namespace nda;

/**
 * @class DiagramEvaluator
 * @brief Class for evaluating a diagram of a given order and topology in block-sparse storage
 * This class is used to evaluate all the backbone decompositions of a given order and topology. It reads the information from a Backbone object
 * and contains the Green's functions and creation/annihilation operators needed to actually compute the diagram. It also contains temporary 
 * data structures required for computation.
 */
class DiagramEvaluator {
  private:
  imtime_ops itops;           // DLR imaginary time object
  nda::vector<double> dlr_it; // DLR imaginary time nodes in relative ordering
  BlockDiagOpFun Gt;          // Green's function at imaginary time nodes
  BlockOpSymQuartet Fq;       // BlockOpSymQuartet (field operators with and without bars)
  BlockDiagOpFun Sigma;       // array for storing self-energy contribution (final result)
  mesh::dlr_imtime tau_mesh;  // imaginary time mesh

  // routines for any diagram
  void multiply_vertex_block(Backbone &backbone, int v_ix, nda::vector_const_view<int> ind_path, nda::vector_const_view<int> block_dims);
  // for block b_ix, multiply by a single vertex, v_ix, in a backbone diagram
  void compose_with_edge_block(Backbone &backbone, int e_ix, nda::vector_const_view<int> ind_path, nda::vector_const_view<int> block_dims);
  // for block b_ix, convolve with a single edge, e_ix, in a backbone diagram
  void multiply_prefactor(Backbone &backbone);

  // routines for self-energy diagrams
  void multiply_zero_vertex_block(Backbone &backbone, bool is_forward, int b_ix_0, int p_kap, int p_mu, nda::vector_const_view<int> ind_path,
                                  nda::vector_const_view<int> block_dims); // multiply by the zero vertex and the vertex connected to zero
  BlockDiagOpFun &get_self_energy();                                       // get the self-energy result
  void find_path_self_energy(Backbone &backbone, int f_ix, nda::vector_view<int> ind_path, nda::vector_view<int> block_dims);
  void eval_self_energy(Backbone &backbone, int f_ix); // evaluate a particular backbone diagram
  void eval_self_energy(Backbone &backbone);           // evaluate a diagram of a given order and topology in block-sparse storage
  void eval_self_energy_fixed_indices(
     Backbone &backbone, int b_ix, int p_kap, int p_mu, nda::vector_const_view<int> ind_path,
     nda::vector_const_view<int>
        block_dims); // evaluate a diagram with fixed orbital indices, poles, and line directions in dense storage, including prefactor

  // routines for correlator diagrams
  void multiply_vertex_corr_block(CorrelatorBackbone &backbone, int v_ix, nda::vector_const_view<int> ind_path,
                                  nda::vector_const_view<int> block_dims);
  // multiply_vertex_block on the left side of a correlator diagram
  void compose_with_edge_corr_block(CorrelatorBackbone &backbone, int e_ix, nda::vector_const_view<int> ind_path,
                                    nda::vector_const_view<int> block_dims);
  // compose_with_edge_block on the left side of a correlator diagram

  public:
  double beta;                   // inverse temperature
  int r;                         // rank of the DLR imaginary time object
  int n;                         // number of orbitals
  int q;                         // number of symmetry sets
  int Nmax;                      // maximum block size in the Green's function
  nda::array<dcomplex, 3> hyb;   // hybridization function at imaginary time nodes
  nda::vector<double> hyb_poles; // hybridization poles
  nda::array<dcomplex, 3> T;     // array for storing intermediate result
  nda::array<dcomplex, 3> U;     // array for storing intermediate result (tau-beta side of correlator diagram)
  nda::array<dcomplex, 3> GKt;   // array for storing result of edge computation
  nda::array<dcomplex, 4> Tkaps; // intermediate storage array
  nda::array<dcomplex, 3> Tmu;   // intermediate storage array

  // routines for any diagram
  void reset(); // reset all arrays to zero

  // routines for self-energy diagrams
  int get_num_self_energy_backbones(nda::array_const_view<int, 2> topology);                  // get number of backbones for given topology
  block_gf<dlr_imtime> compute_self_energy(nda::array_const_view<int, 2> topology);           // compute self-energy for given topology
  block_gf<dlr_imtime> compute_self_energy(nda::array_const_view<int, 2> topology, int f_ix); // compute self-energy for given topology and flat index
  void print_self_energy_backbone(nda::array_const_view<int, 2> topology, int f_ix); // print the backbone corresponding to a given flat index for debugging

  // routines for correlator diagrams
  int get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology); // get number of backbones for given topology
  C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(CorrelatorBackbone &backbone, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops);
  C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(CorrelatorBackbone &backbone, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops,
                                                      int f_ix);
  nda::array<dcomplex, 3> compute_single_ptcle_gf(nda::array_const_view<int, 2> topology);
  // compute single particle Green's function for given topology
  nda::array<dcomplex, 3> compute_single_ptcle_gf(nda::array_const_view<int, 2> topology, int f_ix);
  // compute single particle Green's function for given topology and flat index
  void print_single_ptcle_gf_backbone(nda::array_const_view<int, 2> topology, int f_ix); // print the backbone corresponding to a given flat index for debugging

  /**
   * @brief Constructor for DiagramEvaluator
   * @param[in] beta inverse temperature
   * @param[in] Lambda DLR imaginary time cutoff
   * @param[in] eps DLR imaginary time accuracy
   * @param[in] hyb_poles hybridization poles
   * @param[in] hyb_coeffs hybridization function coefficients at imaginary time nodes
   * @param[in] G_ppsc pseudo-particle Green's function at imaginary time nodes
   * @param[in] ad atom_diag object with Hamiltonian and field operators
   */
  DiagramEvaluator(double beta, double Lambda, double eps, nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                   block_gf_view<dlr_imtime> G_ppsc, triqs::atom_diag::atom_diag<false> const &ad);

  /**
   * @brief Old constructor for DiagramEvaluator
   * @param[in] beta inverse temperature
   * @param[in] Lambda DLR imaginary time cutoff
   * @param[in] eps DLR imaginary time accuracy
   * @param[in] hyb hybridization function at imaginary time nodes
   * @param[in] Gt Green's function at imaginary time nodes
   * @param[in] Fset BlockOpSymQuartet (cre/ann operators with and without bars)
   */
  C2PY_IGNORE DiagramEvaluator(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> hyb,
                               nda::vector_const_view<double> hyb_poles, BlockDiagOpFun &Gt, BlockOpSymQuartet &Fq);

  virtual ~DiagramEvaluator() = default;
};