#pragma once
#include <cppdlr/dlr_kernels.hpp>
#include <nda/declarations.hpp>
#include <triqs_xca/block_sparse.hpp>
#include <nda/nda.hpp>
#include <triqs_xca/backbone.hpp>

using namespace nda;

/**
 * @class DiagramEvaluator
 * @brief Class for evaluating a diagram of a given order and topology
 * This class is used to evaluate all the backbone decompositions of a given
 * order and topology. It reads the information from a Backbone object and
 * contains the Green's functions and creation/annihilation operators needed to
 * actually compute the diagram. It also contains temporary arrays required for
 * computation. 
 */
class DiagramEvaluator {
  public:
  double beta;                      // inverse temperature
  imtime_ops itops;                 // DLR imaginary time object
  nda::array<dcomplex, 3> hyb;      // hybridization function at imaginary time nodes
  nda::array<dcomplex, 3> hyb_refl; // hybridization function at (beta - tau) nodes
  nda::array<dcomplex, 3> Gt;       // Green's function at imaginary time nodes
  DenseFSet Fset;                   // DenseFSet (cre/ann operators with and without bars)
  nda::vector<double> dlr_it;       // DLR imaginary time nodes in relative ordering
  int r;                            // DLR rank
  nda::vector<double> hyb_poles;    // hybridization poles
  nda::array<dcomplex, 3> Sigma;    // array for storing self-energy contribution (final result)
  nda::array<dcomplex, 3> T;        // array for storing intermediate result
  nda::array<dcomplex, 3> GKt;      // array for storing result of edge computation
  nda::array<dcomplex, 4> Tkaps;    // intermediate storage array
  nda::array<dcomplex, 3> Tmu;      // intermediate storage array

  void multiply_vertex_dense(Backbone &backbone,
                             int v_ix);                           // multiply by a single vertex, v_ix, in a backbone diagram using dense storage
  void compose_with_edge_dense(Backbone &backbone, int e_ix);     // convolve with a single edge, e_ix, in a backbone diagram using dense storage
  void reset();                                                   // reset all arrays to zero
  void multiply_zero_vertex(Backbone &backbone, bool is_forward); // multiply by the zero vertex and the vertex connected to zero
  void eval_diagram_dense(
     Backbone &
        backbone); // evaluate a diagram of a given order and topology in dense storage (i.e., evaluate and sum all backbones with different orbital indices, poles, and hybridization line directions)
  void eval_backbone_fixed_indices_dense(
     Backbone &backbone); // evaluate a diagram with fixed orbital indices, poles, and line directions in dense storage, including prefactor

  /**
   * @brief Constructor for DiagramEvaluator
   * 
   * @param[in] beta inverse temperature
   * @param[in] itops DLR imaginary time object
   * @param[in] hyb hybridization function at imaginary time nodes
   * @param[in] hyb_refl hybridization function at (beta - tau) nodes
   * @param[in] Gt Green's function at imaginary time nodes
   * @param[in] Fset DenseFSet (cre/ann operators with and without bars)
   */
  DiagramEvaluator(double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl,
                   nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset);
};

/**
 * @class GreensFunctionDiagramEvaluator
 * @brief Class for evaluating a Green's function diagram of a given order and topology
 * This class is used to evaluate all the backbone decompositions of a given
 * order and topology for a Green's function diagram. It reads the information
 * from a GreensFunctionBackbone object and contains the pseudoparticle Green's functions and
 * creation/annihilation operators needed to actually compute the diagram. It also contains
 * temporary arrays required for computation.
 */
class GreensFunctionDiagramEvaluator {
  public:
  double beta;                      // inverse temperature
  imtime_ops itops;                 // DLR imaginary time object
  nda::array<dcomplex, 3> hyb;      // hybridization function at imaginary time nodes
  nda::array<dcomplex, 3> hyb_refl; // hybridization function at (beta - tau) nodes
  nda::array<dcomplex, 3> Gt;       // Green's function at imaginary time nodes
  DenseFSet Fset;                   // DenseFSet (cre/ann operators with and without bars)
  nda::vector<double> dlr_it;       // DLR imaginary time nodes in relative ordering
  int r;                            // DLR rank
  nda::vector<double> hyb_poles;    // hybridization poles
  nda::array<dcomplex, 3> Sigma;    // array for storing self-energy contribution (final result)
  nda::array<dcomplex, 3> T;        // array for storing intermediate result
  nda::array<dcomplex, 3> GKt;      // array for storing result of edge computation
  nda::array<dcomplex, 4> Tkaps;    // intermediate storage array
  nda::array<dcomplex, 3> Tmu;      // intermediate storage array
};