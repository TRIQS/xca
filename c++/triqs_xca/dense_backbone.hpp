#pragma once
#include <cppdlr/dlr_kernels.hpp>
#include <nda/declarations.hpp>
#include <triqs_xca/block_sparse.hpp>
#include <nda/nda.hpp>
#include <triqs_xca/backbone.hpp>

using namespace nda;

/**
 * @class DenseDiagramEvaluator
 * @brief Class for evaluating a diagram of a given order and topology
 * This class is used to evaluate all the backbone decompositions of a given
 * order and topology. It reads the information from a Backbone object and
 * contains the Green's functions and creation/annihilation operators needed to
 * actually compute the diagram. It also contains temporary arrays required for
 * computation. 
 */
class DenseDiagramEvaluator {
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
  nda::array<dcomplex, 3> U;        // array for storing intermediate result (left side of correlator diagram)

  // routines for any diagram
  void reset(); // reset all arrays to zero
  void multiply_left_vertex(Backbone &backbone,
                       int v_ix);                       // multiply by a single vertex, v_ix, in a backbone diagram using dense storage
  void integrate_left_edge(Backbone &backbone, int e_ix); // convolve with a single edge, e_ix, in a backbone diagram using dense storage
  void multiply_prefactor(Backbone &backbone);          // multiply by the prefactor associated with the backbone

  // routines for self-energy diagrams
  void multiply_left_vertex_and_right_zero_vertex(Backbone &backbone, bool is_forward); // multiply by the zero vertex and the vertex connected to zero
  void eval_self_energy(Backbone &backbone);                      // evaluate a diagram of a given order and topology in dense storage
  // (i.e., evaluate and sum all backbones with different orbital indices, poles, and hybridization line directions)
  void eval_self_energy_fixed_indices(Backbone &backbone);
  // evaluate a diagram with fixed orbital indices, poles, and line directions in dense storage, including prefactor

  // routines for correlator diagrams
  void multiply_right_vertex(Backbone &backbone, int v_ix);
  // multiply by a single vertex, v_ix, on the tau-beta side of a correlator backbone diagram using dense storage
  void integrate_right_edge(Backbone &backbone, int e_ix);
  // convolve with a single edge, e_ix, on the tau-beta side of a correlator backbone diagram using dense storage

  nda::array<dcomplex, 3> eval_correlator(CorrelatorBackbone &backbone, nda::array<dcomplex, 3> mu_ops, nda::array<dcomplex, 3> kap_ops);
  // evaluate the mu, kap entries of a correlator for a diagram of a given order and topology in dense storage
  void eval_correlator_fixed_indices(CorrelatorBackbone &backbone);
  // evaluate a correlator diagram with fixed orbital indices, poles, and line directions in dense storage, including prefactor

  /**
   * @brief Constructor for DenseDiagramEvaluator
   * 
   * @param[in] beta inverse temperature
   * @param[in] itops DLR imaginary time object
   * @param[in] hyb hybridization function at imaginary time nodes
   * @param[in] hyb_refl hybridization function at (beta - tau) nodes
   * @param[in] hyb_poles hybridization poles
   * @param[in] Gt Green's function at imaginary time nodes
   * @param[in] Fset DenseFSet (cre/ann operators with and without bars)
   */
  DenseDiagramEvaluator(double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl,
                        nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset);

  virtual ~DenseDiagramEvaluator() = default;
};
