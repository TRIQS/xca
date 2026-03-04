#pragma once
#include <nda/nda.hpp>

#include <cppdlr/dlr_imtime.hpp>

#include <triqs/gfs.hpp>

#include "triqs_xca/dense.hpp"
#include "triqs_xca/backbone.hpp"
#include "triqs_xca/atom_diag_utils.hpp"

#ifdef __clang__
#define C2PY_IGNORE __attribute__((annotate("c2py_ignore")))
#else
#define C2PY_IGNORE
#endif

namespace triqs_xca::dense {

    using namespace nda;

    using imtime_ops = cppdlr::imtime_ops;

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
      triqs::mesh::dlr_imtime tau_mesh;  // imaginary time mesh
      double beta;                      // inverse temperature
      imtime_ops itops;                 // DLR imaginary time object
      nda::vector<double> dlr_it;       // DLR imaginary time nodes in relative ordering

      nda::array<dcomplex, 3> hyb;      // hybridization function at imaginary time nodes
      nda::array<dcomplex, 3> hyb_refl; // hybridization function at (beta - tau) nodes
      nda::vector<double> hyb_poles;    // hybridization poles

      nda::array<dcomplex, 3> Gt;       // Green's function at imaginary time nodes
      C2PY_IGNORE DenseFSet Fset;       // DenseFSet (cre/ann operators with and without bars)

      int r;                            // DLR rank
      int n;                            // number of orbitals
      int N;                            // number of many body states

      nda::array<dcomplex, 3> Sigma;    // array for storing self-energy contribution (final result)
      nda::array<dcomplex, 3> T;        // array for storing intermediate result
      nda::array<dcomplex, 3> U;        // array for storing intermediate result (left side of correlator diagram)
      nda::array<dcomplex, 3> GKt;      // array for storing result of edge computation
      nda::array<dcomplex, 4> Tkaps;    // intermediate storage array
      nda::array<dcomplex, 3> Tmu;      // intermediate storage array

      private:

      // routines for any diagram
      
      // multiply by a single vertex, v_ix, in a backbone diagram using dense storage
      void multiply_left_vertex(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone, int v_ix); 
      // convolve with a single edge, e_ix, in a backbone diagram using dense storage
      void integrate_left_edge(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone, int e_ix); 
      // multiply by the prefactor associated with the backbone
      void multiply_prefactor(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone); 

      // routines for self-energy diagrams
      
      // multiply by the zero vertex and the vertex connected to zero
      void multiply_left_vertex_and_right_zero_vertex(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone, int vct0); 

      // routines for correlator diagrams

      // multiply by a single vertex, v_ix, on the tau-beta side of a correlator backbone diagram using dense storage
      void multiply_right_vertex(nda::array_view<dcomplex, 3> U_buf, Backbone &backbone, int v_ix);
      // convolve with a single edge, e_ix, on the tau-beta side of a correlator backbone diagram using dense storage
      void integrate_right_edge(nda::array_view<dcomplex, 3> U_buf, Backbone &backbone, int e_ix);

      public:

      void reset(); // reset all arrays to zero

      C2PY_IGNORE int get_num_self_energy_backbones(Backbone &backbone);                  // get number of backbones for given topology
      C2PY_IGNORE void eval_self_energy(Backbone &backbone);                        // evaluate a diagram of a given order and topology in dense storage
      // (i.e., evaluate and sum all backbones with different orbital indices, poles, and hybridization line directions)
      C2PY_IGNORE void eval_self_energy_fixed_indices(Backbone &backbone, int f_ix);
      // evaluate a diagram with fixed orbital indices, poles, and line directions in dense storage, including prefactor

      int get_num_self_energy_backbones(nda::array_const_view<int, 2> topology);                  // get number of backbones for given topology
      triqs::gfs::gf<triqs::mesh::dlr_imtime> compute_self_energy(nda::array_const_view<int, 2> topology);           // compute self-energy for given topology
      triqs::gfs::gf<triqs::mesh::dlr_imtime> compute_self_energy(nda::array_const_view<int, 2> topology, int f_ix); // compute self-energy for given topology and flat index

      C2PY_IGNORE nda::array<dcomplex, 3> eval_correlator(CorrelatorBackbone &backbone, nda::array<dcomplex, 3> mu_ops, nda::array<dcomplex, 3> kap_ops);
      // evaluate the mu, kap entries of a correlator for a diagram of a given order and topology in dense storage
      C2PY_IGNORE void eval_correlator_fixed_indices(CorrelatorBackbone &backbone);
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
      C2PY_IGNORE DenseDiagramEvaluator(double beta, double eps, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl,
                            nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset);

      /**
       * @brief Constructor for DiagramEvaluator
       * @param[in] hyb_poles hybridization poles
       * @param[in] hyb_coeffs hybridization function coefficients (at poles)
       * @param[in] G_ppsc TRIQS pseudo-particle Green's function
       * @param[in] ad atom_diag object with Hamiltonian and field operators
       */
      DenseDiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                            triqs::gfs::gf_view<triqs::mesh::dlr_imtime> G_ppsc, triqs::atom_diag::atom_diag<false> const &ad);

      virtual ~DenseDiagramEvaluator() = default;
    };

} // namespace triqs_xca::dense