#include "triqs_xca/backbone.hpp"
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <cppdlr/utils.hpp>
#include <iostream>
#include <nda/algorithms.hpp>
#include <nda/declarations.hpp>
#include <nda/mapped_functions.hxx>
#include <nda/nda.hpp>
#include <triqs/mesh/dlr_imtime.hpp>
#include <triqs/mesh/imtime.hpp>
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs_xca/atom_diag_utils.hpp>

DiagramEvaluator::DiagramEvaluator(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> hyb,
                                   nda::vector_const_view<double> hyb_poles, BlockDiagOpFun &Gt, BlockOpSymQuartet &Fq)
   : itops(imtime_ops(Lambda, build_dlr_rf(Lambda, eps))),
     Gt(Gt),
     Fq(Fq),
     Sigma(itops.rank(), Gt.get_block_sizes()),
     tau_mesh(mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps)),
     beta(beta),
     r(itops.rank()),
     n(hyb.extent(1)),
     q(nda::max_element(Fq.sym_set_labels) + 1),
     Nmax(Gt.get_max_block_size()),
     hyb(hyb),
     hyb_poles(beta * hyb_poles) {

  dlr_it = itops.get_itnodes();

  // allocate arrays
  T     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  U     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  GKt   = nda::zeros<dcomplex>(r, Nmax, Nmax);
  Tkaps = nda::zeros<dcomplex>(n, r, Nmax, Nmax);
  Tmu   = nda::zeros<dcomplex>(r, Nmax, Nmax);
}

DiagramEvaluator::DiagramEvaluator(double beta, double Lambda, double eps, nda::vector_const_view<double> hyb_poles,
                                   nda::array_const_view<dcomplex, 3> hyb_coeffs, block_gf_view<dlr_imtime> G_ppsc,
                                   triqs::atom_diag::atom_diag<false> const &ad)
   : itops(imtime_ops(Lambda, build_dlr_rf(Lambda, eps))),
     dlr_it(itops.get_itnodes()),
     Gt(BlockDiagOpFun(G_ppsc)),
     Fq(std::get<0>(get_operators(ad, hyb_coeffs))),
     Sigma(itops.rank(), Gt.get_block_sizes()),
     tau_mesh(mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps)),
     beta(beta),
     r(itops.rank()),
     n(hyb_coeffs.extent(1)),
     q(nda::max_element(Fq.sym_set_labels) + 1),
     Nmax(Gt.get_max_block_size()),
     hyb_poles(beta * hyb_poles) {

  // hyb on DLR imaginary time nodes
  hyb = aaa_coefs2vals(beta, Lambda, eps, hyb_coeffs, hyb_poles);

  // allocate arrays
  T     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  U     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  GKt   = nda::zeros<dcomplex>(r, Nmax, Nmax);
  Tkaps = nda::zeros<dcomplex>(n, r, Nmax, Nmax);
  Tmu   = nda::zeros<dcomplex>(r, Nmax, Nmax);
}

// ----------- Private routines for any diagram ==========

void DiagramEvaluator::multiply_vertex_block(Backbone &backbone, int v_ix, nda::vector_const_view<int> ind_path,
                                             nda::vector_const_view<int> block_dims) {
  int o_ix = backbone.get_vertex_orb(v_ix); // orbital_index
  // split backbone orbital index into symmetry set index and orbital index within the symmetry set
  // i.e. have mapping between backbone orbital index and symmetry set index
  int q_ix    = static_cast<int>(Fq.sym_set_labels(o_ix)); // symmetry set index
  int qo_ix   = static_cast<int>(Fq.sym_set_inds(o_ix));   // index within the symmetry set
  int l_ix    = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));
  int n_col_r = v_ix < backbone.get_topology(0, 1) ? block_dims(1) : block_dims(0);
  int b_ix    = ind_path(v_ix - 1); // block index for the vertex v_ix

  if (backbone.has_vertex_bar(v_ix)) {   // F has bar
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           matmul(Fq.F_dag_bars[q_ix].get_block(b_ix)(qo_ix, l_ix, _, _), T(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    } else {
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           -matmul(Fq.F_bars_refl[q_ix].get_block(b_ix)(qo_ix, l_ix, _, _), T(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    }
  } else {
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           matmul(Fq.F_dags[q_ix].get_block(b_ix)(qo_ix, _, _), T(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    } else {
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           matmul(Fq.Fs[q_ix].get_block(b_ix)(qo_ix, _, _), T(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    }
  }

  // K factor
  int bv      = backbone.get_vertex_Ksign(v_ix); // sign on K
  double pole = hyb_poles(l_ix);
  if (backbone.get_vertex_direction(v_ix) == 0) pole = -pole; // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR BACKWARD LINES
  if (bv != 0) {
    for (int t = 0; t < r; t++) {
      T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) = k_it(dlr_it(t), bv * pole) * T(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r));
    }
  }
}

void DiagramEvaluator::compose_with_edge_block(Backbone &backbone, int e_ix, nda::vector_const_view<int> ind_path,
                                               nda::vector_const_view<int> block_dims) {

  int n_col_r                                                            = e_ix < backbone.get_topology(0, 1) ? block_dims(1) : block_dims(0);
  int b_ix                                                               = ind_path(e_ix); // block index for the edge e_ix
  GKt(_, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1))) = Gt.get_block(b_ix);
  int m                                                                  = backbone.m;
  for (int x = 0; x < m - 1; x++) {
    int be      = backbone.get_edge(e_ix, x); // sign on K
    double pole = hyb_poles(backbone.get_pole_ind(x));
    if (backbone.get_fb(x + 1) == 0) pole = -pole; // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR
    if (be != 0) {
      for (int t = 0; t < r; t++) {
        GKt(t, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1))) =
           k_it(dlr_it(t), be * pole) * GKt(t, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1)));
      }
    }
  }
  T(_, range(0, block_dims(e_ix + 1)), range(0, n_col_r)) =
     itops.convolve(beta, itops.vals2coefs(GKt(_, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1)))),
                    itops.vals2coefs(T(_, range(0, block_dims(e_ix + 1)), range(0, n_col_r))), TIME_ORDERED);
}

void DiagramEvaluator::multiply_prefactor(Backbone &backbone) {
  int m = backbone.m;
  // Multiply by prefactor
  for (int m_ix = 0; m_ix < m - 1; m_ix++) {
    int exp = backbone.get_prefactor_Kexp(m_ix);
    if (exp != 0) {
      int Ksign = backbone.get_prefactor_Ksign(m_ix);
      double om = hyb_poles(backbone.get_pole_ind(m_ix));
      if (backbone.get_fb(m_ix + 1) == 0) om = -om; // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR BACKWARD LINES
      double k = k_it(0, Ksign * om);
      for (int x = 0; x < exp; x++) T /= k;
    }
  }
}

// ========== Private self-energy routines ==========

void DiagramEvaluator::multiply_zero_vertex_block(Backbone &backbone, bool is_forward, int b_ix_0, int p_kap, int p_mu,
                                                  nda::vector_const_view<int> ind_path, nda::vector_const_view<int> block_dims) {

  int b_ix_mu = ind_path(backbone.get_topology(0, 1) - 1); // block index for F_mu
  if (is_forward) {
    for (int kap = 0; kap < Fq.sym_set_sizes(p_kap); kap++) {
      for (int t = 0; t < r; t++) {
        Tkaps(kap, t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))) =
           matmul(T(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(1))), Fq.Fs[p_kap].get_block(b_ix_0)(kap, _, _));
      }
    }
    T = 0;
    for (int mu = 0; mu < Fq.sym_set_sizes(p_mu); mu++) {
      Tmu = 0;
      for (int kap = 0; kap < Fq.sym_set_sizes(p_kap); kap++) {
        for (int t = 0; t < r; t++) {
          Tmu(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))) +=
             hyb(t, Fq.sym_set_to_orb(p_mu, mu), Fq.sym_set_to_orb(p_kap, kap))
             * Tkaps(kap, t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0)));
        }
      }
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(backbone.get_topology(0, 1) + 1)), range(0, block_dims(0))) +=
           matmul(Fq.F_dags[p_mu].get_block(b_ix_mu)(mu, _, _), Tmu(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))));
      }
    }
  } else {
    for (int kap = 0; kap < Fq.sym_set_sizes(p_kap); kap++) {
      for (int t = 0; t < r; t++) {
        Tkaps(kap, t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))) =
           matmul(T(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(1))), Fq.F_dags[p_kap].get_block(b_ix_0)(kap, _, _));
      }
    }
    T = 0;
    for (int mu = 0; mu < Fq.sym_set_sizes(p_mu); mu++) {
      Tmu = 0;
      for (int kap = 0; kap < Fq.sym_set_sizes(p_kap); kap++) {
        for (int t = 0; t < r; t++) {
          Tmu(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))) -=
             hyb(t, Fq.sym_set_to_orb(p_mu, mu), Fq.sym_set_to_orb(p_kap, kap))
             * Tkaps(kap, t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0)));
        }
      }
      for (int t = 0; t < r; t++) {
        T(t, range(0, block_dims(backbone.get_topology(0, 1) + 1)), range(0, block_dims(0))) +=
           matmul(Fq.Fs[p_mu].get_block(b_ix_mu)(mu, _, _), Tmu(t, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(0))));
      }
    }
  }
}

BlockDiagOpFun &DiagramEvaluator::get_self_energy() { return Sigma; }

void DiagramEvaluator::find_path_self_energy(Backbone &backbone, int f_ix, nda::vector_view<int> ind_path, nda::vector_view<int> block_dims) {
  int m = backbone.m;
  backbone.set_flat_index(f_ix, hyb_poles); // set directions, pole indices, and orbital indices from a single integer index
  /*  Example of block_dims for m = 2 (OCA): each number is an index of block_dims, and each square represents a block of a matrix 
          3            3            2            2            1            1            0
      --------     --------     --------     --------     --------     --------     --------
    4 |   F  |   3 |   G  |   3 |   F  |   2 |   G  |   2 |   F  |   1 |   G  |   1 |   F  |
      |      |     |      |     |      |     |      |     |      |     |      |     |      |
      --------     --------     --------     --------     --------     --------     --------
  */
  // ind_path can diverge at the vertex connected to vertex 0
  for (int b_ix = 0; b_ix < Gt.get_num_block_cols(); b_ix++) { // loop over blocks of self-energy
    for (int p_kap = 0; p_kap < q; p_kap++) {                  // loop over symmetry sets on the zero vertex
      bool path_all_nonzero = true;
      int w = 0, ip = 0; // w loops over the vertices and edges, ip is the current block index

      if (backbone.has_vertex_dag(0)) { // if line is connected to zero is backward
        ip = Fq.F_dags[p_kap].get_block_index(b_ix);
        if (ip != -1) {
          block_dims(0) = Fq.F_dags[p_kap].get_block_size(b_ix, 1);
          block_dims(1) = Fq.F_dags[p_kap].get_block_size(b_ix, 0);
        } else {
          path_all_nonzero = false;
        }
      } else {
        ip = Fq.Fs[p_kap].get_block_index(b_ix);
        if (ip != -1) {
          block_dims(0) = Fq.Fs[p_kap].get_block_size(b_ix, 1);
          block_dims(1) = Fq.Fs[p_kap].get_block_size(b_ix, 0);
        } else {
          path_all_nonzero = false;
        }
      }
      // traverse factors in two halves
      // first half: before vertex connected to zero
      while (w < backbone.get_topology(0, 1) && path_all_nonzero) { // only continue if we have not hit a zero block
        if (w != 0) {
          ip = (backbone.has_vertex_dag(w)) ? Fq.F_dags[Fq.sym_set_labels(backbone.get_orb_ind(w))].get_block_index(ip) :
                                              Fq.Fs[Fq.sym_set_labels(backbone.get_orb_ind(w))].get_block_index(ip); // update block index
        }
        if (ip == -1 || (w < 2 * m - 1 && Gt.get_zero_block_index(ip) == -1)) { // check if we hit a zero block in F or Gt
          path_all_nonzero = false;
        } else {                                      // inner 'if' block unnecessary in first half
          ind_path(w) = ip;                           // store the block index for the current vertex/edge, unless we are at the last vertex
          if (w != backbone.get_topology(0, 1) - 1) { // if so, then orb_ind = -1, because w is the vertex connected to zero
            block_dims(w + 2) = (backbone.has_vertex_dag(w + 1)) ? Fq.F_dags[Fq.sym_set_labels(backbone.get_orb_ind(w + 1))].get_block_size(ip, 0) :
                                                                   Fq.Fs[Fq.sym_set_labels(backbone.get_orb_ind(w + 1))].get_block_size(ip, 0);
          }
        }
        w += 1;
      }

      int ip1 = ip;
      // second half
      if (path_all_nonzero) {
        for (int p_mu = 0; p_mu < q; p_mu++) {
          bool fork_all_nonzero = true;
          w                     = backbone.get_topology(0, 1); // reset w to the vertex connected to vertex 0
          // save block_dims(backbone.get_topology(0, 1) + 1)
          block_dims(w + 1) = (backbone.has_vertex_dag(w)) ? Fq.F_dags[p_mu].get_block_size(ip1, 0) : Fq.Fs[p_mu].get_block_size(ip1, 0);
          ip                = (backbone.has_vertex_dag(w)) ? Fq.F_dags[p_mu].get_block_index(ip1) :
                                                             Fq.Fs[p_mu].get_block_index(ip1); // update block index for the vertex connected to zero
          while (w < 2 * m && fork_all_nonzero) {
            if (w != backbone.get_topology(0, 1)) {
              ip = (backbone.has_vertex_dag(w)) ? Fq.F_dags[Fq.sym_set_labels(backbone.get_orb_ind(w))].get_block_index(ip) :
                                                  Fq.Fs[Fq.sym_set_labels(backbone.get_orb_ind(w))].get_block_index(ip); // update block index
            }
            if (ip == -1 || (w < 2 * m - 1 && Gt.get_zero_block_index(ip) == -1)) { // check if we hit a zero block in F or Gt
              fork_all_nonzero = false;
            } else {
              if (w < 2 * m - 1) {      // only store the block index if we are not at the last vertex
                ind_path(w)       = ip; // store the block index for the current vertex/edge
                block_dims(w + 2) = (backbone.has_vertex_dag(w + 1)) ?
                   Fq.F_dags[Fq.sym_set_labels(backbone.get_orb_ind(w + 1))].get_block_size(ip, 0) :
                   Fq.Fs[Fq.sym_set_labels(backbone.get_orb_ind(w + 1))].get_block_size(ip, 0);
              }
            }
            w += 1;
          }
          if (fork_all_nonzero) {
            // evaluate the diagram with these directions, poles, and orbital indices
            // b_ix is the block index for the first edge
            eval_self_energy_fixed_indices(backbone, b_ix, p_kap, p_mu, ind_path, block_dims);
          }
        }
      }
    }
  }
  backbone.reset_all_inds(); // reset directions, pole indices, and orbital indices for the next iteration
}

void DiagramEvaluator::eval_self_energy(Backbone &backbone, int f_ix) {
  int m        = backbone.m;
  int f_ix_max = static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb_poles.size(), m - 1));
  if (f_ix < 0 || f_ix >= f_ix_max) { throw std::runtime_error("DiagramEvaluator::eval_self_energy: f_ix out of range"); }

  nda::vector<int> ind_path(2 * m - 1);   // tracks block indices of factors for computing a particular block of the self-energy
  nda::vector<int> block_dims(2 * m + 1); // tracks the dimensions of the blocks in these factors

  find_path_self_energy(backbone, f_ix, ind_path, block_dims);
  Sigma.set_zero_block_indices(); // set zero_block_indices according to current blocks
}

void DiagramEvaluator::eval_self_energy(Backbone &backbone) {
  int m = backbone.m;
  nda::vector<int> ind_path(2 * m - 1);   // tracks block indices of factors for computing a particular block of the self-energy
  nda::vector<int> block_dims(2 * m + 1); // tracks the dimensions of the blocks in these factors

  // loop over all flat indices
  int f_ix_max = static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb_poles.size(), m - 1));
  for (int f_ix = 0; f_ix < f_ix_max; f_ix++) { find_path_self_energy(backbone, f_ix, ind_path, block_dims); }
  Sigma.set_zero_block_indices(); // set zero_block_indices according to current blocks
}

void DiagramEvaluator::eval_self_energy_fixed_indices(Backbone &backbone, int b_ix, int p_kap, int p_mu, nda::vector_const_view<int> ind_path,
                                                      nda::vector_const_view<int> block_dims) {

  int m = backbone.m;

  T(_, range(0, block_dims(1)), range(0, block_dims(1))) = Gt.get_block(ind_path(0));
  for (int v = 1; v < backbone.get_topology(0, 1); v++) {
    multiply_vertex_block(backbone, v, ind_path, block_dims);
    compose_with_edge_block(backbone, v, ind_path, block_dims);
  }

  multiply_zero_vertex_block(backbone, (not backbone.has_vertex_dag(0)), b_ix, p_kap, p_mu, ind_path, block_dims);

  for (int v = backbone.get_topology(0, 1) + 1; v < 2 * m; v++) {
    compose_with_edge_block(backbone, v - 1, ind_path, block_dims);
    multiply_vertex_block(backbone, v, ind_path, block_dims);
  }

  multiply_prefactor(backbone);
  int diag_order_sign = (m % 2 == 0) ? -1 : 1;
  if (backbone.get_fb(0) == 0) diag_order_sign *= -1;
  T(_, range(0, block_dims(2 * m)), range(0, block_dims(0))) *= diag_order_sign * backbone.prefactor_sign;
  // TODO: temporary fix: have backward pass consider sparsity of hybridization function (hyb) during zero vertex
  if (nda::max_element(nda::abs(T(_, range(0, block_dims(2 * m)), range(0, block_dims(0))))) > 1e-16)
    Sigma.add_block(b_ix, T(_, range(0, block_dims(2 * m)), range(0, block_dims(0))));
}

block_gf<dlr_imtime> DiagramEvaluator::compute_self_energy(nda::array_const_view<int, 2> topology) {
  Backbone backbone(topology, n);
  eval_self_energy(backbone);
  BlockDiagOpFun sig = get_self_energy();
  std::vector<gf<dlr_imtime>> sig_blocks(sig.get_num_block_cols());
  for (int i = 0; i < sig.get_num_block_cols(); ++i) {
    if (sig.get_zero_block_index(i) == -1) {
      sig_blocks[i] = gf<dlr_imtime>(tau_mesh, 0 * Gt.get_block(i)); // zero block
    } else {
      sig_blocks[i] = gf<dlr_imtime>(tau_mesh, sig.get_block(i));
    }
  }
  reset();
  return {sig_blocks};
}

// ========== Private correlator routines ==========

void DiagramEvaluator::multiply_vertex_corr_block(CorrelatorBackbone &backbone, int v_ix, nda::vector_const_view<int> ind_path,
                                                  nda::vector_const_view<int> block_dims) {

  int o_ix = backbone.get_vertex_orb(v_ix); // orbital_index
  // split backbone orbital index into symmetry set index and orbital index within the symmetry set
  // i.e. have mapping between backbone orbital index and symmetry set index
  int q_ix    = static_cast<int>(Fq.sym_set_labels(o_ix)); // symmetry set index
  int qo_ix   = static_cast<int>(Fq.sym_set_inds(o_ix));   // index within the symmetry set
  int l_ix    = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));
  int n_col_r = block_dims(backbone.get_topology(0, 1) + 1); // number of columns for the left-hand side of the diagram
  int b_ix    = ind_path(v_ix - 1);                          // block index for the vertex v_ix

  if (backbone.has_vertex_bar(v_ix)) {   // F has bar
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) {
        U(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           matmul(Fq.F_dag_bars[q_ix].get_block(b_ix)(qo_ix, l_ix, _, _), U(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    } else {
      for (int t = 0; t < r; t++) {
        U(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           -matmul(Fq.F_bars_refl[q_ix].get_block(b_ix)(qo_ix, l_ix, _, _), U(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    }
  } else {
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) {
        U(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           matmul(Fq.F_dags[q_ix].get_block(b_ix)(qo_ix, _, _), U(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    } else {
      for (int t = 0; t < r; t++) {
        U(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           matmul(Fq.Fs[q_ix].get_block(b_ix)(qo_ix, _, _), U(t, range(0, block_dims(v_ix)), range(0, n_col_r)));
      }
    }
  }

  // K factor
  // skip if last vertex
  if (v_ix != 2 * backbone.m - 1) {
    int bv      = backbone.get_vertex_Ksign(v_ix); // sign on K
    double pole = hyb_poles(l_ix);
    if (backbone.get_vertex_direction(v_ix) == 0) pole = -pole; // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR BACKWARD LINES
    if (bv != 0) {
      for (int t = 0; t < r; t++) {
        U(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) =
           k_it(dlr_it(t), bv * pole) * U(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r));
      }
    }
  }
}

void DiagramEvaluator::compose_with_edge_corr_block(CorrelatorBackbone &backbone, int e_ix, nda::vector_const_view<int> ind_path,
                                                    nda::vector_const_view<int> block_dims) {

  int n_col_r                                                            = e_ix < backbone.get_topology(0, 1) ? block_dims(1) : block_dims(0);
  int b_ix                                                               = ind_path(e_ix); // block index for the edge e_ix
  GKt(_, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1))) = Gt.get_block(b_ix);
  int m                                                                  = backbone.m;
  for (int x = 0; x < m - 1; x++) {
    int be      = backbone.get_edge(e_ix, x); // sign on K
    double pole = hyb_poles(backbone.get_pole_ind(x));
    if (backbone.get_fb(x + 1) == 0) pole = -pole; // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR BACKWARD LINES
    if (be != 0) {
      for (int t = 0; t < r; t++) {
        GKt(t, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1))) =
           k_it(dlr_it(t), be * pole) * GKt(t, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1)));
      }
    }
  }
  U(_, range(0, block_dims(e_ix + 1)), range(0, n_col_r)) =
     itops.convolve(beta, itops.vals2coefs(GKt(_, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1)))),
                    itops.vals2coefs(U(_, range(0, block_dims(e_ix + 1)), range(0, n_col_r))), TIME_ORDERED);
}

void DiagramEvaluator::find_path_correlator(CorrelatorBackbone &backbone, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops, int f_ix,
                                            nda::vector_view<int> ind_path, nda::vector_view<int> block_dims, nda::array_view<dcomplex, 3> Tmuop,
                                            nda::array_view<dcomplex, 3> correlator) {
  int m    = backbone.m;
  int vct0 = backbone.get_topology(0, 1);   // vertex connected to vertex 0
  backbone.set_flat_index(f_ix, hyb_poles); // set directions, pole indices, and orbital indices from a single integer index
  /*  Example of block_dims for m = 2 (OCA): each number is an index of block_dims, and each square represents a block of a matrix 
          4            3            3            2            2            1            1            0
      --------     --------     --------     --------     --------     --------     --------     --------
    4 |   G  |   4 |   F  |   3 |   G  |   3 |   F  |   2 |   G  |   2 |   F  |   1 |   G  |   1 |   F  |
      |      |     |      |     |      |     |      |     |      |     |      |     |      |     |      |
      --------     --------     --------     --------     --------     --------     --------     --------
  */
  // ind_path can diverge at the vertex connected to vertex 0
  // TODO: implement the full correlator evaluation by looping over all block indices and accumulating contributions
  // nda::array<int, 2> ind_paths_right(Gt.get_num_block_cols(), 2 * m - 3), ind_paths_left(Gt.get_num_block_cols(), 2 * m - 3);

  /*
  for (int b_ix = 0; b_ix < Gt.get_num_block_cols(); ++b_ix) { // loop over blocks of matrix product GF...GF
    // fill ind_paths_right
    // fill ind_paths_left
  }
  // loop over blocks of mu and kap operators
  // follow dense_backbone version
  for (int mu = 0; mu < mu_ops.size(); ++mu) {
    
  }*/
  for (int b_ix = 0; b_ix < Gt.get_num_block_cols(); b_ix++) { // loop over blocks of matrix product GF...GF
    for (int kap = 0; kap < kap_ops.size(); ++kap) {           // loop over symmetry sets of rightmost operator
      const auto &kap_op    = kap_ops[kap];
      bool path_all_nonzero = true;
      int w = 1, ip = 0; // w loops over the vertices and edges, ip is the current block index

      // find block index for rightmost F operator
      ip         = kap_op.get_block_index(b_ix);
      int ip_old = b_ix;
      if (ip == -1 || Gt.get_zero_block_index(ip) == -1) { // only continue if kap_ops[kap] and Gt have nonzero blocks in right place
        path_all_nonzero = false;
      } else {
        block_dims(0) = kap_op.get_block_size(ip_old, 1);
        block_dims(1) = kap_op.get_block_size(ip_old, 0);

        ind_path(0) = ip; // store block index for rightmost G

        // traverse factors on right-hand side of diagram
        while (w < vct0 && path_all_nonzero) { // only continue if we have not hit a zero block
          ip_old = ip;
          ip     = (backbone.has_vertex_dag(w) ? Fq.F_dags[Fq.sym_set_labels(backbone.get_orb_ind(w))].get_block_index(ip) :
                                                 Fq.Fs[Fq.sym_set_labels(backbone.get_orb_ind(w))].get_block_index(ip)); // update block index
          if (ip == -1 || Gt.get_zero_block_index(ip) == -1) { // check if we hit a zero block in F or Gt
            path_all_nonzero = false;
          } else {
            block_dims(w + 1) = (backbone.has_vertex_dag(w)) ? Fq.F_dags[Fq.sym_set_labels(backbone.get_orb_ind(w))].get_block_size(ip_old, 0) :
                                                               Fq.Fs[Fq.sym_set_labels(backbone.get_orb_ind(w))].get_block_size(ip_old, 0);
            ind_path(w)       = ip;
          }
          ++w;
        }
        int ip_vct0 = ip;

        // now consider mu_ops[mu]
        if (path_all_nonzero) {
          for (int mu = 0; mu < mu_ops.size(); ++mu) {
            path_all_nonzero  = true; // reset for each mu
            const auto &mu_op = mu_ops[mu];
            ip_old            = ip_vct0;
            ip                = mu_op.get_block_index(ip_vct0);
            if (ip == -1 || Gt.get_zero_block_index(ip) == -1) { // only continue if mu_ops[mu] and Gt have nonzero blocks in right place
              path_all_nonzero = false;
            } else {
              block_dims(vct0 + 1) = mu_op.get_block_size(ip_old, 0);
              ind_path(vct0)       = ip;

              // traverse factors on left-hand side of diagram
              w = vct0 + 1; // reset w to the vertex connected to vertex 0
              while (w < 2 * m && path_all_nonzero) {
                ip_old = ip;
                ip     = (backbone.has_vertex_dag(w) ? Fq.F_dags[Fq.sym_set_labels(backbone.get_orb_ind(w))].get_block_index(ip) :
                                                       Fq.Fs[Fq.sym_set_labels(backbone.get_orb_ind(w))].get_block_index(ip)); // update block index
                if (ip == -1) {
                  path_all_nonzero = false;
                } else {
                  block_dims(w + 1) = (backbone.has_vertex_dag(w)) ? Fq.F_dags[Fq.sym_set_labels(backbone.get_orb_ind(w))].get_block_size(ip_old, 0) :
                                                                     Fq.Fs[Fq.sym_set_labels(backbone.get_orb_ind(w))].get_block_size(ip_old, 0);
                  ind_path(w)       = ip;
                }
                ++w;
              }
              if (path_all_nonzero && b_ix == ind_path(2 * m - 1)) {
                // evaluate the diagram with these directions, poles, and orbital indices
                // b_ix is the block index for the first edge
                eval_correlator_fixed_indices(backbone, ind_path, block_dims);
                for (int t = 0; t < r; ++t) {
                  Tmuop(t, range(0, block_dims(2 * m)), range(0, block_dims(1))) =
                     matmul(U(t, range(0, block_dims(2 * m)), range(0, block_dims(vct0 + 1))),
                            matmul(mu_op.get_block(ind_path(vct0 - 1)), T(t, range(0, block_dims(vct0)), range(0, block_dims(1)))));
                  correlator(t, mu, kap) += trace(matmul(Tmuop(t, range(0, block_dims(2 * m)), range(0, block_dims(1))), kap_op.get_block(b_ix)));
                }
              }
            }
          }
        }
      }
    }
  }
  backbone.reset_all_inds(); // reset directions, pole indices, and orbital indices for the next iteration
}

nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(CorrelatorBackbone &backbone, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops) {
  int m = backbone.m;
  nda::vector<int> ind_path(2 * m);       // tracks block indices of factors for computing a particular block's contribution to the correlator
  nda::vector<int> block_dims(2 * m + 1); // tracks the dimensions of the blocks in these factors
  int f_ix_max                       = static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb_poles.size(), m - 1));
  nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());
  nda::array<dcomplex, 3> Tmuop      = nda::zeros<dcomplex>(r, Nmax, Nmax);
  // loop over all flat indices
  for (int f_ix = 0; f_ix < f_ix_max; ++f_ix) { find_path_correlator(backbone, mu_ops, kap_ops, f_ix, ind_path, block_dims, Tmuop, correlator); }
  return correlator;
}

nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(CorrelatorBackbone &backbone, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops,
                                                          int f_ix) {
  int m = backbone.m;
  nda::vector<int> ind_path(2 * m);       // tracks block indices of factors for computing a particular block's contribution to the correlator
  nda::vector<int> block_dims(2 * m + 1); // tracks the dimensions of the blocks in these factors
  nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());
  nda::array<dcomplex, 3> Tmuop      = nda::zeros<dcomplex>(r, Nmax, Nmax);
  // loop over all flat indices
  find_path_correlator(backbone, mu_ops, kap_ops, f_ix, ind_path, block_dims, Tmuop, correlator);
  return correlator;
}

void DiagramEvaluator::eval_correlator_fixed_indices(CorrelatorBackbone &backbone, nda::vector_const_view<int> ind_path,
                                                     nda::vector_const_view<int> block_dims) {

  int m = backbone.m;

  // evaluate the first sequence of backbone products and convolutions from tau_1 to tau, proceeding right to left, not inculding the creation and
  // annihilation matrices at the end points. the result is an N x N matrix-valued function of tau.
  T(_, range(0, block_dims(1)), range(0, block_dims(1))) = Gt.get_block(ind_path(0));
  for (int v = 1; v < backbone.get_topology(0, 1); ++v) {
    multiply_vertex_block(backbone, v, ind_path, block_dims);
    compose_with_edge_block(backbone, v, ind_path, block_dims);
  }

  // evaluate the second sequence of backbone products and convolutions from tau to beta, using a change of variables to perform convolutions. the
  // result is another N x N matrix-valued function of tau.
  U(_, range(0, block_dims(backbone.get_topology(0, 1) + 1)), range(0, block_dims(backbone.get_topology(0, 1) + 1))) =
     Gt.get_block(ind_path(backbone.get_topology(0, 1)));
  if (m > 1) { // only do this for second-order and higher
    int be = 0;
    for (int i = 0; i < m - 1; ++i) {
      be          = backbone.get_edge(backbone.get_topology(0, 1), i);
      double pole = hyb_poles(backbone.get_pole_ind(i));
      if (backbone.get_fb(i + 1) == 0) pole = -pole; // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR BACKWARD LINES
      if (be != 0) {
        for (int t = 0; t < r; t++) {
          U(t, range(0, block_dims(backbone.get_topology(0, 1) + 1)), range(0, block_dims(backbone.get_topology(0, 1) + 1))) =
            k_it(dlr_it(t), be * pole)
            * U(t, range(0, block_dims(backbone.get_topology(0, 1) + 1)), range(0, block_dims(backbone.get_topology(0, 1) + 1)));
        }
      }
    }
    for (int v = backbone.get_topology(0, 1) + 1; v < 2 * m - 1; ++v) {
      multiply_vertex_corr_block(backbone, v, ind_path, block_dims);
      compose_with_edge_corr_block(backbone, v, ind_path, block_dims);
    }
    // multiply by the last vertex
    multiply_vertex_corr_block(backbone, 2 * m - 1, ind_path, block_dims);
    // convolve with last edge
    GKt(_, range(0, block_dims(2 * m)), range(0, block_dims(2 * m))) = Gt.get_block(ind_path(2 * m - 1));
    int bv                                                           = backbone.get_vertex_Ksign(2 * m - 1); // sign on K
    int l_ix                                                         = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(2 * m - 1));
    double pole                                                      = hyb_poles(l_ix);
    if (backbone.get_vertex_direction(2 * m - 1) == 0) pole = -pole; // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR BACKWARD LINES
    if (bv != 0) {
      for (int t = 0; t < r; t++) {
        GKt(t, range(0, block_dims(2 * m)), range(0, block_dims(2 * m))) =
          k_it(dlr_it(t), -bv * pole) * GKt(t, range(0, block_dims(2 * m)), range(0, block_dims(2 * m)));
      }
    }
    U(_, range(0, block_dims(2 * m)), range(0, block_dims(backbone.get_topology(0, 1) + 1))) =
      itops.convolve(beta, itops.vals2coefs(GKt(_, range(0, block_dims(2 * m)), range(0, block_dims(2 * m)))),
                      itops.vals2coefs(U(_, range(0, block_dims(2 * m)), range(0, block_dims(backbone.get_topology(0, 1) + 1)))), TIME_ORDERED);
  }
  U = itops.reflect(U);

  multiply_prefactor(backbone);
  int diag_order_sign = 1; // (m % 2 == 1) ? -1 : 1;
  // if (backbone.get_fb(0) == 0) diag_order_sign *= -1;
  T(_, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(1))) *= diag_order_sign * backbone.prefactor_sign;
}

// ========= Public self-energy routines ==========

void DiagramEvaluator::reset() {
  T     = 0;
  U     = 0;
  GKt   = 0;
  Tkaps = 0;
  Tmu   = 0;
  for (int i = 0; i < Sigma.get_num_block_cols(); i++) {
    Sigma.set_block(i, nda::zeros<dcomplex>(r, Sigma.get_block_size(i), Sigma.get_block_size(i)));
  }
}

int DiagramEvaluator::get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) {
  Backbone backbone(topology, n);
  return static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb_poles.size(), backbone.m - 1));
}

block_gf<dlr_imtime> DiagramEvaluator::compute_self_energy(nda::array_const_view<int, 2> topology, int f_ix) {
  Backbone backbone(topology, n);
  eval_self_energy(backbone, f_ix);
  BlockDiagOpFun sig = get_self_energy();
  std::vector<gf<dlr_imtime>> sig_blocks(sig.get_num_block_cols());
  for (int i = 0; i < sig.get_num_block_cols(); ++i) {
    if (sig.get_zero_block_index(i) == -1) {
      sig_blocks[i] = gf<dlr_imtime>(tau_mesh, 0 * Gt.get_block(i)); // zero block
    } else {
      sig_blocks[i] = gf<dlr_imtime>(tau_mesh, sig.get_block(i));
    }
  }
  reset();
  return {sig_blocks};
}

int DiagramEvaluator::get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology) {
  CorrelatorBackbone backbone(topology, n);
  return static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb_poles.size(), backbone.m - 1));
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(nda::array_const_view<int, 2> topology) {
  CorrelatorBackbone backbone(topology, n);
  std::vector<BlockOp> mu_ops, kap_ops;
  for (auto &F : Fq.Fs) {
    for (int i = 0; i < F.get_size_sym_set(); ++i) {
      std::vector<nda::array<dcomplex, 2>> mu_blocks;
      for (int j = 0; j < F.get_num_block_cols(); ++j) {
        if (F.get_block_index(j) != -1) {
          mu_blocks.emplace_back(F.get_block(j)(i, _, _));
        } else {
          mu_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
        }
      }
      nda::vector<int> block_indices = F.get_block_indices()(_);
      BlockOp mu_op(block_indices, mu_blocks);
      mu_ops.push_back(mu_op);
    }
  }
  for (auto &F_dag : Fq.F_dags) {
    for (int i = 0; i < F_dag.get_size_sym_set(); ++i) {
      std::vector<nda::array<dcomplex, 2>> kap_blocks;
      for (int j = 0; j < F_dag.get_num_block_cols(); ++j) {
        if (F_dag.get_block_index(j) != -1) {
          kap_blocks.emplace_back(F_dag.get_block(j)(i, _, _));
        } else {
          kap_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
        }
      }
      nda::vector<int> block_indices = F_dag.get_block_indices()(_);
      BlockOp kap_op(block_indices, kap_blocks);
      kap_ops.push_back(kap_op);
    }
  }
  return eval_correlator(backbone, mu_ops, kap_ops);
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(nda::array_const_view<int, 2> topology, int f_ix) {
  CorrelatorBackbone backbone(topology, n);
  std::vector<BlockOp> mu_ops, kap_ops;
  for (auto &F : Fq.Fs) {
    for (int i = 0; i < F.get_size_sym_set(); ++i) {
      std::vector<nda::array<dcomplex, 2>> mu_blocks;
      for (int j = 0; j < F.get_num_block_cols(); ++j) {
        if (F.get_block_index(j) != -1) {
          mu_blocks.emplace_back(F.get_block(j)(i, _, _));
        } else {
          mu_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
        }
      }
      nda::vector<int> block_indices = F.get_block_indices()(_);
      BlockOp mu_op(block_indices, mu_blocks);
      mu_ops.push_back(mu_op);
    }
  }
  for (auto &F_dag : Fq.F_dags) {
    for (int i = 0; i < F_dag.get_size_sym_set(); ++i) {
      std::vector<nda::array<dcomplex, 2>> kap_blocks;
      for (int j = 0; j < F_dag.get_num_block_cols(); ++j) {
        if (F_dag.get_block_index(j) != -1) {
          kap_blocks.emplace_back(F_dag.get_block(j)(i, _, _));
        } else {
          kap_blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
        }
      }
      nda::vector<int> block_indices = F_dag.get_block_indices()(_);
      BlockOp kap_op(block_indices, kap_blocks);
      kap_ops.push_back(kap_op);
    }
  }
  return eval_correlator(backbone, mu_ops, kap_ops, f_ix);
}