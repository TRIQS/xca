#include "triqs_xca/backbone.hpp"
#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <cppdlr/utils.hpp>
#include <iostream>
#include <nda/algorithms.hpp>
#include <nda/declarations.hpp>
#include <nda/mapped_functions.hxx>
#include <nda/nda.hpp>
#include <triqs/mesh/imtime.hpp>
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/block_sparse_backbone.hpp>
#include <triqs_xca/atom_diag_utils.hpp>

DiagramEvaluator::DiagramEvaluator(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> hyb,
                                   nda::array_const_view<dcomplex, 3> hyb_refl, nda::vector_const_view<double> hyb_poles, BlockDiagOpFun &Gt,
                                   BlockOpSymQuartet &Fq)
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
     hyb_refl(hyb_refl),
     hyb_poles(hyb_poles) {

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
     tau_mesh(mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps)),
     beta(beta),
     n(hyb_coeffs.extent(1)),
     q(nda::max_element(Fq.sym_set_labels) + 1),
     r(itops.rank()),
     Nmax(Gt.get_max_block_size()),
     hyb_poles(hyb_poles),
     Sigma(itops.rank(), Gt.get_block_sizes()) {

  // hyb and hyb_refl on DLR imaginary time nodes
  hyb      = aaa_coefs2vals(beta, Lambda, eps, hyb_coeffs, nda::make_regular(hyb_poles / beta));
  hyb_refl = hyb; // TODO correctly handle reflection

  // allocate arrays
  T     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  U     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  GKt   = nda::zeros<dcomplex>(r, Nmax, Nmax);
  Tkaps = nda::zeros<dcomplex>(n, r, Nmax, Nmax);
  Tmu   = nda::zeros<dcomplex>(r, Nmax, Nmax);
}

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

void DiagramEvaluator::multiply_prefactor(Backbone &backbone, nda::vector_const_view<int> block_dims) {
  int m = backbone.m;
  // Multiply by prefactor
  for (int m_ix = 0; m_ix < m - 1; m_ix++) {
    int exp = backbone.get_prefactor_Kexp(m_ix);
    if (exp != 0) {
      int Ksign = backbone.get_prefactor_Ksign(m_ix);
      double om = hyb_poles(backbone.get_pole_ind(m_ix));
      if (backbone.get_fb(m_ix + 1) == 0) om = -om; // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR BACKWARD LINES
      double k = k_it(0, Ksign * om);
      for (int x = 0; x < exp; x++) T(_, range(0, block_dims(2 * m)), range(0, block_dims(0))) /= k;
    }
  }
}

// ========== Self-energy routines ==========

BlockDiagOpFun &DiagramEvaluator::get_self_energy() { return Sigma; }

void DiagramEvaluator::eval_self_energy(Backbone &backbone) {

  int m = backbone.m;
  nda::vector<int> ind_path(2 * m - 1);   // tracks block indices of factors for computing a particular block of the self-energy
  nda::vector<int> block_dims(2 * m + 1); // tracks the dimensions of the blocks in these factors

  // loop over all flat indices
  int f_ix_max = static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb_poles.size(), m - 1));
  for (int f_ix = 0; f_ix < f_ix_max; f_ix++) {
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

  multiply_prefactor(backbone, block_dims);
  int diag_order_sign = (m % 2 == 0) ? -1 : 1;
  if (backbone.get_fb(0) == 0) diag_order_sign *= -1;
  T(_, range(0, block_dims(2 * m)), range(0, block_dims(0))) *= diag_order_sign * backbone.prefactor_sign;
  // TODO: temporary fix: have backward pass consider sparsity of hybridization function (hyb, hyb_refl) during zero vertex
  if (nda::max_element(nda::abs(T(_, range(0, block_dims(2 * m)), range(0, block_dims(0))))) > 1e-16)
    Sigma.add_block(b_ix, T(_, range(0, block_dims(2 * m)), range(0, block_dims(0))));
}

block_gf<dlr_imtime> DiagramEvaluator::compute_self_energy(nda::array<int, 2> topology) {
  int n = hyb.extent(1);
  Backbone backbone(topology, n);
  eval_self_energy(backbone);
  BlockDiagOpFun Sigma = get_self_energy();
  std::vector<gf<dlr_imtime>> Sigma_blocks(Sigma.get_num_block_cols());
  for (int i = 0; i < Sigma.get_num_block_cols(); ++i) {
    if (Sigma.get_zero_block_index(i) == -1) {
      Sigma_blocks[i] = gf<dlr_imtime>(tau_mesh); // zero block
    } else {
      Sigma_blocks[i] = gf<dlr_imtime>(tau_mesh, Sigma.get_block(i));
    }
  }
  return {Sigma_blocks};
}

// ========== Correlator routines ==========

void DiagramEvaluator::multiply_vertex_corr_block(Backbone &backbone, int v_ix, nda::vector_const_view<int> ind_path,
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
  int bv      = backbone.get_vertex_Ksign(v_ix); // sign on K
  double pole = hyb_poles(l_ix);
  if (backbone.get_vertex_direction(v_ix) == 0) pole = -pole; // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR BACKWARD LINES
  if (bv != 0) {
    for (int t = 0; t < r; t++) {
      U(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r)) = k_it(dlr_it(t), bv * pole) * U(t, range(0, block_dims(v_ix + 1)), range(0, n_col_r));
    }
  }
}

void DiagramEvaluator::compose_with_edge_corr_block(Backbone &backbone, int e_ix, nda::vector_const_view<int> ind_path,
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

nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(Backbone &backbone, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops) {
  int m = backbone.m;
  nda::vector<int> ind_path(2 * m);       // tracks block indices of factors for computing a particular block's contribution to the correlator
  nda::vector<int> block_dims(2 * m + 1); // tracks the dimensions of the blocks in these factors
  int f_ix_max                       = static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb_poles.size(), m - 1));
  nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());
  nda::array<dcomplex, 3> Tmuop      = nda::zeros<dcomplex>(r, Nmax, Nmax);
  // loop over all flat indices
  for (int f_ix = 0; f_ix < f_ix_max; ++f_ix) {
    backbone.set_flat_index(f_ix, hyb_poles); // set directions, pole indices, and orbital indices from a single integer index
    /*  Example of block_dims for m = 2 (OCA): each number is an index of block_dims, and each square represents a block of a matrix 
            4            3            3            2            2            1            1            0
        --------     --------     --------     --------     --------     --------     --------     --------
      4 |   G  |   4 |   F  |   3 |   G  |   3 |   F  |   2 |   G  |   2 |   F  |   1 |   G  |   1 |   F  |
        |      |     |      |     |      |     |      |     |      |     |      |     |      |     |      |
        --------     --------     --------     --------     --------     --------     --------     --------
    */
  }
  // TODO: implement the full correlator evaluation by looping over all block indices and accumulating contributions
  return correlator;
}

void DiagramEvaluator::eval_correlator_fixed_indices(Backbone &backbone, int b_ix, int p_kap, int p_mu, nda::vector_const_view<int> ind_path,
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
         k_it(dlr_it(t), bv * pole) * GKt(t, range(0, block_dims(2 * m)), range(0, block_dims(2 * m)));
    }
  }
  U(_, range(0, block_dims(2 * m)), range(0, block_dims(backbone.get_topology(0, 1) + 1))) =
     itops.convolve(beta, itops.vals2coefs(GKt(_, range(0, block_dims(2 * m)), range(0, block_dims(2 * m)))),
                    itops.vals2coefs(U(_, range(0, block_dims(2 * m)), range(0, block_dims(backbone.get_topology(0, 1) + 1)))), TIME_ORDERED);
  U = itops.reflect(U);

  multiply_prefactor(backbone, block_dims);
  int diag_order_sign = (m % 2 == 0) ? -1 : 1;
  if (backbone.get_fb(0) == 0) diag_order_sign *= -1;
  T(_, range(0, block_dims(backbone.get_topology(0, 1))), range(0, block_dims(1))) *= diag_order_sign * backbone.prefactor_sign;
}