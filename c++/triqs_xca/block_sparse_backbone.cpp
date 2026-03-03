#include <iostream>

#include "triqs_xca/atom_diag_utils.hpp"

#include "triqs_xca/block_sparse_backbone.hpp"

namespace triqs_xca::block_sparse {

using cppdlr::_;

using nda::trace;
using nda::range;
using nda::linalg::matmul;

DiagramEvaluator::DiagramEvaluator(double beta, double Lambda, double eps, nda::array_const_view<dcomplex, 3> hyb,
                                   nda::vector_const_view<double> hyb_poles, BlockDiagOpFun &Gt, BlockOpSymQuartet &Fq)
   : itops(imtime_ops(Lambda, cppdlr::build_dlr_rf(Lambda, eps))),
     Gt(Gt),
     Fq(Fq),
     Sigma(itops.rank(), Gt.get_block_sizes()),
     tau_mesh(triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps)),
     beta(beta),
     r(itops.rank()),
     n(hyb.extent(1)),
     q(nda::max_element(Fq.sym_set_labels) + 1),
     Nmax(Gt.get_max_block_size()),
     hyb(hyb),
     hyb_poles(beta * hyb_poles) {

  dlr_it      = itops.get_itnodes();
  hyb_reflect = itops.reflect(hyb);

  // allocate arrays
  T     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  U     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  GKt   = nda::zeros<dcomplex>(r, Nmax, Nmax);
  Tkaps = nda::zeros<dcomplex>(n, r, Nmax, Nmax);
  Tmu   = nda::zeros<dcomplex>(r, Nmax, Nmax);
}

DiagramEvaluator::DiagramEvaluator(double beta, double Lambda, double eps, nda::vector_const_view<double> hyb_poles,
                                   nda::array_const_view<dcomplex, 3> hyb_coeffs, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
                                   triqs::atom_diag::atom_diag<false> const &ad)
   : itops(imtime_ops(Lambda, cppdlr::build_dlr_rf(Lambda, eps))),
     dlr_it(itops.get_itnodes()),
     Gt(BlockDiagOpFun(G_ppsc)),
     Fq(std::get<0>(get_operators(ad, hyb_coeffs))),
     Sigma(itops.rank(), Gt.get_block_sizes()),
     tau_mesh(triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps)),
     beta(beta),
     r(itops.rank()),
     n(hyb_coeffs.extent(1)),
     q(nda::max_element(Fq.sym_set_labels) + 1),
     Nmax(Gt.get_max_block_size()),
     hyb_poles(beta * hyb_poles) {

  // hyb on DLR imaginary time nodes
  hyb         = aaa_coefs2vals(beta, Lambda, eps, hyb_coeffs, hyb_poles);
  hyb_reflect = itops.reflect(hyb);

  // allocate arrays
  T     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  U     = nda::zeros<dcomplex>(r, Nmax, Nmax);
  GKt   = nda::zeros<dcomplex>(r, Nmax, Nmax);
  Tkaps = nda::zeros<dcomplex>(n, r, Nmax, Nmax);
  Tmu   = nda::zeros<dcomplex>(r, Nmax, Nmax);
}

// ----------- Private routines for any diagram ==========

void DiagramEvaluator::multiply_left_vertex(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone, int v_ix, nda::vector_const_view<int> ind_path,
                                            nda::vector_const_view<int> block_dims) {
  int vct0 = backbone.get_topology(0, 1);   // vertex connnected to time zero
  int o_ix = backbone.get_vertex_orb(v_ix); // orbital_index
  // split backbone orbital index into symmetry set index and orbital index within the symmetry set
  // i.e. have mapping between backbone orbital index and symmetry set index
  int q_ix    = static_cast<int>(Fq.sym_set_labels(o_ix)); // symmetry set index
  int qo_ix   = static_cast<int>(Fq.sym_set_inds(o_ix));   // index within the symmetry set
  int l_ix    = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));
  int n_col_r = v_ix < vct0 ? block_dims(1) : block_dims(0); // number of columns in T: depends on whether v_ix is before or after vct0
  int b_ix    = ind_path(v_ix - 1);                          // block index for the vertex v_ix

  // Get the current operator matrix F using the flags and indices of the vertex with index v_ix

  bool has_bar = backbone.has_vertex_bar(v_ix);
  bool has_dag = backbone.has_vertex_dag(v_ix);

  auto F_selector = [&]() {
    if (has_bar)
      return has_dag ? Fq.F_dag_bars[q_ix].get_block(b_ix)(qo_ix, l_ix, _, _) : Fq.F_bars_refl[q_ix].get_block(b_ix)(qo_ix, l_ix, _, _);
    else
      return has_dag ? Fq.F_dags[q_ix].get_block(b_ix)(qo_ix, _, _) : Fq.Fs[q_ix].get_block(b_ix)(qo_ix, _, _);
  };

  nda::array_const_view<dcomplex, 2> F{F_selector()};

  // Get views on temporary storage for input and output

  nda::array_view<dcomplex, 3> T_v  = T_buf(_, range(0, block_dims(v_ix)), range(0, n_col_r));
  nda::array_view<dcomplex, 3> T_vp = T_buf(_, range(0, block_dims(v_ix + 1)), range(0, n_col_r));

  // Multiply with the operator matrix F from the left

  for (int t = 0; t < r; t++) T_vp(t, _, _) = matmul(F, T_v(t, _, _));

  // Multiply with scalar K(\tau) factor of the pole with index l_ix

  int Ksign = backbone.get_vertex_Ksign(v_ix); // sign on K

  if (Ksign != 0) {
    for (int t = 0; t < r; t++) T_vp(t, _, _) *= cppdlr::k_it(dlr_it(t), Ksign * hyb_poles(l_ix));
  }
}

void DiagramEvaluator::integrate_left_edge(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone, int e_ix, nda::vector_const_view<int> ind_path,
                                           nda::vector_const_view<int> block_dims) {
  int m       = backbone.m;
  int b_ix    = ind_path(e_ix); // block index for the edge e_ix
  int vct0    = backbone.get_topology(0, 1);
  int n_col_r = e_ix < vct0 ? block_dims(1) : block_dims(0);

  nda::array_view<dcomplex, 3> GKt_ep = GKt(_, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1)));

  GKt_ep = Gt.get_block(b_ix);

  for (int x = 0; x < m - 1; x++) {
    int Ksign = backbone.get_edge(e_ix, x); // sign on K
    if (Ksign != 0) {
      for (int t = 0; t < r; t++) GKt_ep(t, _, _) *= cppdlr::k_it(dlr_it(t), Ksign * hyb_poles(backbone.get_pole_ind(x)));
    }
  }
  nda::array_view<dcomplex, 3> T_ep = T_buf(_, range(0, block_dims(e_ix + 1)), range(0, n_col_r));
  
  T_ep = itops.convolve(beta, itops.vals2coefs(GKt_ep), itops.vals2coefs(T_ep), cppdlr::TIME_ORDERED);
}

void DiagramEvaluator::multiply_prefactor(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone) {
  // Apply total prefactor comprised of inverse powers of the kernel evaluated at tau=0 and the current hybridization poles
  int m = backbone.m;
  for (int m_ix = 0; m_ix < m - 1; m_ix++) {
    int exp = backbone.get_prefactor_Kexp(m_ix);
    if (exp != 0) {
      int Ksign = backbone.get_prefactor_Ksign(m_ix);
      T_buf *= std::pow(cppdlr::k_it(0, Ksign * hyb_poles(backbone.get_pole_ind(m_ix))), -exp);
    }
  }
}

// ========== Private self-energy routines ==========

void DiagramEvaluator::multiply_left_vertex_and_right_zero_vertex(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone, bool is_forward, int b_ix_0,
                                                                  int p_kap, int p_mu, nda::vector_const_view<int> ind_path,
                                                                  nda::vector_const_view<int> block_dims) {
  int v_ix    = backbone.get_topology(0, 1);
  int b_ix_mu = ind_path(v_ix - 1); // block index for F_mu

  nda::array_view<dcomplex, 3> T_v  = T_buf(_, range(0, block_dims(v_ix)), range(0, block_dims(1)));
  nda::array_view<dcomplex, 3> T_vp = T_buf(_, range(0, block_dims(v_ix + 1)), range(0, block_dims(0)));

  nda::array_view<dcomplex, 4> Tkaps_v = Tkaps(_, _, range(0, block_dims(v_ix)), range(0, block_dims(0)));
  nda::array_view<dcomplex, 3> Tmu_v   = Tmu(_, range(0, block_dims(v_ix)), range(0, block_dims(0)));

  auto F_selector = [&](bool is_forward, int b_ix_ix, int p_ix, int ix) {
    return is_forward ? Fq.Fs[p_ix].get_block(b_ix_ix)(ix, _, _) : Fq.F_dags[p_ix].get_block(b_ix_ix)(ix, _, _);
  };

  // Apply F_kap operator at tau = 0 from the right
  for (int kap = 0; kap < Fq.sym_set_sizes(p_kap); kap++) {
    auto F_kap = F_selector(is_forward, b_ix_0, p_kap, kap);
    for (int t = 0; t < r; t++) Tkaps_v(kap, t, _, _) = matmul(T_v(t, _, _), F_kap);
  }

  T_vp         = 0; // With intermediate result now in Tkaps, reuse T for final result
  int hyb_sign = is_forward ? +1. : -1.;

  for (int mu = 0; mu < Fq.sym_set_sizes(p_mu); mu++) {
    Tmu_v = 0;
    // Multiply with hybridization function
    for (int kap = 0; kap < Fq.sym_set_sizes(p_kap); kap++) {
      nda::array_const_view<dcomplex, 1> hyb_oo = is_forward ? hyb(_, Fq.sym_set_to_orb(p_mu, mu), Fq.sym_set_to_orb(p_kap, kap)) :
                                                               hyb_reflect(_, Fq.sym_set_to_orb(p_mu, mu), Fq.sym_set_to_orb(p_kap, kap));
      for (int t = 0; t < r; t++) Tmu_v(t, _, _) += hyb_sign * hyb_oo(t) * Tkaps_v(kap, t, _, _);
    }
    // Apply F_mu operator from the left at vertex connected to tau = 0
    auto F_mu = F_selector(!is_forward, b_ix_mu, p_mu, mu);
    for (int t = 0; t < r; t++) T_vp(t, _, _) += matmul(F_mu, Tmu_v(t, _, _));
  }
}

BlockDiagOpFun &DiagramEvaluator::get_self_energy() { return Sigma; }

void DiagramEvaluator::find_path_self_energy(Backbone &backbone, int f_ix, nda::vector_view<int> ind_path, nda::vector_view<int> block_dims) {

  int m = backbone.m; // Diagram order
  int vct0 = backbone.get_topology(0, 1); // vertex connected to zero

  backbone.set_flat_index(f_ix, hyb_poles); // set directions, pole indices, and orbital indices from a single integer index

  /*  Example of block_dims for m = 2 (OCA): each number is an index of block_dims, and each square represents a block of a matrix 
          3            3            2            2            1            1            0
      --------     --------     --------     --------     --------     --------     --------
    4 |   F  |   3 |   G  |   3 |   F  |   2 |   G  |   2 |   F  |   1 |   G  |   1 |   F  |
      |      |     |      |     |      |     |      |     |      |     |      |     |      |
      --------     --------     --------     --------     --------     --------     --------
  */

  auto F_selector_p_ix = [&](int w, int p_ix) { return backbone.has_vertex_dag(w) ? Fq.F_dags[p_ix] : Fq.Fs[p_ix]; };
  auto F_selector      = [&](int w) { return F_selector_p_ix(w, Fq.sym_set_labels(backbone.get_orb_ind(w))); };

  bool incomplete_path    = false;
  auto is_path_incomplete = [&](int w, int ip) { return (ip == -1 || (w < 2 * m - 1 && Gt.get_zero_block_index(ip) == -1)); };

  for (int b_ix = 0; b_ix < Gt.get_num_block_cols(); b_ix++) { // loop over blocks of self-energy

    for (int p_kap = 0; p_kap < q; p_kap++) { // loop over symmetry sets on the zero vertex

      // traverse factors in two halves
      // -----------------------------------------------------------------------------------
      // first half: all vertices before vertex connected to zero

      // ind_path setup 1

      int ip = -1; // running block index

      for (int w = 0; w < vct0; w++) {
        ip              = (w == 0) ? F_selector_p_ix(w, p_kap).get_block_index(b_ix) : F_selector(w).get_block_index(ip);
        incomplete_path = is_path_incomplete(w, ip);
        if (incomplete_path) break;
        ind_path(w) = ip;
      }

      if (incomplete_path) continue;

      // block_dims setup 1

      block_dims(0) = F_selector_p_ix(0, p_kap).get_block_size(b_ix, 1);
      block_dims(1) = F_selector_p_ix(0, p_kap).get_block_size(b_ix, 0);

      for (int w = 0; w < vct0 - 1; w++) { block_dims(w + 2) = F_selector(w + 1).get_block_size(ind_path(w), 0); }

      // -----------------------------------------------------------------------------------
      // second half: vertex connected to zero and above

      for (int p_mu = 0; p_mu < q; p_mu++) { // loop over symmetry sets on the vertex connected to vertex 0?

        // ind_path setup 2

        ip = ind_path(vct0 - 1);

        for (int w = vct0; w < 2 * m; w++) {
          ip              = (w == vct0) ? F_selector_p_ix(w, p_mu).get_block_index(ip) : F_selector(w).get_block_index(ip);
          incomplete_path = is_path_incomplete(w, ip);
          if (incomplete_path) break;
          if (w < 2 * m - 1) ind_path(w) = ip;
        }

        if (incomplete_path) continue;

        // block_dims setup 2

        block_dims(vct0 + 1) = F_selector_p_ix(vct0, p_mu).get_block_size(ind_path(vct0 - 1), 0);

        for (int w = vct0; w < 2 * m - 1; w++) { block_dims(w + 2) = F_selector(w + 1).get_block_size(ind_path(w), 0); }

        // evaluate the diagram with these directions, poles, and orbital indices
        // b_ix is the block index for the first edge
        eval_self_energy_fixed_indices(backbone, b_ix, p_kap, p_mu, ind_path, block_dims);
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

  int m = backbone.m, vct0 = backbone.get_topology(0, 1); // vertex connnected to time zero

  T(_, range(0, block_dims(1)), range(0, block_dims(1))) = Gt.get_block(ind_path(0));
  for (int v = 1; v < vct0; v++) {
    multiply_left_vertex(T, backbone, v, ind_path, block_dims);
    integrate_left_edge(T, backbone, v, ind_path, block_dims);
  }

  multiply_left_vertex_and_right_zero_vertex(T, backbone, (not backbone.has_vertex_dag(0)), b_ix, p_kap, p_mu, ind_path, block_dims);

  for (int v = vct0 + 1; v < 2 * m; v++) {
    integrate_left_edge(T, backbone, v - 1, ind_path, block_dims);
    multiply_left_vertex(T, backbone, v, ind_path, block_dims);
  }

  multiply_prefactor(T, backbone);

  int diag_order_sign = (m % 2 == 0) ? -1 : 1;
  if (backbone.get_fb(0) == 0) diag_order_sign *= -1;

  nda::array_view<dcomplex, 3> T_out = T(_, range(0, block_dims(2 * m)), range(0, block_dims(0)));
  T_out *= diag_order_sign * backbone.prefactor_sign;

  // TODO: temporary fix: have backward pass consider sparsity of hybridization function (hyb) during zero vertex
  if (nda::max_element(nda::abs(T_out)) > 1e-16) Sigma.add_block(b_ix, T_out);
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(nda::array_const_view<int, 2> topology) {
  Backbone backbone(topology, n);
  eval_self_energy(backbone);
  BlockDiagOpFun sig = get_self_energy();
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> sig_blocks(sig.get_num_block_cols());
  for (int i = 0; i < sig.get_num_block_cols(); ++i) {
    if (sig.get_zero_block_index(i) == -1) {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, 0 * Gt.get_block(i)); // zero block
    } else {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, sig.get_block(i));
    }
  }
  reset();
  return {sig_blocks};
}

// ========== Private correlator routines ==========

void DiagramEvaluator::multiply_right_vertex(nda::array_view<dcomplex, 3> U_buf, CorrelatorBackbone &backbone, int v_ix,
                                             nda::vector_const_view<int> ind_path, nda::vector_const_view<int> block_dims) {

  int o_ix = backbone.get_vertex_orb(v_ix); // orbital_index
  // split backbone orbital index into symmetry set index and orbital index within the symmetry set
  // i.e. have mapping between backbone orbital index and symmetry set index
  int q_ix    = static_cast<int>(Fq.sym_set_labels(o_ix)); // symmetry set index
  int qo_ix   = static_cast<int>(Fq.sym_set_inds(o_ix));   // index within the symmetry set
  int l_ix    = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));
  int n_row_l = block_dims(2 * backbone.m); // number of rows for the left-hand side of the diagram
  int b_ix    = ind_path(v_ix - 1);         // block index for the vertex v_ix

  // Get the current operator matrix F using the flags and indices of the vertex with index v_ix

  bool has_bar = backbone.has_vertex_bar(v_ix);
  bool has_dag = backbone.has_vertex_dag(v_ix);

  auto F_selector = [&]() {
    if (has_bar)
      return has_dag ? Fq.F_dag_bars[q_ix].get_block(b_ix)(qo_ix, l_ix, _, _) : Fq.F_bars_refl[q_ix].get_block(b_ix)(qo_ix, l_ix, _, _);
    else
      return has_dag ? Fq.F_dags[q_ix].get_block(b_ix)(qo_ix, _, _) : Fq.Fs[q_ix].get_block(b_ix)(qo_ix, _, _);
  };

  nda::array_const_view<dcomplex, 2> F{F_selector()};

  // Get views on temporary storage for input and output

  nda::array_view<dcomplex, 3> U_v  = U_buf(_, range(0, n_row_l), range(0, block_dims(v_ix)));
  nda::array_view<dcomplex, 3> U_vp = U_buf(_, range(0, n_row_l), range(0, block_dims(v_ix + 1)));

  // Multiply with the operator matrix F from the right

  for (int t = 0; t < r; t++) U_v(t, _, _) = matmul(U_vp(t, _, _), F);

  // K factor

  int Ksign = backbone.get_vertex_Ksign(v_ix);
  if (Ksign != 0) {
    for (int t = 0; t < r; t++) U_v(t, _, _) *= cppdlr::k_it(dlr_it(t), -Ksign * hyb_poles(l_ix)); // extra sign for K(t) -> K(beta - t)
  }
}

void DiagramEvaluator::integrate_right_edge(nda::array_view<dcomplex, 3> U_buf, CorrelatorBackbone &backbone, int e_ix,
                                            nda::vector_const_view<int> ind_path, nda::vector_const_view<int> block_dims) {

  int m       = backbone.m;
  int b_ix    = ind_path(e_ix);             // block index for the edge e_ix
  int n_row_l = block_dims(2 * backbone.m); // number of rows for the left-hand side of the diagram

  nda::array_view<dcomplex, 3> GKt_ep = GKt(_, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1)));

  GKt_ep = Gt.get_block(b_ix);

  for (int x = 0; x < m - 1; x++) {
    int Ksign = backbone.get_edge(e_ix, x);
    if (Ksign != 0) {
      for (int t = 0; t < r; t++) GKt_ep(t, _, _) *= cppdlr::k_it(dlr_it(t), Ksign * hyb_poles(backbone.get_pole_ind(x)));
    }
  }

  nda::array_view<dcomplex, 3> U_e = U_buf(_, range(0, n_row_l), range(0, block_dims(e_ix + 1)));
  
  U_e = itops.convolve(beta, itops.vals2coefs(U_e), itops.vals2coefs(GKt_ep), cppdlr::TIME_ORDERED);
}

nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(CorrelatorBackbone &backbone, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops) {
  int m        = backbone.m;
  int f_ix_max = static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb_poles.size(), m - 1));

  nda::array<dcomplex, 3> correlator(r, mu_ops.size(), kap_ops.size()), Tmuop(r, Nmax, Nmax);
  correlator = 0;

  for (int f_ix = 0; f_ix < f_ix_max; ++f_ix) { correlator += eval_correlator(backbone, mu_ops, kap_ops, f_ix); }
  return correlator;
}

nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(CorrelatorBackbone &backbone, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops,
                                                          int f_ix) {
  int m = backbone.m;
  int vct0 = backbone.get_topology(0, 1);

  nda::vector<int> ind_path(2 * m);       // tracks block indices of factors for computing a particular block's contribution to the correlator
  nda::vector<int> block_dims(2 * m + 1); // tracks the dimensions of the blocks in these factors
  
  nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());
  nda::array<dcomplex, 3> Tmuop      = nda::zeros<dcomplex>(r, Nmax, Nmax);

  // blocks and block indices for intermediate storage
  std::vector<nda::array<dcomplex, 3>> right(Gt.get_num_block_cols(), nda::array<dcomplex, 3>(r, Nmax, Nmax));
  std::vector<nda::array<dcomplex, 3>> left(Gt.get_num_block_cols(), nda::array<dcomplex, 3>(r, Nmax, Nmax));  

  auto right_inds = nda::vector<int>(Gt.get_num_block_cols());
  auto left_inds  = nda::vector<int>(Gt.get_num_block_cols());

  nda::vector<int> ind_path_end(3), block_dims_end(4);

  backbone.set_flat_index(f_ix, hyb_poles); // set directions, pole indices, and orbital indices from a single integer index

  /*  Example of block_dims for m = 2 (OCA): each number is an index of block_dims, and each square represents a block of a matrix 
            4            3            3            2            2            1            1            0
        --------     --------     --------     --------     --------     --------     --------     --------
      4 |   G  |   4 |   F  |   3 |   G  |   3 |   F  |   2 |   G  |   2 |   F  |   1 |   G  |   1 |   F  |
        |      |     |      |     |      |     |      |     |      |     |      |     |      |     |      |
        --------     --------     --------     --------     --------     --------     --------     --------
    */

  auto F_selector_p_ix = [&](int w, int p_ix) { return backbone.has_vertex_dag(w) ? Fq.F_dags[p_ix] : Fq.Fs[p_ix]; };
  auto F_selector      = [&](int w) { return F_selector_p_ix(w, Fq.sym_set_labels(backbone.get_orb_ind(w))); };

  auto is_path_incomplete = [&](int ip) { return (ip == -1 || Gt.get_zero_block_index(ip) == -1); };

  bool incomplete_path = false;
  
  right_inds = -1;
  left_inds  = -1;

  // -- Right-hand side of diagram, with vertices on [tau, 0]
  
  for (int b_ix = 0; b_ix < Gt.get_num_block_cols(); ++b_ix) { // loop over blocks of right-hand side of diagram

    // ind_path setup (right)
    
    int ip = b_ix;
    ind_path(0) = ip;
    
    for(int w = 1; w < vct0; w++) {
      ip = F_selector(w).get_block_index(ip);
      incomplete_path = is_path_incomplete(ip);
      if (incomplete_path) break;
      ind_path(w) = ip;
    }

    if (incomplete_path) continue;

    // block_dims setup (right)

    // block_dims(0) = ?? // Not initalized, is block_dims(0) unused ??
    block_dims(1) = Gt.get_block_size(b_ix);
    for(int w = 1; w < vct0; w++) block_dims(w + 1) = F_selector(w).get_block_size(ind_path(w - 1), 0);    

    // Evaluate: right-hand side edges and vertices (to buffer T)
    
    T(_, range(0, block_dims(1)), range(0, block_dims(1))) = Gt.get_block(ind_path(0)); // first edge at 0
    
    for (int v = 1; v < vct0; v++) {
      multiply_left_vertex(T, backbone, v, ind_path, block_dims);
      integrate_left_edge(T, backbone, v, ind_path, block_dims);
    }
    multiply_prefactor(T, backbone);

    nda::array_view<dcomplex, 3> T_out = T(_, range(0, block_dims(vct0)), range(0, block_dims(1)));
    T_out *= backbone.prefactor_sign; // include sign from diag order and prefactor

    // Store result in T to the "right" buffer
    right[b_ix] = T_out;
    right_inds(b_ix) = ind_path(vct0 - 1);
  }

  // -- Left-hand side of diagram, with vertices on [beta, tau]

  for (int b_ix = 0; b_ix < Gt.get_num_block_cols(); ++b_ix) {
    
    // ind_path setup (left)
    
    int ip = b_ix;
    ind_path(vct0) = ip; // store block index for vertex connected to vertex 0

    for(int w = vct0 + 1; w < 2 * m; w++) {
      ip = F_selector(w).get_block_index(ip);
      incomplete_path = is_path_incomplete(ip);
      if (incomplete_path) break;
      ind_path(w) = ip;
    }

    if (incomplete_path) continue;
    
    // block_dims setup (left)

    block_dims(vct0 + 1) = Gt.get_block_size(b_ix);
    for(int w = vct0 + 1; w < 2 * m; w++) block_dims(w + 1) = F_selector(w).get_block_size(ind_path(w - 1), 0);

    // Evaluate: left-hand side edges and vertices (to buffer U) going from beta to tau 

    nda::array_view<dcomplex, 3> U_beta = U(_, range(0, block_dims(2 * m)), range(0, block_dims(2 * m)));

    U_beta = Gt.get_block(ind_path(2 * m - 1));

    for (int v = 2 * m - 1; v > vct0; v--) {
      multiply_right_vertex(U, backbone, v, ind_path, block_dims);
      integrate_right_edge(U, backbone, v - 1, ind_path, block_dims);
    }

    U = itops.reflect(U); // Reflect result to account for outer [beta, tau] integral order
    
    // Store result in U to the "left" buffer
    left[ind_path(vct0)] = U(_, range(0, block_dims(2 * m)), range(0, block_dims(vct0 + 1)));
    left_inds(b_ix)      = ind_path(2 * m - 1);
  }

  for (int mu = 0; mu < mu_ops.size(); ++mu) {
    for (int b_ix = 0; b_ix < Gt.get_num_block_cols(); ++b_ix) { // loop over blocks of right product

      // setup ind_path_end

      int ip = right_inds(b_ix);
      if (is_path_incomplete(ip)) continue;
      ind_path_end(0) = ip;
      
      ip = mu_ops[mu].get_block_index(ip);
      if (is_path_incomplete(ip)) continue;
      ind_path_end(1) = ip;
      
      ip = left_inds(ip);
      if (is_path_incomplete(ip)) continue;
      ind_path_end(2) = ip;      
      
      // setup block_dims_end

      block_dims_end(0) = right[b_ix].shape(2); // Is this ever different from Nmax?
      block_dims_end(1) = right[b_ix].shape(1); // Is this ever different from Nmax?
      block_dims_end(2) = mu_ops[mu].get_block_size(ind_path_end(0), 0);
      block_dims_end(3) = left[ind_path_end(1)].shape(1);
      
      // Evaluate: product trace, correlator = Tr[left O_mu right O_kap]

      nda::array_const_view<dcomplex, 3> left_b = left[ind_path_end(1)](_, range(0, block_dims_end(3)), range(0, block_dims_end(2)));
      nda::array_const_view<dcomplex, 2> O_mu_b = mu_ops[mu].get_block(ind_path_end(0));
      nda::array_const_view<dcomplex, 3> right_b = right[b_ix](_, range(0, block_dims_end(1)), range(0, block_dims_end(0)));

      nda::array_view<dcomplex, 3> Tmuop_b = Tmuop(_, range(0, block_dims_end(3)), range(0, block_dims_end(0)));

      for (int t = 0; t < r; ++t) Tmuop_b(t, _, _) = matmul(left_b(t, _, _), matmul(O_mu_b, right_b(t, _, _)));
      
      for (int kap = 0; kap < kap_ops.size(); ++kap) {
        for (int c_ix = 0; c_ix < Gt.get_num_block_cols(); ++c_ix) { // Can this loop be removed using c_ix = ind_path_end(2) ?
          if (c_ix == ind_path_end(2) && kap_ops[kap].get_block_index(c_ix) == b_ix) {

            nda::array_const_view<dcomplex, 2> O_kap_b = kap_ops[kap].get_block(c_ix);

            for (int t = 0; t < r; ++t) correlator(t, mu, kap) += trace(matmul(Tmuop_b(t, _, _), O_kap_b));
            
          }
        }
      }
      
    }
  }
  
  backbone.reset_all_inds(); // reset directions, pole indices, and orbital indices for the next iteration

  return correlator;
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

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(nda::array_const_view<int, 2> topology, int f_ix) {
  Backbone backbone(topology, n);
  eval_self_energy(backbone, f_ix);
  BlockDiagOpFun sig = get_self_energy();
  std::vector<triqs::gfs::gf<triqs::mesh::dlr_imtime>> sig_blocks(sig.get_num_block_cols());
  for (int i = 0; i < sig.get_num_block_cols(); ++i) {
    if (sig.get_zero_block_index(i) == -1) {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, 0 * Gt.get_block(i)); // zero block
    } else {
      sig_blocks[i] = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, sig.get_block(i));
    }
  }
  reset();
  return {sig_blocks};
}

void DiagramEvaluator::print_self_energy_backbone(nda::array_const_view<int, 2> topology, int f_ix) {
  Backbone backbone(topology, n);
  backbone.set_flat_index(f_ix, hyb_poles);
  std::cout << "Self-energy backbone for f_ix = " << f_ix << ":\n";
  std::cout << backbone << std::endl;
}

int DiagramEvaluator::get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology) {
  CorrelatorBackbone backbone(topology, n);
  return static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb_poles.size(), backbone.m - 1));
}

std::vector<BlockOp> DiagramEvaluator::setup_mu_ops_for_single_ptcle_gf() {
  std::vector<BlockOp> mu_ops;
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
  return mu_ops;
}

std::vector<BlockOp> DiagramEvaluator::setup_kap_ops_for_single_ptcle_gf() {
  std::vector<BlockOp> kap_ops;
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
  return kap_ops;
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(nda::array_const_view<int, 2> topology) {
  CorrelatorBackbone backbone(topology, n);
  auto mu_ops  = setup_mu_ops_for_single_ptcle_gf();
  auto kap_ops = setup_kap_ops_for_single_ptcle_gf();
  return eval_correlator(backbone, mu_ops, kap_ops);
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(nda::array_const_view<int, 2> topology, int f_ix) {
  CorrelatorBackbone backbone(topology, n);
  auto mu_ops  = setup_mu_ops_for_single_ptcle_gf();
  auto kap_ops = setup_kap_ops_for_single_ptcle_gf();
  return eval_correlator(backbone, mu_ops, kap_ops, f_ix);
}

void DiagramEvaluator::print_single_ptcle_gf_backbone(nda::array_const_view<int, 2> topology, int f_ix) {
  CorrelatorBackbone backbone(topology, n);
  backbone.set_flat_index(f_ix, hyb_poles);
  std::cout << "Single-particle Green's function backbone for f_ix = " << f_ix << ":\n";
  std::cout << backbone << std::endl;
}

} // namespace triqs_xca::block_sparse
