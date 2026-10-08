#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <iostream>

#include <itertools/itertools.hpp>

#include "triqs_xca/block_sparse/atom_diag.hpp"

#include "triqs_xca/block_sparse/diagram_evaluator.hpp"
#include "triqs_xca/block_sparse/dynint.hpp"

#include "triqs_xca/hyb.hpp"
#include "triqs_xca/operator_statistics.hpp"

namespace triqs_xca::block_sparse {

using cppdlr::_;

using nda::trace;
using nda::range;
using nda::linalg::matmul;


namespace {

  // an out-of-range n_int would give a negative n_hyb
  void check_n_int(int n_int, long n) {
    if (n_int < 0 || n_int > n)
      throw std::invalid_argument("DiagramEvaluator: n_int = " + std::to_string(n_int) + " is out of range for a quartet with "
                                  + std::to_string(n) + " orbital indices");
  }

  // Validate the coefficients of the Fq-only constructor against the quartet and return n, the atom_diag constructors build both from the
  // same array and need no check
  long check_fq_only_coeffs(nda::array_const_view<dcomplex, 3> hyb_coeffs, nda::vector_const_view<double> hyb_poles,
                            BlockOpSymQuartet const &Fq) {
    long n = nda::sum(Fq.sym_set_sizes);
    long p = hyb_poles.size();
    // all three extents, a wrong pole count is silently reinterpreted by the reshape in coefs2vals
    if (hyb_coeffs.extent(0) != p || hyb_coeffs.extent(1) != n || hyb_coeffs.extent(2) != n) {
      throw std::invalid_argument("DiagramEvaluator: hyb_coeffs must have shape (p, n, n) with p = hyb_poles.size() = "
                                  + std::to_string(p) + " and n = sum(Fq.sym_set_sizes) = " + std::to_string(n) + ", got ("
                                  + std::to_string(hyb_coeffs.extent(0)) + ", " + std::to_string(hyb_coeffs.extent(1)) + ", "
                                  + std::to_string(hyb_coeffs.extent(2)) + ")");
    }
    check_sym_set_block_diagonal(hyb_coeffs, Fq.sym_set_labels, "DiagramEvaluator");
    return n;
  }

  // the leading index kap of Tkaps runs within one symmetry set, so the largest set is the bound rather than n
  long max_sym_set_size(BlockOpSymQuartet const &Fq) { return nda::max_element(Fq.sym_set_sizes); }

  // largest block dimension over all symmetry sets, Fs[0] alone underestimates it when set 0 misses the largest subspace
  int max_block_dim(BlockOpSymQuartet const &Fq) {
    int N = 0;
    for (auto const &F : Fq.Fs) N = std::max(N, nda::max_element(F.get_block_sizes()));
    for (auto const &F : Fq.F_dags) N = std::max(N, nda::max_element(F.get_block_sizes()));
    return N;
  }

  // dimension of each invariant subspace, read off the blocks of the operators acting on it; -1 for a subspace no operator acts on
  nda::vector<long> subspace_dims_of(BlockOpSymQuartet const &Fq) {
    long n_sub = Fq.Fs[0].get_num_block_cols();
    nda::vector<long> dims(n_sub);
    dims = -1;
    for (auto const *sets : {&Fq.Fs, &Fq.F_dags})
      for (auto const &F : *sets)
        for (long b = 0; b < n_sub; ++b)
          if (F.get_block_index(b) >= 0) dims(b) = F.get_block_size(b, 1);
    return dims;
  }

  template <bool isComplex>
  nda::vector<long> subspace_dims_of(triqs::atom_diag::atom_diag<isComplex> const &ad) {
    auto const dims = ad.get_subspace_dims();
    nda::vector<long> result(dims.size());
    for (std::size_t b = 0; b < dims.size(); ++b) result(b) = dims[b];
    return result;
  }

  // largest invariant-subspace dimension, i.e. the largest block of the pseudo-particle propagator
  template <bool isComplex>
  int max_subspace_dim(triqs::atom_diag::atom_diag<isComplex> const &ad) {
    auto const dims = ad.get_subspace_dims();
    return dims.empty() ? 0 : *std::max_element(dims.begin(), dims.end());
  }

} // namespace

DiagramEvaluator::DiagramEvaluator(double beta, double Lambda, double eps, 
                                   nda::vector_const_view<double> hyb_poles, 
                                   nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                   BlockOpSymQuartet &Fq, int n_int)
   : 
     tau_mesh(triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, Lambda / beta, eps, false)),
     itops(tau_mesh.dlr_it()),
     dlr_it(itops.get_itnodes()),
     Fq(Fq),
     Sigma({}, {}),
     subspace_dims(subspace_dims_of(Fq)),
     beta(beta),
     r(itops.rank()),
     n(check_fq_only_coeffs(hyb_coeffs, hyb_poles, Fq)), // number of spin-orbitals; also validates hyb_coeffs
     // n_int is the constructor parameter here, the member of the same name is not yet initialized
     n_hyb((check_n_int(n_int, nda::sum(Fq.sym_set_sizes)), nda::sum(Fq.sym_set_sizes) - n_int)),
     n_int(n_int),
     q(nda::max_element(Fq.sym_set_labels) + 1),
     Nmax(max_block_dim(Fq)),
     hyb(tau_mesh, hyb_poles, hyb_coeffs),
     // allocate arrays
     T(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     U(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     GKt(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     Tkaps(nda::zeros<dcomplex>(max_sym_set_size(Fq), r, Nmax, Nmax)), // kap runs within one symmetry set, not over all n
     Tmu(nda::zeros<dcomplex>(r, Nmax, Nmax))
     {}

template<bool isComplex>
DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double> hyb_poles,
  nda::array_const_view<dcomplex, 3> hyb_coeffs, 
  triqs::mesh::dlr_imtime tau_mesh,
  triqs::atom_diag::atom_diag<isComplex> const &ad)
   : 
     tau_mesh(tau_mesh),
     itops(tau_mesh.dlr_it()),
     dlr_it(itops.get_itnodes()),
     Fq(std::get<0>(atom_diag::get_operators(ad, hyb_coeffs))),
     Sigma({}, {}),
     subspace_dims(subspace_dims_of(ad)),
     beta(tau_mesh.beta()),
     r(itops.rank()),
     n(ad.get_fops().size()), // number of fermion flavours (spin-orbitals)
     n_hyb(ad.get_fops().size()),
     n_int(0), // no dynamical interactions in this constructor
     q(nda::max_element(Fq.sym_set_labels) + 1),
     Nmax(max_subspace_dim(ad)),
     hyb(tau_mesh, hyb_poles, hyb_coeffs),
     // allocate arrays
     T(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     U(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     GKt(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     Tkaps(nda::zeros<dcomplex>(max_sym_set_size(Fq), r, Nmax, Nmax)), // kap runs within one symmetry set, not over all n
     Tmu(nda::zeros<dcomplex>(r, Nmax, Nmax))
     {}

template DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double> hyb_poles,
  nda::array_const_view<dcomplex, 3> hyb_coeffs, 
  triqs::mesh::dlr_imtime tau_mesh,
  triqs::atom_diag::atom_diag<true> const &ad);

template DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double> hyb_poles,
  nda::array_const_view<dcomplex, 3> hyb_coeffs, 
  triqs::mesh::dlr_imtime tau_mesh,
  triqs::atom_diag::atom_diag<false> const &ad);  

template<bool isComplex>
DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double> hyb_poles,
  nda::array_const_view<dcomplex, 3> hyb_coeffs,
  triqs::mesh::dlr_imtime tau_mesh,
  triqs::atom_diag::atom_diag<isComplex> const &ad,
  std::vector<triqs::operators::many_body_operator_real> const &dynint_ops,
  nda::array_const_view<dcomplex, 3> dynint_coeffs)
   :
     tau_mesh(tau_mesh),
     itops(tau_mesh.dlr_it()),
     dlr_it(itops.get_itnodes()),
     Fq(std::get<0>(dynint::get_operators_and_interactions(ad, hyb_coeffs, dynint_coeffs, dynint_ops))),
     Sigma({}, {}),
     subspace_dims(subspace_dims_of(ad)),
     beta(tau_mesh.beta()),
     r(itops.rank()),
     n(ad.get_fops().size() + dynint_ops.size()), // the extended flavour space
     n_hyb(ad.get_fops().size()),
     n_int(dynint_ops.size()),
     q(nda::max_element(Fq.sym_set_labels) + 1),
     Nmax(max_subspace_dim(ad)),
     // no refl_sign argument: block-sparse keeps +1.0 and applies the backward-line sign explicitly
     hyb(tau_mesh, hyb_poles, hyb::get_extended_coefficients(hyb_coeffs, dynint_coeffs)),
     // allocate arrays
     T(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     U(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     GKt(nda::zeros<dcomplex>(r, Nmax, Nmax)),
     Tkaps(nda::zeros<dcomplex>(max_sym_set_size(Fq), r, Nmax, Nmax)), // kap runs within one symmetry set, not over all n
     Tmu(nda::zeros<dcomplex>(r, Nmax, Nmax))
     {}

template DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double> hyb_poles,
  nda::array_const_view<dcomplex, 3> hyb_coeffs,
  triqs::mesh::dlr_imtime tau_mesh,
  triqs::atom_diag::atom_diag<true> const &ad,
  std::vector<triqs::operators::many_body_operator_real> const &dynint_ops,
  nda::array_const_view<dcomplex, 3> dynint_coeffs);

template DiagramEvaluator::DiagramEvaluator(
  nda::vector_const_view<double> hyb_poles,
  nda::array_const_view<dcomplex, 3> hyb_coeffs,
  triqs::mesh::dlr_imtime tau_mesh,
  triqs::atom_diag::atom_diag<false> const &ad,
  std::vector<triqs::operators::many_body_operator_real> const &dynint_ops,
  nda::array_const_view<dcomplex, 3> dynint_coeffs);

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
  hyb.multiply_kernel_on_vertex(T_vp, backbone, v_ix, l_ix);
}

void DiagramEvaluator::integrate_left_edge(nda::array_view<dcomplex, 3> T_buf, BlockDiagOpFun &Gt, Backbone &backbone, int e_ix, nda::vector_const_view<int> ind_path,
                                           nda::vector_const_view<int> block_dims) {
  int b_ix    = ind_path(e_ix); // block index for the edge e_ix
  int vct0    = backbone.get_topology(0, 1);
  int n_col_r = e_ix < vct0 ? block_dims(1) : block_dims(0);

  nda::array_view<dcomplex, 3> GKt_ep = GKt(_, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1)));
  GKt_ep = Gt.get_block(b_ix);

  hyb.multiply_kernels_on_edge(GKt_ep, backbone, e_ix);
  
  nda::array_view<dcomplex, 3> T_ep = T_buf(_, range(0, block_dims(e_ix + 1)), range(0, n_col_r));
  T_ep = itops.convolve(beta, itops.vals2coefs(GKt_ep), itops.vals2coefs(T_ep), cppdlr::TIME_ORDERED);
}

void DiagramEvaluator::multiply_prefactor(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone) {
  // Apply total prefactor comprised of inverse powers of the kernel evaluated at tau=0 and the current hybridization poles
  hyb.multiply_kernels_prefactor(T_buf, backbone);
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
      // The backward branch transposes the coefficient index, as the barred operators do for the interior lines, see dense/diagram_evaluator.cpp.
      // Reading the same order on both branches breaks unitary invariance under a complex basis rotation
      nda::array_const_view<dcomplex, 1> hyb_oo = is_forward ?
        hyb.values(_, Fq.sym_set_to_orb(p_mu, mu), Fq.sym_set_to_orb(p_kap, kap)) :
        hyb.values_reflect(_, Fq.sym_set_to_orb(p_kap, kap), Fq.sym_set_to_orb(p_mu, mu));

      // set the orbital indices of the vertex connected to zero
      // and compute the fermionic permutation parity of the resulting diagram.

      backbone.set_orb_inds_of_0_and_vct0(Fq.sym_set_to_orb(p_kap, kap), Fq.sym_set_to_orb(p_mu, mu));
      int fermionic_parity = backbone.get_parity();

      for (int t = 0; t < r; t++) Tmu_v(t, _, _) += fermionic_parity * hyb_sign * hyb_oo(t) * Tkaps_v(kap, t, _, _);
    }
    // Apply F_mu operator from the left at vertex connected to tau = 0
    auto F_mu = F_selector(!is_forward, b_ix_mu, p_mu, mu);
    for (int t = 0; t < r; t++) T_vp(t, _, _) += matmul(F_mu, Tmu_v(t, _, _));
  }
}

void DiagramEvaluator::find_path_self_energy(BlockDiagOpFun &Gt, Backbone &backbone, long f_ix, nda::vector_view<int> ind_path, nda::vector_view<int> block_dims) {

  int m = backbone.m; // Diagram order
  int vct0 = backbone.get_topology(0, 1); // vertex connected to zero

  backbone.set_flat_index(f_ix, hyb.poles); // set directions, pole indices, and orbital indices from a single integer index

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

      for (int p_mu = 0; p_mu < q; p_mu++) { // loop over symmetry sets on the vertex connected to vertex 0

        // Only the diagonal (p_kap, p_mu) pairs contribute, since every coefficient array reaching an evaluator has passed
        // check_sym_set_block_diagonal. The cross-set remainder below sym_set_coupling_tol is not harmless: its T_out need not return to
        // block b_ix and would be accumulated into a block of the wrong shape, so it is skipped here rather than by the absolute guard below
        if (p_mu != p_kap) continue;

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
        eval_self_energy_fixed_indices(Gt, backbone, b_ix, p_kap, p_mu, ind_path, block_dims);
      }
    }
  }
  backbone.reset_all_inds(); // reset directions, pole indices, and orbital indices for the next iteration
}

void DiagramEvaluator::eval_self_energy(BlockDiagOpFun &Gt, Backbone &backbone, long f_ix) {
  int m = backbone.m;

  nda::vector<int> ind_path(2 * m - 1);   // tracks block indices of factors for computing a particular block of the self-energy
  nda::vector<int> block_dims(2 * m + 1); // tracks the dimensions of the blocks in these factors

  find_path_self_energy(Gt, backbone, f_ix, ind_path, block_dims);
  Sigma.set_zero_block_indices(); // set zero_block_indices according to current blocks
}

void DiagramEvaluator::eval_self_energy(BlockDiagOpFun &Gt, Backbone &backbone) {
  int m = backbone.m;
  nda::vector<int> ind_path(2 * m - 1);   // tracks block indices of factors for computing a particular block of the self-energy
  nda::vector<int> block_dims(2 * m + 1); // tracks the dimensions of the blocks in these factors

  // loop over all flat indices
  long f_ix_max = backbone.num_flat_indices(hyb.poles.size());
  for (long f_ix = 0; f_ix < f_ix_max; f_ix++) { find_path_self_energy(Gt, backbone, f_ix, ind_path, block_dims); }
  Sigma.set_zero_block_indices(); // set zero_block_indices according to current blocks
}

void DiagramEvaluator::eval_self_energy_fixed_indices(BlockDiagOpFun &Gt, Backbone &backbone, int b_ix, int p_kap, int p_mu, nda::vector_const_view<int> ind_path,
                                                      nda::vector_const_view<int> block_dims) {

  int m = backbone.m, vct0 = backbone.get_topology(0, 1); // vertex connnected to time zero

  T(_, range(0, block_dims(1)), range(0, block_dims(1))) = Gt.get_block(ind_path(0));
  for (int v = 1; v < vct0; v++) {
    multiply_left_vertex(T, backbone, v, ind_path, block_dims);
    integrate_left_edge(T, Gt, backbone, v, ind_path, block_dims);
  }

  multiply_left_vertex_and_right_zero_vertex(T, backbone, (not backbone.has_vertex_dag(0)), b_ix, p_kap, p_mu, ind_path, block_dims);

  for (int v = vct0 + 1; v < 2 * m; v++) {
    integrate_left_edge(T, Gt, backbone, v - 1, ind_path, block_dims);
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

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology) 
  {
  auto Gt = native_propagator(G_ppsc);
  return compute_self_energy(Gt, topology);
  }

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology) 
  {
  check_propagator(Gt);
  // Allocate Sigma and set to zero
  Sigma = Gt; 
  Sigma *= 0;

  Backbone backbone(topology, n, n_int);
  eval_self_energy(Gt, backbone);

  // This converts a BlockDiagOpFun to a triqs::gfs::block_gf (move into BlockDiagOpFun?)
  auto sig = Sigma;
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

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology, long f_ix) 
  {
  auto Gt = native_propagator(G_ppsc);
  return compute_self_energy(Gt, topology, f_ix);
}

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology, 
  nda::array_const_view<long, 1> f_ix_vec) 
  {
  auto Gt = native_propagator(G_ppsc);

  // Allocate Sigma and set to zero
  Sigma = Gt;
  Sigma *= 0;

  Backbone backbone(topology, n, n_int);

  for( auto f_ix : f_ix_vec ) {
    eval_self_energy(Gt, backbone, f_ix);
  }

  // This converts a BlockDiagOpFun to a triqs::gfs::block_gf (move into BlockDiagOpFun?)
  auto sig = Sigma;
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

triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(
  BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology, long f_ix) 
  {
  check_propagator(Gt);
  // Allocate Sigma and set to zero
  Sigma = Gt;
  Sigma *= 0;

  Backbone backbone(topology, n, n_int);
  eval_self_energy(Gt, backbone, f_ix);

  // This converts a BlockDiagOpFun to a triqs::gfs::block_gf (move into BlockDiagOpFun?)
  auto sig = Sigma;
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

  hyb.multiply_kernel_on_vertex(U_v, backbone, v_ix, l_ix, -1.0);
}

void DiagramEvaluator::integrate_right_edge(nda::array_view<dcomplex, 3> U_buf, BlockDiagOpFun &Gt, CorrelatorBackbone &backbone, int e_ix,
                                            nda::vector_const_view<int> ind_path, nda::vector_const_view<int> block_dims) {

  int b_ix    = ind_path(e_ix);             // block index for the edge e_ix
  int n_row_l = block_dims(2 * backbone.m); // number of rows for the left-hand side of the diagram

  nda::array_view<dcomplex, 3> GKt_ep = GKt(_, range(0, block_dims(e_ix + 1)), range(0, block_dims(e_ix + 1)));

  GKt_ep = Gt.get_block(b_ix);

  hyb.multiply_kernels_on_edge(GKt_ep, backbone, e_ix);

  nda::array_view<dcomplex, 3> U_e = U_buf(_, range(0, n_row_l), range(0, block_dims(e_ix + 1)));
  
  U_e = itops.convolve(beta, itops.vals2coefs(U_e), itops.vals2coefs(GKt_ep), cppdlr::TIME_ORDERED);
}

nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(BlockDiagOpFun &Gt, CorrelatorBackbone &backbone,
    std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops, bool is_fermionic) {
  check_propagator(Gt);

  long f_ix_max = backbone.num_flat_indices(hyb.poles.size());

  nda::array<dcomplex, 3> correlator(r, mu_ops.size(), kap_ops.size()), Tmuop(r, Nmax, Nmax);
  correlator = 0;

  for (long f_ix = 0; f_ix < f_ix_max; ++f_ix) { correlator += eval_correlator(Gt, backbone, mu_ops, kap_ops, f_ix, is_fermionic); }
  return correlator;
}

nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(BlockDiagOpFun &Gt, CorrelatorBackbone &backbone, std::vector<BlockOp> mu_ops, std::vector<BlockOp> kap_ops,
                                                          long f_ix, bool is_fermionic) {
  check_propagator(Gt);
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

  backbone.set_flat_index(f_ix, hyb.poles); // set directions, pole indices, and orbital indices from a single integer index

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

  right_inds = -1;
  left_inds  = -1;

  // -- Right-hand side of diagram, with vertices on [tau, 0]
  
  for (int b_ix = 0; b_ix < Gt.get_num_block_cols(); ++b_ix) { // loop over blocks of right-hand side of diagram

    // Each block loop owns its flag: the walk is empty when vct0 == 1 here and when vct0 == 2m-1 in the [beta, tau] loop, and a shared
    // flag would carry the previous loop's last block into the next one, dropping the whole [beta, tau] side
    bool incomplete_path = false;

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
      integrate_left_edge(T, Gt, backbone, v, ind_path, block_dims);
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

    bool incomplete_path = false; // per block; see the [tau, 0] loop above

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
      integrate_right_edge(U, Gt, backbone, v - 1, ind_path, block_dims);
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
      
      // the block index of the kap operator is fixed by the end of the path, which is_path_incomplete(ip) above has checked
      int const c_ix = ind_path_end(2);
      for (int kap = 0; kap < kap_ops.size(); ++kap) {
        if (kap_ops[kap].get_block_index(c_ix) == b_ix) {

          nda::array_const_view<dcomplex, 2> O_kap_b = kap_ops[kap].get_block(c_ix);

          for (int t = 0; t < r; ++t) correlator(t, mu, kap) += trace(matmul(Tmuop_b(t, _, _), O_kap_b));
        }
      }
      
    }
  }
  
  // use special orbital index values to indicate whether the vertex connected to zero
  // is fermionic or bosonic, since this affects the fermionic parity calculation.
  // -2 for fermionic, -3 for bosonic

  int orb_idx_flag = is_fermionic ? -2 : -3;
  backbone.set_orb_inds_of_0_and_vct0(orb_idx_flag, orb_idx_flag);
  int fermionic_sign = backbone.get_parity();
  correlator *= fermionic_sign;

  backbone.reset_all_inds(); // reset directions, pole indices, and orbital indices for the next iteration

  return correlator;
}

// ========= Public self-energy routines ==========

namespace {

  void check_block_count(long n_blocks, long n_sub) {
    if (n_blocks != n_sub)
      throw std::invalid_argument("block_sparse::DiagramEvaluator: the propagator must have one block per invariant subspace, i.e. "
                                  + std::to_string(n_sub) + " blocks, got " + std::to_string(n_blocks));
  }

  // a propagator block has shape (r, d, d); only its rank r is checked where the subspace dimension d is unknown (d < 0)
  void check_block_shape(long b, std::array<long, 3> const &shape, long r, long dim) {
    if (shape[0] != r || (dim >= 0 && (shape[1] != dim || shape[2] != dim)))
      throw std::invalid_argument("block_sparse::DiagramEvaluator: block " + std::to_string(b) + " of the propagator has shape (" + std::to_string(shape[0])
                                  + ", " + std::to_string(shape[1]) + ", " + std::to_string(shape[2]) + "), expected "
                                  + (dim >= 0 ? "(r, d, d) = (" + std::to_string(r) + ", " + std::to_string(dim) + ", " + std::to_string(dim) + ")"
                                              : "rank r = " + std::to_string(r)));
  }

} // namespace

BlockDiagOpFun DiagramEvaluator::native_propagator(triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc) const {
  check_block_count(G_ppsc.size(), subspace_dims.size());
  for (long b = 0; b < subspace_dims.size(); ++b) check_block_shape(b, G_ppsc[b].data().shape(), r, subspace_dims(b));
  return BlockDiagOpFun(G_ppsc);
}

void DiagramEvaluator::check_propagator(BlockDiagOpFun const &Gt) const {
  check_block_count(Gt.get_num_block_cols(), subspace_dims.size());
  for (long b = 0; b < subspace_dims.size(); ++b) check_block_shape(b, Gt.get_block(b).shape(), r, subspace_dims(b));
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

long DiagramEvaluator::get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) {
  Backbone backbone(topology, n, n_int);
  return backbone.num_flat_indices(hyb.poles.size());
}

void DiagramEvaluator::print_self_energy_backbone(nda::array_const_view<int, 2> topology, long f_ix) {
  Backbone backbone(topology, n, n_int);
  backbone.set_flat_index(f_ix, hyb.poles);
  std::cout << "Self-energy backbone for f_ix = " << f_ix << ":\n";
  std::cout << backbone << std::endl;
}

long DiagramEvaluator::get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology) {
  CorrelatorBackbone backbone(topology, n, n_int);
  return backbone.num_flat_indices(hyb.poles.size());
}

// Order the operators by orbital index, not by symmetry set, since they index the external legs of the single-particle Green's function
std::vector<BlockOp> DiagramEvaluator::setup_mu_ops_for_single_ptcle_gf() {
  std::vector<BlockOp> mu_ops;
  // the interaction flavours are the last n_int orbital indices and are not external legs of the single-particle Green's function
  for (int o_ix = 0; o_ix < n_hyb; ++o_ix) {
    auto &F = Fq.Fs[Fq.sym_set_labels(o_ix)];
    int i   = static_cast<int>(Fq.sym_set_inds(o_ix));
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
  return mu_ops;
}

std::vector<BlockOp> DiagramEvaluator::setup_kap_ops_for_single_ptcle_gf() {
  std::vector<BlockOp> kap_ops;
  for (int o_ix = 0; o_ix < n_hyb; ++o_ix) {
    auto &F_dag = Fq.F_dags[Fq.sym_set_labels(o_ix)];
    int i       = static_cast<int>(Fq.sym_set_inds(o_ix));
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
  return kap_ops;
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology) {
  auto Gt = native_propagator(G_ppsc);
  return compute_single_ptcle_gf(Gt, topology);
}

  nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(
  BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology) 
  {
  check_propagator(Gt);
  CorrelatorBackbone backbone(topology, n, n_int);
  auto mu_ops  = setup_mu_ops_for_single_ptcle_gf();
  auto kap_ops = setup_kap_ops_for_single_ptcle_gf();
  return eval_correlator(Gt, backbone, mu_ops, kap_ops);
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology, long f_ix) {
  auto Gt = native_propagator(G_ppsc);
  return compute_single_ptcle_gf(Gt, topology, f_ix);
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(
  BlockDiagOpFun &Gt, nda::array_const_view<int, 2> topology, long f_ix) 
  {
  check_propagator(Gt);
  CorrelatorBackbone backbone(topology, n, n_int);
  auto mu_ops  = setup_mu_ops_for_single_ptcle_gf();
  auto kap_ops = setup_kap_ops_for_single_ptcle_gf();
  return eval_correlator(Gt, backbone, mu_ops, kap_ops, f_ix);
}

nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(
  triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> f_ix_vec) 
  {
  auto Gt = native_propagator(G_ppsc);
  CorrelatorBackbone backbone(topology, n, n_int);
  auto mu_ops  = setup_mu_ops_for_single_ptcle_gf();
  auto kap_ops = setup_kap_ops_for_single_ptcle_gf();
  nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());
  for( auto f_ix : f_ix_vec ) {
    correlator += eval_correlator(Gt, backbone, mu_ops, kap_ops, f_ix);
  }
  return correlator;
}

void DiagramEvaluator::print_single_ptcle_gf_backbone(nda::array_const_view<int, 2> topology, long f_ix) {
  CorrelatorBackbone backbone(topology, n, n_int);
  backbone.set_flat_index(f_ix, hyb.poles);
  std::cout << "Single-particle Green's function backbone for f_ix = " << f_ix << ":\n";
  std::cout << backbone << std::endl;
}

template<bool isComplex>
std::vector<BlockOp> setup_ops_from_triqs_2nd_quant_ops(
  std::vector<triqs::operators::many_body_operator_real> const & ops,
  triqs::atom_diag::atom_diag<isComplex> const &ad) {

  std::vector<BlockOp> bops;

  for (auto &op : ops) {
    auto op_blocks = ad.get_op_mat(op);

    std::vector<nda::array<dcomplex, 2>> blocks;

    for (auto [b, op_block_mat] : itertools::enumerate(op_blocks.block_mat)) {

      int bc = op_blocks.connection(b);
    
      if (bc != -1) {
        auto UR = ad.get_unitary_matrix(b);
        auto UL = ad.get_unitary_matrix(bc);
        auto op_block_mat_transf = UL * op_block_mat * nda::conj(nda::transpose(UR));
        blocks.emplace_back(op_block_mat_transf);
      } else {
        blocks.emplace_back(nda::zeros<dcomplex>(1, 1));
      }
    }
    
    nda::vector<int> block_indices = op_blocks.connection(_);

    BlockOp mu_op(block_indices, blocks);
    bops.push_back(mu_op);
  }
  return bops;
}

template<bool isComplex>
nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, 
    std::vector<triqs::operators::many_body_operator_real> const & ops_tau, 
    std::vector<triqs::operators::many_body_operator_real> const & ops_0, 
    triqs::atom_diag::atom_diag<isComplex> const &ad,
    nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> f_ix_vec){

  // the statistics is the fermion parity of the operators, not whether they commute, see operator_statistics.hpp
  bool is_fermionic = correlator_statistics(ops_tau, ops_0, "compute_one_time_correlator");

  auto Gt = native_propagator(G_ppsc);
  CorrelatorBackbone backbone(topology, n, n_int);

  // the operator blocks are placed by the subspaces of ad, which have to be those of the evaluator
  auto ad_dims = subspace_dims_of(ad);
  bool same    = ad_dims.size() == subspace_dims.size();
  for (long b = 0; same && b < ad_dims.size(); ++b) same = subspace_dims(b) < 0 || ad_dims(b) == subspace_dims(b);
  if (!same) throw std::invalid_argument("compute_one_time_correlator: the subspaces of the atom_diag differ from those the evaluator was built for");

  auto mu_ops  = setup_ops_from_triqs_2nd_quant_ops(ops_tau, ad);
  auto kap_ops = setup_ops_from_triqs_2nd_quant_ops(ops_0, ad);

  nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.size(), kap_ops.size());

  for (auto f_ix : f_ix_vec) {
    correlator += eval_correlator(Gt, backbone, mu_ops, kap_ops, f_ix, is_fermionic);
  }

  return correlator;
}

template
nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, 
    std::vector<triqs::operators::many_body_operator_real> const & ops_tau, 
    std::vector<triqs::operators::many_body_operator_real> const & ops_0, 
    triqs::atom_diag::atom_diag<false> const &ad,
    nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> f_ix_vec);


template
nda::array<dcomplex, 3> DiagramEvaluator::compute_one_time_correlator(
    triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc, 
    std::vector<triqs::operators::many_body_operator_real> const & ops_tau, 
    std::vector<triqs::operators::many_body_operator_real> const & ops_0, 
    triqs::atom_diag::atom_diag<true> const &ad,
    nda::array_const_view<int, 2> topology, nda::array_const_view<long, 1> f_ix_vec);


} // namespace triqs_xca::block_sparse
