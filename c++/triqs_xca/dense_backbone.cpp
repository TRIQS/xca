#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>
#include <cppdlr/utils.hpp>
#include <nda/nda.hpp>
#include <triqs_xca/block_sparse.hpp>
#include <triqs_xca/dense_backbone.hpp>

DenseDiagramEvaluator::DenseDiagramEvaluator(double beta, imtime_ops &itops, nda::array_const_view<dcomplex, 3> hyb,
                                             nda::array_const_view<dcomplex, 3> hyb_refl, nda::vector_const_view<double> hyb_poles,
                                             nda::array_const_view<dcomplex, 3> Gt, DenseFSet &Fset)
   : beta(beta), itops(itops), hyb(hyb), hyb_refl(hyb_refl), Gt(Gt), Fset(Fset), r(itops.rank()), hyb_poles(hyb_poles) {

  dlr_it = itops.get_itnodes();

  // allocate arrays
  int n = hyb.extent(1);
  int N = Gt.extent(1);
  T     = nda::zeros<dcomplex>(r, N, N);
  U     = nda::zeros<dcomplex>(r, N, N);
  GKt   = nda::zeros<dcomplex>(r, N, N);
  Tkaps = nda::zeros<dcomplex>(n, r, N, N);
  Tmu   = nda::zeros<dcomplex>(r, N, N);
  Sigma = nda::zeros<dcomplex>(r, N, N);
}

void DenseDiagramEvaluator::reset() {
  T     = 0;
  U     = 0;
  GKt   = 0;
  Tkaps = 0;
  Tmu   = 0;
  Sigma = 0;
}

void DenseDiagramEvaluator::multiply_vertex(Backbone &backbone, int v_ix) {
  int o_ix = backbone.get_vertex_orb(v_ix); // orbital index
  int l_ix = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));
  // backbone.get_vertex_hyb_ind(v_ix) = i, where i is the # of primes on l
  // l_ix = value of l with i primes

  if (backbone.has_vertex_bar(v_ix)) {   // F has bar
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fset.F_dag_bars(o_ix, l_ix, _, _), T(t, _, _));
    } else {
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fset.F_bars_refl(o_ix, l_ix, _, _), T(t, _, _));
    }
  } else {
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fset.F_dags(o_ix, _, _), T(t, _, _));
    } else {
      for (int t = 0; t < r; t++) T(t, _, _) = matmul(Fset.Fs(o_ix, _, _), T(t, _, _));
    }
  }

  // K factor
  int bv      = backbone.get_vertex_Ksign(v_ix); // sign on K
  double pole = hyb_poles(l_ix);
  // if (backbone.get_vertex_direction(v_ix) == 0) pole = -pole; // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR BACKWARD LINES
  if (bv != 0) {
    for (int t = 0; t < r; t++) { T(t, _, _) = k_it(dlr_it(t), bv * pole) * T(t, _, _); }
  }
}

void DenseDiagramEvaluator::compose_with_edge(Backbone &backbone, int e_ix) {
  GKt   = Gt;
  int m = backbone.m;
  for (int x = 0; x < m - 1; x++) {
    int be      = backbone.get_edge(e_ix, x); // sign on K
    double pole = hyb_poles(backbone.get_pole_ind(x));
    // if (backbone.get_fb(x + 1) == 0) pole = -pole; // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR BACKWARD LINES
    if (be != 0) {
      for (int t = 0; t < r; t++) { GKt(t, _, _) = k_it(dlr_it(t), be * pole) * GKt(t, _, _); }
    }
  }
  T = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T), TIME_ORDERED);
}

void DenseDiagramEvaluator::multiply_prefactor(Backbone &backbone) {
  int m = backbone.m;
  // Multiply by prefactor
  for (int m_ix = 0; m_ix < m - 1; m_ix++) {     // loop over hybridization indices
    int exp = backbone.get_prefactor_Kexp(m_ix); // exponent on K for this hybridization index
    if (exp != 0) {
      int Ksign = backbone.get_prefactor_Ksign(m_ix);     // sign on K for this hybridization index
      double om = hyb_poles(backbone.get_pole_ind(m_ix)); // DLR frequency for this value of this hybridization index
      // if (backbone.get_fb(m_ix + 1) == 0) om = -om;       // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR BACKWARD LINES
      for (int q = 0; q < exp; q++) T /= k_it(0, Ksign * om);
    }
  }
}

void DenseDiagramEvaluator::multiply_zero_vertex(Backbone &backbone, bool is_forward) {
  int n = backbone.n;
  if (is_forward) {
    for (int kap = 0; kap < n; kap++) {
      for (int t = 0; t < r; t++) { Tkaps(kap, t, _, _) = matmul(T(t, _, _), Fset.Fs(kap, _, _)); }
    }
    T = 0;
    for (int mu = 0; mu < n; mu++) {
      Tmu = 0;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) { Tmu(t, _, _) += hyb(t, mu, kap) * Tkaps(kap, t, _, _); }
      }
      for (int t = 0; t < r; t++) { T(t, _, _) += matmul(Fset.F_dags(mu, _, _), Tmu(t, _, _)); }
    }
  } else {
    for (int kap = 0; kap < n; kap++) {
      for (int t = 0; t < r; t++) { Tkaps(kap, t, _, _) = matmul(T(t, _, _), Fset.F_dags(kap, _, _)); }
    }
    T = 0;
    for (int mu = 0; mu < n; mu++) {
      Tmu = 0;
      for (int kap = 0; kap < n; kap++) {
        for (int t = 0; t < r; t++) { Tmu(t, _, _) -= hyb_refl(t, mu, kap) * Tkaps(kap, t, _, _); }
      }
      for (int t = 0; t < r; t++) { T(t, _, _) += matmul(Fset.Fs(mu, _, _), Tmu(t, _, _)); }
    }
  }
}

void DenseDiagramEvaluator::eval_self_energy(Backbone &backbone) {
  int m = backbone.m;
  // loop over all flat indices
  int f_ix_max = static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb_poles.size(), m - 1));
  for (int f_ix = 0; f_ix < f_ix_max; f_ix++) {
    backbone.set_flat_index(f_ix, hyb_poles); // set directions, pole indices, and orbital indices from a single integer index
    eval_self_energy_fixed_indices(backbone); // evaluate the diagram with these directions, poles, and orbital indices
    backbone.reset_all_inds();                // reset directions, pole indices, and orbital indices for the next iteration
  }
}

void DenseDiagramEvaluator::eval_self_energy_fixed_indices(Backbone &backbone) {
  int m = backbone.m;

  // 1. Starting from tau_1, proceed right to left, performing multiplications at vertices and convolutions at edges, until reaching the vertex
  // containing the undecomposed hybridization line Delta_{mu kappa}.
  T = Gt; // T stores the result moving left to right
  // T is initialized to Gt, which is always the function at the rightmost edge
  for (int v = 1; v < backbone.get_topology(0, 1); v++) { // loop from the first vertex to before the special vertex
    multiply_vertex(backbone, v);
    compose_with_edge(backbone, v);
  }

  // 2. For each kappa, multiply by F_kappa(^dag). Then for each mu, kappa, multiply by Delta_{mu kappa}, and sum over kappa. Finally for each mu,
  // multiply F_mu[^dag] and sum over mu.
  multiply_zero_vertex(backbone, (not backbone.has_vertex_dag(0)));

  // 3. Continue right to left until the final vertex multiplication is complete.
  for (int v = backbone.get_topology(0, 1) + 1; v < 2 * m; v++) { // loop from the special vertex to the last vertex
    compose_with_edge(backbone, v - 1);
    multiply_vertex(backbone, v);
  }

  multiply_prefactor(backbone);
  int diag_order_sign = (m % 2 == 0) ? -1 : 1;
  if (backbone.get_fb(0) == 0) diag_order_sign *= -1; // if the first hybridization line is backward, there is an additional sign change
  T *= diag_order_sign * backbone.prefactor_sign;
  Sigma += T;
}

void DenseDiagramEvaluator::multiply_vertex_corr(Backbone &backbone, int v_ix) {
  int o_ix = backbone.get_vertex_orb(v_ix); // orbital index
  int l_ix = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));
  // backbone.get_vertex_hyb_ind(v_ix) = i, where i is the # of primes on l
  // l_ix = value of l with i primes

  if (backbone.has_vertex_bar(v_ix)) {   // F has bar
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) U(t, _, _) = matmul(Fset.F_dag_bars(o_ix, l_ix, _, _), U(t, _, _));
    } else {
      for (int t = 0; t < r; t++) U(t, _, _) = matmul(Fset.F_bars_refl(o_ix, l_ix, _, _), U(t, _, _));
    }
  } else {
    if (backbone.has_vertex_dag(v_ix)) { // F has dagger
      for (int t = 0; t < r; t++) U(t, _, _) = matmul(Fset.F_dags(o_ix, _, _), U(t, _, _));
    } else {
      for (int t = 0; t < r; t++) U(t, _, _) = matmul(Fset.Fs(o_ix, _, _), U(t, _, _));
    }
  }

  // K factor
  // skip if last vertex
  if (v_ix != 2 * backbone.m - 1) {
    int bv      = backbone.get_vertex_Ksign(v_ix); // sign on K
    double pole = hyb_poles(l_ix);
    // if (backbone.get_vertex_direction(v_ix) == 0) pole = -pole; // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR BACKWARD LINES
    if (bv != 0) {
      for (int t = 0; t < r; t++) { U(t, _, _) = k_it(dlr_it(t), bv * pole) * U(t, _, _); }
    }
  }
}

void DenseDiagramEvaluator::multiply_vertex_corr_reverse(Backbone &backbone, int v_ix) {

  int o_ix = backbone.get_vertex_orb(v_ix); // orbital index
  int l_ix = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));

  // backbone.get_vertex_hyb_ind(v_ix) = i, where i is the # of primes on l
  // l_ix = value of l with i primes

  bool has_bar = backbone.has_vertex_bar(v_ix);
  bool has_dag = backbone.has_vertex_dag(v_ix);

  auto F_selector = [&]() {
    if (has_bar)
      return has_dag ? Fset.F_dag_bars(o_ix, l_ix, _, _) : Fset.F_bars_refl(o_ix, l_ix, _, _);
    else
      return has_dag ? Fset.F_dags(o_ix, _, _) : Fset.Fs(o_ix, _, _);
  };

  nda::array_const_view<dcomplex, 2> F = F_selector();

  for (int t = 0; t < r; t++) U(t, _, _) = matmul(U(t, _, _), F);

  // K factor
  int Ksign = backbone.get_vertex_Ksign(v_ix); // sign on K
  if (Ksign != 0) {
    for (int t = 0; t < r; t++) U(t, _, _) *= k_it(dlr_it(t), -Ksign * hyb_poles(l_ix));
  }
}

void DenseDiagramEvaluator::compose_with_edge_corr(Backbone &backbone, int e_ix) {
  GKt   = Gt;
  int m = backbone.m;
  for (int x = 0; x < m - 1; x++) {
    int be      = backbone.get_edge(e_ix, x); // sign on K
    double pole = hyb_poles(backbone.get_pole_ind(x));
    // if (backbone.get_fb(x + 1) == 0) pole = -pole; // MODIFIED LOGIC -- HYB_POLES->(-HYB_POLES) FOR BACKWARD LINES
    if (be != 0) {
      for (int t = 0; t < r; t++) { GKt(t, _, _) = k_it(dlr_it(t), be * pole) * GKt(t, _, _); }
    }
  }
  U = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(U), TIME_ORDERED);
}

void DenseDiagramEvaluator::compose_with_edge_corr_reverse(Backbone &backbone, int e_ix) {
  GKt   = Gt;
  int m = backbone.m;
  for (int x = 0; x < m - 1; x++) {
    int Ksign = backbone.get_edge(e_ix, x); // sign on K
    if (Ksign != 0) {
      double pole = hyb_poles(backbone.get_pole_ind(x));
      for (int t = 0; t < r; t++) GKt(t, _, _) *= k_it(dlr_it(t), Ksign * pole);
    }
  }
  U = itops.convolve(beta, itops.vals2coefs(U), itops.vals2coefs(GKt), TIME_ORDERED);
}

nda::array<dcomplex, 3> DenseDiagramEvaluator::eval_correlator(CorrelatorBackbone &backbone, nda::array<dcomplex, 3> mu_ops,
                                                               nda::array<dcomplex, 3> kap_ops) {
  int m = backbone.m;
  // loop over all flat indices
  int f_ix_max                       = static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb_poles.size(), m - 1));
  nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.extent(0), kap_ops.extent(0));
  nda::array<dcomplex, 3> Tmuop      = nda::zeros<dcomplex>(r, Gt.extent(1), Gt.extent(1));
  for (int f_ix = 0; f_ix < f_ix_max; ++f_ix) {
    backbone.set_flat_index(f_ix, hyb_poles); // set directions, pole indices, and orbital indices from a single integer index
    eval_correlator_fixed_indices(backbone);  // evaluate the diagram with these directions, poles, and orbital indices
    for (int mu = 0; mu < mu_ops.extent(0); ++mu) {
      for (int t = 0; t < r; ++t) { Tmuop(t, _, _) = matmul(U(t, _, _), matmul(mu_ops(mu, _, _), T(t, _, _))); }
      for (int kap = 0; kap < kap_ops.extent(0); ++kap) {
        for (int t = 0; t < r; ++t) { correlator(t, mu, kap) += trace(matmul(Tmuop(t, _, _), kap_ops(kap, _, _))); }
      }
    }
    backbone.reset_all_inds(); // reset directions, pole indices, and orbital indices for the next iteration
  }
  return correlator;
}

void DenseDiagramEvaluator::eval_correlator_fixed_indices(CorrelatorBackbone &backbone) {
  int m = backbone.m;

  // evaluate the first sequence of backbone products and convolutions from tau_1 to tau, proceeding right to left, not inculding the creation and
  // annihilation matrices at the end points. the result is an N x N matrix-valued function of tau.
  T = Gt; // T stores the result moving right to left
  // T is initialized to Gt, which is always the function at the rightmost edge
  for (int v = 1; v < backbone.get_topology(0, 1); ++v) { // loop from the first vertex to before the special vertex
    multiply_vertex(backbone, v);
    compose_with_edge(backbone, v);
  }

  // evaluate the second sequence of backbone products and convolutions from tau to beta, using a change of variables to perform convolutions. the
  // result is another N x N matrix-valued function of tau.
  U = Gt; // U stores the result moving right to left

  for (int v = 2 * m - 1; v > backbone.get_topology(0, 1); v--) {
    multiply_vertex_corr_reverse(backbone, v);
    compose_with_edge_corr_reverse(backbone, v - 1);
  }

  U = itops.reflect(U);

  multiply_prefactor(backbone);
  int diag_order_sign = 1; // (m % 2 == 1) ? -1 : 1;
  T *= diag_order_sign * backbone.prefactor_sign;
}
