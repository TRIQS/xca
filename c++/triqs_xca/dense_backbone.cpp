#include <nda/nda.hpp>

#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>

#include "triqs_xca/hyb.hpp"
#include "triqs_xca/dense_backbone.hpp"

namespace triqs_xca::dense {

  using cppdlr::_;

  using nda::trace;
  using nda::linalg::matmul;

  using triqs_xca::atom_diag::get_operators_dense;

  DenseDiagramEvaluator::DenseDiagramEvaluator(double beta, double eps, imtime_ops &itops, nda::vector_const_view<double> hyb_poles,
                                               nda::array_const_view<dcomplex, 3> hyb_coeffs, DenseFSet &Fset)
     : tau_mesh(triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, itops.lambda() / beta, eps)),
       beta(beta),
       itops(itops),
       dlr_it(itops.get_itnodes()),
       hyb(tau_mesh, nda::make_regular(hyb_poles / beta), hyb_coeffs, -1.0), // Scaling of poles by beta?
       Fset(Fset),
       r(itops.rank()),
       n(hyb_coeffs.extent(1)),
       //n(Fset.Fs.extent(0)),
       N(Fset.Fs.extent(1)),
       // allocate arrays
       Sigma(nda::zeros<dcomplex>(r, N, N)),
       T(nda::zeros<dcomplex>(r, N, N)),
       U(nda::zeros<dcomplex>(r, N, N)),
       GKt(nda::zeros<dcomplex>(r, N, N)),
       Tkaps(nda::zeros<dcomplex>(n, r, N, N)), // Largest memory footprint, speeding up multiply_left_vertex_and_right_zero_vertex
       Tmu(nda::zeros<dcomplex>(r, N, N)) {}

  DenseDiagramEvaluator::DenseDiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                               triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<true> const &ad)
     : tau_mesh(tau_mesh),
       beta(tau_mesh.beta()),
       itops(tau_mesh.dlr_it()),
       dlr_it(itops.get_itnodes()),
       hyb(tau_mesh, hyb_poles, hyb_coeffs, -1.0),
       Fset(get_operators_dense(ad, hyb_coeffs)),
       r(itops.rank()),
       n(ad.get_fops().size()), // number of fermion flavours (spin-orbitals)
       N(ad.get_full_hilbert_space_dim()),
       // allocate arrays
       Sigma(nda::zeros<dcomplex>(r, N, N)),
       T(nda::zeros<dcomplex>(r, N, N)),
       U(nda::zeros<dcomplex>(r, N, N)),
       GKt(nda::zeros<dcomplex>(r, N, N)),
       Tkaps(nda::zeros<dcomplex>(n, r, N, N)), // Largest memory footprint, speeding up multiply_left_vertex_and_right_zero_vertex
       Tmu(nda::zeros<dcomplex>(r, N, N)) {}

  DenseDiagramEvaluator::DenseDiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                               triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<false> const &ad)
     : tau_mesh(tau_mesh),
       beta(tau_mesh.beta()),
       itops(tau_mesh.dlr_it()),
       dlr_it(itops.get_itnodes()),
       hyb(tau_mesh, hyb_poles, hyb_coeffs, -1.0),
       Fset(get_operators_dense(ad, hyb_coeffs)),
       r(itops.rank()),
       n(ad.get_fops().size()), // number of fermion flavours (spin-orbitals)
       N(ad.get_full_hilbert_space_dim()),
       // allocate arrays
       Sigma(nda::zeros<dcomplex>(r, N, N)),
       T(nda::zeros<dcomplex>(r, N, N)),
       U(nda::zeros<dcomplex>(r, N, N)),
       GKt(nda::zeros<dcomplex>(r, N, N)),
       Tkaps(nda::zeros<dcomplex>(n, r, N, N)), // Largest memory footprint, speeding up multiply_left_vertex_and_right_zero_vertex
       Tmu(nda::zeros<dcomplex>(r, N, N)) {}

  void DenseDiagramEvaluator::reset() {
    T     = 0;
    U     = 0;
    GKt   = 0;
    Tkaps = 0;
    Tmu   = 0;
    Sigma = 0;
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DenseDiagramEvaluator::compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology) {
    Backbone backbone(topology, n);
    eval_self_energy(G_ppsc[0].data(), backbone);
    auto sigma_gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DenseDiagramEvaluator::compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology,
                                                                                           int f_ix) {
    Backbone backbone(topology, n);
    eval_self_energy_fixed_indices(G_ppsc[0].data(), backbone, f_ix); // evaluate the diagram with these directions, poles, and orbital indices
    auto sigma_gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  void DenseDiagramEvaluator::multiply_left_vertex(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone, int v_ix) {
    int o_ix = backbone.get_vertex_orb(v_ix); // orbital index
    int l_ix = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));
    // backbone.get_vertex_hyb_ind(v_ix) = i, where i is the # of primes on l
    // l_ix = value of l with i primes
    auto F = Fset.get_operator(backbone, v_ix, o_ix, l_ix);
    for (int t = 0; t < r; t++) T_buf(t, _, _) = matmul(F, T_buf(t, _, _));
    hyb.multiply_kernel_on_vertex(T_buf, backbone, v_ix, l_ix);
  }

  void DenseDiagramEvaluator::integrate_left_edge(nda::array_view<dcomplex, 3> T_buf, nda::array_const_view<dcomplex, 3> Gt, Backbone &backbone,
                                                  int e_ix) {
    GKt = Gt;
    hyb.multiply_kernels_on_edge(GKt, backbone, e_ix);
    T_buf = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T_buf), cppdlr::TIME_ORDERED);
  }

  void DenseDiagramEvaluator::multiply_prefactor(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone) {
    hyb.multiply_kernels_prefactor(T_buf, backbone);
  }

  void DenseDiagramEvaluator::multiply_left_vertex_and_right_zero_vertex(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone, int vct0) {

    bool is_forward                            = backbone.has_vertex_dag(vct0);
    nda::array_const_view<dcomplex, 3> hyb_too = is_forward ? hyb.values : hyb.values_reflect;

    // Save compute by precomputing Tkaps = T_buf * F_kap for all kappa, since this is needed for each mu
    // at the cost of storing an n x r x N x N array (Tkaps) instead of an r x N x N array

    for (int kap = 0; kap < n; kap++) {
      nda::array_const_view<dcomplex, 2> F_kap = Fset.get_operator(backbone, 0, kap);
      for (int t = 0; t < r; t++) Tkaps(kap, t, _, _) = matmul(T_buf(t, _, _), F_kap);
    }

    T_buf = 0;

    for (int mu = 0; mu < n; mu++) {
      Tmu = 0;
      for (int kap = 0; kap < n; kap++) {
        nda::array_const_view<dcomplex, 1> hyb_t = hyb_too(_, mu, kap);
        for (int t = 0; t < r; t++) Tmu(t, _, _) += hyb_t(t) * Tkaps(kap, t, _, _);
      }
      nda::array_const_view<dcomplex, 2> F_mu = Fset.get_operator(backbone, vct0, mu);
      for (int t = 0; t < r; t++) T_buf(t, _, _) += matmul(F_mu, Tmu(t, _, _));
    }
  }

  int DenseDiagramEvaluator::get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) {
    Backbone backbone(topology, n);
    return get_num_self_energy_backbones(backbone);
  }

  int DenseDiagramEvaluator::get_num_self_energy_backbones(Backbone &backbone) {
    int f_ix_max = static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb.poles.size(), backbone.m - 1));
    return f_ix_max;
  }

  void DenseDiagramEvaluator::eval_self_energy(nda::array_const_view<dcomplex, 3> Gt, Backbone &backbone) {
    // loop over all flat indices
    int f_ix_max = get_num_self_energy_backbones(backbone);
    for (int f_ix = 0; f_ix < f_ix_max; f_ix++) {
      eval_self_energy_fixed_indices(Gt, backbone, f_ix); // evaluate the diagram with these directions, poles, and orbital indices
    }
  }

  void DenseDiagramEvaluator::eval_self_energy_fixed_indices(nda::array_const_view<dcomplex, 3> Gt, Backbone &backbone, int f_ix) {
    int m    = backbone.m;
    int vct0 = backbone.get_topology(0, 1); // Vertex Connected To zero

    backbone.set_flat_index(f_ix, hyb.poles); // set directions, pole indices, and orbital indices from a single integer index

    // 1. Starting from tau_1, proceed right to left, performing multiplications at vertices and convolutions at edges, until reaching the vertex
    // containing the undecomposed hybridization line Delta_{mu kappa}.
    T = Gt;
    // T is initialized to Gt, which is always the function at the rightmost edge
    for (int v = 1; v < vct0; v++) { // loop from the first vertex to before the special vertex
      multiply_left_vertex(T, backbone, v);
      integrate_left_edge(T, Gt, backbone, v);
    }

    // 2. For each kappa, multiply by F_kappa(^dag). Then for each mu, kappa, multiply by Delta_{mu kappa}, and sum over kappa. Finally for each mu,
    // multiply F_mu[^dag] and sum over mu.
    multiply_left_vertex_and_right_zero_vertex(T, backbone, vct0);

    // 3. Continue right to left until the final vertex multiplication is complete.
    for (int v = vct0 + 1; v < 2 * m; v++) { // loop from the special vertex to the last vertex
      integrate_left_edge(T, Gt, backbone, v - 1);
      multiply_left_vertex(T, backbone, v);
    }

    multiply_prefactor(T, backbone);
    int diag_order_sign = (m % 2 == 0) ? -1 : 1;
    if (backbone.get_fb(0) == 0) diag_order_sign *= -1; // if the first hybridization line is backward, there is an additional sign change
    T *= diag_order_sign * backbone.prefactor_sign;
    Sigma += T;

    backbone.reset_all_inds(); // reset directions, pole indices, and orbital indices for the next iteration
  }

  void DenseDiagramEvaluator::multiply_right_vertex(nda::array_view<dcomplex, 3> U_buf, Backbone &backbone, int v_ix) {
    int o_ix = backbone.get_vertex_orb(v_ix); // orbital index
    int l_ix = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));
    // backbone.get_vertex_hyb_ind(v_ix) = i, where i is the # of primes on l
    // l_ix = value of l with i primes
    auto F = Fset.get_operator(backbone, v_ix, o_ix, l_ix);
    for (int t = 0; t < r; t++) U_buf(t, _, _) = matmul(U_buf(t, _, _), F);
    hyb.multiply_kernel_on_vertex(U_buf, backbone, v_ix, l_ix, -1.0);
  }

  void DenseDiagramEvaluator::integrate_right_edge(nda::array_view<dcomplex, 3> U_buf, nda::array_const_view<dcomplex, 3> Gt, Backbone &backbone,
                                                   int e_ix) {
    GKt = Gt;
    hyb.multiply_kernels_on_edge(GKt, backbone, e_ix);
    U_buf = itops.convolve(beta, itops.vals2coefs(U_buf), itops.vals2coefs(GKt), cppdlr::TIME_ORDERED);
  }

  nda::array<dcomplex, 3> DenseDiagramEvaluator::eval_correlator(nda::array_const_view<dcomplex, 3> Gt, CorrelatorBackbone &backbone,
                                                                 nda::array<dcomplex, 3> mu_ops, nda::array<dcomplex, 3> kap_ops) {
    int m        = backbone.m;
    int f_ix_max = static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb.poles.size(), m - 1));

    nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.extent(0), kap_ops.extent(0));

    // loop over all flat indices
    for (int f_ix = 0; f_ix < f_ix_max; ++f_ix) {
      correlator += eval_correlator(Gt, backbone, mu_ops, kap_ops, f_ix); // evaluate the diagram with these directions, poles, and orbital indices
    }

    return correlator;
  }

  nda::array<dcomplex, 3> DenseDiagramEvaluator::eval_correlator(nda::array_const_view<dcomplex, 3> Gt, CorrelatorBackbone &backbone,
                                                                 nda::array<dcomplex, 3> mu_ops, nda::array<dcomplex, 3> kap_ops, int f_ix) {

    int m = backbone.m;

    backbone.set_flat_index(f_ix, hyb.poles); // set directions, pole indices, and orbital indices from a single integer index

    nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.extent(0), kap_ops.extent(0));

    // evaluate the first sequence of backbone products and convolutions from tau_1 to tau, proceeding right to left, not inculding the creation and
    // annihilation matrices at the end points. the result is an N x N matrix-valued function of tau.
    T = Gt;
    // T is initialized to Gt, which is always the function at the rightmost edge
    for (int v = 1; v < backbone.get_topology(0, 1); ++v) { // loop from the first vertex to before the special vertex
      multiply_left_vertex(T, backbone, v);
      integrate_left_edge(T, Gt, backbone, v);
    }

    // evaluate the second sequence of backbone products and convolutions from tau to beta, using a change of variables to perform convolutions. the
    // result is another N x N matrix-valued function of tau.
    U = Gt;

    for (int v = 2 * m - 1; v > backbone.get_topology(0, 1); v--) {
      multiply_right_vertex(U, backbone, v);
      integrate_right_edge(U, Gt, backbone, v - 1);
    }

    U = itops.reflect(U);

    multiply_prefactor(T, backbone);
    int diag_order_sign = 1; // (m % 2 == 1) ? -1 : 1;
    T *= diag_order_sign * backbone.prefactor_sign;

    nda::array<dcomplex, 3> Tmuop = nda::zeros<dcomplex>(r, Gt.extent(1), Gt.extent(1));

    for (int mu = 0; mu < mu_ops.extent(0); ++mu) {
      for (int t = 0; t < r; ++t) Tmuop(t, _, _) = matmul(U(t, _, _), matmul(mu_ops(mu, _, _), T(t, _, _)));

      for (int kap = 0; kap < kap_ops.extent(0); ++kap) {
        for (int t = 0; t < r; ++t) correlator(t, mu, kap) += trace(matmul(Tmuop(t, _, _), kap_ops(kap, _, _)));
      }
    }

    backbone.reset_all_inds(); // reset directions, pole indices, and orbital indices for the next iteration

    return correlator;
  }

  int DenseDiagramEvaluator::get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology) {
    CorrelatorBackbone backbone(topology, n);
    return static_cast<int>(backbone.fb_ix_max * backbone.o_ix_max * pow(hyb.poles.size(), backbone.m - 1));
  }

  nda::array<dcomplex, 3> DenseDiagramEvaluator::compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology) {
    CorrelatorBackbone backbone(topology, n);

    auto mu_ops  = Fset.Fs;
    auto kap_ops = Fset.F_dags;

    /*
      auto correlator = triqs::gfs::gf<triqs::mesh::dlr_imtime>(
        tau_mesh, eval_correlator(backbone, mu_ops, kap_ops));
      return correlator;
      */

    return eval_correlator(G_ppsc[0].data(), backbone, mu_ops, kap_ops);
  }

  nda::array<dcomplex, 3> DenseDiagramEvaluator::compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology, int f_ix) {
    CorrelatorBackbone backbone(topology, n);
    auto mu_ops  = Fset.Fs;
    auto kap_ops = Fset.F_dags;

    /*
      auto correlator = triqs::gfs::gf<triqs::mesh::dlr_imtime>(
        tau_mesh, eval_correlator(backbone, mu_ops, kap_ops, f_ix));
      return correlator;
      */

    return eval_correlator(G_ppsc[0].data(), backbone, mu_ops, kap_ops, f_ix);
  }
} // namespace triqs_xca::dense