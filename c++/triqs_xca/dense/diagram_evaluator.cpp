#include <stdexcept>
#include <string>

#include <nda/nda.hpp>

#include <cppdlr/dlr_imtime.hpp>
#include <cppdlr/dlr_kernels.hpp>

#include "triqs_xca/hyb.hpp"
#include "triqs_xca/dense/dynint.hpp"
#include "triqs_xca/dense/diagram_evaluator.hpp"
#include "triqs_xca/dense/atom_diag.hpp"
#include "triqs_xca/operator_statistics.hpp"

namespace triqs_xca::dense {

  using cppdlr::_;

  using nda::trace;
  using nda::linalg::matmul;


  DiagramEvaluator::DiagramEvaluator(double beta, double eps, imtime_ops &itops, nda::vector_const_view<double> hyb_poles,
                                               nda::array_const_view<dcomplex, 3> hyb_coeffs, FSet &Fset)
     : tau_mesh(triqs::mesh::dlr_imtime(beta, triqs::mesh::Fermion, itops.lambda() / beta, eps, false)),
       beta(beta),
       itops(itops),
       dlr_it(itops.get_itnodes()),
       hyb(tau_mesh, nda::make_regular(hyb_poles / beta), hyb_coeffs, -1.0), // Scaling of poles by beta?
       Fset(Fset),
       r(itops.rank()),
       n(hyb_coeffs.extent(1)),
       n_hyb(n),
       n_int(0), // No dynamic interactions in this constructor
       //n(Fset.Fs.extent(0)),
       N(Fset.Fs.extent(1)),
       // allocate arrays
       Sigma(nda::zeros<dcomplex>(r, N, N)),
       T(nda::zeros<dcomplex>(r, N, N)),
       U(nda::zeros<dcomplex>(r, N, N)),
       GKt(nda::zeros<dcomplex>(r, N, N)),
       Tkaps(nda::zeros<dcomplex>(n, r, N, N)), // Largest memory footprint, speeding up multiply_left_vertex_and_right_zero_vertex
       Tmu(nda::zeros<dcomplex>(r, N, N)) {}

  template <bool isComplex>
  DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                               triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<isComplex> const &ad)
     : tau_mesh(tau_mesh),
       beta(tau_mesh.beta()),
       itops(tau_mesh.dlr_it()),
       dlr_it(itops.get_itnodes()),
       hyb(tau_mesh, hyb_poles, hyb_coeffs, -1.0),
       Fset(atom_diag::get_operators(ad, hyb_coeffs)),
       r(itops.rank()),
       n(ad.get_fops().size()), // number of fermion flavours (spin-orbitals)
       n_hyb(n),
       n_int(0), // No dynamic interactions in this constructor
       N(ad.get_full_hilbert_space_dim()),
       // allocate arrays
       Sigma(nda::zeros<dcomplex>(r, N, N)),
       T(nda::zeros<dcomplex>(r, N, N)),
       U(nda::zeros<dcomplex>(r, N, N)),
       GKt(nda::zeros<dcomplex>(r, N, N)),
       Tkaps(nda::zeros<dcomplex>(n, r, N, N)), // Largest memory footprint, speeding up multiply_left_vertex_and_right_zero_vertex
       Tmu(nda::zeros<dcomplex>(r, N, N)) {}

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                                        triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<true> const &ad);

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                                        triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<false> const &ad);

  template <bool isComplex>
  DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                               triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<isComplex> const &ad,
                                               std::vector<triqs::operators::many_body_operator_real> const &dynint_ops,
                                               nda::array_const_view<dcomplex, 3> dynint_coeffs)
     : tau_mesh(tau_mesh),
       beta(tau_mesh.beta()),
       itops(tau_mesh.dlr_it()),
       dlr_it(itops.get_itnodes()),
       hyb(tau_mesh, hyb_poles, hyb::get_extended_coefficients(hyb_coeffs, dynint_coeffs), -1.0),
       Fset(dynint::get_operators_and_interactions(ad, hyb_coeffs, dynint_coeffs, dynint_ops)),
       r(itops.rank()),
       n(ad.get_fops().size() + dynint_ops.size()), // number of fermion flavours (spin-orbitals)
       n_hyb(ad.get_fops().size()),
       n_int(dynint_ops.size()), // Number of dynamic interactions
       N(ad.get_full_hilbert_space_dim()),
       // allocate arrays
       Sigma(nda::zeros<dcomplex>(r, N, N)),
       T(nda::zeros<dcomplex>(r, N, N)),
       U(nda::zeros<dcomplex>(r, N, N)),
       GKt(nda::zeros<dcomplex>(r, N, N)),
       Tkaps(nda::zeros<dcomplex>(n, r, N, N)), // Largest memory footprint, speeding up multiply_left_vertex_and_right_zero_vertex
       Tmu(nda::zeros<dcomplex>(r, N, N)) {}

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                                        triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<true> const &ad,
                                                        std::vector<triqs::operators::many_body_operator_real> const &dynint_ops,
                                                        nda::array_const_view<dcomplex, 3> dynint_coeffs);

  template DiagramEvaluator::DiagramEvaluator(nda::vector_const_view<double> hyb_poles, nda::array_const_view<dcomplex, 3> hyb_coeffs,
                                                        triqs::mesh::dlr_imtime tau_mesh, triqs::atom_diag::atom_diag<false> const &ad,
                                                        std::vector<triqs::operators::many_body_operator_real> const &dynint_ops,
                                                        nda::array_const_view<dcomplex, 3> dynint_coeffs);

  void DiagramEvaluator::reset() {
    T     = 0;
    U     = 0;
    GKt   = 0;
    Tkaps = 0;
    Tmu   = 0;
    Sigma = 0;
  }

  nda::array_const_view<dcomplex, 3> DiagramEvaluator::native_propagator(gf_vt G_ppsc) const {
    if (G_ppsc.size() != 1)
      throw std::invalid_argument("dense::DiagramEvaluator: G_ppsc must be a single block over the full Hilbert space, got "
                                  + std::to_string(G_ppsc.size()) + " blocks");
    auto G = G_ppsc[0].data();
    if (G.extent(0) != r || G.extent(1) != N || G.extent(2) != N)
      throw std::invalid_argument("dense::DiagramEvaluator: G_ppsc has shape (" + std::to_string(G.extent(0)) + ", " + std::to_string(G.extent(1))
                                  + ", " + std::to_string(G.extent(2)) + "), expected (r, N, N) = (" + std::to_string(r) + ", " + std::to_string(N)
                                  + ", " + std::to_string(N) + ")");
    return G;
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology) {
    auto Gt = native_propagator(G_ppsc);
    reset(); // a previous call that threw may have left a partial sum in Sigma
    Backbone backbone(topology, n, n_int);
    eval_self_energy(Gt, backbone);
    auto sigma_gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology,
                                                                                      long f_ix) {
    auto Gt = native_propagator(G_ppsc);
    reset();
    Backbone backbone(topology, n, n_int);
    eval_self_energy_fixed_indices(Gt, backbone, f_ix); // evaluate the diagram with these directions, poles, and orbital indices
    auto sigma_gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy(gf_vt G_ppsc, nda::array_const_view<int, 2> topology,
                                                                                      nda::array_const_view<long, 1> f_ix_vec) {
    auto Gt = native_propagator(G_ppsc);
    reset();
    Backbone backbone(topology, n, n_int);
    for (long f_ix : f_ix_vec)
      eval_self_energy_fixed_indices(Gt, backbone, f_ix); // evaluate the diagram with these directions, poles, and orbital indices
    auto sigma_gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy_by_pairs(gf_vt G_ppsc,
                                                                                                    nda::array_const_view<int, 2> topology) {
    auto Gt = native_propagator(G_ppsc);
    reset();
    Backbone backbone(topology, n, n_int);
    eval_self_energy_by_pairs(Gt, backbone);
    auto sigma_gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy_by_pairs(gf_vt G_ppsc, nda::array_const_view<int, 2> topology,
                                                                                               long f_ix) {
    auto Gt = native_propagator(G_ppsc);
    reset();
    Backbone backbone(topology, n, n_int);
    eval_self_energy_fixed_index_pair(Gt, backbone, f_ix);
    auto sigma_gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  triqs::gfs::block_gf<triqs::mesh::dlr_imtime> DiagramEvaluator::compute_self_energy_by_pairs(gf_vt G_ppsc, nda::array_const_view<int, 2> topology,
                                                                                               nda::array_const_view<long, 1> f_ix_vec) {
    auto Gt = native_propagator(G_ppsc);
    reset();
    Backbone backbone(topology, n, n_int);
    for (long f_ix : f_ix_vec) eval_self_energy_fixed_index_pair(Gt, backbone, f_ix);
    auto sigma_gf = triqs::gfs::gf<triqs::mesh::dlr_imtime>(tau_mesh, this->Sigma);
    reset();
    return std::vector{sigma_gf};
  }

  void DiagramEvaluator::multiply_left_vertex(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone, int v_ix) {
    int o_ix = backbone.get_vertex_orb(v_ix); // orbital index
    int l_ix = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));
    // backbone.get_vertex_hyb_ind(v_ix) = i, where i is the # of primes on l
    // l_ix = value of l with i primes
    auto F = Fset.get_operator(backbone, v_ix, o_ix, l_ix);
    for (int t = 0; t < r; t++) T_buf(t, _, _) = matmul(F, T_buf(t, _, _));
    hyb.multiply_kernel_on_vertex(T_buf, backbone, v_ix, l_ix);
  }

  void DiagramEvaluator::integrate_left_edge(nda::array_view<dcomplex, 3> T_buf, nda::array_const_view<dcomplex, 3> Gt, Backbone &backbone,
                                                  int e_ix) {
    GKt = Gt;
    hyb.multiply_kernels_on_edge(GKt, backbone, e_ix);
    T_buf = itops.convolve(beta, itops.vals2coefs(GKt), itops.vals2coefs(T_buf), cppdlr::TIME_ORDERED);
  }

  void DiagramEvaluator::multiply_prefactor(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone) {
    hyb.multiply_kernels_prefactor(T_buf, backbone);
  }

  void DiagramEvaluator::multiply_left_vertex_and_right_zero_vertex(nda::array_view<dcomplex, 3> T_buf, Backbone &backbone, int vct0) {

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
        // The backward branch transposes the coefficient index, as the barred operators of the interior lines do: forward is
        // F_dag(mu) ... F(kap) with the daggered index first, backward is F(mu) ... F_dag(kap) and needs (kap, mu). Reading (mu, kap) on
        // both branches breaks unitary invariance under a complex basis rotation, which the legacy path avoids by transposing at construction
        nda::array_const_view<dcomplex, 1> hyb_t = is_forward ? hyb_too(_, mu, kap) : hyb_too(_, kap, mu);

        // set the orbital indices of the vertex connected to zero
        // and compute the fermionic permutation parity of the resulting diagram.

        backbone.set_orb_inds_of_0_and_vct0(kap, mu);
        int fermionic_parity = backbone.get_parity();

        for (int t = 0; t < r; t++) Tmu(t, _, _) += fermionic_parity * hyb_t(t) * Tkaps(kap, t, _, _);
      }
      nda::array_const_view<dcomplex, 2> F_mu = Fset.get_operator(backbone, vct0, mu);
      for (int t = 0; t < r; t++) T_buf(t, _, _) += matmul(F_mu, Tmu(t, _, _));
    }
  }

  long DiagramEvaluator::get_num_self_energy_backbones(nda::array_const_view<int, 2> topology) {
    Backbone backbone(topology, n, n_int);
    return get_num_self_energy_backbones(backbone);
  }

  long DiagramEvaluator::get_num_self_energy_backbones(Backbone &backbone) {
    long f_ix_max = backbone.num_flat_indices(hyb.poles.size());
    return f_ix_max;
  }

  void DiagramEvaluator::eval_self_energy(nda::array_const_view<dcomplex, 3> Gt, Backbone &backbone) {
    // loop over all flat indices
    long f_ix_max = get_num_self_energy_backbones(backbone);
    for (long f_ix = 0; f_ix < f_ix_max; f_ix++) {
      eval_self_energy_fixed_indices(Gt, backbone, f_ix); // evaluate the diagram with these directions, poles, and orbital indices
    }
  }

  void DiagramEvaluator::eval_self_energy_by_pairs(nda::array_const_view<dcomplex, 3> Gt, Backbone &backbone) {
    // eval_self_energy_fixed_index_pair(f_ix) evaluates both values of fb(0) (the direction of the
    // hybridization line connected to vertex 0) for the (orbital, pole, fb(1), ...) combination
    // encoded by f_ix. To cover every diagram exactly once, only call it for f_ix whose own
    // fb(0) == 0, i.e. fb_ix = f_ix / (o_ix_max * p_ix_max) is even.
    long f_ix_max = get_num_self_energy_backbones(backbone);
    long n_p      = f_ix_max / backbone.fb_ix_max; // o_ix_max * p_ix_max
    for (long f_ix = 0; f_ix < f_ix_max; ++f_ix) {
      if ((f_ix / n_p) % 2 == 0) { eval_self_energy_fixed_index_pair(Gt, backbone, f_ix); }
    }
  }

  void DiagramEvaluator::eval_self_energy_fixed_index_pair(nda::array_const_view<dcomplex, 3> Gt, Backbone &backbone, long f_ix) {
    int m    = backbone.m;
    int vct0 = backbone.get_topology(0, 1);

    backbone.set_flat_index(f_ix, hyb.poles);

    T = Gt;
    for (int v = 1; v < vct0; v++) {
      multiply_left_vertex(T, backbone, v);
      integrate_left_edge(T, Gt, backbone, v);
    }
    U(_, _, _) = T(_, _, _);
    multiply_left_vertex_and_right_zero_vertex(T, backbone, vct0);
    backbone.reverse_hyb_line_zero();
    multiply_left_vertex_and_right_zero_vertex(U, backbone, vct0);
    backbone.reverse_hyb_line_zero();
    T = T - U;

    for (int v = vct0 + 1; v < 2 * m; v++) {
      integrate_left_edge(T, Gt, backbone, v - 1);
      multiply_left_vertex(T, backbone, v);
    }

    multiply_prefactor(T, backbone);
    int diag_order_sign = (m % 2 == 0) ? -1 : 1;
    if (backbone.get_fb(0) == 0) diag_order_sign *= -1;
    T *= diag_order_sign * backbone.prefactor_sign;
    Sigma += T;

    backbone.reset_all_inds();
  }

  void DiagramEvaluator::eval_self_energy_fixed_indices(nda::array_const_view<dcomplex, 3> Gt, Backbone &backbone, long f_ix) {
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

  void DiagramEvaluator::multiply_right_vertex(nda::array_view<dcomplex, 3> U_buf, Backbone &backbone, int v_ix) {
    int o_ix = backbone.get_vertex_orb(v_ix); // orbital index
    int l_ix = backbone.get_pole_ind(backbone.get_vertex_hyb_ind(v_ix));
    // backbone.get_vertex_hyb_ind(v_ix) = i, where i is the # of primes on l
    // l_ix = value of l with i primes
    auto F = Fset.get_operator(backbone, v_ix, o_ix, l_ix);
    for (int t = 0; t < r; t++) U_buf(t, _, _) = matmul(U_buf(t, _, _), F);
    hyb.multiply_kernel_on_vertex(U_buf, backbone, v_ix, l_ix, -1.0);
  }

  void DiagramEvaluator::integrate_right_edge(nda::array_view<dcomplex, 3> U_buf, nda::array_const_view<dcomplex, 3> Gt, Backbone &backbone,
                                                   int e_ix) {
    GKt = Gt;
    hyb.multiply_kernels_on_edge(GKt, backbone, e_ix);
    U_buf = itops.convolve(beta, itops.vals2coefs(U_buf), itops.vals2coefs(GKt), cppdlr::TIME_ORDERED);
  }

  nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(nda::array_const_view<dcomplex, 3> Gt, CorrelatorBackbone &backbone,
                                                                 nda::array<dcomplex, 3> mu_ops, nda::array<dcomplex, 3> kap_ops, bool is_fermionic) {
    long f_ix_max = backbone.num_flat_indices(hyb.poles.size());

    nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.extent(0), kap_ops.extent(0));

    // loop over all flat indices
    for (long f_ix = 0; f_ix < f_ix_max; ++f_ix) {
      correlator +=
         eval_correlator(Gt, backbone, mu_ops, kap_ops, f_ix, is_fermionic); // evaluate the diagram with these directions, poles, and orbital indices
    }

    return correlator;
  }

  nda::array<dcomplex, 3> DiagramEvaluator::eval_correlator(nda::array_const_view<dcomplex, 3> Gt, CorrelatorBackbone &backbone,
                                                            nda::array<dcomplex, 3> mu_ops, nda::array<dcomplex, 3> kap_ops, long f_ix,
                                                            bool is_fermionic) {

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

    // use special orbital index values to indicate whether the vertex connected to zero
    // is fermionic or bosonic, since this affects the fermionic parity calculation.
    // -2 for fermionic, -3 for bosonic

    int orb_idx_flag = is_fermionic ? -2 : -3;
    backbone.set_orb_inds_of_0_and_vct0(orb_idx_flag, orb_idx_flag);
    int fermionic_sign = backbone.get_parity();
    correlator *= fermionic_sign;

    /*
    std::cout << "n = " << n << ", n_int = " << n_int << std::endl;
    std::cout << "f_ix = " << f_ix << ", fermionic_sign = " << fermionic_sign << std::endl;
    //std::cout << "topology = " << backbone.get_topology() << std::endl;
    std::cout << "orb_idx_flag = " << orb_idx_flag << " is_fermionic = " << is_fermionic << std::endl;
    */

    backbone.reset_all_inds(); // reset directions, pole indices, and orbital indices for the next iteration

    return correlator;
  }

  long DiagramEvaluator::get_num_single_ptcle_gf_backbones(nda::array_const_view<int, 2> topology) {
    CorrelatorBackbone backbone(topology, n, n_int);
    return backbone.num_flat_indices(hyb.poles.size());
  }

  nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology) {
    auto Gt = native_propagator(G_ppsc);
    CorrelatorBackbone backbone(topology, n, n_int);

    // the external legs are the fermionic flavours only, not the dynamical-interaction operators appended to Fset
    auto mu_ops  = Fset.Fs(nda::range(n_hyb), _, _);
    auto kap_ops = Fset.F_dags(nda::range(n_hyb), _, _);

    /*
      auto correlator = triqs::gfs::gf<triqs::mesh::dlr_imtime>(
        tau_mesh, eval_correlator(backbone, mu_ops, kap_ops));
      return correlator;
      */

    return eval_correlator(Gt, backbone, mu_ops, kap_ops);
  }

  nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology, long f_ix) {
    auto Gt = native_propagator(G_ppsc);
    CorrelatorBackbone backbone(topology, n, n_int);
    auto mu_ops  = Fset.Fs(nda::range(n_hyb), _, _);
    auto kap_ops = Fset.F_dags(nda::range(n_hyb), _, _);

    /*
      auto correlator = triqs::gfs::gf<triqs::mesh::dlr_imtime>(
        tau_mesh, eval_correlator(backbone, mu_ops, kap_ops, f_ix));
      return correlator;
      */

    return eval_correlator(Gt, backbone, mu_ops, kap_ops, f_ix);
  }

  nda::array<dcomplex, 3> DiagramEvaluator::compute_single_ptcle_gf(gf_vt G_ppsc, nda::array_const_view<int, 2> topology,
                                                                    nda::array_const_view<long, 1> f_ix_vec) {
    auto Gt = native_propagator(G_ppsc);
    CorrelatorBackbone backbone(topology, n, n_int);
    auto mu_ops  = Fset.Fs(nda::range(n_hyb), _, _);
    auto kap_ops = Fset.F_dags(nda::range(n_hyb), _, _);

    /*
      auto correlator = triqs::gfs::gf<triqs::mesh::dlr_imtime>(
        tau_mesh, eval_correlator(backbone, mu_ops, kap_ops, f_ix));
      return correlator;
      */

    nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.extent(0), kap_ops.extent(0));

    for (auto f_ix : f_ix_vec) { correlator += eval_correlator(Gt, backbone, mu_ops, kap_ops, f_ix); }

    return correlator;
  }

  template <bool isComplex>
  nda::array<dcomplex, 3>
  DiagramEvaluator::compute_one_time_correlator(gf_vt G_ppsc, std::vector<triqs::operators::many_body_operator_real> const &ops_tau,
                                                std::vector<triqs::operators::many_body_operator_real> const &ops_0,
                                                triqs::atom_diag::atom_diag<isComplex> const &ad, nda::array_const_view<int, 2> topology,
                                                nda::array_const_view<long, 1> f_ix_vec) {

    CorrelatorBackbone backbone(topology, n, n_int);

    // The operator matrices are read from subspace 0 only
    if (ad.n_subspaces() != 1)
      throw std::invalid_argument("compute_one_time_correlator: requires an atom_diag with a single subspace, got " + std::to_string(ad.n_subspaces())
                                  + ". Build it with an empty list of conserved operators.");
    if (ad.get_full_hilbert_space_dim() != N)
      throw std::invalid_argument("compute_one_time_correlator: the atom_diag has " + std::to_string(ad.get_full_hilbert_space_dim())
                                  + " states, the evaluator was built for " + std::to_string(N));

    auto Gt = native_propagator(G_ppsc);

    auto U = ad.get_unitary_matrix(0);

    nda::array<dcomplex, 3> mu_ops  = nda::zeros<dcomplex>(ops_tau.size(), N, N);
    nda::array<dcomplex, 3> kap_ops = nda::zeros<dcomplex>(ops_0.size(), N, N);

    for (auto [i, op] : itertools::enumerate(ops_tau)) { mu_ops(i, _, _) = U * ad.get_op_mat(op).block_mat[0] * nda::conj(nda::transpose(U)); }

    for (auto [i, op] : itertools::enumerate(ops_0)) { kap_ops(i, _, _) = U * ad.get_op_mat(op).block_mat[0] * nda::conj(nda::transpose(U)); }

    nda::array<dcomplex, 3> correlator = nda::zeros<dcomplex>(r, mu_ops.extent(0), kap_ops.extent(0));

    // Figure out whether ops_tau and ops_0 are fermionic or bosonic.
    // Require that all operators have the same statistics

    // The statistics is the fermion parity of the operators, not whether they commute: a commutator test calls (S+, S-) fermionic
    // since [S+, S-] = 2 Sz != 0, see operator_statistics.hpp
    bool is_fermionic = correlator_statistics(ops_tau, ops_0, "compute_one_time_correlator");

    for (auto f_ix : f_ix_vec) { correlator += eval_correlator(Gt, backbone, mu_ops, kap_ops, f_ix, is_fermionic); }

    return correlator;
  }

  template nda::array<dcomplex, 3>
  DiagramEvaluator::compute_one_time_correlator(gf_vt G_ppsc, std::vector<triqs::operators::many_body_operator_real> const &ops_tau,
                                                std::vector<triqs::operators::many_body_operator_real> const &ops_0,
                                                triqs::atom_diag::atom_diag<false> const &ad, nda::array_const_view<int, 2> topology,
                                                nda::array_const_view<long, 1> f_ix_vec);

  template nda::array<dcomplex, 3>
  DiagramEvaluator::compute_one_time_correlator(gf_vt G_ppsc, std::vector<triqs::operators::many_body_operator_real> const &ops_tau,
                                                std::vector<triqs::operators::many_body_operator_real> const &ops_0,
                                                triqs::atom_diag::atom_diag<true> const &ad, nda::array_const_view<int, 2> topology,
                                                nda::array_const_view<long, 1> f_ix_vec);

} // namespace triqs_xca::dense