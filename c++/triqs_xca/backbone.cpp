#include <algorithm>

#include "triqs_xca/backbone.hpp"
#include "triqs_xca/topology.hpp"

namespace triqs_xca::backbone {

  using nda::range;

  BackboneVertex::BackboneVertex() : bar(false), dag(false), hyb_ind(0), Ksign(0), orb(0) { ; }

  bool BackboneVertex::has_bar() { return bar; }
  bool BackboneVertex::has_dag() { return dag; }
  int BackboneVertex::get_hyb_ind() { return hyb_ind; }
  int BackboneVertex::get_Ksign() { return Ksign; }
  int BackboneVertex::get_orb() { return orb; }

  void BackboneVertex::set_bar(bool b) { bar = b; }
  void BackboneVertex::set_dag(bool b) { dag = b; }
  void BackboneVertex::set_hyb_ind(int i) { hyb_ind = i; }
  void BackboneVertex::set_Ksign(int i) { Ksign = i; }
  void BackboneVertex::set_orb(int i) { orb = i; }

  Backbone::Backbone(nda::array<int, 2> topology, int n, int n_int)
     : m(topology.extent(0)),
       n(n),
       n_hyb(n - n_int), // number of hybridization operators is total number of operators minus number of interaction operators
       n_int(n_int),
       fb_ix_max(static_cast<int>(pow(2, m))),
       o_ix_max(static_cast<int>(pow(n, m - 1))),
       prefactor_sign(1),
       vertices(std::vector<BackboneVertex>(2 * m)),
       prefactor_Ksigns(nda::vector<int>(m - 1, 0)),
       prefactor_Kexps(nda::vector<int>(m - 1, 0)),
       edges(nda::zeros<int>(2 * m - 1, m - 1)),
       topology(topology),
       fb(nda::vector<int>(m, 0)),
       f_ix(0) {

    for (int i = 0; i < m - 1; i++) {
      if (topology(i + 1, 0) <= topology(i, 0)) { throw std::invalid_argument("first pair indices must be in increasing order"); }
    }

    for (int i = 0; i < m; i++) {
      if (topology(i, 0) >= topology(i, 1)) { throw std::invalid_argument("first row of topology must contain smaller-numbered vertices"); }
    }

    if (topology(0, 0) != 0) throw std::invalid_argument("topology(0,0) must be 0");

    // Test that sorts all indices and compares to range(0, 2*m)
    // to check that all vertices are present and there are no duplicates.
    std::vector<int> vertices_sorted(2 * m);
    for (int i = 0; i < m; i++) {
      vertices_sorted[2 * i]     = topology(i, 0);
      vertices_sorted[2 * i + 1] = topology(i, 1);
    }
    std::ranges::sort(vertices_sorted);
    for (int i = 0; i < 2 * m; i++) {
      if (vertices_sorted[i] != i) { throw std::invalid_argument("topology must contain each vertex index from 0 to 2m-1 exactly once"); }
    }
  }

  void Backbone::set_directions(nda::vector_const_view<int> fb_vec) {

    this->fb = fb_vec;
    if (m != fb_vec.size()) { throw std::invalid_argument("topology and fb must have same # of vertices"); }

    // set operator flags for each vertex, depending on fb
    vertices[0].set_bar(false);              // operator on vertex 0 has no bar
    vertices[topology(0, 1)].set_bar(false); // operator on vertex connected to 0 has no bar
    if (fb_vec(0) == 1) {
      vertices[0].set_dag(false);             // annihilation operator on vertex 0
      vertices[topology(0, 1)].set_dag(true); // creation operator on vertex connected to 0
    } else {
      vertices[0].set_dag(true);               // creation operator on vertex 0
      vertices[topology(0, 1)].set_dag(false); // annihilation operator on vertex connected to 0
    }

    for (int i = 1; i < m; i++) {
      vertices[topology(i, 0)].set_bar(false); // operator on vertex i has no bar
      vertices[topology(i, 1)].set_bar(true);  // operator on vertex connected to i has a bar
      if (fb_vec(i) == 1) {
        vertices[topology(i, 0)].set_dag(false); // annihilation operator on vertex i
        vertices[topology(i, 1)].set_dag(true);  // creation operator on vertex i
      } else {
        vertices[topology(i, 0)].set_dag(true);  // creation operator on vertex i
        vertices[topology(i, 1)].set_dag(false); // annihilation operator on vertex i
      }
    }
  }

  void Backbone::set_directions(int fb_ix) {

    auto fb_vec = nda::vector<int>(m);
    for (int i = 0; i < m; i++) {
      fb_vec(i) = fb_ix % 2; // 0 for backward, 1 for forward
      fb_ix /= 2;
    }
    set_directions(fb_vec);
  }

  void Backbone::reset_directions() {
    fb = 0;
    for (int i = 0; i < 2 * m; i++) {
      vertices[i].set_bar(false);
      vertices[i].set_dag(false);
    }
  }

  void Backbone::set_pole_inds(nda::vector_const_view<int> pole_inds_vec, nda::vector_const_view<double> hyb_poles) {

    this->pole_inds = pole_inds_vec; // values of l, l`, etc.
    for (int i = 1; i < m; i++) {
      if (fb(i) == 1) { // line i is forward
        if (hyb_poles(pole_inds(i - 1)) <= 0) {
          // step 4(a)
          // place K^-_l F_nu at the right vertex
          vertices[topology(i, 0)].set_hyb_ind(i - 1);
          vertices[topology(i, 0)].set_Ksign(-1);
          // place K^+_l F^bar^dag_nu_l at the left vertex
          vertices[topology(i, 1)].set_hyb_ind(i - 1);
          vertices[topology(i, 1)].set_Ksign(1);
          // divide by K^-_l(0)
          prefactor_Ksigns(i - 1) = -1;
          prefactor_Kexps(i - 1)  = 1;
        } else {
          // step 4(b)
          // no K's on vertices, but F bar on left vertex
          vertices[topology(i, 1)].set_hyb_ind(i - 1);
          // place K^+_l on each edge between the two vertices
          for (int j = topology(i, 0); j < topology(i, 1); j++) edges(j, i - 1) = 1;
          // divide by (K^+_l(0))^(# edges between vertices - 1)
          prefactor_Ksigns(i - 1) = 1;
          prefactor_Kexps(i - 1)  = topology(i, 1) - topology(i, 0) - 1;
        }
      } else { // line i is backward
        if (hyb_poles(pole_inds(i - 1)) >= 0) {
          // step 4(a)
          // place K^+_l F^dag_pi at the right vertex
          vertices[topology(i, 0)].set_hyb_ind(i - 1);
          vertices[topology(i, 0)].set_Ksign(1);
          // place K^-_l F^bar_pi_l at the left vertex
          vertices[topology(i, 1)].set_hyb_ind(i - 1);
          vertices[topology(i, 1)].set_Ksign(-1);
          // divide by -K^-+l(0)
          prefactor_sign *= -1;
          prefactor_Ksigns(i - 1) = 1;
          prefactor_Kexps(i - 1)  = 1;
        } else {
          // step 4(b)
          // no K's on vertices, but F bar on left vertex
          vertices[topology(i, 1)].set_hyb_ind(i - 1);
          // place K^-_l on each edge between the two vertices
          for (int j = topology(i, 0); j < topology(i, 1); j++) edges(j, i - 1) = -1;
          // divide by (-K^-_l(0))^(# edges between vertices - 1)
          prefactor_sign *= -1;
          // if ((topology(i,1) - topology(i,0) - 1) % 2 == 1) prefactor_sign *= -1;
          prefactor_Ksigns(i - 1) = -1;
          prefactor_Kexps(i - 1)  = topology(i, 1) - topology(i, 0) - 1;
        }
      }
    }
  }

  void Backbone::set_pole_inds(int p_ix, nda::vector_const_view<double> hyb_poles) {

    int p              = hyb_poles.size();
    auto pole_inds_vec = nda::vector<int>(m - 1);
    for (int i = 0; i < m - 1; i++) {
      pole_inds_vec(i) = p_ix % p;
      p_ix /= p;
    }
    set_pole_inds(pole_inds_vec, hyb_poles);
  }

  void Backbone::reset_pole_inds() {
    pole_inds        = 0;
    edges            = 0;
    prefactor_sign   = 1;
    prefactor_Ksigns = 0;
    prefactor_Kexps  = 0;
    for (int i = 0; i < 2 * m; i++) {
      vertices[i].set_hyb_ind(0);
      vertices[i].set_Ksign(0);
      vertices[i].set_orb(0);
    }
  }

  void Backbone::set_orb_inds(nda::vector_const_view<int> orb_inds_vec) {
    // orb_inds = orbital indices (e.g. lambda, mu indices), going
    // right to left, excluding the ones associated with the special vertex
    this->orb_inds = orb_inds_vec;
    for (int i = 0; i < 2 * m; i++) {
      if (i != 0 && i != topology(0, 1)) { vertices[i].set_orb(orb_inds_vec(i)); }
    }
  }

  void Backbone::set_orb_inds(int o_ix) {
    // set orbital indices from a single integer index
    auto orb_inds_vec            = nda::vector<int>(2 * m);
    orb_inds_vec(0)              = -1;
    orb_inds_vec(topology(0, 1)) = -1; // special vertex 0 and the one connected to it have no orbital indices explicitly summed over
    for (int i = 1; i < m; i++) {      // loop over lines, skipping the one connected to vertex 0
      orb_inds_vec(topology(i, 0)) = o_ix % n;
      orb_inds_vec(topology(i, 1)) = o_ix % n;
      // orbital indices on vertices connected by a line are the same
      o_ix /= n;
    }
    set_orb_inds(orb_inds_vec);
  }

  void Backbone::reset_orb_inds() {
    for (int i = 0; i < 2 * m; i++) vertices[i].set_orb(0);
  }

  void Backbone::set_orb_inds_of_0_and_vct0(int orb_ind_0, int orb_ind_vct0) {
    // set the orbital indices of vertex 0 and the vertex connected to it, which are not included in orb_inds
    vertices[0].set_orb(orb_ind_0);
    vertices[topology(0, 1)].set_orb(orb_ind_vct0);
  }

  int Backbone::get_parity() {

    // get the fermionic permutation parity of the diagram represented by this backbone,
    // based on the topology and orbital indices

    assert(vertices[0].get_orb() != -1);              // make sure orbital index of vertex 0 is set
    assert(vertices[topology(0, 1)].get_orb() != -1); // make sure orbital index of vertex connected to 0 is set

    // loop over orbital indices and check which ones are fermionic by checking i < n_hyb
    // put the result in a boolean vector of size m, where true means the line is fermionic
    nda::vector<bool> is_fermionic(2 * m);

    for (auto vertex_idx : range(0, 2 * m)) {
      int orb_ind = vertices[vertex_idx].get_orb();

      if (orb_ind >= 0) {
        is_fermionic[vertex_idx] = orb_ind < n_hyb; // fermionic vertex when orbital index is less than n_hyb
      } else if (orb_ind == -2) {
        is_fermionic[vertex_idx] = true; // if orbital index is -2, it's a fermionic vertex
      } else if (orb_ind == -3) {
        is_fermionic[vertex_idx] = false; // if orbital index is -3, it's a bosonic vertex
      } else {
        throw std::invalid_argument("Orbital index of vertex " + std::to_string(vertex_idx) + " is not set");
      }
    }
    auto fermionic_topology = topology::fermionic_topology(topology, is_fermionic);
    int fermionic_parity    = topology::topology_parity(fermionic_topology);

    return fermionic_parity;
  }

  void Backbone::set_flat_index(int flat_ix, nda::vector_const_view<double> hyb_poles) {
    // set directions, pole indices, and orbital indices from a single integer index.
    // In terms of fb_ix, p_ix, and o_ix,
    // f_ix = o_ix + n^(m-1) * p_ix + (n * r)^(m-1) * fb_ix, where r is the number of hybridization indices.

    this->f_ix = flat_ix;
    int p      = hyb_poles.size();
    int rem    = flat_ix;
    int o_ix   = rem % o_ix_max; // orbital indices
    rem /= o_ix_max;
    int p_ix_max = static_cast<int>(pow(p, m - 1));
    int p_ix     = rem % p_ix_max; // pole indices
    int fb_ix    = rem / p_ix_max; // directions

    set_directions(fb_ix);
    set_pole_inds(p_ix, hyb_poles);
    set_orb_inds(o_ix);
  }

  void Backbone::reverse_hyb_line_zero() {
    fb(0)    = 1 - fb(0);
    int vct0 = topology(0, 1);
    vertices[0].set_dag(!vertices[0].has_dag());
    vertices[vct0].set_dag(!vertices[vct0].has_dag());
  }

  void Backbone::reset_all_inds() {
    reset_directions();
    reset_pole_inds();
    reset_orb_inds();
  }

  int Backbone::get_prefactor_Ksign(int i) { return prefactor_Ksigns(i); }
  int Backbone::get_prefactor_Kexp(int i) { return prefactor_Kexps(i); }
  bool Backbone::has_vertex_bar(int i) { return vertices[i].has_bar(); }
  bool Backbone::has_vertex_dag(int i) { return vertices[i].has_dag(); }
  int Backbone::get_vertex_hyb_ind(int i) { return vertices[i].get_hyb_ind(); }
  int Backbone::get_vertex_Ksign(int i) { return vertices[i].get_Ksign(); }
  int Backbone::get_vertex_orb(int i) { return vertices[i].get_orb(); }
  int Backbone::get_edge(int num, int pole_ind) { return edges(num, pole_ind); }
  int Backbone::get_topology(int i, int j) { return topology(i, j); }
  int Backbone::get_pole_ind(int i) { return pole_inds(i); }
  int Backbone::get_fb(int i) { return fb(i); }
  int Backbone::get_orb_ind(int i) { return orb_inds(i); }
  int Backbone::get_flat_index() { return f_ix; }

  CorrelatorBackbone::CorrelatorBackbone(nda::array<int, 2> topology, int n) : Backbone(topology, n) { fb_ix_max = static_cast<int>(pow(2, m - 1)); }

  void CorrelatorBackbone::set_directions(int fb_ix) {

    auto fb_vec = nda::vector<int>(m - 1);
    for (int i = 0; i < m - 1; i++) {
      fb_vec(i) = fb_ix % 2; // 0 for backward, 1 for forward
      fb_ix /= 2;
    }
    set_directions(fb_vec);
  }

  void CorrelatorBackbone::set_directions(nda::vector_const_view<int> fb_vec) {
    // @param[in] fb forward/backward line information
    // same logic as the Backbone method, but no line connected to vertex 0 -- just the "for loop" part

    if (m - 1 != fb_vec.size()) { throw std::invalid_argument("fb must have m - 1 elements"); }
    this->fb(nda::range(1, m)) = fb_vec;
    this->fb(0)                = -1; // no line connected to vertex 0

    vertices[topology(0, 0)].set_bar(false); // operator on vertex 0 has no bar
    vertices[topology(0, 1)].set_bar(false); // operator on vertex connected to 0 has no bar
    vertices[topology(0, 0)].set_dag(true);  // creation operator on vertex 0
    vertices[topology(0, 1)].set_dag(false); // annihilation operator on vertex connected to 0
    for (int i = 1; i < m; i++) {
      vertices[topology(i, 0)].set_bar(false); // operator on vertex i has no bar
      vertices[topology(i, 1)].set_bar(true);  // operator on vertex connected to i has a bar
      if (fb(i) == 1) {
        vertices[topology(i, 0)].set_dag(false); // annihilation operator on vertex i
        vertices[topology(i, 1)].set_dag(true);  // creation operator on vertex i
      } else {
        vertices[topology(i, 0)].set_dag(true);  // creation operator on vertex i
        vertices[topology(i, 1)].set_dag(false); // annihilation operator on vertex i
      }
    }
  }

  void CorrelatorBackbone::set_flat_index(int flat_ix, nda::vector_const_view<double> hyb_poles) {
    // set directions, pole indices, and orbital indices from a single integer index.
    // In terms of fb_ix, p_ix, and o_ix,
    // f_ix = o_ix + n^(m-1) * p_ix + (n * r)^(m-1) * fb_ix, where r is the number of hybridization indices.

    this->f_ix = flat_ix;
    int p      = hyb_poles.size();
    int rem    = flat_ix;
    int o_ix   = rem % o_ix_max; // orbital indices
    rem /= o_ix_max;
    int p_ix_max = static_cast<int>(pow(p, m - 1));
    int p_ix     = rem % p_ix_max; // pole indices
    int fb_ix    = rem / p_ix_max; // directions

    set_directions(fb_ix);
    set_pole_inds(p_ix, hyb_poles);
    set_orb_inds(o_ix);
  }

  std::ostream &operator<<(std::ostream &os, Backbone &B) {

    std::string p_str = "1 / (";
    int sign          = B.prefactor_sign;
    if (sign == -1) p_str = "-" + p_str;
    for (int i = 0; i < B.m - 1; i++) {
      if (B.get_prefactor_Kexp(i) >= 1) {
        p_str += "K_{l";
        for (int j = 0; j < i; j++) p_str += "`";
        p_str += "}";
        if (B.get_prefactor_Ksign(i) == 1)
          p_str += "^+";
        else
          p_str += "^-";
      }
      if (B.get_prefactor_Kexp(i) > 1) { p_str += "^" + std::to_string(B.get_prefactor_Kexp(i)); }
      p_str += "(0)";
      if (i < B.m - 2) p_str += " ";
    }
    p_str += ")";

    int diag_str_cent    = 30;
    std::string diag_str = "";
    for (int i = 0; i < diag_str_cent; i++) diag_str += " ";
    diag_str += "0\n";
    std::string v_str_tmp = "", e_str_tmp = "";
    for (int i = 0; i < 2 * B.m; i++) {
      // K factor
      if (B.get_vertex_Ksign(i) != 0) {
        v_str_tmp += "K_{l";
        for (int j = 0; j < B.get_vertex_hyb_ind(i); j++) v_str_tmp += "`";
        v_str_tmp += "}";
        if (B.get_vertex_Ksign(i) == 1)
          v_str_tmp += "^+";
        else
          v_str_tmp += "^-";
      }

      // F operator
      v_str_tmp += "F";
      if (B.has_vertex_bar(i) || B.has_vertex_dag(i)) v_str_tmp += "^{";
      if (B.has_vertex_bar(i)) v_str_tmp += "bar";
      if (B.has_vertex_dag(i)) v_str_tmp += "dag";
      if (B.has_vertex_bar(i) || B.has_vertex_dag(i)) v_str_tmp += "}";
      if (not B.has_vertex_bar(i))
        v_str_tmp += "_" + std::to_string(i);
      else {
        for (int j = 0; j < B.m; j++)
          if (B.get_topology(j, 1) == i) v_str_tmp += "_" + std::to_string(B.get_topology(j, 0));
        v_str_tmp += "l";
        for (int j = 0; j < B.get_vertex_hyb_ind(i); j++) v_str_tmp += "`";
      }
      // hybridization
      if (i == B.get_topology(0, 1)) {
        v_str_tmp += " Delta_{";
        if (B.has_vertex_dag(i) == 1)
          v_str_tmp += "0," + std::to_string(i) + "} ";
        else
          v_str_tmp += std::to_string(i) + ",0}";
      } else
        v_str_tmp += " ";
      int vlen0 = v_str_tmp.size();
      for (int j = vlen0; j < diag_str_cent - 2; j++) v_str_tmp = " " + v_str_tmp;
      diag_str += v_str_tmp + "--| \n";
      v_str_tmp = "";

      // edges --> e_str
      if (i < 2 * B.m - 1) {
        for (int j = 0; j < B.m - 1; j++) {
          if (B.get_edge(i, j) != 0) {
            e_str_tmp += "K_{l";
            for (int k = 0; k < j; k++) e_str_tmp += "`";
            e_str_tmp += "}";
            if (B.get_edge(i, j) == 1)
              e_str_tmp += "^+ ";
            else
              e_str_tmp += "^- ";
          }
        }
        e_str_tmp += "G ";
        for (int j = 0; j < diag_str_cent; j++) diag_str += " ";
        diag_str += "| " + e_str_tmp + "\n";
        e_str_tmp = "";
      }
    }
    for (int i = 0; i < diag_str_cent - 1; i++) diag_str += " ";
    diag_str += "tau";

    os << "\nPrefactor: " << p_str << "\nDiagram: \n" << diag_str;
    return os;
  }

  std::ostream &operator<<(std::ostream &os, CorrelatorBackbone &B) {

    std::string p_str = "1 / (";
    int sign          = B.prefactor_sign;
    if (sign == -1) p_str = "-" + p_str;
    for (int i = 0; i < B.m - 1; i++) {
      if (B.get_prefactor_Kexp(i) >= 1) {
        p_str += "K_{l";
        for (int j = 0; j < i; j++) p_str += "`";
        p_str += "}";
        if (B.get_prefactor_Ksign(i) == 1)
          p_str += "^+";
        else
          p_str += "^-";
      }
      if (B.get_prefactor_Kexp(i) > 1) { p_str += "^" + std::to_string(B.get_prefactor_Kexp(i)); }
      p_str += "(0)";
      if (i < B.m - 2) p_str += " ";
    }
    p_str += ")";

    int diag_str_cent    = 30;
    std::string diag_str = "";
    for (int i = 0; i < diag_str_cent; i++) diag_str += " ";
    diag_str += "0\n";
    std::string v_str_tmp = "", e_str_tmp = "";
    for (int i = 0; i < 2 * B.m; i++) {
      // K factor
      if (B.get_vertex_Ksign(i) != 0) {
        v_str_tmp += "K_{l";
        for (int j = 0; j < B.get_vertex_hyb_ind(i); j++) v_str_tmp += "`";
        v_str_tmp += "}";
        if (B.get_vertex_Ksign(i) == 1)
          v_str_tmp += "^+";
        else
          v_str_tmp += "^-";
      }

      // F operator
      v_str_tmp += "F";
      if (B.has_vertex_bar(i) || B.has_vertex_dag(i)) v_str_tmp += "^{";
      if (B.has_vertex_bar(i)) v_str_tmp += "bar";
      if (B.has_vertex_dag(i)) v_str_tmp += "dag";
      if (B.has_vertex_bar(i) || B.has_vertex_dag(i)) v_str_tmp += "}";
      if (not B.has_vertex_bar(i))
        v_str_tmp += "_" + std::to_string(i);
      else {
        for (int j = 0; j < B.m; j++)
          if (B.get_topology(j, 1) == i) v_str_tmp += "_" + std::to_string(B.get_topology(j, 0));
        v_str_tmp += "l";
        for (int j = 0; j < B.get_vertex_hyb_ind(i); j++) v_str_tmp += "`";
      }

      v_str_tmp += " ";
      int vlen0 = v_str_tmp.size();
      for (int j = vlen0; j < diag_str_cent - 2; j++) v_str_tmp = " " + v_str_tmp;
      diag_str += v_str_tmp + "--| \n";
      v_str_tmp = "";

      // edges --> e_str
      if (i < 2 * B.m - 1) {
        for (int j = 0; j < B.m - 1; j++) {
          if (B.get_edge(i, j) != 0) {
            e_str_tmp += "K_{l";
            for (int k = 0; k < j; k++) e_str_tmp += "`";
            e_str_tmp += "}";
            if (B.get_edge(i, j) == 1)
              e_str_tmp += "^+ ";
            else
              e_str_tmp += "^- ";
          }
        }
        e_str_tmp += "G ";
        for (int j = 0; j < diag_str_cent; j++) diag_str += " ";
        diag_str += "| " + e_str_tmp + "\n";
        e_str_tmp = "";
      }
    }
    for (int i = 0; i < diag_str_cent; i++) diag_str += " ";
    diag_str += "| G\n";
    for (int i = 0; i < diag_str_cent - 1; i++) diag_str += " ";
    diag_str += "beta";

    os << "\nPrefactor: " << p_str << "\nDiagram: \n" << diag_str;
    return os;
  }

} // namespace triqs_xca::backbone