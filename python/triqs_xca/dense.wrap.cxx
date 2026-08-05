
// C.f. https://numpy.org/doc/1.21/reference/c-api/array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL _cpp2py_ARRAY_API
#ifndef CLAIR_C2PY_WRAP_GEN
#ifdef __clang__
// #pragma clang diagnostic ignored "-W#warnings"
#endif
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wcpp"
#endif

#define C2PY_VERSION_MAJOR 0
#define C2PY_VERSION_MINOR 1

#include <c2py/c2py.hpp>

using c2py::operator""_a;

// ==================== enums =====================

// ==================== module classes =====================

// --------- class _c2py_cls_0 -----------
using _c2py_cls_0                                            = triqs_xca::dense::DenseDiagramEvaluator;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_0>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_0> = "triqs_xca.dense.DenseDiagramEvaluator";
static auto _c2py_init_0                                     = c2py::dispatcher_c_kw_t{
   c2py::c_constructor<
                                          _c2py_cls_0,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad"),
   c2py::c_constructor<
                                          _c2py_cls_0,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad"),
   c2py::c_constructor<
                                          _c2py_cls_0,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<1> &, const std::vector<triqs::operators::many_body_operator_real> &,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad", "dynint_ops", "dynint_coeffs"),
   c2py::c_constructor<
                                          _c2py_cls_0,
                                          nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>,
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &, const std::vector<triqs::operators::many_body_operator_real> &,
                                          nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                nda::borrowed<nda::mem::AddressSpace::Host>>>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad", "dynint_ops",
                                                                                                              "dynint_coeffs")};
template <> constexpr initproc c2py::tp_init<_c2py_cls_0> = c2py::pyfkw_constructor<_c2py_init_0>;
template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_0> = _c2py_init_0.doc(
   R"DOC(
Constructor for DiagramEvaluator

Parameters
----------
hyb_poles : {par_0}
   hybridization poles
hyb_coeffs : {par_1}
   hybridization function coefficients (at poles)
tau_mesh : {par_2}
   TRIQS imagnary time DLR mesh
ad : {par_3}
   TRIQS atom_diag object with Hamiltonian and field operators
dynint_ops : {par_4}
   vector of many_body_operator_real objects representing the dynamic interactions
dynint_coeffs : {par_5}
   array of coefficients for the dynamic interactions (also using hyb_poles)
)DOC",
   {{c2py::python_typename<
       nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<triqs::mesh::dlr_imtime>()},
    {c2py::python_typename<const triqs::atom_diag::atom_diag<1> &>(), c2py::python_typename<const triqs::atom_diag::atom_diag<0> &>()},
    {c2py::python_typename<const std::vector<triqs::operators::many_body_operator_real> &>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()}});
// compute_one_time_correlator
static auto const _c2py_fun_0 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::DenseDiagramEvaluator::gf_vt G_ppsc,
         const std::vector<triqs::operators::many_body_operator_real> &ops_tau, const std::vector<triqs::operators::many_body_operator_real> &ops_0,
         const triqs::atom_diag::atom_diag<1> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const int, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> f_ix_vec)
         -> decltype(auto) { return self.template compute_one_time_correlator<true>(G_ppsc, ops_tau, ops_0, ad, topology, f_ix_vec); },
      "self", "G_ppsc", "ops_tau", "ops_0", "ad", "topology", "f_ix_vec"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::DenseDiagramEvaluator::gf_vt G_ppsc,
         const std::vector<triqs::operators::many_body_operator_real> &ops_tau, const std::vector<triqs::operators::many_body_operator_real> &ops_0,
         const triqs::atom_diag::atom_diag<0> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const int, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> f_ix_vec)
         -> decltype(auto) { return self.template compute_one_time_correlator<false>(G_ppsc, ops_tau, ops_0, ad, topology, f_ix_vec); },
      "self", "G_ppsc", "ops_tau", "ops_0", "ad", "topology", "f_ix_vec")};

// compute_self_energy
static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::DenseDiagramEvaluator::gf_vt G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.compute_self_energy(G_ppsc, topology); },
      "self", "G_ppsc", "topology"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::DenseDiagramEvaluator::gf_vt G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         int f_ix) -> decltype(auto) { return self.compute_self_energy(G_ppsc, topology, f_ix); },
      "self", "G_ppsc", "topology", "f_ix"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::DenseDiagramEvaluator::gf_vt G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const int, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> f_ix_vec)
         -> decltype(auto) { return self.compute_self_energy(G_ppsc, topology, f_ix_vec); },
      "self", "G_ppsc", "topology", "f_ix_vec")};

// compute_self_energy_by_pairs
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::DenseDiagramEvaluator::gf_vt G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.compute_self_energy_by_pairs(G_ppsc, topology); },
      "self", "G_ppsc", "topology"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::DenseDiagramEvaluator::gf_vt G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         int f_ix) -> decltype(auto) { return self.compute_self_energy_by_pairs(G_ppsc, topology, f_ix); },
      "self", "G_ppsc", "topology", "f_ix"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::DenseDiagramEvaluator::gf_vt G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const int, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> f_ix_vec)
         -> decltype(auto) { return self.compute_self_energy_by_pairs(G_ppsc, topology, f_ix_vec); },
      "self", "G_ppsc", "topology", "f_ix_vec")};

// compute_single_ptcle_gf
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::DenseDiagramEvaluator::gf_vt G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.compute_single_ptcle_gf(G_ppsc, topology); },
      "self", "G_ppsc", "topology"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::DenseDiagramEvaluator::gf_vt G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         int f_ix) -> decltype(auto) { return self.compute_single_ptcle_gf(G_ppsc, topology, f_ix); },
      "self", "G_ppsc", "topology", "f_ix"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs_xca::dense::DenseDiagramEvaluator::gf_vt G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const int, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> f_ix_vec)
         -> decltype(auto) { return self.compute_single_ptcle_gf(G_ppsc, topology, f_ix_vec); },
      "self", "G_ppsc", "topology", "f_ix_vec")};

// get_num_self_energy_backbones
static auto const _c2py_fun_4 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.get_num_self_energy_backbones(topology); },
   "self", "topology")};

// get_num_single_ptcle_gf_backbones
static auto const _c2py_fun_5 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.get_num_single_ptcle_gf_backbones(topology); },
   "self", "topology")};

// reset
static auto const _c2py_fun_6 = c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 &self) -> decltype(auto) { return self.reset(); }, "self")};

static const auto _c2py_doc_0 = _c2py_fun_0.doc(R"DOC()DOC");
static const auto _c2py_doc_1 = _c2py_fun_1.doc(R"DOC()DOC");
static const auto _c2py_doc_2 = _c2py_fun_2.doc(R"DOC()DOC");
static const auto _c2py_doc_3 = _c2py_fun_3.doc(R"DOC()DOC");
static const auto _c2py_doc_4 = _c2py_fun_4.doc(R"DOC()DOC");
static const auto _c2py_doc_5 = _c2py_fun_5.doc(R"DOC()DOC");
static const auto _c2py_doc_6 = _c2py_fun_6.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_0>[] = {
   {"compute_one_time_correlator", (PyCFunction)c2py::pyfkw<_c2py_fun_0>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_0.c_str()},
   {"compute_self_energy", (PyCFunction)c2py::pyfkw<_c2py_fun_1>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
   {"compute_self_energy_by_pairs", (PyCFunction)c2py::pyfkw<_c2py_fun_2>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
   {"compute_single_ptcle_gf", (PyCFunction)c2py::pyfkw<_c2py_fun_3>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_3.c_str()},
   {"get_num_self_energy_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_4>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_4.c_str()},
   {"get_num_single_ptcle_gf_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_5>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_5.c_str()},
   {"reset", (PyCFunction)c2py::pyfkw<_c2py_fun_6>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_6.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

constexpr auto _c2py_doc_member_0  = R"DOC()DOC";
constexpr auto _c2py_doc_member_1  = R"DOC()DOC";
constexpr auto _c2py_doc_member_2  = R"DOC()DOC";
constexpr auto _c2py_doc_member_3  = R"DOC()DOC";
constexpr auto _c2py_doc_member_4  = R"DOC()DOC";
constexpr auto _c2py_doc_member_5  = R"DOC()DOC";
constexpr auto _c2py_doc_member_6  = R"DOC()DOC";
constexpr auto _c2py_doc_member_7  = R"DOC()DOC";
constexpr auto _c2py_doc_member_8  = R"DOC()DOC";
constexpr auto _c2py_doc_member_9  = R"DOC()DOC";
constexpr auto _c2py_doc_member_10 = R"DOC()DOC";
constexpr auto _c2py_doc_member_11 = R"DOC()DOC";
constexpr auto _c2py_doc_member_12 = R"DOC()DOC";
constexpr auto _c2py_doc_member_13 = R"DOC()DOC";
constexpr auto _c2py_doc_member_14 = R"DOC()DOC";

// ----- Member and property table ----

template <>
constinit PyGetSetDef c2py::tp_getset<_c2py_cls_0>[] = {
   c2py::getsetdef_from_member<&_c2py_cls_0::tau_mesh, _c2py_cls_0>("tau_mesh", _c2py_doc_member_0),
   c2py::getsetdef_from_member<&_c2py_cls_0::beta, _c2py_cls_0>("beta", _c2py_doc_member_1),
   c2py::getsetdef_from_member<&_c2py_cls_0::itops, _c2py_cls_0>("itops", _c2py_doc_member_2),
   c2py::getsetdef_from_member<&_c2py_cls_0::dlr_it, _c2py_cls_0>("dlr_it", _c2py_doc_member_3),
   c2py::getsetdef_from_member<&_c2py_cls_0::r, _c2py_cls_0>("r", _c2py_doc_member_4),
   c2py::getsetdef_from_member<&_c2py_cls_0::n, _c2py_cls_0>("n", _c2py_doc_member_5),
   c2py::getsetdef_from_member<&_c2py_cls_0::n_hyb, _c2py_cls_0>("n_hyb", _c2py_doc_member_6),
   c2py::getsetdef_from_member<&_c2py_cls_0::n_int, _c2py_cls_0>("n_int", _c2py_doc_member_7),
   c2py::getsetdef_from_member<&_c2py_cls_0::N, _c2py_cls_0>("N", _c2py_doc_member_8),
   c2py::getsetdef_from_member<&_c2py_cls_0::Sigma, _c2py_cls_0>("Sigma", _c2py_doc_member_9),
   c2py::getsetdef_from_member<&_c2py_cls_0::T, _c2py_cls_0>("T", _c2py_doc_member_10),
   c2py::getsetdef_from_member<&_c2py_cls_0::U, _c2py_cls_0>("U", _c2py_doc_member_11),
   c2py::getsetdef_from_member<&_c2py_cls_0::GKt, _c2py_cls_0>("GKt", _c2py_doc_member_12),
   c2py::getsetdef_from_member<&_c2py_cls_0::Tkaps, _c2py_cls_0>("Tkaps", _c2py_doc_member_13),
   c2py::getsetdef_from_member<&_c2py_cls_0::Tmu, _c2py_cls_0>("Tmu", _c2py_doc_member_14),

   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<_c2py_cls_0> = R"DOC(Class for evaluating a diagram of a given order and topology
This class is used to evaluate all the backbone decompositions of a given
order and topology. It reads the information from a Backbone object and
contains the Green's functions and creation/annihilation operators needed to
actually compute the diagram. It also contains temporary arrays required for
computation.)DOC"
   + std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<_c2py_cls_0>;

// ==================== module functions ====================

// NCA_dense
static auto const _c2py_fun_7 =
   c2py::dispatcher_f_kw_t{c2py::cfun([](nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                               nda::borrowed<nda::mem::AddressSpace::Host>>
                                            hyb,
                                         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                               nda::borrowed<nda::mem::AddressSpace::Host>>
                                            hyb_refl,
                                         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                               nda::borrowed<nda::mem::AddressSpace::Host>>
                                            Gt,
                                         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                               nda::borrowed<nda::mem::AddressSpace::Host>>
                                            Fs,
                                         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                               nda::borrowed<nda::mem::AddressSpace::Host>>
                                            F_dags) { return triqs_xca::block_sparse::NCA_dense(hyb, hyb_refl, Gt, Fs, F_dags); },
                                      "hyb", "hyb_refl", "Gt", "Fs", "F_dags")};

// OCA_dense
static auto const _c2py_fun_8 = c2py::dispatcher_f_kw_t{
   c2py::cfun([](nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                       nda::borrowed<nda::mem::AddressSpace::Host>>
                    hyb,
                 cppdlr::imtime_ops itops, double beta,
                 nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                       nda::borrowed<nda::mem::AddressSpace::Host>>
                    Gt,
                 nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                       nda::borrowed<nda::mem::AddressSpace::Host>>
                    Fs,
                 nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                       nda::borrowed<nda::mem::AddressSpace::Host>>
                    F_dags) { return triqs_xca::block_sparse::OCA_dense(hyb, itops, beta, Gt, Fs, F_dags); },
              "hyb", "itops", "beta", "Gt", "Fs", "F_dags"),
   c2py::cfun(
      [](nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            hyb,
         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            hyb_coeffs,
         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            hyb_refl,
         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            hyb_refl_coeffs,
         nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>
            hyb_poles,
         cppdlr::imtime_ops &itops, double beta,
         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            Gt,
         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            Fs,
         nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                               nda::borrowed<nda::mem::AddressSpace::Host>>
            F_dags) {
        return triqs_xca::block_sparse::OCA_dense(hyb, hyb_coeffs, hyb_refl, hyb_refl_coeffs, hyb_poles, itops, beta, Gt, Fs, F_dags);
      },
      "hyb", "hyb_coeffs", "hyb_refl", "hyb_refl_coeffs", "hyb_poles", "itops", "beta", "Gt", "Fs", "F_dags")};

static const auto _c2py_doc_7 =
   _c2py_fun_7.doc(R"DOC(
Evaluate NCA self-energy term using dense storage

Parameters
----------
hyb : {par_0}
   hybridization function
hyb_refl : {par_1}
   hyb evaluated at (beta - tau)
Gt : {par_2}
   Greens function
Fs : {par_3}
   vector of annihilation operators
F_dags : {par_4}
   vector of creation operators
)DOC",
                   {{c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
                    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
                    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
                    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
                    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()}});
static const auto _c2py_doc_8 = _c2py_fun_8.doc(
   R"DOC(
[1] Evaluate OCA using dense storage

------

[2] Evaluate OCA using dense storage and allow user to provide hybridization poles and coefficients

------

Parameters
----------
hyb : {par_0}
   hybridization function at imaginary time nodes
itops : {par_1}
   cppdlr imaginary time object
beta : {par_2}
   inverse temperature
Gt : {par_3}
   Greens function
Fs : {par_4}
   F operator
hyb_coeffs : {par_5}
   hybridization coefficients
hyb_refl : {par_6}
   hybridization function eval'd at negative imag. times
hyb_refl_coeffs : {par_7}
   hybridization coefficients at negative imag. times
hyb_poles : {par_8}
   hybridization poles

Returns
-------
{ret_0}
   OCA term of self-energy
)DOC",
   {{c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<cppdlr::imtime_ops>(), c2py::python_typename<cppdlr::imtime_ops &>()},
    {c2py::python_typename<double>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<
       nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>>()}},
   {c2py::python_typename<
      nda::basic_array<std::complex<double>, 3, nda::C_layout, 'A', nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>>()});
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {"NCA_dense", (PyCFunction)c2py::pyfkw<_c2py_fun_7>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_7.c_str()},
   {"OCA_dense", (PyCFunction)c2py::pyfkw<_c2py_fun_8>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_8.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT,
                                        "dense",           /* name of module */
                                        R"RAWDOC()RAWDOC", /* module documentation, may be NULL */
                                        -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
                                        module_methods,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *PyInit_dense() {

  if (not c2py::check_python_version("dense")) return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_0>) < 0) return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] = &c2py::wrap_pytype<c2py::py_range>;
#define _add_type(T, N) c2py::add_type_object_to_main<T>(N, m, conv_table)
  _add_type(_c2py_cls_0, "DenseDiagramEvaluator");
#undef _add_type

  return m;
}
#endif
// CLAIR_WRAP_GEN
