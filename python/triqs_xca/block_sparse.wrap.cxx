
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
using _c2py_cls_0                                            = triqs_xca::block_sparse::DiagramEvaluator;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_0>     = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_0> = "triqs_xca.block_sparse.DiagramEvaluator";
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
                                          triqs::mesh::dlr_imtime, const triqs::atom_diag::atom_diag<0> &>("hyb_poles", "hyb_coeffs", "tau_mesh", "ad")};
template <> constexpr initproc c2py::tp_init<_c2py_cls_0> = c2py::pyfkw_constructor<_c2py_init_0>;
template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_0> = _c2py_init_0.doc(
   R"DOC(
Constructor for DiagramEvaluator

Parameters
----------
beta : {par_0}
   inverse temperature
Lambda : {par_1}
   DLR imaginary time cutoff
eps : {par_2}
   DLR imaginary time accuracy
hyb_poles : {par_3}
   hybridization poles
hyb_coeffs : {par_4}
   hybridization function coefficients at imaginary time nodes
G_ppsc : {par_5}
   pseudo-particle Green's function at imaginary time nodes
ad : {par_6}
   atom_diag object with Hamiltonian and field operators
)DOC",
   {{},
    {},
    {},
    {c2py::python_typename<
       nda::basic_array_view<const double, 1, nda::C_stride_layout, 'V', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {c2py::python_typename<nda::basic_array_view<const std::complex<double>, 3, nda::C_stride_layout, 'A', nda::default_accessor,
                                                 nda::borrowed<nda::mem::AddressSpace::Host>>>()},
    {},
    {c2py::python_typename<const triqs::atom_diag::atom_diag<1> &>(), c2py::python_typename<const triqs::atom_diag::atom_diag<0> &>()}});
// compute_one_time_correlator
static auto const _c2py_fun_0 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         const std::vector<triqs::operators::many_body_operator_real> &ops_tau, const std::vector<triqs::operators::many_body_operator_real> &ops_0,
         const triqs::atom_diag::atom_diag<1> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const int, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> f_ix_vec)
         -> decltype(auto) { return self.template compute_one_time_correlator<true>(G_ppsc, ops_tau, ops_0, ad, topology, f_ix_vec); },
      "self", "G_ppsc", "ops_tau", "ops_0", "ad", "topology", "f_ix_vec"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         const std::vector<triqs::operators::many_body_operator_real> &ops_tau, const std::vector<triqs::operators::many_body_operator_real> &ops_0,
         const triqs::atom_diag::atom_diag<0> &ad,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const int, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> f_ix_vec)
         -> decltype(auto) { return self.template compute_one_time_correlator<false>(G_ppsc, ops_tau, ops_0, ad, topology, f_ix_vec); },
      "self", "G_ppsc", "ops_tau", "ops_0", "ad", "topology", "f_ix_vec")};

// compute_self_energy
static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.compute_self_energy(G_ppsc, topology); },
      "self", "G_ppsc", "topology"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         int f_ix) -> decltype(auto) { return self.compute_self_energy(G_ppsc, topology, f_ix); },
      "self", "G_ppsc", "topology", "f_ix"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const int, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> f_ix_vec)
         -> decltype(auto) { return self.compute_self_energy(G_ppsc, topology, f_ix_vec); },
      "self", "G_ppsc", "topology", "f_ix_vec")};

// compute_single_ptcle_gf
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
         -> decltype(auto) { return self.compute_single_ptcle_gf(G_ppsc, topology); },
      "self", "G_ppsc", "topology"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         int f_ix) -> decltype(auto) { return self.compute_single_ptcle_gf(G_ppsc, topology, f_ix); },
      "self", "G_ppsc", "topology", "f_ix"),
   c2py::cmethod(
      [](_c2py_cls_0 &self, triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc,
         nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
         nda::basic_array_view<const int, 1, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> f_ix_vec)
         -> decltype(auto) { return self.compute_single_ptcle_gf(G_ppsc, topology, f_ix_vec); },
      "self", "G_ppsc", "topology", "f_ix_vec")};

// get_num_self_energy_backbones
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.get_num_self_energy_backbones(topology); },
   "self", "topology")};

// get_num_single_ptcle_gf_backbones
static auto const _c2py_fun_4 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology)
      -> decltype(auto) { return self.get_num_single_ptcle_gf_backbones(topology); },
   "self", "topology")};

// print_self_energy_backbone
static auto const _c2py_fun_5 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
      int f_ix) -> decltype(auto) { return self.print_self_energy_backbone(topology, f_ix); },
   "self", "topology", "f_ix")};

// print_single_ptcle_gf_backbone
static auto const _c2py_fun_6 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](_c2py_cls_0 &self,
      nda::basic_array_view<const int, 2, nda::C_stride_layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::AddressSpace::Host>> topology,
      int f_ix) -> decltype(auto) { return self.print_single_ptcle_gf_backbone(topology, f_ix); },
   "self", "topology", "f_ix")};

// reset
static auto const _c2py_fun_7 = c2py::dispatcher_f_kw_t{c2py::cmethod([](_c2py_cls_0 &self) -> decltype(auto) { return self.reset(); }, "self")};

static const auto _c2py_doc_0 = _c2py_fun_0.doc(R"DOC()DOC");
static const auto _c2py_doc_1 = _c2py_fun_1.doc(R"DOC()DOC");
static const auto _c2py_doc_2 = _c2py_fun_2.doc(R"DOC()DOC");
static const auto _c2py_doc_3 = _c2py_fun_3.doc(R"DOC()DOC");
static const auto _c2py_doc_4 = _c2py_fun_4.doc(R"DOC()DOC");
static const auto _c2py_doc_5 = _c2py_fun_5.doc(R"DOC()DOC");
static const auto _c2py_doc_6 = _c2py_fun_6.doc(R"DOC()DOC");
static const auto _c2py_doc_7 = _c2py_fun_7.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_0>[] = {
   {"compute_one_time_correlator", (PyCFunction)c2py::pyfkw<_c2py_fun_0>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_0.c_str()},
   {"compute_self_energy", (PyCFunction)c2py::pyfkw<_c2py_fun_1>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
   {"compute_single_ptcle_gf", (PyCFunction)c2py::pyfkw<_c2py_fun_2>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
   {"get_num_self_energy_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_3>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_3.c_str()},
   {"get_num_single_ptcle_gf_backbones", (PyCFunction)c2py::pyfkw<_c2py_fun_4>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_4.c_str()},
   {"print_self_energy_backbone", (PyCFunction)c2py::pyfkw<_c2py_fun_5>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_5.c_str()},
   {"print_single_ptcle_gf_backbone", (PyCFunction)c2py::pyfkw<_c2py_fun_6>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_6.c_str()},
   {"reset", (PyCFunction)c2py::pyfkw<_c2py_fun_7>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_7.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

constexpr auto _c2py_doc_member_0 = R"DOC()DOC";
constexpr auto _c2py_doc_member_1 = R"DOC()DOC";
constexpr auto _c2py_doc_member_2 = R"DOC()DOC";
constexpr auto _c2py_doc_member_3 = R"DOC()DOC";
constexpr auto _c2py_doc_member_4 = R"DOC()DOC";
constexpr auto _c2py_doc_member_5 = R"DOC()DOC";
constexpr auto _c2py_doc_member_6 = R"DOC()DOC";
constexpr auto _c2py_doc_member_7 = R"DOC()DOC";
constexpr auto _c2py_doc_member_8 = R"DOC()DOC";
constexpr auto _c2py_doc_member_9 = R"DOC()DOC";

// ----- Member and property table ----

template <>
constinit PyGetSetDef c2py::tp_getset<_c2py_cls_0>[] = {c2py::getsetdef_from_member<&_c2py_cls_0::beta, _c2py_cls_0>("beta", _c2py_doc_member_0),
                                                        c2py::getsetdef_from_member<&_c2py_cls_0::r, _c2py_cls_0>("r", _c2py_doc_member_1),
                                                        c2py::getsetdef_from_member<&_c2py_cls_0::n, _c2py_cls_0>("n", _c2py_doc_member_2),
                                                        c2py::getsetdef_from_member<&_c2py_cls_0::q, _c2py_cls_0>("q", _c2py_doc_member_3),
                                                        c2py::getsetdef_from_member<&_c2py_cls_0::Nmax, _c2py_cls_0>("Nmax", _c2py_doc_member_4),
                                                        c2py::getsetdef_from_member<&_c2py_cls_0::T, _c2py_cls_0>("T", _c2py_doc_member_5),
                                                        c2py::getsetdef_from_member<&_c2py_cls_0::U, _c2py_cls_0>("U", _c2py_doc_member_6),
                                                        c2py::getsetdef_from_member<&_c2py_cls_0::GKt, _c2py_cls_0>("GKt", _c2py_doc_member_7),
                                                        c2py::getsetdef_from_member<&_c2py_cls_0::Tkaps, _c2py_cls_0>("Tkaps", _c2py_doc_member_8),
                                                        c2py::getsetdef_from_member<&_c2py_cls_0::Tmu, _c2py_cls_0>("Tmu", _c2py_doc_member_9),

                                                        {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<_c2py_cls_0> = R"DOC(Class for evaluating a diagram of a given order and topology in block-sparse storage
This class is used to evaluate all the backbone decompositions of a given order and topology. It reads the information from a Backbone object
and contains the Green's functions and creation/annihilation operators needed to actually compute the diagram. It also contains temporary 
data structures required for computation.)DOC"
   + std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<_c2py_cls_0>;

// ==================== module functions ====================

// convolve_ppsc
static auto const _c2py_fun_8 = c2py::dispatcher_f_kw_t{
   c2py::cfun([](triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G1,
                 triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G2) { return triqs_xca::block_sparse::convolve_ppsc(G1, G2); },
              "G1", "G2")};

// expectation_value
static auto const _c2py_fun_9 = c2py::dispatcher_f_kw_t{
   c2py::cfun([](const triqs::operators::many_body_operator_real &op, const triqs::atom_diag::atom_diag<0> &ad,
                 triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G_ppsc) { return triqs_xca::block_sparse::expectation_value(op, ad, G_ppsc); },
              "op", "ad", "G_ppsc")};

// trace
static auto const _c2py_fun_10 =
   c2py::dispatcher_f_kw_t{c2py::cfun([](triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime> G) { return triqs_xca::block_sparse::trace(G); }, "G")};

static const auto _c2py_doc_8  = _c2py_fun_8.doc(R"DOC(
Compute the Volterra "convolution" of two pseudo-particle Green's functions

Parameters
----------
G1 : {par_0}
   Left Green's function
G2 : {par_1}
   Right Green's function

Returns
-------
{ret_0}
   Convolution (G1 * G2)()
)DOC",
                                                 {{c2py::python_typename<triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>>()},
                                                  {c2py::python_typename<triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>>()}},
                                                 {c2py::python_typename<triqs::gfs::block_gf<triqs::mesh::dlr_imtime>>()});
static const auto _c2py_doc_9  = _c2py_fun_9.doc(R"DOC(
Compute the expectation value of the 2nd quantized operator op, <O> = -Tr[O G()]
using the AtomDiag instance ad to generate a block representation
and tracing with the many-body density matrix of the pseudo particle Green's function G_ppsc

Parameters
----------
op : {par_0}
   2nd quantization operator
ad : {par_1}
   AtomDiag object
G_ppsc : {par_2}
   Pseudo-particle Green's function

Returns
-------
{ret_0}
   Expectation value -Tr[G() O]
)DOC",
                                                 {{c2py::python_typename<const triqs::operators::many_body_operator_real &>()},
                                                  {c2py::python_typename<const triqs::atom_diag::atom_diag<0> &>()},
                                                  {c2py::python_typename<triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>>()}},
                                                 {c2py::python_typename<nda::dcomplex>()});
static const auto _c2py_doc_10 = _c2py_fun_10.doc(R"DOC(
Take the trace of a pseudo-particle Green's function

Parameters
----------
G : {par_0}
   Pseudo-particle Green's function
)DOC",
                                                  {{c2py::python_typename<triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>>()}});
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {"convolve_ppsc", (PyCFunction)c2py::pyfkw<_c2py_fun_8>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_8.c_str()},
   {"expectation_value", (PyCFunction)c2py::pyfkw<_c2py_fun_9>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_9.c_str()},
   {"trace", (PyCFunction)c2py::pyfkw<_c2py_fun_10>, METH_VARARGS | METH_KEYWORDS, _c2py_doc_10.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT,
                                        "block_sparse",    /* name of module */
                                        R"RAWDOC()RAWDOC", /* module documentation, may be NULL */
                                        -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
                                        module_methods,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *PyInit_block_sparse() {

  if (not c2py::check_python_version("block_sparse")) return NULL;

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
  _add_type(_c2py_cls_0, "DiagramEvaluator");
#undef _add_type

  return m;
}
#endif
// CLAIR_WRAP_GEN
