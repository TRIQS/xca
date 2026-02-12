
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

// ==================== Wrapped classes =====================

template <> constexpr bool c2py::is_wrapped<DiagramEvaluator> = true;

// ==================== enums =====================

// ==================== module classes =====================

template <> inline constexpr auto c2py::tp_name<DiagramEvaluator> = "triqs_xca.module.DiagramEvaluator";
static auto init_0                                                = c2py::dispatcher_c_kw_t{
   c2py::c_constructor<DiagramEvaluator, double, double, double, nda::vector_const_view<double>, nda::array_const_view<dcomplex, 3>,
                                                                      triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>, const triqs::atom_diag::atom_diag<0> &>(
      "beta", "Lambda", "eps", "hyb_poles", "hyb_coeffs", "G_ppsc", "ad")};
template <> constexpr initproc c2py::tp_init<DiagramEvaluator> = c2py::pyfkw_constructor<init_0>;
template <>
const std::string c2py::tp_ctor_doc<DiagramEvaluator> = init_0.doc(R"DOC(
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
                                                                   {{c2py::python_typename<double>()},
                                                                    {c2py::python_typename<double>()},
                                                                    {c2py::python_typename<double>()},
                                                                    {c2py::python_typename<nda::vector_const_view<double>>()},
                                                                    {c2py::python_typename<nda::array_const_view<dcomplex, 3>>()},
                                                                    {c2py::python_typename<triqs::gfs::block_gf_view<triqs::mesh::dlr_imtime>>()},
                                                                    {c2py::python_typename<const triqs::atom_diag::atom_diag<0> &>()}});
// compute_self_energy
static auto const fun_0 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](DiagramEvaluator &self, nda::array_const_view<int, 2> topology) { return self.compute_self_energy(topology); }, "self",
                 "topology"),
   c2py::cmethod([](DiagramEvaluator &self, nda::array_const_view<int, 2> topology, int f_ix) { return self.compute_self_energy(topology, f_ix); },
                 "self", "topology", "f_ix")};

// compute_single_ptcle_gf
static auto const fun_1 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](DiagramEvaluator &self, nda::array_const_view<int, 2> topology) { return self.compute_single_ptcle_gf(topology); }, "self",
                 "topology"),
   c2py::cmethod(
      [](DiagramEvaluator &self, nda::array_const_view<int, 2> topology, int f_ix) { return self.compute_single_ptcle_gf(topology, f_ix); }, "self",
      "topology", "f_ix")};

// get_num_self_energy_backbones
static auto const fun_2 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](DiagramEvaluator &self, nda::array_const_view<int, 2> topology) { return self.get_num_self_energy_backbones(topology); }, "self", "topology")};

// get_num_single_ptcle_gf_backbones
static auto const fun_3 = c2py::dispatcher_f_kw_t{
   c2py::cmethod([](DiagramEvaluator &self, nda::array_const_view<int, 2> topology) { return self.get_num_single_ptcle_gf_backbones(topology); },
                 "self", "topology")};

// print_self_energy_backbone
static auto const fun_4 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](DiagramEvaluator &self, nda::array_const_view<int, 2> topology, int f_ix) { return self.print_self_energy_backbone(topology, f_ix); }, "self",
   "topology", "f_ix")};

// print_single_ptcle_gf_backbone
static auto const fun_5 = c2py::dispatcher_f_kw_t{c2py::cmethod(
   [](DiagramEvaluator &self, nda::array_const_view<int, 2> topology, int f_ix) { return self.print_single_ptcle_gf_backbone(topology, f_ix); },
   "self", "topology", "f_ix")};

// reset
static auto const fun_6 = c2py::dispatcher_f_kw_t{c2py::cmethod([](DiagramEvaluator &self) { return self.reset(); }, "self")};

static const auto doc_d_0 = fun_0.doc(R"DOC()DOC");
static const auto doc_d_1 = fun_1.doc(R"DOC()DOC");
static const auto doc_d_2 = fun_2.doc(R"DOC()DOC");
static const auto doc_d_3 = fun_3.doc(R"DOC()DOC");
static const auto doc_d_4 = fun_4.doc(R"DOC()DOC");
static const auto doc_d_5 = fun_5.doc(R"DOC()DOC");
static const auto doc_d_6 = fun_6.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<DiagramEvaluator>[] = {
   {"compute_self_energy", (PyCFunction)c2py::pyfkw<fun_0>, METH_VARARGS | METH_KEYWORDS, doc_d_0.c_str()},
   {"compute_single_ptcle_gf", (PyCFunction)c2py::pyfkw<fun_1>, METH_VARARGS | METH_KEYWORDS, doc_d_1.c_str()},
   {"get_num_self_energy_backbones", (PyCFunction)c2py::pyfkw<fun_2>, METH_VARARGS | METH_KEYWORDS, doc_d_2.c_str()},
   {"get_num_single_ptcle_gf_backbones", (PyCFunction)c2py::pyfkw<fun_3>, METH_VARARGS | METH_KEYWORDS, doc_d_3.c_str()},
   {"print_self_energy_backbone", (PyCFunction)c2py::pyfkw<fun_4>, METH_VARARGS | METH_KEYWORDS, doc_d_4.c_str()},
   {"print_single_ptcle_gf_backbone", (PyCFunction)c2py::pyfkw<fun_5>, METH_VARARGS | METH_KEYWORDS, doc_d_5.c_str()},
   {"reset", (PyCFunction)c2py::pyfkw<fun_6>, METH_VARARGS | METH_KEYWORDS, doc_d_6.c_str()},
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

constexpr auto doc_member_0  = R"DOC()DOC";
constexpr auto doc_member_1  = R"DOC()DOC";
constexpr auto doc_member_2  = R"DOC()DOC";
constexpr auto doc_member_3  = R"DOC()DOC";
constexpr auto doc_member_4  = R"DOC()DOC";
constexpr auto doc_member_5  = R"DOC()DOC";
constexpr auto doc_member_6  = R"DOC()DOC";
constexpr auto doc_member_7  = R"DOC()DOC";
constexpr auto doc_member_8  = R"DOC()DOC";
constexpr auto doc_member_9  = R"DOC()DOC";
constexpr auto doc_member_10 = R"DOC()DOC";
constexpr auto doc_member_11 = R"DOC()DOC";

// ----- Method table ----

template <>
constinit PyGetSetDef c2py::tp_getset<DiagramEvaluator>[] = {
   c2py::getsetdef_from_member<&DiagramEvaluator::beta, DiagramEvaluator>("beta", doc_member_0),
   c2py::getsetdef_from_member<&DiagramEvaluator::r, DiagramEvaluator>("r", doc_member_1),
   c2py::getsetdef_from_member<&DiagramEvaluator::n, DiagramEvaluator>("n", doc_member_2),
   c2py::getsetdef_from_member<&DiagramEvaluator::q, DiagramEvaluator>("q", doc_member_3),
   c2py::getsetdef_from_member<&DiagramEvaluator::Nmax, DiagramEvaluator>("Nmax", doc_member_4),
   c2py::getsetdef_from_member<&DiagramEvaluator::hyb, DiagramEvaluator>("hyb", doc_member_5),
   c2py::getsetdef_from_member<&DiagramEvaluator::hyb_poles, DiagramEvaluator>("hyb_poles", doc_member_6),
   c2py::getsetdef_from_member<&DiagramEvaluator::T, DiagramEvaluator>("T", doc_member_7),
   c2py::getsetdef_from_member<&DiagramEvaluator::U, DiagramEvaluator>("U", doc_member_8),
   c2py::getsetdef_from_member<&DiagramEvaluator::GKt, DiagramEvaluator>("GKt", doc_member_9),
   c2py::getsetdef_from_member<&DiagramEvaluator::Tkaps, DiagramEvaluator>("Tkaps", doc_member_10),
   c2py::getsetdef_from_member<&DiagramEvaluator::Tmu, DiagramEvaluator>("Tmu", doc_member_11),

   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<DiagramEvaluator> = R"DOC(Class for evaluating a diagram of a given order and topology in block-sparse storage
This class is used to evaluate all the backbone decompositions of a given order and topology. It reads the information from a Backbone object
and contains the Green's functions and creation/annihilation operators needed to actually compute the diagram. It also contains temporary 
data structures required for computation.)DOC"
   + std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<DiagramEvaluator>;

// ==================== module functions ====================

//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT,
                                        "module",          /* name of module */
                                        R"RAWDOC()RAWDOC", /* module documentation, may be NULL */
                                        -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
                                        module_methods,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *PyInit_module() {

  if (not c2py::check_python_version("module")) return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0) return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<DiagramEvaluator>) < 0) return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] = &c2py::wrap_pytype<c2py::py_range>;
  c2py::add_type_object_to_main<DiagramEvaluator>("DiagramEvaluator", m, conv_table);

  return m;
}
#endif
// CLAIR_WRAP_GEN
