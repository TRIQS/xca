
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

template <> constexpr bool c2py::is_wrapped<triqs_xca::dense::DenseDiagramEvaluator> = true;

// ==================== enums =====================

// ==================== module classes =====================

template <> inline constexpr auto c2py::tp_name<triqs_xca::dense::DenseDiagramEvaluator> = "triqs_xca.dense.DenseDiagramEvaluator";
static auto init_0                                                                       = c2py::dispatcher_c_kw_t{
   c2py::c_constructor<triqs_xca::dense::DenseDiagramEvaluator, double, double, double, nda::vector_const_view<double>,
                                                                                             nda::array_const_view<nda::dcomplex, 3>, triqs::gfs::gf_view<triqs::mesh::dlr_imtime>, const triqs::atom_diag::atom_diag<0> &>(
      "beta", "Lambda", "eps", "hyb_poles", "hyb_coeffs", "G_ppsc", "ad")};
template <> constexpr initproc c2py::tp_init<triqs_xca::dense::DenseDiagramEvaluator> = c2py::pyfkw_constructor<init_0>;
template <>
const std::string c2py::tp_ctor_doc<triqs_xca::dense::DenseDiagramEvaluator> =
   init_0.doc(R"DOC(
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
               {c2py::python_typename<nda::array_const_view<nda::dcomplex, 3>>()},
               {c2py::python_typename<triqs::gfs::gf_view<triqs::mesh::dlr_imtime>>()},
               {c2py::python_typename<const triqs::atom_diag::atom_diag<0> &>()}});
// reset
static auto const fun_0 = c2py::dispatcher_f_kw_t{c2py::cmethod([](triqs_xca::dense::DenseDiagramEvaluator &self) { return self.reset(); }, "self")};

static const auto doc_d_0 = fun_0.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<triqs_xca::dense::DenseDiagramEvaluator>[] = {
   {"reset", (PyCFunction)c2py::pyfkw<fun_0>, METH_VARARGS | METH_KEYWORDS, doc_d_0.c_str()},
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
constexpr auto doc_member_12 = R"DOC()DOC";
constexpr auto doc_member_13 = R"DOC()DOC";

// ----- Method table ----

template <>
constinit PyGetSetDef c2py::tp_getset<triqs_xca::dense::DenseDiagramEvaluator>[] = {
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::beta, triqs_xca::dense::DenseDiagramEvaluator>("beta", doc_member_0),
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::itops, triqs_xca::dense::DenseDiagramEvaluator>("itops", doc_member_1),
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::hyb, triqs_xca::dense::DenseDiagramEvaluator>("hyb", doc_member_2),
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::hyb_refl, triqs_xca::dense::DenseDiagramEvaluator>("hyb_refl", doc_member_3),
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::Gt, triqs_xca::dense::DenseDiagramEvaluator>("Gt", doc_member_4),
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::dlr_it, triqs_xca::dense::DenseDiagramEvaluator>("dlr_it", doc_member_5),
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::r, triqs_xca::dense::DenseDiagramEvaluator>("r", doc_member_6),
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::hyb_poles, triqs_xca::dense::DenseDiagramEvaluator>("hyb_poles",
                                                                                                                             doc_member_7),
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::Sigma, triqs_xca::dense::DenseDiagramEvaluator>("Sigma", doc_member_8),
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::T, triqs_xca::dense::DenseDiagramEvaluator>("T", doc_member_9),
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::GKt, triqs_xca::dense::DenseDiagramEvaluator>("GKt", doc_member_10),
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::Tkaps, triqs_xca::dense::DenseDiagramEvaluator>("Tkaps", doc_member_11),
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::Tmu, triqs_xca::dense::DenseDiagramEvaluator>("Tmu", doc_member_12),
   c2py::getsetdef_from_member<&triqs_xca::dense::DenseDiagramEvaluator::U, triqs_xca::dense::DenseDiagramEvaluator>("U", doc_member_13),

   {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<triqs_xca::dense::DenseDiagramEvaluator> = R"DOC(Class for evaluating a diagram of a given order and topology
This class is used to evaluate all the backbone decompositions of a given
order and topology. It reads the information from a Backbone object and
contains the Green's functions and creation/annihilation operators needed to
actually compute the diagram. It also contains temporary arrays required for
computation.)DOC"
   + std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<triqs_xca::dense::DenseDiagramEvaluator>;

// ==================== module functions ====================

// NCA_dense
static auto const fun_1 = c2py::dispatcher_f_kw_t{
   c2py::cfun([](nda::array_const_view<nda::dcomplex, 3> hyb, nda::array_const_view<nda::dcomplex, 3> hyb_refl,
                 nda::array_const_view<nda::dcomplex, 3> Gt, nda::array_const_view<nda::dcomplex, 3> Fs,
                 nda::array_const_view<nda::dcomplex, 3> F_dags) { return triqs_xca::block_sparse::NCA_dense(hyb, hyb_refl, Gt, Fs, F_dags); },
              "hyb", "hyb_refl", "Gt", "Fs", "F_dags")};

// OCA_dense
static auto const fun_2 = c2py::dispatcher_f_kw_t{
   c2py::cfun([](nda::array_const_view<nda::dcomplex, 3> hyb, cppdlr::imtime_ops itops, double beta, nda::array_const_view<nda::dcomplex, 3> Gt,
                 nda::array_const_view<nda::dcomplex, 3> Fs,
                 nda::array_const_view<nda::dcomplex, 3> F_dags) { return triqs_xca::block_sparse::OCA_dense(hyb, itops, beta, Gt, Fs, F_dags); },
              "hyb", "itops", "beta", "Gt", "Fs", "F_dags"),
   c2py::cfun(
      [](nda::array_const_view<nda::dcomplex, 3> hyb, nda::array_const_view<nda::dcomplex, 3> hyb_coeffs,
         nda::array_const_view<nda::dcomplex, 3> hyb_refl, nda::array_const_view<nda::dcomplex, 3> hyb_refl_coeffs,
         nda::vector_const_view<double> hyb_poles, cppdlr::imtime_ops &itops, double beta, nda::array_const_view<nda::dcomplex, 3> Gt,
         nda::array_const_view<nda::dcomplex, 3> Fs, nda::array_const_view<nda::dcomplex, 3> F_dags) {
        return triqs_xca::block_sparse::OCA_dense(hyb, hyb_coeffs, hyb_refl, hyb_refl_coeffs, hyb_poles, itops, beta, Gt, Fs, F_dags);
      },
      "hyb", "hyb_coeffs", "hyb_refl", "hyb_refl_coeffs", "hyb_poles", "itops", "beta", "Gt", "Fs", "F_dags")};

static const auto doc_d_1 = fun_1.doc(R"DOC(
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
                                      {{c2py::python_typename<nda::array_const_view<nda::dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<nda::dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<nda::dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<nda::dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<nda::dcomplex, 3>>()}});
static const auto doc_d_2 = fun_2.doc(R"DOC(
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
                                      {{c2py::python_typename<nda::array_const_view<nda::dcomplex, 3>>()},
                                       {c2py::python_typename<cppdlr::imtime_ops>(), c2py::python_typename<cppdlr::imtime_ops &>()},
                                       {c2py::python_typename<double>()},
                                       {c2py::python_typename<nda::array_const_view<nda::dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<nda::dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<nda::dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<nda::dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<nda::dcomplex, 3>>()},
                                       {c2py::python_typename<nda::vector_const_view<double>>()}},
                                      {c2py::python_typename<nda::array<nda::dcomplex, 3>>()});
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {"NCA_dense", (PyCFunction)c2py::pyfkw<fun_1>, METH_VARARGS | METH_KEYWORDS, doc_d_1.c_str()},
   {"OCA_dense", (PyCFunction)c2py::pyfkw<fun_2>, METH_VARARGS | METH_KEYWORDS, doc_d_2.c_str()},
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
  if (PyType_Ready(&c2py::wrap_pytype<triqs_xca::dense::DenseDiagramEvaluator>) < 0) return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] = &c2py::wrap_pytype<c2py::py_range>;
  c2py::add_type_object_to_main<triqs_xca::dense::DenseDiagramEvaluator>("DenseDiagramEvaluator", m, conv_table);

  return m;
}
#endif
// CLAIR_WRAP_GEN
