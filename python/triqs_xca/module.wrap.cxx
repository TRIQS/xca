
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

// ==================== enums =====================

// ==================== module classes =====================

// ==================== module functions ====================

// compute_self_energy
static auto const fun_0 = c2py::dispatcher_f_kw_t{c2py::cfun(
   [](double beta, double Lambda, double eps, nda::array<dcomplex, 3> hyb, nda::vector<double> hyb_poles, nda::array<dcomplex, 3> hyb_coeffs,
      triqs::atom_diag::atom_diag<0> ad, int order) { return compute_self_energy(beta, Lambda, eps, hyb, hyb_poles, hyb_coeffs, ad, order); },
   "beta", "Lambda", "eps", "hyb", "hyb_poles", "hyb_coeffs", "ad", "order")};

static const auto doc_d_0 = fun_0.doc(R"DOC(
Solve for the self-energy up to a given order

Parameters
----------
beta : {par_0}
   Inverse temperature
Lambda : {par_1}
   DLR Lambda parameter
eps : {par_2}
   DLR epsilon parameter
hyb : {par_3}
   Hybridization function
hyb_poles : {par_4}
   Hybridization poles
hyb_coeffs : {par_5}
   Hybridization coefficients
ad : {par_6}
   Atom diagonalization object
order : {par_7}
   Perturbation order (1, 2, or 3)

Returns
-------
{ret_0}
   std::vector<nda::array<dcomplex, 3>> Self-energy blocks
)DOC",
                                      {{c2py::python_typename<double>()},
                                       {c2py::python_typename<double>()},
                                       {c2py::python_typename<double>()},
                                       {c2py::python_typename<nda::array<dcomplex, 3>>()},
                                       {c2py::python_typename<nda::vector<double>>()},
                                       {c2py::python_typename<nda::array<dcomplex, 3>>()},
                                       {c2py::python_typename<triqs::atom_diag::atom_diag<0>>()},
                                       {c2py::python_typename<int>()}},
                                      {c2py::python_typename<std::vector<nda::array<dcomplex, 3>>>()});
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {"compute_self_energy", (PyCFunction)c2py::pyfkw<fun_0>, METH_VARARGS | METH_KEYWORDS, doc_d_0.c_str()},
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

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] = &c2py::wrap_pytype<c2py::py_range>;

  return m;
}
#endif
// CLAIR_WRAP_GEN
