
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

// NCA_dense
static auto const fun_0 = c2py::dispatcher_f_kw_t{c2py::cfun(
   [](nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_refl, nda::array_const_view<dcomplex, 3> Gt,
      nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags) { return NCA_dense(hyb, hyb_refl, Gt, Fs, F_dags); },
   "hyb", "hyb_refl", "Gt", "Fs", "F_dags")};

// OCA_dense
static auto const fun_1 = c2py::dispatcher_f_kw_t{
   c2py::cfun([](nda::array_const_view<dcomplex, 3> hyb, cppdlr::imtime_ops itops, double beta, nda::array_const_view<dcomplex, 3> Gt,
                 nda::array_const_view<dcomplex, 3> Fs,
                 nda::array_const_view<dcomplex, 3> F_dags) { return OCA_dense(hyb, itops, beta, Gt, Fs, F_dags); },
              "hyb", "itops", "beta", "Gt", "Fs", "F_dags"),
   c2py::cfun(
      [](nda::array_const_view<dcomplex, 3> hyb, nda::array_const_view<dcomplex, 3> hyb_coeffs, nda::array_const_view<dcomplex, 3> hyb_refl,
         nda::array_const_view<dcomplex, 3> hyb_refl_coeffs, nda::vector_const_view<double> hyb_poles, cppdlr::imtime_ops &itops, double beta,
         nda::array_const_view<dcomplex, 3> Gt, nda::array_const_view<dcomplex, 3> Fs, nda::array_const_view<dcomplex, 3> F_dags) {
        return OCA_dense(hyb, hyb_coeffs, hyb_refl, hyb_refl_coeffs, hyb_poles, itops, beta, Gt, Fs, F_dags);
      },
      "hyb", "hyb_coeffs", "hyb_refl", "hyb_refl_coeffs", "hyb_poles", "itops", "beta", "Gt", "Fs", "F_dags")};

static const auto doc_d_0 = fun_0.doc(R"DOC(
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
                                      {{c2py::python_typename<nda::array_const_view<dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<dcomplex, 3>>()}});
static const auto doc_d_1 = fun_1.doc(R"DOC(
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
                                      {{c2py::python_typename<nda::array_const_view<dcomplex, 3>>()},
                                       {c2py::python_typename<cppdlr::imtime_ops>(), c2py::python_typename<cppdlr::imtime_ops &>()},
                                       {c2py::python_typename<double>()},
                                       {c2py::python_typename<nda::array_const_view<dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<dcomplex, 3>>()},
                                       {c2py::python_typename<nda::array_const_view<dcomplex, 3>>()},
                                       {c2py::python_typename<nda::vector_const_view<double>>()}},
                                      {c2py::python_typename<nda::array<dcomplex, 3>>()});
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
   {"NCA_dense", (PyCFunction)c2py::pyfkw<fun_0>, METH_VARARGS | METH_KEYWORDS, doc_d_0.c_str()},
   {"OCA_dense", (PyCFunction)c2py::pyfkw<fun_1>, METH_VARARGS | METH_KEYWORDS, doc_d_1.c_str()},
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

  m = PyModule_Create(&module_def);
  if (m == NULL) return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] = &c2py::wrap_pytype<c2py::py_range>;

  return m;
}
#endif
// CLAIR_WRAP_GEN
