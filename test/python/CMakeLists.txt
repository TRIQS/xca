# Copy h5 files to binary dir
file(GLOB_RECURSE all_h5_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref.h5)
foreach(file ${all_h5_ref_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# List of all tests
set(all_tests
  cppdlr_imtime_ops
  triqs_soehyb/load_module
  triqs_soehyb/spinless_dimer
  triqs_soehyb/spinless_dimer_solver
  triqs_soehyb/offdiagonal_nca
  diag
  triqs_soehyb/unitary_invariance
  unitary_invariance
  triqs_soehyb/h5_io
  h5_io
  dynint
  dynint_spin_flip
  dynint_transverse
  topology
  triqs_soehyb/triqs_solver
  triqs_soehyb/one_spinful_fermion
  triqs_soehyb/analytic_diagrams
  solver_copy
  api
  api_one_fermion
  api_two_fermions
  bethe_self_cons
  dense_manual_NCA_OCA
  normalization
  spgf_ordering
  bug_bs_one_fermion
  volterra_integral
  one_time_correlator
  expectation_value
  order_conv
  )

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  add_test(NAME Py_${test} COMMAND ${TRIQS_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${test_dir}/${test_name}.py WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  set_property(TEST Py_${test} APPEND PROPERTY ENVIRONMENT_MODIFICATION
    "PYTHONPATH=path_list_prepend:${pyed_SOURCE_DIR}"
    "PYTHONPATH=path_list_prepend:${adapol_SOURCE_DIR}/src"
    "PYTHONPATH=path_list_prepend:${PROJECT_BINARY_DIR}/python")
  set_property(TEST Py_${test} APPEND PROPERTY ENVIRONMENT ${SANITIZER_RT_PRELOAD})
endforeach()
