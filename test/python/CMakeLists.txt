add_subdirectory(exp_order_conv)

# Copy h5 files to binary dir
file(GLOB_RECURSE all_h5_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref.h5)
foreach(file ${all_h5_ref_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# List of all tests
set(all_tests
  test_cppdlr
  triqs_soehyb/load_module
  triqs_soehyb/spinless_dimer
  triqs_soehyb/spinless_dimer_solver
  triqs_soehyb/offdiagonal_nca
  test_diag
  triqs_soehyb/unitary_invariance
  unitary_invariance_bs
  triqs_soehyb/h5_io
  h5_io_bs
  dynint
  dynint_spin_flip
  dynint_transverse
  topology
  triqs_soehyb/triqs_solver
  triqs_soehyb/one_spinful_fermion
  triqs_soehyb/analytic_diagrams
  block_sparse_copy
  block_sparse_api
  block_sparse_api_one_fermion
  block_sparse_api_two_fermions
  block_sparse_bethe_self_cons
  dense_manual_NCA_OCA
  block_sparse_normalization
  block_sparse_spgf_ordering
  bug_bs_one_fermion
  volterra_integral
  one_time_correlator
  expectation_value
  exp_order_conv/n_level_non_int_discrete_bath
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
