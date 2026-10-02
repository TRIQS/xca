# Copy h5 files to binary dir
file(GLOB_RECURSE all_h5_ref_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.ref.h5)
foreach(file ${all_h5_ref_files})
  configure_file(${file} ${file} COPYONLY)
endforeach()

# Helper library of the tests, in namespace triqs_xca::test_utils
file(GLOB test_utils_sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} test_utils/*.cpp)
add_library(${PROJECT_NAME}_test_utils STATIC ${test_utils_sources})
target_include_directories(${PROJECT_NAME}_test_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME}_test_utils PUBLIC ${PROJECT_NAME}::${PROJECT_NAME}_c PRIVATE ${PROJECT_NAME}_warnings)

# List of all tests
file(GLOB_RECURSE all_tests RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
list(FILTER all_tests EXCLUDE REGEX "^test_utils/")

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  # Tests in a subdirectory are listed as <dir>/<name> in ctest, with the target <dir>_<name>
  if(test_dir)
    set(test_label ${test_dir}/${test_name})
    string(REPLACE "/" "_" test_target ${test_dir}_${test_name})
  else()
    set(test_label ${test_name})
    set(test_target ${test_name})
  endif()
  add_executable(${test_target} ${test})
  set_target_properties(${test_target} PROPERTIES OUTPUT_NAME ${test_name})
  target_link_libraries(${test_target} ${PROJECT_NAME}::${PROJECT_NAME}_c ${PROJECT_NAME}_warnings gtest_main)
  if(test_dir STREQUAL "triqs_soehyb")
    target_link_libraries(${test_target} ${PROJECT_NAME}::triqs_soehyb_c)
  else()
    target_link_libraries(${test_target} ${PROJECT_NAME}_test_utils)
  endif()
  set_property(TARGET ${test_target} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  add_test(NAME ${test_label} COMMAND ${test_target} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  # Run clang-tidy if found
  if(CLANG_TIDY_EXECUTABLE)
    set_target_properties(${test_target} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXECUTABLE}")
  endif()
  # Run cppcheck if found
  if(CPPCHECK_EXECUTABLE)
    add_custom_command(
      TARGET ${test_target}
      COMMAND ${CPPCHECK_EXECUTABLE}
      --enable=warning,style,performance,portability
      --std=c++20
      --template=gcc
      --verbose
      --force
      --quiet
      ${CMAKE_CURRENT_SOURCE_DIR}/${test}
    )
  endif()
endforeach()
