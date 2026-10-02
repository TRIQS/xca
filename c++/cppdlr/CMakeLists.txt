# Header-only additions to the external cppdlr library. The headers are installed into
# include/cppdlr, next to (overlaying) the headers of cppdlr proper.
#
# This is also the common base target of the C++ libraries and Python modules: it carries the
# compile definitions shared by all of them.
add_library(${PROJECT_NAME}_cppdlr INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME}_cppdlr ALIAS ${PROJECT_NAME}_cppdlr)

target_link_libraries(${PROJECT_NAME}_cppdlr INTERFACE triqs)
target_include_directories(${PROJECT_NAME}_cppdlr INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/c++>)
target_include_directories(${PROJECT_NAME}_cppdlr SYSTEM INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)
target_compile_definitions(${PROJECT_NAME}_cppdlr INTERFACE
  TRIQS_XCA_GIT_HASH=${PROJECT_GIT_HASH}
  TRIQS_GIT_HASH=${TRIQS_GIT_HASH}
  $<$<CONFIG:Debug>:TRIQS_XCA_DEBUG>
  $<$<CONFIG:Debug>:TRIQS_DEBUG>
  $<$<CONFIG:Debug>:TRIQS_ARRAYS_ENFORCE_BOUNDCHECK>
)

install(TARGETS ${PROJECT_NAME}_cppdlr EXPORT ${PROJECT_NAME}-targets)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.hpp")
