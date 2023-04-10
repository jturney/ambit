#if (NOT STATIC_ONLY)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python 3.7 COMPONENTS Interpreter Development NumPy REQUIRED)

if (APPLE)
    set(base "@loader_path")
else()
    set(base "$ORIGIN")
endif()
file(RELATIVE_PATH relDir ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/ambit
                          ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH ${base} ${base}/${relDir})

message(STATUS "ambit rpath: ${CMAKE_INSTALL_RPATH}")

# build static libray
add_library(pyambit SHARED bindings.cc)
target_link_libraries(pyambit PUBLIC ambit::ambit)

if(ENABLE_CYCLOPS)
    target_link_libraries(pyambit PUBLIC ${CYCLOPS}/lib/libctf.a ${ELEMENTAL}/libEl.a ${ELEMENTAL}/external/pmrrr/libpmrrr.a ${MPI_LIBRARIES})
    set_target_properties(pyambit PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    set_target_properties(pyambit PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

set_target_properties(pyambit PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")

target_link_libraries(pyambit PUBLIC tgt::lapack)

target_link_libraries(
  pyambit
  PRIVATE
    Python::Module
    pybind11::module
    pybind11::lto
    pybind11::windows_extras
)

pybind11_extension(pyambit)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(pyambit)
endif()
set_target_properties(pyambit PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN 1)

# Ensure the Python module is appropriately named
#set_target_properties(pyambit PROPERTIES PREFIX "")
#set_target_properties(pyambit PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})

install(
  TARGETS
    pyambit
  EXPORT
    python_interface
  LIBRARY DESTINATION
    ${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/ambit
  )
install(
  EXPORT
    python_interface
  FILE
    "ambitTargets-python.cmake"
  NAMESPACE
    "ambit::"
  DESTINATION
    ${CMAKECONFIG_INSTALL_DIR}
  )

#endif()
