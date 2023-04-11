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

pybind11_add_module(pyambit MODULE bindings.cc)
target_compile_features(pyambit PRIVATE "cxx_std_17")
#if (pybind11_VERSION VERSION_GREATER_EQUAL 2.7.0)
    # suppress error in L2
    # * https://github.com/pybind/pybind11/pull/2919
    target_compile_definitions(pyambit PRIVATE NDEBUG=1)
#endif()
#target_link_libraries(pyambit PRIVATE ambit::ambit)

install(TARGETS pyambit
        EXPORT "ambitTargets"
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/ambit)
#install(EXPORT "${PN}Targets"
#        NAMESPACE "${PN}::"
#        DESTINATION ${PYLIBEFP_CMAKECONFIG_INSTALL_DIR})

#set_target_properties(pyambit PROPERTIES
#                           OUTPUT_NAME core
#                           EXPORT_NAME core
#                           INSTALL_RPATH "${psi4_RPATH}"
#                           BUILD_WITH_INSTALL_RPATH ON)

## build static libray
#add_library(pyambit SHARED bindings.cc)
##add_dependencies(pyambit ambit::ambit)
#target_link_libraries(pyambit PUBLIC ambit::ambit)
#
#if(ENABLE_CYCLOPS)
#    target_link_libraries(pyambit PUBLIC ${CYCLOPS}/lib/libctf.a ${ELEMENTAL}/libEl.a ${ELEMENTAL}/external/pmrrr/libpmrrr.a ${MPI_LIBRARIES})
#    set_target_properties(pyambit PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#    set_target_properties(pyambit PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
#endif()



    if(MSVC)
        target_compile_definitions(pyambit PUBLIC _USE_MATH_DEFINES)
        target_compile_options(pyambit PUBLIC "/EHsc")
        set_target_properties(
          pyambit
          PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS ON
          )
    endif()

    target_link_libraries(
      pyambit
      PRIVATE
        ambit::ambit
      )



################

set_target_properties(pyambit PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")

#target_link_libraries(pyambit PUBLIC tgt::lapack)

# target_link_libraries(pyambit PRIVATE ambit::ambit)
# target_link_libraries(pyambit PUBLIC tgt::lapack)
#  ERROR (pyambit,Lib/site-packages/ambit/pyambit.cp39-win_amd64.pyd): Needed DSO Library/bin/libblas.dll found in ['libblas']
#  ERROR (pyambit,Lib/site-packages/ambit/pyambit.cp39-win_amd64.pyd): .. but ['libblas'] not in reqs/run, (i.e. it is overlinking) (likely) or a missing dependency (less likely)
#  ERROR (pyambit,Lib/site-packages/ambit/pyambit.cp39-win_amd64.pyd): Needed DSO Library/bin/liblapack.dll found in ['liblapack']
#  ERROR (pyambit,Lib/site-packages/ambit/pyambit.cp39-win_amd64.pyd): .. but ['liblapack'] not in reqs/run, (i.e. it is overlinking) (likely) or a missing dependency (less likely)

#target_link_libraries(
#  pyambit
#  PRIVATE
#    Python::Module
#    pybind11::module
#    pybind11::lto
#    pybind11::windows_extras
#)
#
#pybind11_extension(pyambit)
#if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
#    # Strip unnecessary sections of the binary on Linux/macOS
#    pybind11_strip(pyambit)
#endif()
#set_target_properties(pyambit PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN 1)
##target_link_libraries(pyambit PUBLIC ambit-lib)

# Ensure the Python module is appropriately named
#set_target_properties(pyambit PROPERTIES PREFIX "")
#set_target_properties(pyambit PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})

#install(TARGETS pyambit DESTINATION ${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/ambit)
#install(
#  TARGETS
#    pyambit
#  EXPORT
#    python_interface
#  LIBRARY DESTINATION
#    ${CMAKE_INSTALL_LIBDIR}${PYMOD_INSTALL_LIBDIR}/ambit
#  )
#install(
#  EXPORT
#    python_interface
#  FILE
#    "ambitTargets-python.cmake"
#  NAMESPACE
#    "ambit::"
#  DESTINATION
#    ${CMAKECONFIG_INSTALL_DIR}
#  )

#endif()
