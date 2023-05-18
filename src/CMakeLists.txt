set(TENSOR_HEADERS
        ${PROJECT_SOURCE_DIR}/include/ambit/print.h
        ${PROJECT_SOURCE_DIR}/include/ambit/tensor.h
        ${PROJECT_SOURCE_DIR}/include/ambit/timer.h
        ${PROJECT_SOURCE_DIR}/include/ambit/blocked_tensor.h
        ${PROJECT_SOURCE_DIR}/include/ambit/sym_blocked_tensor.h
        ${PROJECT_SOURCE_DIR}/include/ambit/common_types.h
        ${PROJECT_SOURCE_DIR}/include/ambit/settings.h

        ../include/ambit/io/hdf5.h
        ../include/ambit/io/hdf5/attribute.h
        ../include/ambit/io/hdf5/dataset.h
        ../include/ambit/io/hdf5/dataspace.h
        ../include/ambit/io/hdf5/file.h
        ../include/ambit/io/hdf5/group.h
        ../include/ambit/io/hdf5/location.h
        #../include/ambit/io/psi4/file.h
        #../include/ambit/io/psi4/io.h
        #../include/ambit/io/psi4/iwl.h
#       # ../include/ambit/io/psi4/manager.h

#${PROJECT_SOURCE_DIR}/include/ambit/helpers/psi4/io.h

        math/math.h

        tensor/core/core.h
        tensor/disk/disk.h
        #tensor/indices.h
        tensor/globals.h
        tensor/macros.h
        tensor/tensorimpl.h
        )

set(TENSOR_SOURCES
    #helpers/psi4/io.cc

        io/hdf5/attribute.cc
        io/hdf5/dataset.cc
        io/hdf5/dataspace.cc
        io/hdf5/file.cc
        io/hdf5/group.cc
        io/hdf5/location.cc

        #io/psi4/file.cc
        #io/psi4/iwl.cc
        #io/psi4/manager.cc

        math/blas.cc
        math/lapack.cc

        tensor/core/core.cc
        tensor/disk/disk.cc

        tensor/indices.cc
        tensor/globals.cc
        tensor/labeled_tensor.cc
        tensor/print.cc
        tensor/slice.cc
        tensor/sliced_tensor.cc
        tensor/tensor.cc
        tensor/tensorimpl.cc
        tensor/timer.cc

        blocked_tensor/blocked_tensor.cc
        blocked_tensor/sym_blocked_tensor.cc
        )

# if we have MPI and Cyclops is enabled
if (ENABLE_CYCLOPS)
    list(APPEND TENSOR_HEADERS tensor/cyclops/cyclops.h)
    list(APPEND TENSOR_SOURCES tensor/cyclops/cyclops.cc)
endif ()

#if (ENABLE_PSI4)
#    list(APPEND TENSOR_HEADERS
#            ${PROJECT_SOURCE_DIR}/include/ambit/helpers/psi4/integrals.h
#            ${PROJECT_SOURCE_DIR}/include/ambit/helpers/psi4/convert.h)
#    list(APPEND TENSOR_SOURCES
#            helpers/psi4/integrals.cc
#            helpers/psi4/convert.cc)
#endif ()

list(SORT TENSOR_SOURCES)

if(MSVC)
    # MSVC does not include <cmath> constants, unless _USE_MATH_DEFINES is defined.
    # _CRT_* to squash some getenv, strdup, strncpy, ctime, fopen warnings
    add_compile_definitions(
      _USE_MATH_DEFINES
      _CRT_NONSTDC_NO_DEPRECATE
      _CRT_NONSTDC_NO_WARNINGS
      _CRT_SECURE_NO_WARNINGS
      )
    # Set the exception handling model (allows "throw") (this isn't working; still need it in `-D CMAKE_CXX_FLAGS`)
    add_compile_options("/EHsc")
endif()

# build library
set(export_properties
  "ambit_VERSION"
  )

# LAB note: I've gone back and forth over the years on whether tgt::hdf5 and tgt::lapack should be PUBLIC or PRIVATE links of libambit.
# * My present conclusion is that hdf5 is PUBLIC (appears in ambit headers) and lapack is PRIVATE (not in headers; ambit consumer is indifferent).
# * This conclusion is supported by overlinking errors in the conda-build when both are PUBLIC.
# * But, when lapack is PRIVATE-ized, the ambit GHA CI check for Linux fails.
# * The failure cause is the unsetting of LDFLAGS (set upon conda activate in an env including conda compilers).
#   This removes -rpath-link that correctly handled using lapack as a secondary dependency when linking pyambit.
# * Best explanation I've found is at: http://www.kaizou.org/2015/01/linux-libraries.html
# * So, I understand what's happening Linux-wise, but I've failed at getting CMake to set this up right w/o the external LDFLAGS.
#   I strongly suspect IMPORTED_LINK_DEPENDENT_LIBRARIES is part of the answer.
# * Final state is that I'm leaving both PUBLIC on the grounds that a robust build is most important, and most ppl don't
#   care about overlinking. The conda package has a little patch that turns lapack PRIVATE.

if (NOT STATIC_ONLY)
    add_library(ambit-shared SHARED ${TENSOR_SOURCES})
    target_link_libraries (
      ambit-shared
      PUBLIC
        tgt::hdf5
        tgt::lapack
      )
    set_target_properties (
      ambit-shared
      PROPERTIES
        OUTPUT_NAME "ambit"
        EXPORT_NAME "ambit"
        ambit_VERSION ${ambit_VERSION}
      )
    set_property(TARGET ambit-shared APPEND PROPERTY EXPORT_PROPERTIES "${export_properties}")
endif()

if (NOT SHARED_ONLY)
    add_library(ambit-static STATIC ${TENSOR_SOURCES})
    target_link_libraries (
      ambit-static
      PUBLIC
        tgt::hdf5
        tgt::lapack
      )
    set_target_properties (
      ambit-static
      PROPERTIES
        OUTPUT_NAME "ambit"
        EXPORT_NAME "ambit"
        POSITION_INDEPENDENT_CODE ${BUILD_FPIC}
        ambit_VERSION ${ambit_VERSION}
      )
    set_property(TARGET ambit-static APPEND PROPERTY EXPORT_PROPERTIES "${export_properties}")
endif()

# LAB: not addressed
if (ENABLE_CYCLOPS)
    if (NOT STATIC_ONLY)
        target_link_libraries(ambit-shared ${CYCLOPS}/lib/libctf.a ${ELEMENTAL}/libEl.a ${ELEMENTAL}/external/pmrrr/libpmrrr.a ${MPI_LIBRARIES})
    endif()
    target_link_libraries(ambit-static ${CYCLOPS}/lib/libctf.a ${ELEMENTAL}/libEl.a ${ELEMENTAL}/external/pmrrr/libpmrrr.a ${MPI_LIBRARIES})
endif ()

#if (NOT STATIC_ONLY)
#    target_link_libraries(ambit-shared
#        ${LAPACK_LIBRARIES}
#        ${BLAS_LIBRARIES}
#        ${HDF5_LIBRARIES}
#        )
#endif()
#target_link_libraries(ambit-static
#        ${LAPACK_LIBRARIES}
#        ${BLAS_LIBRARIES}
#        ${HDF5_LIBRARIES}
#        )

if (STATIC_ONLY)
    add_library(ambit-lib ALIAS ambit-static)
    add_library(ambit::ambit ALIAS ambit-static)
else()
    add_library(ambit-lib ALIAS ambit-shared)
    add_library(ambit::ambit ALIAS ambit-shared)
endif()

# <<<  Install  >>>

if (NOT STATIC_ONLY)
    install(
      TARGETS
        ambit-shared
      EXPORT
        shared_interface
      RUNTIME DESTINATION
        ${CMAKE_INSTALL_BINDIR}
      ARCHIVE DESTINATION
        ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION
        ${CMAKE_INSTALL_LIBDIR}
      PUBLIC_HEADER DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}
      )
endif()

if (NOT SHARED_ONLY)
    install(
      TARGETS
        ambit-static
      EXPORT
        static_interface
      RUNTIME DESTINATION
        ${CMAKE_INSTALL_BINDIR}
      ARCHIVE DESTINATION
        ${CMAKE_INSTALL_LIBDIR}
      LIBRARY DESTINATION
        ${CMAKE_INSTALL_LIBDIR}
      PUBLIC_HEADER DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}
      )
endif()

# <<<  Export interface  >>>

if (NOT STATIC_ONLY)
    if (APPLE)
        set_target_properties (ambit-shared PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif()
    target_compile_definitions (ambit-shared INTERFACE USING_${PROJECT_NAME})
    target_include_directories (ambit-shared INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
endif()

if (NOT SHARED_ONLY)
    target_compile_definitions (ambit-static INTERFACE USING_${PROJECT_NAME})
    target_include_directories (ambit-static INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
endif()

# <<<  Export Config  >>>

if (NOT STATIC_ONLY)
    install(
      EXPORT
        shared_interface
      FILE
        "${PROJECT_NAME}Targets-shared.cmake"
      NAMESPACE
        "${PROJECT_NAME}::"
      DESTINATION
        ${CMAKECONFIG_INSTALL_DIR}
      )
endif()

if (NOT SHARED_ONLY)
    install(
      EXPORT
        static_interface
      FILE
        "${PROJECT_NAME}Targets-static.cmake"
      NAMESPACE
        "${PROJECT_NAME}::"
      DESTINATION
        ${CMAKECONFIG_INSTALL_DIR}
      )
endif()

