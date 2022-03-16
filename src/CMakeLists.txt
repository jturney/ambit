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
        tensor/indices.h
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

# build library
if (NOT STATIC_ONLY)
    add_library(ambit-shared SHARED ${TENSOR_SOURCES} ${TENSOR_HEADERS})
    target_link_libraries (ambit-shared PUBLIC tgt::lapack
                                                tgt::hdf5)
    set_target_properties (ambit-shared PROPERTIES OUTPUT_NAME "ambit"
                                                   EXPORT_NAME "ambit")
endif()

if (NOT SHARED_ONLY)
    add_library(ambit-static STATIC ${TENSOR_SOURCES} ${TENSOR_HEADERS})
    target_link_libraries (ambit-static PUBLIC tgt::lapack
                                                tgt::hdf5)
    set_target_properties (ambit-static PROPERTIES OUTPUT_NAME "ambit"
                                                   EXPORT_NAME "ambit"
                                                   POSITION_INDEPENDENT_CODE ${BUILD_FPIC})
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

message("ambitambit a/so ${STATIC_ONLY} ${SHARED_ONLY}")
if (STATIC_ONLY)
    add_library(ambit-lib ALIAS ambit-static)
else()
    add_library(ambit-lib ALIAS ambit-shared)
endif()

add_subdirectory(python)

# <<<  Install  >>>

if (NOT STATIC_ONLY)
    install (TARGETS ambit-shared
                     EXPORT "ambitTargets-shared"
                     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if (NOT SHARED_ONLY)
    install (TARGETS ambit-static
                     EXPORT "ambitTargets-static"
                     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

# <<<  Export interface  >>>

if (NOT STATIC_ONLY)
    if (APPLE)
        set_target_properties (ambit-shared PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    endif()
    target_compile_definitions (ambit-shared INTERFACE USING_${PROJECT_NAME})
    target_include_directories (ambit-shared INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                                                       $<TARGET_PROPERTY:tgt::hdf5,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

if (NOT SHARED_ONLY)
    target_compile_definitions (ambit-static INTERFACE USING_${PROJECT_NAME})
    target_include_directories (ambit-static INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                                                       $<TARGET_PROPERTY:tgt::hdf5,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

# <<<  Export Config  >>>

set(CMAKECONFIG_INSTALL_DIR "share/cmake/${PROJECT_NAME}")
if (NOT STATIC_ONLY)
    install (EXPORT "${PROJECT_NAME}Targets-shared"
                    NAMESPACE "${PROJECT_NAME}::"
                    DESTINATION ${CMAKECONFIG_INSTALL_DIR})
endif()

if (NOT SHARED_ONLY)
    install (EXPORT "${PROJECT_NAME}Targets-static"
                    NAMESPACE "${PROJECT_NAME}::"
                    DESTINATION ${CMAKECONFIG_INSTALL_DIR})
endif()

