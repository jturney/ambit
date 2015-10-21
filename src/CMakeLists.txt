set(TENSOR_HEADERS
    ${PROJECT_SOURCE_DIR}/include/ambit/print.h
    ${PROJECT_SOURCE_DIR}/include/ambit/tensor.h
    ${PROJECT_SOURCE_DIR}/include/ambit/timer.h
    ${PROJECT_SOURCE_DIR}/include/ambit/blocked_tensor.h
    ${PROJECT_SOURCE_DIR}/include/ambit/common_types.h
    ${PROJECT_SOURCE_DIR}/include/ambit/settings.h
    ${PROJECT_SOURCE_DIR}/include/ambit/composite_tensor.h
    ${PROJECT_SOURCE_DIR}/include/ambit/symblocked_tensor.h

    ${PROJECT_SOURCE_DIR}/include/ambit/io/file.h
    ${PROJECT_SOURCE_DIR}/include/ambit/io/io.h
    ${PROJECT_SOURCE_DIR}/include/ambit/io/iwl.h
    ${PROJECT_SOURCE_DIR}/include/ambit/io/manager.h

    ${PROJECT_SOURCE_DIR}/include/ambit/helpers/psi4/io.h

    math/math.h

    tensor/core/core.h
    tensor/disk/disk.h
    tensor/indices.h
    tensor/globals.h
    tensor/macros.h
    tensor/tensorimpl.h
)

set(TENSOR_SOURCES
    helpers/psi4/io.cc

    io/file.cc
    io/iwl.cc
    io/manager.cc

    math/blas.cc
    math/lapack.cc

    tensor/core/core.cc
    tensor/disk/disk.cc

    tensor/composite_tensor.cc
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
)

# if we have MPI and Cyclops is enabled
if(ENABLE_CYCLOPS)
    list(APPEND TENSOR_HEADERS tensor/cyclops/cyclops.h)
    list(APPEND TENSOR_SOURCES tensor/cyclops/cyclops.cc)
endif()

# build libray
add_library(ambit SHARED ${TENSOR_SOURCES} ${TENSOR_HEADERS})

if(ENABLE_CYCLOPS)
    target_link_libraries(ambit ${CYCLOPS}/lib/libctf.a ${ELEMENTAL}/libEl.a ${ELEMENTAL}/external/pmrrr/libpmrrr.a ${MPI_LIBRARIES})
endif()

target_link_libraries(ambit ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${Boost_LIBRARIES})

install(TARGETS ambit LIBRARY DESTINATION lib)

if(ENABLE_PYTHON)
    add_subdirectory(python)
endif()

