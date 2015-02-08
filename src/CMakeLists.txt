set(TENSOR_HEADERS
    ${PROJECT_SOURCE_DIR}/include/tensor/tensor.h
    ${PROJECT_SOURCE_DIR}/include/tensor/io/file.h
    ${PROJECT_SOURCE_DIR}/include/tensor/io/io.h
    ${PROJECT_SOURCE_DIR}/include/tensor/io/iwl.h
    ${PROJECT_SOURCE_DIR}/include/tensor/io/manager.h
    math/math.h
    tensor/core.h
    tensor/indices.h
    tensor/macros.h
    tensor/memory.h
    tensor/tensorimpl.h
)

set(TENSOR_SOURCES
    io/file.cc
    io/iwl.cc
    io/manager.cc
    math/blas.cc
    math/lapack.cc
    tensor/blocked_tensor.cc
    tensor/core.cc
    tensor/disk.cc
    tensor/slice.cc
    tensor/indices.cc
    tensor/labeled_tensor.cc
    tensor/sliced_tensor.cc
    tensor/tensor.cc
    tensor/tensorimpl.cc
)

# if we have MPI and Cyclops is enabled
if(ENABLE_CYCLOPS)
    list(APPEND TENSOR_HEADERS tensor/cyclops/cyclops.h)
    list(APPEND TENSOR_SOURCES tensor/cyclops/cyclops.cc)
endif()

# build static libray
add_library(tensor STATIC ${TENSOR_SOURCES} ${TENSOR_HEADERS})

if(ENABLE_CYCLOPS)
    target_link_libraries(tensor ${CYCLOPS}/lib/libctf.a ${MPI_LIBRARIES})
    set_target_properties(tensor PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    set_target_properties(tensor PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

target_link_libraries(tensor ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})

install(TARGETS tensor ARCHIVE DESTINATION lib)

if(ENABLE_PYTHON)
    add_subdirectory(python)
endif()

