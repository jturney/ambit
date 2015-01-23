set(TENSOR_HEADERS
    ${PROJECT_SOURCE_DIR}/include/tensor/tensor.h
    math/math.h
    tensor/core.h
    tensor/indices.h
    tensor/macros.h
    tensor/memory.h
    tensor/tensorimpl.h
)

set(TENSOR_SOURCES
    math/blas.cc
    math/lapack.cc
    tensor/core.cc
    tensor/indices.cc
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

install(TARGETS tensor ARCHIVE DESTINATION lib)

