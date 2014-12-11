set(TENSOR_HEADERS
    ${PROJECT_SOURCE_DIR}/include/tensor/tensor.h
    math/math.h
    tensor/core.h
    tensor/tensorimpl.h
)

set(TENSOR_SOURCES
    math/blas.cc
    math/lapack.cc
    tensor/core.cc
    tensor/tensor.cc
    tensor/tensorimpl.cc
)

# if we have MPI and Cyclops is enabled
#if(ENABLE_CYCLOPS)
#    list(APPEND TENSOR_HEADERS cyclops/cyclopsimpl.h)
#    list(APPEND TENSOR_SOURCES cyclops/cyclopsimpl.cc)
#endif()

# build static libray
add_library(tensor STATIC ${TENSOR_SOURCES} ${TENSOR_HEADERS})

# specify dependencies for the library (if any)
#add_dependencies(tensor )
install(TARGETS tensor ARCHIVE DESTINATION lib)
