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
if(ENABLE_CTF)
    list(APPEND TENSOR_HEADERS tensor/cyclops/cyclops.h)
    list(APPEND TENSOR_SOURCES tensor/cyclops/cyclops.cc)
endif()

# build static libray
add_library(tensor STATIC ${TENSOR_SOURCES} ${TENSOR_HEADERS})

if(ENABLE_CTF)
    target_link_libraries(tensor ${CTF}/lib/libctf.a)
endif()

install(TARGETS tensor ARCHIVE DESTINATION lib)
