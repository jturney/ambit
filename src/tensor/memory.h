#if !defined(TENSOR_MEMORY_H)
#define TENSOR_MEMORY_H

#include <mm_malloc.h>

namespace tensor {

namespace memory {

/*
 * Intel(R) recommends data alignment to 64 bytes for MIC architecture.
 *
 * 16 byte alignment for SSE
 * 32 byte alignment for AVX
 */
enum { kMemoryAlignmentBytes = 64 };

template <typename DataType>
DataType* allocate(size_t size)
{
    DataType *tmp = (DataType*)_mm_malloc(size * sizeof(DataType), kMemoryAlignmentBytes);

    if (tmp == nullptr) {
        throw tensor::detail::OutOfMemoryException();
    }

    return tmp;
}

template <typename DataType>
void free(DataType* data)
{
    _mm_free(data);
}

}

}

#endif
