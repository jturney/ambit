#if !defined(TENSOR_MEMORY_H)
#define TENSOR_MEMORY_H

#include <cstddef>
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
    // Add extra padding of the length of the Alignment.
    // This allows us to use -opt-assume-safe-padding on the newer Intel compilers
    // when targeting MICs. It allows the compiler to forgo adding code to handle
    // loop remainders.
    DataType *tmp = (DataType*)_mm_malloc(size * sizeof(DataType) + kMemoryAlignmentBytes, kMemoryAlignmentBytes);

    if (tmp == nullptr) {
        throw std::bad_alloc();
    }

    return tmp;
}

template <typename DataType>
void free(DataType* data)
{
    _mm_free(data);
}


/**
 * Allocator for aligned data.
 */
template <typename T, std::size_t Alignment=kMemoryAlignmentBytes>
struct aligned_allocator
{
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    pointer address(reference r) const
    {
        return &r;
    }

    const_pointer address(const_reference r) const
    {
        return &r;
    }

    size_type max_size() const
    {
        return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
    }

    template <typename U>
    struct rebind
    {
        typedef aligned_allocator<U, Alignment> other;
    };

    bool operator!=(const aligned_allocator& other) const
    {
        return !(*this == other);
    }

    void construct(T * const p, const T& t) const
    {
        void * const pv = static_cast<void*>(p);

        new (pv) T(t);
    }

    void destroy(T * const p) const
    {
        p->~T();
    }

    bool operator==(const aligned_allocator& other) const
    {
        return (*this == other);
    }

    // Default construct, copy constructor, rebinding constructor, and destructor.
    // Empty for stateless allocators.
    aligned_allocator() {}

    aligned_allocator(const aligned_allocator&) {}

    template<typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) {}

    ~aligned_allocator() {}

    pointer allocate(const size_type n) const
    {
        // The return value of allocate(0) is unspecified.
        // aligned_allocator returns NULL in order to avoid depending
        // on malloc(0)'s implementation-defined behavior.
        if (n == 0)
            return NULL;

        // All allocators should contain an integer overflow check,
        // The Standardization Committee recommends that std::length_error
        // be thrown in the case of integer overflow
        if (n > max_size())
            throw std::length_error("aligned_allocator<T>::allocate() - Integer overflow.");

        // Add extra padding of the length of the Alignment.
        // This allows us to use -opt-assume-safe-padding on the newer Intel compilers
        // when targeting MICs. It allows the compiler to forgo adding code to handle
        // loop remainders.
        void * const pv = _mm_malloc(n * sizeof(T) + Alignment, Alignment);

        if (pv == NULL)
            throw std::bad_alloc();

        return static_cast<pointer>(pv);
    }

    void deallocate(T * const p, const std::size_t /*n*/) const
    {
        _mm_free(p);
    }

    template<typename U>
    pointer allocate(const size_type n, const U * /*const hint*/) const
    {
        return allocate(n);
    }

private:
    aligned_allocator& operator=(const_reference);
};

}

template<typename T>
using aligned_vector = typename std::vector<T, memory::aligned_allocator<T, memory::kMemoryAlignmentBytes>>;

}

#endif
