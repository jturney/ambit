#if !defined(TENSOR_MACROS_H)
#define TENSOR_MACROS_H

#if defined(__INTEL_COMPILER)
#define VECTORIZED_LOOP _Pragma("simd")
#elif defined(__GNUC__)
#define VECTORIZED_LOOP _Pragma("GCC ivdep")
#else
#define VECTORIZED_LOOP
#endif

#if defined(_OPENMP)
#define OMP_STATIC_LOOP _Pragma("omp parallel for schedule(static)")
#define OMP_GUIDED_LOOP _Pragma("omp parallel for schedule(guided)")
#define OMP_CYCLIC_LOOP _Pragma("omp parallel for schedule(static,1)")
#define OMP_VECTORIZED_STATIC_LOOP VECTORIZED_LOOP OMP_STATIC_LOOP
#else
#define OMP_STATIC_LOOP
#define OMP_GUIDED_LOOP
#define OMP_CYCLIC_LOOP
#define OMP_VECTORIZED_STATIC_LOOP VECTORIZED_LOOP
#endif

#if defined(__GNUC__) || (defined(__MWERKS__) && (__MWERKS__ >= 0x3000)) ||    \
    (defined(__ICC) && (__ICC >= 600)) || defined(__ghs__)

#define CURRENT_FUNCTION __PRETTY_FUNCTION__

#elif defined(__DMC__) && (__DMC__ >= 0x810)

#define CURRENT_FUNCTION __PRETTY_FUNCTION__

#elif defined(__FUNCSIG__)

#define CURRENT_FUNCTION __FUNCSIG__

#elif(defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 600)) ||               \
    (defined(__IBMCPP__) && (__IBMCPP__ >= 500))

#define CURRENT_FUNCTION __FUNCTION__

#elif defined(__BORLANDC__) && (__BORLANDC__ >= 0x550)

#define CURRENT_FUNCTION __FUNC__

#elif defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901)

#define CURRENT_FUNCTION __func__

#elif defined(__cplusplus) && (__cplusplus >= 201103)

#define CURRENT_FUNCTION __func__

#else

#define CURRENT_FUNCTION "(unknown)"

#endif

#endif
