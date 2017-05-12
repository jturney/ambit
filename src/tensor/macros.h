/*
 * @BEGIN LICENSE
 *
 * ambit: C++ library for the implementation of tensor product calculations
 *        through a clean, concise user interface.
 *
 * Copyright (c) 2014-2017 Ambit developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of ambit.
 *
 * Ambit is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Ambit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with ambit; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

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
