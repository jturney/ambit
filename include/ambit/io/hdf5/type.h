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

//
// Created by Justin Turney on 1/14/16.
//

#ifndef AMBIT_TYPE_H_H
#define AMBIT_TYPE_H_H

#include <ambit/common_types.h>
#include <hdf5.h>

namespace ambit {

namespace io {

namespace hdf5 {

namespace detail {

template<typename T>
struct ctype;

#define MAKE_CTYPE(T, H5T) \
    template<> \
    struct ctype<T> \
    { \
        static hid_t hid() \
        { \
            return H5Tcopy(H5T); \
        } \
    };

MAKE_CTYPE( char,                  H5T_NATIVE_CHAR );
MAKE_CTYPE( signed char,           H5T_NATIVE_SCHAR );
MAKE_CTYPE( unsigned char,         H5T_NATIVE_UCHAR );
MAKE_CTYPE( short,                 H5T_NATIVE_SHORT );
MAKE_CTYPE( unsigned short,        H5T_NATIVE_USHORT );
MAKE_CTYPE( int,                   H5T_NATIVE_INT );
MAKE_CTYPE( unsigned int,          H5T_NATIVE_UINT );
MAKE_CTYPE( long,                  H5T_NATIVE_LONG );
MAKE_CTYPE( unsigned long,         H5T_NATIVE_ULONG );
MAKE_CTYPE( long long,             H5T_NATIVE_LLONG );
MAKE_CTYPE( unsigned long long,    H5T_NATIVE_ULLONG );
MAKE_CTYPE( float,                 H5T_NATIVE_FLOAT );
MAKE_CTYPE( double,                H5T_NATIVE_DOUBLE );
MAKE_CTYPE( long double,           H5T_NATIVE_LDOUBLE );
MAKE_CTYPE( bool,                  H5T_NATIVE_CHAR ); // ignore unsupported type HBOOL

#undef MAKE_CTYPE

} // namespace detail

} // namespace hdf5

} // namespace io

} // namespace ambit
#endif //AMBIT_TYPE_H_H
