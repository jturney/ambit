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
