# ambit
C++ library for the implementation of tensor product calculations through a clean, concise user interface.

Build status: [![Build Status](https://travis-ci.org/jturney/ambit.svg?branch=master)](https://travis-ci.org/jturney/ambit)

Primary test environments
=========================

Continuous integration builds
-----------------------------

- Ubuntu 12.04 LTS 64-bit with Python 2.7.3, CMake 3.3.2
  this is the environment offered by [Travis CI](https://travis-ci.org) pulling
  in various PPA. The following compilers are used, both in release and debug:

  1. GCC 4.7
  2. GCC 4.8
  3. GCC 4.9
  4. GCC 5.1, with and without coverage analysis in debug mode
  5. Clang 3.5 and GFortran 4.6
  6. Clang 3.6 and GFortran 4.6
  7. Clang 3.7 and GFortran 4.6
  8. Clang 3.8 and GFortran 4.6

- Mac OS X 10.9.5 with Python 2.7.10, CMake 3.2.3
  this is the environment offered by [Travis CI](https://travis-ci.org)
  The following compilers are used, both in release and debug:

  1. XCode 6.4 with Clang and GFortran 5.2
  2. XCode 6.4 with GCC 5.2
  3. XCode 7.0 with Clang and GFortran 5.2
  4. XCode 7.0 with GCC 5.2

Nightly builds
--------------

