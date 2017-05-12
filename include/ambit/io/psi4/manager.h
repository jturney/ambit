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

#if !defined(TENSOR_IO_MANAGER)
#define TENSOR_IO_MANAGER

#include "file.h"

#include <map>
#include <string>

namespace ambit
{
namespace io
{
namespace psi4 {

struct Manager
{
    /** Constructor for the io manager.
    * Will create base_directory, if needed.
    * \param base_directory The directory the scratch files will be created in.
    */
    Manager(const std::string& base_directory);

    /** Returns a file object associated with the basename given.
    * \param basename The name of the file to be created.
    */
    File scratch_file(const std::string& basename);

private:
    /** Base directory for the scratch files.
    * When in an MPI environment this is the directory for "global" files.
    */
    std::string base_directory_;

    /** Base directory for MPI process scratch files.
    * When not in an MPI environment this is equivalent to base_directory_.
    */
    std::string mpi_base_directory_;
};
}
}
}

#endif
