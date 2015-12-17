/*
 *  Copyright (C) 2013  Justin Turney
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.

 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.

 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-151 USA.
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

struct Manager
{
    /** Constructor for the io manager.
    * Will create base_directory, if needed.
    * \param base_directory The directory the scratch files will be created in.
    */
    Manager(const std::string &base_directory);

    /** Returns a file object associated with the basename given.
    * \param basename The name of the file to be created.
    */
    File scratch_file(const std::string &basename);

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

#endif
