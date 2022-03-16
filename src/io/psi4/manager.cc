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

#include <filesystem>

#include <ambit/io/psi4/io.h>

#include <util/print.h>

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

namespace ambit
{
namespace io
{
namespace psi4 {

namespace {

void create_directory(const std::string& directory)
{
    if (std::filesystem::exists(directory) == false) {
        // create directory
        std::filesystem::create_directory(directory);
    }
    else if (std::filesystem::is_directory(directory) == false) {
        // it exists and is not a directory
        throw std::runtime_error("base directory name given already exists and "
                                         "is not a directory: " +
                                 directory);
    }
}
}

Manager::Manager(const std::string& base_directory)
        : base_directory_(base_directory)
{
    create_directory(base_directory_);

// if we are running with MPI create a "local" scratch directory
#if defined(HAVE_MPI)
    int flag = 0;
    MPI_Initialized(&flag);
    if (flag)
    {
        mpi_base_directory_ =
            base_directory_ + "/" +
            std::to_string(MPI::COMM_WORLD.Get_rank());
        create_directory(mpi_base_directory_);
    }
    else
        mpi_base_directory_ = base_directory_;
#else
    mpi_base_directory_ = base_directory_;
#endif

    // append tailing '/' to the directories to make our lives easier
    base_directory_ += "/";
    mpi_base_directory_ += "/";

    printf("base_directory_ %s\n", base_directory_.c_str());
    printf("mpi_base_directory_ %s\n", mpi_base_directory_.c_str());
}

File Manager::scratch_file(const std::string& basename)
{
    return File(mpi_base_directory_ + basename, kOpenModeOpenExisting);
}
}
}
}
