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

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <ambit/io/io.h>

//#include <util/print.h>

#if defined(HAVE_MPI)
#include <mpi.h>
#endif

namespace ambit
{
namespace io
{

namespace
{

void create_directory(const std::string &directory)
{
    if (boost::filesystem::exists(directory) == false)
    {
        // create directory
        boost::filesystem::create_directory(directory);
    }
    else if (boost::filesystem::is_directory(directory) == false)
    {
        // it exists and is not a directory
        throw std::runtime_error("base directory name given already exists and "
                                 "is not a directory: " +
                                 directory);
    }
}
}

Manager::Manager(const std::string &base_directory)
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
            boost::lexical_cast<std::string>(MPI::COMM_WORLD.Get_rank());
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

File Manager::scratch_file(const std::string &basename)
{
    return File(mpi_base_directory_ + basename, kOpenModeOpenExisting);
}
}
}
