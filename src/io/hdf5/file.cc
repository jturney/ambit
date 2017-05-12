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
// Created by Justin Turney on 1/8/16.
//

#include <ambit/io/hdf5/file.h>
#include <ambit/print.h>

namespace ambit {

namespace io {

namespace hdf5 {

File::File(const string& filename, OpenMode om, DeleteMode dm)
        : Location(), filename_(filename)
{
    open(filename, om, dm);
}

File::~File()
{
    close();
}

void File::open(const string& filename, OpenMode om, DeleteMode dm)
{
    delete_mode_ = dm;

    if (om == kOpenModeCreateNew)
        id_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    else
        id_ = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
}

void File::close()
{
    if (id_ != -1) {
        H5Fclose(id_);

        if (delete_mode_ == kDeleteModeDeleteOnClose)
            remove(filename_.c_str());

        id_ = -1;
    }
}

} // namespace hdf6

} // namespace io

} // namespace ambit
