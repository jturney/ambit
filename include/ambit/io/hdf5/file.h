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

#ifndef AMBIT_FILE_H
#define AMBIT_FILE_H

#include <ambit/common_types.h>
#include <ambit/io/hdf5/group.h>
#include <ambit/io/hdf5/location.h>

namespace ambit {

namespace io {

namespace hdf5 {

enum OpenMode
{
    kOpenModeCreateNew,
    kOpenModeOpenExisting
};

enum DeleteMode
{
    kDeleteModeKeepOnClose,
    kDeleteModeDeleteOnClose
};

struct File
        : public Location
{
    File() = default;
    File(const string& filename,
         OpenMode om,
         DeleteMode dm = kDeleteModeKeepOnClose);

    virtual ~File();

    void open(const string& filename,
              OpenMode om,
              DeleteMode dm = kDeleteModeKeepOnClose);

    void close();

private:
    string filename_;
    DeleteMode delete_mode_;
};

} // namespace hdf5

} // namespace io

} // namespace ambit
#endif //AMBIT_FILE_H
