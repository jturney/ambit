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
// Created by Justin Turney on 1/9/16.
//

#include <ambit/common_types.h>
#include <ambit/io/hdf5/group.h>

namespace ambit {

namespace io {

namespace hdf5 {

Group::Group(const Location& loc, const string& name)
{
    if (loc.has_link(name))
        open(loc, name);
    else
        create(loc, name);
}

Group::~Group()
{
    close();
}

void Group::create(const Location& loc, const string& name)
{
    close();
    id_ = H5Gcreate2(loc.id(), name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    assert(id_  >= 0);
}

void Group::open(const Location& loc, const string& name)
{
    close();
    id_ = H5Gopen2(loc.id(), name.c_str(), H5P_DEFAULT);
    assert(id_ >= 0);
}

void Group::close()
{
    if (id_ >= 0)
        H5Gclose(id_);
    id_ = -1;
}

size_t Group::size() const
{
    hsize_t size;
    H5Gget_num_objs(id(), &size);
    return static_cast<size_t>(size);
}

} // namespace hdf5

} // namespace io

} // namespace ambit
