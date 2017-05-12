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
// Created by Justin Turney on 1/20/16.
//

#include <ambit/io/hdf5/attribute.h>

namespace ambit {

namespace io {

namespace hdf5 {

Attribute::Attribute(Location const& location, const string& name)
        : id_(-1), location_(location), name_(name)
{ }

Attribute::~Attribute()
{
    if (id_ != -1) {
        H5Aclose(id_);
        id_ = -1;
    }
}

bool Attribute::exists() const
{
    return H5Aexists(location_.id(), name_.c_str()) > 0;
}

} // namespace hdf5

} // namespace io

} // namespace ambit
