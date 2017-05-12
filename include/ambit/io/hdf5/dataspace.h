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
// Created by Justin Turney on 1/11/16.
//

#ifndef AMBIT_DATASPACE_H_H
#define AMBIT_DATASPACE_H_H

#include <ambit/common_types.h>
#include <ambit/tensor.h>
#include <hdf5.h>

namespace ambit {

namespace io {

namespace hdf5 {

struct Dataspace
{

    Dataspace(const Tensor& tensor);
    Dataspace(const Dimension& current_dims);
    Dataspace(const Dimension& current_dims, const Dimension& maximum_dims);

    virtual ~Dataspace();

    hid_t id() const
    {
        return dataspace_id_;
    }

private:
    hid_t dataspace_id_;
};

} // namespace hdf5

} // namespace io

} // namespace ambit
#endif //AMBIT_DATASPACE_H_H
