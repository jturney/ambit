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

#ifndef AMBIT_DATASET_H
#define AMBIT_DATASET_H

#include <ambit/common_types.h>
#include <ambit/io/hdf5/location.h>
#include <ambit/io/hdf5/dataspace.h>
#include <ambit/io/hdf5/type.h>
#include <hdf5.h>

namespace ambit
{

namespace io
{

namespace hdf5
{

template <typename T> struct Dataset
{
    Dataset() : id_(-1) {}

    Dataset(hid_t id) : id_(id) {}

    Dataset(const Location &location, const string &name) : id_(-1)
    {
        open(location, name);
    }

    Dataset(const Location &location, const string &name,
            const Dataspace &space)
        : id_(-1)
    {
        create(location, name, space);
    }

    virtual ~Dataset() { close(); }

    void open(const Location &location, const string &name)
    {
        close();

        id_ = H5Dopen2(location.id(), name.c_str(), H5P_DEFAULT);
        if (id_ == -1)
            throw std::runtime_error("Unable to open dataset");
    }

    void create(const Location &location, const string &name,
                const Dataspace &space)
    {
        close();

        id_ = H5Dcreate2(location.id(), name.c_str(), detail::ctype<T>::hid(),
                        space.id(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        if (id_ == -1)
            throw std::runtime_error("Unable to create dataset");
    }

    void close()
    {
        if (id_ != -1)
        {
            H5Dclose(id_);
            id_ = -1;
        }
    }

    void write(const vector<T> &data)
    {
        H5Dwrite(id_, detail::ctype<T>::hid(), H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 data.data());
    }

    void write(const Tensor &data)
    {
        if (data.type() != CoreTensor)
        {
            throw std::runtime_error(
                "Only able to write CoreTensor's to disk.");
        }

        write(data.data());
    }

    void read(const vector<T> &data)
    {
        //        H5Dread(id_, detail::ctype<T>::hid(), )
    }

    const hid_t &id() const { return id_; }

    static void write(const Location& location, const Tensor& data)
    {
        Dataspace space(data);
        Dataset<T> set(location, data.name(), space);
        set.write(data);
    }

  private:
    hid_t id_;
};

inline void write(const Location& location, const Tensor& data)
{
    Dataset<double>::write(location, data);
}

} // namespace hdf5

} // namespace io

} // namespace ambit
#endif // AMBIT_DATASET_H
