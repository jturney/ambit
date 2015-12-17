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

#include <ambit/io/iwl.h>
#include <stdexcept>

namespace ambit
{
namespace io
{

IWL::IWL(File &&f, double cutoff, bool psi34_compatible)
    : File(std::move(f)), nintegral(0), last_buffer(0),
      values(details::integrals_per_buffer__),
      p(details::integrals_per_buffer__), q(details::integrals_per_buffer__),
      r(details::integrals_per_buffer__), s(details::integrals_per_buffer__),
      labels_(details::integrals_per_buffer__ * 4),
      psi34_compatible_(psi34_compatible), cutoff_(cutoff),
      read_position_({0, 0})
{
    // ensure the iwl buffer exists in the file.
    if (open_mode_ == kOpenModeOpenExisting)
    {
        if (toc().exists(details::buffer_key__) == false)
            throw std::runtime_error("IWL buffer does not exist in file: " +
                                     name_);
        // go ahead and fetch the first buffer
        fetch();
    }
}

IWL::IWL(const std::string &full_pathname, enum OpenMode om, enum DeleteMode dm,
         double cutoff, bool psi34_compatible)
    : File(full_pathname, om, dm), nintegral(0), last_buffer(0),
      values(details::integrals_per_buffer__),
      p(details::integrals_per_buffer__), q(details::integrals_per_buffer__),
      r(details::integrals_per_buffer__), s(details::integrals_per_buffer__),
      labels_(details::integrals_per_buffer__ * 4),
      psi34_compatible_(psi34_compatible), cutoff_(cutoff),
      read_position_({0, 0})
{
    // ensure the iwl buffer exists in the file.
    if (om == kOpenModeOpenExisting)
    {
        if (toc().exists(details::buffer_key__) == false)
            throw std::runtime_error("IWL buffer does not exist in file: " +
                                     full_pathname);

        // go ahead and fetch the first buffer
        fetch();
    }
}

IWL::~IWL() {}

void IWL::fetch()
{
    read_entry_stream(details::buffer_key__, read_position_,
                      (int *)&last_buffer, 1);
    read_entry_stream(details::buffer_key__, read_position_, (int *)&nintegral,
                      1);

    if (psi34_compatible_)
        read_entry_stream(details::buffer_key__, read_position_, labels_.data(),
                          4 * details::integrals_per_buffer__);
    else
        throw std::runtime_error("Not implemented");

    read_entry_stream(details::buffer_key__, read_position_, values.data(),
                      details::integrals_per_buffer__);

    // distribute the labels to their respective p, q, r, s
    for (int i = 0; i < details::integrals_per_buffer__; ++i)
    {
        p[i] = labels_[4 * i + 0];
        q[i] = labels_[4 * i + 1];
        r[i] = labels_[4 * i + 2];
        s[i] = labels_[4 * i + 3];
    }
}

void IWL::read_one(File &io, const std::string &label, Tensor &tensor)
{
    // ensure the tensor object is only 2D.
    if (tensor.rank() != 2)
        throw std::runtime_error("tensor must be 2 dimension.");
    if (tensor.dims()[0] != tensor.dims()[1])
        throw std::runtime_error("tensor must be square.");

    // psi stores lower triangle full block (may be symmetry blocked, but we
    // don't care).

    std::vector<double> &data = tensor.data();
    size_t n = tensor.dims()[0];
    size_t ntri = n * (n + 1) / 2;

    std::vector<double> ints(ntri);
    io.read(label, ints);

    // Walk through lower triangle and mirror it to the upper triangle
    for (size_t x = 0, xy = 0; x < n; ++x)
    {
        for (size_t y = 0; y <= x; ++y, ++xy)
        {
            data[x * n + y] = ints[xy];
            data[y * n + x] = ints[xy];
        }
    }
}

namespace
{
size_t position(size_t dim, short int p, short int q, short int r, short int s)
{
    //    return ((p * dim + q) * dim + r) * dim + s;
    return p * dim * dim * dim + q * dim * dim + r * dim + s;
}
}

void IWL::read_two(IWL &io, Tensor &tensor)
{
    // ensure the tensor object is only 4D.
    if (tensor.rank() != 4)
        throw std::runtime_error("tensor must be rank 4");

    size_t dim = tensor.dim(0);
    for (size_t i = 1; i < 4; ++i)
    {
        if (dim != tensor.dim(i))
            throw std::runtime_error(
                "tensor must have equivalent length indices");
    }

    // psi stores 1 of the 8 possible permutations

    std::vector<double> &values = tensor.data();

    size_t count = 0;
    do
    {
        size_t c = 0;

        for (int i = 0; i < io.nintegral; ++i)
        {
            short int p, q, r, s;
            p = io.p[i];
            q = io.q[i];
            r = io.r[i];
            s = io.s[i];

            values[position(dim, p, q, r, s)] = io.values[i];
            values[position(dim, p, q, s, r)] = io.values[i];
            values[position(dim, q, p, r, s)] = io.values[i];
            values[position(dim, q, p, s, r)] = io.values[i];
            values[position(dim, r, s, p, q)] = io.values[i];
            values[position(dim, r, s, q, p)] = io.values[i];
            values[position(dim, s, r, p, q)] = io.values[i];
            values[position(dim, s, r, q, p)] = io.values[i];
        }

        count += c;

        values.resize(c);

        if (io.last_buffer)
            break;
        io.fetch();
    } while (1);
}
}
}
